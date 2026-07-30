import asyncio
import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from litellm._logging import verbose_router_logger
from litellm.constants import DEFAULT_CHARS_PER_TOKEN
from litellm.router_strategy.complexity_router.cache_warming.eligibility import (
    min_prompt_cache_tokens_for_warm_set,
    resolve_warm_models,
)
from litellm.router_strategy.complexity_router.cache_warming.types import (
    CACHE_WARMING_REPLAY_MARKER_KEY,
    CACHE_WARMING_REPLAY_TAG,
    CacheWarmingAttribution,
    CacheWarmingPayload,
    compress_payload,
)
from litellm.litellm_core_utils.core_helpers import (
    get_caller_scope,
    get_request_metadata_field,
    iter_request_metadata_dicts,
)

if TYPE_CHECKING:
    from litellm.router_strategy.complexity_router.complexity_router import ComplexityRouter

_MAX_UNCOMPRESSED_RATIO = 8
_ATTRIBUTION_KEYS = (
    "user_api_key",
    "user_api_key_hash",
    "user_api_key_user_id",
    "user_api_key_team_id",
    "user_api_key_org_id",
    "user_api_key_project_id",
    "user_api_key_end_user_id",
)


@lru_cache(maxsize=64)
def _warn_privacy_gate_blocked(auto_router_model_name: str) -> None:
    verbose_router_logger.warning(
        "cache_warming is enabled for auto-router %s but prompt retention is not permitted "
        "(store_prompts_in_spend_logs is off or message redaction is active); capture is skipped",
        auto_router_model_name,
    )


@lru_cache(maxsize=4096)
def _warn_payload_too_large(auto_router_model_name: str, session_id: str) -> None:
    verbose_router_logger.warning(
        "cache_warming: session %s on auto-router %s exceeds max_payload_bytes; not warming this session",
        session_id,
        auto_router_model_name,
    )


def _capture_allowed(kwargs: Mapping[str, object]) -> bool:
    """One consent predicate for one question: may this request's prompt content be
    retained? Honors both halves of the operator's stated policy: the redaction
    opt-out (turn_off_message_logging, including per-request and header forms) and
    the prompt-retention opt-in (store_prompts_in_spend_logs; SDK use without the
    proxy consents through cache_warming.enabled itself).

    should_redact_message_logging reads a model_call_details dict whose shape is
    implicit: the header and global forms come off litellm_params, but the
    per-request form is read from standard_callback_dynamic_params at the TOP level,
    so passing only litellm_params silently loses the caller's own opt-out. That
    value is owned by the logging object, which initializes it from the request in
    its constructor, so it is read off the object here rather than re-derived; the
    key is spelled exactly as the request path spells it when it builds the same
    dict for the same predicate (litellm_logging.py:565)."""
    from litellm.litellm_core_utils.redact_messages import (
        should_redact_message_logging,  # pyright: ignore[reportUnknownVariableType]  # legacy-untyped helper
    )

    model_call_details = {  # mutable-ok: read-only view for the predicate
        "litellm_params": kwargs,
        "standard_callback_dynamic_params": getattr(
            kwargs.get("litellm_logging_obj"), "standard_callback_dynamic_params", None
        ),
    }
    if should_redact_message_logging(model_call_details):
        return False
    try:
        from litellm.proxy.spend_tracking.spend_tracking_utils import (
            _should_store_prompts_and_responses_in_spend_logs,  # pyright: ignore[reportPrivateUsage]  # canonical proxy consent gate; no public accessor exists
        )
    except ImportError:
        return True
    return _should_store_prompts_and_responses_in_spend_logs()


def _is_replay(metadata_dicts: Sequence[Mapping[str, object]]) -> bool:
    for metadata in metadata_dicts:
        if metadata.get(CACHE_WARMING_REPLAY_MARKER_KEY):
            return True
        tags = metadata.get("tags")
        if isinstance(tags, list) and CACHE_WARMING_REPLAY_TAG in tags:
            return True
    return False


def _call_surface(request_kwargs: Mapping[str, object]) -> Literal["chat_completions", "anthropic_messages"]:
    """Two in-band signals, checked in order. Through the proxy the logging object
    rides the request into deployment selection and its call_type carries the route
    (function_setup stamps the entry function's name), which is the reliable signal
    because the fallback machinery strips original_function keys before the
    pre-routing hook runs. The generic-dispatch stamp is kept as the SDK-direct
    fallback for call shapes that never enter the proxy layer."""
    logging_call_type = getattr(request_kwargs.get("litellm_logging_obj"), "call_type", None)
    if logging_call_type in ("anthropic_messages", "aanthropic_messages"):
        return "anthropic_messages"
    generic_function = request_kwargs.get("original_generic_function")
    if getattr(generic_function, "__name__", None) == "anthropic_messages":
        return "anthropic_messages"
    return "chat_completions"


def _prompt_chars(payload: CacheWarmingPayload) -> int:
    """Every character the provider will cache: prompt text via the shared flattener (which reads both
    string and content-block shapes), plus tool schemas and tool_call arguments. Tool schemas dominate
    agent sessions, so omitting them made those sessions fail the min-token gate and never warm.
    The limiter's own _estimate_tokens_for_request omits tools too and is proxy-side, so no owner."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import get_str_from_messages

    system = payload.system if isinstance(payload.system, str) else list(payload.system or ())
    system_message = ({"role": "system", "content": system},) if payload.system is not None else ()
    text_chars = len(get_str_from_messages([*system_message, *payload.messages]))  # pyright: ignore[reportArgumentType]  # captured wire shape
    tool_chars = sum(len(json.dumps(tool, default=str)) for tool in payload.tools or ())
    tool_call_chars = sum(
        len(json.dumps(call, default=str))
        for message in payload.messages
        for call in (message.get("tool_calls") or ())
        if isinstance(call, Mapping)
    )
    return text_chars + tool_chars + tool_call_chars


def _gate_and_compress(
    payload: CacheWarmingPayload, max_payload_bytes: int, min_tokens: int
) -> "tuple[str, str, int] | Literal['too_large', 'too_small']":
    """Size gates plus compression, run off the event loop. The uncompressed bound
    runs before any compression so adversarial highly-compressible content cannot
    buy unbounded CPU with a small compressed result, and the chars/4 token
    estimate deliberately avoids running a real tokenizer over a full multi-turn
    conversation in the request path; counting prompt TEXT keeps the JSON envelope out of it."""
    serialized_chars = len(payload.model_dump_json())
    if serialized_chars > _MAX_UNCOMPRESSED_RATIO * max_payload_bytes:
        return "too_large"
    token_estimate = max(1, _prompt_chars(payload) // DEFAULT_CHARS_PER_TOKEN)
    if token_estimate < min_tokens:
        return "too_small"
    blob, sha = compress_payload(payload)
    if len(blob) > max_payload_bytes:
        return "too_large"
    return (blob, sha, token_estimate)


def _extract_attribution(metadata_dicts: Sequence[Mapping[str, object]]) -> CacheWarmingAttribution:
    """Identity comes from the ONE proxy-stamped slot, never merged: the proxy strips only the
    ``user_api_key_`` prefix, so a bare ``user_api_key`` in the other slot would forge billing identity.
    ``user_api_key_hash`` cannot be client-injected, so it marks the authoritative slot."""
    stamped = next((metadata for metadata in metadata_dicts if "user_api_key_hash" in metadata), None)
    if stamped is None:
        return CacheWarmingAttribution()
    return CacheWarmingAttribution(
        **{  # mutable-ok: handed straight to pydantic, never retained
            key: str(value) for key, value in stamped.items() if key in _ATTRIBUTION_KEYS and value is not None
        }
    )


def _build_payload(
    request_kwargs: Mapping[str, object],
    messages: "Sequence[Mapping[str, object]] | None",
    call_surface: Literal["chat_completions", "anthropic_messages"],
    routed_model: str,
) -> CacheWarmingPayload | None:
    if not messages:
        return None
    return CacheWarmingPayload.model_validate(
        {  # mutable-ok: pydantic input, never retained
            "model": routed_model,
            "messages": messages,
            "system": request_kwargs.get("system") if call_surface == "anthropic_messages" else None,
            "tools": request_kwargs.get("tools"),
            "tool_choice": request_kwargs.get("tool_choice"),
            "call_surface": call_surface,
        }
    )


async def capture_session(
    strategy: "ComplexityRouter",
    request_kwargs: Mapping[str, object],
    messages: "Sequence[Mapping[str, object]] | None",
    routed_model: str,
) -> None:
    metadata_dicts = iter_request_metadata_dicts(request_kwargs)
    if _is_replay(metadata_dicts):
        return
    if not _capture_allowed(request_kwargs):
        _warn_privacy_gate_blocked(strategy.model_name)
        return
    session_id = get_request_metadata_field(request_kwargs, "session_id")
    if session_id is None:
        return
    payload = _build_payload(request_kwargs, messages, _call_surface(request_kwargs), routed_model)
    if payload is None:
        return
    config = strategy.config.cache_warming
    warm_models = resolve_warm_models(strategy.config)
    gated = await asyncio.to_thread(
        _gate_and_compress, payload, config.max_payload_bytes, min_prompt_cache_tokens_for_warm_set(warm_models)
    )
    match gated:
        case "too_large":
            _warn_payload_too_large(strategy.model_name, session_id)
            return
        case "too_small":
            return
        case (blob, sha, token_estimate):
            pass
    store = strategy.get_cache_warming_store()
    if store is None:
        return
    caller_scope = get_caller_scope(request_kwargs)
    await store.upsert_session(
        caller_scope=caller_scope,
        session_id=session_id,
        payload_compressed=blob,
        payload_sha256=sha,
        token_estimate=token_estimate,
        served_model=routed_model,
        attribution=_extract_attribution(metadata_dicts),
        ttl_seconds=config.session_ttl_seconds,
        max_sessions=config.max_sessions,
    )
