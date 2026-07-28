import asyncio
import uuid
import weakref
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Literal

from pydantic import TypeAdapter

from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_strategy.complexity_router.cache_warming.eligibility import (
    min_prompt_cache_tokens_for_warm_set,
    resolve_warm_models,
)
from litellm.router_strategy.complexity_router.request_metadata import (
    get_session_id_from_request_kwargs,
    get_user_api_key_hash_from_request_kwargs,
    iter_metadata_dicts,
)
from litellm.router_strategy.complexity_router.cache_warming.types import (
    CACHE_WARMING_MARKER_KEY,
    CACHE_WARMING_REPLAY_MARKER_KEY,
    CACHE_WARMING_REPLAY_TAG,
    CacheWarmingAttribution,
    CacheWarmingPayload,
    compress_payload,
)
from litellm.types.utils import CallTypes

if TYPE_CHECKING:
    from litellm.router_strategy.complexity_router.complexity_router import ComplexityRouter

_CAPTURE_CALL_TYPES = frozenset(
    (CallTypes.completion, CallTypes.acompletion, CallTypes.anthropic_messages, CallTypes.aanthropic_messages)
)
_ANTHROPIC_CALL_TYPES = frozenset((CallTypes.anthropic_messages, CallTypes.aanthropic_messages))
_MAX_UNCOMPRESSED_RATIO = 8
_ATTRIBUTION_KEYS = (
    "user_api_key",
    "user_api_key_hash",
    "user_api_key_user_id",
    "user_api_key_team_id",
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
    proxy consents through cache_warming.enabled itself)."""
    from litellm.litellm_core_utils.redact_messages import should_redact_message_logging

    if should_redact_message_logging({"litellm_params": kwargs}):  # mutable-ok: read-only view for the predicate
        return False
    try:
        from litellm.proxy.spend_tracking.spend_tracking_utils import (
            _should_store_prompts_and_responses_in_spend_logs,  # pyright: ignore[reportPrivateUsage]  # canonical proxy consent gate; no public accessor exists
        )
    except ImportError:
        return True
    return _should_store_prompts_and_responses_in_spend_logs()


def _resolve_stamp(metadata_dicts: Sequence[Mapping[str, object]]) -> "tuple[ComplexityRouter, str] | None":
    marker = next(
        (value for metadata in metadata_dicts if isinstance(value := metadata.get(CACHE_WARMING_MARKER_KEY), dict)),
        None,
    )
    if marker is None:
        return None
    typed = TypeAdapter(dict[str, object]).validate_python(marker)
    ref = typed.get("strategy_ref")
    routed_model = typed.get("routed_model")
    if not isinstance(ref, str) or not isinstance(routed_model, str):
        return None
    strategy = _WARMING_STRATEGIES.get(ref)
    if strategy is None:
        verbose_router_logger.debug(
            "cache_warming: stamped strategy is no longer registered (config reload between routing and "
            "capture); skipping capture for this in-flight turn"
        )
        return None
    if typed.get("auto_router_model_name") != strategy.model_name:
        return None
    return (strategy, routed_model)


def _is_replay(metadata_dicts: Sequence[Mapping[str, object]]) -> bool:
    for metadata in metadata_dicts:
        if metadata.get(CACHE_WARMING_REPLAY_MARKER_KEY):
            return True
        tags = metadata.get("tags")
        if isinstance(tags, list) and CACHE_WARMING_REPLAY_TAG in tags:
            return True
    return False


def _gate_and_compress(
    payload: CacheWarmingPayload, max_payload_bytes: int, min_tokens: int
) -> "tuple[str, str, int] | Literal['too_large', 'too_small']":
    """Size gates plus compression, run off the event loop. The uncompressed bound
    runs before any compression so adversarial highly-compressible content cannot
    buy unbounded CPU with a small compressed result, and the chars/4 token
    estimate deliberately avoids running a real tokenizer over a full multi-turn
    conversation in the request path."""
    serialized_chars = len(payload.model_dump_json())
    if serialized_chars > _MAX_UNCOMPRESSED_RATIO * max_payload_bytes:
        return "too_large"
    token_estimate = serialized_chars // 4
    if token_estimate < min_tokens:
        return "too_small"
    blob, sha = compress_payload(payload)
    if len(blob) > max_payload_bytes:
        return "too_large"
    return (blob, sha, token_estimate)


def _extract_attribution(metadata_dicts: Sequence[Mapping[str, object]]) -> CacheWarmingAttribution:
    merged = dict(  # mutable-ok: handed straight to pydantic, never retained
        (key, str(value))
        for metadata in reversed(metadata_dicts)
        for key, value in metadata.items()
        if key in _ATTRIBUTION_KEYS and value is not None
    )
    return CacheWarmingAttribution(**merged)


_WARMING_STRATEGIES: "weakref.WeakValueDictionary[str, ComplexityRouter]" = weakref.WeakValueDictionary()


def register_warming_strategy(strategy: "ComplexityRouter") -> str:
    """Give a warming-enabled complexity router a per-request-resolvable identity.

    Capture is dispatched through one process-wide stateless hook; each request's
    pre-routing stamp carries the UUID returned here and capture resolves it back
    to the exact strategy object that routed the request. Replaced strategies
    simply fall out of the weak registry, so there is no hook lifecycle to sync
    on set_model_list or upsert_deployment and no per-Router hook instances to
    dedupe or remove."""
    import litellm

    ref = uuid.uuid4().hex
    _WARMING_STRATEGIES[ref] = strategy
    litellm.logging_callback_manager.add_litellm_callback(_get_dispatcher())
    return ref


@lru_cache(maxsize=1)
def _get_dispatcher() -> "ComplexityCacheWarmingCaptureHook":
    return ComplexityCacheWarmingCaptureHook()


class ComplexityCacheWarmingCaptureHook(CustomLogger):
    def __init__(self, privacy_gate: "Callable[[Mapping[str, object]], bool]" = _capture_allowed) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # CustomLogger.__init__ is legacy-untyped
        self.privacy_gate = privacy_gate

    async def async_pre_call_deployment_hook(self, kwargs: Mapping[str, object], call_type: CallTypes | None) -> None:
        try:
            await self._capture(kwargs, call_type)
        except Exception:  # noqa: BLE001  # a capture failure must never fail the user's request
            verbose_router_logger.exception("cache_warming capture failed; the request continues unaffected")

    async def _capture(self, kwargs: Mapping[str, object], call_type: CallTypes | None) -> None:
        if call_type not in _CAPTURE_CALL_TYPES:
            return
        metadata_dicts = iter_metadata_dicts(kwargs)
        if _is_replay(metadata_dicts):
            return
        resolved = _resolve_stamp(metadata_dicts)
        if resolved is None:
            return
        strategy, routed_model = resolved
        if not self.privacy_gate(kwargs):
            _warn_privacy_gate_blocked(strategy.model_name)
            return
        session_id = get_session_id_from_request_kwargs(kwargs)
        if session_id is None:
            return
        payload = self._build_payload(kwargs, call_type, routed_model)
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
        caller_scope = get_user_api_key_hash_from_request_kwargs(kwargs) or "unscoped"
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

    @staticmethod
    def _build_payload(
        kwargs: Mapping[str, object], call_type: CallTypes | None, routed_model: str
    ) -> CacheWarmingPayload | None:
        messages = kwargs.get("messages")
        if not isinstance(messages, list) or not messages:
            return None
        is_anthropic_surface = call_type in _ANTHROPIC_CALL_TYPES
        return CacheWarmingPayload.model_validate(
            {  # mutable-ok: pydantic input, never retained
                "model": routed_model,
                "messages": messages,
                "system": kwargs.get("system") if is_anthropic_surface else None,
                "tools": kwargs.get("tools"),
                "tool_choice": kwargs.get("tool_choice"),
                "call_surface": "anthropic_messages" if is_anthropic_surface else "chat_completions",
            }
        )
