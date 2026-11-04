import json
import os
from collections.abc import Iterator, Mapping, Sequence
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeVar

from pydantic import ConfigDict, TypeAdapter, ValidationError
from typing_extensions import ReadOnly

import litellm
from litellm._logging import verbose_logger
from litellm.constants import X_LITELLM_DISABLE_CALLBACKS
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.internal_call_metadata import sanitize_user_api_key_auth
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.utils import TRUSTED_CALLBACK_VARS_FIELD, ModelInfo, StandardCallbackDynamicParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObject

OPENAI_MAX_PROMPT_CACHE_KEY_LENGTH: Final = 64

_EFFORT_DEGRADATION_CHAIN: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "max": ("max", "xhigh", "high"),
        "xhigh": ("xhigh", "high"),
        "minimal": ("minimal", "low"),
    }
)
REASONING_EFFORT_OFF: Final = "none"
RESPONSES_API_PROVIDERS: Final = frozenset({"openai"})
_LOWEST_REASONING_TIERS: Final = ("minimal", "low")
_UNCONFIRMED_MINIMAL_LOWEST_TIERS: Final = ("low",)

ThinkingOffSwitch = Literal["disabled", "between_tools"]

_JsonValue: Final = TypeVar("_JsonValue")


def json_copy_or_original(value: _JsonValue) -> _JsonValue:
    try:
        return json.loads(json.dumps(value, default=dict))
    except (TypeError, ValueError, RecursionError):
        return value


class _ClaudeCodeUserId(LiteLLMBaseModel):
    """The JSON Claude Code packs into ``metadata.user_id``; only ``session_id`` is per conversation."""

    model_config = ConfigDict(frozen=True)

    session_id: str


def prompt_cache_key_from_user_id(user_id: object) -> str | None:
    """The per-session key Claude Code carries inside ``metadata.user_id``, or nothing.

    Anthropic defines ``user_id`` as an opaque end-user id, so a plain string names a person, not
    a conversation. Keying the provider cache on it pins every parallel session and subagent of that
    person to one slot, which caches worse than the provider's own prompt-prefix hashing does.
    """
    if not isinstance(user_id, str):
        return None
    try:
        return _ClaudeCodeUserId.model_validate_json(user_id).session_id[:OPENAI_MAX_PROMPT_CACHE_KEY_LENGTH] or None
    except ValidationError:
        return None


PROXY_SUB_CALL_METADATA_KEYS: Final = (
    "user_api_key",
    "user_api_key_hash",
    "user_api_key_alias",
    "user_api_key_team_id",
    "user_api_key_team_alias",
    "user_api_key_user_id",
    "user_api_key_user_email",
    "user_api_key_org_id",
    "user_api_key_project_id",
    "user_api_key_end_user_id",
    "user_api_end_user_max_budget",
    "user_api_key_model_max_budget",
    "user_api_key_team_model_max_budget",
    "user_api_key_user_model_max_budget",
    "user_api_key_team_member_model_max_budget",
    "user_api_key_end_user_model_max_budget",
    "litellm_call_id",
    "litellm_parent_otel_span",
    "headers",
)
_USER_API_KEY_AUTH_METADATA_KEY: Final = "user_api_key_auth"
_OBJECT_MAPPING: Final = TypeAdapter(Mapping[str, object])
_OBJECT_LIST: Final = TypeAdapter(list[object])


_NO_SETTINGS: Final[Mapping[str, object]] = MappingProxyType({})


def proxy_general_settings() -> Mapping[str, object]:
    """The running proxy's ``general_settings``; empty for SDK callers and before the proxy config loads."""
    try:
        from litellm.proxy.proxy_server import general_settings
    except ImportError:
        return _NO_SETTINGS
    return general_settings or _NO_SETTINGS


def proxy_sub_call_metadata(
    parent_litellm_metadata: Mapping[str, object] | None,
) -> dict[str, object]:  # mutable-ok: the router writes model_group and fallback keys into the metadata it is handed
    """The parent request's auth, budget, and header fields, so a sub-call bills the same key, team, and end
    user, the router filters deployments by the caller's access groups, and header-driven logging options
    such as message redaction apply to it. The caller's auth object goes without its budget reservation,
    which belongs to the parent request."""
    if not parent_litellm_metadata:
        return {}
    carried: Final = {
        key: parent_litellm_metadata[key] for key in PROXY_SUB_CALL_METADATA_KEYS if key in parent_litellm_metadata
    }
    caller: Final = parent_litellm_metadata.get(_USER_API_KEY_AUTH_METADATA_KEY)
    if caller is None:
        return carried
    return {**carried, _USER_API_KEY_AUTH_METADATA_KEY: sanitize_user_api_key_auth(caller)}


class ProxySubCallLoggingKwargs(StandardCallbackDynamicParams, total=False):
    litellm_trusted_callback_vars: ReadOnly[Mapping[str, object]]
    proxy_server_request: ReadOnly[Mapping[str, Mapping[str, str]]]


def proxy_sub_call_logging_kwargs(parent_kwargs: Mapping[str, object]) -> ProxySubCallLoggingKwargs:
    """The parent request's callback params and proxy-stamped callback vars, so a sub-call logs under the same
    message-logging and callback-disable settings as the request it serves. Of the parent's ``proxy_server_request``
    only the ``x-litellm-disable-callbacks`` header is carried, since dispatch reads it there: the router reads
    per-request routing inputs from the rest, such as the Claude Code session that would pull a sub-call off its
    configured model, and redaction headers already ride in ``proxy_sub_call_metadata``."""
    parent_logging: Final = litellm_logging_obj_from_kwargs(parent_kwargs)
    parent_params: Final[ProxySubCallLoggingKwargs] = (
        {**parent_logging.standard_callback_dynamic_params} if parent_logging is not None else {}
    )
    disable_header: Final = _disable_callbacks_header(parent_kwargs)
    dynamic_params: Final[ProxySubCallLoggingKwargs] = (
        parent_params
        if disable_header is None
        else {**parent_params, "proxy_server_request": {"headers": {X_LITELLM_DISABLE_CALLBACKS: disable_header}}}
    )
    trusted_vars: Final = _object_mapping(parent_kwargs.get(TRUSTED_CALLBACK_VARS_FIELD))
    if trusted_vars is None:
        return dynamic_params
    return {**dynamic_params, "litellm_trusted_callback_vars": trusted_vars}


def proxy_sub_call_trace_id(parent_kwargs: Mapping[str, object]) -> str | None:
    """The trace id the parent call resolved, whether the caller sent it in the body or the proxy minted it, so a
    sub-call passes a team's ``require_trace_id`` check whenever the parent did."""
    body_trace_id: Final = parent_kwargs.get("litellm_trace_id")
    if isinstance(body_trace_id, str) and body_trace_id:
        return body_trace_id
    parent_logging: Final = litellm_logging_obj_from_kwargs(parent_kwargs)
    logged_trace_id: Final = parent_logging.litellm_trace_id if parent_logging is not None else None
    return logged_trace_id if isinstance(logged_trace_id, str) and logged_trace_id else None


def _disable_callbacks_header(parent_kwargs: Mapping[str, object]) -> str | None:
    parent_request: Final = _object_mapping(parent_kwargs.get("proxy_server_request"))
    headers: Final = _object_mapping(parent_request.get("headers")) if parent_request is not None else None
    header: Final = headers.get(X_LITELLM_DISABLE_CALLBACKS) if headers is not None else None
    return header if isinstance(header, str) else None


def _object_mapping(value: object) -> Mapping[str, object] | None:
    try:
        return _OBJECT_MAPPING.validate_python(value)
    except ValidationError:
        return None


def litellm_logging_obj_from_kwargs(kwargs: Mapping[str, object]) -> "LiteLLMLoggingObject | None":
    """The logging object the bridged call logs through, when the caller supplied one."""
    from litellm.litellm_core_utils.litellm_logging import Logging

    candidate: Final = kwargs.get("litellm_logging_obj")
    return candidate if isinstance(candidate, Logging) else None


def anthropic_sse_frame(event: Mapping[str, object]) -> bytes:
    return f"event: {event.get('type', 'message')}\ndata: {json.dumps(event)}\n\n".encode()


def local_model_name(model: str, custom_llm_provider: object) -> str:
    """The id the provider itself knows, for reporting back to the caller in ``message_start``."""
    return model.removeprefix(f"{custom_llm_provider}/") if isinstance(custom_llm_provider, str) else model


def is_reasoning_auto_summary_enabled() -> bool:
    """Check whether the default 'summary: detailed' injection is enabled (opt-in)."""
    return litellm.reasoning_auto_summary or os.getenv("LITELLM_REASONING_AUTO_SUMMARY", "false").lower() == "true"


def requested_reasoning_summary(thinking: Mapping[str, object]) -> str | None:
    if thinking.get("type") == "disabled":
        return None
    summary: Final = thinking.get("summary")
    if isinstance(summary, str):
        return summary
    display: Final = thinking.get("display")
    if display is not None:
        return None if display == "omitted" else "auto"
    return "auto" if thinking.get("type") == "between_tools" else None


def _list_items(value: object) -> Sequence[object]:
    try:
        return _OBJECT_LIST.validate_python(value, strict=True)
    except ValidationError:
        return ()


def content_blocks(content: object) -> tuple[Mapping[str, object], ...]:
    return tuple(block for block in map(as_str_mapping, _list_items(content)) if block is not None)


def _final_user_turn(messages: object) -> Iterator[Mapping[str, object]]:
    for message in map(as_str_mapping, reversed(_list_items(messages))):
        if message is None or message.get("role") != "user":
            return
        yield message


def is_between_tool_calls(request: Mapping[str, object]) -> bool:
    blocks: Final = chain.from_iterable(
        content_blocks(message.get("content")) for message in _final_user_turn(request.get("messages"))
    )
    return any(block.get("type") == "tool_result" for block in blocks)


def thinking_off_switch(request: Mapping[str, object]) -> ThinkingOffSwitch | None:
    thinking: Final = as_str_mapping(request.get("thinking"))
    thinking_type: Final = thinking.get("type") if thinking is not None else None
    if thinking_type == "disabled":
        return "disabled"
    if thinking_type == "between_tools" and not is_between_tool_calls(request):
        return "between_tools"
    return None


def _mapped_model_info(model: str, custom_llm_provider: str | None) -> ModelInfo | None:
    from litellm.utils import get_model_info

    try:
        return get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    except Exception:
        return None


def _deployment_model_info(
    model: str, custom_llm_provider: str | None, deployment_model_info: object
) -> Mapping[str, object] | None:
    from litellm.utils import catalog_entry

    own_info: Final = as_str_mapping(deployment_model_info)
    if own_info is None:
        return _mapped_model_info(model, custom_llm_provider)
    declared: Final = {key: value for key, value in own_info.items() if value is not None}
    return {**(catalog_entry(model, custom_llm_provider) or {}), **declared}


def _resolved_reasoning_efforts(
    model_info: Mapping[str, object] | None, model: str, custom_llm_provider: str | None
) -> tuple[str, ...] | None:
    from litellm.litellm_core_utils.reasoning_effort_capability import resolve_supported_reasoning_efforts
    from litellm.utils import has_catalog_entry

    if model_info is None:
        return None
    return resolve_supported_reasoning_efforts(
        model_info, deployment_is_mapped=has_catalog_entry(model, custom_llm_provider)
    )


def model_reasoning_efforts(model: str, custom_llm_provider: str | None) -> tuple[str, ...] | None:
    return _resolved_reasoning_efforts(_mapped_model_info(model, custom_llm_provider), model, custom_llm_provider)


def deployment_reasoning_efforts(
    model: str, custom_llm_provider: str | None, deployment_model_info: object
) -> tuple[str, ...] | None:
    return _resolved_reasoning_efforts(
        _deployment_model_info(model, custom_llm_provider, deployment_model_info), model, custom_llm_provider
    )


def _declares(model_info: Mapping[str, object] | None, capability: str) -> bool:
    return model_info is not None and model_info.get(capability) is True


def _confirms_minimal(model_info: Mapping[str, object] | None) -> bool:
    from litellm.litellm_core_utils.reasoning_effort_capability import declared_reasoning_efforts

    return model_info is not None and (
        declared_reasoning_efforts(model_info) is not None or _declares(model_info, "supports_minimal_reasoning_effort")
    )


def _lowest_tier_instead_of_off(levels: tuple[str, ...] | None, model_info: Mapping[str, object] | None) -> str | None:
    if not levels or REASONING_EFFORT_OFF in levels:
        return None
    tiers: Final = _LOWEST_REASONING_TIERS if _confirms_minimal(model_info) else _UNCONFIRMED_MINIMAL_LOWEST_TIERS
    return next((tier for tier in tiers if tier in levels), None)


def reasoning_off_level(
    model: str,
    custom_llm_provider: str | None,
    *,
    deployment_model_info: object,
    drop_params: bool,
    between_tools: bool,
) -> str | None:
    model_info: Final = _deployment_model_info(model, custom_llm_provider, deployment_model_info)
    levels: Final = _resolved_reasoning_efforts(model_info, model, custom_llm_provider)
    if levels == () or (between_tools and levels is None):
        return None
    lowest_tier: Final = _lowest_tier_instead_of_off(levels, model_info)
    if lowest_tier is not None:
        return lowest_tier
    if (
        not drop_params
        or not _declares(model_info, "supports_reasoning")
        or _declares(model_info, "supports_none_reasoning_effort")
    ):
        return REASONING_EFFORT_OFF
    verbose_logger.warning(
        "Dropping reasoning_effort=none for %s, its model cost map entry does not set "
        "supports_none_reasoning_effort, because drop_params is set",
        model,
    )
    return None


def maps_bare_output_effort(model: str, custom_llm_provider: str | None, *, deployment_model_info: object) -> bool:
    from litellm.litellm_core_utils.reasoning_effort_capability import resolve_supported_reasoning_efforts

    model_info: Final = _deployment_model_info(model, custom_llm_provider, deployment_model_info)
    return (
        model_info is not None
        and model_info.get("default_reasoning_effort") != REASONING_EFFORT_OFF
        and resolve_supported_reasoning_efforts(model_info, deployment_is_mapped=True) != ()
    )


def normalize_reasoning_effort_value(
    effort: str,
    model: str,
    custom_llm_provider: str | None = None,
    deployment_model_info: object = None,
) -> str:
    """Lower a tier the deployment does not accept to the nearest one it does, leaving others alone.

    A deployment whose own reasoning_effort_levels name a strength tier keeps the tier as sent, since
    completion() and responses() round it onto those levels.

    The accepted set is resolved by the same owner that answers ``/model_group/info``, so a level
    the proxy advertises is a level this path forwards.

    A deployment that refuses every step of a chain falls back to an accepted level read off that
    same set rather than to an assumed one, since an entry naming its levels outright can exclude
    the tiers the per-level flags treat as unconditional. ``none`` is never that fallback and is
    never degraded to, being an off switch rather than a tier; an always-on-thinking model is
    handled where the thinking block is built. A deployment accepting no tier at all keeps the
    chain's floor, which is what every deployment degraded to before there was anything to ask.
    """
    chain: Final = _EFFORT_DEGRADATION_CHAIN.get(effort)
    if chain is None:
        return effort

    from litellm.litellm_core_utils.reasoning_effort_capability import deployment_rounds_reasoning_effort

    if deployment_rounds_reasoning_effort(deployment_model_info):
        return effort

    supported: Final = model_reasoning_efforts(model, custom_llm_provider)
    if not supported:
        return chain[-1]

    accepted_tiers: Final = tuple(level for level in supported if level != REASONING_EFFORT_OFF)
    return next((level for level in (*chain, *accepted_tiers) if level in supported), chain[-1])
