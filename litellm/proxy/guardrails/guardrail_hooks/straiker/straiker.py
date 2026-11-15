from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NoReturn, Protocol, TypeAlias
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm._version import version as litellm_version
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.exceptions import (
    BadRequestError,
    GuardrailRaisedException,
    ModifyResponseException,
    Timeout,
)
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    get_session_id_from_request_data,
    log_guardrail_information,
)
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.litellm_core_utils.prompt_templates.factory import resolve_structured_messages
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import SpecialProxyStrings
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.proxy.guardrails.guardrail_hooks.straiker import (
    STRAIKER_WEBHOOK_SCHEMA_VERSION,
    StraikerGuardrailConfigModel,
    StraikerStreamMode,
    StraikerWebhookApplication,
    StraikerWebhookContent,
    StraikerWebhookContext,
    StraikerWebhookEvent,
    StraikerWebhookIdentity,
    StraikerWebhookRequest,
    StraikerWebhookResponse,
    StraikerWebhookStream,
    StraikerWebhookUsage,
)
from litellm.types.utils import (
    CallTypes,
    GenericGuardrailAPIInputs,
    GuardrailStatus,
    GuardrailTracingDetail,
    ModelResponse,
    TextCompletionResponse,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

GUARDRAIL_NAME: Final = "straiker"
DEFAULT_BLOCK_MESSAGE: Final = "Content violates policy"
DEFAULT_API_BASE: Final = "https://api.prod.straiker.ai"
DEFAULT_MAX_PAYLOAD_BYTES: Final = 524288
WEBHOOK_PATH: Final = "/api/v1/detect/webhook"
V3_DETECT_PATH: Final = "/api/v3/detect"
V3_KEY_PREFIX: Final = "sk_agt_"
V3_SESSION_HEADER: Final = "x-claude-code-session-id"
V3_CLIENT_HEADER: Final = "x-s6r-client"
V3_FORMAT_HEADER: Final = "x-s6r-format"
# (User-Agent prefix, entrypoints, Straiker client value, display name). Claude's clients send
# `claude-cli/<version> (<user type>, <entrypoint>[, agent-sdk/<version>])` on every call, title
# and topic side calls included, and the entrypoint names the product. Only Claude Code's are
# listed: the terminal CLI (`cli`), `claude -p` (`sdk-cli`), the VS Code extension and the desktop
# app's Code tab. Cowork and Chat in the desktop app send `local-agent`, and applications built on
# the Claude Agent SDK send `sdk-ts` or `sdk-py`. They share the prefix but are not Claude Code,
# so they are not named here.
_V3_CLIENT_BY_USER_AGENT: Final = (
    (
        "claude-cli/",
        frozenset({"cli", "sdk-cli", "claude-vscode", "claude-desktop", "claude-desktop-3p"}),
        "claude",
        "Claude",
    ),
)
V3_GATEWAY_NAME: Final = "LiteLLM"
V3_DERIVED_SESSION_PREFIX: Final = "litellm-"
V3_AGENT_HEADER: Final = "x-s6r-agent"
V3_RESPONSE_PHASE: Final = "response-sync"
V3_BLOCK_DECISIONS: Final = frozenset({"block", "deny"})
V3_UNDECIDED: Final = frozenset({"ask"})
V3_BLOCKED_TURN_MEMORY: Final = 10_000
V3_BLOCKED_TURN_TTL_SECONDS: Final = 24 * 60 * 60
# An allowlist: the hook's request dict merges the client body with proxy state (`deployment`
# carries the resolved credential), so only fields named here are relayed.
_V3_PROVIDER_BODY_KEYS: Final = frozenset(
    {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "functions",
        "function_call",
        "temperature",
        "top_p",
        "n",
        "stream",
        "stream_options",
        "stop",
        "max_tokens",
        "max_completion_tokens",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "response_format",
        "seed",
        "logprobs",
        "top_logprobs",
        "parallel_tool_calls",
        "reasoning_effort",
        "modalities",
        "audio",
        "prediction",
        "store",
        "service_tier",
        "web_search_options",
        "prompt",
        "suffix",
        "echo",
        "best_of",
        "system",
        "stop_sequences",
        "top_k",
        "thinking",
        "container",
        "mcp_servers",
        "context_management",
        "output_format",
        "input",
        "instructions",
        "previous_response_id",
        "truncation",
        "text",
        "include",
        "reasoning",
        "max_output_tokens",
        "background",
        "conversation",
        "session_id",
    }
)
# The scrub of these is one level deep on purpose: a function schema that defines a `token` or
# `headers` property lives under `function.parameters` and must be relayed as sent.
_V3_CREDENTIAL_FIELDS: Final = frozenset({"authorization_token", "authorization", "headers"})
_V3_REDACTED_VALUE: Final = "[redacted]"
_V3_REDACTED_KEYS: Final = frozenset({"tools", "mcp_servers"})
_V3_IDENTITY_METADATA_KEYS: Final = (
    "user_api_key_end_user_id",
    "user_api_key_user_email",
    "user_api_key_user_id",
    "user_api_key_alias",
    "user_api_key_team_id",
)
#: Statuses that mean Straiker could not give a verdict right now. They are retried and, once the
#: retries are spent, follow `unreachable_fallback` like a timeout or a connection error. Any
#: other status is an error (`fail_on_error`): a bad key or a rejected request does not change
#: on a retry.
TRANSIENT_STATUS: Final = frozenset({408, 429, 500, 502, 503, 504})
_APPLICATION_METADATA_KEYS: Final = frozenset({"agent_id", "app_name"})
_OPAQUE_METADATA_SCALAR_TYPES: Final = (str, int, float, bool)
_JSON_DICT_ADAPTER: Final = TypeAdapter(dict[str, object])


@dataclass(frozen=True, slots=True)
class _WebhookFailure:
    message: str
    is_unreachable: bool
    retryable: bool = False
    #: Straiker answered, but with no allow or block decision in the reply.
    no_verdict: bool = False


@dataclass(frozen=True, slots=True)
class _RememberedBlock:
    """A blocked prompt, kept so that a resend is blocked again and logged as a repeat of it."""

    message: str
    blocked_by: tuple[str, ...]
    turn_id: str | None


#: What one phase logs as `action`, in the labels Straiker's other gateway integrations log.
_V3LogAction: TypeAlias = Literal["allow", "detect", "block", "degraded", "unknown"]

#: LiteLLM's guardrail status for each logged action. A detect-mode finding is flagged, not
#: intervened: the request went through. A phase that was never scored failed to respond,
#: whether the request then failed open or closed.
_V3_LOG_STATUS: Final[Mapping[str, GuardrailStatus]] = MappingProxyType(
    {
        "allow": "success",
        "detect": "guardrail_flagged",
        "block": "guardrail_intervened",
        "degraded": "guardrail_failed_to_respond",
        "unknown": "guardrail_failed_to_respond",
    }
)


def _v3_log_details(
    parsed: StraikerWebhookResponse | None, replay_of: _RememberedBlock | None
) -> tuple[tuple[str, object], ...]:
    """The turn's details for the log entry. A phase that was never scored has none; a block
    repeated from this guardrail's memory names the turn it repeats and has no turn of its own."""
    if parsed is not None:
        return (
            ("turn_id", parsed.turn_id),
            ("controls", list(parsed.controls)),
            ("blocked_by", list(parsed.blocked_by)),
            ("events_scored", parsed.events_scored),
        )
    if replay_of is not None:
        return (("blocked_by", list(replay_of.blocked_by)), ("replay_of", replay_of.turn_id))
    return ()


def _v3_log_action(parsed: StraikerWebhookResponse) -> _V3LogAction:
    if parsed.action == "BLOCKED":
        return "block"
    return "detect" if parsed.verdict == "detect" else "allow"


def _status_failure(status: int, text: str) -> _WebhookFailure:
    return _WebhookFailure(
        f"HTTP {status}: {text[:200]}",
        is_unreachable=status in TRANSIENT_STATUS,
        retryable=status in TRANSIENT_STATUS,
    )


def _error_response_text(response: httpx.Response) -> str:
    try:
        return response.text
    except Exception:  # noqa: BLE001  # a masked response may carry no body
        return ""


class _JsonResponse(Protocol):
    def json(self) -> object: ...


def _json_body(response: _JsonResponse) -> object:
    return response.json()


def _as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _merged_metadata(request_data: Mapping[str, object]) -> dict:
    return {
        **_as_dict(request_data.get("metadata")),
        **_as_dict(request_data.get("litellm_metadata")),
    }


def _as_optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _build_webhook_metadata(request_data: dict, default_metadata: dict[str, str]) -> dict[str, object] | None:
    out: Final[dict[str, object]] = {}
    for key, value in _as_dict(request_data.get("metadata")).items():
        if key in _APPLICATION_METADATA_KEYS or key.startswith("user_api"):
            continue
        if key == "session_id":
            continue
        if isinstance(value, _OPAQUE_METADATA_SCALAR_TYPES):
            out[key] = value
    out.update(default_metadata)
    return out or None


def _extract_identity(request_data: dict) -> StraikerWebhookIdentity:
    meta: Final = _merged_metadata(request_data)
    return StraikerWebhookIdentity(
        litellm_key=_as_optional_str(meta.get("user_api_key_alias"))
        or _as_optional_str(meta.get("user_api_key_hash"))
        or _as_optional_str(meta.get("user_api_key_token")),
        litellm_team=_as_optional_str(meta.get("user_api_key_team_alias"))
        or _as_optional_str(meta.get("user_api_key_team_id")),
        litellm_user_id=_as_optional_str(meta.get("user_api_key_user_id")),
        litellm_user_email=_as_optional_str(meta.get("user_api_key_user_email")),
        litellm_org_id=_as_optional_str(meta.get("user_api_key_org_id")),
        end_user_id=_as_optional_str(meta.get("user_api_key_end_user_id")),
    )


def _resolve_provider(request_data: dict, model: str | None) -> str | None:
    litellm_params: Final = _as_dict(request_data.get("litellm_params"))
    custom_llm_provider: Final = request_data.get("custom_llm_provider") or litellm_params.get("custom_llm_provider")
    if custom_llm_provider:
        return custom_llm_provider
    if not model:
        return None
    try:
        _, provider, _, _ = get_llm_provider(
            model=model,
            api_base=request_data.get("api_base") or litellm_params.get("api_base"),
            api_key=request_data.get("api_key") or litellm_params.get("api_key"),
        )
    except BadRequestError:
        return None
    return provider or None


def _resolve_destination(request_data: dict) -> str | None:
    litellm_params: Final = _as_dict(request_data.get("litellm_params"))
    api_base: Final = request_data.get("api_base") or litellm_params.get("api_base")
    if not isinstance(api_base, str):
        return None
    try:
        return urlsplit(api_base).hostname
    except ValueError:
        return None


def _route_has_translation(request_data: dict) -> bool:
    from litellm.litellm_core_utils.api_route_to_call_types import get_call_types_for_route
    from litellm.llms import load_guardrail_translation_mappings

    route: Final = _as_dict(request_data.get("litellm_metadata")).get("user_api_key_request_route")
    if not isinstance(route, str) or not route:
        return False
    mappings: Final = load_guardrail_translation_mappings()
    return any(call_type in mappings for call_type in get_call_types_for_route(route) or ())


def _request_structured_messages(request_data: dict) -> list[dict[str, object]] | None:
    messages: Final = request_data.get("messages")
    if messages:
        return messages if isinstance(messages, list) else None
    if not _route_has_translation(request_data):
        return None
    return resolve_structured_messages(messages=None, request_kwargs=request_data)


def _hook_name(value: object) -> str:
    return value.value if isinstance(value, GuardrailEventHooks) else str(value)


def _configured_modes(event_hook: object) -> list[str] | None:
    if isinstance(event_hook, list):
        names = [_hook_name(v) for v in event_hook]
    elif isinstance(event_hook, (str, GuardrailEventHooks)):
        names = [_hook_name(event_hook)]
    elif isinstance(event_hook, Mode):
        default: Final = event_hook.default if isinstance(event_hook.default, list) else [event_hook.default]
        tags: Final = [v for value in event_hook.tags.values() for v in (value if isinstance(value, list) else [value])]
        names = [_hook_name(v) for v in (*default, *tags) if v is not None]
    else:
        return None
    return list(dict.fromkeys(names)) or None


#: The call types whose route answers a `ModifyResponseException` as a reply in the caller's
#: own API format. Every other caller of `apply_guardrail` (embeddings, the standalone
#: /guardrails/apply_guardrail check and others) turns it into HTTP 500.
_REPLY_CALL_SURFACES: Final = frozenset(
    {
        CallTypes.completion.value,
        CallTypes.acompletion.value,
        CallTypes.text_completion.value,
        CallTypes.atext_completion.value,
        CallTypes.anthropic_messages.value,
        CallTypes.responses.value,
        CallTypes.aresponses.value,
    }
)


def _resolve_call_surface(logging_obj: LiteLLMLoggingObj | None, request_data: dict) -> str:
    call_type: Final = (
        (getattr(logging_obj, "call_type", None) if logging_obj is not None else None)
        or request_data.get("call_type")
        or request_data.get("litellm_call_type")
    )
    return call_type if isinstance(call_type, str) and call_type else "unknown"


def _jsonable_dict(value: object) -> dict[str, object] | None:
    if isinstance(value, BaseModel):
        return _JSON_DICT_ADAPTER.validate_python(value.model_dump(mode="json", exclude_none=True))
    if isinstance(value, dict):
        return _JSON_DICT_ADAPTER.validate_python(value)
    return None


def _opaque_dict_list(value: object) -> list[dict[str, object]] | None:
    if not isinstance(value, list):
        return None
    items: Final = tuple(plain for item in value if (plain := _jsonable_dict(item)) is not None)
    return list(items) if items else None


def _choice_terminal_reason(choice: object) -> str | None:
    if isinstance(choice, dict):
        return _as_optional_str(choice.get("finish_reason")) or _as_optional_str(choice.get("stop_reason"))
    return _as_optional_str(getattr(choice, "finish_reason", None)) or _as_optional_str(
        getattr(choice, "stop_reason", None)
    )


def _response_finish_reason(response: object) -> str | None:
    if response is None:
        return None
    if isinstance(response, dict):
        top = _as_optional_str(response.get("finish_reason")) or _as_optional_str(response.get("stop_reason"))
        if top:
            return top
        choices = response.get("choices")
        if not isinstance(choices, list):
            return None
        for choice in choices:
            reason = _choice_terminal_reason(choice)
            if reason:
                return reason
        return None

    top = _as_optional_str(getattr(response, "finish_reason", None)) or _as_optional_str(
        getattr(response, "stop_reason", None)
    )
    if top:
        return top
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list):
        return None
    for choice in choices:
        reason = _choice_terminal_reason(choice)
        if reason:
            return reason
    return None


def _as_optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _usage_token_count(usage: object, openai_key: str, anthropic_key: str) -> int | None:
    get: Final = usage.get if isinstance(usage, dict) else lambda key: getattr(usage, key, None)
    openai_count: Final = _as_optional_int(get(openai_key))
    return openai_count if openai_count is not None else _as_optional_int(get(anthropic_key))


def _build_usage(response: object) -> StraikerWebhookUsage | None:
    usage: Final = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    if usage is None:
        return None
    input_tokens: Final = _usage_token_count(usage, "prompt_tokens", "input_tokens")
    output_tokens: Final = _usage_token_count(usage, "completion_tokens", "output_tokens")
    if input_tokens is None and output_tokens is None:
        return None
    return StraikerWebhookUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def _is_streamed_request(request_data: dict) -> bool:
    if request_data.get("stream") is True:
        return True
    body: Final = _as_dict(_as_dict(request_data.get("proxy_server_request")).get("body"))
    return body.get("stream") is True


# What the proxy stamps on a master-key call in place of a person. Sent onward, either
# would be recorded as an identity and every master-key turn filed under it.
_PLACEHOLDER_IDENTITIES: Final = frozenset({SpecialProxyStrings.default_user_id.value, "litellm_proxy_master_key"})


def _real_identity(value: object) -> str | None:
    """LiteLLM's proxy-admin placeholders are not a person."""
    identity: Final = _as_optional_str(value)
    return None if identity in _PLACEHOLDER_IDENTITIES else identity


def _request_header(request_data: Mapping[str, object], name: str | None) -> str | None:
    """A header from the inbound request, when LiteLLM kept it on the request data."""
    if not name:
        return None
    proxy_request: Final = request_data.get("proxy_server_request")
    headers: Final = proxy_request.get("headers") if isinstance(proxy_request, Mapping) else None
    if not isinstance(headers, Mapping):
        return None
    wanted: Final = name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted and isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _frozen(pairs: Iterable[tuple[str, object]]) -> Mapping[str, object]:
    return MappingProxyType(dict(pairs))


def _json_default(value: object) -> object:
    if isinstance(value, Mapping):
        return dict(value)
    return str(value)


def _v3_identity_metadata(request_data: Mapping[str, object]) -> Mapping[str, str]:
    """The proxy-resolved identity fields, and only those, for the relayed body."""
    merged: Final = _merged_metadata(request_data)
    return MappingProxyType(
        {key: value for key in _V3_IDENTITY_METADATA_KEYS if (value := _real_identity(merged.get(key)))}
    )


def _v3_request_body(request_data: Mapping[str, object], request_texts: Iterable[object] = ()) -> Mapping[str, object]:
    """The provider body LiteLLM received, stripped of everything the proxy added.

    The hook sees the client's request merged with proxy bookkeeping: logging objects,
    the resolved key, the inbound headers. Only the provider body is Straiker's to read,
    and the client's Authorization header must not travel. Identity survives as the
    metadata subset the Straiker LiteLLM adapter reads. A call with no provider body, such
    as /guardrails/apply_guardrail with only `text`, relays `request_texts` as user turns.
    """
    identity: Final = _v3_identity_metadata(request_data)
    turns: Final = (
        _v3_prompt_as_messages(request_data.get("prompt"))
        if _v3_text_completion_route(request_data) and "messages" not in request_data
        else None
    )
    texts: Final = tuple(text for text in request_texts if isinstance(text, str) and text)
    no_conversation: Final = (
        not request_data.get("messages") and "prompt" not in request_data and "input" not in request_data
    )
    text_turns: Final = (
        tuple(_frozen((("role", "user"), ("content", text))) for text in texts)
        if turns is None and no_conversation
        else ()
    )
    provider: Final = tuple(
        (key, _v3_without_credentials(value) if key in _V3_REDACTED_KEYS else value)
        for key, value in request_data.items()
        if key in _V3_PROVIDER_BODY_KEYS
        and not (turns is not None and key == "prompt")
        and not (text_turns and key == "messages")
    )
    prompt_turns: Final = (
        (("messages", turns),) if turns is not None else ((("messages", text_turns),) if text_turns else ())
    )
    return _frozen((*provider, *prompt_turns, *((("metadata", identity),) if identity else ())))


def _v3_without_credentials(entries: object) -> object:
    if not isinstance(entries, (list, tuple)):
        return entries
    return tuple(
        _frozen(
            (str(key), _V3_REDACTED_VALUE if str(key).lower() in _V3_CREDENTIAL_FIELDS else item)
            for key, item in entry.items()
        )
        if isinstance(entry, Mapping)
        else entry
        for entry in entries
    )


def _v3_route_is(request_data: Mapping[str, object], call_type: CallTypes) -> bool:
    from litellm.litellm_core_utils.api_route_to_call_types import get_call_types_for_route

    route: Final = _merged_metadata(request_data).get("user_api_key_request_route")
    if not isinstance(route, str) or not route:
        return False
    return call_type in (get_call_types_for_route(route) or ())


def _v3_anthropic_messages_route(request_data: Mapping[str, object]) -> bool:
    return _v3_route_is(request_data, CallTypes.anthropic_messages)


def _v3_text_completion_route(request_data: Mapping[str, object]) -> bool:
    return _v3_route_is(request_data, CallTypes.text_completion)


def _v3_is_token_list(value: object) -> bool:
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(isinstance(token, int) and not isinstance(token, bool) for token in value)
    )


def _v3_decode_tokens(tokens: Iterable[object]) -> str | None:
    ids: Final = [token for token in tokens if isinstance(token, int)]
    try:
        import tiktoken

        return tiktoken.encoding_for_model("text-davinci-003").decode(ids)
    except Exception:  # noqa: BLE001  # no tokenizer available: the raw prompt is relayed instead
        return None


def _v3_prompt_texts(prompt: object) -> tuple[str, ...] | None:
    """The text the model receives for a completions `prompt`, in the proxy's own terms.

    LiteLLM accepts a string, a list of strings, a list of token ids, or a list of token-id
    lists, and decodes token ids with the text-davinci-003 tokenizer before calling the model.
    The same decoding here means Straiker screens what the model gets. None when the prompt
    is a shape this cannot render, so the caller relays it untouched rather than screening
    something else.
    """
    if isinstance(prompt, str):
        return (prompt,)
    if not isinstance(prompt, (list, tuple)) or not prompt:
        return None
    if all(isinstance(item, str) for item in prompt):
        return tuple(str(item) for item in prompt)
    if _v3_is_token_list(prompt):
        decoded: Final = _v3_decode_tokens(prompt)
        return (decoded,) if decoded is not None else None
    if all(_v3_is_token_list(item) for item in prompt):
        decoded_each: Final = tuple(_v3_decode_tokens(item) for item in prompt)
        return None if any(text is None for text in decoded_each) else tuple(text or "" for text in decoded_each)
    return None


def _v3_prompt_as_messages(prompt: object) -> tuple[Mapping[str, object], ...] | None:
    texts: Final = _v3_prompt_texts(prompt)
    if texts is None:
        return None
    return tuple(_frozen((("role", "user"), ("content", text))) for text in texts)


def _v3_answer(request_data: Mapping[str, object], model: str | None) -> Mapping[str, object] | None:
    """The answer in the API shape the client spoke, which is what a relay forwards.

    On a streamed Messages call the proxy rebuilds the answer as a chat completion before
    the hook runs. Straiker's coding-agent reader parses a Messages answer, so a Claude Code
    turn sent as a chat completion scores nothing; the proxy's own adapter turns it back.
    """
    response: Final = request_data.get("response")
    if isinstance(response, TextCompletionResponse):
        return _v3_text_completion_as_chat(response)
    if not isinstance(response, ModelResponse) or not _v3_anthropic_messages_route(request_data):
        return _jsonable_dict(response)
    from litellm.llms.anthropic.messages.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )

    translated: Final = LiteLLMAnthropicMessagesAdapter().translate_openai_response_to_anthropic(response=response)
    re_keyed: Final = dict(translated, model=response.model or model)
    return _jsonable_dict(re_keyed)


def _v3_text_completion_as_chat(response: TextCompletionResponse) -> Mapping[str, object]:
    """A legacy completion answer in the chat shape the platform scores.

    Straiker has no reader for a `text_completion` answer on a gateway: the request phase
    of a /v1/completions call is scored, the response phase is refused. A completion is one
    user turn and one assistant turn, so both phases are presented as that exchange.
    """
    choices: Final = tuple(
        _frozen(
            (
                ("index", index),
                ("finish_reason", getattr(choice, "finish_reason", None)),
                ("message", _frozen((("role", "assistant"), ("content", getattr(choice, "text", "") or "")))),
            )
        )
        for index, choice in enumerate(response.choices)
    )
    usage: Final = _jsonable_dict(getattr(response, "usage", None))
    return _frozen(
        (
            ("id", response.id),
            ("object", "chat.completion"),
            ("created", response.created),
            ("model", response.model),
            ("choices", choices),
            *((("usage", usage),) if usage else ()),
        )
    )


def _v3_answer_json(
    inputs: GenericGuardrailAPIInputs, request_data: Mapping[str, object], model: str | None
) -> str | None:
    """The model's answer as the raw response body Straiker parses on the response phase.

    The real response object carries tool calls, which a coding-agent turn is scored on,
    so it is preferred. A streamed answer reaches the hook already assembled into texts,
    and those become a minimal chat completion so the answer is still scored.
    """
    response: Final = _v3_answer(request_data, model)
    if response:
        return json.dumps(response, default=_json_default)
    texts: Final = tuple(t for t in (inputs.get("texts") or []) if t)
    if not texts:
        return None
    message: Final = _frozen((("role", "assistant"), ("content", "\n".join(texts))))
    choice: Final = _frozen((("index", 0), ("finish_reason", "stop"), ("message", message)))
    return json.dumps(_frozen((("object", "chat.completion"), ("choices", (choice,)))), default=_json_default)


def _v3_payload(
    envelope: StraikerWebhookRequest,
    inputs: GenericGuardrailAPIInputs,
    request_data: Mapping[str, object],
    input_type: Literal["request", "response"],
    request_body: Mapping[str, object],
) -> Mapping[str, object]:
    """The /api/v3/detect body for one phase of a turn, the unified Kong plugin's contract.

    Request phase: the provider body itself. Response phase: the answer beside the request
    it answers, `{straiker_phase, sse, model, request}`, which is how Straiker classifies a
    tool call the model just made. Straiker parses either and derives prompt, answer, agent
    and archetype from the traffic; nothing is pre-digested here. Identity and session ride
    on both phases the way Kong sends them.
    """
    context: Final = envelope.context
    answer_json: Final = _v3_answer_json(inputs, request_data, context.model) if input_type == "response" else None
    phase: Final = (
        tuple(request_body.items())
        if input_type == "request"
        else (
            ("straiker_phase", V3_RESPONSE_PHASE),
            ("model", context.model),
            ("request", request_body),
            *((("sse", answer_json),) if answer_json is not None else ()),
        )
    )
    session: Final = _v3_session_id(envelope, request_data, request_body)
    user: Final = _v3_user(envelope)
    return _frozen(
        (
            *phase,
            *((("session_id", session),) if session else ()),
            *(
                (("original", _frozen((("processed", _frozen((("Meta", _frozen((("user", user),))),))),))),)
                if user
                else ()
            ),
        )
    )


def _v3_conversation_prefixes(request_body: Mapping[str, object]) -> tuple[str, ...]:
    """A fingerprint of the conversation after each of its messages, first to last.

    The last one names the conversation as sent; the earlier ones let a request that
    carries a blocked exchange as its history be recognised, not only an exact resend.
    A `prompt` or a string `input` has one fingerprint.
    """
    messages: Final = _v3_messages(request_body)
    if messages:
        digest: Final = hashlib.sha256()

        def after(message: object) -> str:
            digest.update(json.dumps(message, sort_keys=True, default=str).encode("utf-8"))
            digest.update(b"\x1e")
            return digest.copy().hexdigest()

        return tuple(after(message) for message in messages)
    plain: Final = request_body.get("input") if "input" in request_body else request_body.get("prompt")
    if plain is None:
        return ()
    return (hashlib.sha256(json.dumps(plain, sort_keys=True, default=str).encode("utf-8")).hexdigest(),)


def _v3_session_id(
    envelope: StraikerWebhookRequest,
    request_data: Mapping[str, object],
    request_body: Mapping[str, object],
) -> str | None:
    """A stable id for the conversation, in Kong's order of precedence.

    Claude Code names its session on the wire and that wins. Then the session LiteLLM
    resolved from its own metadata. Then, for a conversation that states none, a hash of
    the principal, the system prompt and the first message: a chat client replays the
    whole conversation on every turn, so that triple is constant for its lifetime and
    groups the turns. A fresh synthetic id per request would group nothing.

    The principal is in the hash because Straiker skips turns it has already scored for a
    session. Two users who open with the same words are two conversations; hashed on the
    words alone they shared one session, and the second user's copy of an attack came
    back as a replay, unscored and allowed (measured 2026-09-20).
    """
    supplied: Final = _request_header(request_data, V3_SESSION_HEADER)
    if supplied:
        return supplied
    if envelope.context.session_id:
        return envelope.context.session_id
    conversation: Final = f"{_v3_system_text(request_body) or ''}\0{_v3_first_message_text(request_body)}"
    if conversation == "\0":
        return None
    seed: Final = f"{_v3_user(envelope) or ''}\0{conversation}"
    return V3_DERIVED_SESSION_PREFIX + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


_V3_PREAMBLE_ROLES: Final = frozenset({"system", "developer"})


def _v3_message_text(message: object) -> str:
    """Every text block of a message, so a turn that opens with an image or a document still
    seeds on what the user wrote."""
    content: Final = message.get("content") if isinstance(message, Mapping) else None
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "\n".join(
            str(block["text"]) for block in content if isinstance(block, Mapping) and isinstance(block.get("text"), str)
        )
    return ""


def _v3_messages(request_body: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    messages: Final = request_body.get("messages") or request_body.get("input")
    if isinstance(messages, (list, tuple)):
        return tuple(message for message in messages if isinstance(message, Mapping))
    return ()


def _v3_system_text(request_body: Mapping[str, object]) -> str | None:
    """The preamble, wherever the API puts it: Anthropic's `system`, the Responses API's
    `instructions`, or the leading system or developer message of an OpenAI chat body."""
    system: Final = request_body.get("system")
    if isinstance(system, str):
        return system
    if system is not None:
        return json.dumps(system, default=str)
    instructions: Final = request_body.get("instructions")
    if isinstance(instructions, str):
        return instructions
    preamble: Final = next((m for m in _v3_messages(request_body) if m.get("role") in _V3_PREAMBLE_ROLES), None)
    return _v3_message_text(preamble) if preamble is not None else None


def _v3_first_message_text(request_body: Mapping[str, object]) -> str:
    """What the user first said: the first `user` message, never the system prompt that an
    OpenAI chat body carries as `messages[0]`, else a Responses `input` string, else `prompt`."""
    first_user: Final = next((m for m in _v3_messages(request_body) if m.get("role") == "user"), None)
    if first_user is not None:
        return _v3_message_text(first_user)
    plain: Final = (
        request_body.get("input") if isinstance(request_body.get("input"), str) else request_body.get("prompt")
    )
    return plain if isinstance(plain, str) else ""


def _v3_user(envelope: StraikerWebhookRequest) -> str | None:
    """Who is asking: the key's own user first, then the end user the request named.

    The key is the authenticated principal, the way a Kong consumer is, so a per-user key
    names the person even when the client packs something else into the body. Claude Code
    packs a hashed account-and-session token into `metadata.user_id`, which is what the end
    user resolves to when nothing better is set; it is a session, not a person, and only
    surfaces when the key names nobody. A master-key call resolves to LiteLLM's
    `default_user_id`; sent as an identity it would become one.
    """
    identity: Final = envelope.identity
    for candidate in (identity.litellm_user_email, identity.litellm_user_id, identity.end_user_id):
        real = _real_identity(candidate)
        if real:
            return real
    return None


def _v3_user_agent_comment(user_agent: str) -> frozenset[str]:
    """The comma-separated fields inside a User-Agent's first `( ... )`, e.g. `cli`."""
    _, _, rest = user_agent.partition("(")
    return frozenset(field.strip() for field in rest.partition(")")[0].split(",") if field.strip())


def _v3_client_from_user_agent(request_data: Mapping[str, object]) -> tuple[str, str] | None:
    """`(client, agent name)` for a User-Agent this gateway recognises, else None.

    The prefix alone is not enough: the entrypoint inside the parentheses has to be one of the
    product's own, so a client that shares the prefix is left unnamed.
    """
    user_agent: Final = (_request_header(request_data, "user-agent") or "").strip().lower()
    comment: Final = _v3_user_agent_comment(user_agent)
    return next(
        (
            (client, f"{display} ({V3_GATEWAY_NAME})")
            for prefix, entrypoints, client, display in _V3_CLIENT_BY_USER_AGENT
            if user_agent.startswith(prefix) and not comment.isdisjoint(entrypoints)
        ),
        None,
    )


def _v3_headers(
    request_data: Mapping[str, object],
    agent_ref: str | None = None,
    client: str | None = None,
    format_hint: str | None = None,
) -> Mapping[str, str]:
    """Per-call routing hints, the unified Kong plugin's set. All optional.

    `x-s6r-agent` names ONE application when a gateway fronts several: the route's
    `agent_ref`, else the caller's own header, else the agent this gateway names from a Claude
    Code User-Agent. The operator's value comes first because the header is caller-supplied,
    and honouring it over a pinned route would let any key file its traffic under another
    application's agent and controls. `x-s6r-client` is the route's `client` config, else the
    client the User-Agent names. Cowork, Chat and Claude Agent SDK applications share Claude
    Code's User-Agent prefix but not its entrypoints, so they get neither header and Straiker
    identifies them from the request. `x-s6r-format` comes from config alone. Claude Code's own
    session header is forwarded when the client sent it, which is how a coding session groups
    the way the native hook would.
    """
    session: Final = _request_header(request_data, V3_SESSION_HEADER)
    recognised: Final = _v3_client_from_user_agent(request_data)
    agent: Final = (
        agent_ref or _request_header(request_data, V3_AGENT_HEADER) or (recognised[1] if recognised else None)
    )
    named_client: Final = client or (recognised[0] if recognised else None)
    candidates: Final = (
        (V3_SESSION_HEADER, session),
        (V3_AGENT_HEADER, agent),
        (V3_CLIENT_HEADER, named_client),
        (V3_FORMAT_HEADER, format_hint),
    )
    return MappingProxyType({name: value for name, value in candidates if value})


def _v3_decision(body: Mapping[str, object]) -> tuple[str | None, Mapping[str, object]]:
    """``(decision, verdict)``: the enforceable decision and the object carrying it.

    Straiker answers in two envelopes. A relayed body gets the hook contract,
    `hookSpecificOutput.permissionDecision`, with the flat fields nested under `straiker`;
    a flat call answers `action` at the top level. Reading only one of them would silently
    make block mode a no-op on the other.
    """
    nested: Final = body.get("straiker")
    verdict: Final = nested if isinstance(nested, Mapping) else body
    hook: Final = body.get("hookSpecificOutput")
    decision: Final = hook.get("permissionDecision") if isinstance(hook, Mapping) else None
    if isinstance(decision, str) and decision:
        return decision.lower(), verdict
    action: Final = verdict.get("action")
    return (action.lower() if isinstance(action, str) and action else None), verdict


def _v3_control_list(verdict: Mapping[str, object], key: Literal["controls", "blocked_by"]) -> tuple[str, ...]:
    raw: Final = verdict.get(key)
    return tuple(sorted(str(control) for control in raw)) if isinstance(raw, list) else ()


def _v3_blocked_by(verdict: Mapping[str, object]) -> tuple[str, ...]:
    return _v3_control_list(verdict, "blocked_by")


def _v3_events_scored(verdict: Mapping[str, object]) -> int | None:
    raw: Final = verdict.get("events_scored")
    return raw if isinstance(raw, int) and not isinstance(raw, bool) else None


def _v3_response(body: Mapping[str, object]) -> StraikerWebhookResponse | None:
    """Map a v3 verdict onto the action the guardrail already acts on.

    A detect-mode control fires into `controls` without changing the decision, so it
    correctly reads NONE. `blocked_by` is the block-mode subset and is honoured even if a
    build answers it without flipping the decision. None when Straiker stated no verdict:
    a missing decision, or `ask`, which a gateway has no one to put to.
    """
    decision, verdict = _v3_decision(body)
    blocked_by: Final = _v3_blocked_by(verdict)
    if (decision is None or decision in V3_UNDECIDED) and not blocked_by:
        return None
    blocked: Final = decision in V3_BLOCK_DECISIONS or bool(blocked_by)
    stated: Final = (verdict.get("block_message"), verdict.get("deny_reason"), body.get("stopReason"))
    reason: Final = (
        next(
            (text.strip() for text in stated if isinstance(text, str) and text.strip()),
            f"Straiker blocked this turn: {', '.join(blocked_by) or 'policy'}",
        )
        if blocked
        else None
    )
    stated_action: Final = verdict.get("action")
    detected: Final = isinstance(stated_action, str) and stated_action.lower() == "detect"
    return StraikerWebhookResponse(
        action="BLOCKED" if blocked else "NONE",
        blocked_reason=reason,
        blocked_by=blocked_by,
        turnId=_as_optional_str(verdict.get("turn_id")) or _as_optional_str(body.get("turn_id")),
        verdict="block" if blocked else ("detect" if detected else "allow"),
        controls=_v3_control_list(verdict, "controls"),
        events_scored=_v3_events_scored(verdict),
        session_id=_as_optional_str(verdict.get("session_id")),
    )


class StraikerGuardrail(CustomGuardrail):
    @staticmethod
    def get_config_model() -> type[GuardrailConfigModel]:
        return StraikerGuardrailConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
        ]

    def __init__(
        self,
        api_key: str,
        api_base: str = DEFAULT_API_BASE,
        api_version: Literal["v1", "v3"] | None = None,
        agent_ref: str | None = None,
        client: str | None = None,
        format_hint: Literal["anthropic.messages", "openai.chat"] | None = None,
        source: str = "LiteLLM Gateway",
        timeout: float = 5.0,
        max_retries: int = 2,
        initial_backoff: float = 0.1,
        max_backoff: float = 2.0,
        unreachable_fallback: Literal["fail_open", "fail_closed"] = "fail_closed",
        fail_on_error: bool = True,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        stream_mode: StraikerStreamMode = "buffered",
        custom_headers: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
        verbose: bool = False,
        async_handler: httpx.AsyncClient | None = None,
        **kwargs: object,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be non-empty")
        if unreachable_fallback not in ("fail_open", "fail_closed"):
            raise ValueError(f"unreachable_fallback must be 'fail_open' or 'fail_closed'; got {unreachable_fallback!r}")
        if stream_mode not in ("buffered", "streaming"):
            raise ValueError(f"stream_mode must be 'buffered' or 'streaming'; got {stream_mode!r}")
        if api_version not in (None, "v1", "v3"):
            raise ValueError(f"api_version must be 'v1' or 'v3'; got {api_version!r}")
        # The v1 webhook rejects an sk_agt_ key, so an sk_agt_ key always means v3. Guardrails
        # saved on 1.101.3 or older carry api_version 'v1' from the old shared default.
        is_v3_key: Final = api_key.startswith(V3_KEY_PREFIX)
        if is_v3_key and api_version == "v1":
            verbose_proxy_logger.warning(
                "Straiker guardrail: api_version 'v1' cannot use an sk_agt_ key, routing to /api/v3/detect"
            )

        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.api_version: Literal["v1", "v3"] = "v3" if is_v3_key else (api_version or "v1")
        self.agent_ref = _as_optional_str(agent_ref)
        self.client = _as_optional_str(client)
        if format_hint is not None and format_hint not in ("anthropic.messages", "openai.chat"):
            raise ValueError(f"format_hint must be 'anthropic.messages' or 'openai.chat'; got {format_hint!r}")
        self.format_hint = format_hint
        # Blocked conversations by session, so a resend or a conversation grown past a blocked
        # turn is blocked again here: Straiker de-duplicates turns it has already scored per
        # session and answers a replay `allow`, whatever the original verdict was (measured
        # 2026-09-20). Per process; a replica that did not see the block asks Straiker.
        self._v3_blocked_turns = InMemoryCache(
            max_size_in_memory=V3_BLOCKED_TURN_MEMORY, default_ttl=V3_BLOCKED_TURN_TTL_SECONDS
        )
        self.source = source
        self.max_retries = max(0, int(max_retries))
        self.initial_backoff = max(0.0, float(initial_backoff))
        self.max_backoff = max(self.initial_backoff, float(max_backoff))
        self.unreachable_fallback = unreachable_fallback
        self.fail_on_error = fail_on_error
        self.max_payload_bytes = int(max_payload_bytes)
        self.custom_headers = dict(custom_headers) if custom_headers else {}
        self.default_metadata = dict(metadata) if metadata else {}
        self.verbose = bool(verbose)

        # Both modes check a streamed response once, assembled, at the end of the stream:
        # LiteLLM's per-chunk sampling would post a growing partial answer every few chunks.
        # `buffered` withholds every chunk until that check passes. `streaming` releases chunks
        # as they arrive and holds back only the stream's final frames, which a block replaces
        # with the block message. These are the flags LiteLLM's streaming hook reads; they are
        # set from `stream_mode` alone rather than exposed as options of their own.
        self.stream_mode: StraikerStreamMode = stream_mode
        self.streaming_end_of_stream_only = True
        self.streaming_buffer_until_moderated = stream_mode == "buffered"

        self.async_handler = async_handler or get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )

        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))
        super().__init__(**kwargs)  # pyright: ignore[reportArgumentType]  # kwargs splat carries object-typed values
        self.timeout = float(timeout)

        self.configured_modes = _configured_modes(self.event_hook)

    def _webhook_url(self) -> str:
        return f"{self.api_base}{V3_DETECT_PATH if self.api_version == 'v3' else WEBHOOK_PATH}"

    def _headers(self) -> dict[str, str]:
        reserved: Final = {"authorization", "content-type", "x-straiker-webhook-format"}
        extra: Final = {k: v for k, v in self.custom_headers.items() if k.lower() not in reserved}
        headers: Final = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.api_version != "v3":
            headers["X-Straiker-Webhook-Format"] = "litellm"
        return {**headers, **extra}

    def _build_application(self, request_data: dict) -> StraikerWebhookApplication:
        meta: Final = _merged_metadata(request_data)
        agent_id: Final = _as_optional_str(meta.get("agent_id"))
        return StraikerWebhookApplication(
            source=agent_id or self.source,
            name=_as_optional_str(meta.get("app_name")),
        )

    def _build_context(
        self,
        request_data: dict,
        model: str | None,
        logging_obj: LiteLLMLoggingObj | None,
    ) -> StraikerWebhookContext:
        return StraikerWebhookContext(
            call_surface=_resolve_call_surface(logging_obj, request_data),
            mode=self.configured_modes,
            model=model,
            model_provider=_resolve_provider(request_data, model),
            destination=_resolve_destination(request_data),
            session_id=get_session_id_from_request_data(request_data),
            litellm_call_id=getattr(logging_obj, "litellm_call_id", None) if logging_obj else None,
            litellm_trace_id=getattr(logging_obj, "litellm_trace_id", None) if logging_obj else None,
            litellm_version=litellm_version,
        )

    def _build_envelope(
        self,
        *,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None,
    ) -> StraikerWebhookRequest:
        model: Final = inputs.get("model") or request_data.get("model")
        call_id: Final = getattr(logging_obj, "litellm_call_id", None) if logging_obj else None
        event_id: Final = f"{call_id or 'litellm'}:{input_type}"

        content: Final = StraikerWebhookContent(
            texts=list(inputs.get("texts") or []),
            images=list(inputs.get("images") or []),
            structured_messages=_opaque_dict_list(inputs.get("structured_messages")),
            tools=_opaque_dict_list(inputs.get("tools")),
            tool_calls=_opaque_dict_list(inputs.get("tool_calls")),
        )

        if input_type == "request":
            event = StraikerWebhookEvent(type="pre_call", id=event_id)
            return StraikerWebhookRequest(
                event=event,
                request=content,
                context=self._build_context(request_data, model, logging_obj),
                identity=_extract_identity(request_data),
                application=self._build_application(request_data),
                metadata=_build_webhook_metadata(request_data, self.default_metadata),
            )

        response_obj: Final = request_data.get("response")
        content.finish_reason = _response_finish_reason(response_obj)
        request_content: Final = StraikerWebhookContent(
            structured_messages=_opaque_dict_list(_request_structured_messages(request_data)),
        )
        phase: Final[Literal["none", "assembled"]] = "assembled" if _is_streamed_request(request_data) else "none"
        event = StraikerWebhookEvent(type="post_call", id=event_id, stream=StraikerWebhookStream(phase=phase))
        return StraikerWebhookRequest(
            event=event,
            request=request_content,
            response=content,
            context=self._build_context(request_data, model, logging_obj),
            identity=_extract_identity(request_data),
            application=self._build_application(request_data),
            usage=_build_usage(response_obj),
            metadata=_build_webhook_metadata(request_data, self.default_metadata),
        )

    async def _post_webhook(
        self, payload: Mapping[str, object], headers: Mapping[str, str] | None = None
    ) -> tuple[StraikerWebhookResponse | None, _WebhookFailure | None]:
        try:
            body: Final = json.dumps(payload, default=_json_default).encode("utf-8")
        except (TypeError, ValueError, OverflowError) as error:
            return None, _WebhookFailure(f"request serialization failed: {error}", is_unreachable=False)
        body_bytes: Final = len(body)
        if body_bytes > self.max_payload_bytes:
            return None, _WebhookFailure(
                f"payload {body_bytes}B exceeds max_payload_bytes {self.max_payload_bytes}",
                is_unreachable=False,
            )

        url: Final = self._webhook_url()
        merged_headers: Final = {**self._headers(), **(headers or {})}
        attempts: Final = self.max_retries + 1
        last_failure: _WebhookFailure | None = None

        if self.verbose:
            verbose_proxy_logger.info(
                json.dumps(
                    {
                        "event": "straiker.webhook_request",
                        "url": url,
                        "bytes": body_bytes,
                        "payload": payload,
                    },
                    default=_json_default,
                )
            )

        for attempt in range(attempts):
            parsed, last_failure = await self._attempt(url, body, merged_headers)
            if last_failure is None or not last_failure.retryable:
                return parsed, last_failure
            if attempt < attempts - 1:
                backoff: float = min(self.initial_backoff * (1 << attempt), self.max_backoff)
                await asyncio.sleep(random.uniform(0, backoff))

        return None, last_failure or _WebhookFailure("unknown error", is_unreachable=True)

    async def _attempt(
        self, url: str, body: bytes, headers: dict[str, str]
    ) -> tuple[StraikerWebhookResponse | None, _WebhookFailure | None]:
        try:
            resp: Final = await self.async_handler.post(url, content=body, headers=headers, timeout=self.timeout)
        except httpx.HTTPStatusError as status_error:
            return None, _status_failure(status_error.response.status_code, _error_response_text(status_error.response))
        except (httpx.TimeoutException, asyncio.TimeoutError, Timeout) as e:
            # Unreachable, but not retried: the attempt has already waited the whole `timeout`,
            # so a retry would multiply the delay before a verdict rather than shorten it.
            # Before httpx.RequestError, which httpx.TimeoutException subclasses.
            return None, _WebhookFailure(f"{type(e).__name__}: {e}", is_unreachable=True)
        except httpx.RequestError as e:
            return None, _WebhookFailure(f"{type(e).__name__}: {e}", is_unreachable=True, retryable=True)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            return None, _WebhookFailure(f"{type(e).__name__}: {e}", is_unreachable=False)
        if resp is None:
            return None, _WebhookFailure("no response", is_unreachable=True, retryable=True)
        if resp.status_code == 200:
            return self._parse_verdict(resp)
        return None, _status_failure(resp.status_code, resp.text)

    def _parse_verdict(self, resp: httpx.Response) -> tuple[StraikerWebhookResponse | None, _WebhookFailure | None]:
        try:
            body: Final = _json_body(resp)
            body_mapping: Final = as_str_mapping(body)
            if body_mapping is None:
                return None, _WebhookFailure(
                    f"invalid response schema: expected an object, got {type(body).__name__}", is_unreachable=False
                )
            parsed: Final = (
                _v3_response(body_mapping)
                if self.api_version == "v3"
                else StraikerWebhookResponse.model_validate(body_mapping)
            )
        except (ValidationError, json.JSONDecodeError) as ve:
            return None, _WebhookFailure(f"invalid response schema: {ve}", is_unreachable=False)
        if parsed is None:
            return None, _WebhookFailure(
                "invalid response schema: no allow or block decision", is_unreachable=False, no_verdict=True
            )
        if self.verbose:
            verbose_proxy_logger.info(
                json.dumps(
                    {"event": "straiker.webhook_response", "status_code": resp.status_code, "body": body},
                    default=_json_default,
                )
            )
        return parsed, None

    def _record(
        self,
        *,
        request_data: dict,
        logging_obj: LiteLLMLoggingObj | None,
        parsed: StraikerWebhookResponse,
    ) -> None:
        if not self.verbose:
            return
        response_obj: Final = request_data.get("response")
        hidden: Final = getattr(response_obj, "_hidden_params", None)
        if isinstance(hidden, dict):
            straiker_hidden: Final = hidden.setdefault("straiker", {})
            if isinstance(straiker_hidden, dict):
                straiker_hidden.update({"action": parsed.action, "turn_id": parsed.turn_id})

    def _log_v3_phase(
        self,
        *,
        request_data: dict[str, object],  # mutable-ok: the logging helper appends to it in place
        input_type: Literal["request", "response"],
        action: _V3LogAction,
        started: float,
        session: str | None,
        parsed: StraikerWebhookResponse | None = None,
        replay_of: _RememberedBlock | None = None,
    ) -> None:
        """Add this phase's verdict to LiteLLM's standard logging payload, which every logging
        callback and the spend logs read, with the field names and labels Straiker's other gateway
        integrations log. It takes the place of the generic entry `log_guardrail_information`
        would otherwise write, which reads `allow` even when the guardrail failed open.

        `guardrail_response` holds the session, the action and the turn's details. The action
        and the controls are also set as `guardrail_action` and `violation_categories`, which
        spend logs keep when they redact `guardrail_response`.
        """
        entry: Final = {
            key: value
            for key, value in (
                ("session_id", (parsed.session_id if parsed is not None else None) or session),
                ("action", action),
                *_v3_log_details(parsed, replay_of),
            )
            if value is not None
        }
        fired: Final = parsed.controls if parsed is not None else ()
        blocked_by: Final = (
            parsed.blocked_by if parsed is not None else (replay_of.blocked_by if replay_of is not None else ())
        )
        violations: Final = sorted({*fired, *blocked_by})
        tracing: Final[GuardrailTracingDetail] = (
            {"guardrail_action": action, "violation_categories": violations}
            if violations
            else {"guardrail_action": action}
        )
        ended: Final = time.time()
        # Logging must never decide the request: if this fails, the decorator writes its
        # generic entry instead and the verdict is enforced as usual.
        try:
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_json_response=entry,
                request_data=request_data,
                guardrail_status=_V3_LOG_STATUS[action],
                start_time=started,
                end_time=ended,
                duration=ended - started,
                guardrail_provider=GUARDRAIL_NAME,
                event_type=GuardrailEventHooks.pre_call if input_type == "request" else GuardrailEventHooks.post_call,
                tracing_detail=tracing,
            )
        except Exception as error:  # noqa: BLE001  # a logging failure must not change the verdict
            verbose_proxy_logger.warning(
                json.dumps({"event": "straiker.log_failed", "input_type": input_type, "error": str(error)})
            )

    def _fail(
        self,
        *,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        error: str,
        is_unreachable: bool,
    ) -> GenericGuardrailAPIInputs:
        fail_open: Final = (is_unreachable and self.unreachable_fallback == "fail_open") or not self.fail_on_error
        verbose_proxy_logger.error(
            json.dumps(
                {
                    "event": "straiker.error",
                    "input_type": input_type,
                    "error": error,
                    "fail_open": fail_open,
                },
                default=_json_default,
            )
        )
        if fail_open:
            return inputs
        self._block(
            request_data=request_data,
            input_type=input_type,
            message=f"Straiker detection unavailable: {error}",
        )

    def _block(
        self,
        *,
        request_data: dict,
        input_type: Literal["request", "response"],
        message: str,
        blocked_content: bool = False,
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> NoReturn:
        # A verdict on the prompt is answered as a reply, the way an answer-side block already
        # is. GuardrailRaisedException reaches the client as HTTP 400, which some clients
        # report as a malformed request without the reason. A backend failure stays an error,
        # so a consumer can still tell an outage from a verdict, and so does a verdict on a
        # route that has no reply to give.
        replies: Final = blocked_content and _resolve_call_surface(logging_obj, request_data) in _REPLY_CALL_SURFACES
        if input_type == "request" and not replies:
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name or GUARDRAIL_NAME,
                message=message,
                should_wrap_with_default_message=False,
                blocked_content=blocked_content,
            )
        reply: Final = ModifyResponseException(
            message=message,
            model=request_data.get("model", "unknown") or "unknown",
            request_data=request_data,
            guardrail_name=self.guardrail_name or GUARDRAIL_NAME,
            original_response=request_data.get("response") if input_type == "response" else None,
        )
        # A route that cannot send a reply maps this exception to an HTTP error and reads the
        # status from it. Without one the block is reported as a 500.
        reply.status_code = 400  # pyright: ignore[reportAttributeAccessIssue]  # the class declares no status
        raise reply

    @staticmethod
    def _intervened_inputs(
        inputs: GenericGuardrailAPIInputs,
        parsed: StraikerWebhookResponse,
    ) -> GenericGuardrailAPIInputs:
        return_inputs: Final[GenericGuardrailAPIInputs] = {}
        return_inputs.update(inputs)
        if parsed.texts is not None:
            return_inputs["texts"] = parsed.texts
        return return_inputs

    async def _apply_v3(
        self,
        *,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None,
    ) -> GenericGuardrailAPIInputs:
        """One phase of a turn against /api/v3/detect: relay, read the decision, log it, enforce."""
        started: Final = time.time()
        try:
            envelope: Final = self._build_envelope(
                inputs=inputs,
                request_data=request_data,
                input_type=input_type,
                logging_obj=logging_obj,
            )
            request_body: Final = _v3_request_body(
                request_data, (inputs.get("texts") or ()) if input_type == "request" else ()
            )
            payload: Final = _v3_payload(envelope, inputs, request_data, input_type, request_body)
            headers: Final = _v3_headers(request_data, self.agent_ref, self.client, self.format_hint)
            # The memory is scoped by the principal (the user, else the key) and the session
            # together; a request that has neither is never remembered, so two callers never
            # share a block.
            session: Final = _v3_session_id(envelope, request_data, request_body)
            principal: Final = _v3_user(envelope) or envelope.identity.litellm_key
            scope: Final = f"{principal or ''}\0{session or ''}" if session or principal else ""
            prefixes: Final = _v3_conversation_prefixes(request_body) if scope else ()
        except (ValidationError, TypeError, ValueError) as error:
            self._log_v3_phase(
                request_data=request_data, input_type=input_type, action="degraded", started=started, session=None
            )
            return self._fail(
                inputs=inputs,
                request_data=request_data,
                input_type=input_type,
                error=str(error),
                is_unreachable=False,
            )

        replayed: Final = self._v3_replayed_block(scope, prefixes) if input_type == "request" else None
        if replayed is not None:
            self._log_v3_phase(
                request_data=request_data,
                input_type=input_type,
                action="block",
                started=started,
                session=session,
                replay_of=replayed,
            )
            self._block(
                request_data=request_data,
                input_type=input_type,
                message=replayed.message,
                blocked_content=True,
                logging_obj=logging_obj,
            )

        parsed, failure = await self._post_webhook(payload, headers)
        if failure is not None or parsed is None:
            self._log_v3_phase(
                request_data=request_data,
                input_type=input_type,
                action="unknown" if failure is not None and failure.no_verdict else "degraded",
                started=started,
                session=session,
            )
            return self._fail(
                inputs=inputs,
                request_data=request_data,
                input_type=input_type,
                error=failure.message if failure is not None else "empty response from Straiker",
                is_unreachable=failure.is_unreachable if failure is not None else False,
            )
        self._record(request_data=request_data, logging_obj=logging_obj, parsed=parsed)
        self._log_v3_phase(
            request_data=request_data,
            input_type=input_type,
            action=_v3_log_action(parsed),
            started=started,
            session=session,
            parsed=parsed,
        )
        if parsed.action == "BLOCKED":
            message: Final = parsed.blocked_reason or DEFAULT_BLOCK_MESSAGE
            # Only a block that names a control is remembered. The same words are the same
            # attack tomorrow, but a block that comes from state -- an engaged kill switch,
            # a governance action -- is lifted by an administrator, and a remembered copy
            # would keep refusing a conversation the platform now allows. A blocked answer is
            # not remembered: the question that produced it may be harmless.
            if input_type == "request" and prefixes and parsed.blocked_by:
                self._v3_blocked_turns.set_cache(
                    f"{scope}\0{prefixes[-1]}", _RememberedBlock(message, parsed.blocked_by, parsed.turn_id)
                )
            self._block(
                request_data=request_data,
                input_type=input_type,
                message=message,
                blocked_content=True,
                logging_obj=logging_obj,
            )
        return inputs

    def _v3_replayed_block(self, scope: str, prefixes: tuple[str, ...]) -> _RememberedBlock | None:
        """The block a conversation already earned, when this request repeats or extends a
        conversation this process blocked in the same scope (session or principal)."""
        for prefix in prefixes:
            remembered: object = self._v3_blocked_turns.get_cache(f"{scope}\0{prefix}")
            if isinstance(remembered, _RememberedBlock):
                if self.verbose:
                    verbose_proxy_logger.info(
                        json.dumps({"event": "straiker.replay_blocked", "scope": scope, "prefix": prefix})
                    )
                return remembered
        return None

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        if self.api_version == "v3":
            return await self._apply_v3(
                inputs=inputs, request_data=request_data, input_type=input_type, logging_obj=logging_obj
            )
        try:
            envelope: Final = self._build_envelope(
                inputs=inputs,
                request_data=request_data,
                input_type=input_type,
                logging_obj=logging_obj,
            )
            payload: Final = envelope.model_dump(mode="json", exclude_none=True)
        except (ValidationError, TypeError, ValueError) as error:
            return self._fail(
                inputs=inputs,
                request_data=request_data,
                input_type=input_type,
                error=str(error),
                is_unreachable=False,
            )

        parsed, failure = await self._post_webhook(payload)
        if failure is not None:
            return self._fail(
                inputs=inputs,
                request_data=request_data,
                input_type=input_type,
                error=failure.message,
                is_unreachable=failure.is_unreachable,
            )

        if parsed is None:
            return self._fail(
                inputs=inputs,
                request_data=request_data,
                input_type=input_type,
                error="empty response from Straiker",
                is_unreachable=False,
            )
        self._record(request_data=request_data, logging_obj=logging_obj, parsed=parsed)

        if parsed.schema_version is not None and parsed.schema_version != STRAIKER_WEBHOOK_SCHEMA_VERSION:
            verbose_proxy_logger.warning(
                json.dumps(
                    {
                        "event": "straiker.schema_drift",
                        "expected": STRAIKER_WEBHOOK_SCHEMA_VERSION,
                        "received": parsed.schema_version,
                    }
                )
            )

        if parsed.action == "BLOCKED":
            self._block(
                request_data=request_data,
                input_type=input_type,
                message=parsed.blocked_reason or DEFAULT_BLOCK_MESSAGE,
                blocked_content=True,
                logging_obj=logging_obj,
            )
        if parsed.action == "GUARDRAIL_INTERVENED":
            is_streamed_response: Final = input_type == "response" and _is_streamed_request(request_data)
            if parsed.texts is None or is_streamed_response:
                self._block(
                    request_data=request_data,
                    input_type=input_type,
                    message=parsed.blocked_reason or DEFAULT_BLOCK_MESSAGE,
                    blocked_content=True,
                    logging_obj=logging_obj,
                )
            return self._intervened_inputs(inputs, parsed)
        return inputs
