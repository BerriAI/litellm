from __future__ import annotations

import asyncio
import hashlib
import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, NoReturn
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm._version import version as litellm_version
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
from litellm.types.utils import CallTypes, GenericGuardrailAPIInputs, ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

GUARDRAIL_NAME: Final = "straiker"
DEFAULT_BLOCK_MESSAGE: Final = "Content violates policy"
DEFAULT_API_BASE: Final = "https://api.prod.straiker.ai"
DEFAULT_MAX_PAYLOAD_BYTES: Final = 524288
WEBHOOK_PATH: Final = "/api/v1/detect/webhook"
# The v3 platform exposes one detect route and it parses the gateway's own traffic:
# the request phase relays the provider body LiteLLM received, the response phase wraps
# the model's answer beside that request, and Straiker derives prompt, answer, agent and
# archetype server-side (the same contract the unified Kong plugin speaks). There is no
# v3 webhook envelope (/api/v3/detect/webhook is a 404).
V3_DETECT_PATH: Final = "/api/v3/detect"
# Integration keys minted by the v3 platform. A v1 collection key is a UUID; the prefix
# never collides, so it can select the API when the config does not say.
V3_KEY_PREFIX: Final = "sk_agt_"
# No ingress header on v3: the platform parses the relayed body itself and reads the agent
# out of it (cc_entrypoint), the same as the unified Kong plugin. `x-tool` is the v1
# native-hook selector and is not sent.
V3_SESSION_HEADER: Final = "x-claude-code-session-id"
# Routing hints, each optional. The same headers the unified Kong plugin sends, so a
# tenant's agents enumerate identically whichever gateway the traffic came through.
V3_CLIENT_HEADER: Final = "x-s6r-client"
V3_FORMAT_HEADER: Final = "x-s6r-format"
# Prefix for a session id derived from the conversation itself, when the client states none.
V3_DERIVED_SESSION_PREFIX: Final = "litellm-"
# Which agent this turn belongs to, when one gateway fronts several applications. A name for
# ONE agent, never a kind of agent: Straiker keys per-agent state on it, so a value shared by
# several applications merges them into one. Forwarded from the client when it sends one, else
# the `agent_ref` config value. The same header the Kong plugin sends, so a tenant's agents
# enumerate identically whichever gateway the traffic came through.
V3_AGENT_HEADER: Final = "x-s6r-agent"
V3_RESPONSE_PHASE: Final = "response-sync"
# A v3 verdict blocks on `permissionDecision` (gateway envelope) or `action` (flat body).
V3_BLOCK_DECISIONS: Final = frozenset({"block", "deny"})
# Provider body fields, for every surface the proxy fronts. An allowlist rather than a
# denylist: the hook sees the client body merged with proxy state (`deployment` carries
# the resolved credential, `proxy_server_request` the client's Authorization header), and
# a field this list does not know is not relayed. Detection reads messages, system, tools,
# input and instructions; the rest travels so Straiker records the turn as the client sent it.
_V3_PROVIDER_BODY_KEYS: Final = frozenset(
    {
        # OpenAI chat completions
        "model", "messages", "tools", "tool_choice", "functions", "function_call", "temperature",
        "top_p", "n", "stream", "stream_options", "stop", "max_tokens", "max_completion_tokens",
        "presence_penalty", "frequency_penalty", "logit_bias", "user", "response_format", "seed",
        "logprobs", "top_logprobs", "parallel_tool_calls", "reasoning_effort", "modalities", "audio",
        "prediction", "store", "service_tier", "web_search_options",
        # Anthropic messages
        "system", "stop_sequences", "top_k", "thinking", "container", "mcp_servers",
        "context_management", "output_format",
        # OpenAI responses
        "input", "instructions", "previous_response_id", "truncation", "text", "include",
        "reasoning", "max_output_tokens", "background", "conversation",
        # Conversation grouping a client may state itself
        "session_id",
    }
)
# The identity fields Straiker's LiteLLM adapter reads from `metadata`, most specific first.
_V3_IDENTITY_METADATA_KEYS: Final = (
    "user_api_key_end_user_id",
    "user_api_key_user_email",
    "user_api_key_user_id",
    "user_api_key_alias",
    "user_api_key_team_id",
)
RETRY_STATUS: Final = frozenset({408, 429, 500, 502, 503, 504})
UNREACHABLE_STATUS: Final = frozenset({502, 503, 504})
_APPLICATION_METADATA_KEYS: Final = frozenset({"agent_id", "app_name"})
_OPAQUE_METADATA_SCALAR_TYPES: Final = (str, int, float, bool)
_JSON_DICT_ADAPTER: Final = TypeAdapter(dict[str, object])


@dataclass(frozen=True, slots=True)
class _WebhookFailure:
    message: str
    is_unreachable: bool


def _as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _merged_metadata(request_data: dict) -> dict:
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


def _request_structured_messages(request_data: dict) -> list[dict[str, Any]] | None:
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


def _response_finish_reason(response: Any) -> str | None:
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
    identity = _as_optional_str(value)
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


def _v3_identity_metadata(request_data: Mapping[str, object]) -> dict[str, str]:
    """The proxy-resolved identity fields, and only those, for the relayed body."""
    merged: Final = _merged_metadata(dict(request_data))
    identity: dict[str, str] = {}
    for key in _V3_IDENTITY_METADATA_KEYS:
        value = _real_identity(merged.get(key))
        if value:
            identity[key] = value
    return identity


def _v3_request_body(request_data: Mapping[str, object]) -> dict[str, object]:
    """The provider body LiteLLM received, stripped of everything the proxy added.

    The hook sees the client's request merged with proxy bookkeeping: logging objects,
    the resolved key, the inbound headers. Only the provider body is Straiker's to read,
    and the client's Authorization header must not travel. Identity survives as the
    metadata subset the Straiker LiteLLM adapter reads.
    """
    body: dict[str, object] = {key: value for key, value in request_data.items() if key in _V3_PROVIDER_BODY_KEYS}
    identity: Final = _v3_identity_metadata(request_data)
    if identity:
        body["metadata"] = identity
    return body


def _v3_anthropic_messages_route(request_data: Mapping[str, object]) -> bool:
    from litellm.litellm_core_utils.api_route_to_call_types import get_call_types_for_route

    route: Final = _merged_metadata(dict(request_data)).get("user_api_key_request_route")
    if not isinstance(route, str) or not route:
        return False
    return CallTypes.anthropic_messages in (get_call_types_for_route(route) or ())


def _v3_answer(request_data: Mapping[str, object], model: str | None) -> dict[str, object] | None:
    """The answer in the API shape the client spoke, which is what a relay forwards.

    On a streamed Messages call the proxy rebuilds the answer as a chat completion before
    the hook runs. Straiker's coding-agent reader parses a Messages answer, so a Claude Code
    turn sent as a chat completion scores nothing; the proxy's own adapter turns it back.
    """
    response: Final = request_data.get("response")
    if not isinstance(response, ModelResponse) or not _v3_anthropic_messages_route(request_data):
        return _jsonable_dict(response)
    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )

    translated: Final = LiteLLMAnthropicMessagesAdapter().translate_openai_response_to_anthropic(response=response)
    return _jsonable_dict({**translated, "model": response.model or model})


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
        return json.dumps(response, default=str)
    texts: Final = [t for t in (inputs.get("texts") or []) if isinstance(t, str) and t]
    if not texts:
        return None
    assembled: Final = {
        "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "\n".join(texts)}}],
    }
    return json.dumps(assembled)


def _v3_payload(
    envelope: StraikerWebhookRequest,
    inputs: GenericGuardrailAPIInputs,
    request_data: Mapping[str, object],
    input_type: Literal["request", "response"],
) -> dict[str, object]:
    """The /api/v3/detect body for one phase of a turn, the unified Kong plugin's contract.

    Request phase: the provider body itself. Response phase: the answer beside the request
    it answers, `{straiker_phase, sse, model, request}`, which is how Straiker classifies a
    tool call the model just made. Straiker parses either and derives prompt, answer, agent
    and archetype from the traffic; nothing is pre-digested here. Identity and session ride
    on both phases the way Kong sends them.
    """
    context: Final = envelope.context
    request_body: Final = _v3_request_body(request_data)

    if input_type == "request":
        payload: dict[str, object] = dict(request_body)
    else:
        payload = {
            "straiker_phase": V3_RESPONSE_PHASE,
            "model": context.model,
            "request": request_body,
        }
        answer_json: Final = _v3_answer_json(inputs, request_data, context.model)
        if answer_json is not None:
            payload["sse"] = answer_json

    session: Final = _v3_session_id(envelope, request_data, request_body)
    if session:
        payload["session_id"] = session
    user: Final = _v3_user(envelope)
    if user:
        payload["original"] = {"processed": {"Meta": {"user": user}}}
    return payload


def _v3_session_id(
    envelope: StraikerWebhookRequest,
    request_data: Mapping[str, object],
    request_body: Mapping[str, object],
) -> str | None:
    """A stable id for the conversation, in Kong's order of precedence.

    Claude Code names its session on the wire and that wins. Then the session LiteLLM
    resolved from its own metadata. Then, for a conversation that states none, a hash of
    the system prompt and the first message: a chat client replays the whole conversation
    on every turn, so that pair is constant for its lifetime and groups the turns. A fresh
    synthetic id per request would group nothing.
    """
    supplied: Final = _request_header(request_data, V3_SESSION_HEADER)
    if supplied:
        return supplied
    if envelope.context.session_id:
        return envelope.context.session_id
    system = request_body.get("system")
    if not isinstance(system, str):
        system = json.dumps(system, default=str) if system is not None else None
    if system is None and isinstance(request_body.get("instructions"), str):
        system = request_body["instructions"]
    first = ""
    messages: Final = request_body.get("messages") or request_body.get("input")
    if isinstance(messages, list) and messages and isinstance(messages[0], Mapping):
        content = messages[0].get("content")
        if isinstance(content, str):
            first = content
        elif isinstance(content, list) and content and isinstance(content[0], Mapping):
            first = str(content[0].get("text") or "")
    elif isinstance(request_body.get("prompt"), str):
        first = str(request_body["prompt"])
    seed: Final = f"{system or ''}\0{first}"
    if seed == "\0":
        return None
    return V3_DERIVED_SESSION_PREFIX + hashlib.md5(seed.encode("utf-8"), usedforsecurity=False).hexdigest()


def _v3_user(envelope: StraikerWebhookRequest) -> str | None:
    """Who is asking, most specific first, never a proxy placeholder.

    A master-key call resolves to LiteLLM's `default_user_id`; sent as an identity it
    would become one, and a session with a real end user on other turns would be filed
    under the placeholder.
    """
    identity: Final = envelope.identity
    for candidate in (identity.litellm_user_email, identity.end_user_id, identity.litellm_user_id):
        real = _real_identity(candidate)
        if real:
            return real
    return None


def _v3_headers(
    request_data: Mapping[str, object],
    agent_ref: str | None = None,
    client: str | None = None,
    format_hint: str | None = None,
) -> dict[str, str]:
    """Per-call routing hints, the unified Kong plugin's set. All optional.

    `x-s6r-agent` names ONE application when a gateway fronts several; a client-supplied
    value wins over the route's `agent_ref`. `x-s6r-client` and `x-s6r-format` come from
    config alone. Claude Code's own session header is forwarded when the client sent it,
    which is how a coding session groups the way the native hook would.
    """
    headers: dict[str, str] = {}
    session: Final = _request_header(request_data, V3_SESSION_HEADER)
    if session:
        headers[V3_SESSION_HEADER] = session
    agent: Final = _request_header(request_data, V3_AGENT_HEADER) or agent_ref
    if agent:
        headers[V3_AGENT_HEADER] = agent
    if client:
        headers[V3_CLIENT_HEADER] = client
    if format_hint:
        headers[V3_FORMAT_HEADER] = format_hint
    return headers


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
    if isinstance(hook, Mapping):
        decision = hook.get("permissionDecision")
        if isinstance(decision, str) and decision:
            return decision.lower(), verdict
    action: Final = verdict.get("action")
    return (action.lower() if isinstance(action, str) and action else None), verdict


def _v3_response(body: Mapping[str, object]) -> StraikerWebhookResponse:
    """Map a v3 verdict onto the action the guardrail already acts on.

    A detect-mode control fires into `controls` without changing the decision, so it
    correctly reads NONE. `blocked_by` is the block-mode subset and is honoured even if a
    build answers it without flipping the decision.
    """
    decision, verdict = _v3_decision(body)
    raw_blocked_by: Final = verdict.get("blocked_by")
    blocked_by: Final = tuple(sorted(str(c) for c in raw_blocked_by)) if isinstance(raw_blocked_by, list) else ()
    blocked: Final = decision in V3_BLOCK_DECISIONS or bool(blocked_by)
    reason: str | None = None
    if blocked:
        for key in ("block_message", "deny_reason", "stopReason"):
            value = verdict.get(key) if key != "stopReason" else body.get(key)
            if isinstance(value, str) and value.strip():
                reason = value.strip()
                break
        if reason is None:
            reason = f"Straiker blocked this turn: {', '.join(blocked_by) or 'policy'}"
    return StraikerWebhookResponse.model_validate(
        {
            "action": "BLOCKED" if blocked else "NONE",
            "blocked_reason": reason,
            "turnId": verdict.get("turn_id") or body.get("turn_id"),
        }
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
        if api_version is None:
            # The key names the platform: a v3 integration key cannot call v1 and a v1
            # collection key cannot call v3, so an unset version follows the key.
            api_version = "v3" if api_key.startswith(V3_KEY_PREFIX) else "v1"
        if api_version not in ("v1", "v3"):
            raise ValueError(f"api_version must be 'v1' or 'v3'; got {api_version!r}")

        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.api_version = api_version
        self.agent_ref = _as_optional_str(agent_ref)
        self.client = _as_optional_str(client)
        if format_hint is not None and format_hint not in ("anthropic.messages", "openai.chat"):
            raise ValueError(f"format_hint must be 'anthropic.messages' or 'openai.chat'; got {format_hint!r}")
        self.format_hint = format_hint
        self.source = source
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        self.initial_backoff = max(0.0, float(initial_backoff))
        self.max_backoff = max(self.initial_backoff, float(max_backoff))
        self.unreachable_fallback = unreachable_fallback
        self.fail_on_error = fail_on_error
        self.max_payload_bytes = int(max_payload_bytes)
        self.custom_headers = dict(custom_headers) if custom_headers else {}
        self.default_metadata = dict(metadata) if metadata else {}
        self.verbose = bool(verbose)

        self.streaming_end_of_stream_only = True
        self.streaming_buffer_until_moderated = True

        self.async_handler = async_handler or get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )

        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))
        super().__init__(**kwargs)

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
        # v3 relays the provider body and has no envelope dialect to declare.
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

        is_request: Final = input_type == "request"
        content: Final = StraikerWebhookContent(
            texts=list(inputs.get("texts") or []),
            images=list(inputs.get("images") or []),
            structured_messages=_opaque_dict_list(inputs.get("structured_messages")) if is_request else None,
            tools=_opaque_dict_list(inputs.get("tools")) if is_request else None,
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
        self, payload: dict, headers: Mapping[str, str] | None = None
    ) -> tuple[StraikerWebhookResponse | None, _WebhookFailure | None]:
        try:
            body = json.dumps(payload).encode("utf-8")
        except (TypeError, ValueError, OverflowError) as error:
            return None, _WebhookFailure(f"request serialization failed: {error}", is_unreachable=False)
        body_bytes: Final = len(body)
        if body_bytes > self.max_payload_bytes:
            return None, _WebhookFailure(
                f"payload {body_bytes}B exceeds max_payload_bytes {self.max_payload_bytes}",
                is_unreachable=False,
            )

        url: Final = self._webhook_url()
        headers: Final = {**self._headers(), **(headers or {})}
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
                    default=str,
                )
            )

        for attempt in range(attempts):
            try:
                try:
                    resp = await self.async_handler.post(url, content=body, headers=headers, timeout=self.timeout)
                except httpx.HTTPStatusError as status_error:
                    # LiteLLM's client raises on any non-2xx, so an error status never reaches
                    # the branch below on its own. Treat it as the same failure a plain client
                    # would have returned: retryable when the status says so, otherwise final.
                    status: int = status_error.response.status_code
                    text: str = ""
                    try:
                        text = status_error.response.text
                    except Exception:  # noqa: BLE001 - masked responses may lack a body
                        text = ""
                    last_failure = _WebhookFailure(
                        f"HTTP {status}: {text[:200]}", is_unreachable=status in UNREACHABLE_STATUS
                    )
                    if status not in RETRY_STATUS:
                        return None, last_failure
                    if attempt < attempts - 1:
                        backoff = min(self.initial_backoff * (2**attempt), self.max_backoff)
                        await asyncio.sleep(random.uniform(0, backoff))
                    continue
                if resp.status_code == 200:
                    try:
                        body = resp.json()
                        parsed = _v3_response(body) if self.api_version == "v3" else StraikerWebhookResponse.model_validate(body)
                    except (ValidationError, json.JSONDecodeError) as ve:
                        return None, _WebhookFailure(f"invalid response schema: {ve}", is_unreachable=False)
                    if self.verbose:
                        verbose_proxy_logger.info(
                            json.dumps(
                                {
                                    "event": "straiker.webhook_response",
                                    "status_code": resp.status_code,
                                    "body": body,
                                },
                                default=str,
                            )
                        )
                    return parsed, None
                last_failure = _WebhookFailure(
                    f"HTTP {resp.status_code}: {resp.text[:200]}",
                    is_unreachable=resp.status_code in UNREACHABLE_STATUS,
                )
                if resp.status_code not in RETRY_STATUS:
                    return None, last_failure
            except (httpx.RequestError, asyncio.TimeoutError, Timeout) as e:
                last_failure = _WebhookFailure(f"{type(e).__name__}: {e}", is_unreachable=True)
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                return None, _WebhookFailure(f"{type(e).__name__}: {e}", is_unreachable=False)

            if attempt < attempts - 1:
                backoff = min(self.initial_backoff * (2**attempt), self.max_backoff)
                await asyncio.sleep(random.uniform(0, backoff))

        return None, last_failure or _WebhookFailure("unknown error", is_unreachable=True)

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
                default=str,
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
    ) -> NoReturn:
        if input_type == "request":
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name or GUARDRAIL_NAME,
                message=message,
                should_wrap_with_default_message=False,
                blocked_content=blocked_content,
            )
        raise ModifyResponseException(
            message=message,
            model=request_data.get("model", "unknown") or "unknown",
            request_data=request_data,
            guardrail_name=self.guardrail_name or GUARDRAIL_NAME,
            original_response=request_data.get("response"),
        )

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
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None,
    ) -> GenericGuardrailAPIInputs:
        """One phase of a turn against /api/v3/detect: relay, read the decision, enforce."""
        try:
            envelope: Final = self._build_envelope(
                inputs=inputs,
                request_data=request_data,
                input_type=input_type,
                logging_obj=logging_obj,
            )
            payload: Final = _v3_payload(envelope, inputs, request_data, input_type)
            headers: Final = _v3_headers(request_data, self.agent_ref, self.client, self.format_hint)
        except (ValidationError, TypeError, ValueError) as error:
            return self._fail(
                inputs=inputs,
                request_data=request_data,
                input_type=input_type,
                error=str(error),
                is_unreachable=False,
            )

        parsed, failure = await self._post_webhook(payload, headers)
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
        if parsed.action == "BLOCKED":
            self._block(
                request_data=request_data,
                input_type=input_type,
                message=parsed.blocked_reason or DEFAULT_BLOCK_MESSAGE,
                blocked_content=True,
            )
        return inputs

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
            )
        if parsed.action == "GUARDRAIL_INTERVENED":
            is_streamed_response: Final = input_type == "response" and _is_streamed_request(request_data)
            if parsed.texts is None or is_streamed_response:
                self._block(
                    request_data=request_data,
                    input_type=input_type,
                    message=parsed.blocked_reason or DEFAULT_BLOCK_MESSAGE,
                    blocked_content=True,
                )
            return self._intervened_inputs(inputs, parsed)
        return inputs
