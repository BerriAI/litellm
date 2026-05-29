"""Typed span-data inputs: frozen dataclasses the engine and mappers consume."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Mapping, cast
from urllib.parse import urlsplit

from litellm.integrations.otel.semconv import (
    GenAIOperation,
    resolve_operation,
    resolve_provider,
)
from litellm.integrations.otel.utils import (
    as_bool,
    as_float,
    as_int,
    as_str,
    as_str_tuple,
)

if TYPE_CHECKING:
    from litellm.types.services import ServiceLoggerPayload
    from litellm.types.utils import (
        StandardLoggingGuardrailInformation,
        StandardLoggingPayload,
    )


# --- typed sub-structures ---------------------------------------------------- #


@dataclass(frozen=True)
class LLMRequestParams:
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop_sequences: tuple[str, ...] | None = None
    seed: int | None = None

    @classmethod
    def from_model_parameters(cls, params: Mapping[str, object]) -> "LLMRequestParams":
        max_tokens = as_int(params.get("max_tokens"))
        if max_tokens is None:
            max_tokens = as_int(params.get("max_completion_tokens"))
        return cls(
            temperature=as_float(params.get("temperature")),
            top_p=as_float(params.get("top_p")),
            top_k=as_int(params.get("top_k")),
            max_tokens=max_tokens,
            frequency_penalty=as_float(params.get("frequency_penalty")),
            presence_penalty=as_float(params.get("presence_penalty")),
            stop_sequences=as_str_tuple(params.get("stop")),
            seed=as_int(params.get("seed")),
        )


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class SpanError:
    error_type: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class ServerInfo:
    address: str | None = None
    port: int | None = None

    @classmethod
    def from_api_base(cls, api_base: str | None) -> ServerInfo | None:
        if not api_base:
            return None
        parsed = urlsplit(api_base if "://" in api_base else f"//{api_base}")
        if not parsed.hostname:
            return None
        return cls(address=parsed.hostname, port=parsed.port)


@dataclass(frozen=True)
class RequestIdentity:
    call_id: str | None = None
    team_id: str | None = None
    team_alias: str | None = None
    key_hash: str | None = None
    end_user: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: "StandardLoggingPayload") -> "RequestIdentity":
        raw_meta = cast(Mapping[str, object], payload.get("metadata") or {})
        metadata = {
            key: str(value)
            for key, value in raw_meta.items()
            if isinstance(value, (str, bool, int, float))
        }
        return cls(
            call_id=as_str(payload.get("litellm_call_id")) or as_str(payload.get("id")),
            # StandardLoggingMetadata's canonical key is ``user_api_key_team_id``;
            # the bare ``team_id`` is a legacy alias and is often empty, so prefer
            # the canonical key and fall back to the alias.
            team_id=as_str(raw_meta.get("user_api_key_team_id"))
            or as_str(raw_meta.get("team_id")),
            team_alias=as_str(raw_meta.get("user_api_key_team_alias"))
            or as_str(raw_meta.get("team_alias")),
            key_hash=as_str(raw_meta.get("user_api_key_hash")),
            end_user=as_str(payload.get("end_user"))
            or as_str(raw_meta.get("user_api_key_end_user_id")),
            metadata=metadata,
        )

    @classmethod
    def from_user_api_key_auth(cls, auth: object) -> "RequestIdentity":
        """Identity from a ``UserAPIKeyAuth`` (duck-typed to keep this module
        free of a proxy import).

        Used in the pre-call hook to seed Baggage early — before any LLM,
        guardrail, or service span is created — so the whole request's spans
        inherit identity, not just the LLM-call span. Metadata sub-keys use the
        ``user_api_key_*`` names that ``baggage.DEFAULT_BAGGAGE_METADATA_KEYS``
        promotes.
        """
        get = lambda name: getattr(auth, name, None)  # noqa: E731
        metadata = {
            meta_key: str(value)
            for meta_key, attr in (
                ("user_api_key_user_id", "user_id"),
                ("user_api_key_org_id", "org_id"),
                ("user_api_key_alias", "key_alias"),
                ("user_api_key_end_user_id", "end_user_id"),
            )
            if (value := get(attr))
        }
        return cls(
            team_id=as_str(get("team_id")),
            team_alias=as_str(get("team_alias")),
            key_hash=as_str(get("api_key")),
            end_user=as_str(get("end_user_id")),
            metadata=metadata,
        )


@dataclass(frozen=True)
class GuardrailSpanData:
    guardrail_name: str
    mode: str | None = None
    status: str | None = None
    masked_entity_count: int | None = None
    provider: str | None = None
    action: str | None = None
    # The guardrail verdict / provider response (e.g. the moderation result),
    # JSON-serialized. This is the detail that belongs on the guardrail span.
    response_json: str | None = None
    violation_categories: tuple[str, ...] = ()
    confidence_score: float | None = None
    risk_score: float | None = None
    duration: float | None = None
    # Provider-agnostic configuration/detection metadata (see
    # ``StandardLoggingGuardrailInformation``). Present for any guardrail that
    # populates them, not just one provider's shape.
    guardrail_id: str | None = None
    policy_template: str | None = None
    detection_method: str | None = None
    # Set when the guardrail intervened/blocked or failed, so the emitter marks
    # the span ERROR — a blocking guardrail is an error outcome for that span.
    error: SpanError | None = None

    # Guardrail statuses that mean the guardrail did not pass the request through.
    _ERROR_STATUSES: ClassVar[frozenset[str]] = frozenset(
        {"guardrail_intervened", "guardrail_failed_to_respond"}
    )

    @classmethod
    def from_logging_entry(
        cls, entry: "StandardLoggingGuardrailInformation"
    ) -> "GuardrailSpanData":
        """Build from one ``standard_logging_guardrail_information`` entry.

        Reads the canonical, provider-agnostic ``StandardLoggingGuardrailInformation``
        keys only — no guessing at a single provider's field names. Values that are
        typed as enums or lists (e.g. ``guardrail_mode``) are normalized to a
        stable string rather than assumed to already be plain strings.
        """
        get = cast(Mapping[str, object], entry).get
        status = as_str(get("guardrail_status"))
        response = get("guardrail_response")
        error = (
            SpanError(error_type=status, message=as_str(get("guardrail_action")))
            if status in cls._ERROR_STATUSES
            else None
        )
        return cls(
            guardrail_name=as_str(get("guardrail_name")) or "guardrail",
            mode=_guardrail_mode_str(get("guardrail_mode")),
            status=status,
            masked_entity_count=_total_masked_entities(get("masked_entity_count")),
            provider=as_str(get("guardrail_provider")),
            action=as_str(get("guardrail_action")),
            response_json=_json_or_none(response) if response is not None else None,
            violation_categories=as_str_tuple(get("violation_categories")) or (),
            confidence_score=as_float(get("confidence_score")),
            risk_score=as_float(get("risk_score")),
            duration=as_float(get("duration")),
            guardrail_id=as_str(get("guardrail_id")),
            policy_template=as_str(get("policy_template")),
            detection_method=as_str(get("detection_method")),
            error=error,
        )


@dataclass(frozen=True)
class ServiceSpanData:
    service_name: str
    call_type: str | None = None
    error: SpanError | None = None
    # Caller-supplied attributes to stamp on the service span, passed through
    # from ``async_service_*_hook(event_metadata=...)``. The mapper owns how
    # these are namespaced: the canonical vocabulary uses ``litellm.metadata.*``
    # keys, the semconv-ai / Traceloop vocabulary uses the bare key names.
    event_metadata: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        payload: "ServiceLoggerPayload",
        event_metadata: Mapping[str, object] | None = None,
    ) -> "ServiceSpanData":
        # ``payload.service`` is a ``ServiceTypes(str, Enum)`` and ``error`` is
        # ``Optional[str]`` on the Pydantic model — no defensive reads needed.
        # ``str(value)`` covers every case (``str(None) == "None"``).
        coerced = {key: str(value) for key, value in (event_metadata or {}).items()}
        return cls(
            service_name=payload.service.value,
            call_type=payload.call_type,
            error=SpanError(message=payload.error) if payload.error else None,
            event_metadata=coerced,
        )


@dataclass(frozen=True)
class ProxyRequestSpanData:
    http_method: str
    route: str
    url_path: str | None = None
    status_code: int | None = None
    identity: RequestIdentity | None = None


# --- the primary LLM-call model ---------------------------------------------- #


@dataclass(frozen=True)
class ToolDefinition:
    """A single function/tool declared on a chat-completion request."""

    name: str
    description: str | None = None
    parameters_json: str | None = (
        None  # JSON-serialized schema (str so it's an AttrValue)
    )


@dataclass(frozen=True)
class LLMCallSpanData:
    operation: GenAIOperation
    provider: str
    request_model: str
    response_model: str | None
    response_id: str | None
    request_params: LLMRequestParams
    usage: LLMUsage
    finish_reasons: tuple[str, ...]
    error: SpanError | None
    response_cost: float | None
    server: ServerInfo | None
    identity: RequestIdentity
    is_streaming: bool | None = None
    tools: tuple[ToolDefinition, ...] = ()
    # Raw messages and response, needed by vendor mappers (OpenInference,
    # Langfuse, Weave) that stamp message-level attributes. ``messages_in`` is
    # the request payload; ``choices_out`` mirrors ``response.choices`` from
    # the StandardLoggingPayload. Both are tuples of immutable mappings so the
    # dataclass stays hashable and frozen.
    messages_in: tuple[Mapping[str, object], ...] = ()
    choices_out: tuple[Mapping[str, object], ...] = ()
    system_fingerprint: str | None = None

    @classmethod
    def from_standard_logging_payload(
        cls, payload: "StandardLoggingPayload", capture_content: bool = False
    ) -> "LLMCallSpanData":
        params = cast(Mapping[str, object], payload.get("model_parameters") or {})
        hidden = cast(Mapping[str, object], payload.get("hidden_params") or {})
        # Normalize ``response`` to a dict once so every field read below is a
        # plain ``.get`` — no repeated ``isinstance`` guards.
        raw_response = payload.get("response")
        response = cast(
            Mapping[str, object], raw_response if isinstance(raw_response, dict) else {}
        )
        choices_out = _dicts(response.get("choices"))
        # ``finish_reasons`` is metadata, not content, so derive it from
        # ``choices_out`` before gating. The raw message/choice bodies are only
        # retained when content capture is enabled (see ``capture_span_content``);
        # otherwise the content-bearing mappers receive empty sequences and emit
        # no prompt/response text.
        finish_reasons = _finish_reasons(choices_out)
        return cls(
            operation=resolve_operation(as_str(payload.get("call_type"))),
            provider=resolve_provider(as_str(payload.get("custom_llm_provider"))),
            request_model=as_str(payload.get("model")) or "",
            response_model=as_str(response.get("model")),
            response_id=as_str(response.get("id")),
            request_params=LLMRequestParams.from_model_parameters(params),
            usage=LLMUsage(
                input_tokens=as_int(payload.get("prompt_tokens")),
                output_tokens=as_int(payload.get("completion_tokens")),
                total_tokens=as_int(payload.get("total_tokens")),
            ),
            finish_reasons=finish_reasons,
            error=_parse_error(payload),
            response_cost=as_float(payload.get("response_cost")),
            server=ServerInfo.from_api_base(
                as_str(payload.get("api_base")) or as_str(hidden.get("api_base"))
            ),
            identity=RequestIdentity.from_payload(payload),
            is_streaming=as_bool(payload.get("stream")),
            tools=_extract_tools(params),
            messages_in=_dicts(payload.get("messages")) if capture_content else (),
            choices_out=choices_out if capture_content else (),
            system_fingerprint=as_str(response.get("system_fingerprint")),
        )


def _json_or_none(value: object) -> str | None:
    """JSON-serialize ``value`` (already-string values pass through). ``None`` on failure."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except Exception:
        return None


def _guardrail_mode_str(value: object) -> str | None:
    """Normalize ``guardrail_mode`` to a stable string.

    ``guardrail_mode`` is typed as a ``GuardrailEventHooks`` enum, a list of them,
    or a ``GuardrailMode`` — not a plain string. Emit the enum *value* (e.g.
    ``"pre_call"``) rather than ``str(enum)`` (``"GuardrailEventHooks.pre_call"``),
    and join a list of modes so a guardrail that runs at multiple hooks is
    represented faithfully.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parts = [part for item in value if (part := _guardrail_mode_str(item))]
        return ",".join(parts) or None
    if isinstance(value, Enum):
        return as_str(value.value)
    return as_str(value)


def _total_masked_entities(value: object) -> int | None:
    """``masked_entity_count`` is a ``{entity_type: count}`` map — sum to a total."""
    if isinstance(value, Mapping):
        total = sum(v for v in value.values() if isinstance(v, int))
        return total or None
    return as_int(value)


def _dicts(value: object) -> tuple[Mapping[str, object], ...]:
    """The dict items of ``value`` (when it's a list), as a tuple. Else empty."""
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _finish_reasons(choices: tuple[Mapping[str, object], ...]) -> tuple[str, ...]:
    """Non-empty ``finish_reason`` of each response choice."""
    return tuple(r for c in choices if (r := as_str(c.get("finish_reason"))))


def _parse_error(payload: "StandardLoggingPayload") -> SpanError | None:
    """A ``SpanError`` for a failed request, or ``None`` on success."""
    if payload.get("status") != "failure":
        return None
    info = cast(Mapping[str, object], payload.get("error_information") or {})
    return SpanError(
        error_type=as_str(info.get("error_class")) or as_str(info.get("error_code")),
        message=as_str(info.get("error_message")) or as_str(payload.get("error_str")),
    )


def _tool_from_entry(entry: object) -> ToolDefinition | None:
    """One ``tools``/``functions`` entry → ``ToolDefinition``, or ``None`` if unusable."""
    if not isinstance(entry, dict):
        return None
    fn = entry.get("function") if "function" in entry else entry
    if not isinstance(fn, dict):
        return None
    name = as_str(fn.get("name"))
    if not name:
        return None
    params = fn.get("parameters")
    parameters_json: str | None = None
    if params is not None:
        try:
            parameters_json = json.dumps(params, default=str)
        except Exception:
            parameters_json = None
    return ToolDefinition(
        name=name,
        description=as_str(fn.get("description")),
        parameters_json=parameters_json,
    )


def _extract_tools(
    model_parameters: Mapping[str, object],
) -> tuple[ToolDefinition, ...]:
    """Pull declared tools from request params (OpenAI / Anthropic shape).

    Accepts the chat-completion ``tools=[{"type":"function", "function":
    {...}}, ...]`` shape, and falls back to the ``functions=[...]`` shape.
    Returns an empty tuple when neither is present.
    """
    raw_tools = model_parameters.get("tools")
    if not isinstance(raw_tools, list):
        raw_tools = model_parameters.get("functions")  # ``functions`` shape
    if not isinstance(raw_tools, list):
        return ()
    return tuple(t for entry in raw_tools if (t := _tool_from_entry(entry)) is not None)
