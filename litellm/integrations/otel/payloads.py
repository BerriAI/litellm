"""Source of truth #3 for the LiteLLM OpenTelemetry instrumentation: typed inputs.

Every span is built from a frozen, typed model defined here. The ``from_*``
classmethods are the single chokepoint where untyped litellm internals (the
``StandardLoggingPayload`` / ``ServiceLoggerPayload`` dicts) are read and
normalized; everything downstream of them is fully typed. This module imports no
``opentelemetry`` symbols.
"""

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    cast,
)
from urllib.parse import urlsplit

from litellm.integrations.otel.semconv import (
    DEFAULT_BAGGAGE_METADATA_KEYS,
    GenAI,
    GenAIOperation,
    LiteLLM,
    resolve_operation,
    resolve_provider,
)

if TYPE_CHECKING:
    from litellm.types.services import ServiceLoggerPayload
    from litellm.types.utils import StandardLoggingPayload


# --- small typed coercion helpers (localize reading of heterogeneous dicts) -- #


def _as_str(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _as_int(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _as_float(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _as_bool(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return bool(value)


def _as_str_tuple(value: object) -> Optional[Tuple[str, ...]]:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return None


# --- typed sub-structures ---------------------------------------------------- #


@dataclass(frozen=True)
class LLMRequestParams:
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop_sequences: Optional[Tuple[str, ...]] = None
    seed: Optional[int] = None

    @classmethod
    def from_model_parameters(cls, params: Mapping[str, object]) -> "LLMRequestParams":
        max_tokens = _as_int(params.get("max_tokens"))
        if max_tokens is None:
            max_tokens = _as_int(params.get("max_completion_tokens"))
        return cls(
            temperature=_as_float(params.get("temperature")),
            top_p=_as_float(params.get("top_p")),
            top_k=_as_int(params.get("top_k")),
            max_tokens=max_tokens,
            frequency_penalty=_as_float(params.get("frequency_penalty")),
            presence_penalty=_as_float(params.get("presence_penalty")),
            stop_sequences=_as_str_tuple(params.get("stop")),
            seed=_as_int(params.get("seed")),
        )


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass(frozen=True)
class SpanError:
    error_type: Optional[str] = None
    message: Optional[str] = None


@dataclass(frozen=True)
class ServerInfo:
    address: Optional[str] = None
    port: Optional[int] = None

    @classmethod
    def from_api_base(cls, api_base: Optional[str]) -> Optional["ServerInfo"]:
        if not api_base:
            return None
        parsed = urlsplit(api_base if "://" in api_base else f"//{api_base}")
        if not parsed.hostname:
            return None
        return cls(address=parsed.hostname, port=parsed.port)


@dataclass(frozen=True)
class RequestIdentity:
    """Request-scoped identity. The promotable subset rides Baggage to all spans."""

    call_id: Optional[str] = None
    team_id: Optional[str] = None
    team_alias: Optional[str] = None
    key_hash: Optional[str] = None
    end_user: Optional[str] = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: "StandardLoggingPayload") -> "RequestIdentity":
        raw_meta = cast(Mapping[str, object], payload.get("metadata") or {})
        metadata: Dict[str, str] = {}
        for key, value in raw_meta.items():
            if isinstance(value, (str, bool, int, float)):
                metadata[key] = str(value)
        return cls(
            call_id=_as_str(payload.get("litellm_call_id"))
            or _as_str(payload.get("id")),
            team_id=_as_str(raw_meta.get("team_id")),
            team_alias=_as_str(raw_meta.get("team_alias")),
            key_hash=_as_str(raw_meta.get("user_api_key_hash")),
            end_user=_as_str(payload.get("end_user")),
            metadata=metadata,
        )


@dataclass(frozen=True)
class GuardrailSpanData:
    guardrail_name: str
    mode: Optional[str] = None
    status: Optional[str] = None
    masked_entity_count: Optional[int] = None


@dataclass(frozen=True)
class ServiceSpanData:
    service_name: str
    call_type: Optional[str] = None
    error: Optional[SpanError] = None

    @classmethod
    def from_payload(cls, payload: "ServiceLoggerPayload") -> "ServiceSpanData":
        service = getattr(payload, "service", None)
        service_name = getattr(service, "value", None) or str(service or "service")
        error_text = getattr(payload, "error", None)
        return cls(
            service_name=service_name,
            call_type=getattr(payload, "call_type", None),
            error=SpanError(message=_as_str(error_text)) if error_text else None,
        )


@dataclass(frozen=True)
class ProxyRequestSpanData:
    http_method: str
    route: str
    url_path: Optional[str] = None
    status_code: Optional[int] = None
    identity: Optional[RequestIdentity] = None


@dataclass(frozen=True)
class ManagementSpanData:
    route: str
    error: Optional[SpanError] = None


# --- the primary LLM-call model ---------------------------------------------- #


@dataclass(frozen=True)
class LLMCallSpanData:
    operation: GenAIOperation
    provider: str
    request_model: str
    response_model: Optional[str]
    response_id: Optional[str]
    request_params: LLMRequestParams
    usage: LLMUsage
    finish_reasons: Tuple[str, ...]
    error: Optional[SpanError]
    response_cost: Optional[float]
    server: Optional[ServerInfo]
    identity: RequestIdentity
    is_streaming: Optional[bool] = None

    @classmethod
    def from_standard_logging_payload(
        cls, payload: "StandardLoggingPayload"
    ) -> "LLMCallSpanData":
        model_parameters = cast(
            Mapping[str, object], payload.get("model_parameters") or {}
        )
        response = payload.get("response")
        response_id: Optional[str] = None
        response_model: Optional[str] = None
        finish_reasons: List[str] = []
        if isinstance(response, dict):
            response_id = _as_str(response.get("id"))
            response_model = _as_str(response.get("model"))
            choices = response.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if isinstance(choice, dict):
                        reason = _as_str(choice.get("finish_reason"))
                        if reason:
                            finish_reasons.append(reason)

        error: Optional[SpanError] = None
        if payload.get("status") == "failure":
            error_info = cast(
                Mapping[str, object], payload.get("error_information") or {}
            )
            error = SpanError(
                error_type=_as_str(error_info.get("error_class"))
                or _as_str(error_info.get("error_code")),
                message=_as_str(error_info.get("error_message"))
                or _as_str(payload.get("error_str")),
            )

        hidden_params = cast(Mapping[str, object], payload.get("hidden_params") or {})
        return cls(
            operation=resolve_operation(_as_str(payload.get("call_type"))),
            provider=resolve_provider(_as_str(payload.get("custom_llm_provider"))),
            request_model=_as_str(payload.get("model")) or "",
            response_model=response_model,
            response_id=response_id,
            request_params=LLMRequestParams.from_model_parameters(model_parameters),
            usage=LLMUsage(
                input_tokens=_as_int(payload.get("prompt_tokens")),
                output_tokens=_as_int(payload.get("completion_tokens")),
                total_tokens=_as_int(payload.get("total_tokens")),
            ),
            finish_reasons=tuple(finish_reasons),
            error=error,
            response_cost=_as_float(payload.get("response_cost")),
            server=ServerInfo.from_api_base(
                _as_str(payload.get("api_base"))
                or _as_str(hidden_params.get("api_base"))
            ),
            identity=RequestIdentity.from_payload(payload),
            is_streaming=_as_bool(payload.get("stream")),
        )


def promoted_baggage(
    identity: RequestIdentity,
    request_model: Optional[str],
    promoted_keys: Tuple[str, ...],
    metadata_keys: Tuple[str, ...] = DEFAULT_BAGGAGE_METADATA_KEYS,
) -> Dict[str, str]:
    """Assemble the bounded set of values to write into Baggage for promotion.

    Only keys in ``promoted_keys`` (and metadata sub-keys in ``metadata_keys``)
    are included — never the full metadata blob, and never ``http.*``.
    """
    candidate: Dict[str, Optional[str]] = {
        LiteLLM.TEAM_ID: identity.team_id,
        LiteLLM.TEAM_ALIAS: identity.team_alias,
        LiteLLM.KEY_HASH: identity.key_hash,
        LiteLLM.END_USER: identity.end_user,
        GenAI.REQUEST_MODEL: request_model,
    }
    out: Dict[str, str] = {
        key: value for key, value in candidate.items() if key in promoted_keys and value
    }
    for meta_key in metadata_keys:
        value = identity.metadata.get(meta_key)
        if value:
            out[f"{LiteLLM.METADATA_PREFIX}{meta_key}"] = value
    return out
