"""Typed, semconv-aligned OpenTelemetry instrumentation for LiteLLM.

The three sources of truth — attribute keys (:mod:`semconv`), the span and
hierarchy registry (:mod:`spans`), and the typed span-data inputs
(:mod:`payloads`) — plus :mod:`config` are exported here and are free of any
``opentelemetry`` import. The engine layer (``emitter``, ``providers``,
``context``, ``metrics``) and the ``CustomLogger`` adapter (``logger``) are
reached via their submodule paths so that importing this package never
requires the OTel SDK.

The ``LITELLM_OTEL_V2`` env var gates whether the factory in
``litellm_core_utils.litellm_logging`` constructs the ``OpenTelemetryV2``
class (from :mod:`logger`).
"""

from litellm.integrations.otel.model.config import (
    OTEL_V2_ENV,
    OpenTelemetryV2Config,
    is_otel_v2_enabled,
)
from litellm.integrations.otel.model.baggage import (
    BAGGAGE_PROMOTED_KEYS,
    DEFAULT_BAGGAGE_METADATA_KEYS,
    promoted_baggage,
)
from litellm.integrations.otel.model.metadata import (
    RequestContext,
    RequestIdentity,
)
from litellm.integrations.otel.model.payloads import (
    GuardrailSpanData,
    LLMCallSpanData,
    LLMRequestParams,
    LLMUsage,
    ProxyRequestSpanData,
    ServerInfo,
    ServiceSpanData,
    SpanError,
)
from litellm.integrations.otel.model.semconv import (
    DB,
    Error,
    GenAI,
    GenAIOperation,
    GenAIProvider,
    HTTP,
    LiteLLM,
    Metric,
    Server,
    resolve_operation,
    resolve_provider,
)
from litellm.integrations.otel.model.spans import (
    SPAN_REGISTRY,
    LiteLLMSpanKind,
    SpanRole,
    SpanSpec,
    db_system,
    span_role_for_service,
    validate_registry,
)

__all__ = [
    # config
    "OTEL_V2_ENV",
    "OpenTelemetryV2Config",
    "is_otel_v2_enabled",
    # semconv
    "BAGGAGE_PROMOTED_KEYS",
    "DB",
    "DEFAULT_BAGGAGE_METADATA_KEYS",
    "Error",
    "GenAI",
    "GenAIOperation",
    "GenAIProvider",
    "HTTP",
    "LiteLLM",
    "Metric",
    "Server",
    "resolve_operation",
    "resolve_provider",
    # spans
    "SPAN_REGISTRY",
    "LiteLLMSpanKind",
    "SpanRole",
    "SpanSpec",
    "db_system",
    "span_role_for_service",
    "validate_registry",
    # payloads
    "GuardrailSpanData",
    "LLMCallSpanData",
    "LLMRequestParams",
    "LLMUsage",
    "ProxyRequestSpanData",
    "RequestContext",
    "RequestIdentity",
    "ServerInfo",
    "ServiceSpanData",
    "SpanError",
    "promoted_baggage",
]
