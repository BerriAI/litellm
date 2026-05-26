"""Typed, semconv-aligned OpenTelemetry instrumentation for LiteLLM.

Phase 0-1 scaffolding (see ``RFC.md``). The three sources of truth — attribute
keys (:mod:`semconv`), the span+hierarchy registry (:mod:`spans`), and the typed
span-data inputs (:mod:`payloads`) — plus :mod:`config` are exported here and are
free of any ``opentelemetry`` import. The engine layer (``emitter``, ``providers``,
``context``, ``metrics``) is imported via its submodule paths so that importing
this package never requires the OTel SDK.

This package is inert until ``LITELLM_OTEL_V2`` is enabled; nothing here is wired
into the request path yet.
"""

from litellm.integrations.otel.config import (
    OTEL_V2_ENV,
    OpenTelemetryV2Config,
    is_otel_v2_enabled,
)
from litellm.integrations.otel.payloads import (
    GuardrailSpanData,
    LLMCallSpanData,
    LLMRequestParams,
    LLMUsage,
    ManagementSpanData,
    ProxyRequestSpanData,
    RequestIdentity,
    ServerInfo,
    ServiceSpanData,
    SpanError,
    promoted_baggage,
)
from litellm.integrations.otel.semconv import (
    BAGGAGE_PROMOTED_KEYS,
    DEFAULT_BAGGAGE_METADATA_KEYS,
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
from litellm.integrations.otel.spans import (
    SPAN_REGISTRY,
    LiteLLMSpanKind,
    SpanRole,
    SpanSpec,
    validate_registry,
)

__all__ = [
    # config
    "OTEL_V2_ENV",
    "OpenTelemetryV2Config",
    "is_otel_v2_enabled",
    # semconv
    "BAGGAGE_PROMOTED_KEYS",
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
    "validate_registry",
    # payloads
    "GuardrailSpanData",
    "LLMCallSpanData",
    "LLMRequestParams",
    "LLMUsage",
    "ManagementSpanData",
    "ProxyRequestSpanData",
    "RequestIdentity",
    "ServerInfo",
    "ServiceSpanData",
    "SpanError",
    "promoted_baggage",
]
