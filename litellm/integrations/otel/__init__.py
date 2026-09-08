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

from litellm.integrations.otel.model.baggage import (
    BAGGAGE_PROMOTED_KEYS,
    DEFAULT_BAGGAGE_METADATA_KEYS,
    promoted_baggage,
)
from litellm.integrations.otel.model.config import (
    OTEL_V2_ENV,
    OpenTelemetryV2Config,
    is_otel_v2_enabled,
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
    MCPListToolsSpanData,
    MCPToolCallSpanData,
    ProxyRequestSpanData,
    ServerInfo,
    ServiceSpanData,
    SpanError,
    is_mcp_list_tools,
    is_mcp_tool_call,
)
from litellm.integrations.otel.model.semconv import (
    DB,
    HTTP,
    MCP,
    Client,
    Error,
    GenAI,
    GenAIOperation,
    GenAIOutputType,
    GenAIProvider,
    JsonRpc,
    LiteLLM,
    LiteLLMError,
    MCPMethod,
    Metric,
    Network,
    NetworkTransport,
    RpcSystem,
    Server,
    resolve_operation,
    resolve_output_type,
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
    "BAGGAGE_PROMOTED_KEYS",
    "DB",
    "DEFAULT_BAGGAGE_METADATA_KEYS",
    "HTTP",
    "MCP",
    "OTEL_V2_ENV",
    "SPAN_REGISTRY",
    "Client",
    "Error",
    "GenAI",
    "GenAIOperation",
    "GenAIOutputType",
    "GenAIProvider",
    "GuardrailSpanData",
    "JsonRpc",
    "LLMCallSpanData",
    "LLMRequestParams",
    "LLMUsage",
    "LiteLLM",
    "LiteLLMError",
    "LiteLLMSpanKind",
    "MCPListToolsSpanData",
    "MCPMethod",
    "MCPToolCallSpanData",
    "Metric",
    "Network",
    "NetworkTransport",
    "OpenTelemetryV2Config",
    "ProxyRequestSpanData",
    "RequestContext",
    "RequestIdentity",
    "RpcSystem",
    "Server",
    "ServerInfo",
    "ServiceSpanData",
    "SpanError",
    "SpanRole",
    "SpanSpec",
    "db_system",
    "is_mcp_list_tools",
    "is_mcp_tool_call",
    "is_otel_v2_enabled",
    "promoted_baggage",
    "resolve_operation",
    "resolve_output_type",
    "resolve_provider",
    "span_role_for_service",
    "validate_registry",
]
