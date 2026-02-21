import os
from typing import TYPE_CHECKING, Any, Optional, Union

from litellm._logging import verbose_logger
from litellm.integrations.arize import _utils
from litellm.integrations.arize._utils import ArizeOTELAttributes
from litellm.types.integrations.arize_phoenix import ArizePhoenixConfig

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import Span as _Span
    from opentelemetry.trace import SpanKind

    from litellm.integrations.opentelemetry import OpenTelemetry as _OpenTelemetry
    from litellm.integrations.opentelemetry import OpenTelemetryConfig as _OpenTelemetryConfig
    from litellm.types.integrations.arize import Protocol as _Protocol

    Protocol = _Protocol
    OpenTelemetryConfig = _OpenTelemetryConfig
    Span = Union[_Span, Any]
    OpenTelemetry = _OpenTelemetry
else:
    Protocol = Any
    OpenTelemetryConfig = Any
    Span = Any
    TracerProvider = Any
    SpanKind = Any
    # Import OpenTelemetry at runtime
    try:
        from litellm.integrations.opentelemetry import OpenTelemetry
    except ImportError:
        OpenTelemetry = None  # type: ignore


ARIZE_HOSTED_PHOENIX_ENDPOINT = "https://otlp.arize.com/v1/traces"


class ArizePhoenixLogger(OpenTelemetry):  # type: ignore
    """
    Arize Phoenix logger that sends traces to a Phoenix endpoint.

    Creates its own dedicated TracerProvider so it can coexist with the
    generic ``otel`` callback (or any other OTEL-based integration) without
    fighting over the global ``opentelemetry.trace`` TracerProvider singleton.
    """

    def _init_tracing(self, tracer_provider):
        """
        Override to always create a *private* TracerProvider for Arize Phoenix.

        The base ``OpenTelemetry._init_tracing`` falls back to the global
        TracerProvider when one already exists.  That causes whichever
        integration initialises second to silently reuse the first one's
        exporter, so spans only reach one destination.

        By creating our own provider we guarantee Arize Phoenix always gets
        its own exporter pipeline, regardless of initialisation order.
        """
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.trace import SpanKind

        if tracer_provider is not None:
            # Explicitly supplied (e.g. in tests) — honour it.
            self.tracer = tracer_provider.get_tracer("litellm")
            self.span_kind = SpanKind
            return

        # Always create a dedicated provider — never touch the global one.
        provider = TracerProvider(resource=self._get_litellm_resource(self.config))
        provider.add_span_processor(self._get_span_processor())
        self.tracer = provider.get_tracer("litellm")
        self.span_kind = SpanKind
        verbose_logger.debug(
            "ArizePhoenixLogger: Created dedicated TracerProvider "
            "(endpoint=%s, exporter=%s)",
            self.config.endpoint,
            self.config.exporter,
        )

    def _init_otel_logger_on_litellm_proxy(self):
        """
        Override: Arize Phoenix should NOT overwrite the proxy's
        ``open_telemetry_logger``.  That attribute is reserved for the
        primary ``otel`` callback which handles proxy-level parent spans.
        """
        pass

    def set_attributes(self, span: Span, kwargs, response_obj: Optional[Any]):
        ArizePhoenixLogger.set_arize_phoenix_attributes(span, kwargs, response_obj)
        return

    @staticmethod
    def set_arize_phoenix_attributes(span: Span, kwargs, response_obj):
        from litellm.integrations.opentelemetry_utils.base_otel_llm_obs_attributes import safe_set_attribute

        _utils.set_attributes(span, kwargs, response_obj, ArizeOTELAttributes)

        # Dynamic project name: check metadata first, then fall back to env var config
        dynamic_project_name = ArizePhoenixLogger._get_dynamic_project_name(kwargs)
        if dynamic_project_name:
            safe_set_attribute(span, "openinference.project.name", dynamic_project_name)
        else:
            # Fall back to static config from env var
            config = ArizePhoenixLogger.get_arize_phoenix_config()
            if config.project_name:
                safe_set_attribute(span, "openinference.project.name", config.project_name)

        return

    @staticmethod
    def _get_dynamic_project_name(kwargs) -> Optional[str]:
        """
        Retrieve dynamic Phoenix project name from request metadata.

        Users can set `metadata.phoenix_project_name` in their request to route
        traces to different Phoenix projects dynamically.
        """
        standard_logging_payload = kwargs.get("standard_logging_object")
        if isinstance(standard_logging_payload, dict):
            metadata = standard_logging_payload.get("metadata")
            if isinstance(metadata, dict):
                project_name = metadata.get("phoenix_project_name")
                if project_name:
                    return str(project_name)

        # Also check litellm_params.metadata for SDK usage
        litellm_params = kwargs.get("litellm_params")
        if isinstance(litellm_params, dict):
            metadata = litellm_params.get("metadata") or {}
        else:
            metadata = {}
        if isinstance(metadata, dict):
            project_name = metadata.get("phoenix_project_name")
            if project_name:
                return str(project_name)

        return None

    def _handle_success(self, kwargs, response_obj, start_time, end_time):
        """
        Override to prevent creating duplicate litellm_request spans when a proxy parent span exists.
        
        ArizePhoenixLogger should reuse the proxy parent span instead of creating a new litellm_request span,
        to maintain a shallow span hierarchy as expected by Arize Phoenix.
        """
        from opentelemetry.trace import Status, StatusCode
        from litellm.secret_managers.main import get_secret_bool
        from litellm.integrations.opentelemetry import LITELLM_PROXY_REQUEST_SPAN_NAME
        
        verbose_logger.debug(
            "ArizePhoenixLogger: Logging kwargs: %s, OTEL config settings=%s",
            kwargs,
            self.config,
        )
        ctx, parent_span = self._get_span_context(kwargs)

        # ArizePhoenixLogger NEVER creates a litellm_request span when a proxy parent span exists
        # This is different from the base OpenTelemetry behavior which respects USE_OTEL_LITELLM_REQUEST_SPAN
        should_create_primary_span = parent_span is None or (
            parent_span.name != LITELLM_PROXY_REQUEST_SPAN_NAME
            and get_secret_bool("USE_OTEL_LITELLM_REQUEST_SPAN")
        )

        if should_create_primary_span:
            # Create a new litellm_request span
            span = self._start_primary_span(
                kwargs, response_obj, start_time, end_time, ctx
            )
            # Raw-request sub-span (if enabled) - child of litellm_request span
            self._maybe_log_raw_request(
                kwargs, response_obj, start_time, end_time, span
            )
            # Ensure proxy-request parent span is annotated with the actual operation kind
            if (
                parent_span is not None
                and parent_span.name == LITELLM_PROXY_REQUEST_SPAN_NAME
            ):
                self.set_attributes(parent_span, kwargs, response_obj)
        else:
            # Do not create primary span (keep hierarchy shallow when parent exists)
            span = None
            # Only set attributes if the span is still recording (not closed)
            # Note: parent_span is guaranteed to be not None here
            if parent_span.is_recording():
                parent_span.set_status(Status(StatusCode.OK))
                self.set_attributes(parent_span, kwargs, response_obj)
            # Raw-request as direct child of parent_span
            self._maybe_log_raw_request(
                kwargs, response_obj, start_time, end_time, parent_span
            )

        # 3. Guardrail span
        self._create_guardrail_span(kwargs=kwargs, context=ctx)

        # 4. Metrics & cost recording
        self._record_metrics(kwargs, response_obj, start_time, end_time)

        # 5. Semantic logs.
        if self.config.enable_events:
            log_span = span if span is not None else parent_span
            if log_span is not None:
                self._emit_semantic_logs(kwargs, response_obj, log_span)

        # 6. Do NOT end parent span - it should be managed by its creator
        # External spans (from Langfuse, user code, HTTP headers, global context) must not be closed by LiteLLM
        # However, proxy-created spans should be closed here
        if (
            parent_span is not None
            and parent_span.name == LITELLM_PROXY_REQUEST_SPAN_NAME
        ):
            parent_span.end(end_time=self._to_ns(end_time))

    @staticmethod
    def get_arize_phoenix_config() -> ArizePhoenixConfig:
        """
        Retrieves the Arize Phoenix configuration based on environment variables.
        Returns:
            ArizePhoenixConfig: A Pydantic model containing Arize Phoenix configuration.
        """
        api_key = os.environ.get("PHOENIX_API_KEY", None)

        collector_endpoint = os.environ.get("PHOENIX_COLLECTOR_HTTP_ENDPOINT", None)

        if not collector_endpoint:
            grpc_endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", None)
            http_endpoint = os.environ.get("PHOENIX_COLLECTOR_HTTP_ENDPOINT", None)
            collector_endpoint = http_endpoint or grpc_endpoint

        endpoint = None
        protocol: Protocol = "otlp_http"

        if collector_endpoint:
            # Parse the endpoint to determine protocol
            if collector_endpoint.startswith("grpc://") or (":4317" in collector_endpoint and "/v1/traces" not in collector_endpoint):
                endpoint = collector_endpoint
                protocol = "otlp_grpc"
            else:
                # Phoenix Cloud endpoints (app.phoenix.arize.com) include the space in the URL
                if "app.phoenix.arize.com" in collector_endpoint:
                    endpoint = collector_endpoint
                    protocol = "otlp_http"
                # For other HTTP endpoints, ensure they have the correct path
                elif "/v1/traces" not in collector_endpoint:
                    if collector_endpoint.endswith("/v1"):
                        endpoint = collector_endpoint + "/traces"
                    elif collector_endpoint.endswith("/"):
                        endpoint = f"{collector_endpoint}v1/traces"
                    else:
                        endpoint = f"{collector_endpoint}/v1/traces"
                else:
                    endpoint = collector_endpoint
                protocol = "otlp_http"
        else:
            # If no endpoint specified, self hosted phoenix
            endpoint = "http://localhost:6006/v1/traces"
            protocol = "otlp_http"
            verbose_logger.debug(
                f"No PHOENIX_COLLECTOR_ENDPOINT found, using default local Phoenix endpoint: {endpoint}"
            )

        otlp_auth_headers = None
        if api_key is not None:
            otlp_auth_headers = f"Authorization=Bearer {api_key}"
        elif "app.phoenix.arize.com" in endpoint:
            # Phoenix Cloud requires an API key
            raise ValueError(
                "PHOENIX_API_KEY must be set when using Phoenix Cloud (app.phoenix.arize.com)."
            )

        project_name = os.environ.get("PHOENIX_PROJECT_NAME", "default")

        return ArizePhoenixConfig(
            otlp_auth_headers=otlp_auth_headers,
            protocol=protocol,
            endpoint=endpoint,
            project_name=project_name,
        )
    
    ## cannot suppress additional proxy server spans, removed previous methods.

    async def async_health_check(self):

        config = self.get_arize_phoenix_config()

        if not config.otlp_auth_headers:
            return {
                "status": "unhealthy",
                "error_message": "PHOENIX_API_KEY environment variable not set",
            }

        return {
            "status": "healthy",
            "message": "Arize-Phoenix credentials are configured properly",
        }