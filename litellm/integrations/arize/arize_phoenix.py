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
    from litellm.integrations.opentelemetry import (
        OpenTelemetryConfig as _OpenTelemetryConfig,
    )
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

    async def async_post_call_success_hook(
        self,
        data,
        user_api_key_dict,
        response,
    ):
        """
        skipping this hook prevents both the orphan and the duplicate guardrail spans.
        already handled in global tracer provider
        """
        return response

    def set_attributes(self, span: Span, kwargs, response_obj: Optional[Any]):
        ArizePhoenixLogger.set_arize_phoenix_attributes(span, kwargs, response_obj)
        return

    @staticmethod
    def set_arize_phoenix_attributes(span: Span, kwargs, response_obj):
        from litellm.integrations.opentelemetry_utils.base_otel_llm_obs_attributes import (
            safe_set_attribute,
        )

        _utils.set_attributes(span, kwargs, response_obj, ArizeOTELAttributes)

        # Dynamic project name: check metadata first, then fall back to env var config
        dynamic_project_name = ArizePhoenixLogger._get_dynamic_project_name(kwargs)
        if dynamic_project_name:
            safe_set_attribute(span, "openinference.project.name", dynamic_project_name)
        else:
            # Fall back to static config from env var
            config = ArizePhoenixLogger.get_arize_phoenix_config()
            if config.project_name:
                safe_set_attribute(
                    span, "openinference.project.name", config.project_name
                )

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

    # ------------------------------------------------------------------
    # Router-level parent span — one per logical user request so all
    # retry + fallback attempt spans nest under a single root in Phoenix.
    # The span object is stored on the logger instance (never in kwargs /
    # metadata) to avoid JSON-serialisation errors when request data is
    # forwarded to providers.
    # ------------------------------------------------------------------

    def _router_span_registry(self) -> dict:
        if not hasattr(self, "_router_parent_spans"):
            self._router_parent_spans: dict = {}
        return self._router_parent_spans

    @staticmethod
    def _router_call_id(kwargs) -> Optional[str]:
        call_id = kwargs.get("litellm_trace_id") or kwargs.get("litellm_call_id")
        if call_id:
            return str(call_id)
        litellm_params = kwargs.get("litellm_params") or {}
        call_id = litellm_params.get("litellm_trace_id") or litellm_params.get(
            "litellm_call_id"
        )
        return str(call_id) if call_id else None

    def start_router_parent_span(self, kwargs: dict) -> None:
        import uuid

        from opentelemetry import trace
        from opentelemetry.trace import SpanKind

        call_id = self._router_call_id(kwargs)
        if call_id is None:
            call_id = str(uuid.uuid4())
            kwargs["litellm_trace_id"] = call_id

        registry = self._router_span_registry()
        if call_id in registry:
            return

        # Evict stale ended entries (span=None) to bound registry size.
        if len(registry) > 200:
            stale = [k for k, v in registry.items() if v[0] is None]
            for k in stale:
                del registry[k]

        litellm_params = kwargs.get("litellm_params") or {}
        proxy_server_request = litellm_params.get("proxy_server_request") or {}
        headers = proxy_server_request.get("headers") or {}
        traceparent_ctx = (
            self.get_traceparent_from_header(headers=headers)
            if headers.get("traceparent")
            else None
        )

        kind = SpanKind.SERVER if proxy_server_request else SpanKind.INTERNAL

        span = self.tracer.start_span(
            name="litellm_proxy_request",
            context=traceparent_ctx,
            kind=kind,
        )
        span.set_attribute("openinference.span.kind", "CHAIN")
        span.set_attribute("llm.request.model", str(kwargs.get("model", "")))

        ctx = trace.set_span_in_context(span)
        registry[call_id] = (span, ctx)

    def end_router_parent_span(
        self, kwargs: dict, exception: Optional[BaseException] = None
    ) -> None:
        from opentelemetry.trace import Status, StatusCode

        call_id = self._router_call_id(kwargs)
        if call_id is None:
            return

        registry = self._router_span_registry()
        entry = registry.get(call_id)
        if entry is None:
            return

        span, ctx = entry
        if span is None:
            return  # already ended

        if exception is not None:
            span.set_status(Status(StatusCode.ERROR))
            try:
                span.record_exception(exception)
            except Exception:
                pass
        else:
            span.set_status(Status(StatusCode.OK))

        span.end()

        # Keep ctx in registry so async success/failure callbacks that fire
        # after this finally-block can still attach child spans to the same
        # trace. An ended span's context (trace_id + span_id) remains valid
        # for parenting. Stale entries are evicted in start_router_parent_span.
        registry[call_id] = (None, ctx)

    def _get_router_parent_ctx(self, kwargs):
        """Return (ctx, None) if a router parent exists for this call, else (None, None)."""
        registry = getattr(self, "_router_parent_spans", None)
        if not registry:
            return None, None
        call_id = self._router_call_id(kwargs)
        if not call_id:
            return None, None
        entry = registry.get(call_id)
        if entry is None:
            return None, None
        _span, ctx = entry
        return ctx, None

    def _get_phoenix_context(self, kwargs):
        """
        Build a trace context for Phoenix's dedicated TracerProvider.

        Priority:
        1. Router parent span (covers all retries + fallbacks under one root).
        2. Incoming ``traceparent`` header (distributed tracing).
        3. Proxy mode — create a per-call ``litellm_proxy_request`` parent.
        4. SDK mode — root span, no parent.
        """
        from opentelemetry import trace

        # 1. Router parent already open for this call — attach as child.
        ctx, _ = self._get_router_parent_ctx(kwargs)
        if ctx is not None:
            return ctx, None

        litellm_params = kwargs.get("litellm_params", {}) or {}
        proxy_server_request = litellm_params.get("proxy_server_request", {}) or {}
        headers = proxy_server_request.get("headers", {}) or {}

        # 2. Propagate distributed trace context if the caller sent a traceparent
        traceparent_ctx = (
            self.get_traceparent_from_header(headers=headers)
            if headers.get("traceparent")
            else None
        )

        is_proxy_mode = bool(proxy_server_request)

        if is_proxy_mode:
            # 3. Create a per-call parent on Phoenix's own tracer.
            start_time_val = kwargs.get("start_time", kwargs.get("api_call_start_time"))
            parent_span = self.tracer.start_span(
                name="litellm_proxy_request",
                start_time=(
                    self._to_ns(start_time_val) if start_time_val is not None else None
                ),
                context=traceparent_ctx,
                kind=self.span_kind.SERVER,
            )
            ctx = trace.set_span_in_context(parent_span)
            return ctx, parent_span

        # 4. SDK mode — no parent span needed
        return traceparent_ctx, None

    def _handle_success(self, kwargs, response_obj, start_time, end_time):
        """
        Override to always create spans on ArizePhoenixLogger's dedicated TracerProvider.

        The base class's ``_get_span_context`` would find the parent span created by
        the ``otel`` callback on the *global* TracerProvider.  That span is invisible
        in Phoenix (different exporter pipeline), so we ignore it and build our own
        hierarchy via ``_get_phoenix_context``.
        """
        from opentelemetry.trace import Status, StatusCode

        verbose_logger.debug(
            "ArizePhoenixLogger: Logging kwargs: %s, OTEL config settings=%s",
            kwargs,
            self.config,
        )

        ctx, parent_span = self._get_phoenix_context(kwargs)

        # Create litellm_request span (child of our parent when in proxy mode)
        span = self.tracer.start_span(
            name=self._get_span_name(kwargs),
            start_time=self._to_ns(start_time),
            context=ctx,
        )
        span.set_status(Status(StatusCode.OK))
        self.set_attributes(span, kwargs, response_obj)

        # Raw-request sub-span (if enabled) — must be created before
        # ending the parent span so the hierarchy is valid.
        self._maybe_log_raw_request(kwargs, response_obj, start_time, end_time, span)

        # Guardrail context: in proxy mode it's a sibling of litellm_request
        # under litellm_proxy_request. In SDK mode there is no proxy parent,
        # so we parent it to the litellm_request span to avoid an orphan.
        from opentelemetry import trace as _trace

        guardrail_ctx = (
            ctx if parent_span is not None else _trace.set_span_in_context(span)
        )

        span.end(end_time=self._to_ns(end_time))

        # Guardrail span (always parented — no orphan roots)
        self._create_guardrail_span(kwargs=kwargs, context=guardrail_ctx)

        # Annotate and close our proxy parent span.
        # Only session/request metadata goes on the parent — the child span
        # already carries the full LLM payload (messages, tokens, response).
        # Duplicating them here double-counts tokens.
        if parent_span is not None:
            parent_span.set_status(Status(StatusCode.OK))
            _utils.set_parent_span_attributes(parent_span, kwargs, response_obj)
            parent_span.end(end_time=self._to_ns(end_time))

        # Metrics & cost recording
        self._record_metrics(kwargs, response_obj, start_time, end_time)

        # Semantic logs
        if self.config.enable_events:
            self._emit_semantic_logs(kwargs, response_obj, span)

    def _handle_failure(self, kwargs, response_obj, start_time, end_time):
        """
        Override to always create failure spans on ArizePhoenixLogger's dedicated
        TracerProvider.  Mirrors ``_handle_success`` but sets ERROR status.
        """
        from opentelemetry.trace import Status, StatusCode

        verbose_logger.debug(
            "ArizePhoenixLogger: Failure - Logging kwargs: %s, OTEL config settings=%s",
            kwargs,
            self.config,
        )

        ctx, parent_span = self._get_phoenix_context(kwargs)

        # Create litellm_request span (child of our parent when in proxy mode)
        span = self.tracer.start_span(
            name=self._get_span_name(kwargs),
            start_time=self._to_ns(start_time),
            context=ctx,
        )
        span.set_status(Status(StatusCode.ERROR))
        self.set_attributes(span, kwargs, response_obj)
        self._record_exception_on_span(span=span, kwargs=kwargs)

        # See _handle_success for guardrail context rationale.
        from opentelemetry import trace as _trace

        guardrail_ctx = (
            ctx if parent_span is not None else _trace.set_span_in_context(span)
        )

        span.end(end_time=self._to_ns(end_time))

        # Guardrail span (always parented, no orphan roots)
        self._create_guardrail_span(kwargs=kwargs, context=guardrail_ctx)

        # Annotate and close our proxy parent span (see _handle_success for
        # why only parent-scoped attributes go here).
        if parent_span is not None:
            parent_span.set_status(Status(StatusCode.ERROR))
            _utils.set_parent_span_attributes(parent_span, kwargs, response_obj)
            self._record_exception_on_span(span=parent_span, kwargs=kwargs)
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
            if collector_endpoint.startswith("grpc://") or (
                ":4317" in collector_endpoint and "/v1/traces" not in collector_endpoint
            ):
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

    async def log_success_fallback_event(
        self,
        original_model_group: str,
        kwargs: dict,
        original_exception: Exception,
    ):
        from opentelemetry.trace import Status, StatusCode

        fallback_model = kwargs.get("model")
        fallback_depth = kwargs.get("fallback_depth", 1)
        status_code = getattr(original_exception, "status_code", None)
        exception_class = original_exception.__class__.__name__

        ctx, _ = self._get_router_parent_ctx(kwargs)
        span = self.tracer.start_span(
            name=f"fallback: {original_model_group} -> {fallback_model}",
            context=ctx,
        )
        span.set_attribute("llm.fallback.event_type", "success")
        span.set_attribute("llm.fallback.original_model", str(original_model_group))
        span.set_attribute("llm.fallback.fallback_model", str(fallback_model))
        span.set_attribute("llm.fallback.attempt_number", fallback_depth)
        span.set_attribute("llm.fallback.error_class", exception_class)
        if status_code is not None:
            span.set_attribute("llm.fallback.error_status_code", int(status_code))
        span.add_event(
            "fallback_triggered",
            attributes={
                "trigger.model": str(original_model_group),
                "trigger.error_class": exception_class,
                "trigger.status_code": str(status_code or ""),
                "fallback.model": str(fallback_model),
            },
        )
        span.set_status(Status(StatusCode.OK))
        span.end()

    async def log_failure_fallback_event(
        self,
        original_model_group: str,
        kwargs: dict,
        original_exception: Exception,
    ):
        from opentelemetry.trace import Status, StatusCode

        fallback_attempted = kwargs.get("model")
        max_fallbacks = kwargs.get("max_fallbacks", 0)
        status_code = getattr(original_exception, "status_code", None)
        exception_class = original_exception.__class__.__name__

        ctx, _ = self._get_router_parent_ctx(kwargs)
        span = self.tracer.start_span(
            name=f"fallback_failed: {original_model_group}",
            context=ctx,
        )
        span.set_attribute("llm.fallback.event_type", "failure")
        span.set_attribute("llm.fallback.original_model", str(original_model_group))
        span.set_attribute("llm.fallback.attempted_model", str(fallback_attempted))
        span.set_attribute("llm.fallback.max_fallbacks", int(max_fallbacks))
        span.set_attribute("llm.fallback.error_class", exception_class)
        if status_code is not None:
            span.set_attribute("llm.fallback.error_status_code", int(status_code))
        span.record_exception(original_exception)
        span.set_status(Status(StatusCode.ERROR))
        span.end()
