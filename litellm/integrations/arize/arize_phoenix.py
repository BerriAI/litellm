import os
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

from litellm._logging import verbose_logger
from litellm.integrations.arize import _utils
from litellm.integrations.arize._utils import ArizeOTELAttributes
from litellm.types.integrations.arize_phoenix import ArizePhoenixConfig

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SpanProcessor
    from opentelemetry.trace import Span as _Span
    from opentelemetry.trace import SpanKind
    from opentelemetry.trace import Tracer

    from litellm.integrations.opentelemetry import OpenTelemetry as _OpenTelemetry
    from litellm.integrations.opentelemetry import (
        OpenTelemetryConfig as _OpenTelemetryConfig,
    )
    from litellm.types.integrations.arize import Protocol as _Protocol

    Protocol = _Protocol
    OpenTelemetryConfig = _OpenTelemetryConfig
    Span = Union[_Span, Any]
    OpenTelemetry = _OpenTelemetry
    LITELLM_TRACER_NAME: str
else:
    Protocol = Any
    OpenTelemetryConfig = Any
    Span = Any
    Tracer = Any
    TracerProvider = Any
    SpanKind = Any
    SpanProcessor = Any
    try:
        from litellm.integrations.opentelemetry import (
            LITELLM_TRACER_NAME,
            OpenTelemetry,
        )
    except ImportError:
        LITELLM_TRACER_NAME = "litellm"
        OpenTelemetry = None  # type: ignore


ARIZE_HOSTED_PHOENIX_ENDPOINT = "https://otlp.arize.com/v1/traces"
_MAX_PROJECT_PROVIDERS = 64


class ArizePhoenixLogger(OpenTelemetry):  # type: ignore
    """
    Arize Phoenix logger that sends traces to a Phoenix endpoint.

    Creates its own dedicated TracerProvider so it can coexist with the
    generic ``otel`` callback (or any other OTEL-based integration) without
    fighting over the global ``opentelemetry.trace`` TracerProvider singleton.
    """

    def _init_tracing(self, tracer_provider):
        """
        Override to create per-project TracerProviders (LRU-cached) for Arize Phoenix.

        The base ``OpenTelemetry._init_tracing`` falls back to the global
        TracerProvider when one already exists.  That causes whichever
        integration initialises second to silently reuse the first one's
        exporter, so spans only reach one destination.
        """
        from opentelemetry.trace import SpanKind

        if tracer_provider is not None:
            self._use_injected_tracer_provider = True
            self._shared_span_processor = None
            self.tracer = tracer_provider.get_tracer(LITELLM_TRACER_NAME)
            self.span_kind = SpanKind
            return

        self._use_injected_tracer_provider = False
        self._project_providers: OrderedDict[str, TracerProvider] = OrderedDict()
        self._project_providers_lock = threading.Lock()
        self._shared_span_processor = self._get_span_processor()
        self.span_kind = SpanKind

        default_project = self._resolve_project_name({})
        self.tracer = self._get_tracer_for(default_project)
        verbose_logger.debug(
            "ArizePhoenixLogger: Initialized per-project TracerProvider cache "
            "(default_project=%s, endpoint=%s, exporter=%s)",
            default_project,
            self.config.endpoint,
            self.config.exporter,
        )

    def flush_tracer_providers(self) -> None:
        """
        Flush all cached per-project providers and the shared span processor.

        Call on graceful proxy shutdown. Do not call on LRU eviction — in-flight
        spans may still reference evicted providers.
        """
        if getattr(self, "_use_injected_tracer_provider", False):
            return

        shared_processor = getattr(self, "_shared_span_processor", None)
        if shared_processor is not None:
            try:
                shared_processor.force_flush()
            except Exception as e:
                verbose_logger.debug(
                    "ArizePhoenixLogger: shared span processor force_flush failed: %s",
                    e,
                )

        with getattr(self, "_project_providers_lock", threading.Lock()):
            providers = list(getattr(self, "_project_providers", {}).values())

        for provider in providers:
            try:
                provider.force_flush()
            except Exception as e:
                verbose_logger.debug("ArizePhoenixLogger: TracerProvider force_flush failed: %s", e)

    def _get_litellm_resource_for_project(self, project_name: str):
        """
        Build an OTEL Resource with project routing attrs that win over env detector.

        Phoenix uses ``openinference.project.name``; Arize AX uses ``model_id`` and
        ``service.name``. Project attrs are merged last so OTEL_RESOURCE_ATTRIBUTES
        from init does not pin every provider to one project.
        """
        from opentelemetry.sdk.resources import OTELResourceDetector, Resource

        project_attributes: dict[str, str] = {
            "openinference.project.name": project_name,
            "model_id": project_name,
            "service.name": project_name,
        }
        deployment_environment = getattr(self.config, "deployment_environment", None)
        if deployment_environment is not None:
            project_attributes["deployment.environment"] = deployment_environment

        env_resource = OTELResourceDetector().detect()
        project_resource = Resource.create(project_attributes)  # type: ignore[arg-type]
        return env_resource.merge(project_resource)

    def _build_tracer_provider_for_project(self, project_name: str) -> TracerProvider:
        """Create a TracerProvider for *project_name* (caller holds no cache lock)."""
        from opentelemetry.sdk.trace import TracerProvider

        provider = TracerProvider(resource=self._get_litellm_resource_for_project(project_name))
        provider.add_span_processor(self._shared_span_processor)
        return provider

    def _get_tracer_for(self, project_name: str) -> Tracer:
        """Return a tracer for *project_name*, creating/caching a provider on miss."""
        if getattr(self, "_use_injected_tracer_provider", False):
            return self.tracer

        with self._project_providers_lock:
            if project_name in self._project_providers:
                self._project_providers.move_to_end(project_name)
                return self._project_providers[project_name].get_tracer(LITELLM_TRACER_NAME)

        # OTELResourceDetector().detect() is synchronous; build outside the lock so
        # concurrent requests for other projects are not blocked on cache misses.
        new_provider = self._build_tracer_provider_for_project(project_name)

        with self._project_providers_lock:
            if project_name in self._project_providers:
                self._project_providers.move_to_end(project_name)
                return self._project_providers[project_name].get_tracer(LITELLM_TRACER_NAME)

            if len(self._project_providers) >= _MAX_PROJECT_PROVIDERS:
                self._project_providers.popitem(last=False)

            self._project_providers[project_name] = new_provider
            return new_provider.get_tracer(LITELLM_TRACER_NAME)

    def _resolve_tracer_for_kwargs(self, kwargs: dict) -> Tuple[str, Tracer]:
        """Resolve project name once and return the matching tracer."""
        project_name = self._resolve_project_name(kwargs)
        return project_name, self._get_tracer_for(project_name)

    def get_tracer_to_use_for_request(self, kwargs: dict) -> Tracer:
        """Route guardrail/raw-request spans to the same per-project tracer as the request."""
        if getattr(self, "_use_injected_tracer_provider", False):
            return self.tracer
        return self._resolve_tracer_for_kwargs(kwargs)[1]

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
        _utils.set_attributes(span, kwargs, response_obj, ArizeOTELAttributes)
        return

    @staticmethod
    def _normalize_project_name(name: Optional[str]) -> Optional[str]:
        if name is None:
            return None
        normalized = str(name).strip()
        return normalized if normalized else None

    @staticmethod
    def _iter_metadata_dicts_from_kwargs(kwargs: dict):
        """Yield request metadata dicts; standard_logging_object before litellm_params."""
        for key in ("standard_logging_object", "litellm_params"):
            found_key = kwargs.get(key)
            if not isinstance(found_key, dict):
                continue
            metadata = found_key.get("metadata")
            if isinstance(metadata, dict):
                yield metadata

    @staticmethod
    def _is_proxy_request(kwargs: dict) -> bool:
        """True when the call is routed through the LiteLLM proxy.

        Proxy mode is determined solely by the server-set ``proxy_server_request``
        field in ``litellm_params``.  Checking request metadata for
        ``user_api_key_auth_metadata`` is intentionally avoided: that field is
        user-supplied and would let an authenticated caller fake proxy-mode
        detection to route their telemetry into arbitrary Arize/Phoenix projects.
        """
        litellm_params = kwargs.get("litellm_params")
        return isinstance(litellm_params, dict) and bool(litellm_params.get("proxy_server_request"))

    @staticmethod
    def _project_from_metadata_dict(metadata: dict, metadata_key: str, *, proxy_mode: bool) -> Optional[str]:
        """
        Read a Phoenix project field from proxy/SDK metadata.

        On the proxy, only ``user_api_key_auth_metadata`` (team/key config) may
        select the project. SDK callers may still set project fields directly on
        ``metadata``.
        """
        auth_metadata = metadata.get("user_api_key_auth_metadata")
        if isinstance(auth_metadata, dict):
            project = ArizePhoenixLogger._normalize_project_name(auth_metadata.get(metadata_key))
            if project:
                return project

        if not proxy_mode:
            return ArizePhoenixLogger._normalize_project_name(metadata.get(metadata_key))
        return None

    @staticmethod
    def _metadata_project_from_kwargs(kwargs: dict, metadata_key: str) -> Optional[str]:
        proxy_mode = ArizePhoenixLogger._is_proxy_request(kwargs)
        for metadata in ArizePhoenixLogger._iter_metadata_dicts_from_kwargs(kwargs):
            project = ArizePhoenixLogger._project_from_metadata_dict(metadata, metadata_key, proxy_mode=proxy_mode)
            if project:
                return project
        return None

    @staticmethod
    def _resolve_project_name(kwargs: dict) -> str:
        """
        Resolve the target Phoenix/Arize project for this request.

        Proxy priority: ``user_api_key_auth_metadata.phoenix_project_name_override``,
        ``user_api_key_auth_metadata.phoenix_project_name``, env, then ``default``.
        SDK priority: request metadata fields, then env, then ``default``.
        """
        override = ArizePhoenixLogger._metadata_project_from_kwargs(kwargs, "phoenix_project_name_override")
        if override:
            return override

        phoenix_name = ArizePhoenixLogger._metadata_project_from_kwargs(kwargs, "phoenix_project_name")
        if phoenix_name:
            return phoenix_name

        env_name = ArizePhoenixLogger._normalize_project_name(
            os.environ.get("PHOENIX_PROJECT_NAME") or os.environ.get("ARIZE_PROJECT_NAME")
        )
        if env_name:
            return env_name

        return "default"

    def _get_phoenix_context(self, kwargs, tracer: Optional[Tracer] = None):
        """
        Build a trace context for Phoenix's dedicated TracerProvider.

        The base ``_get_span_context`` returns parent spans from the global
        TracerProvider (the ``otel`` callback).  Those spans live on a
        *different* TracerProvider, so they won't appear in Phoenix — using
        them as parents just creates broken links.

        Instead we:
        1. Honour an incoming ``traceparent`` HTTP header (distributed tracing).
        2. In proxy mode, create our *own* parent span on Phoenix's tracer
           so the hierarchy is visible end-to-end inside Phoenix.
        3. In SDK (non-proxy) mode, just return (None, None) for a root span.
        """
        from opentelemetry import trace

        if tracer is None:
            tracer = self._resolve_tracer_for_kwargs(kwargs)[1]

        litellm_params = kwargs.get("litellm_params", {}) or {}
        proxy_server_request = litellm_params.get("proxy_server_request", {}) or {}
        headers = proxy_server_request.get("headers", {}) or {}

        traceparent_ctx = self.get_traceparent_from_header(headers=headers) if headers.get("traceparent") else None

        is_proxy_mode = bool(proxy_server_request)

        if is_proxy_mode:
            start_time_val = kwargs.get("start_time", kwargs.get("api_call_start_time"))
            parent_span = tracer.start_span(
                name="litellm_proxy_request",
                start_time=(self._to_ns(start_time_val) if start_time_val is not None else None),
                context=traceparent_ctx,
                kind=self.span_kind.SERVER,
            )
            ctx = trace.set_span_in_context(parent_span)
            return ctx, parent_span

        return traceparent_ctx, None

    def _handle_success(self, kwargs, response_obj, start_time, end_time):
        self._handle_phoenix_trace(kwargs, response_obj, start_time, end_time, success=True)

    def _handle_failure(self, kwargs, response_obj, start_time, end_time):
        self._handle_phoenix_trace(kwargs, response_obj, start_time, end_time, success=False)

    def _handle_phoenix_trace(
        self,
        kwargs,
        response_obj,
        start_time,
        end_time,
        *,
        success: bool,
    ):
        from opentelemetry.trace import Status, StatusCode

        verbose_logger.debug(
            "ArizePhoenixLogger: %s - kwargs: %s, OTEL config settings=%s",
            "success" if success else "failure",
            kwargs,
            self.config,
        )

        _project_name, tracer = self._resolve_tracer_for_kwargs(kwargs)
        ctx, parent_span = self._get_phoenix_context(kwargs, tracer=tracer)

        status = Status(StatusCode.OK if success else StatusCode.ERROR)

        span = tracer.start_span(
            name=self._get_span_name(kwargs),
            start_time=self._to_ns(start_time),
            context=ctx,
        )
        span.set_status(status)
        self.set_attributes(span, kwargs, response_obj)
        if not success:
            self._record_exception_on_span(span=span, kwargs=kwargs)

        if success:
            self._maybe_log_raw_request(kwargs, response_obj, start_time, end_time, span)
        span.end(end_time=self._to_ns(end_time))

        self._create_guardrail_span(kwargs=kwargs, context=ctx)

        if parent_span is not None:
            parent_span.set_status(status)
            self.set_attributes(parent_span, kwargs, response_obj)
            if not success:
                self._record_exception_on_span(span=parent_span, kwargs=kwargs)
            parent_span.end(end_time=self._to_ns(end_time))

        if success:
            self._record_metrics(kwargs, response_obj, start_time, end_time)

            if self.config.enable_events:
                self._emit_semantic_logs(kwargs, response_obj, span)

    @staticmethod
    def get_arize_phoenix_config() -> ArizePhoenixConfig:
        """
        Retrieves the Arize Phoenix configuration based on environment variables.
        Returns:
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
            if collector_endpoint.startswith("grpc://") or (
                ":4317" in collector_endpoint and "/v1/traces" not in collector_endpoint
            ):
                endpoint = collector_endpoint
                protocol = "otlp_grpc"
            else:
                if "app.phoenix.arize.com" in collector_endpoint:
                    endpoint = collector_endpoint
                    protocol = "otlp_http"
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
            endpoint = "http://localhost:6006/v1/traces"
            protocol = "otlp_http"
            verbose_logger.debug(
                f"No PHOENIX_COLLECTOR_ENDPOINT found, using default local Phoenix endpoint: {endpoint}"
            )

        otlp_auth_headers = None
        if api_key is not None:
            otlp_auth_headers = f"Authorization=Bearer {api_key}"
        elif "app.phoenix.arize.com" in endpoint:
            raise ValueError("PHOENIX_API_KEY must be set when using Phoenix Cloud (app.phoenix.arize.com).")

        project_name = os.environ.get("PHOENIX_PROJECT_NAME") or "default"

        return ArizePhoenixConfig(
            otlp_auth_headers=otlp_auth_headers,
            protocol=protocol,
            endpoint=endpoint,
            project_name=project_name,
        )

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
