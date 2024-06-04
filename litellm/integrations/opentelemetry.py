import os
from typing import Optional
from dataclasses import dataclass

from litellm.integrations.custom_logger import CustomLogger
from litellm._logging import verbose_logger

LITELLM_TRACER_NAME = "litellm"
LITELLM_RESOURCE = {"service.name": "litellm"}


@dataclass
class OpenTelemetryConfig:
    from opentelemetry.sdk.trace.export import SpanExporter

    exporter: str | SpanExporter = "console"
    endpoint: Optional[str] = None
    headers: Optional[str] = None

    @classmethod
    def from_env(cls):
        """
        OTEL_HEADERS=x-honeycomb-team=B85YgLm9****
        OTEL_EXPORTER="otlp_http"
        OTEL_ENDPOINT="https://api.honeycomb.io/v1/traces"

        OTEL_HEADERS gets sent as headers = {"x-honeycomb-team": "B85YgLm96******"}
        """
        return cls(
            exporter=os.getenv("OTEL_EXPORTER", "console"),
            endpoint=os.getenv("OTEL_ENDPOINT"),
            headers=os.getenv(
                "OTEL_HEADERS"
            ),  # example: OTEL_HEADERS=x-honeycomb-team=B85YgLm96VGdFisfJVme1H"
        )


class OpenTelemetry(CustomLogger):
    def __init__(self, config=OpenTelemetryConfig.from_env()):
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        self.config = config
        self.OTEL_EXPORTER = self.config.exporter
        self.OTEL_ENDPOINT = self.config.endpoint
        self.OTEL_HEADERS = self.config.headers
        provider = TracerProvider(resource=Resource(attributes=LITELLM_RESOURCE))
        provider.add_span_processor(self._get_span_processor())

        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(LITELLM_TRACER_NAME)

        if bool(os.getenv("DEBUG_OTEL", False)) is True:
            # Set up logging
            import logging

            logging.basicConfig(level=logging.DEBUG)
            logger = logging.getLogger(__name__)

            # Enable OpenTelemetry logging
            otel_exporter_logger = logging.getLogger("opentelemetry.sdk.trace.export")
            otel_exporter_logger.setLevel(logging.DEBUG)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_sucess(kwargs, response_obj, start_time, end_time)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_failure(kwargs, response_obj, start_time, end_time)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_sucess(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_failure(kwargs, response_obj, start_time, end_time)

    def _handle_sucess(self, kwargs, response_obj, start_time, end_time):
        from opentelemetry.trace import Status, StatusCode

        verbose_logger.debug(
            "OpenTelemetry Logger: Logging kwargs: %s, OTEL config settings=%s",
            kwargs,
            self.config,
        )

        span = self.tracer.start_span(
            name=self._get_span_name(kwargs),
            start_time=self._to_ns(start_time),
            context=self._get_span_context(kwargs),
        )
        span.set_status(Status(StatusCode.OK))
        self.set_attributes(span, kwargs, response_obj)
        span.end(end_time=self._to_ns(end_time))

    def _handle_failure(self, kwargs, response_obj, start_time, end_time):
        from opentelemetry.trace import Status, StatusCode

        span = self.tracer.start_span(
            name=self._get_span_name(kwargs),
            start_time=self._to_ns(start_time),
            context=self._get_span_context(kwargs),
        )
        span.set_status(Status(StatusCode.ERROR))
        self.set_attributes(span, kwargs, response_obj)
        span.end(end_time=self._to_ns(end_time))

    def set_attributes(self, span, kwargs, response_obj):
        for key in ["model", "api_base", "api_version"]:
            if key in kwargs:
                span.set_attribute(key, kwargs[key])

    def _to_ns(self, dt):
        return int(dt.timestamp() * 1e9)

    def _get_span_name(self, kwargs):
        return f"litellm-{kwargs.get('call_type', 'completion')}"

    def _get_span_context(self, kwargs):
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )

        litellm_params = kwargs.get("litellm_params", {}) or {}
        proxy_server_request = litellm_params.get("proxy_server_request", {}) or {}
        headers = proxy_server_request.get("headers", {}) or {}
        traceparent = headers.get("traceparent", None)

        if traceparent is None:
            return None
        else:
            carrier = {"traceparent": traceparent}
            return TraceContextTextMapPropagator().extract(carrier=carrier)

    def _get_span_processor(self):
        from opentelemetry.sdk.trace.export import (
            SpanExporter,
            SimpleSpanProcessor,
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterHTTP,
        )
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterGRPC,
        )

        verbose_logger.debug(
            "OpenTelemetry Logger, initializing span processor \nself.OTEL_EXPORTER: %s\nself.OTEL_ENDPOINT: %s\nself.OTEL_HEADERS: %s",
            self.OTEL_EXPORTER,
            self.OTEL_ENDPOINT,
            self.OTEL_HEADERS,
        )
        _split_otel_headers = {}
        if self.OTEL_HEADERS is not None and isinstance(self.OTEL_HEADERS, str):
            _split_otel_headers = self.OTEL_HEADERS.split("=")
            _split_otel_headers = {_split_otel_headers[0]: _split_otel_headers[1]}

        if isinstance(self.OTEL_EXPORTER, SpanExporter):
            verbose_logger.debug(
                "OpenTelemetry: intiializing SpanExporter. Value of OTEL_EXPORTER: %s",
                self.OTEL_EXPORTER,
            )
            return SimpleSpanProcessor(self.OTEL_EXPORTER)

        if self.OTEL_EXPORTER == "console":
            verbose_logger.debug(
                "OpenTelemetry: intiializing console exporter. Value of OTEL_EXPORTER: %s",
                self.OTEL_EXPORTER,
            )
            return BatchSpanProcessor(ConsoleSpanExporter())
        elif self.OTEL_EXPORTER == "otlp_http":
            verbose_logger.debug(
                "OpenTelemetry: intiializing http exporter. Value of OTEL_EXPORTER: %s",
                self.OTEL_EXPORTER,
            )
            return BatchSpanProcessor(
                OTLPSpanExporterHTTP(
                    endpoint=self.OTEL_ENDPOINT, headers=_split_otel_headers
                )
            )
        elif self.OTEL_EXPORTER == "otlp_grpc":
            verbose_logger.debug(
                "OpenTelemetry: intiializing grpc exporter. Value of OTEL_EXPORTER: %s",
                self.OTEL_EXPORTER,
            )
            return BatchSpanProcessor(
                OTLPSpanExporterGRPC(
                    endpoint=self.OTEL_ENDPOINT, headers=_split_otel_headers
                )
            )
        else:
            verbose_logger.debug(
                "OpenTelemetry: intiializing console exporter. Value of OTEL_EXPORTER: %s",
                self.OTEL_EXPORTER,
            )
            return BatchSpanProcessor(ConsoleSpanExporter())
