import os
from typing import Optional
from dataclasses import dataclass

from litellm.integrations.custom_logger import CustomLogger


LITELLM_TRACER_NAME = "litellm"
LITELLM_RESOURCE = {"service.name": "litellm"}


@dataclass
class OpenTelemetryConfig:
    from opentelemetry.sdk.trace.export import SpanExporter

    exporter: str | SpanExporter = "console"
    endpoint: Optional[str] = None
    bearer_token: Optional[str] = None

    @classmethod
    def from_env(cls):
        return cls(
            exporter=os.getenv("OTEL_EXPORTER", "console"),
            endpoint=os.getenv("OTEL_ENDPOINT"),
            bearer_token=os.getenv("OTEL_BEARER_TOKEN"),
        )


class OpenTelemetry(CustomLogger):
    def __init__(self, config=OpenTelemetryConfig.from_env()):
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        self.config = config
        provider = TracerProvider(resource=Resource(attributes=LITELLM_RESOURCE))
        provider.add_span_processor(self._get_span_processor())

        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(LITELLM_TRACER_NAME)

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

        if isinstance(self.config.exporter, SpanExporter):
            return SimpleSpanProcessor(self.config.exporter)

        if self.config.exporter == "console":
            return BatchSpanProcessor(ConsoleSpanExporter())
        elif self.config.exporter == "otlp_http":
            return BatchSpanProcessor(
                OTLPSpanExporterHTTP(
                    endpoint=self.OTEL_ENDPOINT,
                    headers={"Authorization": f"Bearer {self.OTEL_BEARER_TOKEN}"},
                )
            )
        elif self.config.exporter == "otlp_grpc":
            return BatchSpanProcessor(
                OTLPSpanExporterGRPC(
                    endpoint=self.OTEL_ENDPOINT,
                    headers={"Authorization": f"Bearer {self.OTEL_BEARER_TOKEN}"},
                )
            )
        else:
            return BatchSpanProcessor(ConsoleSpanExporter())
