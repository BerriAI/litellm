import os

import litellm
from litellm.integrations.custom_logger import CustomLogger

from opentelemetry import trace

from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)


LITELLM_TRACER_NAME = "litellm"
LITELLM_RESOURCE = {"service.name": "litellm"}

MOCK_TRACE_PARENT = {"traceparent": "SOMETHING_FROM_PROXY_REQUEST"}
MOCK_SPAN_NAME = "TODO"


class OpenTelemetry(CustomLogger):
    def __init__(self):
        provider = TracerProvider(resource=Resource(attributes=LITELLM_RESOURCE))
        provider.add_span_processor(self.get_span_processor())

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
        span = self.tracer.start_span(
            MOCK_SPAN_NAME,
            start_time=self._to_ns(start_time),
            context=TraceContextTextMapPropagator().extract(carrier=MOCK_TRACE_PARENT),
        )
        span.set_status(Status(StatusCode.OK))
        self._set_attributes(span, kwargs, response_obj)
        span.end(end_time=self._to_ns(end_time))

    def _handle_failure(self, kwargs, response_obj, start_time, end_time):
        span = self.tracer.start_span(
            MOCK_SPAN_NAME,
            start_time=self._to_ns(start_time),
            context=TraceContextTextMapPropagator().extract(carrier=MOCK_TRACE_PARENT),
        )
        span.set_status(Status(StatusCode.ERROR))
        self._set_attributes(span, kwargs, response_obj)
        span.end(end_time=self._to_ns(end_time))

    def _to_ns(self, dt):
        return int(dt.timestamp() * 1e9)

    def _set_attributes(self, span, kwargs, response_obj):
        keys = ["model", "api_base", "api_version"]
        for key in keys:
            if key in kwargs:
                span.set_attribute("model", kwargs[key])

    def get_span_processor(self):
        if litellm.set_verbose:
            BatchSpanProcessor(ConsoleSpanExporter())
        else:
            BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint=os.getenv("OTEL_ENDPOINT"),
                    headers={
                        "Authorization": f"Bearer {os.getenv('OTEL_BEARER_TOKEN')}"
                    },
                )
            )
