# mypy: ignore-errors
import openai
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import SpanKind
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


trace.set_tracer_provider(TracerProvider())
memory_exporter = InMemorySpanExporter()
span_processor = SimpleSpanProcessor(memory_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
tracer = trace.get_tracer(__name__)

# create an otel traceparent header
tracer = trace.get_tracer(__name__)
with tracer.start_as_current_span("ishaan-local-dev-app") as span:
    span.set_attribute("generation_name", "ishaan-generation-openai-client")
    client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")
    extra_headers = {}
    context = trace.set_span_in_context(span)
    traceparent = TraceContextTextMapPropagator()
    traceparent.inject(carrier=extra_headers, context=context)
    print("EXTRA HEADERS: ", extra_headers)
    _trace_parent = extra_headers.get("traceparent")
    trace_id = _trace_parent.split("-")[1]
    print("Trace ID: ", trace_id)

    # # request sent to model set on litellm proxy, `litellm --model`
    response = client.chat.completions.create(
        model="llama3",
        messages=[
            {"role": "user", "content": "this is a test request, write a short poem"}
        ],
        extra_headers=extra_headers,
    )

    print(response)
