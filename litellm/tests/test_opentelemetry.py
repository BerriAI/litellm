import asyncio
import litellm

from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def test_otel_callback():
    exporter = InMemorySpanExporter()

    litellm.callbacks = [OpenTelemetry(OpenTelemetryConfig(exporter=exporter))]

    litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
    )

    asyncio.run(
        litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
        )
    )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1 + 1
