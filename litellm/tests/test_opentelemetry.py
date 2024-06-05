import asyncio
import litellm

from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from litellm._logging import verbose_logger
import logging
import time
import pytest

verbose_logger.setLevel(logging.DEBUG)


@pytest.mark.skip(reason="new test")
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

    time.sleep(4)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1 + 1
