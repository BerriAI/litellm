from contextlib import ExitStack
from typing import Final

import pytest
import pytest_asyncio
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from litellm.integrations.generic_api.generic_api_callback import GenericAPILogger
from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig
from litellm.integrations.prometheus import PrometheusLogger
from tests.test_litellm_rust.conftest import isolate_rust_state
from tests.test_litellm_rust.integrations import GenericAPIExportHarness, OtelHarness, Route, provider_response
from tests.test_litellm_rust.support.recording_server import RecordingServer, recording_service


@pytest.fixture
def provider(recording_server: RecordingServer, route: Route) -> RecordingServer:
    recording_server.default_response = provider_response(route)
    return recording_server


@pytest.fixture
def otel(isolate_rust_state: ExitStack) -> OtelHarness:
    exporter: Final = InMemorySpanExporter()
    tracer_provider: Final = TracerProvider()
    isolate_rust_state.callback(tracer_provider.shutdown)
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    logger: Final = OpenTelemetry(config=OpenTelemetryConfig(exporter=exporter), tracer_provider=tracer_provider)
    return OtelHarness(logger=logger, exporter=exporter)


@pytest.fixture
def prometheus() -> PrometheusLogger:
    return PrometheusLogger()


@pytest_asyncio.fixture
async def generic_api_export(isolate_rust_state: ExitStack) -> GenericAPIExportHarness:
    server: Final = isolate_rust_state.enter_context(recording_service())
    server.expected_requests = None
    logger: Final = GenericAPILogger(endpoint=f"{server.base_url}/logs", batch_size=1, log_format="single")
    return GenericAPIExportHarness(server=server, logger=logger)
