from collections.abc import Generator
from functools import partial
from typing import Final

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import litellm
from litellm.integrations.otel.logger import OpenTelemetryV2
from litellm.integrations.otel.model.config import OpenTelemetryV2Config
from litellm.integrations.otel.plumbing.providers import build_tracer_provider
from litellm.rust_bridge.catalog import Route
from tests.test_litellm_rust.messages.data import MESSAGES_RESPONSE, call
from tests.test_litellm_rust.support.caching import Call
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.routing import native_route


@pytest.fixture
def server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.serve(ResponseSpec(body=MESSAGES_RESPONSE))
    return recording_server


@pytest.fixture(autouse=True)
def native() -> Generator[None]:
    with native_route(Route.MESSAGES):
        yield


@pytest.fixture
def native_call(server: RecordingServer) -> Call:
    return partial(call, server)


@pytest.fixture
def otel() -> Generator[tuple[OpenTelemetryV2, InMemorySpanExporter]]:
    from litellm.proxy import proxy_server

    config: Final = OpenTelemetryV2Config(exporter="in_memory", enable_metrics=False, enable_events=False)
    exporter: Final = InMemorySpanExporter()
    provider: Final = build_tracer_provider(config, exporter=exporter, use_simple_processor=True)
    with (
        rebound(proxy_server, "open_telemetry_logger", None),
        rebound(litellm, "service_callback", []),
    ):
        try:
            yield OpenTelemetryV2(config=config, tracer_provider=provider), exporter
        finally:
            provider.shutdown()
