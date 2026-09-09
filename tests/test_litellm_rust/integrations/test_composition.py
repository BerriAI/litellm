import json
from dataclasses import dataclass
from typing import Final

import pytest
from opentelemetry.trace import StatusCode

from litellm.integrations.generic_api.generic_api_callback import GenericAPILogger
from litellm.integrations.prometheus import PrometheusLogger
from tests.test_litellm_rust.conftest import Backend, isolated_backend
from tests.test_litellm_rust.integrations import (
    ASYNC_ROUTES,
    MESSAGES_ROUTE,
    OCR_ASYNC,
    AsyncBoundaryLogger,
    MutatingFailingLogger,
    OtelHarness,
    Route,
    gcs_literalai_harness,
    metric_value,
    provider_response,
    route_id,
    wait_for_callback,
)
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger
from tests.test_litellm_rust.support.provenance import has_rust_response_marker
from tests.test_litellm_rust.support.recording_server import RecordingServer, recording_service

pytestmark = pytest.mark.requires_rust_extension

MESSAGES_TOOLS: Final = [
    {
        "name": "get_weather",
        "description": "Get current weather",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ASYNC_ROUTES, ids=lambda route: f"otel-prometheus-generic-api-{route.name}-success")
async def test_otel_prometheus_and_generic_api_each_export_one_success(
    route: Route, recording_server: RecordingServer, otel: OtelHarness, prometheus: PrometheusLogger
) -> None:
    recording_server.default_response = provider_response(route)
    before: Final = metric_value("litellm_requests_metric_total", model=route.provider_model)
    recorder: Final = RecordingLogger()

    with recording_service() as sink:
        logger: Final = GenericAPILogger(endpoint=f"{sink.base_url}/logs", batch_size=1, log_format="single")
        response: Final = await route.invoke(recording_server, callbacks=[otel.logger, prometheus, logger, recorder])
        await wait_for_callback(route, recorder)

        exports: Final = tuple(request for request in sink.requests if request.path == "/logs")
        spans: Final = await otel.wait_for_spans()
        assert len(exports) == 1
        assert exports[0].body["call_type"] == route.call_type
        assert exports[0].body["response_cost"] == pytest.approx(route.expected_cost)
        assert len(spans) == 1
        assert spans[0].status.status_code is StatusCode.OK
        assert metric_value("litellm_requests_metric_total", model=route.provider_model) == before + 1
        assert has_rust_response_marker(response)


@pytest.mark.asyncio
async def test_failing_callback_preserves_prior_mutation_and_later_exporters() -> None:
    with recording_service() as provider, recording_service() as sink:
        provider.default_response = provider_response(OCR_ASYNC)
        logger: Final = GenericAPILogger(endpoint=f"{sink.base_url}/logs", batch_size=1, log_format="single")
        recorder: Final = RecordingLogger()
        response: Final = await OCR_ASYNC.invoke(provider, callbacks=[MutatingFailingLogger(), logger, recorder])
        events: Final = await wait_for_callback(OCR_ASYNC, recorder)
        exports: Final = tuple(request for request in sink.requests if request.path == "/logs")
        assert response.pages[0].markdown == OCR_ASYNC.expected_text
        assert events[0].kwargs["standard_logging_object"]["metadata"]["composition_marker"] == "visible-before-failure"
        assert len(exports) == 1
        assert exports[0].body["metadata"]["composition_marker"] == "visible-before-failure"
        assert has_rust_response_marker(response)


@dataclass(frozen=True, slots=True)
class SerializationSchedule:
    callback_order: tuple[str, str]
    flush_immediately: bool
    gcs_has_tools: bool


SERIALIZATION_SCHEDULES: Final = (
    pytest.param(
        SerializationSchedule(("gcs", "literalai"), True, True), id="gcs-serializes-before-literalai-mutation"
    ),
    pytest.param(
        SerializationSchedule(("gcs", "literalai"), False, False), id="gcs-serializes-after-literalai-mutation"
    ),
    pytest.param(SerializationSchedule(("literalai", "gcs"), False, False), id="literalai-mutates-before-gcs-enqueue"),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("python", "rust"))
@pytest.mark.parametrize("schedule", SERIALIZATION_SCHEDULES)
async def test_gcs_literalai_serialization_schedule(
    backend: Backend, schedule: SerializationSchedule, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "premium_user", True)
    monkeypatch.setenv("GCS_BATCH_SIZE", "1" if schedule.flush_immediately else "100")
    monkeypatch.setenv("GCS_BUCKET_NAME", "composition-bucket")
    monkeypatch.setenv("GCS_FLUSH_INTERVAL", "3600")
    monkeypatch.setenv("GCS_USE_BATCHED_LOGGING", "true")
    monkeypatch.setenv("LITERAL_BATCH_SIZE", "1")

    async with isolated_backend(backend):
        with recording_service() as provider:
            provider.default_response = provider_response(MESSAGES_ROUTE)
            async with gcs_literalai_harness() as harness:
                callbacks_by_name: Final = {"gcs": harness.gcs, "literalai": harness.literal}
                ordered_callbacks: Final = [callbacks_by_name[name] for name in schedule.callback_order]
                callbacks: Final = (
                    [harness.gcs, AsyncBoundaryLogger(harness.gcs.flush_queue), harness.literal]
                    if schedule.flush_immediately
                    else ordered_callbacks
                )
                recorder: Final = RecordingLogger()
                response: Final = await MESSAGES_ROUTE.invoke(
                    provider, tools=MESSAGES_TOOLS, callbacks=[*callbacks, recorder]
                )
                await wait_for_callback(MESSAGES_ROUTE, recorder)
                if not schedule.flush_immediately:
                    await harness.gcs.flush_queue()

                gcs_payload_value: Final = harness.storage.posts[0].body
                if not isinstance(gcs_payload_value, str):
                    raise TypeError(f"Expected serialized GCS payload, got {type(gcs_payload_value).__name__}")
                gcs_payload: Final = json.loads(gcs_payload_value)
                literal_body: Final = harness.literal_sink.posts[0].body
                if not isinstance(literal_body, dict):
                    raise TypeError(f"Expected LiteralAI request body, got {type(literal_body).__name__}")
                generation: Final = literal_body["variables"]["generation_0"]
                assert ("tools" in gcs_payload["model_parameters"]) is schedule.gcs_has_tools
                assert generation["tools"] == MESSAGES_TOOLS
                assert len(harness.storage.posts) == 1
                assert len(harness.literal_sink.posts) == 1
                assert has_rust_response_marker(response) is (backend == "rust")
