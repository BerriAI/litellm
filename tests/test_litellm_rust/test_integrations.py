import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import pytest
from opentelemetry.trace import StatusCode

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.generic_api.generic_api_callback import GenericAPILogger
from litellm.integrations.prometheus import PrometheusLogger
from litellm.litellm_core_utils.custom_logger_registry import CustomLoggerRegistry
from litellm.proxy.guardrails.guardrail_registry import guardrail_initializer_registry
from litellm.rust_bridge.provenance import has_rust_response_marker
from litellm.types.guardrails import SupportedGuardrailIntegrations
from litellm.types.utils import CallTypes
from tests.test_litellm_rust.callback_recorder import (
    LiveReferenceLogger,
    RecordingLogger,
    SecondaryLiveReferenceLogger,
    drain_logging,
)
from tests.test_litellm_rust.conftest import Backend, isolated_backend
from tests.test_litellm_rust.contracts import MESSAGES_EVENTS
from tests.test_litellm_rust.integrations import (
    ASYNC_ROUTES,
    DISCOVERED_ONLY_GUARDRAIL_NAMES,
    ENTERPRISE_LOGGER_NAMES,
    GUARDRAIL_NAMES,
    GUARDRAIL_OBLIGATIONS,
    LOGGER_OBLIGATIONS,
    MESSAGES_ROUTE,
    MESSAGES_STREAM,
    OCR_ASYNC,
    OCR_SYNC,
    OSS_LOGGER_NAMES,
    AsyncBoundaryLogger,
    MutatingFailingLogger,
    OtelHarness,
    Route,
    RunObservation,
    gcs_literalai_harness,
    metric_value,
    provider_response,
    route_id,
    wait_for_callback,
)
from tests.test_litellm_rust.recording_server import RecordingServer, ResponseSpec, recording_service

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


async def observe_backend(route: Route, backend: Backend) -> RunObservation:
    async with isolated_backend(backend):
        with recording_service() as provider:
            provider.default_response = provider_response(route)
            recorder: Final = RecordingLogger()
            response: Final = await route.invoke(provider, callbacks=[recorder])
            event: Final = (await wait_for_callback(route, recorder))[0]
            payload: Final = event.kwargs["standard_logging_object"]
            provider_body: Final = provider.requests[0].body
            if not isinstance(provider_body, dict):
                raise TypeError(f"Expected provider object body, got {type(provider_body).__name__}")
            return RunObservation(
                call_type=payload["call_type"],
                model=payload["model"],
                response_cost=payload["response_cost"],
                response_text=route.response_text(response),
                provider_body=provider_body,
                rust_dispatch=has_rust_response_marker(response),
            )


def test_logger_catalogue_reconciles_with_production_registry() -> None:
    actual_names: Final = frozenset(CustomLoggerRegistry.CALLBACK_CLASS_STR_TO_CLASS_TYPE)
    catalogued_names: Final = frozenset(
        name for obligation in LOGGER_OBLIGATIONS.values() for name in obligation.registration_names
    )
    assert OSS_LOGGER_NAMES <= actual_names
    assert actual_names <= OSS_LOGGER_NAMES | ENTERPRISE_LOGGER_NAMES
    assert catalogued_names == OSS_LOGGER_NAMES | ENTERPRISE_LOGGER_NAMES


def test_guardrail_catalogue_reconciles_enum_and_runtime_discovery() -> None:
    enum_names: Final = frozenset(integration.value for integration in SupportedGuardrailIntegrations)
    initializer_names: Final = frozenset(guardrail_initializer_registry)
    assert enum_names == GUARDRAIL_NAMES
    assert initializer_names == GUARDRAIL_NAMES | DISCOVERED_ONLY_GUARDRAIL_NAMES
    assert frozenset(GUARDRAIL_OBLIGATIONS) == initializer_names


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ASYNC_ROUTES, ids=lambda route: f"otel-prometheus-generic-api-{route.name}-success")
async def test_export_composition(
    route: Route, recording_server: RecordingServer, otel: OtelHarness, prometheus: PrometheusLogger
) -> None:
    recording_server.default_response = provider_response(route)
    before: Final = metric_value("litellm_requests_metric_total", model=route.provider_model)
    recorder: Final = RecordingLogger()

    with recording_service() as sink:
        logger: Final = GenericAPILogger(endpoint=f"{sink.base_url}/logs", batch_size=1, log_format="single")
        await route.invoke(recording_server, callbacks=[otel.logger, prometheus, logger, recorder])
        await wait_for_callback(route, recorder)

        exports: Final = tuple(request for request in sink.requests if request.path == "/logs")
        spans: Final = await otel.wait_for_spans()
        assert len(exports) == 1
        assert exports[0].body["call_type"] == route.call_type
        assert exports[0].body["response_cost"] == pytest.approx(route.expected_cost)
        assert len(spans) == 1
        assert spans[0].status.status_code is StatusCode.OK
        assert metric_value("litellm_requests_metric_total", model=route.provider_model) == before + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("route", (OCR_SYNC, OCR_ASYNC, MESSAGES_ROUTE), ids=route_id)
async def test_public_sdk_python_rust_composition_parity(route: Route) -> None:
    python: Final = await observe_backend(route, "python")
    rust: Final = await observe_backend(route, "rust")
    assert python.call_type == rust.call_type == route.call_type
    assert python.model == rust.model == route.provider_model
    assert python.response_cost == pytest.approx(rust.response_cost)
    assert python.response_text == rust.response_text == route.expected_text
    python_body: Final = {**python.provider_body, "stream": python.provider_body.get("stream", False)}
    rust_body: Final = {**rust.provider_body, "stream": rust.provider_body.get("stream", False)}
    assert python_body == rust_body
    assert python.rust_dispatch is False
    assert rust.rust_dispatch is True


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("python", "rust"))
async def test_terminal_callbacks_share_live_objects_within_one_run(backend: Backend) -> None:
    async with isolated_backend(backend):
        with recording_service() as provider:
            provider.default_response = provider_response(MESSAGES_ROUTE)
            first: Final = LiveReferenceLogger()
            second: Final = SecondaryLiveReferenceLogger()
            response: Final = await MESSAGES_ROUTE.invoke(provider, callbacks=[first, second])
            first_event: Final = (await first.wait_for_async())[0]
            second_event: Final = (await second.wait_for_async())[0]
            assert first_event.kwargs is second_event.kwargs
            assert first_event.response is second_event.response
            assert has_rust_response_marker(response) is (backend == "rust")
            first.release()
            second.release()


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


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("python", "rust"))
async def test_interrupted_stream_emits_terminal_only_when_consumer_closes(backend: Backend, otel: OtelHarness) -> None:
    async with isolated_backend(backend):
        with recording_service() as provider:
            provider.default_response = ResponseSpec(body=None, events=MESSAGES_EVENTS)
            stream: Final = await MESSAGES_STREAM.open_stream(provider, callbacks=[otel.logger])
            first_chunk: Final = await anext(stream)
            await drain_logging()
            assert otel.spans() == ()
            assert has_rust_response_marker(stream) is (backend == "rust")
            await stream.aclose()
            await drain_logging()
            assert first_chunk is not None
            assert len(otel.spans()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("python", "rust"))
@pytest.mark.parametrize(
    ("route", "phase"),
    (
        (OCR_ASYNC, "pre"),
        (MESSAGES_ROUTE, "pre"),
        (MESSAGES_STREAM, "pre"),
        (OCR_ASYNC, "success"),
        (MESSAGES_ROUTE, "success"),
    ),
    ids=("ocr-pre", "messages-pre", "messages-stream-pre", "ocr-success", "messages-success"),
)
async def test_deployment_rejection_notifies_failure_before_terminal(
    backend: Backend, route: Route, phase: str
) -> None:
    class Reject(CustomLogger):
        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: CallTypes | None
        ) -> dict[str, object]:
            if phase == "pre":
                raise litellm.BadRequestError("deployment rejected", "test-provider", route.provider_model)
            return kwargs

        async def async_post_call_success_deployment_hook(
            self, request_data: Mapping[str, object], response: object, call_type: CallTypes | None
        ) -> object:
            raise litellm.BadRequestError("deployment rejected", "test-provider", route.provider_model)

    async with isolated_backend(backend):
        with recording_service() as provider:
            provider.expected_requests = 0 if phase == "pre" else 1
            provider.default_response = provider_response(route)
            recorder: Final = RecordingLogger()
            litellm.callbacks.extend((Reject(), recorder))

            with pytest.raises(litellm.BadRequestError, match="deployment rejected"):
                await route.invoke(provider)
            await recorder.wait_for_async("async_log_failure_event")

            assert len(provider.requests) == provider.expected_requests
            assert "async_log_success_event" not in recorder.names
            assert recorder.names.count("async_post_call_failure_deployment_hook") == 1
            assert recorder.names.index("async_post_call_failure_deployment_hook") < recorder.names.index(
                "async_log_failure_event"
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("python", "rust"))
@pytest.mark.parametrize("ending", ("provider_error", "truncated", "close"))
async def test_established_stream_failure_notifies_deployment_once(backend: Backend, ending: str) -> None:
    async with isolated_backend(backend):
        with recording_service() as provider:
            provider.default_response = ResponseSpec(
                body=None,
                events=(
                    (
                        *MESSAGES_EVENTS[:1],
                        (
                            "error",
                            {"type": "error", "error": {"type": "overloaded_error", "message": "upstream overloaded"}},
                        ),
                    )
                    if ending == "provider_error"
                    else MESSAGES_EVENTS[:-1]
                    if ending == "truncated"
                    else MESSAGES_EVENTS
                ),
            )
            release: Final = threading.Event()
            if ending == "close":
                provider.enqueue(
                    ResponseSpec(
                        body=None,
                        chunks=tuple(
                            f"event: {event}\ndata: {json.dumps(data)}\n\n".encode() for event, data in MESSAGES_EVENTS
                        ),
                        release=release,
                    )
                )
            recorder: Final = RecordingLogger()
            litellm.callbacks.append(recorder)
            try:
                stream: Final = await MESSAGES_STREAM.open_stream(provider)
                assert has_rust_response_marker(stream) is (backend == "rust")
                assert recorder.names.count("async_pre_call_deployment_hook") == 1
                assert "async_post_call_failure_deployment_hook" not in recorder.names

                if ending == "close":
                    assert await anext(stream)
                    await stream.aclose()
                else:
                    try:
                        async for _ in stream:
                            pass
                    except litellm.APIError:
                        pass
                await drain_logging()
            finally:
                release.set()

            assert len(provider.requests) == 1
            assert recorder.names.count("async_log_failure_event") == 1, recorder.names
            assert "async_log_success_event" not in recorder.names
            assert recorder.names.count("async_post_call_failure_deployment_hook") == 1
            assert recorder.names.index("async_post_call_failure_deployment_hook") < recorder.names.index(
                "async_log_failure_event"
            )
