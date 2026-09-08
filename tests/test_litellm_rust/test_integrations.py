import json
from typing import Final

import pytest
from opentelemetry.trace import StatusCode
from prometheus_client import CollectorRegistry, Counter

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.litellm_core_utils import litellm_logging
from litellm.types.utils import CallTypes
from tests._prometheus_helpers import isolated_prometheus_registry
from tests.test_litellm_rust.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.integrations import (
    ALL_ROUTES,
    ASYNC_ROUTES,
    MESSAGES_ROUTE,
    NON_STREAM_ASYNC_ROUTES,
    OCR_ASYNC,
    OCR_SYNC,
    OtelHarness,
    RecordingGuardrail,
    ReviewGuardrail,
    Route,
    metric_value,
    route_id,
)
from tests.test_litellm_rust.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension

FAILURE_RESPONSE: Final = ResponseSpec(body={"message": "provider unavailable"}, status=500)


def test_prometheus_registry_restores_collectors_after_failure() -> None:
    registry: Final = CollectorRegistry()
    original: Final = Counter("original", "Original collector", registry=registry)
    original.inc(2)

    def failing_test() -> None:
        with isolated_prometheus_registry(registry):
            assert registry.get_sample_value("original_total") is None
            Counter("original", "Temporary replacement", registry=registry).inc(7)
            Counter("temporary", "Temporary collector", registry=registry).inc()
            raise RuntimeError("test failed")

    with pytest.raises(RuntimeError, match="test failed"):
        failing_test()

    assert registry.get_sample_value("original_total") == 2
    assert registry.get_sample_value("temporary_total") is None
    registry.unregister(original)
    assert registry.get_sample_value("original_total") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ALL_ROUTES, ids=route_id)
async def test_otel_emits_one_request_span_on_success(
    route: Route, provider: RecordingServer, otel: OtelHarness
) -> None:
    await route.invoke(provider, callbacks=[otel.logger])

    spans: Final = await otel.wait_for_spans()
    assert len(spans) == 1
    span: Final = spans[0]
    assert span.status.status_code is StatusCode.OK
    assert span.attributes["llm.request.type"] == route.call_type
    assert span.attributes["gen_ai.request.model"] == route.provider_model
    assert json.loads(span.attributes["hidden_params"])["response_cost"] == pytest.approx(route.expected_cost)


@pytest.mark.asyncio
async def test_otel_stream_span_appears_only_after_exhaustion(
    otel: OtelHarness, recording_server: RecordingServer
) -> None:
    recording_server.default_response = ResponseSpec(body=MESSAGES_ROUTE.provider_response)

    stream: Final = await MESSAGES_ROUTE.invoke(recording_server, callbacks=[otel.logger], stream=True)
    await drain_logging()
    assert otel.spans() == ()

    chunks: Final = [chunk async for chunk in stream]

    assert chunks
    assert len(await otel.wait_for_spans()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("route", (OCR_SYNC, OCR_ASYNC), ids=route_id)
async def test_otel_emits_one_error_span_on_provider_failure(
    route: Route, provider: RecordingServer, otel: OtelHarness
) -> None:
    provider.enqueue(FAILURE_RESPONSE)

    with pytest.raises(litellm.InternalServerError):
        await route.invoke(provider, callbacks=[otel.logger])

    spans: Final = await otel.wait_for_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code is StatusCode.ERROR
    exception_events: Final = [event for event in spans[0].events if event.name == "exception"]
    assert len(exception_events) == 1
    assert "InternalServerError" in exception_events[0].attributes["exception.type"]
    assert spans[0].attributes["llm.request.type"] == route.call_type


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ALL_ROUTES, ids=route_id)
async def test_otel_and_custom_logger_observe_same_standard_logging_object(
    route: Route, provider: RecordingServer, otel: OtelHarness
) -> None:
    recorder: Final = RecordingLogger()

    await route.invoke(provider, callbacks=[otel.logger, recorder])

    events: Final = (
        await recorder.wait_for_async("async_log_success_event")
        if route.fires_async_hooks
        else recorder.wait_for("log_success_event")
    )
    payload: Final = events[0].kwargs["standard_logging_object"]
    span: Final = (await otel.wait_for_spans())[0]
    assert payload["call_type"] == route.call_type
    assert payload["model"] == route.provider_model
    assert json.loads(span.attributes["hidden_params"]) == payload["hidden_params"]
    assert span.attributes["llm.request.type"] == payload["call_type"]
    assert span.attributes["litellm.provider.model"] == payload["model"]


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ASYNC_ROUTES, ids=route_id)
async def test_prometheus_counts_one_successful_request(
    route: Route, provider: RecordingServer, prometheus: PrometheusLogger
) -> None:
    before: Final = metric_value("litellm_requests_metric_total", model=route.provider_model)
    recorder: Final = RecordingLogger()

    await route.invoke(provider, callbacks=[prometheus, recorder])
    await recorder.wait_for_async("async_log_success_event")

    assert metric_value("litellm_requests_metric_total", model=route.provider_model) == before + 1
    assert metric_value("litellm_llm_api_failed_requests_metric_total", model=route.provider_model) == 0


@pytest.mark.asyncio
async def test_prometheus_counts_tokens_from_messages_usage(
    recording_server: RecordingServer, prometheus: PrometheusLogger
) -> None:
    recording_server.default_response = ResponseSpec(body=MESSAGES_ROUTE.provider_response)
    recorder: Final = RecordingLogger()

    await MESSAGES_ROUTE.invoke(recording_server, callbacks=[prometheus, recorder])
    await recorder.wait_for_async("async_log_success_event")

    assert metric_value("litellm_input_tokens_metric_total", model=MESSAGES_ROUTE.provider_model) == 5
    assert metric_value("litellm_output_tokens_metric_total", model=MESSAGES_ROUTE.provider_model) == 4


@pytest.mark.asyncio
async def test_prometheus_counts_one_failed_request(
    otel: OtelHarness, prometheus: PrometheusLogger, recording_server: RecordingServer
) -> None:
    recording_server.enqueue(FAILURE_RESPONSE)

    with pytest.raises(litellm.InternalServerError):
        await OCR_ASYNC.invoke(recording_server, callbacks=[prometheus, otel.logger])

    assert len(await otel.wait_for_spans()) == 1
    assert metric_value("litellm_llm_api_failed_requests_metric_total", model=OCR_ASYNC.provider_model) == 1
    assert metric_value("litellm_requests_metric_total", model=OCR_ASYNC.provider_model) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("route", NON_STREAM_ASYNC_ROUTES, ids=route_id)
async def test_prometheus_by_string_name_is_initialized_once(route: Route, provider: RecordingServer) -> None:
    provider.expected_requests = 2
    litellm.success_callback = ["prometheus"]  # test-quality-ok: tests public string-name registration; isolate_rust_state restores this registry
    recorder: Final = RecordingLogger()

    await route.invoke(provider, callbacks=[recorder])
    await route.invoke(provider, callbacks=[recorder])
    await recorder.wait_for_async("async_log_success_event", count=2)

    instances: Final = [cb for cb in litellm_logging._in_memory_loggers if isinstance(cb, PrometheusLogger)]  # pyright: ignore[reportPrivateUsage]  # string-name cache has no public accessor
    assert len(instances) == 1
    assert metric_value("litellm_requests_metric_total", model=route.provider_model) == 2
    assert "prometheus" not in litellm.success_callback
    assert instances[0] in litellm._async_success_callback  # pyright: ignore[reportPrivateUsage]  # callback registry has no public accessor


@pytest.mark.asyncio
async def test_sync_ocr_reaches_sync_hooks_only(
    recording_server: RecordingServer, otel: OtelHarness, prometheus: PrometheusLogger
) -> None:
    recording_server.default_response = ResponseSpec(body=OCR_SYNC.provider_response)

    await OCR_SYNC.invoke(recording_server, callbacks=[otel.logger, prometheus])

    assert len(await otel.wait_for_spans()) == 1
    assert metric_value("litellm_requests_metric_total", model=OCR_SYNC.provider_model) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("route", NON_STREAM_ASYNC_ROUTES, ids=route_id)
async def test_logging_only_guardrail_verdict_reaches_otel_and_custom_logger(
    route: Route, provider: RecordingServer, otel: OtelHarness
) -> None:
    guardrail: Final = RecordingGuardrail()
    recorder: Final = RecordingLogger()

    await route.invoke(provider, callbacks=[guardrail, otel.logger, recorder])

    payload: Final = (await recorder.wait_for_async("async_log_success_event"))[0].kwargs["standard_logging_object"]
    assert guardrail.observations == [route.logging_only_scan]
    verdicts: Final = payload["guardrail_information"]
    assert [verdict["guardrail_name"] for verdict in verdicts] == ["rust-review"]
    assert verdicts[0]["guardrail_status"] == "success"
    guardrail_spans: Final = await otel.wait_for_spans("guardrail")
    assert len(guardrail_spans) == 1
    assert guardrail_spans[0].attributes["guardrail_name"] == "rust-review"
    assert guardrail_spans[0].attributes["guardrail_status"] == "success"


@pytest.mark.asyncio
@pytest.mark.parametrize("route", NON_STREAM_ASYNC_ROUTES, ids=route_id)
async def test_logging_only_guardrail_failure_does_not_block_loggers(
    route: Route, provider: RecordingServer, otel: OtelHarness, prometheus: PrometheusLogger
) -> None:
    guardrail: Final = RecordingGuardrail(fail_with=RuntimeError("review service unavailable"))
    recorder: Final = RecordingLogger()

    response: Final = await route.invoke(provider, callbacks=[guardrail, otel.logger, prometheus, recorder])

    assert response is not None
    assert len(await otel.wait_for_spans()) == 1
    assert metric_value("litellm_requests_metric_total", model=route.provider_model) == 1
    payload: Final = (await recorder.wait_for_async("async_log_success_event"))[0].kwargs["standard_logging_object"]
    assert payload["guardrail_information"][0]["guardrail_status"] == "guardrail_failed_to_respond"


@pytest.mark.asyncio
@pytest.mark.parametrize("route", NON_STREAM_ASYNC_ROUTES, ids=route_id)
async def test_post_call_guardrail_replacement_is_what_loggers_see(
    route: Route, provider: RecordingServer, otel: OtelHarness
) -> None:
    async def review(response: object) -> object:
        match route.name:
            case "ocr-async":
                return response.model_copy(
                    update={"pages": [response.pages[0].model_copy(update={"markdown": "Reviewed OCR"})]}
                )
            case _:
                return {**response, "content": [{"type": "text", "text": "Reviewed Messages"}]}

    guardrail: Final = ReviewGuardrail(review)
    recorder: Final = RecordingLogger()
    litellm.callbacks.append(guardrail)

    response: Final = await route.invoke(provider, callbacks=[otel.logger, recorder], guardrails=["rust-review"])

    assert guardrail.call_types == [CallTypes(route.call_type)]
    event: Final = (await recorder.wait_for_async("async_log_success_event"))[0]
    span: Final = (await otel.wait_for_spans())[0]
    match route.name:
        case "ocr-async":
            assert response.pages[0].markdown == "Reviewed OCR"
            assert event.response.pages[0].markdown == "Reviewed OCR"
        case _:
            assert response["content"][0]["text"] == "Reviewed Messages"
            assert event.response.choices[0].message.content == "Reviewed Messages"
            assert "Reviewed Messages" in json.dumps(dict(span.attributes))
    assert "guardrails" not in provider.requests[0].body
