import json
from typing import Final

import pytest
from opentelemetry.trace import StatusCode
from prometheus_client import CollectorRegistry, Counter

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.litellm_core_utils import litellm_logging
from tests._prometheus_helpers import isolated_prometheus_registry
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.requests import MESSAGES_EVENTS
from tests.test_litellm_rust.integrations import (
    ALL_ROUTES,
    ASYNC_ROUTES,
    MESSAGES_ROUTE,
    MESSAGES_STREAM,
    NON_STREAM_ASYNC_ROUTES,
    OCR_ASYNC,
    OCR_SYNC,
    GenericAPIExportHarness,
    OtelHarness,
    Route,
    metric_value,
    route_id,
    wait_for_callback,
)
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension

FAILURE_RESPONSE: Final = ResponseSpec(body={"message": "provider unavailable"}, status=500)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ASYNC_ROUTES, ids=route_id)
async def test_generic_api_logger_exports_success_over_http(
    route: Route,
    provider: RecordingServer,
    generic_api_export: GenericAPIExportHarness,
) -> None:
    recorder: Final = RecordingLogger()
    await route.invoke(provider, callbacks=[generic_api_export.logger, recorder])
    await wait_for_callback(route, recorder)

    assert len(generic_api_export.exports) == 1
    payload: Final = generic_api_export.exports[0].body
    assert payload["status"] == "success"
    assert payload["call_type"] == route.call_type
    assert payload["model"] == route.provider_model
    assert payload["response_cost"] == pytest.approx(route.expected_cost)
    assert route.expected_text in json.dumps(payload["response"])


@pytest.mark.asyncio
@pytest.mark.parametrize("route", NON_STREAM_ASYNC_ROUTES, ids=route_id)
async def test_generic_api_logger_exports_provider_failure_over_http(
    route: Route,
    provider: RecordingServer,
    generic_api_export: GenericAPIExportHarness,
) -> None:
    provider.enqueue(FAILURE_RESPONSE)
    recorder: Final = RecordingLogger()

    try:
        with pytest.raises(litellm.InternalServerError):
            await route.invoke(provider, callbacks=[generic_api_export.logger, recorder], num_retries=0)
    finally:
        await drain_logging()
    await wait_for_callback(route, recorder, outcome="failure")

    assert len(provider.requests) == 1
    assert len(generic_api_export.exports) == 1
    payload: Final = generic_api_export.exports[0].body
    assert payload["status"] == "failure"
    assert payload["call_type"] == route.call_type
    assert payload["error_information"]["error_class"] == "InternalServerError"
    assert payload["error_information"]["error_code"] == "500"


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
    recording_server.default_response = ResponseSpec(body=None, events=MESSAGES_EVENTS)
    stream: Final = await MESSAGES_STREAM.open_stream(recording_server, callbacks=[otel.logger])
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
    payload: Final = (await wait_for_callback(route, recorder))[0].kwargs["standard_logging_object"]
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
    await wait_for_callback(route, recorder)
    assert metric_value("litellm_requests_metric_total", model=route.provider_model) == before + 1
    assert metric_value("litellm_llm_api_failed_requests_metric_total", model=route.provider_model) == 0


@pytest.mark.asyncio
async def test_prometheus_counts_tokens_from_messages_usage(
    recording_server: RecordingServer, prometheus: PrometheusLogger
) -> None:
    recording_server.default_response = ResponseSpec(body=MESSAGES_ROUTE.provider_response)
    recorder: Final = RecordingLogger()
    await MESSAGES_ROUTE.invoke(recording_server, callbacks=[prometheus, recorder])
    await wait_for_callback(MESSAGES_ROUTE, recorder)
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
    litellm.success_callback = ["prometheus"]  # test-quality-ok: public registration; fixture restores globals
    recorder: Final = RecordingLogger()
    await route.invoke(provider, callbacks=[recorder])
    await route.invoke(provider, callbacks=[recorder])
    await wait_for_callback(route, recorder, count=2)
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
