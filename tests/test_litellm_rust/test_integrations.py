import json
from typing import Final

import pytest
from fastapi import HTTPException
from opentelemetry.trace import StatusCode
from prometheus_client import CollectorRegistry, Counter

import litellm
from litellm.integrations.generic_api.generic_api_callback import GenericAPILogger
from litellm.integrations.prometheus import PrometheusLogger
from litellm.litellm_core_utils import litellm_logging
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.generic_guardrail_api import GenericGuardrailAPI
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import ContentFilterGuardrail
from litellm.types.guardrails import BlockedWord, ContentFilterAction, GuardrailEventHooks
from litellm.types.utils import CallTypes
from tests._prometheus_helpers import isolated_prometheus_registry
from tests.test_litellm_rust.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.integrations import (
    ALL_ROUTES,
    ASYNC_ROUTES,
    AZURE_MODERATION_ALLOW_RESPONSE,
    AZURE_MODERATION_BLOCK_RESPONSE,
    MESSAGES_ROUTE,
    NON_STREAM_ASYNC_ROUTES,
    OCR_ASYNC,
    OCR_SYNC,
    OtelHarness,
    RecordingGuardrail,
    ReviewGuardrail,
    Route,
    azure_text_moderation,
    metric_value,
    route_id,
)
from tests.test_litellm_rust.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension

FAILURE_RESPONSE: Final = ResponseSpec(body={"message": "provider unavailable"}, status=500)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ASYNC_ROUTES, ids=route_id)
async def test_generic_api_logger_exports_success_over_http(route: Route, provider: RecordingServer) -> None:
    provider.expected_requests = 2
    logger: Final = GenericAPILogger(endpoint=f"{provider.base_url}/logs", batch_size=1, log_format="single")
    recorder: Final = RecordingLogger()

    await route.invoke(provider, callbacks=[logger, recorder])
    await recorder.wait_for_async("async_log_success_event")

    exports: Final = [request for request in provider.requests if request.path == "/logs"]
    assert len(exports) == 1
    payload: Final = exports[0].body
    assert payload["status"] == "success"
    assert payload["call_type"] == route.call_type
    assert payload["model"] == route.provider_model
    assert payload["response_cost"] == pytest.approx(route.expected_cost)
    assert route.response_text in json.dumps(payload["response"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route",
    (
        OCR_ASYNC,
        pytest.param(
            MESSAGES_ROUTE,
            marks=pytest.mark.xfail(
                reason="Rust Messages retries HTTP 500 despite num_retries=0",
                strict=True,
            ),
        ),
    ),
    ids=route_id,
)
async def test_generic_api_logger_exports_provider_failure_over_http(route: Route, provider: RecordingServer) -> None:
    provider.expected_requests = None
    provider.enqueue(FAILURE_RESPONSE)
    logger: Final = GenericAPILogger(endpoint=f"{provider.base_url}/logs", batch_size=1, log_format="single")
    recorder: Final = RecordingLogger()

    try:
        with pytest.raises(litellm.InternalServerError):
            await route.invoke(provider, callbacks=[logger, recorder], num_retries=0)
    finally:
        await drain_logging()
    await recorder.wait_for_async("async_log_failure_event")

    exports: Final = [request for request in provider.requests if request.path == "/logs"]
    assert len(provider.requests) == 2
    assert len(exports) == 1
    payload: Final = exports[0].body
    assert payload["status"] == "failure"
    assert payload["call_type"] == route.call_type
    assert payload["error_information"]["error_class"] == "InternalServerError"
    assert payload["error_information"]["error_code"] == "500"


@pytest.mark.asyncio
@pytest.mark.parametrize("route", NON_STREAM_ASYNC_ROUTES, ids=route_id)
@pytest.mark.parametrize(
    ("verdict", "expected_status"),
    (
        (ResponseSpec(body={"action": "NONE"}), "success"),
        (ResponseSpec(body={"action": "BLOCKED", "blocked_reason": "policy violation"}), "guardrail_intervened"),
        (ResponseSpec(body={"action": "GUARDRAIL_INTERVENED", "texts": ["redacted"]}), "success"),
        (FAILURE_RESPONSE, "guardrail_failed_to_respond"),
    ),
    ids=("allow", "block", "rewrite", "unavailable"),
)
async def test_generic_guardrail_logging_only_verdict_is_exported_over_http(
    route: Route, provider: RecordingServer, verdict: ResponseSpec, expected_status: str
) -> None:
    provider.expected_requests = 3
    provider.enqueue(ResponseSpec(body=route.provider_response))
    provider.enqueue(verdict)
    provider.enqueue(ResponseSpec(body={}))
    guardrail: Final = GenericGuardrailAPI(
        api_base=provider.base_url,
        guardrail_name="http-review",
        event_hook=GuardrailEventHooks.logging_only,
        default_on=True,
    )
    logger: Final = GenericAPILogger(endpoint=f"{provider.base_url}/logs", batch_size=1, log_format="single")
    recorder: Final = RecordingLogger()

    response: Final = await route.invoke(provider, callbacks=[guardrail, logger, recorder])
    await recorder.wait_for_async("async_log_success_event")

    scan: Final = provider.requests[1]
    assert scan.path == "/beta/litellm_basic_guardrail_api"
    assert scan.body["input_type"] == route.logging_only_scan[0]
    assert scan.body["texts"] == list(route.logging_only_scan[1])
    assert provider.requests[2].path == "/logs"
    payload: Final = provider.requests[2].body
    assert payload["status"] == "success"
    assert [(entry["guardrail_name"], entry["guardrail_status"]) for entry in payload["guardrail_information"]] == [
        ("http-review", expected_status)
    ]
    assert route.response_text in json.dumps(payload["response"])
    if route == OCR_ASYNC:
        assert response.pages[0].markdown == route.response_text
    else:
        assert response["content"][0]["text"] == route.response_text
    assert "guardrails" not in provider.requests[0].body


@pytest.mark.asyncio
@pytest.mark.parametrize("route", NON_STREAM_ASYNC_ROUTES, ids=route_id)
@pytest.mark.parametrize("action", (ContentFilterAction.BLOCK, ContentFilterAction.MASK))
async def test_content_filter_logging_only_detects_real_content_without_changing_response(
    route: Route, provider: RecordingServer, action: ContentFilterAction
) -> None:
    guardrail: Final = ContentFilterGuardrail(
        guardrail_name="content-review",
        event_hook=GuardrailEventHooks.logging_only,
        default_on=True,
        blocked_words=[BlockedWord(keyword=route.logging_only_scan[1][0], action=action)],
    )
    recorder: Final = RecordingLogger()

    response: Final = await route.invoke(provider, callbacks=[guardrail, recorder])
    payload: Final = (await recorder.wait_for_async("async_log_success_event"))[0].kwargs["standard_logging_object"]

    assert len(payload["guardrail_information"]) == 1
    verdict: Final = payload["guardrail_information"][0]
    assert verdict["guardrail_name"] == "content-review"
    assert verdict["guardrail_status"] == ("guardrail_intervened" if action == ContentFilterAction.BLOCK else "success")
    assert verdict["guardrail_response"] == [
        {"action": action.value, "keyword": route.logging_only_scan[1][0].lower(), "type": "blocked_word"}
    ]
    assert route.response_text in json.dumps(payload["response"])
    if route == OCR_ASYNC:
        assert response.pages[0].markdown == route.response_text
    else:
        assert response["content"][0]["text"] == route.response_text


@pytest.mark.asyncio
@pytest.mark.parametrize("route", NON_STREAM_ASYNC_ROUTES, ids=route_id)
@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="CustomGuardrail deployment post-call hook does not dispatch to production apply_guardrail implementations",
)
async def test_content_filter_post_call_blocks_provider_response(route: Route, provider: RecordingServer) -> None:
    guardrail: Final = ContentFilterGuardrail(
        guardrail_name="enforced-content-review",
        event_hook=GuardrailEventHooks.post_call,
        blocked_words=[BlockedWord(keyword=route.response_text, action=ContentFilterAction.BLOCK)],
    )
    litellm.callbacks.append(guardrail)

    with pytest.raises(HTTPException, match="Content blocked") as blocked:
        await route.invoke(provider, guardrails=["enforced-content-review"])
    assert blocked.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Azure text moderation only scans ModelResponse and skips native Messages responses",
)
async def test_azure_text_moderation_allows_messages_response_over_http(
    recording_server: RecordingServer, otel: OtelHarness
) -> None:
    recording_server.expected_requests = None
    recording_server.default_response = ResponseSpec(body=AZURE_MODERATION_ALLOW_RESPONSE)
    recording_server.enqueue(ResponseSpec(body=MESSAGES_ROUTE.provider_response))
    guardrail: Final = azure_text_moderation(recording_server)
    recorder: Final = RecordingLogger()
    litellm.callbacks.append(guardrail)

    response: Final = await MESSAGES_ROUTE.invoke(
        recording_server,
        callbacks=[otel.logger, recorder],
        guardrails=[guardrail.guardrail_name],
    )
    await recorder.wait_for_async("async_log_success_event")

    assert len(recording_server.requests) == 2
    moderation_request: Final = recording_server.requests[1]
    assert moderation_request.path == "/contentsafety/text:analyze?api-version=2024-09-01"
    assert moderation_request.headers["ocp-apim-subscription-key"] == "test-azure-key"
    assert moderation_request.body == {
        "text": MESSAGES_ROUTE.response_text,
        "categories": ["Hate", "Sexual", "SelfHarm", "Violence"],
        "blocklistNames": None,
        "haltOnBlocklistHit": False,
        "outputType": "FourSeverityLevels",
    }
    assert response["content"][0]["text"] == MESSAGES_ROUTE.response_text
    assert len(await otel.wait_for_spans()) == 1


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="Azure text moderation only scans ModelResponse and skips native Messages responses",
)
async def test_azure_text_moderation_blocks_messages_response_over_http(recording_server: RecordingServer) -> None:
    recording_server.expected_requests = None
    recording_server.default_response = ResponseSpec(body=AZURE_MODERATION_BLOCK_RESPONSE)
    recording_server.enqueue(ResponseSpec(body=MESSAGES_ROUTE.provider_response))
    guardrail: Final = azure_text_moderation(recording_server)
    litellm.callbacks.append(guardrail)

    with pytest.raises(HTTPException, match="Violence crossed severity 2") as blocked:
        await MESSAGES_ROUTE.invoke(recording_server, guardrails=[guardrail.guardrail_name])

    assert blocked.value.status_code == 400
    assert recording_server.requests[1].body["text"] == MESSAGES_ROUTE.response_text


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
    litellm.success_callback = ["prometheus"]  # test-quality-ok: public registration; fixture restores globals
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
