import asyncio
from datetime import datetime, timezone
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.spend_tracking.cost_finalization import finalize_response_cost
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage

NOW: Final = datetime(2026, 10, 20, 12, 30, tzinfo=timezone.utc)
MODEL: Final = "gemini-finalization-test"


class Counter:
    def __init__(self, total: float = 0.0, fail: bool = False) -> None:
        self.total = total
        self.calls = 0
        self.fail = fail

    async def async_increment(self, key: str, value: float, ttl: int | None = None) -> float:
        self.calls += 1
        if self.fail:
            raise ConnectionError("counter unavailable")
        self.total += value
        return self.total


class BlockingCounter(Counter):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def async_increment(self, key: str, value: float, ttl: int | None = None) -> float:
        self.started.set()
        await self.release.wait()
        return await super().async_increment(key, value, ttl)


@pytest.fixture
def logging_obj(monkeypatch: pytest.MonkeyPatch) -> Logging:
    monkeypatch.setattr(litellm, "apply_search_free_tier", True)
    monkeypatch.setitem(
        litellm.model_cost,
        MODEL,
        {
            "litellm_provider": "vertex_ai",
            "mode": "chat",
            "input_cost_per_token": 0.01,
            "output_cost_per_token": 0.02,
            "web_search_billing_unit": "per_query",
            "search_context_cost_per_query": {"search_context_size_medium": 0.01},
            "search_free_requests": {"count": 2, "period": "month", "shared_group": "finalization-test"},
        },
    )
    logging: Final = Logging(
        model=MODEL,
        messages=[],
        stream=False,
        call_type="acompletion",
        start_time=NOW.timestamp(),
        litellm_call_id="finalization-call",
        function_id="finalization-function",
    )
    logging.update_environment_variables(
        model=MODEL, custom_llm_provider="vertex_ai", litellm_params={}, optional_params={}
    )
    return logging


def response(cost: float = 0.5) -> ModelResponse:
    result: Final = ModelResponse(
        model=MODEL,
        usage=Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            prompt_tokens_details=PromptTokensDetailsWrapper(web_search_requests=3),
        ),
    )
    result._hidden_params["response_cost"] = cost
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize(("used", "expected"), [(0, 0.48), (1, 0.49), (2, 0.5)])
async def test_header_and_logging_share_one_allowance_allocation(
    logging_obj: Logging, used: int, expected: float
) -> None:
    counter: Final = Counter(used)
    result: Final = response()
    costs: Final = await asyncio.gather(*(finalize_response_cost(logging_obj, result, counter, NOW) for _ in range(3)))
    assert costs[0] is costs[1] is costs[2]
    assert costs[0] is not None
    assert costs[0].amount == pytest.approx(expected)
    assert counter.calls == 1
    logging_obj._success_handler_helper_fn(result=result, start_time=NOW, end_time=NOW)
    payload: Final = logging_obj.model_call_details["standard_logging_object"]
    headers: Final = ProxyBaseLLMRequestProcessing.get_custom_headers(
        user_api_key_dict=UserAPIKeyAuth(), response_cost=costs[0].amount, litellm_logging_obj=logging_obj
    )
    assert float(headers["x-litellm-response-cost"]) == pytest.approx(payload["response_cost"])
    assert payload["cost_finalized"] is True
    assert payload["cost_breakdown"] == costs[0].logging_breakdown()


@pytest.mark.asyncio
async def test_failed_counter_freezes_full_price_without_a_retry(logging_obj: Logging) -> None:
    counter: Final = Counter(fail=True)
    result: Final = response()
    finalized: Final = await finalize_response_cost(logging_obj, result, counter, NOW)
    assert finalized is not None and finalized.amount == 0.5
    assert await finalize_response_cost(logging_obj, result, counter, NOW) is finalized
    assert counter.calls == 1


@pytest.mark.asyncio
async def test_guardrail_cost_is_added_once_after_allowance(logging_obj: Logging) -> None:
    result: Final = response()
    finalized: Final = await finalize_response_cost(logging_obj, result, Counter(), NOW, guardrail_cost=0.2)
    assert finalized is not None and finalized.amount == pytest.approx(0.68)
    logging_obj._success_handler_helper_fn(result=result, start_time=NOW, end_time=NOW)
    payload: Final = logging_obj.model_call_details["standard_logging_object"]
    assert payload["response_cost"] == pytest.approx(0.68)
    assert payload["cost_breakdown"]["guardrail_cost"] == 0.2


@pytest.mark.asyncio
async def test_cache_hit_finalizes_zero_without_allocating(logging_obj: Logging) -> None:
    result: Final = response()
    result._hidden_params["cache_hit"] = True
    result._hidden_params["response_cost"] = 0.0
    counter: Final = Counter()
    finalized: Final = await finalize_response_cost(logging_obj, result, counter, NOW)
    assert finalized is not None and finalized.amount == 0.0
    assert counter.calls == 0


@pytest.mark.asyncio
async def test_both_callback_paths_wait_for_finalization(logging_obj: Logging) -> None:
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    from litellm.utils import dispatch_success_logging

    loop: Final = asyncio.get_running_loop()
    sync_cost: Final = loop.create_future()
    async_cost: Final = loop.create_future()

    class SyncObserver(CustomLogger):
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            loop.call_soon_threadsafe(sync_cost.set_result, kwargs["standard_logging_object"]["response_cost"])

    class AsyncObserver(CustomLogger):
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            async_cost.set_result(kwargs["standard_logging_object"]["response_cost"])

    logging_obj.add_dynamic_callback(SyncObserver())
    logging_obj.add_dynamic_callback(AsyncObserver())
    logging_obj.defer_async_logging = True
    inner: Final = response()
    outer: Final = response(2.0)
    for result in (inner, outer):
        dispatch_success_logging(logging_obj, result, NOW, NOW, False, False)
    assert not logging_obj._sync_success_scheduled
    assert not logging_obj._async_success_enqueued
    assert logging_obj.deferred_success_result is not None
    assert logging_obj.deferred_success_result[0] is inner
    counter: Final = BlockingCounter()
    pending: Final = asyncio.create_task(finalize_response_cost(logging_obj, outer, counter, NOW))
    await counter.started.wait()
    dispatch_success_logging(logging_obj, outer, NOW, NOW, False, False)
    assert not logging_obj._sync_success_scheduled
    assert not logging_obj._async_success_enqueued
    concurrent: Final = asyncio.create_task(finalize_response_cost(logging_obj, outer, counter, NOW))
    counter.release.set()
    finalized: Final = await pending
    assert finalized is not None
    assert await concurrent is finalized
    assert counter.calls == 1
    ProxyBaseLLMRequestProcessing._flush_deferred_async_logging(logging_obj, exception_raised=False)
    observed: Final = await asyncio.wait_for(asyncio.gather(sync_cost, async_cost), timeout=10)
    await GLOBAL_LOGGING_WORKER.flush()
    assert observed == pytest.approx([0.48, 0.48])
    assert logging_obj.enqueue_deferred_logging is None


@pytest.mark.asyncio
async def test_rejected_response_does_not_release_success_callbacks(logging_obj: Logging) -> None:
    from litellm.utils import dispatch_success_logging

    logging_obj.defer_async_logging = True
    dispatch_success_logging(logging_obj, response(), NOW, NOW, False, False)
    ProxyBaseLLMRequestProcessing._flush_deferred_async_logging(logging_obj, exception_raised=True)
    assert not logging_obj._sync_success_scheduled
    assert not logging_obj._async_success_enqueued
    assert logging_obj.enqueue_deferred_logging is None


@pytest.mark.asyncio
async def test_prebuilt_payload_is_reconciled_with_final_cost(logging_obj: Logging) -> None:
    result: Final = response()
    prepared: Final = logging_obj.prepare_success_logging(result, NOW, NOW, None, None, True, False)
    original: Final = prepared[3]
    assert original is not None and original["response_cost"] == 0.5
    finalized: Final = await finalize_response_cost(logging_obj, result, Counter(), NOW)
    assert finalized is not None
    logging_obj._success_handler_helper_fn(result, NOW, NOW, standard_logging_object=original)
    payload: Final = logging_obj.model_call_details["standard_logging_object"]
    assert payload["response_cost"] == pytest.approx(0.48)
    assert original["response_cost"] == 0.5


@pytest.mark.asyncio
async def test_cached_response_waits_for_guardrail_cost(logging_obj: Logging) -> None:
    from litellm.caching.caching_handler import LLMCachingHandler

    result: Final = response(0.0)
    result._hidden_params["cache_hit"] = True
    logging_obj.defer_async_logging = True
    handler: Final = LLMCachingHandler(original_function=response, request_kwargs={}, start_time=NOW)
    handler._async_log_cache_hit_on_callbacks(logging_obj, result, NOW, NOW, True)
    assert not logging_obj._sync_success_scheduled
    assert not logging_obj._async_success_enqueued
    counter: Final = Counter()
    finalized: Final = await finalize_response_cost(logging_obj, result, counter, NOW, guardrail_cost=0.2)
    assert finalized is not None and finalized.amount == 0.2
    assert counter.calls == 0
    logging_obj._success_handler_helper_fn(result, NOW, NOW)
    assert logging_obj.model_call_details["standard_logging_object"]["response_cost"] == 0.2
    ProxyBaseLLMRequestProcessing._flush_deferred_async_logging(logging_obj, exception_raised=True)


@pytest.mark.asyncio
async def test_collector_event_preserves_finalized_cost(logging_obj: Logging) -> None:
    from litellm.proxy.spend_tracking.spend_event import (
        SpendEvent,
        build_spend_event,
        decode_spend_event,
        spend_event_callback_args,
    )

    result: Final = response()
    await finalize_response_cost(logging_obj, result, Counter(), NOW)
    logging_obj._success_handler_helper_fn(result, NOW, NOW)
    event: Final = build_spend_event(logging_obj.model_call_details, result, NOW, NOW, store_bodies=False)
    assert isinstance(event, bytes)
    decoded: Final = decode_spend_event(event)
    assert isinstance(decoded, SpendEvent)
    restored: Final = spend_event_callback_args(decoded).kwargs["standard_logging_object"]
    assert restored is not None
    assert restored["cost_finalized"] is True
    assert restored["response_cost"] == pytest.approx(0.48)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_count", [1, 2])
async def test_cancellation_finishes_an_inflight_allocation_before_releasing_callbacks(
    logging_obj: Logging, cancel_count: int
) -> None:
    counter: Final = BlockingCounter()
    result: Final = response()
    pending: Final = asyncio.create_task(finalize_response_cost(logging_obj, result, counter, NOW))
    await counter.started.wait()
    for _ in range(cancel_count):
        pending.cancel()
        await asyncio.sleep(0)
    assert not pending.done()
    counter.release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert logging_obj.finalized_cost is not None
    assert logging_obj.finalized_cost.amount == pytest.approx(0.48)
    assert await finalize_response_cost(logging_obj, result, counter, NOW) is logging_obj.finalized_cost
    assert counter.calls == 1


@pytest.mark.asyncio
async def test_existing_calculator_prices_usage_before_finalizing(logging_obj: Logging) -> None:
    result: Final = response()
    result._hidden_params.pop("response_cost")
    finalized: Final = await finalize_response_cost(logging_obj, result, Counter(), NOW)
    assert finalized is not None and finalized.amount == pytest.approx(0.21)
    logging_obj._success_handler_helper_fn(result, NOW, NOW)
    assert logging_obj.model_call_details["standard_logging_object"]["response_cost"] == pytest.approx(0.21)


@pytest.mark.asyncio
@pytest.mark.parametrize("known_cost", [True, False])
async def test_guardrail_json_response_uses_finalized_header_when_cost_is_known(
    logging_obj: Logging, monkeypatch: pytest.MonkeyPatch, known_cost: bool
) -> None:
    import httpx
    from fastapi import Response

    from litellm.caching.dual_cache import DualCache
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.proxy.utils import ProxyLogging

    class IdentityGuardrail(CustomGuardrail):
        async def async_post_call_success_hook(self, data, user_api_key_dict, response):
            return response

    guardrail: Final = IdentityGuardrail(guardrail_name="identity", event_hook="post_call", default_on=True)
    monkeypatch.setattr(litellm, "callbacks", [guardrail])
    monkeypatch.setattr(litellm, "apply_search_free_tier", False)
    ProxyLogging._callback_capabilities_cache.clear()
    normalized: Final = response()
    if not known_cost:
        normalized._hidden_params["response_cost"] = None
    logging_obj.deferred_success_result = (normalized, NOW, NOW)
    upstream: Final = httpx.Response(200, json={"message": "hello"}, headers={"x-request-id": "upstream-id"})
    processor: Final = ProxyBaseLLMRequestProcessing(data={"custom_llm_provider": "bedrock"})
    result: Final = await processor._handle_non_streaming_allm_passthrough_route(
        response=upstream,
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
        user_api_key_dict=UserAPIKeyAuth(),
        custom_headers={},
        request_headers={},
        logging_obj=logging_obj,
    )
    assert isinstance(result, Response)
    assert result.headers["x-request-id"] == "upstream-id"
    assert result.body == b'{"message": "hello"}'
    if known_cost:
        assert result.headers["x-litellm-response-cost"] == "0.5"
        assert logging_obj.finalized_cost is not None and logging_obj.finalized_cost.amount == 0.5
    else:
        assert "x-litellm-response-cost" not in result.headers
        assert logging_obj.finalized_cost is None
    ProxyLogging._callback_capabilities_cache.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("inline", [True, False])
async def test_malformed_passthrough_logging_preserves_response_dispatch_contract(
    logging_obj: Logging, inline: bool
) -> None:
    import httpx

    from litellm.llms.anthropic.common_utils import AnthropicError
    from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging
    from litellm.types.passthrough_endpoints.pass_through_endpoints import PassthroughStandardLoggingPayload

    url: Final = "https://api.anthropic.com/v1/messages"
    upstream: Final = httpx.Response(200, content=b"invalid-json", request=httpx.Request("POST", url))
    pending: Final = PassThroughEndpointLogging().pass_through_async_success_handler(
        httpx_response=upstream,
        response_body={},
        logging_obj=logging_obj,
        url_route=url,
        result="",
        start_time=NOW,
        end_time=NOW,
        cache_hit=False,
        request_body={},
        passthrough_logging_payload=PassthroughStandardLoggingPayload(url=url, request_method="POST", request_body={}),
        finalize_before_dispatch=inline,
    )
    if inline:
        assert await pending is None
    else:
        with pytest.raises(AnthropicError, match="Unable to get json response"):
            await pending
    assert logging_obj.finalized_cost is None
    assert not logging_obj._async_success_enqueued


@pytest.mark.asyncio
async def test_finalization_failure_leaves_the_request_and_callback_billing_intact(
    logging_obj: Logging, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(**kwargs: object) -> object:
        raise RuntimeError("payload build failed")

    monkeypatch.setattr(logging_obj, "prepare_success_logging", explode)
    counter: Final = Counter()
    assert await finalize_response_cost(logging_obj, response(), counter, NOW) is None
    assert logging_obj.finalized_cost is None
    assert counter.calls == 0


@pytest.mark.asyncio
async def test_deferred_cache_hit_finalizes_zero_for_results_without_hidden_params(logging_obj: Logging) -> None:
    from litellm import completion
    from litellm.caching.caching_handler import LLMCachingHandler

    logging_obj.defer_async_logging = True
    cached_result: Final = response().model_dump()
    LLMCachingHandler(
        original_function=completion, request_kwargs={}, start_time=datetime.now()
    )._async_log_cache_hit_on_callbacks(
        logging_obj=logging_obj, cached_result=cached_result, start_time=NOW, end_time=NOW, cache_hit=True
    )
    counter: Final = Counter()
    finalized: Final = await finalize_response_cost(logging_obj, cached_result, counter, NOW)
    assert finalized is not None and finalized.amount == 0.0
    assert counter.calls == 0
    logging_obj._success_handler_helper_fn(cached_result, NOW, NOW)
    payload: Final = logging_obj.model_call_details["standard_logging_object"]
    assert payload["cache_hit"] is True
    assert payload["saved_cache_cost"] > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("post-call failed"), asyncio.CancelledError(), "cancel_scope"])
@pytest.mark.parametrize("guardrail_cost", [0.0, 0.2])
@pytest.mark.parametrize("mutate_usage", [False, True])
@pytest.mark.parametrize("disable_error_logs", [False, True])
@pytest.mark.parametrize("reserved_cost", [0.0, 0.4])
@pytest.mark.parametrize("disable_spend_updates", [False, True])
async def test_completed_call_keeps_its_charge_when_post_call_processing_fails(
    logging_obj: Logging,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException | str,
    guardrail_cost: float,
    mutate_usage: bool,
    disable_error_logs: bool,
    reserved_cost: float,
    disable_spend_updates: bool,
) -> None:
    from unittest.mock import create_autospec
    import anyio

    from fastapi import Request, Response

    from litellm.proxy._types import ProxyException

    from litellm.caching.caching import DualCache
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
    from litellm.proxy.hooks.proxy_track_cost_callback import ProxyDBLogger
    from litellm.proxy.proxy_server import ProxyConfig
    from litellm.proxy import proxy_server
    from litellm.proxy.utils import ProxyLogging

    scope: Final = anyio.CancelScope()

    class RejectCompletedCall(CustomLogger):
        async def async_post_call_success_hook(
            self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
        ) -> object:
            if mutate_usage and isinstance(response, ModelResponse):
                response.usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            if failure == "cancel_scope":
                scope.cancel()
                await anyio.sleep(0)
            assert isinstance(failure, BaseException)
            raise failure

    monkeypatch.setattr(
        proxy_server,
        "general_settings",
        {
            "disable_error_logs": disable_error_logs,
            "disable_spend_updates": disable_spend_updates,
        },
    )
    writer: Final = create_autospec(DBSpendUpdateWriter, instance=True)
    monkeypatch.setattr(litellm, "callbacks", [RejectCompletedCall(), ProxyDBLogger(spend_writer=lambda: writer)])
    monkeypatch.setattr(litellm, "apply_search_free_tier", False)
    data: Final = {
        "model": MODEL,
        "custom_llm_provider": "vertex_ai",
        "messages": [{"role": "user", "content": "hello"}],
        "mock_response": response(),
        "proxy_server_request": {"headers": {"user-agent": "budget-client"}},
        "metadata": {
            "standard_logging_guardrail_information": [
                {"guardrail_name": "charged-guardrail", "guardrail_cost": guardrail_cost, "guardrail_status": "success"}
            ]
        },
        "litellm_logging_obj": logging_obj,
        "litellm_call_id": logging_obj.litellm_call_id,
    }
    processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    proxy_logging: Final = ProxyLogging(user_api_key_cache=DualCache())
    counters: Final = DualCache()
    for key in ("spend:key:completed-key", "spend:team:completed-team"):
        counters.set_cache(key=key, value=1.5 + reserved_cost)
    counters.set_cache(key="spend:tag:User-Agent: budget-client", value=1.5)
    monkeypatch.setattr(proxy_server, "spend_counter_cache", counters)
    reservation: Final = (
        {
            "reserved_cost": reserved_cost,
            "entries": [
                {"counter_key": key, "entity_type": "Key", "entity_id": key, "reserved_cost": reserved_cost}
                for key in ("spend:key:completed-key", "spend:team:completed-team")
            ],
        }
        if reserved_cost
        else None
    )
    auth: Final = UserAPIKeyAuth(
        request_route="/v1/chat/completions",
        api_key="completed-key",
        team_id="completed-team",
        budget_reservation=reservation,
    )
    request: Final = Request({"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []})
    with scope, pytest.raises(asyncio.CancelledError if failure == "cancel_scope" else type(failure)):
        await processor._process_llm_request(
            request=request,
            fastapi_response=Response(),
            user_api_key_dict=auth,
            route_type="acompletion",
            proxy_logging_obj=proxy_logging,
            general_settings={},
            proxy_config=ProxyConfig(),
            user_model=MODEL,
            skip_pre_call_logic=True,
        )
    if isinstance(failure, Exception):
        with pytest.raises(ProxyException) as error:
            await processor.handle_llm_api_exception(e=failure, user_api_key_dict=auth, proxy_logging_obj=proxy_logging)
        assert float(error.value.headers["x-litellm-response-cost"]) == pytest.approx(0.5 + guardrail_cost)
    if disable_spend_updates:
        assert writer.update_database.await_count == 0
        for key in ("spend:key:completed-key", "spend:team:completed-team", "spend:tag:User-Agent: budget-client"):
            assert await counters.async_get_cache(key=key) == pytest.approx(1.5)
        if auth.budget_reservation is not None:
            assert auth.budget_reservation["finalized"] is True
        return
    assert writer.update_database.await_count == 1
    recorded: Final = writer.update_database.call_args.kwargs
    assert recorded["response_cost"] == pytest.approx(0.5 + guardrail_cost)
    payload: Final = recorded["kwargs"]["standard_logging_object"]
    assert payload["status"] == "failure"
    assert payload["prompt_tokens"] == 10
    assert payload["completion_tokens"] == 5
    assert payload["response_cost"] == pytest.approx(recorded["response_cost"])
    assert not logging_obj.async_success_log_enqueued
    if auth.budget_reservation is not None:
        assert auth.budget_reservation["finalized"] is True
    assert await counters.async_get_cache(key="spend:key:completed-key") == pytest.approx(2.0 + guardrail_cost)
    assert await counters.async_get_cache(key="spend:team:completed-team") == pytest.approx(2.0 + guardrail_cost)

    assert await counters.async_get_cache(key="spend:tag:User-Agent: budget-client") == pytest.approx(
        2.0 + guardrail_cost
    )


@pytest.mark.asyncio
async def test_redacted_completed_call_preserves_usage_and_search_allowance(
    logging_obj: Logging, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy.spend_tracking.cost_finalization import prepare_response_cost

    monkeypatch.setattr(litellm, "turn_off_message_logging", True)
    result: Final = response()
    prepared: Final = prepare_response_cost(logging_obj, result, NOW)
    assert prepared is not None
    result.usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    counter: Final = Counter()
    finalized: Final = await finalize_response_cost(logging_obj, result, counter, NOW)
    assert finalized is not None
    assert finalized.amount == pytest.approx(0.48)
    assert counter.total == 3
    logging_obj._failure_handler_helper_fn(RuntimeError("post-call failed"), "", NOW, NOW)
    payload: Final = logging_obj.model_call_details["standard_logging_object"]
    assert (payload["prompt_tokens"], payload["completion_tokens"]) == (10, 5)
    assert payload["response_cost"] == pytest.approx(0.48)
    assert payload["response"] == {"text": "redacted-by-litellm"}


@pytest.mark.asyncio
async def test_guardrail_json_rejection_retains_completed_provider_usage(
    logging_obj: Logging, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx
    from litellm.caching.dual_cache import DualCache
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.proxy.utils import ProxyLogging

    class RejectJSON(CustomGuardrail):
        async def async_post_call_success_hook(self, data, user_api_key_dict, response):
            raise RuntimeError("blocked JSON response")

    monkeypatch.setattr(
        litellm, "callbacks", [RejectJSON(guardrail_name="reject", event_hook="post_call", default_on=True)]
    )
    monkeypatch.setattr(litellm, "apply_search_free_tier", False)
    ProxyLogging._callback_capabilities_cache.clear()
    normalized: Final = response()
    logging_obj.deferred_success_result = (normalized, NOW, NOW)
    upstream: Final = httpx.Response(200, json={"message": "hello"})
    processor: Final = ProxyBaseLLMRequestProcessing(data={"custom_llm_provider": "bedrock"})
    with pytest.raises(RuntimeError, match="blocked JSON response"):
        await processor._handle_non_streaming_allm_passthrough_route(
            response=upstream,
            proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
            user_api_key_dict=UserAPIKeyAuth(),
            custom_headers={},
            request_headers={},
            logging_obj=logging_obj,
        )
    assert logging_obj.prepared_cost is not None
    normalized.usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    finalized: Final = await finalize_response_cost(logging_obj, upstream, Counter(), NOW)
    assert finalized is not None and finalized.amount == 0.5
    logging_obj._failure_handler_helper_fn(RuntimeError("blocked JSON response"), "", NOW, NOW)
    payload: Final = logging_obj.model_call_details["standard_logging_object"]
    assert (payload["prompt_tokens"], payload["completion_tokens"]) == (10, 5)
    assert payload["response_cost"] == 0.5
    ProxyLogging._callback_capabilities_cache.clear()
