from decimal import Decimal
from typing import Final

import httpx
import pytest

import litellm
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.tripo_passthrough_logging_handler import (
    poll_tripo_task,
    submitted_tripo_task,
    tripo_cost_per_credit,
    tripo_task_cost,
)
from litellm.types.passthrough_endpoints.tripo import TripoTaskData, tripo_region_from_url


def test_tripo_model_info_uses_3d_generation_mode() -> None:
    assert litellm.get_model_info("tripo/task", custom_llm_provider="tripo")["mode"] == "3d_generation"


def test_tripo_region_and_cost_are_derived_from_upstream_data(monkeypatch: pytest.MonkeyPatch) -> None:
    price: Final = 0.025
    monkeypatch.setitem(litellm.model_cost, "tripo/task", {
        **litellm.model_cost["tripo/task"], "cost_per_credit": price,
    })
    assert tripo_region_from_url("https://openapi.tripo3d.ai/v3") == "global"
    assert tripo_region_from_url("https://openapi.tripo3d.com/v3") == "cn"
    assert tripo_cost_per_credit("tripo/task") == price
    assert tripo_cost_per_credit("tripo/unlisted-test-model") == price
    assert tripo_task_cost(TripoTaskData(task_id="task-1", credits_consumed=Decimal("100")), price) == 2.5


def test_submitted_tripo_task_requires_a_task_id() -> None:
    assert submitted_tripo_task({"code": 0, "data": {"status": "running"}}) is None
    assert submitted_tripo_task({"code": 0, "data": {"task_id": "task-1", "status": "running"}}) is not None


def test_poll_tripo_task_uses_the_submission_domain_and_stops_on_terminal_status() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {"task_id": "task/1", "status": "success", "credits_consumed": 100},
            },
            request=request,
        )

    def client_factory() -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler))

    task = poll_tripo_task(
        httpx.URL("https://openapi.tripo3d.com/v3/generation/text-to-model"),
        "Bearer cn-key",
        "task/1",
        client_factory=client_factory,
        sleep=lambda _: None,
    )

    assert task is not None
    assert task.credits_consumed == Decimal("100")
    assert str(requests[0].url) == "https://openapi.tripo3d.com/v3/tasks/task%2F1"
    assert requests[0].headers["authorization"] == "Bearer cn-key"


@pytest.mark.parametrize("price", [0.0, 0.037])
def test_tripo_model_specific_credit_price_overrides_fallback(monkeypatch: pytest.MonkeyPatch, price: float) -> None:
    monkeypatch.setitem(litellm.model_cost, "tripo/test-priced-model", {
        "litellm_provider": "tripo", "mode": "3d_generation", "cost_per_credit": price,
    })
    assert tripo_cost_per_credit("tripo/test-priced-model") == price


@pytest.mark.asyncio
async def test_completed_task_credit_cost_reaches_success_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    from datetime import datetime

    from pydantic import TypeAdapter

    from litellm.integrations.custom_logger import CustomLogger
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.tripo_passthrough_logging_handler import (
        TripoPassthroughLoggingHandler,
    )
    from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging
    from litellm.types.utils import StandardLoggingPayload

    price: Final = 0.025
    monkeypatch.setitem(litellm.model_cost, "tripo/task", {
        **litellm.model_cost["tripo/task"], "cost_per_credit": price,
    })
    observed: Final[asyncio.Future[StandardLoggingPayload]] = asyncio.get_running_loop().create_future()

    class Observer(CustomLogger):
        async def async_log_success_event(
            self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
        ) -> None:
            observed.set_result(TypeAdapter(StandardLoggingPayload).validate_python(kwargs["standard_logging_object"]))

    states: Final = iter((("queued", 0), ("running", 15), ("success", 20)))

    def upstream(request: httpx.Request) -> httpx.Response:
        assert not observed.done()
        status, credits = next(states)
        return httpx.Response(200, json={
            "code": 0, "data": {"task_id": "completed-task", "status": status, "credits_consumed": credits},
        }, request=request)

    def client_factory() -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(upstream))

    start: Final = datetime(2026, 1, 1)
    logging: Final = Logging(
        model="tripo/unlisted-test-model", messages=[], stream=False, call_type="pass_through_endpoint",
        start_time=start, litellm_call_id="tripo-credit-test", function_id="tripo-credit-test",
        dynamic_async_success_callbacks=[Observer()],
    )
    handler: Final = TripoPassthroughLoggingHandler(client_factory=client_factory, sleep=lambda _: None)
    submission: Final = httpx.Response(200, request=httpx.Request(
        "POST", "https://openapi.tripo3d.com/v3/generation/text-to-model",
        headers={"Authorization": "Bearer upstream-key"},
    ))
    assert handler.schedule(
        submission, {"code": 0, "data": {"task_id": "completed-task"}}, logging, start,
        {"model": "tripo/unlisted-test-model"}, PassThroughEndpointLogging().handle_logging, {},
    )
    await asyncio.gather(*tuple(handler.pending))
    assert observed.done()
    payload: Final = observed.result()
    assert payload["response_cost"] == 0.5
    assert payload["custom_llm_provider"] == "tripo"
    assert logging.model_call_details["response_cost"] == 0.5
    status_response: Final = httpx.Response(200, request=httpx.Request(
        "GET", "https://openapi.tripo3d.com/v3/tasks/completed-task",
    ))
    assert not handler.schedule(
        status_response,
        {"code": 0, "data": {"task_id": "completed-task", "status": "success", "credits_consumed": 20}},
        logging, start, {}, PassThroughEndpointLogging().handle_logging, {},
    )
    assert not handler.pending


def test_running_task_with_credits_does_not_produce_a_final_cost() -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "code": 0, "data": {"task_id": "running-task", "status": "running", "credits_consumed": 20},
        }, request=request)

    task: Final = poll_tripo_task(
        httpx.URL("https://openapi.tripo3d.com/v3/generation/text-to-model"), "Bearer test-key", "running-task",
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(upstream)),
        sleep=lambda _: None, max_attempts=2,
    )
    assert task is None
    assert tripo_task_cost(task, 0.025) is None
