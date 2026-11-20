import asyncio
from decimal import Decimal
from typing import Final

import httpx
import pytest

import litellm
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.tripo_passthrough_logging_handler import (
    poll_tripo_task,
    submitted_tripo_task,
    tripo_task_cost,
)
from litellm.types.passthrough_endpoints.tripo import TripoTaskData, tripo_region_from_url


async def _no_wait(_: float) -> None:
    return None


def test_tripo_model_info_uses_3d_generation_mode() -> None:
    assert litellm.get_model_info("tripo/task", custom_llm_provider="tripo")["mode"] == "3d_generation"


def test_tripo_cost_is_fixed_per_consumed_credit() -> None:
    assert tripo_region_from_url("https://openapi.tripo3d.ai/v3") == "global"
    assert tripo_region_from_url("https://openapi.tripo3d.com/v3") == "cn"
    assert tripo_task_cost(TripoTaskData(task_id="task-1", credits_consumed=Decimal("100"))) == 1.0


def test_submitted_tripo_task_requires_a_task_id() -> None:
    assert submitted_tripo_task({"code": 0, "data": {"status": "running"}}) is None
    assert submitted_tripo_task({"code": 0, "data": {"task_id": "task-1", "status": "running"}}) is not None


@pytest.mark.asyncio
async def test_poll_tripo_task_uses_the_submission_domain_and_stops_on_terminal_status() -> None:
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

    def client_factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    task = await poll_tripo_task(
        httpx.URL("https://openapi.tripo3d.com/v3/generation/text-to-model"),
        "Bearer cn-key",
        "task/1",
        client_factory=client_factory,
        sleep=_no_wait,
    )

    assert task is not None
    assert task.credits_consumed == Decimal("100")
    assert str(requests[0].url) == "https://openapi.tripo3d.com/v3/tasks/task%2F1"
    assert requests[0].headers["authorization"] == "Bearer cn-key"


@pytest.mark.asyncio
async def test_completed_task_credit_cost_reaches_success_callbacks() -> None:
    from datetime import datetime

    from pydantic import TypeAdapter

    from litellm.integrations.custom_logger import CustomLogger
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.tripo_passthrough_logging_handler import (
        TripoPassthroughLoggingHandler,
    )
    from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging
    from litellm.types.utils import StandardLoggingPayload

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
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {"task_id": "completed-task", "status": status, "credits_consumed": credits},
            },
            request=request,
        )

    def client_factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(upstream))

    start: Final = datetime(2026, 1, 1)
    logging: Final = Logging(
        model="tripo/unlisted-test-model",
        messages=[],
        stream=False,
        call_type="pass_through_endpoint",
        start_time=start,
        litellm_call_id="tripo-credit-test",
        function_id="tripo-credit-test",
        dynamic_async_success_callbacks=[Observer()],
    )
    handler: Final = TripoPassthroughLoggingHandler(client_factory=client_factory, sleep=_no_wait)
    submission: Final = httpx.Response(
        200,
        request=httpx.Request(
            "POST",
            "https://openapi.tripo3d.com/v3/generation/text-to-model",
            headers={"Authorization": "Bearer upstream-key"},
        ),
    )
    assert handler.schedule(
        submission,
        {"code": 0, "data": {"task_id": "completed-task"}},
        logging,
        start,
        {"model": "tripo/unlisted-test-model"},
        PassThroughEndpointLogging().handle_logging,
        {},
    )
    await asyncio.gather(*tuple(handler.pending))
    assert observed.done()
    payload: Final = observed.result()
    assert payload["response_cost"] == 0.2
    assert payload["custom_llm_provider"] == "tripo"
    assert logging.model_call_details["response_cost"] == 0.2
    status_response: Final = httpx.Response(
        200,
        request=httpx.Request(
            "GET",
            "https://openapi.tripo3d.com/v3/tasks/completed-task",
        ),
    )
    assert not handler.schedule(
        status_response,
        {"code": 0, "data": {"task_id": "completed-task", "status": "success", "credits_consumed": 20}},
        logging,
        start,
        {},
        PassThroughEndpointLogging().handle_logging,
        {},
    )
    assert not handler.pending


@pytest.mark.asyncio
async def test_running_task_with_credits_does_not_produce_a_final_cost() -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {"task_id": "running-task", "status": "running", "credits_consumed": 20},
            },
            request=request,
        )

    task: Final = await poll_tripo_task(
        httpx.URL("https://openapi.tripo3d.com/v3/generation/text-to-model"),
        "Bearer test-key",
        "running-task",
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(upstream)),
        sleep=_no_wait,
        max_attempts=2,
    )
    assert task is None
    assert tripo_task_cost(task) is None


@pytest.mark.asyncio
async def test_in_flight_polls_wait_together_then_both_bill() -> None:
    from datetime import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.tripo_passthrough_logging_handler import (
        TripoPassthroughLoggingHandler,
    )

    both_waiting: Final = asyncio.Event()
    release: Final = asyncio.Event()
    waiting: list[str] = []
    costs: list[object] = []
    polls: dict[str, int] = {}

    async def wait_for_peer(_: float) -> None:
        waiting.append("asleep")
        if len(waiting) == 2:
            both_waiting.set()
        await release.wait()

    def upstream(request: httpx.Request) -> httpx.Response:
        task_id: Final = request.url.path.removeprefix("/v3/tasks/")
        polls[task_id] = polls.get(task_id, 0) + 1
        status: Final = "queued" if polls[task_id] == 1 else "success"
        return httpx.Response(
            200,
            json={"code": 0, "data": {"task_id": task_id, "status": status, "credits_consumed": 20}},
            request=request,
        )

    async def record(**kwargs: object) -> None:
        costs.append(kwargs["response_cost"])

    def call_log(task_id: str) -> Logging:
        return Logging(
            model="tripo/task",
            messages=[],
            stream=False,
            call_type="pass_through_endpoint",
            start_time=datetime(2026, 1, 1),
            litellm_call_id=task_id,
            function_id=task_id,
        )

    handler: Final = TripoPassthroughLoggingHandler(
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(upstream)),
        sleep=wait_for_peer,
    )
    submission: Final = httpx.Response(
        200,
        request=httpx.Request(
            "POST",
            "https://openapi.tripo3d.com/v3/generation/text-to-model",
            headers={"Authorization": "Bearer upstream-key"},
        ),
    )
    for task_id in ("task-a", "task-b"):
        assert handler.schedule(
            submission,
            {"code": 0, "data": {"task_id": task_id}},
            call_log(task_id),
            datetime(2026, 1, 1),
            {},
            record,
            {},
        )
    await asyncio.wait_for(both_waiting.wait(), timeout=5)
    assert waiting == ["asleep", "asleep"]
    release.set()
    pending: Final = tuple(handler.pending)
    await asyncio.gather(*pending)
    assert [task.exception() for task in pending] == [None, None]
    assert costs == [0.2, 0.2]
