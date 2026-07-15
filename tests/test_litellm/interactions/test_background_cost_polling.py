import asyncio
import time
from typing import Optional

import pytest

from litellm.interactions.background_cost_polling import (
    BackgroundInteractionPollContext,
    maybe_schedule_background_interaction_cost_polling,
    poll_and_log_background_interaction_cost,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
from litellm.types.interactions import InteractionsAPIResponse

USAGE_BLOCK = {
    "total_tokens": 175,
    "total_input_tokens": 100,
    "input_tokens_by_modality": [{"modality": "text", "tokens": 100}],
    "total_cached_tokens": 0,
    "total_output_tokens": 50,
    "output_tokens_by_modality": [{"modality": "text", "tokens": 50}],
    "total_tool_use_tokens": 0,
    "total_thought_tokens": 25,
}


def _logging_obj(
    call_type: str = "acreate_interaction",
    litellm_params: Optional[dict] = None,
) -> LitellmLogging:
    logging_obj = LitellmLogging(
        model="gemini-2.5-flash",
        messages=[],
        stream=False,
        call_type=call_type,
        start_time=time.time(),
        litellm_call_id="bg-interactions-call-id",
        function_id="bg-interactions-fn-id",
    )
    logging_obj.update_environment_variables(
        litellm_params=litellm_params or {},
        optional_params={},
        model="gemini-2.5-flash",
        custom_llm_provider="gemini",
        input="hi",
    )
    return logging_obj


def _reservation() -> dict:
    return {"reserved_cost": 0.05, "entries": [], "finalized": False, "input_cost": 0.001}


def _logging_obj_with_reservation(reservation: dict) -> LitellmLogging:
    return _logging_obj(litellm_params={"metadata": {"user_api_key_budget_reservation": reservation}})


def _context(logging_obj: LitellmLogging, timeout_seconds: float = 1.0) -> BackgroundInteractionPollContext:
    return BackgroundInteractionPollContext(
        interaction_id="interactions/bg-abc",
        custom_llm_provider="gemini",
        logging_obj=logging_obj,
        initial_interval_seconds=0.001,
        max_interval_seconds=0.002,
        timeout_seconds=timeout_seconds,
    )


def _response(status: str, with_usage: bool) -> InteractionsAPIResponse:
    return InteractionsAPIResponse(
        id="interactions/bg-abc",
        model="gemini-2.5-flash",
        status=status,
        steps=[],
        usage=dict(USAGE_BLOCK) if with_usage else None,
    )


def _fetch_sequence(*responses):
    remaining = list(responses)
    calls = []

    async def fetch(context):
        calls.append(context.interaction_id)
        item = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        if isinstance(item, Exception):
            raise item
        return item

    return fetch, calls


@pytest.mark.asyncio
async def test_poller_bills_once_when_interaction_completes():
    logging_obj = _logging_obj()
    fetch, calls = _fetch_sequence(
        _response("in_progress", with_usage=False),
        _response("completed", with_usage=True),
    )

    await poll_and_log_background_interaction_cost(_context(logging_obj), fetch_interaction=fetch)

    assert len(calls) == 2
    assert logging_obj.model_call_details["response_cost"] > 0
    assert logging_obj.model_call_details["standard_logging_object"]["total_tokens"] == 175


@pytest.mark.asyncio
async def test_poller_stops_without_billing_on_terminal_status_without_usage():
    logging_obj = _logging_obj()
    fetch, calls = _fetch_sequence(_response("failed", with_usage=False))

    await poll_and_log_background_interaction_cost(_context(logging_obj), fetch_interaction=fetch)

    assert len(calls) == 1
    assert logging_obj.model_call_details.get("response_cost") is None


@pytest.mark.asyncio
async def test_poller_gives_up_after_timeout_without_billing():
    logging_obj = _logging_obj()
    fetch, calls = _fetch_sequence(_response("in_progress", with_usage=False))

    await poll_and_log_background_interaction_cost(
        _context(logging_obj, timeout_seconds=0.01),
        fetch_interaction=fetch,
    )

    assert len(calls) >= 2
    assert logging_obj.model_call_details.get("response_cost") is None


@pytest.mark.asyncio
async def test_poller_releases_budget_reservation_when_interaction_ends_without_usage():
    reservation = _reservation()
    logging_obj = _logging_obj_with_reservation(reservation)
    fetch, _ = _fetch_sequence(_response("failed", with_usage=False))

    await poll_and_log_background_interaction_cost(_context(logging_obj), fetch_interaction=fetch)

    assert reservation["finalized"] is True


@pytest.mark.asyncio
async def test_poller_releases_budget_reservation_on_timeout_give_up():
    reservation = _reservation()
    logging_obj = _logging_obj_with_reservation(reservation)
    fetch, _ = _fetch_sequence(_response("in_progress", with_usage=False))

    await poll_and_log_background_interaction_cost(
        _context(logging_obj, timeout_seconds=0.01),
        fetch_interaction=fetch,
    )

    assert reservation["finalized"] is True


@pytest.mark.asyncio
async def test_poller_leaves_reservation_reconciliation_to_the_completion_event():
    reservation = _reservation()
    logging_obj = _logging_obj_with_reservation(reservation)
    fetch, _ = _fetch_sequence(
        _response("in_progress", with_usage=False),
        _response("completed", with_usage=True),
    )

    await poll_and_log_background_interaction_cost(_context(logging_obj), fetch_interaction=fetch)

    assert logging_obj.model_call_details["response_cost"] > 0
    assert reservation["finalized"] is False


@pytest.mark.asyncio
async def test_poller_retries_after_fetch_error_and_still_bills():
    logging_obj = _logging_obj()
    fetch, calls = _fetch_sequence(
        RuntimeError("transient network error"),
        _response("completed", with_usage=True),
    )

    await poll_and_log_background_interaction_cost(_context(logging_obj), fetch_interaction=fetch)

    assert len(calls) == 2
    assert logging_obj.model_call_details["response_cost"] > 0


@pytest.mark.asyncio
async def test_schedule_creates_poll_task_for_in_progress_create():
    logging_obj = _logging_obj()
    task = maybe_schedule_background_interaction_cost_polling(
        response=_response("in_progress", with_usage=False),
        create_kwargs={"litellm_logging_obj": logging_obj},
        custom_llm_provider="gemini",
    )

    assert isinstance(task, asyncio.Task)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response,create_kwargs",
    [
        (_response("completed", with_usage=True), {"litellm_logging_obj": "placeholder"}),
        (_response("in_progress", with_usage=False), {}),
        ("not a response", {"litellm_logging_obj": "placeholder"}),
    ],
)
async def test_schedule_skips_non_pollable_results(response, create_kwargs):
    if create_kwargs.get("litellm_logging_obj") == "placeholder":
        create_kwargs = {"litellm_logging_obj": _logging_obj()}

    task = maybe_schedule_background_interaction_cost_polling(
        response=response,
        create_kwargs=create_kwargs,
        custom_llm_provider="gemini",
    )

    assert task is None


@pytest.mark.asyncio
async def test_schedule_respects_kill_switch(monkeypatch):
    import litellm.interactions.background_cost_polling as module

    monkeypatch.setattr(module, "BACKGROUND_INTERACTION_COST_POLLING_ENABLED", False)

    task = maybe_schedule_background_interaction_cost_polling(
        response=_response("in_progress", with_usage=False),
        create_kwargs={"litellm_logging_obj": _logging_obj()},
        custom_llm_provider="gemini",
    )

    assert task is None
