import asyncio
import time
from itertools import islice
from typing import Optional

import pytest

from litellm.interactions.background_cost_polling import (
    _SETTLED_KEY,
    _poll_intervals,
    BackgroundInteractionPollContext,
    maybe_schedule_background_interaction_cost_polling,
    maybe_settle_background_interaction_before_delete,
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


async def _raise_on_billing(result: InteractionsAPIResponse) -> None:
    raise RuntimeError("cost calculation failed for a settled background interaction")


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


@pytest.mark.parametrize(
    "initial, maximum",
    [(0.0, 0.002), (0.001, 0.0), (-1.0, 0.002), (0.0, 0.0)],
)
def test_poll_intervals_stops_instead_of_looping_on_a_non_positive_interval(initial, maximum):
    intervals = list(islice(_poll_intervals(initial=initial, maximum=maximum, timeout=3600.0), 10))

    assert len(intervals) < 10
    assert all(interval > 0 for interval in intervals)


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
async def test_poller_bills_an_interaction_paused_for_a_tool_result():
    logging_obj = _logging_obj()
    fetch, calls = _fetch_sequence(
        _response("in_progress", with_usage=False),
        _response("requires_action", with_usage=True),
    )

    await poll_and_log_background_interaction_cost(_context(logging_obj), fetch_interaction=fetch)

    assert len(calls) == 2
    assert logging_obj.model_call_details["response_cost"] > 0
    assert logging_obj.model_call_details["standard_logging_object"]["total_tokens"] == 175


@pytest.mark.asyncio
async def test_poller_does_not_pin_the_budget_for_an_interaction_paused_for_a_tool_result():
    reservation = _reservation()
    logging_obj = _logging_obj_with_reservation(reservation)
    fetch, _ = _fetch_sequence(_response("requires_action", with_usage=True))

    await poll_and_log_background_interaction_cost(_context(logging_obj), fetch_interaction=fetch)

    assert logging_obj.model_call_details["response_cost"] > 0
    assert reservation["finalized"] is False


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
async def test_poller_releases_budget_reservation_when_billing_raises():
    reservation = _reservation()
    logging_obj = _logging_obj_with_reservation(reservation)
    fetch, _ = _fetch_sequence(_response("completed", with_usage=True))
    logging_obj.async_log_background_interaction_completion = _raise_on_billing

    with pytest.raises(RuntimeError):
        await poll_and_log_background_interaction_cost(_context(logging_obj), fetch_interaction=fetch)

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


def _register_poll(logging_obj: LitellmLogging, poll_fetch=None) -> asyncio.Task:
    import litellm.interactions.background_cost_polling as bg

    if poll_fetch is None:
        poll_fetch, _ = _fetch_sequence(_response("in_progress", with_usage=False))
    context = _context(logging_obj)
    task = asyncio.create_task(poll_and_log_background_interaction_cost(context, fetch_interaction=poll_fetch))
    bg._ACTIVE_POLLS[context.interaction_id] = bg._ActiveBackgroundPoll(task=task, context=context)
    task.add_done_callback(lambda finished: bg._discard_poll(context.interaction_id, finished))
    return task


@pytest.mark.asyncio
async def test_delete_settlement_bills_an_interaction_paused_for_a_tool_result():
    logging_obj = _logging_obj()
    task = _register_poll(logging_obj)
    fetch, calls = _fetch_sequence(_response("requires_action", with_usage=True))

    await maybe_settle_background_interaction_before_delete(
        interaction_id="interactions/bg-abc",
        fetch_interaction=fetch,
    )

    assert len(calls) == 1
    assert logging_obj.model_call_details["response_cost"] > 0
    await asyncio.wait_for(task, timeout=5)


@pytest.mark.asyncio
async def test_delete_settlement_bills_pending_background_interaction():
    logging_obj = _logging_obj()
    task = _register_poll(logging_obj)
    fetch, calls = _fetch_sequence(_response("completed", with_usage=True))

    await maybe_settle_background_interaction_before_delete(
        interaction_id="interactions/bg-abc",
        fetch_interaction=fetch,
    )

    assert len(calls) == 1
    assert logging_obj.model_call_details["response_cost"] > 0
    assert logging_obj.model_call_details["standard_logging_object"]["total_tokens"] == 175
    await asyncio.wait_for(task, timeout=5)


@pytest.mark.asyncio
async def test_delete_settlement_releases_reservation_when_still_in_progress():
    reservation = _reservation()
    logging_obj = _logging_obj_with_reservation(reservation)
    task = _register_poll(logging_obj)
    fetch, _ = _fetch_sequence(_response("in_progress", with_usage=False))

    await maybe_settle_background_interaction_before_delete(
        interaction_id="interactions/bg-abc",
        fetch_interaction=fetch,
    )

    assert reservation["finalized"] is True
    assert logging_obj.model_call_details.get("response_cost") is None
    await asyncio.wait_for(task, timeout=5)


@pytest.mark.asyncio
async def test_delete_settlement_releases_reservation_when_prefetch_fails():
    reservation = _reservation()
    logging_obj = _logging_obj_with_reservation(reservation)
    task = _register_poll(logging_obj)
    fetch, _ = _fetch_sequence(RuntimeError("interaction already deleted"))

    await maybe_settle_background_interaction_before_delete(
        interaction_id="interactions/bg-abc",
        fetch_interaction=fetch,
    )

    assert reservation["finalized"] is True
    assert logging_obj.model_call_details.get("response_cost") is None
    await asyncio.wait_for(task, timeout=5)


@pytest.mark.asyncio
async def test_delete_settlement_releases_reservation_when_billing_raises():
    reservation = _reservation()
    logging_obj = _logging_obj_with_reservation(reservation)
    task = _register_poll(logging_obj)
    fetch, _ = _fetch_sequence(_response("completed", with_usage=True))
    logging_obj.async_log_background_interaction_completion = _raise_on_billing

    with pytest.raises(RuntimeError):
        await maybe_settle_background_interaction_before_delete(
            interaction_id="interactions/bg-abc",
            fetch_interaction=fetch,
        )

    assert reservation["finalized"] is True
    await asyncio.wait_for(task, timeout=5)


@pytest.mark.asyncio
async def test_delete_settlement_ignores_interactions_without_pending_poll():
    fetch, calls = _fetch_sequence(_response("completed", with_usage=True))

    await maybe_settle_background_interaction_before_delete(
        interaction_id="interactions/never-polled",
        fetch_interaction=fetch,
    )

    assert calls == []


@pytest.mark.asyncio
async def test_delete_settlement_noop_after_poll_task_finished():
    logging_obj = _logging_obj()
    poll_fetch, _ = _fetch_sequence(_response("completed", with_usage=True))
    task = _register_poll(logging_obj, poll_fetch=poll_fetch)
    await asyncio.wait_for(task, timeout=5)
    assert logging_obj.model_call_details["response_cost"] > 0

    settle_fetch, settle_calls = _fetch_sequence(_response("completed", with_usage=True))
    await maybe_settle_background_interaction_before_delete(
        interaction_id="interactions/bg-abc",
        fetch_interaction=settle_fetch,
    )

    assert settle_calls == []


@pytest.mark.asyncio
async def test_delete_settlement_does_not_rebill_when_gate_already_claimed():
    logging_obj = _logging_obj()
    logging_obj.model_call_details[_SETTLED_KEY] = True
    task = _register_poll(logging_obj)
    fetch, calls = _fetch_sequence(_response("completed", with_usage=True))

    await maybe_settle_background_interaction_before_delete(
        interaction_id="interactions/bg-abc",
        fetch_interaction=fetch,
    )

    assert len(calls) == 1
    assert logging_obj.model_call_details.get("response_cost") is None
    await asyncio.wait_for(task, timeout=5)


@pytest.mark.asyncio
async def test_poller_exits_without_billing_once_settled_elsewhere():
    logging_obj = _logging_obj()
    logging_obj.model_call_details[_SETTLED_KEY] = True
    fetch, calls = _fetch_sequence(_response("completed", with_usage=True))

    await poll_and_log_background_interaction_cost(_context(logging_obj), fetch_interaction=fetch)

    assert calls == []
    assert logging_obj.model_call_details.get("response_cost") is None


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


def test_every_status_the_api_can_return_is_either_pollable_or_terminal():
    """
    The proxy bills a usage-less create in exactly two ways: it polls the
    interaction until it settles, or it recognises the status as terminal and
    settles immediately. A status in neither set is billed by nobody, alerts
    nobody, and releases its budget reservation, which is the zero-spend bug
    this whole module exists to fix.

    Pinned against the generated spec enum rather than a hand-written list, so
    a status Google adds later breaks this test instead of silently shipping
    another unbilled path.
    """
    from litellm.interactions.background_cost_polling import _POLLABLE_STATUSES, _TERMINAL_STATUSES
    from litellm.types.interactions.generated import Status1

    spec_statuses = {member.value for member in Status1}
    handled = _POLLABLE_STATUSES | _TERMINAL_STATUSES

    assert spec_statuses - handled == set()
    assert handled - spec_statuses == set()


@pytest.mark.asyncio
async def test_schedule_creates_poll_task_for_queued_create():
    """
    ``queued`` is the API's not-started-yet state. It carries no usage, so the
    create cannot bill it, and it is not terminal, so nothing settles it:
    without a poll task it is never charged at all.
    """
    logging_obj = _logging_obj()
    task = maybe_schedule_background_interaction_cost_polling(
        response=_response("queued", with_usage=False),
        create_kwargs={"litellm_logging_obj": logging_obj},
        custom_llm_provider="gemini",
    )

    assert isinstance(task, asyncio.Task)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_poller_bills_an_interaction_that_started_out_queued():
    logging_obj = _logging_obj()
    fetch, calls = _fetch_sequence(
        _response("queued", with_usage=False),
        _response("in_progress", with_usage=False),
        _response("completed", with_usage=True),
    )

    await poll_and_log_background_interaction_cost(_context(logging_obj), fetch_interaction=fetch)

    assert len(calls) == 3
    assert logging_obj.model_call_details["response_cost"] > 0
    assert logging_obj.model_call_details["standard_logging_object"]["total_tokens"] == 175


def test_poll_intervals_double_up_to_the_cap_and_stay_inside_the_timeout():
    """
    The degenerate cases are covered above; this pins the shape the proxy
    actually ships, so an off-by-one in the doubling or in the remaining-budget
    check cannot pass green.
    """
    intervals = list(_poll_intervals(initial=5.0, maximum=60.0, timeout=3600.0))

    assert intervals[:6] == [5.0, 10.0, 20.0, 40.0, 60.0, 60.0]
    assert max(intervals) == 60.0
    assert sum(intervals) <= 3600.0
    assert sum(intervals) + 60.0 > 3600.0


@pytest.mark.asyncio
async def test_giving_up_on_an_unrecognized_status_says_which_status_it_was(monkeypatch):
    """
    A status outside both sets polls for the full timeout and then gives up.
    The give-up line is the only trace it leaves, so it has to name the status
    rather than reporting it as an interaction that was merely still running.
    """
    import litellm.interactions.background_cost_polling as bg

    errors = []
    monkeypatch.setattr(bg.verbose_logger, "error", lambda *args, **kwargs: errors.append(args))

    logging_obj = _logging_obj()
    fetch, _ = _fetch_sequence(_response("halted_for_review", with_usage=False))

    await poll_and_log_background_interaction_cost(
        _context(logging_obj, timeout_seconds=0.01), fetch_interaction=fetch
    )

    assert len(errors) == 1
    assert "halted_for_review" in errors[0]
