import asyncio
import time
from datetime import datetime, timezone
from itertools import islice
from typing import Optional

import pytest

from litellm.interactions.background_cost_polling import (
    _create_context,
    _poll_intervals,
    _rebuild_logging_obj,
    BackgroundInteractionPollContext,
    InMemoryBackgroundSettlementStore,
    maybe_schedule_background_interaction_cost_polling,
    maybe_settle_background_interaction_before_delete,
    PendingBackgroundInteraction,
    poll_and_log_background_interaction_cost,
    PollSchedule,
    resume_unsettled_background_interactions,
)
from litellm.litellm_core_utils.core_helpers import get_litellm_metadata_from_kwargs
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


def _context(
    logging_obj: LitellmLogging,
    timeout_seconds: float = 1.0,
    store: Optional[InMemoryBackgroundSettlementStore] = None,
) -> BackgroundInteractionPollContext:
    return BackgroundInteractionPollContext(
        interaction_id="interactions/bg-abc",
        custom_llm_provider="gemini",
        logging_obj=logging_obj,
        initial_interval_seconds=0.001,
        max_interval_seconds=0.002,
        timeout_seconds=timeout_seconds,
        store=store if store is not None else InMemoryBackgroundSettlementStore(),
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
    task = await maybe_schedule_background_interaction_cost_polling(
        response=_response("in_progress", with_usage=False),
        create_kwargs={"litellm_logging_obj": logging_obj},
        custom_llm_provider="gemini",
        store=InMemoryBackgroundSettlementStore(),
    )

    assert isinstance(task, asyncio.Task)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_schedule_registers_an_agent_only_create_that_names_no_model():
    logging_obj = LitellmLogging(
        model=None,
        messages=None,
        stream=False,
        call_type="acreate_interaction",
        start_time=time.time(),
        litellm_call_id="bg-agent-call-id",
        function_id="bg-agent-fn-id",
    )
    logging_obj.update_environment_variables(litellm_params={}, optional_params={}, custom_llm_provider="gemini")
    store = InMemoryBackgroundSettlementStore()

    task = await maybe_schedule_background_interaction_cost_polling(
        response=_response("in_progress", with_usage=False),
        create_kwargs={"litellm_logging_obj": logging_obj},
        custom_llm_provider="gemini",
        store=store,
    )

    assert isinstance(task, asyncio.Task)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    pending = await store.pending("interactions/bg-abc")
    assert pending is not None
    assert pending.create_context.model is None
    assert _rebuild_logging_obj(pending.create_context).model is None


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

    task = await maybe_schedule_background_interaction_cost_polling(
        response=response,
        create_kwargs=create_kwargs,
        custom_llm_provider="gemini",
        store=InMemoryBackgroundSettlementStore(),
    )

    assert task is None


def _register_poll(logging_obj: LitellmLogging, poll_fetch=None, store=None) -> asyncio.Task:
    import litellm.interactions.background_cost_polling as bg

    if poll_fetch is None:
        poll_fetch, _ = _fetch_sequence(_response("in_progress", with_usage=False))
    context = _context(logging_obj, store=store)
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
        delete_kwargs={},
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
        delete_kwargs={},
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
        delete_kwargs={},
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
        delete_kwargs={},
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
            delete_kwargs={},
            fetch_interaction=fetch,
        )

    assert reservation["finalized"] is True
    await asyncio.wait_for(task, timeout=5)


@pytest.mark.asyncio
async def test_delete_settlement_ignores_interactions_without_pending_poll():
    fetch, calls = _fetch_sequence(_response("completed", with_usage=True))

    await maybe_settle_background_interaction_before_delete(
        interaction_id="interactions/never-polled",
        delete_kwargs={},
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
        delete_kwargs={},
        fetch_interaction=settle_fetch,
    )

    assert settle_calls == []


@pytest.mark.asyncio
async def test_delete_settlement_does_not_rebill_when_gate_already_claimed():
    logging_obj = _logging_obj()
    store = InMemoryBackgroundSettlementStore()
    assert await store.claim("interactions/bg-abc")
    task = _register_poll(logging_obj, store=store)
    fetch, calls = _fetch_sequence(_response("completed", with_usage=True))

    await maybe_settle_background_interaction_before_delete(
        interaction_id="interactions/bg-abc",
        delete_kwargs={},
        fetch_interaction=fetch,
    )

    assert len(calls) == 1
    assert logging_obj.model_call_details.get("response_cost") is None
    await asyncio.wait_for(task, timeout=5)


@pytest.mark.asyncio
async def test_poller_exits_without_billing_once_settled_elsewhere():
    logging_obj = _logging_obj()
    store = InMemoryBackgroundSettlementStore()
    assert await store.claim("interactions/bg-abc")
    fetch, calls = _fetch_sequence(_response("completed", with_usage=True))

    await poll_and_log_background_interaction_cost(_context(logging_obj, store=store), fetch_interaction=fetch)

    assert calls == []
    assert logging_obj.model_call_details.get("response_cost") is None


@pytest.mark.asyncio
async def test_schedule_respects_kill_switch(monkeypatch):
    import litellm.interactions.background_cost_polling as module

    monkeypatch.setattr(module, "BACKGROUND_INTERACTION_COST_POLLING_ENABLED", False)

    task = await maybe_schedule_background_interaction_cost_polling(
        response=_response("in_progress", with_usage=False),
        create_kwargs={"litellm_logging_obj": _logging_obj()},
        custom_llm_provider="gemini",
        store=InMemoryBackgroundSettlementStore(),
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
    task = await maybe_schedule_background_interaction_cost_polling(
        response=_response("queued", with_usage=False),
        create_kwargs={"litellm_logging_obj": logging_obj},
        custom_llm_provider="gemini",
        store=InMemoryBackgroundSettlementStore(),
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


KEY_HASH = "0123456789abcdef" * 4

FAST_SCHEDULE = PollSchedule(initial_interval_seconds=0.001, max_interval_seconds=0.002, timeout_seconds=1.0)


def _capturing_fetch(response: InteractionsAPIResponse):
    captured = []

    async def fetch(context):
        captured.append(context)
        return response

    return fetch, captured


def _create_metadata(**extra) -> dict:
    return {
        "user_api_key": KEY_HASH,
        "user_api_key_team_id": "team-1",
        "user_api_key_auth": object(),
        **extra,
    }


async def _create_on_a_replica_that_then_dies(logging_obj: LitellmLogging, store) -> None:
    import litellm.interactions.background_cost_polling as bg

    poll_fetch, _ = _fetch_sequence(_response("in_progress", with_usage=False))
    task = await maybe_schedule_background_interaction_cost_polling(
        response=_response("in_progress", with_usage=False),
        create_kwargs={"litellm_logging_obj": logging_obj},
        custom_llm_provider="gemini",
        store=store,
        fetch_interaction=poll_fetch,
    )
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert "interactions/bg-abc" not in bg._ACTIVE_POLLS


@pytest.mark.asyncio
async def test_delete_on_another_replica_bills_the_create_from_the_store():
    """
    The regression: the replica that served the create owns the poll task, so
    a delete served by any other replica used to find nothing to settle and
    the work went unbilled. The store carries the create's attribution, never
    its auth object, to whichever replica settles.
    """
    store = InMemoryBackgroundSettlementStore()
    logging_obj = _logging_obj(litellm_params={"metadata": _create_metadata()})
    await _create_on_a_replica_that_then_dies(logging_obj, store)
    fetch, captured = _capturing_fetch(_response("completed", with_usage=True))

    outcome = await maybe_settle_background_interaction_before_delete(
        interaction_id="interactions/bg-abc",
        delete_kwargs={},
        fetch_interaction=fetch,
        store=store,
    )

    assert outcome == "billed"
    settled = captured[0].logging_obj
    assert settled is not logging_obj
    assert settled.model_call_details["response_cost"] > 0
    payload_metadata = settled.model_call_details["standard_logging_object"]["metadata"]
    assert payload_metadata["user_api_key_hash"] == KEY_HASH
    assert payload_metadata["user_api_key_team_id"] == "team-1"
    assert "user_api_key_auth" not in get_litellm_metadata_from_kwargs(kwargs=settled.model_call_details)


@pytest.mark.asyncio
async def test_delete_on_another_replica_releases_the_create_reservation():
    store = InMemoryBackgroundSettlementStore()
    logging_obj = _logging_obj(
        litellm_params={"metadata": _create_metadata(user_api_key_budget_reservation=_reservation())}
    )
    await _create_on_a_replica_that_then_dies(logging_obj, store)
    fetch, captured = _capturing_fetch(_response("in_progress", with_usage=False))

    outcome = await maybe_settle_background_interaction_before_delete(
        interaction_id="interactions/bg-abc",
        delete_kwargs={},
        fetch_interaction=fetch,
        store=store,
    )

    assert outcome == "released"
    settled_metadata = get_litellm_metadata_from_kwargs(kwargs=captured[0].logging_obj.model_call_details)
    assert settled_metadata["user_api_key_budget_reservation"]["finalized"] is True


@pytest.mark.asyncio
async def test_delete_on_another_replica_fails_when_it_cannot_fetch_and_leaves_the_bill_to_the_creating_poll():
    """
    The settling replica fetches with the delete's credentials, never the
    create's, so a fetch it cannot make (a key only the deployment carries)
    says nothing about the interaction. Deleting anyway would strand the bill
    behind a deleted interaction, so the delete fails with the fetch's error
    and the poll on the creating replica still owns the bill.
    """
    store = InMemoryBackgroundSettlementStore()
    logging_obj = _logging_obj(litellm_params={"metadata": _create_metadata()})
    await _create_on_a_replica_that_then_dies(logging_obj, store)
    fetch, _ = _fetch_sequence(RuntimeError("Google API key is required"))

    with pytest.raises(RuntimeError, match="Google API key is required"):
        await maybe_settle_background_interaction_before_delete(
            interaction_id="interactions/bg-abc", delete_kwargs={}, fetch_interaction=fetch, store=store
        )

    assert await store.is_claimed("interactions/bg-abc") is False
    poll_fetch, _ = _fetch_sequence(_response("completed", with_usage=True))
    await asyncio.wait_for(_register_poll(logging_obj, poll_fetch=poll_fetch, store=store), timeout=5)
    assert logging_obj.model_call_details["response_cost"] > 0


@pytest.mark.asyncio
async def test_delete_settles_once_however_many_replicas_try():
    store = InMemoryBackgroundSettlementStore()
    await _create_on_a_replica_that_then_dies(_logging_obj(litellm_params={"metadata": _create_metadata()}), store)
    first_fetch, first_calls = _capturing_fetch(_response("completed", with_usage=True))
    second_fetch, second_calls = _capturing_fetch(_response("completed", with_usage=True))

    first = await maybe_settle_background_interaction_before_delete(
        interaction_id="interactions/bg-abc", delete_kwargs={}, fetch_interaction=first_fetch, store=store
    )
    second = await maybe_settle_background_interaction_before_delete(
        interaction_id="interactions/bg-abc", delete_kwargs={}, fetch_interaction=second_fetch, store=store
    )

    assert (first, second) == ("billed", None)
    assert len(first_calls) == 1
    assert second_calls == []


def test_create_context_carries_no_request_headers():
    logging_obj = _logging_obj(
        litellm_params={
            "metadata": _create_metadata(
                requester_custom_headers={"x-api-key": "sk-customer-secret"},
                proxy_server_request={"headers": {"x-api-key": "sk-customer-secret"}},
            )
        }
    )

    carried = _create_context(logging_obj, "gemini").metadata

    assert carried["user_api_key_team_id"] == "team-1"
    assert "requester_custom_headers" not in carried
    assert "proxy_server_request" not in carried


@pytest.mark.asyncio
async def test_restart_resumes_only_the_rows_no_replica_claimed():
    store = InMemoryBackgroundSettlementStore()
    create_context = _create_context(_logging_obj(litellm_params={"metadata": _create_metadata()}), "gemini")
    for interaction_id in ("interactions/bg-orphaned", "interactions/bg-settled"):
        await store.register(
            PendingBackgroundInteraction(
                interaction_id=interaction_id,
                custom_llm_provider="gemini",
                create_context=create_context,
                created_at=datetime.now(timezone.utc),
            )
        )
    assert await store.claim("interactions/bg-settled")
    fetch, captured = _capturing_fetch(_response("completed", with_usage=True))

    resumed = await resume_unsettled_background_interactions(store, fetch, schedule=FAST_SCHEDULE)

    assert len(resumed) == 1
    assert await asyncio.wait_for(resumed[0], timeout=5) == "billed"
    assert [context.interaction_id for context in captured] == ["interactions/bg-orphaned"]
    assert captured[0].logging_obj.model_call_details["response_cost"] > 0
    assert await store.is_claimed("interactions/bg-orphaned")


class _ClaimAnswersOnlyAfterTheLastFetch:
    def __init__(self):
        self.store = InMemoryBackgroundSettlementStore()
        self.fetches = 0
        self.fetches_at_last_claim = -1

    async def fetch(self, context):
        self.fetches += 1
        return _response("completed", with_usage=True)

    async def register(self, pending):
        await self.store.register(pending)

    async def pending(self, interaction_id):
        return await self.store.pending(interaction_id)

    async def is_claimed(self, interaction_id):
        return await self.store.is_claimed(interaction_id)

    async def claim(self, interaction_id):
        if self.fetches != self.fetches_at_last_claim:
            self.fetches_at_last_claim = self.fetches
            raise RuntimeError("database unavailable")
        return await self.store.claim(interaction_id)

    async def record_outcome(self, interaction_id, outcome):
        return None

    async def unclaimed(self):
        return await self.store.unclaimed()


@pytest.mark.asyncio
async def test_poller_bills_the_completed_response_it_saw_when_the_claim_only_answers_at_the_deadline():
    logging_obj = _logging_obj()
    store = _ClaimAnswersOnlyAfterTheLastFetch()

    outcome = await poll_and_log_background_interaction_cost(
        _context(logging_obj, timeout_seconds=0.01, store=store),
        fetch_interaction=store.fetch,
    )

    assert store.fetches >= 2
    assert outcome == "billed"
    assert logging_obj.model_call_details["response_cost"] > 0


class _DownStore:
    async def register(self, pending):
        raise RuntimeError("database unavailable")

    async def pending(self, interaction_id):
        raise RuntimeError("database unavailable")

    async def is_claimed(self, interaction_id):
        raise RuntimeError("database unavailable")

    async def claim(self, interaction_id):
        raise RuntimeError("database unavailable")

    async def record_outcome(self, interaction_id, outcome):
        raise RuntimeError("database unavailable")

    async def unclaimed(self):
        raise RuntimeError("database unavailable")


class _RegistersThenRaises:
    def __init__(self):
        self.store = InMemoryBackgroundSettlementStore()

    async def register(self, pending):
        await self.store.register(pending)
        raise RuntimeError("connection reset after the row was committed")

    async def pending(self, interaction_id):
        return await self.store.pending(interaction_id)

    async def is_claimed(self, interaction_id):
        return await self.store.is_claimed(interaction_id)

    async def claim(self, interaction_id):
        return await self.store.claim(interaction_id)

    async def record_outcome(self, interaction_id, outcome):
        return None

    async def unclaimed(self):
        return await self.store.unclaimed()


@pytest.mark.asyncio
async def test_create_whose_registration_raised_after_landing_still_claims_the_stored_row():
    """
    A registration that raises after its row committed used to move the poll
    to a private in-memory gate, so the creating worker billed while the
    stored row stayed unclaimed for another replica's delete or the next boot
    to bill again. The row that landed is the gate every settler shares.
    """
    store = _RegistersThenRaises()
    logging_obj = _logging_obj(litellm_params={"metadata": _create_metadata()})
    poll_fetch, _ = _fetch_sequence(_response("in_progress", with_usage=False))
    task = await maybe_schedule_background_interaction_cost_polling(
        response=_response("in_progress", with_usage=False),
        create_kwargs={"litellm_logging_obj": logging_obj},
        custom_llm_provider="gemini",
        store=store,
        fetch_interaction=poll_fetch,
    )
    fetch, _ = _capturing_fetch(_response("completed", with_usage=True))

    outcome = await maybe_settle_background_interaction_before_delete(
        interaction_id="interactions/bg-abc", delete_kwargs={}, fetch_interaction=fetch, store=store
    )

    assert outcome == "billed"
    assert await store.is_claimed("interactions/bg-abc")
    assert await store.unclaimed() == ()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_delete_on_a_worker_that_resumed_the_poll_fails_when_it_cannot_fetch():
    """
    After a restart every worker resumes the unclaimed rows, so none of them
    is the creator whose delete may release and delete on a failed fetch. A
    resumed worker's delete fails like any other replica's, and its own poll
    still bills the interaction once it completes.
    """
    store = InMemoryBackgroundSettlementStore()
    await store.register(
        PendingBackgroundInteraction(
            interaction_id="interactions/bg-abc",
            custom_llm_provider="gemini",
            create_context=_create_context(_logging_obj(litellm_params={"metadata": _create_metadata()}), "gemini"),
            created_at=datetime.now(timezone.utc),
        )
    )
    responses = [_response("in_progress", with_usage=False)]

    async def poll_fetch(context):
        return responses[-1]

    (resumed,) = await resume_unsettled_background_interactions(store, poll_fetch, schedule=FAST_SCHEDULE)
    fetch, _ = _fetch_sequence(RuntimeError("Google API key is required"))

    with pytest.raises(RuntimeError, match="Google API key is required"):
        await maybe_settle_background_interaction_before_delete(
            interaction_id="interactions/bg-abc", delete_kwargs={}, fetch_interaction=fetch, store=store
        )

    assert await store.is_claimed("interactions/bg-abc") is False
    responses.append(_response("completed", with_usage=True))
    assert await asyncio.wait_for(resumed, timeout=5) == "billed"


@pytest.mark.asyncio
async def test_create_still_settles_on_its_own_replica_when_the_store_is_down():
    logging_obj = _logging_obj()
    poll_fetch, _ = _fetch_sequence(_response("in_progress", with_usage=False))
    task = await maybe_schedule_background_interaction_cost_polling(
        response=_response("in_progress", with_usage=False),
        create_kwargs={"litellm_logging_obj": logging_obj},
        custom_llm_provider="gemini",
        store=_DownStore(),
        fetch_interaction=poll_fetch,
    )
    fetch, calls = _fetch_sequence(_response("completed", with_usage=True))

    outcome = await maybe_settle_background_interaction_before_delete(
        interaction_id="interactions/bg-abc",
        delete_kwargs={},
        fetch_interaction=fetch,
        store=_DownStore(),
    )

    assert outcome == "billed"
    assert len(calls) == 1
    assert logging_obj.model_call_details["response_cost"] > 0
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
