import asyncio
import time
from datetime import datetime, timezone
from typing import Optional

import pytest

from litellm.interactions.background_cost_polling import (
    _SETTLED_KEY,
    BackgroundInteractionPollContext,
    BackgroundSettlementContext,
    SettlementOutcome,
    get_settlement_store,
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


class _InMemorySettlementStore:
    def __init__(self, persist_error: Optional[Exception] = None) -> None:
        self.rows: dict[str, dict] = {}
        self.contexts: dict[str, BackgroundSettlementContext] = {}
        self.outcomes: dict[str, SettlementOutcome] = {}
        self.settle_before_delete_calls: list[str] = []
        self.persist_error = persist_error

    async def persist_pending(self, context: BackgroundSettlementContext, timeout_at: datetime) -> None:
        if self.persist_error is not None:
            raise self.persist_error
        existing = self.rows.get(context.interaction_id)
        status = existing["status"] if existing is not None else "pending"
        self.rows[context.interaction_id] = {"status": status, "timeout_at": timeout_at}
        self.contexts[context.interaction_id] = context

    async def claim(self, interaction_id: str) -> bool:
        row = self.rows.get(interaction_id)
        if row is None:
            return True
        if row["status"] != "pending":
            return False
        row["status"] = "settled"
        return True

    async def record_outcome(self, interaction_id: str, outcome: SettlementOutcome) -> None:
        self.outcomes[interaction_id] = outcome

    async def is_pending(self, interaction_id: str) -> bool:
        row = self.rows.get(interaction_id)
        return row is not None and row["status"] == "pending"

    async def settle_pending_before_delete(self, interaction_id: str) -> None:
        self.settle_before_delete_calls.append(interaction_id)


def _attributed_litellm_params(reservation: Optional[dict] = None) -> dict:
    metadata = {
        "user_api_key": "hashed-virtual-key",
        "user_api_key_user_id": "user-123",
        "user_api_key_team_id": "team-456",
        "deployment_model_name": "gemini-3-flash-preview",
    }
    if reservation is not None:
        metadata["user_api_key_budget_reservation"] = reservation
    return {"metadata": metadata}


def test_no_settlement_store_registered_by_default():
    assert get_settlement_store() is None


@pytest.mark.asyncio
async def test_poller_persists_settlement_context_with_attribution_and_reservation():
    store = _InMemorySettlementStore()
    reservation = _reservation()
    logging_obj = _logging_obj(litellm_params=_attributed_litellm_params(reservation))
    fetch, _ = _fetch_sequence(_response("completed", with_usage=True))

    await poll_and_log_background_interaction_cost(_context(logging_obj), fetch_interaction=fetch, store=store)

    context = store.contexts["interactions/bg-abc"]
    assert context.attribution.user_api_key == "hashed-virtual-key"
    assert context.attribution.user_api_key_user_id == "user-123"
    assert context.attribution.user_api_key_team_id == "team-456"
    assert context.model == "gemini-2.5-flash"
    assert context.model_group == "gemini-3-flash-preview"
    assert context.custom_llm_provider == "gemini"
    assert context.call_type == "acreate_interaction"
    assert context.litellm_call_id == "bg-interactions-call-id"
    assert context.budget_reservation is not None
    assert context.budget_reservation.reserved_cost == reservation["reserved_cost"]
    assert store.rows["interactions/bg-abc"]["timeout_at"] > datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_poller_records_billed_outcome_so_sweep_cannot_rebill():
    store = _InMemorySettlementStore()
    logging_obj = _logging_obj()
    fetch, _ = _fetch_sequence(_response("completed", with_usage=True))

    await poll_and_log_background_interaction_cost(_context(logging_obj), fetch_interaction=fetch, store=store)

    assert logging_obj.model_call_details["response_cost"] > 0
    assert store.rows["interactions/bg-abc"]["status"] == "settled"
    assert store.outcomes["interactions/bg-abc"] == "billed"


def _store_with_row_settled_elsewhere(interaction_id: str = "interactions/bg-abc") -> _InMemorySettlementStore:
    store = _InMemorySettlementStore()
    store.rows[interaction_id] = {"status": "settled", "timeout_at": datetime.now(timezone.utc)}
    return store


@pytest.mark.asyncio
async def test_poller_skips_billing_and_finalizes_locally_when_store_claim_lost():
    store = _store_with_row_settled_elsewhere()
    reservation = _reservation()
    logging_obj = _logging_obj_with_reservation(reservation)
    fetch, _ = _fetch_sequence(_response("completed", with_usage=True))

    await poll_and_log_background_interaction_cost(_context(logging_obj), fetch_interaction=fetch, store=store)

    assert logging_obj.model_call_details.get("response_cost") is None
    assert reservation["finalized"] is True
    assert store.outcomes == {}


@pytest.mark.asyncio
async def test_timeout_give_up_does_not_release_when_store_claim_lost():
    store = _store_with_row_settled_elsewhere()
    reservation = _reservation()
    logging_obj = _logging_obj_with_reservation(reservation)
    fetch, _ = _fetch_sequence(_response("in_progress", with_usage=False))

    await poll_and_log_background_interaction_cost(
        _context(logging_obj, timeout_seconds=0.01),
        fetch_interaction=fetch,
        store=store,
    )

    assert reservation["finalized"] is True
    assert store.outcomes == {}
    assert logging_obj.model_call_details.get("response_cost") is None


@pytest.mark.asyncio
async def test_timeout_give_up_records_abandoned_outcome_when_claim_won():
    store = _InMemorySettlementStore()
    reservation = _reservation()
    logging_obj = _logging_obj_with_reservation(reservation)

    await poll_and_log_background_interaction_cost(
        _context(logging_obj, timeout_seconds=0.01),
        fetch_interaction=_fetch_sequence(_response("in_progress", with_usage=False))[0],
        store=store,
    )

    assert reservation["finalized"] is True
    assert store.rows["interactions/bg-abc"]["status"] == "settled"
    assert store.outcomes["interactions/bg-abc"] == "abandoned"


@pytest.mark.asyncio
async def test_poller_continues_in_memory_when_persist_fails():
    store = _InMemorySettlementStore(persist_error=RuntimeError("db unavailable"))
    logging_obj = _logging_obj()
    fetch, calls = _fetch_sequence(_response("completed", with_usage=True))

    await poll_and_log_background_interaction_cost(_context(logging_obj), fetch_interaction=fetch, store=store)

    assert len(calls) == 1
    assert logging_obj.model_call_details["response_cost"] > 0
    assert store.rows == {}
    assert store.outcomes == {}


@pytest.mark.asyncio
async def test_poller_exits_early_on_fetch_error_once_settled_elsewhere():
    store = _store_with_row_settled_elsewhere()
    reservation = _reservation()
    logging_obj = _logging_obj_with_reservation(reservation)
    fetch, calls = _fetch_sequence(RuntimeError("interaction deleted on another pod"))

    await poll_and_log_background_interaction_cost(
        _context(logging_obj, timeout_seconds=1.0),
        fetch_interaction=fetch,
        store=store,
    )

    assert len(calls) == 1
    assert reservation["finalized"] is True
    assert logging_obj.model_call_details.get("response_cost") is None
    assert store.outcomes == {}


@pytest.mark.asyncio
async def test_poll_task_closes_row_when_delete_settled_before_persist():
    import litellm.interactions.background_cost_polling as bg

    store = _InMemorySettlementStore()
    logging_obj = _logging_obj()
    context = _context(logging_obj)
    placeholder = asyncio.create_task(asyncio.sleep(0))
    bg._ACTIVE_POLLS[context.interaction_id] = bg._ActiveBackgroundPoll(task=placeholder, context=context)
    delete_fetch, _ = _fetch_sequence(_response("completed", with_usage=True))
    await maybe_settle_background_interaction_before_delete(
        interaction_id=context.interaction_id,
        fetch_interaction=delete_fetch,
        store=store,
    )
    bg._ACTIVE_POLLS.pop(context.interaction_id, None)
    await placeholder
    assert logging_obj.model_call_details["response_cost"] > 0
    assert store.rows == {}

    poll_fetch, poll_calls = _fetch_sequence(_response("completed", with_usage=True))
    await poll_and_log_background_interaction_cost(context, fetch_interaction=poll_fetch, store=store)

    assert poll_calls == []
    assert store.rows["interactions/bg-abc"]["status"] == "settled"
    assert store.outcomes["interactions/bg-abc"] == "billed"


@pytest.mark.asyncio
async def test_persisted_model_keeps_provider_scoped_form_when_precall_overwrites_it():
    store = _InMemorySettlementStore()
    logging_obj = _logging_obj()
    logging_obj.update_environment_variables(litellm_params={}, optional_params={}, model="gemini/gemini-2.5-flash")
    logging_obj._pre_call(input="hi", api_key=None, model="gemini-2.5-flash")
    fetch, _ = _fetch_sequence(_response("completed", with_usage=True))

    await poll_and_log_background_interaction_cost(_context(logging_obj), fetch_interaction=fetch, store=store)

    assert logging_obj.model_call_details["model"] == "gemini-2.5-flash"
    assert store.contexts["interactions/bg-abc"].model == "gemini/gemini-2.5-flash"


@pytest.mark.asyncio
async def test_delete_registry_miss_delegates_to_store():
    store = _InMemorySettlementStore()

    await maybe_settle_background_interaction_before_delete(
        interaction_id="interactions/created-on-another-pod",
        store=store,
    )

    assert store.settle_before_delete_calls == ["interactions/created-on-another-pod"]


@pytest.mark.asyncio
async def test_delete_registry_hit_records_billed_outcome_in_store():
    store = _InMemorySettlementStore()
    logging_obj = _logging_obj()
    task = _register_poll(logging_obj)
    fetch, _ = _fetch_sequence(_response("completed", with_usage=True))

    await maybe_settle_background_interaction_before_delete(
        interaction_id="interactions/bg-abc",
        fetch_interaction=fetch,
        store=store,
    )

    assert logging_obj.model_call_details["response_cost"] > 0
    assert store.outcomes["interactions/bg-abc"] == "billed"
    await asyncio.wait_for(task, timeout=5)


def test_settlement_reservation_shape_matches_budget_reservation_module():
    from litellm.interactions.background_cost_polling import SettlementReservation
    from litellm.proxy.spend_tracking.budget_reservation import _BudgetCounter, _counter_to_reservation_entry

    entry = _counter_to_reservation_entry(
        counter=_BudgetCounter(
            counter_key="spend:key:hashed-virtual-key",
            max_budget=10.0,
            fallback_spend=0.0,
            entity_type="key",
            entity_id="hashed-virtual-key",
        ),
        reserved_cost=0.05,
    )
    reservation = {"reserved_cost": 0.05, "entries": [entry], "finalized": False, "input_cost": 0.001}

    assert SettlementReservation.model_validate(reservation).model_dump() == reservation


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
