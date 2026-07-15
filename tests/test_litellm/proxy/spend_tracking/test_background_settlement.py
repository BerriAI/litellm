import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from litellm.interactions.background_cost_polling import (
    BackgroundSettlementContext,
    SettlementKeyAttribution,
    SettlementOutcome,
    SettlementReservation,
)
from litellm.proxy.spend_tracking.background_settlement import (
    PendingSettlementRow,
    _parse_row,
    _resolve_settlement_credentials,
    rebuild_logging_for_settlement,
    settle_row_before_delete,
    sweep_pending_settlements,
)
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
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

INTERACTION_ID = "interactions/bg-foreign"


def _settlement_context(
    reservation: Optional[SettlementReservation] = None,
    model_group: Optional[str] = None,
) -> BackgroundSettlementContext:
    return BackgroundSettlementContext(
        interaction_id=INTERACTION_ID,
        custom_llm_provider="gemini",
        model="gemini-2.5-flash",
        model_group=model_group,
        litellm_call_id="original-create-call-id",
        call_type="acreate_interaction",
        attribution=SettlementKeyAttribution(
            user_api_key="hashed-virtual-key",
            user_api_key_user_id="user-123",
            user_api_key_team_id="team-456",
        ),
        budget_reservation=reservation,
    )


def _reservation() -> SettlementReservation:
    return SettlementReservation(reserved_cost=0.05, entries=[], finalized=False, input_cost=0.001)


def _row(
    context: Optional[BackgroundSettlementContext] = None,
    timeout_at: Optional[datetime] = None,
) -> PendingSettlementRow:
    return PendingSettlementRow(
        interaction_id=INTERACTION_ID,
        context=context if context is not None else _settlement_context(),
        timeout_at=timeout_at if timeout_at is not None else datetime.now(timezone.utc) + timedelta(hours=1),
    )


def _response(status: str, with_usage: bool) -> InteractionsAPIResponse:
    return InteractionsAPIResponse(
        id=INTERACTION_ID,
        model="gemini-2.5-flash",
        status=status,
        steps=[],
        usage=dict(USAGE_BLOCK) if with_usage else None,
    )


def _fetch_returning(item):
    calls = []

    async def fetch(context: BackgroundSettlementContext) -> InteractionsAPIResponse:
        calls.append(context.interaction_id)
        if isinstance(item, Exception):
            raise item
        return item

    return fetch, calls


class _FakeRowStore:
    def __init__(self, rows: tuple[PendingSettlementRow, ...]) -> None:
        self.rows = {row.interaction_id: row for row in rows}
        self.status = {row.interaction_id: "pending" for row in rows}
        self.outcomes: dict[str, SettlementOutcome] = {}
        self.claims_won = 0

    async def claim(self, interaction_id: str) -> bool:
        if self.status.get(interaction_id) != "pending":
            return False
        self.status[interaction_id] = "settled"
        self.claims_won += 1
        return True

    async def record_outcome(self, interaction_id: str, outcome: SettlementOutcome) -> None:
        self.outcomes[interaction_id] = outcome

    async def list_due(self, older_than: datetime, limit: int) -> tuple[PendingSettlementRow, ...]:
        due = tuple(row for row_id, row in self.rows.items() if self.status[row_id] == "pending")
        return due[:limit]


@pytest.mark.asyncio
async def test_rebuilt_logging_bills_with_original_attribution_and_request_id():
    logging_obj = rebuild_logging_for_settlement(_settlement_context())
    response = _response("completed", with_usage=True)

    await logging_obj.async_log_background_interaction_completion(result=response)

    assert logging_obj.model_call_details["response_cost"] > 0
    payload = get_logging_payload(
        kwargs=logging_obj.model_call_details,
        response_obj=response,
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
    )
    assert payload["request_id"] == INTERACTION_ID
    assert payload["user"] == "user-123"
    assert payload["team_id"] == "team-456"
    assert payload["api_key"]
    assert payload["spend"] > 0


def test_rebuilt_logging_carries_reservation_for_reconcile():
    context = _settlement_context(reservation=_reservation())
    logging_obj = rebuild_logging_for_settlement(context)

    from litellm.litellm_core_utils.core_helpers import get_litellm_metadata_from_kwargs

    metadata = get_litellm_metadata_from_kwargs(kwargs=logging_obj.model_call_details)
    assert metadata["user_api_key_budget_reservation"]["reserved_cost"] == 0.05
    assert metadata["user_api_key_budget_reservation"]["finalized"] is False


@pytest.mark.asyncio
async def test_sweep_bills_terminal_row_exactly_once_across_concurrent_sweeps():
    store = _FakeRowStore(rows=(_row(),))
    fetch, calls = _fetch_returning(_response("completed", with_usage=True))

    await asyncio.gather(
        sweep_pending_settlements(store=store, fetch=fetch),
        sweep_pending_settlements(store=store, fetch=fetch),
    )

    assert store.claims_won == 1
    assert store.outcomes == {INTERACTION_ID: "billed"}
    assert store.status[INTERACTION_ID] == "settled"
    assert len(calls) >= 1


@pytest.mark.asyncio
async def test_sweep_abandons_row_past_timeout_without_fetching():
    store = _FakeRowStore(rows=(_row(timeout_at=datetime.now(timezone.utc) - timedelta(seconds=1)),))
    fetch, calls = _fetch_returning(_response("completed", with_usage=True))

    await sweep_pending_settlements(store=store, fetch=fetch)

    assert calls == []
    assert store.outcomes == {INTERACTION_ID: "abandoned"}
    assert store.status[INTERACTION_ID] == "settled"


@pytest.mark.asyncio
async def test_sweep_leaves_row_pending_on_fetch_error():
    store = _FakeRowStore(rows=(_row(),))
    fetch, calls = _fetch_returning(RuntimeError("provider unavailable"))

    await sweep_pending_settlements(store=store, fetch=fetch)

    assert len(calls) == 1
    assert store.status[INTERACTION_ID] == "pending"
    assert store.outcomes == {}


@pytest.mark.asyncio
async def test_sweep_leaves_row_pending_while_interaction_still_running():
    store = _FakeRowStore(rows=(_row(),))
    fetch, calls = _fetch_returning(_response("in_progress", with_usage=False))

    await sweep_pending_settlements(store=store, fetch=fetch)

    assert len(calls) == 1
    assert store.status[INTERACTION_ID] == "pending"
    assert store.outcomes == {}


@pytest.mark.asyncio
async def test_delete_settlement_releases_when_interaction_not_terminal():
    row = _row(context=_settlement_context(reservation=_reservation()))
    store = _FakeRowStore(rows=(row,))
    fetch, _ = _fetch_returning(_response("in_progress", with_usage=False))

    await settle_row_before_delete(row=row, store=store, fetch=fetch)

    assert store.outcomes == {INTERACTION_ID: "released"}
    assert store.status[INTERACTION_ID] == "settled"


@pytest.mark.asyncio
async def test_delete_settlement_bills_terminal_interaction():
    row = _row()
    store = _FakeRowStore(rows=(row,))
    fetch, _ = _fetch_returning(_response("completed", with_usage=True))

    await settle_row_before_delete(row=row, store=store, fetch=fetch)

    assert store.outcomes == {INTERACTION_ID: "billed"}


@pytest.mark.asyncio
async def test_delete_settlement_releases_on_fetch_error():
    row = _row(context=_settlement_context(reservation=_reservation()))
    store = _FakeRowStore(rows=(row,))
    fetch, _ = _fetch_returning(RuntimeError("interaction already deleted"))

    await settle_row_before_delete(row=row, store=store, fetch=fetch)

    assert store.outcomes == {INTERACTION_ID: "released"}
    assert store.status[INTERACTION_ID] == "settled"


@pytest.mark.asyncio
async def test_delete_settlement_noop_when_claim_lost():
    row = _row()
    store = _FakeRowStore(rows=(row,))
    store.status[INTERACTION_ID] = "settled"
    fetch, _ = _fetch_returning(_response("completed", with_usage=True))

    await settle_row_before_delete(row=row, store=store, fetch=fetch)

    assert store.outcomes == {}


class _FakeRouter:
    def __init__(self, deployment: Optional[dict] = None, error: Optional[Exception] = None) -> None:
        self.deployment = deployment
        self.error = error
        self.requested_models: list[str] = []

    async def async_get_available_deployment(self, model: str, request_kwargs: dict) -> dict:
        self.requested_models.append(model)
        if self.error is not None:
            raise self.error
        assert self.deployment is not None
        return self.deployment


@pytest.mark.asyncio
async def test_credentials_resolved_from_router_deployment():
    router = _FakeRouter(deployment={"litellm_params": {"api_key": "resolved-key", "api_base": "https://resolved"}})

    credentials = await _resolve_settlement_credentials(
        _settlement_context(model_group="gemini-3-flash-preview"),
        router=router,
    )

    assert credentials.api_key == "resolved-key"
    assert credentials.api_base == "https://resolved"
    assert router.requested_models == ["gemini-3-flash-preview"]


@pytest.mark.asyncio
async def test_credentials_fall_back_to_env_when_router_lookup_fails():
    router = _FakeRouter(error=RuntimeError("no deployments available"))

    credentials = await _resolve_settlement_credentials(
        _settlement_context(model_group="gemini-3-flash-preview"),
        router=router,
    )

    assert credentials.api_key is None
    assert credentials.api_base is None


@pytest.mark.asyncio
async def test_credentials_fall_back_to_env_without_model_group():
    credentials = await _resolve_settlement_credentials(_settlement_context(model_group=None))

    assert credentials.api_key is None
    assert credentials.api_base is None


def test_parse_row_roundtrips_persisted_context_json():
    context = _settlement_context(reservation=_reservation(), model_group="gemini-3-flash-preview")
    timeout_at = datetime.now(timezone.utc)

    row = _parse_row(
        interaction_id=INTERACTION_ID,
        context_value=context.model_dump_json(),
        timeout_at=timeout_at,
    )

    assert row is not None
    assert row.context == context
    assert row.timeout_at == timeout_at


def test_parse_row_returns_none_for_corrupt_context():
    assert (
        _parse_row(
            interaction_id=INTERACTION_ID,
            context_value="{not json",
            timeout_at=datetime.now(timezone.utc),
        )
        is None
    )
