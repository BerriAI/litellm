import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

import litellm.interactions.background_cost_polling as bg
from litellm import Router
from litellm.constants import BACKGROUND_INTERACTION_SETTLEMENT_CLAIM_LEASE_SECONDS
from litellm.interactions.background_cost_polling import (
    _create_context,
    BackgroundInteractionPollContext,
    configure_background_settlement_store,
    maybe_settle_background_interaction_before_delete,
    PendingBackgroundInteraction,
    PollSchedule,
    resume_unsettled_background_interactions,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
from litellm.proxy.spend_tracking.background_interaction_settlement import (
    configure_background_interaction_settlement,
    fetch_with_deployment_credentials,
    install_background_interaction_settlement,
    PrismaBackgroundSettlementStore,
)
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

FAST_SCHEDULE = PollSchedule(initial_interval_seconds=0.001, max_interval_seconds=0.002, timeout_seconds=1.0)


@dataclass
class _Row:
    interaction_id: str
    custom_llm_provider: str
    create_context: object
    created_at: datetime
    claimed_at: Optional[datetime] = None
    claimed_by: Optional[str] = None
    settled_at: Optional[datetime] = None
    outcome: Optional[str] = None


class _FakeSettlementTable:
    """Just enough of prisma's per-model actions: Json is stored as the data it wraps and read back parsed."""

    def __init__(self, rows: tuple[_Row, ...] = ()):
        self.rows = {row.interaction_id: row for row in rows}

    async def create(self, *, data):
        row = _Row(
            interaction_id=data["interaction_id"],
            custom_llm_provider=data["custom_llm_provider"],
            create_context=data["create_context"].data,
            created_at=data["created_at"],
        )
        self.rows[row.interaction_id] = row
        return row

    async def find_unique(self, *, where):
        return self.rows.get(where["interaction_id"])

    async def find_many(self, *, where):
        return self._matching(where)

    async def update_many(self, *, data, where):
        matched = self._matching(where)
        for row in matched:
            for column, value in data.items():
                setattr(row, column, getattr(value, "data", value) if column == "create_context" else value)
        return len(matched)

    def _matching(self, where) -> list:
        return [row for row in self.rows.values() if _matches(row, where)]


def _matches(row: "_Row", where: dict) -> bool:
    return all(_satisfies(row, column, condition) for column, condition in where.items())


def _satisfies(row: "_Row", column: str, condition) -> bool:
    if column == "OR":
        return any(_matches(row, alternative) for alternative in condition)
    value = getattr(row, column)
    if isinstance(condition, dict):
        return value is not None and value < condition["lt"]
    return value == condition


def _logging_obj(metadata: Optional[dict] = None) -> LitellmLogging:
    logging_obj = LitellmLogging(
        model="gemini-2.5-flash",
        messages=[],
        stream=False,
        call_type="acreate_interaction",
        start_time=time.time(),
        litellm_call_id="bg-settlement-call-id",
        function_id="bg-settlement-fn-id",
    )
    logging_obj.update_environment_variables(
        litellm_params={"metadata": metadata or {"user_api_key": "0123456789abcdef" * 4}},
        optional_params={},
        model="gemini-2.5-flash",
        custom_llm_provider="gemini",
        input="hi",
    )
    return logging_obj


def _pending(interaction_id: str) -> PendingBackgroundInteraction:
    return PendingBackgroundInteraction(
        interaction_id=interaction_id,
        custom_llm_provider="gemini",
        create_context=_create_context(_logging_obj(), "gemini"),
        created_at=datetime.now(timezone.utc),
    )


def _stored_row(
    interaction_id: str,
    claimed_at: Optional[datetime] = None,
    settled: bool = False,
    create_context: Optional[object] = None,
) -> _Row:
    return _Row(
        interaction_id=interaction_id,
        custom_llm_provider="gemini",
        create_context=(
            create_context
            if create_context is not None
            else _create_context(_logging_obj(), "gemini").model_dump(mode="json")
        ),
        created_at=datetime.now(timezone.utc),
        claimed_at=claimed_at,
        claimed_by="replica-a:1" if claimed_at is not None else None,
        settled_at=claimed_at if settled else None,
        outcome="billed" if settled else None,
    )


def _minutes_ago(minutes: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(minutes=minutes)


def _past_the_default_lease() -> datetime:
    return datetime.now(timezone.utc) - timedelta(seconds=BACKGROUND_INTERACTION_SETTLEMENT_CLAIM_LEASE_SECONDS + 60)


def _completed(interaction_id: str) -> InteractionsAPIResponse:
    return InteractionsAPIResponse(
        id=interaction_id, model="gemini-2.5-flash", status="completed", steps=[], usage=dict(USAGE_BLOCK)
    )


def _capturing_fetch():
    captured = []

    async def fetch(context):
        captured.append(context)
        return _completed(context.interaction_id)

    return fetch, captured


@pytest.mark.asyncio
async def test_registered_row_reads_back_as_the_same_pending_interaction():
    table = _FakeSettlementTable()
    store = PrismaBackgroundSettlementStore(table=table, claimed_by="replica-a:1")
    pending = _pending("interactions/bg-1")

    await store.register(pending)

    assert await store.pending("interactions/bg-1") == pending
    assert await store.unsettled() == (pending,)


@pytest.mark.asyncio
async def test_claim_is_won_by_exactly_one_settler():
    table = _FakeSettlementTable()
    replica_a = PrismaBackgroundSettlementStore(table=table, claimed_by="replica-a:1")
    replica_b = PrismaBackgroundSettlementStore(table=table, claimed_by="replica-b:1")
    await replica_a.register(_pending("interactions/bg-1"))

    assert await replica_b.claim("interactions/bg-1") is True
    assert await replica_a.claim("interactions/bg-1") is False
    assert await replica_a.is_claimed("interactions/bg-1") is True
    assert await replica_a.pending("interactions/bg-1") is None
    assert table.rows["interactions/bg-1"].claimed_by == "replica-b:1"


class _MissingSettlementTable:
    """Prisma's per-model actions against a database whose migration for this table was held back."""

    async def create(self, *, data):
        raise self._missing()

    async def find_unique(self, *, where):
        raise self._missing()

    async def find_many(self, *, where):
        raise self._missing()

    async def update_many(self, *, data, where):
        raise self._missing()

    def _missing(self):
        from prisma.errors import TableNotFoundError

        return TableNotFoundError(
            {
                "user_facing_error": {
                    "error_code": "P2021",
                    "meta": {"table": "public.LiteLLM_BackgroundInteractionSettlement"},
                    "message": "The table does not exist in the current database.",
                }
            }
        )


@pytest.mark.asyncio
async def test_a_missing_table_holds_no_rows_and_takes_no_registration():
    from prisma.errors import TableNotFoundError

    store = PrismaBackgroundSettlementStore(table=_MissingSettlementTable(), claimed_by="replica-a:1")

    with pytest.raises(TableNotFoundError):
        await store.register(_pending("interactions/bg-1"))
    assert await store.pending("interactions/bg-1") is None
    assert await store.is_claimed("interactions/bg-1") is False
    assert await store.claim("interactions/bg-1") is False
    with pytest.raises(TableNotFoundError):
        await store.unsettled()


@pytest.mark.asyncio
async def test_unsettled_lists_claimed_rows_with_no_outcome_and_when_their_lease_ends():
    claimed_at = _minutes_ago(1)
    table = _FakeSettlementTable(
        rows=(
            _stored_row("interactions/bg-orphaned"),
            _stored_row("interactions/bg-mid-settlement", claimed_at=claimed_at),
            _stored_row("interactions/bg-settled", claimed_at=claimed_at, settled=True),
            _stored_row("interactions/bg-from-the-future", create_context={"schema": "unknown"}),
        )
    )
    store = PrismaBackgroundSettlementStore(table=table, claimed_by="replica-b:1", claim_lease_seconds=300)

    unsettled = await store.unsettled()

    assert [(row.interaction_id, row.claim_expires_at) for row in unsettled] == [
        ("interactions/bg-orphaned", None),
        ("interactions/bg-mid-settlement", claimed_at + timedelta(seconds=300)),
    ]


@pytest.mark.asyncio
async def test_a_claim_whose_settler_died_is_taken_over_once_its_lease_runs_out():
    """
    The regression: a settler that died between its claim and its outcome left
    the row claimed forever, so no replica, restart, or delete ever billed it.
    """
    table = _FakeSettlementTable(
        rows=(
            _stored_row("interactions/bg-dead-settler", claimed_at=_minutes_ago(10)),
            _stored_row("interactions/bg-settled", claimed_at=_minutes_ago(10), settled=True),
        )
    )
    replica_b = PrismaBackgroundSettlementStore(table=table, claimed_by="replica-b:1", claim_lease_seconds=300)
    replica_c = PrismaBackgroundSettlementStore(table=table, claimed_by="replica-c:1", claim_lease_seconds=300)

    assert await replica_b.is_claimed("interactions/bg-dead-settler") is False
    assert await replica_b.pending("interactions/bg-dead-settler") is not None
    assert await replica_b.claim("interactions/bg-dead-settler") is True
    assert await replica_c.claim("interactions/bg-dead-settler") is False
    assert await replica_c.is_claimed("interactions/bg-dead-settler") is True
    assert table.rows["interactions/bg-dead-settler"].claimed_by == "replica-b:1"

    assert await replica_b.is_claimed("interactions/bg-settled") is True
    assert await replica_b.pending("interactions/bg-settled") is None
    assert await replica_b.claim("interactions/bg-settled") is False
    assert table.rows["interactions/bg-settled"].claimed_by == "replica-a:1"


@pytest.mark.asyncio
async def test_record_outcome_keeps_the_audit_trail_and_drops_the_stored_request_context():
    table = _FakeSettlementTable()
    store = PrismaBackgroundSettlementStore(table=table, claimed_by="replica-a:1")
    await store.register(_pending("interactions/bg-1"))
    assert await store.claim("interactions/bg-1")
    assert table.rows["interactions/bg-1"].create_context

    await store.record_outcome("interactions/bg-1", "billed")

    row = table.rows["interactions/bg-1"]
    assert row.outcome == "billed"
    assert row.settled_at is not None
    assert row.claimed_at <= row.settled_at
    assert row.create_context == {}


@pytest.mark.asyncio
async def test_configure_installs_the_store_and_resumes_the_orphaned_rows():
    table = _FakeSettlementTable(
        rows=(
            _stored_row("interactions/bg-orphaned"),
            _stored_row("interactions/bg-dead-settler", claimed_at=_past_the_default_lease()),
            _stored_row("interactions/bg-settled", claimed_at=_minutes_ago(1), settled=True),
        )
    )
    fetch, captured = _capturing_fetch()
    previous_store = bg._STORE.store
    try:
        resumed = await configure_background_interaction_settlement(
            table=table, claimed_by="replica-b:1", fetch_interaction=fetch, schedule=FAST_SCHEDULE
        )

        assert [await asyncio.wait_for(task, timeout=5) for task in resumed] == ["billed", "billed"]
        assert sorted(context.interaction_id for context in captured) == [
            "interactions/bg-dead-settler",
            "interactions/bg-orphaned",
        ]
        for interaction_id in ("interactions/bg-orphaned", "interactions/bg-dead-settler"):
            assert table.rows[interaction_id].claimed_by == "replica-b:1"
            assert table.rows[interaction_id].outcome == "billed"
        assert table.rows["interactions/bg-settled"].claimed_by == "replica-a:1"

        await table.create(
            data={
                "interaction_id": "interactions/bg-created-elsewhere",
                "custom_llm_provider": "gemini",
                "create_context": _JsonLike(_create_context(_logging_obj(), "gemini").model_dump(mode="json")),
                "created_at": datetime.now(timezone.utc),
            }
        )
        outcome = await maybe_settle_background_interaction_before_delete(
            interaction_id="interactions/bg-created-elsewhere", delete_kwargs={}, fetch_interaction=fetch
        )

        assert outcome == "billed"
        assert table.rows["interactions/bg-created-elsewhere"].claimed_by == "replica-b:1"
    finally:
        configure_background_settlement_store(previous_store)


class _PrismaClientWithoutSettlementTable:
    pass


@pytest.mark.asyncio
async def test_install_keeps_booting_when_the_settlement_table_is_unreachable():
    previous_store = bg._STORE.store

    await install_background_interaction_settlement(_PrismaClientWithoutSettlementTable(), router=lambda: None)

    assert bg._STORE.store is previous_store


@dataclass(frozen=True)
class _JsonLike:
    data: object


@pytest.mark.asyncio
async def test_a_resumed_row_still_under_a_live_claim_waits_for_the_lease_before_settling():
    claimed_at = datetime.now(timezone.utc)
    table = _FakeSettlementTable(rows=(_stored_row("interactions/bg-mid-settlement", claimed_at=claimed_at),))
    store = PrismaBackgroundSettlementStore(table=table, claimed_by="replica-b:1", claim_lease_seconds=0.2)
    fetched_at = []

    async def fetch(context):
        fetched_at.append(datetime.now(timezone.utc))
        return _completed(context.interaction_id)

    (resumed,) = await resume_unsettled_background_interactions(store, fetch, schedule=FAST_SCHEDULE)

    assert await asyncio.wait_for(resumed, timeout=5) == "billed"
    assert fetched_at[0] >= claimed_at + timedelta(seconds=0.2)
    assert table.rows["interactions/bg-mid-settlement"].claimed_by == "replica-b:1"


def _router_with_a_deployment_scoped_key() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "gemini-background",
                "litellm_params": {
                    "model": "gemini/gemini-2.5-flash",
                    "api_key": "deployment-scoped-key",
                    "api_base": "https://deployment-scoped.example",
                },
                "model_info": {"id": "deployment-1"},
            }
        ]
    )


def _create_context_served_by(deployment_id: str) -> dict:
    metadata = {"user_api_key": "0123456789abcdef" * 4, "model_info": {"id": deployment_id}}
    return _create_context(_logging_obj(metadata), "gemini").model_dump(mode="json")


@pytest.mark.asyncio
async def test_a_resumed_poll_fetches_with_the_credentials_of_the_deployment_that_served_the_create():
    """
    The regression: a resumed poll has no request to take credentials from, so
    it fetched with the environment's key, which a deployment-scoped key is
    not, and every fetch failed until the row was given up unbilled.
    """
    table = _FakeSettlementTable(
        rows=(_stored_row("interactions/bg-1", create_context=_create_context_served_by("deployment-1")),)
    )
    router = _router_with_a_deployment_scoped_key()
    fetch, captured = _capturing_fetch()
    previous_store = bg._STORE.store
    try:
        (resumed,) = await configure_background_interaction_settlement(
            table=table,
            claimed_by="replica-b:1",
            fetch_interaction=fetch_with_deployment_credentials(lambda: router, fetch),
            schedule=FAST_SCHEDULE,
        )

        assert await asyncio.wait_for(resumed, timeout=5) == "billed"
        assert (captured[0].api_key, captured[0].api_base) == (
            "deployment-scoped-key",
            "https://deployment-scoped.example",
        )
    finally:
        configure_background_settlement_store(previous_store)


@pytest.mark.parametrize(
    "router, deployment_id",
    [
        (_router_with_a_deployment_scoped_key, "deployment-removed-since"),
        (_router_with_a_deployment_scoped_key, None),
        (lambda: None, "deployment-1"),
    ],
)
@pytest.mark.asyncio
async def test_a_resumed_poll_without_a_known_deployment_fetches_with_the_environment_credentials(
    router, deployment_id
):
    live_router = router()
    fetch, captured = _capturing_fetch()
    context = BackgroundInteractionPollContext(
        interaction_id="interactions/bg-1",
        custom_llm_provider="gemini",
        logging_obj=_logging_obj(),
        deployment_id=deployment_id,
    )

    await fetch_with_deployment_credentials(lambda: live_router, fetch)(context)

    assert captured == [context]
