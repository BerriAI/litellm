import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pytest

import litellm.interactions.background_cost_polling as bg
from litellm.interactions.background_cost_polling import (
    _create_context,
    configure_background_settlement_store,
    maybe_settle_background_interaction_before_delete,
    PendingBackgroundInteraction,
    PollSchedule,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
from litellm.proxy.spend_tracking.background_interaction_settlement import (
    configure_background_interaction_settlement,
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
                setattr(row, column, value)
        return len(matched)

    def _matching(self, where) -> list:
        return [row for row in self.rows.values() if all(getattr(row, column) == value for column, value in where.items())]


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


def _stored_row(interaction_id: str, claimed: bool = False, create_context: Optional[object] = None) -> _Row:
    return _Row(
        interaction_id=interaction_id,
        custom_llm_provider="gemini",
        create_context=(
            create_context
            if create_context is not None
            else _create_context(_logging_obj(), "gemini").model_dump(mode="json")
        ),
        created_at=datetime.now(timezone.utc),
        claimed_at=datetime.now(timezone.utc) if claimed else None,
        claimed_by="replica-a:1" if claimed else None,
    )


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
    assert await store.unclaimed() == (pending,)


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


@pytest.mark.asyncio
async def test_unclaimed_skips_claimed_and_unreadable_rows():
    table = _FakeSettlementTable(
        rows=(
            _stored_row("interactions/bg-orphaned"),
            _stored_row("interactions/bg-settled", claimed=True),
            _stored_row("interactions/bg-from-the-future", create_context={"schema": "unknown"}),
        )
    )
    store = PrismaBackgroundSettlementStore(table=table, claimed_by="replica-b:1")

    unclaimed = await store.unclaimed()

    assert [row.interaction_id for row in unclaimed] == ["interactions/bg-orphaned"]


@pytest.mark.asyncio
async def test_record_outcome_keeps_the_audit_trail_on_the_row():
    table = _FakeSettlementTable()
    store = PrismaBackgroundSettlementStore(table=table, claimed_by="replica-a:1")
    await store.register(_pending("interactions/bg-1"))
    assert await store.claim("interactions/bg-1")

    await store.record_outcome("interactions/bg-1", "billed")

    row = table.rows["interactions/bg-1"]
    assert row.outcome == "billed"
    assert row.settled_at is not None
    assert row.claimed_at <= row.settled_at


@pytest.mark.asyncio
async def test_configure_installs_the_store_and_resumes_the_orphaned_rows():
    table = _FakeSettlementTable(
        rows=(_stored_row("interactions/bg-orphaned"), _stored_row("interactions/bg-settled", claimed=True))
    )
    fetch, captured = _capturing_fetch()
    previous_store = bg._STORE.store
    try:
        resumed = await configure_background_interaction_settlement(
            table=table, claimed_by="replica-b:1", fetch_interaction=fetch, schedule=FAST_SCHEDULE
        )

        assert len(resumed) == 1
        assert await asyncio.wait_for(resumed[0], timeout=5) == "billed"
        assert [context.interaction_id for context in captured] == ["interactions/bg-orphaned"]
        assert table.rows["interactions/bg-orphaned"].claimed_by == "replica-b:1"
        assert table.rows["interactions/bg-orphaned"].outcome == "billed"

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


@dataclass(frozen=True)
class _JsonLike:
    data: object
