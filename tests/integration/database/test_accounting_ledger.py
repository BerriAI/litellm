import asyncio
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Final

import psycopg
import pytest
import pytest_asyncio
from psycopg import sql

from litellm.accounting import (
    CostEvent,
    DeliveryState,
    DeliveryTarget,
    EventKind,
    InboundUsageEvent,
    Metric,
    MetricName,
    OutboundUsageEvent,
    Outcome,
    PricingSnapshot,
)
from litellm.proxy.db import accounting_ledger
from litellm.proxy.db.accounting_ledger import AccountingConflictError, AccountingEvent, AccountingLedger
from litellm.types.utils import CallTypes
from tests.integration._support.database import scratch_database
from tests.integration._support.prisma_clients import connected_proxy_prisma

_SCHEMA: Final = Path(__file__).resolve().parents[3] / "litellm-proxy-extras/litellm_proxy_extras/schema.prisma"
_EVENT_INSERT: Final = (
    'INSERT INTO "LiteLLM_AccountingEvent" '
    "(event_id, kind, schema_version, payload_hash, occurred_at, payload) "
    "VALUES ('event', %s, %s, 'hash', '2026-01-01', '{}') RETURNING kind::text, schema_version"
)
_DELIVERY_INSERT: Final = (
    'INSERT INTO "LiteLLM_AccountingDelivery" (event_id, target, version, state, attempts) '
    "VALUES ('event', %s, %s, %s, %s) RETURNING target::text, version, state::text, attempts"
)


@pytest.fixture(scope="module")
def ledger_database() -> Iterator[str]:
    with scratch_database("accounting") as database_url:
        result: Final = subprocess.run(
            [sys.executable, "-I", "-m", "prisma", "migrate", "deploy", "--schema", str(_SCHEMA)],
            env={**os.environ, "DATABASE_URL": database_url},
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        yield database_url


@pytest.fixture
def ledger(ledger_database: str) -> Iterator[psycopg.Connection[tuple[object, ...]]]:
    with psycopg.connect(ledger_database) as connection:
        try:
            yield connection
        finally:
            connection.rollback()


@pytest.mark.parametrize("kind", tuple(EventKind))
def test_ledger_accepts_every_public_event_kind(
    ledger: psycopg.Connection[tuple[object, ...]], kind: EventKind
) -> None:
    row: Final = ledger.execute(_EVENT_INSERT, (kind.value, 1)).fetchone()
    assert row == (kind.value, 1)


@pytest.mark.parametrize("target", tuple(DeliveryTarget))
@pytest.mark.parametrize("state", tuple(DeliveryState))
def test_delivery_accepts_every_public_target_and_state(
    ledger: psycopg.Connection[tuple[object, ...]], target: DeliveryTarget, state: DeliveryState
) -> None:
    row: Final = ledger.execute(_DELIVERY_INSERT, (target.value, 1, state.value, 0)).fetchone()
    assert row == (target.value, 1, state.value, 0)


@pytest.mark.parametrize(
    ("enum_name", "values"),
    [
        ("AccountingEventKind", tuple(value.value for value in EventKind)),
        ("AccountingDeliveryState", tuple(value.value for value in DeliveryState)),
        ("AccountingDeliveryTarget", tuple(value.value for value in DeliveryTarget)),
    ],
)
def test_migrated_enums_match_the_public_contract(
    ledger: psycopg.Connection[tuple[object, ...]], enum_name: str, values: tuple[str, ...]
) -> None:
    rows: Final = ledger.execute(
        "SELECT e.enumlabel FROM pg_enum e JOIN pg_type t ON t.oid = e.enumtypid WHERE t.typname = %s",
        (enum_name,),
    ).fetchall()
    assert frozenset(row[0] for row in rows) == frozenset(values)


@pytest.mark.parametrize("version", [0, -1])
def test_event_rejects_nonpositive_schema_versions(
    ledger: psycopg.Connection[tuple[object, ...]], version: int
) -> None:
    with pytest.raises(psycopg.errors.CheckViolation, match="AccountingEvent_schema_version_check"):
        ledger.execute(_EVENT_INSERT, (EventKind.OUTBOUND_USAGE.value, version))


@pytest.mark.parametrize(
    ("version", "attempts", "constraint"),
    [
        (0, 0, "AccountingDelivery_version_check"),
        (-1, 0, "AccountingDelivery_version_check"),
        (1, -1, "AccountingDelivery_attempts_check"),
    ],
)
def test_delivery_rejects_invalid_versions_and_attempts(
    ledger: psycopg.Connection[tuple[object, ...]], version: int, attempts: int, constraint: str
) -> None:
    with pytest.raises(psycopg.errors.CheckViolation, match=constraint):
        ledger.execute(_DELIVERY_INSERT, (DeliveryTarget.KEY_SPEND.value, version, "pending", attempts))


@pytest.mark.parametrize("column", ["state", "target"])
def test_delivery_rejects_unknown_enum_values(ledger: psycopg.Connection[tuple[object, ...]], column: str) -> None:
    ledger.execute(_DELIVERY_INSERT, (DeliveryTarget.KEY_SPEND.value, 1, "pending", 0))
    with pytest.raises(psycopg.errors.InvalidTextRepresentation, match="invalid input value for enum"):
        ledger.execute(
            sql.SQL('UPDATE "LiteLLM_AccountingDelivery" SET {} = %s').format(sql.Identifier(column)), ("invalid",)
        )


def test_event_rejects_unknown_kind(ledger: psycopg.Connection[tuple[object, ...]]) -> None:
    with pytest.raises(psycopg.errors.InvalidTextRepresentation, match="invalid input value for enum"):
        ledger.execute(_EVENT_INSERT, ("invalid", 1))


def test_new_delivery_defaults_to_pending_with_no_attempts(ledger: psycopg.Connection[tuple[object, ...]]) -> None:
    row: Final = ledger.execute(
        'INSERT INTO "LiteLLM_AccountingDelivery" (event_id, target, version) '
        "VALUES ('event', %s, 1) RETURNING state::text, attempts",
        (DeliveryTarget.KEY_SPEND.value,),
    ).fetchone()
    assert row == ("pending", 0)


_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)
_CUTOFF = datetime(2026, 2, 1, tzinfo=timezone.utc)
_TARGETS = (DeliveryTarget.USER_SPEND, DeliveryTarget.TEAM_SPEND)


def _usage(event_id: str = "usage", *, inbound: bool = False) -> InboundUsageEvent | OutboundUsageEvent:
    fields = dict(
        event_id=event_id,
        request_id="request",
        call_type=CallTypes.completion,
        started_at=_AT,
        ended_at=_AT,
        outcome=Outcome.SUCCEEDED,
        dimensions=(),
    )
    if inbound:
        return InboundUsageEvent(**fields, metrics=(Metric(name=MetricName.REQUESTS, value=Decimal(1)),))
    return OutboundUsageEvent(**fields, attempt_id=event_id, retry_of_attempt_id=None, metrics=None)


def _cost(source: str = "usage", event_id: str = "cost", amount: str = "0.125") -> CostEvent:
    return CostEvent(
        event_id=event_id,
        usage_event_id=source,
        occurred_at=_AT,
        dimensions=(),
        charges=(Metric(name=MetricName.SPEND, value=Decimal(amount), unit="USD"),),
        snapshot=PricingSnapshot(
            calculator_version="fixture",
            model="fixture",
            provider=None,
            pricing_source="cost_map",
            priced_model_key="fixture",
            total_cost=Decimal(amount),
        ),
    )


def _envelope(event: AccountingEvent, **context: object) -> str:
    return json.dumps({"event": event.model_dump(mode="json"), "context": context})


async def _insert(
    store: AccountingLedger, event: AccountingEvent, targets: tuple[DeliveryTarget, ...] = _TARGETS, **context: object
) -> str:
    return await store.insert_event(event, targets, envelope=_envelope(event, **context))


def _rows(database_url: str, statement: str, parameters: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    with psycopg.connect(database_url) as connection:
        cursor = connection.execute(statement, parameters)
        return cursor.fetchall() if cursor.description is not None else []


def _complete_and_age(database_url: str) -> None:
    with psycopg.connect(database_url) as connection:
        connection.execute('UPDATE "LiteLLM_AccountingEvent" SET recorded_at = %s', (_AT,))
        connection.execute("UPDATE \"LiteLLM_AccountingDelivery\" SET state = 'applied', applied_at = %s", (_AT,))


@pytest_asyncio.fixture(loop_scope="function")
async def writer(ledger_database, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", ledger_database)
    monkeypatch.delenv("DATABASE_URL_READ_REPLICA", raising=False)
    _rows(ledger_database, 'TRUNCATE "LiteLLM_AccountingDelivery", "LiteLLM_AccountingEvent"')
    async with connected_proxy_prisma(ledger_database) as client:
        yield client, AccountingLedger.for_client(client)


@pytest.mark.asyncio
async def test_event_and_complete_delivery_set_commit_together(writer, ledger_database):
    _, store = writer
    event = _usage()
    assert await _insert(store, event, requested_region="eu", quantity="1.25") == "inserted"
    rows = _rows(ledger_database, 'SELECT kind::text, source_event_id, payload FROM "LiteLLM_AccountingEvent"')
    assert rows == [("outbound_usage", None, json.loads(_envelope(event, requested_region="eu", quantity="1.25")))]
    deliveries = _rows(ledger_database, 'SELECT target::text, state::text, attempts FROM "LiteLLM_AccountingDelivery"')
    assert set(deliveries) == {(target.value, "pending", 0) for target in _TARGETS}


@pytest.mark.asyncio
async def test_concurrent_duplicates_preserve_completed_deliveries(writer, ledger_database):
    _, store = writer
    event = _usage(inbound=True)
    results = await asyncio.gather(*(_insert(store, event) for _ in range(8)))
    assert results.count("inserted") == 1 and results.count("duplicate") == 7
    _complete_and_age(ledger_database)
    before = _rows(ledger_database, 'SELECT * FROM "LiteLLM_AccountingDelivery" ORDER BY target')
    reordered = json.dumps(json.loads(_envelope(event)), sort_keys=True, indent=2)
    assert await store.insert_event(event, tuple(reversed(_TARGETS)) + _TARGETS, envelope=reordered) == "duplicate"
    assert _rows(ledger_database, 'SELECT * FROM "LiteLLM_AccountingDelivery" ORDER BY target') == before
    assert _rows(ledger_database, 'SELECT count(*) FROM "LiteLLM_AccountingEvent"') == [(1,)]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["payload", "context", "targets", "schema_version", "version"])
async def test_conflicting_acceptance_cannot_overwrite_or_add_work(writer, ledger_database, change):
    _, store = writer
    event = _usage()
    await _insert(store, event)
    altered = event.model_copy(update={"outcome": Outcome.FAILED}) if change == "payload" else event
    targets = (DeliveryTarget.KEY_SPEND,) if change == "targets" else _TARGETS
    context = {"override": "0.5"} if change == "context" else {}
    options = {change: 2} if change in ("schema_version", "version") else {}
    with pytest.raises(AccountingConflictError, match="conflicts"):
        await store.insert_event(altered, targets, envelope=_envelope(altered, **context), **options)
    assert _rows(ledger_database, 'SELECT payload FROM "LiteLLM_AccountingEvent"') == [(json.loads(_envelope(event)),)]
    assert set(_rows(ledger_database, 'SELECT target::text FROM "LiteLLM_AccountingDelivery"')) == {
        (t.value,) for t in _TARGETS
    }


@pytest.mark.asyncio
async def test_competing_cost_identities_accept_only_one_authoritative_charge(writer, ledger_database):
    _, store = writer
    await _insert(store, _usage())
    results = await asyncio.gather(
        _insert(store, _cost()), _insert(store, _cost(event_id="other-cost")), return_exceptions=True
    )
    assert sum(result == "inserted" for result in results) == 1
    assert sum(isinstance(result, AccountingConflictError) for result in results) == 1
    accepted = _rows(ledger_database, "SELECT event_id, payload FROM \"LiteLLM_AccountingEvent\" WHERE kind = 'cost'")
    assert len(accepted) == 1
    event = CostEvent.model_validate_json(json.dumps(accepted[0][1]["event"]))
    assert await _insert(store, event) == "duplicate"
    assert event.snapshot.total_cost == Decimal("0.125")
    with pytest.raises(AccountingConflictError):
        await _insert(store, _cost(event_id=event.event_id, amount="0.5"))


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["missing", "inbound", "cost"])
async def test_cost_requires_an_existing_outbound_source(writer, ledger_database, source):
    _, store = writer
    if source == "inbound":
        await _insert(store, _usage("source", inbound=True))
    if source == "cost":
        await _insert(store, _usage())
        await _insert(store, _cost(event_id="source"))
    with pytest.raises(AccountingConflictError, match="outbound usage"):
        await _insert(store, _cost(source="source", event_id="invalid"))
    assert _rows(ledger_database, "SELECT count(*) FROM \"LiteLLM_AccountingEvent\" WHERE event_id = 'invalid'") == [
        (0,)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["before_commit", "after_commit", "cancellation"])
async def test_interruption_rolls_back_or_retry_recognizes_commit(writer, ledger_database, failure):
    client, store = writer

    @asynccontextmanager
    async def interrupted():
        async with client.writer_db.tx() as tx:
            yield tx
            if failure == "before_commit":
                raise RuntimeError("injected pre-commit failure")
            if failure == "cancellation":
                raise asyncio.CancelledError
        raise RuntimeError("injected lost acknowledgement")

    error = asyncio.CancelledError if failure == "cancellation" else RuntimeError
    with pytest.raises(error):
        await _insert(AccountingLedger(interrupted), _usage())
    expected = 1 if failure == "after_commit" else 0
    assert _rows(ledger_database, 'SELECT count(*) FROM "LiteLLM_AccountingEvent"') == [(expected,)]
    assert _rows(ledger_database, 'SELECT count(*) FROM "LiteLLM_AccountingDelivery"') == [(expected * len(_TARGETS),)]
    assert await _insert(store, _usage()) == ("duplicate" if expected else "inserted")


@pytest.mark.asyncio
async def test_delivery_insert_failure_rolls_back_event(writer, ledger_database):
    from prisma.errors import RawQueryError

    _, store = writer
    with psycopg.connect(ledger_database) as connection:
        connection.execute(
            'ALTER TABLE "LiteLLM_AccountingDelivery" ADD CONSTRAINT reject_delivery CHECK (version < 1)'
        )
    try:
        with pytest.raises(RawQueryError, match="reject_delivery"):
            await _insert(store, _usage())
        assert _rows(ledger_database, 'SELECT count(*) FROM "LiteLLM_AccountingEvent"') == [(0,)]
    finally:
        with psycopg.connect(ledger_database) as connection:
            connection.execute('ALTER TABLE "LiteLLM_AccountingDelivery" DROP CONSTRAINT reject_delivery')


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked", ["pending", "parked", "unpriced", "recent_cost", "recovery"])
async def test_retention_preserves_incomplete_or_recovery_needed_families(writer, ledger_database, blocked):
    _, store = writer
    await _insert(store, _usage())
    if blocked != "unpriced":
        await _insert(store, _cost())
    _complete_and_age(ledger_database)
    with psycopg.connect(ledger_database) as connection:
        if blocked in ("pending", "parked"):
            connection.execute(
                "UPDATE \"LiteLLM_AccountingDelivery\" SET state = %s WHERE event_id = 'cost'", (blocked,)
            )
        if blocked == "recent_cost":
            connection.execute(
                "UPDATE \"LiteLLM_AccountingEvent\" SET recorded_at = %s WHERE event_id = 'cost'", (_CUTOFF,)
            )
    before = _rows(ledger_database, 'SELECT event_id FROM "LiteLLM_AccountingEvent" ORDER BY event_id')
    assert (
        await store.purge_completed(
            redelivery_before=_CUTOFF, recovery_before=_AT if blocked == "recovery" else _CUTOFF
        )
        == 0
    )
    assert _rows(ledger_database, 'SELECT event_id FROM "LiteLLM_AccountingEvent" ORDER BY event_id') == before


@pytest.mark.asyncio
async def test_retention_removes_completed_family_and_receipts_together(writer, ledger_database):
    _, store = writer
    await _insert(store, _usage())
    await _insert(store, _cost())
    _complete_and_age(ledger_database)
    assert await store.purge_completed(redelivery_before=_CUTOFF, recovery_before=_CUTOFF) == 2
    assert _rows(ledger_database, 'SELECT count(*) FROM "LiteLLM_AccountingEvent"') == [(0,)]
    assert _rows(ledger_database, 'SELECT count(*) FROM "LiteLLM_AccountingDelivery"') == [(0,)]


@pytest.mark.asyncio
async def test_retention_respects_batch_bound_and_skips_locked_families(writer, ledger_database):
    client, store = writer
    for name in ("a", "b", "c"):
        await _insert(store, _usage(name, inbound=True))
    _complete_and_age(ledger_database)
    async with client.writer_db.tx() as tx:
        await tx.query_raw("SELECT event_id FROM \"LiteLLM_AccountingEvent\" WHERE event_id = 'a' FOR UPDATE")
        assert await store.purge_completed(redelivery_before=_CUTOFF, recovery_before=_CUTOFF, limit=1) == 1
        assert _rows(ledger_database, 'SELECT event_id FROM "LiteLLM_AccountingEvent" ORDER BY event_id') == [
            ("a",),
            ("c",),
        ]
    assert await store.purge_completed(redelivery_before=_CUTOFF, recovery_before=_CUTOFF) == 2


@pytest.mark.asyncio
async def test_retention_makes_progress_past_pending_and_unpriced_events(writer, ledger_database):
    _, store = writer
    await _insert(store, _usage("a", inbound=True))
    await _insert(store, _usage("b"))
    await _insert(store, _usage("c", inbound=True))
    _complete_and_age(ledger_database)
    _rows(ledger_database, "UPDATE \"LiteLLM_AccountingDelivery\" SET state = 'pending' WHERE event_id = 'a'")
    assert await store.purge_completed(redelivery_before=_CUTOFF, recovery_before=_CUTOFF, limit=1) == 1
    assert _rows(ledger_database, 'SELECT event_id FROM "LiteLLM_AccountingEvent" ORDER BY event_id') == [
        ("a",),
        ("b",),
    ]


@pytest.mark.asyncio
async def test_zero_cost_and_empty_delivery_sets_are_durable(writer, ledger_database):
    _, store = writer
    await _insert(store, _usage(), targets=())
    event = _cost(amount="0")
    assert await _insert(store, event, targets=()) == "inserted"
    assert await _insert(store, event, targets=()) == "duplicate"
    assert _rows(ledger_database, 'SELECT count(*) FROM "LiteLLM_AccountingDelivery"') == [(0,)]
    _complete_and_age(ledger_database)
    assert await store.purge_completed(redelivery_before=_CUTOFF, recovery_before=_CUTOFF) == 2


@pytest.mark.asyncio
async def test_retention_cannot_delete_source_while_cost_acceptance_is_uncommitted(writer, ledger_database):
    client, store = writer
    await _insert(store, _usage())
    _complete_and_age(ledger_database)
    inserted = asyncio.Event()
    release = asyncio.Event()

    @asynccontextmanager
    async def paused_commit():
        async with client.writer_db.tx() as tx:
            yield tx
            inserted.set()
            await release.wait()

    task = asyncio.create_task(_insert(AccountingLedger(paused_commit), _cost()))
    try:
        await asyncio.wait_for(inserted.wait(), timeout=5)
        assert await store.purge_completed(redelivery_before=_CUTOFF, recovery_before=_CUTOFF) == 0
    finally:
        release.set()
        await task
    assert _rows(ledger_database, 'SELECT event_id FROM "LiteLLM_AccountingEvent" ORDER BY event_id') == [
        ("cost",),
        ("usage",),
    ]


@pytest.mark.asyncio
async def test_retention_failure_rolls_back_event_and_receipt_deletion(writer, ledger_database):
    client, store = writer
    await _insert(store, _usage(inbound=True))
    _complete_and_age(ledger_database)

    @asynccontextmanager
    async def interrupted():
        async with client.writer_db.tx() as tx:
            yield tx
            raise RuntimeError("injected cleanup failure")

    with pytest.raises(RuntimeError, match="cleanup failure"):
        await AccountingLedger(interrupted).purge_completed(redelivery_before=_CUTOFF, recovery_before=_CUTOFF)
    assert _rows(ledger_database, 'SELECT count(*) FROM "LiteLLM_AccountingEvent"') == [(1,)]
    assert _rows(ledger_database, 'SELECT count(*) FROM "LiteLLM_AccountingDelivery"') == [(len(_TARGETS),)]
    assert await store.purge_completed(redelivery_before=_CUTOFF, recovery_before=_CUTOFF) == 1


@pytest.mark.asyncio
async def test_ledger_reads_and_writes_use_primary_when_replica_is_configured(writer, ledger_database, monkeypatch):
    with scratch_database("replica") as replica_url:
        monkeypatch.setenv("DATABASE_URL_READ_REPLICA", replica_url)
        async with connected_proxy_prisma(ledger_database) as client:
            store = AccountingLedger.for_client(client)
            await _insert(store, _usage())
            assert await _insert(store, _usage()) == "duplicate"
            assert await _insert(store, _cost()) == "inserted"
            _complete_and_age(ledger_database)
            assert await store.purge_completed(redelivery_before=_CUTOFF, recovery_before=_CUTOFF) == 2
        assert _rows(replica_url, "SELECT to_regclass('public.\"LiteLLM_AccountingEvent\"')") == [(None,)]


def _index_conditions(plan: object) -> list[tuple[object, object]]:
    if isinstance(plan, dict):
        own = [(plan.get("Index Name"), plan.get("Index Cond"))] if "Index Name" in plan else []
        children = [plan.get("Plan"), *plan.get("Plans", [])]
        return own + [pair for child in children if child is not None for pair in _index_conditions(child)]
    return [pair for item in plan for pair in _index_conditions(item)] if isinstance(plan, list) else []


@pytest.mark.parametrize("statement", [accounting_ledger._DELETE_EVENTS, accounting_ledger._DELETE_DELIVERIES])
def test_family_cleanup_lookups_use_the_kind_source_index(ledger_database: str, statement: str) -> None:
    explained = "EXPLAIN (FORMAT JSON) " + statement.replace("$1", "%s")
    with psycopg.connect(ledger_database) as connection:
        connection.execute("SET enable_seqscan = off")
        plan = connection.execute(explained, ("usage",) * statement.count("$1")).fetchone()
    assert plan is not None
    conditions = _index_conditions(plan[0])
    assert any(
        name == "LiteLLM_AccountingEvent_kind_source_event_id_idx" and "source_event_id" in str(condition)
        for name, condition in conditions
    ), conditions
