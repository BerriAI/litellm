import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import psycopg
import pytest
from psycopg import sql

from litellm.accounting import DeliveryState, DeliveryTarget, EventKind
from tests.integration._support.database import scratch_database

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
