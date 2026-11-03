from collections.abc import Iterator
from pathlib import Path
from typing import Final

import psycopg
import pytest

from tests.integration._support.database import scratch_database

_MIGRATIONS: Final = Path(__file__).resolve().parents[3] / "litellm-proxy-extras/litellm_proxy_extras/migrations"


@pytest.fixture
def event_database() -> Iterator[psycopg.Connection[tuple[object, ...]]]:
    with scratch_database("spendlogs_projection") as url, psycopg.connect(url, autocommit=True) as connection:
        connection.execute((_MIGRATIONS / "20261027000000_add_accounting_ledger_tables/migration.sql").read_text())
        connection.execute(
            'INSERT INTO "LiteLLM_AccountingEvent" (event_id, kind, schema_version, payload_hash, occurred_at, payload) '
            "VALUES ('event', 'outbound_usage', 1, 'hash', '2026-11-02', '{}')"
        )
        yield connection


@pytest.mark.parametrize("state", ["pending", "parked"])
def test_event_removal_refuses_unfinished_work(
    event_database: psycopg.Connection[tuple[object, ...]], state: str
) -> None:
    event_database.execute(
        'INSERT INTO "LiteLLM_AccountingDelivery" (event_id, target, version, state) '
        "VALUES ('event', 'LiteLLM_UserTable', 1, %s)",
        (state,),
    )
    with pytest.raises(psycopg.errors.RaiseException, match="Drain pending and parked"):
        event_database.execute((_MIGRATIONS / "20261103000000_remove_accounting_events/migration.sql").read_text())
    event_database.execute("ROLLBACK")
    assert event_database.execute('SELECT event_id FROM "LiteLLM_AccountingEvent"').fetchall() == [("event",)]
    assert event_database.execute('SELECT state::text FROM "LiteLLM_AccountingDelivery"').fetchall() == [(state,)]


def test_event_removal_preserves_applied_delivery_receipts(
    event_database: psycopg.Connection[tuple[object, ...]],
) -> None:
    event_database.execute(
        'INSERT INTO "LiteLLM_AccountingDelivery" (event_id, target, version, state) '
        "VALUES ('event', 'LiteLLM_UserTable', 1, 'applied')"
    )
    event_database.execute((_MIGRATIONS / "20261103000000_remove_accounting_events/migration.sql").read_text())
    assert event_database.execute("SELECT to_regclass('\"LiteLLM_AccountingEvent\"')").fetchone() == (None,)
    assert event_database.execute('SELECT event_id, state::text FROM "LiteLLM_AccountingDelivery"').fetchall() == [
        ("event", "applied")
    ]
