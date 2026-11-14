import os
from pathlib import Path
from typing import Final

import psycopg
import pytest

_MIGRATIONS: Final = Path(__file__).parents[4] / "litellm/proxy-extras/litellm_proxy_extras/migrations"
_OLD_LEDGER: Final = _MIGRATIONS / "20261027000000_add_accounting_ledger_tables/migration.sql"
_SPEND_LOG_DELIVERY: Final = _MIGRATIONS / "20261103010000_spendlogs_accounting_delivery/migration.sql"
_DROP: Final = _MIGRATIONS / "20261104010000_drop_accounting_event_tables/migration.sql"
_DROPPED: Final = (
    "AccountingDeliveryTarget",
    "AccountingEventKind",
    "LiteLLM_AccountingDelivery",
    "LiteLLM_AccountingEvent",
)


def _apply(connection: psycopg.Connection[tuple[object, ...]], migration: Path) -> None:
    with connection.transaction():
        connection.execute(migration.read_bytes())


def _present(connection: psycopg.Connection[tuple[object, ...]], names: tuple[str, ...]) -> tuple[object, ...]:
    return tuple(
        row[0]
        for row in connection.execute(
            "SELECT relname FROM pg_class WHERE relname = ANY(%s) "
            "UNION SELECT typname FROM pg_type WHERE typname = ANY(%s) ORDER BY 1",
            (list(names), list(names)),
        ).fetchall()
    )


@pytest.mark.parametrize("old_states", [(), ("applied", "pending", "parked")], ids=["empty", "with_rows"])
def test_drop_removes_the_old_ledger_and_keeps_spend_log_deliveries(old_states: tuple[str, ...]) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        _apply(connection, _OLD_LEDGER)
        connection.execute('CREATE TABLE "LiteLLM_SpendLogs" (request_id text PRIMARY KEY)')
        _apply(connection, _SPEND_LOG_DELIVERY)
        for index, state in enumerate(old_states):
            connection.execute(
                'INSERT INTO "LiteLLM_AccountingEvent" (event_id, kind, schema_version, payload_hash, occurred_at, '
                "payload) VALUES (%s, 'cost', 1, 'hash', now(), '{}')",
                (f"event-{index}",),
            )
            connection.execute(
                'INSERT INTO "LiteLLM_AccountingDelivery" (event_id, target, version, state) '
                "VALUES (%s, 'LiteLLM_UserTable', 1, %s)",
                (f"event-{index}", state),
            )
        connection.execute(
            'INSERT INTO "LiteLLM_SpendLogDelivery" (request_id, state, accounting_payload, is_batch) '
            "VALUES ('request', 'pending', '{}', false)"
        )

        _apply(connection, _DROP)
        _apply(connection, _DROP)

        assert _present(connection, _DROPPED) == ()
        assert connection.execute('SELECT request_id, state::text FROM "LiteLLM_SpendLogDelivery"').fetchall() == [
            ("request", "pending")
        ]
