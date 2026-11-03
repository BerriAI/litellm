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


@pytest.mark.parametrize("entrypoint", ["proxy", "extras_v1", "extras_v2"])
def test_schema_push_cannot_bypass_unfinished_event_guard(
    event_database: psycopg.Connection[tuple[object, ...]], monkeypatch: pytest.MonkeyPatch, entrypoint: str
) -> None:
    import os
    from urllib.parse import urlsplit, urlunsplit
    from litellm.proxy.db.prisma_client import PrismaManager
    from litellm_proxy_extras.utils import ProxyExtrasDBManager

    event_database.execute(
        'INSERT INTO "LiteLLM_AccountingDelivery" (event_id, target, version, state) '
        "VALUES ('event', 'LiteLLM_UserTable', 1, 'pending')"
    )
    url = urlunsplit(urlsplit(os.environ["DATABASE_URL"])._replace(path=f"/{event_database.info.dbname}"))
    monkeypatch.setenv("DATABASE_URL", url)
    setup_database = PrismaManager.setup_database if entrypoint == "proxy" else ProxyExtrasDBManager.setup_database
    with pytest.raises(RuntimeError, match="accounting events"):
        setup_database(use_migrate=False, use_v2_resolver=entrypoint == "extras_v2")
    assert event_database.execute('SELECT event_id FROM "LiteLLM_AccountingEvent"').fetchall() == [("event",)]
    assert event_database.execute('SELECT state::text FROM "LiteLLM_AccountingDelivery"').fetchall() == [("pending",)]


@pytest.mark.parametrize("interrupt_before_commit", [False, True])
def test_prisma_deploy_preserves_event_removal_transaction(
    event_database: psycopg.Connection[tuple[object, ...]], tmp_path: Path, interrupt_before_commit: bool
) -> None:
    import os
    import subprocess
    import sys
    from urllib.parse import urlsplit, urlunsplit

    url = urlunsplit(urlsplit(os.environ["DATABASE_URL"])._replace(path=f"/{event_database.info.dbname}"))
    schema = tmp_path / "schema.prisma"
    schema.write_text('datasource db {\n  provider = "postgresql"\n  url = env("DATABASE_URL")\n}\n')
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "migration_lock.toml").write_text('provider = "postgresql"\n')
    initial_name = "20261027000000_add_accounting_ledger_tables"
    candidate_name = "20261103000000_remove_accounting_events"
    for name in (initial_name, candidate_name):
        directory = migrations / name
        directory.mkdir()
        statement = (_MIGRATIONS / name / "migration.sql").read_text()
        if name == candidate_name and interrupt_before_commit:
            statement = statement.replace("COMMIT;", "SELECT 1 / 0;\nCOMMIT;")
        (directory / "migration.sql").write_text(statement)
    command = [sys.executable, "-I", "-m", "prisma", "migrate"]
    environment = {**os.environ, "DATABASE_URL": url}
    resolved = subprocess.run(
        command + ["resolve", "--applied", initial_name, "--schema", str(schema)],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert resolved.returncode == 0, resolved.stderr
    result = subprocess.run(
        command + ["deploy", "--schema", str(schema)],
        env=environment,
        capture_output=True,
        text=True,
    )
    present = event_database.execute("SELECT to_regclass('\"LiteLLM_AccountingEvent\"') IS NOT NULL").fetchone()
    if interrupt_before_commit:
        assert result.returncode != 0, result.stdout
        assert present == (True,)
        assert event_database.execute('SELECT event_id FROM "LiteLLM_AccountingEvent"').fetchall() == [("event",)]
    else:
        assert result.returncode == 0, result.stderr
        assert present == (False,)
        assert event_database.execute(
            'SELECT finished_at IS NOT NULL FROM "_prisma_migrations" WHERE migration_name=%s', (candidate_name,)
        ).fetchone() == (True,)


def test_committed_removal_can_replay_after_lost_migration_record(
    event_database: psycopg.Connection[tuple[object, ...]],
) -> None:
    statement = (_MIGRATIONS / "20261103000000_remove_accounting_events/migration.sql").read_text()
    event_database.execute(statement)
    event_database.execute(statement)
    assert event_database.execute("SELECT to_regclass('\"LiteLLM_AccountingEvent\"')").fetchone() == (None,)
    assert event_database.execute('SELECT COUNT(*) FROM "LiteLLM_AccountingDelivery"').fetchone() == (0,)


def test_schema_push_guard_only_blocks_events_in_the_target_schema(
    event_database: psycopg.Connection[tuple[object, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    from urllib.parse import urlsplit, urlunsplit
    from litellm_proxy_extras.utils import ProxyExtrasDBManager

    url = urlunsplit(
        urlsplit(os.environ["DATABASE_URL"])._replace(path=f"/{event_database.info.dbname}", query="schema=other")
    )
    monkeypatch.setenv("DATABASE_URL", url)
    event_database.execute("CREATE SCHEMA other")
    ProxyExtrasDBManager.raise_if_accounting_event_removal_pending()
    assert event_database.execute('SELECT event_id FROM "LiteLLM_AccountingEvent"').fetchall() == [("event",)]
    event_database.execute('CREATE TABLE other."LiteLLM_AccountingEvent" (event_id text)')
    with pytest.raises(RuntimeError, match="accounting events"):
        ProxyExtrasDBManager.raise_if_accounting_event_removal_pending()
