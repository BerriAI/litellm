import os
import uuid
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
from psycopg import sql

from integration._support.client import Gateway, eventually
from integration._support.database import read_rows, scratch_database
from integration._support.process import owned_proxy

pytestmark: Final = pytest.mark.timeout(240)

READER_PASSWORD: Final = "integration-reader-password"
SPEND: Final = 1.25
REFRESHED: Final = {"message": "MonthlyGlobalSpend view refreshed", "status": "success"}
FAILED: Final = {"message": "Failed to refresh materialized view", "status": "failure"}


@dataclass(frozen=True, slots=True)
class SpendDatabase:
    candidate: Gateway
    writer_url: str
    read_url: str


@contextmanager
def readonly_role() -> Generator[str]:
    role: Final = f"integration_reader_{uuid.uuid4().hex}"
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as admin:
        admin.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOINHERIT").format(
                sql.Identifier(role), sql.Literal(READER_PASSWORD)
            )
        )
        try:
            admin.execute(sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(sql.Identifier(role)))
            yield role
        finally:
            admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))


@contextmanager
def spend_database(gateway: Gateway, tmp_path: Path, *, with_reader: bool) -> Generator[SpendDatabase]:
    """A proxy on its own database with its spend views created, reading through a real read-only role when
    `with_reader`, and with no read replica at all otherwise"""
    with readonly_role() as role, scratch_database() as writer_url:
        parsed: Final = urlsplit(writer_url)
        host: Final = parsed.netloc.rpartition("@")[2]
        reader_url: Final = urlunsplit(parsed._replace(netloc=f"{role}:{READER_PASSWORD}@{host}"))
        with psycopg.connect(writer_url, autocommit=True) as owner:
            owner.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(role)))
            owner.execute(
                sql.SQL("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO {}").format(
                    sql.Identifier(role)
                )
            )
        with owned_proxy(
            gateway,
            tmp_path,
            {"DATABASE_URL": writer_url, **({"DATABASE_URL_READ_REPLICA": reader_url} if with_reader else {})},
            remove_environment=("DATABASE_URL_READ_REPLICA",),
        ) as candidate:
            eventually(
                lambda: read_rows(
                    "SELECT relname FROM pg_class WHERE relname=%s AND relkind='v'",
                    ("MonthlyGlobalSpend",),
                    database_url=writer_url,
                ),
                bool,
                seconds=60,
            )
            connected: Final = bool(read_rows("SELECT pid FROM pg_stat_activity WHERE usename=%s", (role,)))
            assert connected == with_reader, f"read-only reader connected={connected}, expected {with_reader}"
            yield SpendDatabase(candidate, writer_url, reader_url if with_reader else writer_url)


def materialize_stock_view(writer_url: str) -> None:
    with psycopg.connect(writer_url, autocommit=True) as owner:
        owner.execute("""
            DO $$
            DECLARE definition text := pg_get_viewdef('"MonthlyGlobalSpend"'::regclass);
            BEGIN
                EXECUTE 'DROP VIEW "MonthlyGlobalSpend"';
                EXECUTE format('CREATE MATERIALIZED VIEW "MonthlyGlobalSpend" AS %s WITH NO DATA', rtrim(definition, '; '));
            END $$
        """)


def insert_spend(writer_url: str) -> None:
    with psycopg.connect(writer_url, autocommit=True) as owner:
        owner.execute(
            'INSERT INTO "LiteLLM_SpendLogs" (request_id, call_type, "startTime", "endTime", spend) '
            "VALUES (%s, 'acompletion', now(), now(), %s)",
            (f"integration-{uuid.uuid4().hex}", SPEND),
        )


def test_global_spend_refresh_runs_on_the_writer_with_a_real_readonly_reader(gateway: Gateway, tmp_path: Path) -> None:
    with spend_database(gateway, tmp_path, with_reader=True) as database:
        materialize_stock_view(database.writer_url)
        insert_spend(database.writer_url)
        with psycopg.connect(database.read_url) as reader:
            with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
                reader.execute('REFRESH MATERIALIZED VIEW "MonthlyGlobalSpend"')

        response: Final = database.candidate.request("POST", "/global/spend/refresh")

        assert response.status_code == 200 and response.json() == REFRESHED, response.text
        assert read_rows('SELECT spend FROM "MonthlyGlobalSpend"', (), database_url=database.read_url) == [
            {"spend": SPEND}
        ]


def test_concurrent_global_spend_refreshes_all_run_on_the_writer(gateway: Gateway, tmp_path: Path) -> None:
    with spend_database(gateway, tmp_path, with_reader=True) as database:
        materialize_stock_view(database.writer_url)
        insert_spend(database.writer_url)

        with ThreadPoolExecutor(max_workers=5) as executor:
            pending: Final = [
                executor.submit(database.candidate.request, "POST", "/global/spend/refresh") for _ in range(5)
            ]
            responses: Final = [future.result() for future in pending]

        assert [(response.status_code, response.json()) for response in responses] == [(200, REFRESHED)] * 5
        assert read_rows('SELECT spend FROM "MonthlyGlobalSpend"', (), database_url=database.read_url) == [
            {"spend": SPEND}
        ]


def test_global_spend_refresh_without_a_read_replica_still_refreshes(gateway: Gateway, tmp_path: Path) -> None:
    with spend_database(gateway, tmp_path, with_reader=False) as database:
        materialize_stock_view(database.writer_url)
        insert_spend(database.writer_url)

        response: Final = database.candidate.request("POST", "/global/spend/refresh")

        assert response.status_code == 200 and response.json() == REFRESHED, response.text
        assert read_rows('SELECT spend FROM "MonthlyGlobalSpend"', (), database_url=database.writer_url) == [
            {"spend": SPEND}
        ]


def test_global_spend_refresh_leaves_the_stock_plain_view_alone(gateway: Gateway, tmp_path: Path) -> None:
    with spend_database(gateway, tmp_path, with_reader=True) as database:
        response: Final = database.candidate.request("POST", "/global/spend/refresh")

        assert response.status_code == 200 and response.json() is None, response.text
        assert read_rows(
            "SELECT relkind FROM pg_class WHERE relname=%s", ("MonthlyGlobalSpend",), database_url=database.writer_url
        ) == [{"relkind": "v"}]


def test_global_spend_refresh_reports_failure_when_the_writer_refresh_errors(gateway: Gateway, tmp_path: Path) -> None:
    with spend_database(gateway, tmp_path, with_reader=True) as database:
        with psycopg.connect(database.writer_url, autocommit=True) as owner:
            owner.execute("CREATE SEQUENCE refresh_attempts")
            owner.execute('DROP VIEW "MonthlyGlobalSpend"')
            owner.execute(
                'CREATE MATERIALIZED VIEW "MonthlyGlobalSpend" AS '
                "SELECT nextval('refresh_attempts') / 0 AS spend WITH NO DATA"
            )

        response: Final = database.candidate.request("POST", "/global/spend/refresh")

        assert response.status_code == 200 and response.json() == FAILED, response.text
        assert read_rows("SELECT is_called FROM refresh_attempts", (), database_url=database.writer_url) == [
            {"is_called": True}
        ], "the refresh never ran on the writer"
        assert read_rows(
            "SELECT relispopulated FROM pg_class WHERE relname=%s",
            ("MonthlyGlobalSpend",),
            database_url=database.writer_url,
        ) == [{"relispopulated": False}]
