from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path
from typing import Final

import psycopg
import pytest
from integration._support.client import Gateway, eventually, list_value, object_value, string_value
from integration._support.database import scratch_database
from integration._support.process import MIGRATE_DEPLOY, owned_proxy_process
from psycopg.rows import class_row
from psycopg.types.json import Jsonb

pytestmark: Final = pytest.mark.timeout(240)

ERROR_INDEX: Final = "LiteLLM_SpendLogs_error_startTime_idx"
INDEXES_IN_PLACE: Final = "Request-log indexes are all in place"
SUCCESS_ROWS: Final = 5000


@dataclass(frozen=True, slots=True)
class ErrorRow:
    request_id: str
    status: str
    error_code: str
    error_message: str


@dataclass(frozen=True, slots=True)
class _ScanRow:
    scans: int


ERROR_ROWS: Final = (
    ErrorRow("failed-500-overloaded", "failure", "500", "upstream overloaded"),
    ErrorRow("failed-500-timeout", "failure", "500", "upstream timed out"),
    ErrorRow("failed-429", "failure", "429", "rate limited by upstream"),
    ErrorRow("client-disconnected", "success", "499", "Client disconnected the request"),
)
FILTERS: Final = (
    ("error_code", "500", frozenset({"failed-500-overloaded", "failed-500-timeout"})),
    ("error_code", "499", frozenset({"client-disconnected"})),
    ("error_message", "upstream", frozenset({"failed-500-overloaded", "failed-500-timeout", "failed-429"})),
)
SESSION_VIEWS: Final = ("false", "true")


def _seed(database_url: str, started: datetime) -> None:
    utc: Final = started.astimezone(timezone.utc).replace(tzinfo=None)
    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute(
            'INSERT INTO "LiteLLM_SpendLogs" (request_id, call_type, "startTime", "endTime", metadata, status) '
            "SELECT 'ok-' || n, 'acompletion', %(started)s + n * interval '1 millisecond', "
            "%(started)s + n * interval '1 millisecond', %(metadata)s, 'success' "
            "FROM generate_series(1, %(rows)s) AS n",
            {"started": utc, "metadata": Jsonb({"error_information": None}), "rows": SUCCESS_ROWS},
        )
        with connection.cursor() as cursor:
            cursor.executemany(
                'INSERT INTO "LiteLLM_SpendLogs" (request_id, call_type, "startTime", "endTime", metadata, status) '
                "VALUES (%s, 'acompletion', %s, %s, %s, %s)",
                tuple(
                    (
                        row.request_id,
                        utc,
                        utc,
                        Jsonb(
                            {
                                "error_information": {
                                    "error_code": row.error_code,
                                    "error_message": row.error_message,
                                    "error_class": "APIError",
                                }
                            }
                        ),
                        row.status,
                    )
                    for row in ERROR_ROWS
                ),
            )
        connection.execute('ANALYZE "LiteLLM_SpendLogs"')


def _error_index_scans(database_url: str) -> int:
    with psycopg.connect(database_url) as connection, connection.cursor(row_factory=class_row(_ScanRow)) as cursor:
        row: Final = cursor.execute(
            "SELECT idx_scan AS scans FROM pg_stat_user_indexes WHERE indexrelname = %s", (ERROR_INDEX,)
        ).fetchone()
    assert row is not None, f"{ERROR_INDEX} was not built"
    return row.scans


def _listed(gateway: Gateway, window: Mapping[str, str], query: Mapping[str, str]) -> frozenset[str]:
    page: Final = gateway.get("/spend/logs/ui", params={**window, **query})
    return frozenset(string_value(object_value(row)["request_id"]) for row in list_value(page["data"]))


def test_error_filters_read_only_the_rows_that_carry_error_information(gateway: Gateway, tmp_path: Path) -> None:
    now: Final = datetime.now(timezone.utc)
    window: Final = {
        "start_date": (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": (now + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
    }
    with scratch_database() as database_url:
        with owned_proxy_process(
            gateway, tmp_path, {"DATABASE_URL": database_url}, database_setup=MIGRATE_DEPLOY
        ) as booted:
            eventually(
                lambda: booted.log.read_text(errors="replace"), lambda text: INDEXES_IN_PLACE in text, seconds=60
            )
            _seed(database_url, now - timedelta(hours=1))
            scans_before: Final = _error_index_scans(database_url)

            listed: Final = {
                (session, name, value): _listed(
                    booted.gateway, window, {name: value, "group_by_session": session}
                )
                for session, (name, value, _) in product(SESSION_VIEWS, FILTERS)
            }
            assert listed == {
                (session, name, value): rows for session, (name, value, rows) in product(SESSION_VIEWS, FILTERS)
            }
            eventually(
                lambda: _error_index_scans(database_url),
                lambda scans: scans - scans_before >= len(listed),
                seconds=30,
            )
