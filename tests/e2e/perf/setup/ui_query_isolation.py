from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

import psycopg
import pytest
from psycopg.rows import dict_row
from pydantic import BaseModel, ConfigDict, TypeAdapter


class ActiveQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    pid: int
    leader_pid: int | None
    application_name: str
    state: str | None
    wait_event_type: str | None
    wait_event: str | None
    seconds: float | None
    query: str


def validate_target(url: str) -> None:
    assert os.environ.get("E2E_UI_PERF_DISPOSABLE") == "1", "Disposable database confirmation required"
    assert os.environ.get("E2E_RESET_SPEND_LOGS") != "1", "Never truncate the snapshot baseline"
    assert re.fullmatch(r"db-perf-perf-[a-z0-9-]+\.litellm\.svc", urlparse(url).hostname or ""), (
        "SQL cancellation is restricted to the disposable perf-lane restore"
    )


def active(connection: psycopg.Connection[dict[str, object]]) -> tuple[ActiveQuery, ...]:
    rows: Final = connection.execute("""SELECT pid, leader_pid, application_name, state,
        wait_event_type, wait_event, extract(epoch from clock_timestamp()-query_start)::float AS seconds,
        left(query, 1500) AS query FROM pg_stat_activity
        WHERE datname = current_database() AND pid <> pg_backend_pid()
        AND backend_type IN ('client backend', 'parallel worker')
        AND state IS DISTINCT FROM 'idle' ORDER BY pid""").fetchall()
    return TypeAdapter(tuple[ActiveQuery, ...]).validate_python(rows)


def wait_idle(connection: psycopg.Connection[dict[str, object]], evidence: Path, *, cancel: bool) -> None:
    consecutive = 0
    for queries in (active(connection) for _ in range(30)):
        with evidence.open("a", encoding="utf-8") as output:
            _ = output.write(
                json.dumps(
                    {"time": time.time(), "cancel": cancel, "active": tuple(query.model_dump() for query in queries)}
                )
                + "\n"
            )
        if queries:
            consecutive = 0  # rebind-ok: consecutive idle observations reset on activity
            for query in queries:
                if cancel and query.leader_pid is None:
                    _ = connection.execute(
                        "SELECT pg_cancel_backend(%s)"
                        if query.state == "active"
                        else "SELECT pg_terminate_backend(%s)",
                        (query.pid,),
                    )
        else:
            consecutive += 1  # rebind-ok: consecutive idle observations across polling iterations
            if consecutive == 3:
                return
        time.sleep(1)
    pytest.exit("Database did not become idle. No further UI measurements will run", returncode=2)


def settled_counter(read: Callable[[], float], evidence: Path) -> float:
    previous: float | None = None
    consecutive = 0
    for value in (read() for _ in range(10)):
        with evidence.open("a", encoding="utf-8") as output:
            _ = output.write(json.dumps({"time": time.time(), "database_failures": value}) + "\n")
        consecutive = consecutive + 1 if value == previous else 1  # rebind-ok: polling state
        if consecutive == 3:
            return value
        previous = value  # rebind-ok: last observed counter for stabilization
        time.sleep(1)
    pytest.exit("Database counters did not settle. No further UI measurements will run", returncode=2)


@contextmanager
def isolated_database(name: str) -> Generator[None]:
    url: Final = os.environ["DATABASE_URL"]
    validate_target(url)
    evidence: Final = Path(os.environ["E2E_ARTIFACT_DIR"]) / f"ui-isolation-{name}.jsonl"
    with psycopg.Connection[dict[str, object]].connect(
        url,
        autocommit=True,
        connect_timeout=10,
        application_name="ui-isolation-monitor",
        options="-c statement_timeout=5000",
        row_factory=dict_row,
    ) as connection:
        try:
            wait_idle(connection, evidence, cancel=False)
            yield
        finally:
            try:
                wait_idle(connection, evidence, cancel=True)
            except Exception as error:
                pytest.exit(f"SQL cleanup failed ({type(error).__name__}). Stopping UI measurements", returncode=2)
