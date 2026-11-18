"""Timing and pg_stat_activity capture for diagnosing slow integration tests, written under the results directory."""

import json
import os
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import psycopg

_ACTIVITY: Final = """
SELECT application_name, state, wait_event_type, wait_event,
       extract(epoch FROM now() - xact_start) AS xact_seconds,
       extract(epoch FROM now() - query_start) AS query_seconds,
       left(query, 160) AS query
FROM pg_stat_activity
WHERE datname = current_database() AND pid <> pg_backend_pid() AND backend_type = 'client backend'
"""


def results_path(name: str) -> Path:
    directory: Final = Path(os.environ.get("INTEGRATION_RESULTS_DIR", "."))
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"diag-{name}-{uuid.uuid4().hex[:8]}.json"


class Timeline:
    def __init__(self) -> None:
        self.origin: Final = time.monotonic()
        self.events: Final[list[dict[str, object]]] = []
        self._lock: Final = threading.Lock()

    def mark(self, label: str, **fields: object) -> None:
        with self._lock:
            self.events.append({"t": round(time.monotonic() - self.origin, 4), "label": label, **fields})


@contextmanager
def activity_sampler(database_url: str, timeline: Timeline, interval: float = 0.1) -> Iterator[None]:
    """Samples pg_stat_activity into ``timeline`` every ``interval`` seconds until the block exits."""
    stop: Final = threading.Event()

    def run() -> None:
        connection: psycopg.Connection | None = None
        while not stop.is_set():
            try:
                if connection is None or connection.closed:
                    connection = psycopg.connect(database_url, autocommit=True, connect_timeout=2)
                rows = connection.execute(_ACTIVITY).fetchall()
                timeline.mark(
                    "pg",
                    rows=[
                        [name, state, wait_type, wait, _round(xact), _round(query_seconds), text]
                        for name, state, wait_type, wait, xact, query_seconds, text in rows
                    ],
                )
            except Exception as error:  # noqa: BLE001  # a sampler must outlive a stopped database
                timeline.mark("pg-error", error=type(error).__name__)
                if connection is not None:
                    connection.close()
                connection = None
            stop.wait(interval)
        if connection is not None:
            connection.close()

    sampler: Final = threading.Thread(target=run, name="pg-activity-sampler", daemon=True)
    sampler.start()
    try:
        yield
    finally:
        stop.set()
        sampler.join(timeout=5)


def _round(value: object) -> float | None:
    return None if value is None else round(float(str(value)), 3)


def write(name: str, timeline: Timeline, **extra: object) -> Path:
    path: Final = results_path(name)
    path.write_text(json.dumps({"events": timeline.events, **extra}, default=str))
    return path
