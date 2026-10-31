from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Final, Literal

import psycopg
from pydantic import BaseModel, ConfigDict, TypeAdapter


class UpgradeState(BaseModel):
    model_config = ConfigDict(frozen=True)

    phase: Literal["schema", "schema_ready", "indexes", "complete", "failed"]
    started_at: float
    schema_seconds: float = 0
    index_seconds: float = 0
    required_indexes: tuple[str, ...]
    required_migrations: tuple[str, ...]
    pending_indexes: tuple[str, ...]
    baseline_rows: dict[str, str]


_STATE: Final = TypeAdapter(tuple[UpgradeState])
_BOOL: Final = TypeAdapter(tuple[bool])
_STRINGS: Final = TypeAdapter(tuple[tuple[str], ...])
_FINGERPRINTS: Final = TypeAdapter(tuple[tuple[str, str], ...])
_FINGERPRINT: Final = """md5(ROW(request_id, api_key, spend, total_tokens, prompt_tokens,
    completion_tokens, model, "startTime", "endTime")::text)"""


def connect() -> psycopg.Connection[tuple[object, ...]]:
    return psycopg.connect(
        os.environ["DATABASE_URL"],
        autocommit=True,
        connect_timeout=10,
        options="-c statement_timeout=10000 -c timezone=UTC",
    )


def fingerprints(connection: psycopg.Connection[tuple[object, ...]], ids: tuple[str, ...]) -> dict[str, str]:
    rows: Final = connection.execute(
        f'SELECT request_id, {_FINGERPRINT} FROM "LiteLLM_SpendLogs" WHERE request_id = ANY(%s)', (list(ids),)
    ).fetchall()
    return dict(_FINGERPRINTS.validate_python(rows))


def missing_indexes(connection: psycopg.Connection[tuple[object, ...]], names: tuple[str, ...]) -> tuple[str, ...]:
    rows: Final = connection.execute(
        """SELECT name FROM unnest(%s::text[]) AS name WHERE NOT EXISTS (
           SELECT 1 FROM pg_index WHERE indexrelid = to_regclass(format('public.%%I', name))
           AND indisvalid AND indisready)""",
        (list(names),),
    ).fetchall()
    return tuple(row[0] for row in _STRINGS.validate_python(rows))


def read_state(connection: psycopg.Connection[tuple[object, ...]]) -> UpgradeState:
    (state,) = _STATE.validate_python(
        connection.execute('SELECT state FROM "_LiteLLMPerfUpgrade" WHERE id=1').fetchone()
    )
    return state


def write_state(connection: psycopg.Connection[tuple[object, ...]], state: UpgradeState) -> None:
    _ = connection.execute(
        """INSERT INTO "_LiteLLMPerfUpgrade" (id, state) VALUES (1, %s::jsonb)
           ON CONFLICT (id) DO UPDATE SET state = EXCLUDED.state""",
        (state.model_dump_json(),),
    )


def release_indexes(connection: psycopg.Connection[tuple[object, ...]]) -> None:
    _ = connection.execute('UPDATE "_LiteLLMPerfUpgrade" SET traffic_ready=true WHERE id=1')


def wait_for_traffic(connection: psycopg.Connection[tuple[object, ...]], deadline: float) -> bool:
    while time.monotonic() < deadline:
        (ready,) = _BOOL.validate_python(
            connection.execute('SELECT traffic_ready FROM "_LiteLLMPerfUpgrade" WHERE id=1').fetchone()
        )
        if ready:
            return True
        time.sleep(1)
    return False


def main() -> int:
    assert os.environ.get("E2E_DATABASE_UPGRADE") == "1", "Only run against a disposable restored baseline"
    assert os.environ.get("DATABASE_PERF_CONFIRM_DISPOSABLE") == "1", "Disposable database confirmation required"
    from litellm_proxy_extras import request_log_indexes
    from litellm_proxy_extras.utils import ProxyExtrasDBManager

    with connect() as connection:
        _ = connection.execute("""CREATE TABLE "_LiteLLMPerfUpgrade" (
            id integer PRIMARY KEY CHECK (id=1), state jsonb NOT NULL, traffic_ready boolean NOT NULL DEFAULT false)""")
        sample: Final = connection.execute("""
            (SELECT request_id FROM "LiteLLM_SpendLogs" ORDER BY request_id LIMIT 32)
            UNION (SELECT request_id FROM "LiteLLM_SpendLogs" ORDER BY request_id DESC LIMIT 32)""").fetchall()
        ids: Final = tuple(row[0] for row in _STRINGS.validate_python(sample))
        assert ids, "Restored baseline has no spend logs"
        names: Final = tuple(index.name for index in request_log_indexes.REQUEST_LOG_INDEXES)
        migrations: Final = Path(request_log_indexes.__file__).parent / "migrations"
        initial: Final = UpgradeState(
            phase="schema",
            started_at=time.time(),
            required_indexes=names,
            required_migrations=tuple(sorted(path.parent.name for path in migrations.glob("*/migration.sql"))),
            pending_indexes=missing_indexes(connection, names),
            baseline_rows=fingerprints(connection, ids),
        )
        assert initial.required_migrations and initial.required_indexes, "Target schema requirements are missing"
        write_state(connection, initial)
        try:
            assert ProxyExtrasDBManager.setup_database(use_migrate=True, use_v2_resolver=True), (
                "Schema migration failed"
            )
            ready: Final = initial.model_copy(
                update={
                    "phase": "schema_ready",
                    "schema_seconds": time.time() - initial.started_at,
                    "pending_indexes": missing_indexes(connection, names),
                }
            )
            write_state(connection, ready)
            assert wait_for_traffic(connection, time.monotonic() + 1800), "Traffic did not start within 30 minutes"
            write_state(connection, ready.model_copy(update={"phase": "indexes"}))
            index_start: Final = time.monotonic()
            assert ProxyExtrasDBManager.build_request_log_indexes(), "Request-log index build failed"
            complete: Final = ready.model_copy(
                update={"phase": "complete", "index_seconds": time.monotonic() - index_start}
            )
            write_state(connection, complete)
            _ = sys.stdout.write(complete.model_dump_json() + "\n")
            return 0
        except Exception as exc:
            write_state(connection, read_state(connection).model_copy(update={"phase": "failed"}))
            _ = sys.stderr.write(json.dumps({"error": type(exc).__name__}) + "\n")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
