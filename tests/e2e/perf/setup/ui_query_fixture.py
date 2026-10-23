from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Iterable
from datetime import datetime, timedelta
from itertools import chain
from typing import Final

import psycopg
from pydantic import AwareDatetime, BaseModel, ConfigDict, TypeAdapter

from setup.ui_query_isolation import validate_target
from setup.upgrade_worker import read_state, release_indexes


class UiFixture(BaseModel):
    model_config = ConfigDict(frozen=True)

    anchor: AwareDatetime
    api_key: str
    """`perf_ui_db_*` JUnit properties naming how much data the journeys ran against."""
    volume: tuple[tuple[str, int], ...] = ()


class TableVolume(BaseModel):
    model_config = ConfigDict(frozen=True)

    table: str
    rows: int
    bytes: int


QUERIED_TABLES: Final = (
    "LiteLLM_SpendLogs",
    "LiteLLM_DailyUserSpend",
    "LiteLLM_DailyTeamSpend",
    "LiteLLM_DailyTagSpend",
    "LiteLLM_VerificationToken",
    "LiteLLM_TeamTable",
    "LiteLLM_UserTable",
    "LiteLLM_ProxyModelTable",
)
_VOLUME_SQL: Final = """
    SELECT c.relname, GREATEST(c.reltuples, 0)::bigint, pg_total_relation_size(c.oid)
    FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public' AND c.relname = ANY(%(tables)s)
"""


SEEDED_TEAMS: Final = 100
SEEDED_USERS: Final = 2000
SEEDED_KEYS: Final = 5000
SEEDED_MODELS: Final = 200
DAILY_TABLES: Final = ("LiteLLM_DailyUserSpend", "LiteLLM_DailyTeamSpend")
_SEED_SQL: Final = (
    """
    INSERT INTO "LiteLLM_TeamTable" (team_id, team_alias, members_with_roles, metadata, model_spend, model_max_budget)
    SELECT 'perf-team-' || n, 'perf-team-' || n, '[]', '{}', '{}', '{}' FROM generate_series(0, %(teams)s - 1) AS n
    ON CONFLICT (team_id) DO NOTHING
    """,
    """
    INSERT INTO "LiteLLM_UserTable" (user_id, user_alias, user_email, user_role, team_id, metadata, model_spend, model_max_budget)
    SELECT 'perf-user-' || n, 'perf-user-' || n, 'perf-user-' || n || '@example.com', 'internal_user',
        'perf-team-' || (n %% %(teams)s), '{}', '{}', '{}'
    FROM generate_series(0, %(users)s - 1) AS n
    ON CONFLICT (user_id) DO NOTHING
    """,
    """
    INSERT INTO "LiteLLM_VerificationToken" (token, key_name, key_alias, team_id, user_id, aliases, config, permissions,
        metadata, model_spend, model_max_budget, created_at, updated_at)
    SELECT md5('perf-key:' || n) || md5('perf-key-tail:' || n), 'sk-...perf' || n, 'perf-key-' || n,
        'perf-team-' || (n %% %(teams)s), 'perf-user-' || (n %% %(users)s), '{}', '{}', '{}', '{}', '{}', '{}',
        %(anchor)s - make_interval(days => n %% 365), %(anchor)s
    FROM generate_series(0, %(keys)s - 1) AS n
    ON CONFLICT (token) DO NOTHING
    """,
    """
    INSERT INTO "LiteLLM_ProxyModelTable" (model_id, model_name, litellm_params, model_info, created_by, updated_by)
    SELECT 'perf-model-' || n, 'perf-chat-' || (n %% 20),
        jsonb_build_object('model', 'openai/perf-chat', 'api_key', 'perf-seed',
            'api_base', 'os.environ/E2E_PERF_MOCK_OPENAI_URL'),
        jsonb_build_object('id', 'perf-model-' || n, 'db_model', true), 'perf-seed', 'perf-seed'
    FROM generate_series(0, %(models)s - 1) AS n
    ON CONFLICT (model_id) DO NOTHING
    """,
)


def volume_properties(volumes: Iterable[TableVolume]) -> tuple[tuple[str, int], ...]:
    """One `perf_ui_db_rows_<table>` and one `perf_ui_db_bytes_<table>` property per table the journeys read,
    so the dashboard can say what the latencies were measured against."""
    return tuple(
        chain.from_iterable(
            ((f"perf_ui_db_rows_{v.table}", v.rows), (f"perf_ui_db_bytes_{v.table}", v.bytes)) for v in volumes
        )
    )


def table_volumes(connection: psycopg.Connection, spend_log_rows: int) -> tuple[TableVolume, ...]:
    """Planner row estimates and on-disk size (indexes included) of the queried tables. The restore's
    `LiteLLM_SpendLogs` count comes from the seed checkpoint, exact where the estimate can be stale."""
    adapter: Final = TypeAdapter(tuple[str, int, int])
    found: Final = tuple(
        adapter.validate_python(row)
        for row in connection.execute(_VOLUME_SQL, {"tables": list(QUERIED_TABLES)}).fetchall()
    )
    return tuple(
        TableVolume(table=table, rows=spend_log_rows if table == "LiteLLM_SpendLogs" else rows, bytes=size)
        for table, rows, size in sorted(found)
    )


def prepare_fixture() -> UiFixture:
    validate_target(os.environ["DATABASE_URL"])
    _finish_upgrade()
    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=10) as connection:
        _ = connection.execute("SET statement_timeout='30s'")
        row: Final = connection.execute("SELECT spec->>'anchor', rows FROM litellm_perf_seed.checkpoint").fetchone()
        assert row is not None and TypeAdapter(int).validate_python(row[1]) > 0, "Missing completed seed rows"
        seed_ui_tables(connection, anchor=TypeAdapter(datetime).validate_python(row[0]))
        fixture: Final = UiFixture(
            anchor=TypeAdapter(datetime).validate_python(row[0]),
            api_key=hashlib.md5(b"perf-key:0", usedforsecurity=False).hexdigest()
            + hashlib.md5(b"perf-key-tail:0", usedforsecurity=False).hexdigest(),
            volume=volume_properties(table_volumes(connection, TypeAdapter(int).validate_python(row[1]))),
        )
        return fixture


def seed_ui_tables(connection: psycopg.Connection, anchor: datetime) -> None:
    """Give the Keys, Teams, Users and Models pages realistic row counts on the disposable restore.

    Names follow the SpendLogs seed (perf-key-N aliases, perf-team-N, perf-user-N) so the Logs filters and
    Team Usage line up with real spend. The Daily* rollups come from the snapshot itself and are only checked.
    """
    window: Final = {
        "since": (anchor - timedelta(days=7)).strftime("%Y-%m-%d"),
        "until": anchor.strftime("%Y-%m-%d"),
    }
    for table in DAILY_TABLES:
        row = connection.execute(f'SELECT count(*) FROM "{table}" WHERE date BETWEEN %(since)s AND %(until)s', window)
        assert _count(row.fetchone()) > 0, f"{table} has no rows in the week before the anchor"
    params: Final = {
        "teams": SEEDED_TEAMS,
        "users": SEEDED_USERS,
        "keys": SEEDED_KEYS,
        "models": SEEDED_MODELS,
        "anchor": anchor,
    }
    for statement in _SEED_SQL:
        _ = connection.execute(statement, params)
    connection.commit()


def _count(row: tuple[object, ...] | None) -> int:
    assert row is not None
    return TypeAdapter(int).validate_python(row[0])


def _released_schema(connection: psycopg.Connection) -> bool:
    """A released tag's migrations image upgrades the restore (schema and request-log indexes) before the
    stack starts, so there is no upgrade worker state to wait on."""
    row: Final = connection.execute("""SELECT to_regclass('public."_LiteLLMPerfUpgrade"') IS NULL""").fetchone()
    return row is None or TypeAdapter(bool).validate_python(row[0])


def _finish_upgrade() -> None:
    if os.environ.get("E2E_DATABASE_UPGRADE") != "1":
        return
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True, connect_timeout=10) as connection:
        if _released_schema(connection):
            return
        release_indexes(connection)
        deadline: Final = time.monotonic() + 6 * 3600
        while time.monotonic() < deadline:
            match read_state(connection).phase:
                case "failed":
                    raise AssertionError("Snapshot upgrade failed")
                case "complete":
                    return
                case _:
                    pass
            time.sleep(5)
    raise AssertionError("Snapshot indexes did not finish before UI measurements")
