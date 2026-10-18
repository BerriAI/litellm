from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Final

import psycopg
from psycopg import sql
from pydantic import AwareDatetime, BaseModel, ConfigDict, TypeAdapter

from setup.ui_query_isolation import validate_target
from setup.upgrade_worker import read_state, release_indexes


class UiFixture(BaseModel):
    model_config = ConfigDict(frozen=True)

    anchor: AwareDatetime
    api_key: str


def prepare_fixture() -> UiFixture:
    validate_target(os.environ["DATABASE_URL"])
    _finish_upgrade()
    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=10) as connection:
        _ = connection.execute("SET statement_timeout='30s'")
        row: Final = connection.execute("SELECT spec->>'anchor', rows FROM litellm_perf_seed.checkpoint").fetchone()
        assert row is not None and TypeAdapter(int).validate_python(row[1]) > 0, "Missing completed seed rows"
        fixture: Final = UiFixture(
            anchor=TypeAdapter(datetime).validate_python(row[0]),
            api_key=hashlib.md5(b"perf-key:0", usedforsecurity=False).hexdigest()
            + hashlib.md5(b"perf-key-tail:0", usedforsecurity=False).hexdigest(),
        )
        for statement in Path(__file__).with_suffix(".sql").read_text().split(";"):
            if statement.strip():
                _ = connection.execute(
                    statement.encode(),
                    {
                        "anchor": fixture.anchor,
                        "api_key": fixture.api_key,
                        "gateway_marker": "litellm_gateway_injected_cache",
                        "request_ids": [f"perf-ui-{i}" for i in range(1, 321)],
                    },
                )
        count: Final = connection.execute(
            'SELECT count(*) FROM "LiteLLM_SpendLogs" WHERE request_id = ANY(%s)',
            ([f"perf-ui-{i}" for i in range(1, 321)],),
        ).fetchone()
        assert count is not None and count[0] == 320, "Incomplete UI fixture"
        _daily_rows(connection, fixture)
        return fixture


def _finish_upgrade() -> None:
    if os.environ.get("E2E_DATABASE_UPGRADE") != "1":
        return
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True, connect_timeout=10) as connection:
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


def _daily_rows(connection: psycopg.Connection[tuple[object, ...]], fixture: UiFixture) -> None:
    tables: Final = (
        ("LiteLLM_DailyUserSpend", "user_id", "perf-user-0"),
        ("LiteLLM_DailyTeamSpend", "team_id", "perf-team-0"),
        ("LiteLLM_DailyTagSpend", "tag", "production"),
        ("LiteLLM_DailyOrganizationSpend", "organization_id", "perf-ui-organization"),
        ("LiteLLM_DailyEndUserSpend", "end_user_id", "perf-end-user-0"),
        ("LiteLLM_DailyAgentSpend", "agent_id", "perf-ui-agent"),
    )
    for table, field, entity in tables:
        _ = connection.execute(
            sql.SQL("""INSERT INTO {} (id, {}, date, api_key, model, model_group, custom_llm_provider,
               spend, prompt_tokens, completion_tokens, api_requests, successful_requests,
               cache_read_input_tokens, cache_creation_input_tokens, updated_at)
               VALUES (%s, %s, %s, %s, 'perf-chat', 'perf-chat', 'openai', 1, 1000, 100, 1, 1, 500, 100, now())
               ON CONFLICT DO NOTHING""").format(sql.Identifier(table), sql.Identifier(field)),
            (f"perf-ui-{table}", entity, fixture.anchor.date().isoformat(), fixture.api_key),
        )
