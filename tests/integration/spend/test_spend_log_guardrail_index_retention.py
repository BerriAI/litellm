import uuid
from pathlib import Path
from typing import Final

import psycopg
import pytest
import yaml
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.database import read_rows, scratch_database
from tests.integration._support.process import MIGRATE_DEPLOY, owned_proxy

pytestmark: Final = pytest.mark.shares_worker_state

CLEANUP_EVERY_MINUTE: Final = "* * * * *"
_MAPPING: Final = TypeAdapter(dict[str, JsonValue])


def _seed_spend_log(database_url: str, request_id: str, days_ago: int) -> None:
    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute(
            'INSERT INTO "LiteLLM_SpendLogs" (request_id, call_type, api_key, spend, "startTime", "endTime") '
            "VALUES (%s, 'acompletion', %s, 0, now() - make_interval(days => %s), now() - make_interval(days => %s))",
            (request_id, f"integration-{request_id}", str(days_ago), str(days_ago)),
        )


def _seed_guardrail_index_row(database_url: str, request_id: str, guardrail_id: str, days_ago: int) -> None:
    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute(
            'INSERT INTO "LiteLLM_SpendLogGuardrailIndex" (request_id, guardrail_id, start_time) '
            "VALUES (%s, %s, now() - make_interval(days => %s))",
            (request_id, guardrail_id, str(days_ago)),
        )


def _guardrail_index_rows(database_url: str, request_ids: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    rows: Final = read_rows(
        'SELECT request_id, guardrail_id FROM "LiteLLM_SpendLogGuardrailIndex" WHERE request_id = ANY(%s) '
        "ORDER BY request_id, guardrail_id",
        (list(request_ids),),
        database_url=database_url,
    )
    return tuple((str(row["request_id"]), str(row["guardrail_id"])) for row in rows)


def _spend_log_present(database_url: str, request_id: str) -> bool:
    return bool(
        read_rows(
            'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (request_id,), database_url=database_url
        )
    )


def _cleanup_config(tmp_path: Path) -> Path:
    base: Final = _MAPPING.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final = {
        **base,
        "general_settings": {
            **_MAPPING.validate_python(base["general_settings"]),
            "maximum_spend_logs_retention_period": "30d",
            "maximum_spend_logs_cleanup_cron": CLEANUP_EVERY_MINUTE,
            "scheduled_job_stagger": {"enabled": False},
        },
    }
    path: Final = tmp_path / "retention.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.timeout(300)
def test_spend_log_retention_prunes_expired_guardrail_index_rows(gateway: Gateway, tmp_path: Path) -> None:
    run: Final = uuid.uuid4().hex
    guardrails: Final = tuple(f"integration-guardrail-{run}-{index}" for index in range(3))
    expired: Final = tuple(f"expired-{run}-{index}" for index in range(3))
    recent: Final = f"recent-{run}"
    with (
        scratch_database() as database_url,
        owned_proxy(
            gateway,
            tmp_path,
            {"DATABASE_URL": database_url},
            config=_cleanup_config(tmp_path),
            database_setup=MIGRATE_DEPLOY,
        ),
    ):
        for request_id, guardrail_id in zip(expired, guardrails, strict=True):
            _seed_guardrail_index_row(database_url, request_id, guardrail_id, days_ago=200)
        _seed_guardrail_index_row(database_url, recent, guardrails[0], days_ago=1)
        _seed_spend_log(database_url, expired[0], days_ago=200)
        eventually(lambda: _spend_log_present(database_url, expired[0]), lambda present: not present, seconds=150)
        remaining: Final = eventually(
            lambda: _guardrail_index_rows(database_url, (*expired, recent)),
            lambda rows: all(request_id not in {row[0] for row in rows} for request_id in expired),
            seconds=30,
            return_last_on_timeout=True,
        )
        assert remaining == ((recent, guardrails[0]),), remaining
