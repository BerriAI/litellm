import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import psycopg
import pytest
import yaml
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.database import read_rows
from tests.integration._support.process import owned_proxy

CLEANUP_EVERY_MINUTE: Final = "* * * * *"
_CONFIG: Final = TypeAdapter(dict[str, dict[str, JsonValue]])


def _day(days_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%d")


def _seed_daily_tag_spend(tag: str, days: tuple[str, ...]) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        for day in days:
            connection.execute(
                'INSERT INTO "LiteLLM_DailyTagSpend" (id, tag, date, api_key, model, spend, updated_at) '
                "VALUES (%s, %s, %s, %s, %s, 1.0, now())",
                (uuid.uuid4().hex, tag, day, f"integration-{tag}", "gpt-4o-mini"),
            )


def _seed_old_spend_log(request_id: str, days_ago: int) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        connection.execute(
            'INSERT INTO "LiteLLM_SpendLogs" (request_id, call_type, api_key, spend, "startTime", "endTime") '
            "VALUES (%s, 'acompletion', %s, 0, now() - make_interval(days => %s), now() - make_interval(days => %s))",
            (request_id, f"integration-{request_id}", str(days_ago), str(days_ago)),
        )


def _delete_daily_tag_spend(tag: str) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        connection.execute('DELETE FROM "LiteLLM_DailyTagSpend" WHERE tag = %s', (tag,))


def _remaining_days(tag: str) -> tuple[str, ...]:
    rows: Final = read_rows('SELECT date FROM "LiteLLM_DailyTagSpend" WHERE tag = %s ORDER BY date', (tag,))
    return tuple(str(row["date"]) for row in rows)


def _spend_log_present(request_id: str) -> bool:
    return bool(read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (request_id,)))


def _cleanup_config(tmp_path: Path, retention: dict[str, JsonValue]) -> Path:
    base: Final = _CONFIG.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final = {
        **base,
        "general_settings": {
            **base["general_settings"],
            **retention,
            "maximum_spend_logs_cleanup_cron": CLEANUP_EVERY_MINUTE,
            "scheduled_job_stagger": {"enabled": False},
        },
    }
    path: Final = tmp_path / "retention.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.covers("spend.daily_tag_spend.retention_prunes_rows_older_than_the_period_and_keeps_the_rest")
def test_daily_tag_spend_retention_prunes_only_rows_older_than_the_period(gateway: Gateway, tmp_path: Path) -> None:
    tag: Final = f"integration-retention-{uuid.uuid4().hex}"
    expired, on_the_cutoff, today = _day(200), _day(30), _day(0)
    _seed_daily_tag_spend(tag, (expired, on_the_cutoff, today))
    try:
        config: Final = _cleanup_config(tmp_path, {"maximum_daily_tag_spend_retention_period": "30d"})
        with owned_proxy(gateway, tmp_path, {}, config=config):
            remaining: Final = eventually(
                lambda: _remaining_days(tag),
                lambda days: expired not in days,
                seconds=150,
            )
        assert remaining == (on_the_cutoff, today), remaining
    finally:
        _delete_daily_tag_spend(tag)


@pytest.mark.covers("spend.daily_tag_spend.unset_retention_never_deletes_even_while_spend_logs_are_pruned")
def test_daily_tag_spend_is_kept_forever_when_its_retention_is_unset(gateway: Gateway, tmp_path: Path) -> None:
    tag: Final = f"integration-retention-{uuid.uuid4().hex}"
    request_id: Final = f"integration-retention-{uuid.uuid4().hex}"
    _seed_daily_tag_spend(tag, (_day(200),))
    _seed_old_spend_log(request_id, days_ago=200)
    try:
        config: Final = _cleanup_config(tmp_path, {"maximum_spend_logs_retention_period": "30d"})
        with owned_proxy(gateway, tmp_path, {}, config=config):
            eventually(lambda: _spend_log_present(request_id), lambda present: not present, seconds=150)
            assert _remaining_days(tag) == (_day(200),)
    finally:
        _delete_daily_tag_spend(tag)
