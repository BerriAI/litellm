import json
import os
import signal
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import psutil
import psycopg
import pytest
import yaml
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import Gateway, eventually, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.process import OwnedProxy, owned_proxy, owned_proxy_process

CLEANUP_EVERY_MINUTE: Final = "* * * * *"
RETENTION_SETTING: Final = "maximum_daily_tag_spend_retention_period"
_MAPPING: Final = TypeAdapter(dict[str, JsonValue])
_SETTINGS: Final = TypeAdapter(list[dict[str, JsonValue]])


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


def _stored_retention_setting() -> JsonValue:
    rows: Final = read_rows(
        'SELECT param_value -> %s AS value FROM "LiteLLM_Config" WHERE param_name = %s',
        (RETENTION_SETTING, "general_settings"),
    )
    return rows[0]["value"] if rows else None


def _store_retention_setting(value: JsonValue) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        if value is None:
            connection.execute(
                'UPDATE "LiteLLM_Config" SET param_value = param_value - %s WHERE param_name = %s',
                (RETENTION_SETTING, "general_settings"),
            )
            return
        connection.execute(
            'UPDATE "LiteLLM_Config" SET param_value = jsonb_set(param_value, ARRAY[%s], %s::jsonb) '
            "WHERE param_name = %s",
            (RETENTION_SETTING, json.dumps(value), "general_settings"),
        )


def _listening_workers(owned: OwnedProxy) -> tuple[psutil.Process, ...]:
    port: Final = owned.gateway.client.base_url.port
    return tuple(
        child
        for child in psutil.Process(owned.process.pid).children(recursive=True)
        if any(conn.status == psutil.CONN_LISTEN and conn.laddr.port == port for conn in child.net_connections("inet"))
    )


def _listed_retention_value(gateway: Gateway) -> JsonValue:
    listed: Final = _SETTINGS.validate_json(
        gateway.request("GET", "/config/list", params={"config_type": "general_settings"}).content
    )
    matching: Final = tuple(entry for entry in listed if entry["field_name"] == RETENTION_SETTING)
    return matching[0]["field_value"] if matching else "not listed"


def _completion_id(gateway: Gateway, model: str) -> str:
    return string_value(gateway.chat(model, text=f"retention audit {uuid.uuid4().hex}")["id"])


def _cleanup_config(tmp_path: Path, retention: dict[str, JsonValue]) -> Path:
    base: Final = _MAPPING.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final = {
        **base,
        "general_settings": {
            **_MAPPING.validate_python(base["general_settings"]),
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


@pytest.mark.covers("spend.daily_tag_spend.runtime_config_update_enables_cleanup_on_a_multi_worker_proxy")
def test_config_update_turns_on_daily_tag_spend_cleanup_without_a_restart(gateway: Gateway, tmp_path: Path) -> None:
    tag: Final = f"integration-retention-{uuid.uuid4().hex}"
    expired, yesterday_of_cutoff, on_the_cutoff, today = _day(200), _day(31), _day(30), _day(0)
    _seed_daily_tag_spend(tag, (expired, yesterday_of_cutoff, on_the_cutoff, today))
    previously_stored: Final = _stored_retention_setting()
    _store_retention_setting(None)
    try:
        config: Final = _cleanup_config(tmp_path, {})
        with owned_proxy(gateway, tmp_path, {}, config=config, workers=2) as owned, owned.scenario() as scenario:
            model: Final = scenario.model()
            assert _listed_retention_value(owned) is None
            owned.post("/config/update", {"general_settings": {RETENTION_SETTING: "30d"}})
            assert _listed_retention_value(owned) == "30d"
            remaining: Final = eventually(
                lambda: _remaining_days(tag),
                lambda days: yesterday_of_cutoff not in days,
                seconds=150,
            )
            assert remaining == (on_the_cutoff, today), remaining
            assert _completion_id(owned, model).startswith("chatcmpl-")
    finally:
        _store_retention_setting(previously_stored)
        _delete_daily_tag_spend(tag)


@pytest.mark.covers("spend.daily_tag_spend.invalid_retention_value_keeps_rows_and_leaves_the_proxy_serving")
def test_unparseable_daily_tag_spend_retention_deletes_nothing_and_keeps_serving(
    gateway: Gateway, tmp_path: Path
) -> None:
    tag: Final = f"integration-retention-{uuid.uuid4().hex}"
    request_id: Final = f"integration-retention-{uuid.uuid4().hex}"
    expired: Final = _day(200)
    _seed_daily_tag_spend(tag, (expired,))
    _seed_old_spend_log(request_id, days_ago=200)
    try:
        config: Final = _cleanup_config(
            tmp_path, {RETENTION_SETTING: "soon", "maximum_spend_logs_retention_period": "30d"}
        )
        with owned_proxy(gateway, tmp_path, {}, config=config) as owned, owned.scenario() as scenario:
            model: Final = scenario.model()
            eventually(lambda: _spend_log_present(request_id), lambda present: not present, seconds=150)
            assert _remaining_days(tag) == (expired,)
            assert _completion_id(owned, model).startswith("chatcmpl-")
    finally:
        _delete_daily_tag_spend(tag)


@pytest.mark.covers("spend.daily_tag_spend.retention_horizon_is_independent_of_the_spend_log_horizon")
def test_daily_tag_spend_keeps_days_the_shorter_spend_log_horizon_already_pruned(
    gateway: Gateway, tmp_path: Path
) -> None:
    tag: Final = f"integration-retention-{uuid.uuid4().hex}"
    request_id: Final = f"integration-retention-{uuid.uuid4().hex}"
    expired, inside_tag_horizon = _day(200), _day(60)
    _seed_daily_tag_spend(tag, (expired, inside_tag_horizon))
    _seed_old_spend_log(request_id, days_ago=60)
    try:
        config: Final = _cleanup_config(
            tmp_path, {RETENTION_SETTING: "90d", "maximum_spend_logs_retention_period": "30d"}
        )
        with owned_proxy(gateway, tmp_path, {}, config=config):
            eventually(lambda: _spend_log_present(request_id), lambda present: not present, seconds=150)
            remaining: Final = eventually(lambda: _remaining_days(tag), lambda days: expired not in days, seconds=150)
        assert remaining == (inside_tag_horizon,), remaining
    finally:
        _delete_daily_tag_spend(tag)


@pytest.mark.covers("spend.daily_tag_spend.cleanup_and_serving_survive_losing_one_of_two_workers")
def test_daily_tag_spend_cleanup_completes_after_one_of_two_workers_is_killed(gateway: Gateway, tmp_path: Path) -> None:
    tag: Final = f"integration-retention-{uuid.uuid4().hex}"
    expired, today = _day(200), _day(0)
    _seed_daily_tag_spend(tag, (expired, today))
    try:
        config: Final = _cleanup_config(tmp_path, {RETENTION_SETTING: "30d"})
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            with owned.gateway.scenario() as scenario:
                model: Final = scenario.model()
                workers: Final = eventually(
                    lambda: _listening_workers(owned), lambda found: len(found) == 2, seconds=30
                )
                workers[0].send_signal(signal.SIGKILL)
                eventually(lambda: workers[0].is_running(), lambda alive: not alive, seconds=10)
                ids: Final = tuple(_completion_id(owned.gateway, model) for _ in range(6))
                assert len(set(ids)) == 6 and all(identity.startswith("chatcmpl-") for identity in ids), ids
                remaining: Final = eventually(
                    lambda: _remaining_days(tag), lambda days: expired not in days, seconds=150
                )
                assert remaining == (today,), remaining
    finally:
        _delete_daily_tag_spend(tag)


@pytest.mark.covers("spend.daily_tag_spend.unset_retention_never_deletes_even_while_spend_logs_are_pruned")
def test_daily_tag_spend_is_kept_forever_when_its_retention_is_unset(gateway: Gateway, tmp_path: Path) -> None:
    tag: Final = f"integration-retention-{uuid.uuid4().hex}"
    request_id: Final = f"integration-retention-{uuid.uuid4().hex}"
    expired: Final = _day(200)
    _seed_daily_tag_spend(tag, (expired,))
    _seed_old_spend_log(request_id, days_ago=200)
    try:
        config: Final = _cleanup_config(tmp_path, {"maximum_spend_logs_retention_period": "30d"})
        with owned_proxy(gateway, tmp_path, {}, config=config):
            eventually(lambda: _spend_log_present(request_id), lambda present: not present, seconds=150)
            assert _remaining_days(tag) == (expired,)
    finally:
        _delete_daily_tag_spend(tag)
