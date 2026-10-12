import json
import os
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Final

import psycopg
import pytest
import yaml
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.spend_counters import counter_ttl, counter_value, key_counter, outlives, rig_cache
from pydantic import JsonValue

ROOT: Final = Path(os.environ.get("INTEGRATION_PROXY_ROOT") or Path(__file__).resolve().parents[3])
RATES: Final = {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002}
ONE_CALL_COST: Final = 0.06
SLOW_UPSTREAM_SECONDS: Final = 3.0
SHORT_REDIS_TTL: Final = 4
LONG_REDIS_TTL: Final = 86400
ENV_REDIS_TTL: Final = "7200"
LONG_BATCH_WRITE_SECONDS: Final = 900
STRICT_KEY_BUDGET: Final = 50.0
CELL_TIMEOUT: Final = int(2 * graceful_stop_seconds() + 120)
FAST_RESET_TICK: Final = MappingProxyType(
    {"PROXY_BUDGET_RESCHEDULER_MIN_TIME": "2", "PROXY_BUDGET_RESCHEDULER_MAX_TIME": "2"}
)


def _config(
    directory: Path,
    *,
    litellm_settings: Mapping[str, JsonValue] = MappingProxyType({}),
    general_settings: Mapping[str, JsonValue] = MappingProxyType({}),
) -> Path:
    shared: Final = object_value(yaml.safe_load((ROOT / "tests/integration/proxy_config.yaml").read_text()))
    configured: Final = {
        **shared,
        "litellm_settings": {**object_value(shared["litellm_settings"]), **litellm_settings},
        "general_settings": {**object_value(shared["general_settings"]), **general_settings},
    }
    path: Final = directory / f"proxy_config_{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(configured, sort_keys=False))
    return path


def _chat(gateway: Gateway, model: str, key: str) -> str:
    served: Final = gateway.chat(model, key=key, text=f"settings {uuid.uuid4().hex}")
    assert served["usage"]["total_tokens"] == 40, served
    return str(served["id"])


def _counter_state(counter: str) -> tuple[float | None, int]:
    with rig_cache() as cache:
        return counter_value(cache, counter), counter_ttl(cache, counter)


def _settled(counter: str, spend: float, expected_ttl: int) -> tuple[float | None, int]:
    return eventually(
        lambda: _counter_state(counter),
        lambda state: state[0] is not None and state[0] == pytest.approx(spend) and outlives(state[1], expected_ttl),
        seconds=15,
    )


def _spend_rows(key: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE api_key=%s', (sha256(key.encode()).hexdigest(),))


@pytest.mark.timeout(CELL_TIMEOUT)
def test_null_default_redis_ttl_falls_back_to_the_hour_floor(gateway: Gateway, tmp_path: Path) -> None:
    config: Final = _config(tmp_path, litellm_settings={"default_redis_ttl": None})
    with owned_proxy(gateway, tmp_path, {}, config=config) as owned, owned.scenario() as scenario:
        model: Final = scenario.model(**RATES)
        key: Final = scenario.key(models=[model], max_budget=5.0)
        assert _chat(owned, model, key)
        assert _settled(key_counter(key), ONE_CALL_COST, 3600)


@pytest.mark.timeout(CELL_TIMEOUT)
def test_a_short_default_redis_ttl_never_shortens_the_counter_and_the_lease_refresh_keeps_it_long(
    gateway: Gateway, tmp_path: Path
) -> None:
    config: Final = _config(tmp_path, litellm_settings={"default_redis_ttl": SHORT_REDIS_TTL})
    with owned_proxy(gateway, tmp_path, {}, config=config) as owned, owned.scenario() as scenario, rig_cache() as cache:
        model: Final = scenario.model(model="openai/slow-endpoint", **RATES)
        key: Final = scenario.key(models=[model], max_budget=5.0)
        counter: Final = key_counter(key)
        with ThreadPoolExecutor(max_workers=1) as pool:
            slow: Final = pool.submit(lambda: (_chat(owned, model, key), time.monotonic())[1])
            eventually(lambda: cache.exists(counter), lambda present: present == 1, seconds=SLOW_UPSTREAM_SECONDS)
            assert cache.expire(counter, 1000)
            refreshed_at: Final = eventually(
                lambda: (time.monotonic(), counter_ttl(cache, counter)),
                lambda state: outlives(state[1]),
                seconds=SLOW_UPSTREAM_SECONDS + 10,
            )[0]
            finished_at: Final = slow.result(timeout=60)
        assert refreshed_at < finished_at, (refreshed_at, finished_at)
        assert _settled(counter, ONE_CALL_COST, 3600)


@pytest.mark.timeout(CELL_TIMEOUT)
def test_a_long_default_redis_ttl_wins_on_every_write_path(gateway: Gateway, tmp_path: Path) -> None:
    config: Final = _config(tmp_path, litellm_settings={"default_redis_ttl": LONG_REDIS_TTL})
    with owned_proxy(gateway, tmp_path, {}, config=config) as owned, owned.scenario() as scenario:
        model: Final = scenario.model(**RATES)
        key: Final = scenario.key(models=[model], max_budget=5.0)
        counter: Final = key_counter(key)
        assert _chat(owned, model, key)
        assert _settled(counter, ONE_CALL_COST, LONG_REDIS_TTL)
        updated: Final = owned.request("POST", "/key/update", {"key": key, "spend": 2.5})
        assert updated.status_code == 200, updated.text
        assert _settled(counter, 2.5, LONG_REDIS_TTL)


@pytest.mark.timeout(CELL_TIMEOUT)
def test_a_default_redis_ttl_given_as_an_environment_string_is_read_as_seconds(
    gateway: Gateway, tmp_path: Path
) -> None:
    config: Final = _config(tmp_path, litellm_settings={"default_redis_ttl": "os.environ/SPEND_COUNTER_REDIS_TTL"})
    overrides: Final = {"SPEND_COUNTER_REDIS_TTL": ENV_REDIS_TTL}
    with owned_proxy(gateway, tmp_path, overrides, config=config) as owned, owned.scenario() as scenario:
        model: Final = scenario.model(**RATES)
        key: Final = scenario.key(models=[model], max_budget=5.0)
        counter: Final = key_counter(key)
        assert _chat(owned, model, key)
        assert _settled(counter, ONE_CALL_COST, int(ENV_REDIS_TTL))
        reset: Final = owned.request("POST", f"/key/{key}/reset_spend", {"reset_to": 0.0})
        assert reset.status_code == 200, reset.text
        assert _settled(counter, 0.0, int(ENV_REDIS_TTL))


@pytest.mark.timeout(CELL_TIMEOUT)
def test_a_long_batch_write_interval_lifts_the_counter_ttl(gateway: Gateway, tmp_path: Path) -> None:
    config: Final = _config(tmp_path, general_settings={"proxy_batch_write_at": LONG_BATCH_WRITE_SECONDS})
    with owned_proxy(gateway, tmp_path, {}, config=config) as owned, owned.scenario() as scenario:
        model: Final = scenario.model(**RATES)
        key: Final = scenario.key(models=[model], max_budget=5.0)
        assert _chat(owned, model, key)
        assert _settled(key_counter(key), ONE_CALL_COST, 10 * LONG_BATCH_WRITE_SECONDS)


@pytest.mark.timeout(CELL_TIMEOUT)
def test_strict_enforcement_repairs_a_stale_low_key_counter_with_the_long_ttl(gateway: Gateway, tmp_path: Path) -> None:
    config: Final = _config(tmp_path, general_settings={"fail_closed_budget_enforcement": True})
    with owned_proxy(gateway, tmp_path, {}, config=config) as owned, owned.scenario() as scenario, rig_cache() as cache:
        model: Final = scenario.model(**RATES)
        key: Final = scenario.key(models=[model], max_budget=STRICT_KEY_BUDGET)
        counter: Final = key_counter(key)
        assert _chat(owned, model, key)
        eventually(
            lambda: read_rows(
                'SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (sha256(key.encode()).hexdigest(),)
            ),
            lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) >= ONE_CALL_COST,
            seconds=70,
        )
        attempts: Final[list[str]] = []

        def stale_then_request() -> tuple[float | None, int]:
            assert cache.set(counter, "0.01")
            attempts.append(_chat(owned, model, key))
            return eventually(
                lambda: (counter_value(cache, counter), counter_ttl(cache, counter)),
                lambda state: state[0] is not None and state[0] < STRICT_KEY_BUDGET / 2,
                seconds=15,
            )

        repaired: Final = eventually(
            stale_then_request,
            lambda state: state[0] is not None and state[0] > 0.01 + ONE_CALL_COST + 1e-6,
            seconds=30,
        )
        lifted: Final = repaired[0]
        assert (
            lifted is not None and 2 * ONE_CALL_COST - 1e-9 <= lifted <= (len(attempts) + 1) * ONE_CALL_COST + 1e-9
        ), (
            repaired,
            attempts,
        )
        assert outlives(repaired[1]), repaired


def _window_reset_at(key: str) -> datetime:
    rows: Final = read_rows(
        'SELECT budget_limits FROM "LiteLLM_VerificationToken" WHERE token=%s', (sha256(key.encode()).hexdigest(),)
    )
    assert len(rows) == 1, rows
    raw: Final = rows[0]["budget_limits"]
    windows: Final = json.loads(raw) if isinstance(raw, str) else raw
    assert isinstance(windows, list) and len(windows) == 1, raw
    return datetime.fromisoformat(str(object_value(windows[0])["reset_at"]).replace("Z", "+00:00"))


def _make_window_due(key: str) -> None:
    due: Final = (datetime.now(UTC) - timedelta(days=1)).replace(tzinfo=None).isoformat()
    windows: Final = [{"max_budget": 1.0, "budget_duration": "1d", "reset_at": due}]
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        connection.execute(
            'UPDATE "LiteLLM_VerificationToken" SET budget_limits = %s::jsonb WHERE token = %s',
            (json.dumps(windows), sha256(key.encode()).hexdigest()),
        )


@pytest.mark.timeout(CELL_TIMEOUT)
def test_a_reset_budget_window_keeps_the_long_ttl(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy(gateway, tmp_path, FAST_RESET_TICK) as owned, owned.scenario() as scenario, rig_cache() as cache:
        model: Final = scenario.model(**RATES)
        key: Final = scenario.key(models=[model], budget_limits=[{"max_budget": 1.0, "budget_duration": "1d"}])
        window_counter: Final = f"{key_counter(key)}:window:1d"
        assert _chat(owned, model, key)
        assert _settled(window_counter, ONE_CALL_COST, 3600)
        _make_window_due(key)
        rolled: Final = eventually(
            lambda: (_window_reset_at(key), counter_value(cache, window_counter), counter_ttl(cache, window_counter)),
            lambda state: state[0].replace(tzinfo=None) > datetime.now(UTC).replace(tzinfo=None),
            seconds=45,
        )
        assert rolled[1] is not None and rolled[1] < ONE_CALL_COST, rolled
        assert outlives(rolled[2]), rolled
