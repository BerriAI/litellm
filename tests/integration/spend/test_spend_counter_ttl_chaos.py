import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.redis_process import owned_redis
from integration._support.spend_counters import (
    counter_ttl,
    counter_value,
    key_counter,
    outlives,
    rig_cache,
)
from pydantic import JsonValue
from redis import Redis

RATES: Final = {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002}
ANTHROPIC_HEADERS: Final = {"anthropic-version": "2023-06-01"}
BURST: Final = 24
UNBOUNDED_BUDGET: Final = 100000.0
RESERVATION_ESTIMATE_CEILING: Final = 40.0
SLOW_UPSTREAM_SECONDS: Final = 3.0
LEAK_KEY_BUDGET: Final = 20.0
CELL_TIMEOUT: Final = int(2 * graceful_stop_seconds() + 120)
RESPAWN_SECONDS: Final = graceful_stop_seconds() + 60


def _chat(gateway: Gateway, model: str, key: str, index: int, stream: bool) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": f"burst chat {index} {uuid.uuid4().hex}"}],
            "stream": stream,
            **({"stream_options": {"include_usage": True}} if stream else {}),
        },
        key=key,
    )


def _messages(gateway: Gateway, model: str, key: str, index: int, stream: bool) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/messages",
        {
            "model": model,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": f"burst messages {index} {uuid.uuid4().hex}"}],
            "stream": stream,
        },
        key=key,
        headers=ANTHROPIC_HEADERS,
    )


def _responses(gateway: Gateway, model: str, key: str, index: int, stream: bool) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/responses",
        {"model": model, "input": f"burst responses {index} {uuid.uuid4().hex}", "stream": stream},
        key=key,
    )


SURFACES: Final[tuple[Callable[[Gateway, str, str, int, bool], httpx.Response], ...]] = (
    _chat,
    _messages,
    _responses,
)


def _spend_rows(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT request_id, spend, status FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
        (sha256(key.encode()).hexdigest(),),
    )


def _successes(key: str, count: int) -> list[dict[str, JsonValue]]:
    rows: Final = eventually(
        lambda: [row for row in _spend_rows(key) if str(row["status"]) == "success"],
        lambda values: len(values) >= count,
        seconds=90,
    )
    assert len(rows) == count, rows
    assert len({str(row["request_id"]) for row in rows}) == count, rows
    return rows


def _counter_matches(cache: Redis, counter: str, total: float) -> tuple[float | None, int]:
    state: Final = eventually(
        lambda: (counter_value(cache, counter), counter_ttl(cache, counter)),
        lambda values: values[0] is not None and values[0] == pytest.approx(total, abs=1e-9),
        seconds=20,
    )
    assert outlives(state[1]), state
    return state


def test_a_burst_with_upstream_failures_lands_each_success_once_with_a_long_ttl(gateway: Gateway) -> None:
    with (
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
        gateway.scenario() as scenario,
        rig_cache() as cache,
    ):
        provider_model: Final = f"burst-{uuid.uuid4().hex}"
        model: Final = scenario.model(model=f"openai/{provider_model}", **RATES)
        key: Final = scenario.key(models=[model], max_budget=UNBOUNDED_BUDGET)
        statuses: Final = [500] + [200, 200, 200] * (BURST // 4 + 2)

        def remove_script() -> None:
            response: Final = upstream.delete(f"/__scripts/{provider_model}")
            assert response.status_code in (200, 404), response.text

        scenario.cleanups.callback(remove_script)
        configured: Final = upstream.post(f"/__scripts/{provider_model}", json={"statuses": statuses})
        assert configured.status_code == 200, configured.text

        def attempt(index: int) -> httpx.Response:
            stream: Final = index % 2 == 1
            return SURFACES[index % len(SURFACES)](gateway, model, key, index, stream)

        with ThreadPoolExecutor(max_workers=BURST) as pool:
            responses: Final = tuple(pool.map(attempt, range(BURST)))
        codes: Final = [response.status_code for response in responses]
        assert all(code in (200, 500) for code in codes), codes
        succeeded: Final = codes.count(200)
        assert 0 < succeeded < BURST, codes
        rows: Final = _successes(key, succeeded)
        total: Final = sum(float(str(row["spend"])) for row in rows)
        assert total == pytest.approx(succeeded * 0.06), rows
        _counter_matches(cache, key_counter(key), total)


def _workers(owned: OwnedProxy) -> tuple[int, ...]:
    return tuple(sorted(owned.worker_pids()))


def _attempt(gateway: Gateway, model: str, key: str, index: int) -> int:
    try:
        return _chat(gateway, model, key, index, False).status_code
    except httpx.HTTPError:
        return 0


@pytest.mark.timeout(CELL_TIMEOUT)
def test_a_killed_worker_does_not_shorten_or_lose_the_counter(gateway: Gateway, tmp_path: Path) -> None:
    with (
        owned_proxy_process(gateway, tmp_path, {}, workers=2) as owned,
        owned.gateway.scenario() as scenario,
        rig_cache() as cache,
    ):
        model: Final = scenario.model(**RATES)
        key: Final = scenario.key(models=[model], max_budget=UNBOUNDED_BUDGET)
        before: Final = _workers(owned)
        assert len(before) == 2, before
        with ThreadPoolExecutor(max_workers=12) as pool:
            first: Final = pool.submit(_attempt, owned.gateway, model, key, 0)
            assert first.result(timeout=60) == 200
            in_flight: Final = [pool.submit(_attempt, owned.gateway, model, key, index) for index in range(1, 12)]
            with owned.replacing_worker(before[0]) as kill:
                kill()
                survivor: Final = _attempt(owned.gateway, model, key, 99)
            codes: Final = [future.result(timeout=120) for future in in_flight]
        assert survivor == 200, survivor
        assert all(code in (0, 200) for code in codes), codes
        replaced: Final = eventually(
            lambda: _workers(owned),
            lambda pids: len(pids) == 2 and before[0] not in pids and before[1] in pids,
            seconds=RESPAWN_SECONDS,
        )
        assert replaced
        after: Final = [_attempt(owned.gateway, model, key, index) for index in range(12, 18)]
        assert after == [200] * 6, after
        answered: Final = 1 + 1 + codes.count(200) + 6
        landed: Final = eventually(
            lambda: [row for row in _spend_rows(key) if str(row["status"]) == "success"],
            lambda rows: len(rows) >= 1 + 1 + 6,
            seconds=90,
        )
        assert len({str(row["request_id"]) for row in landed}) == len(landed) <= answered, landed
        ceiling: Final = answered * 0.06 + len(in_flight) * RESERVATION_ESTIMATE_CEILING
        state: Final = eventually(
            lambda: (counter_value(cache, key_counter(key)), counter_ttl(cache, key_counter(key))),
            lambda values: values[0] is not None and 8 * 0.06 - 1e-9 <= values[0] <= ceiling,
            seconds=20,
        )
        assert outlives(state[1]), state


@pytest.mark.timeout(CELL_TIMEOUT)
def test_a_reservation_leaked_by_a_killed_worker_stays_for_the_counter_ttl(gateway: Gateway, tmp_path: Path) -> None:
    with (
        owned_proxy_process(gateway, tmp_path, {}, workers=2) as owned,
        owned.gateway.scenario() as scenario,
        rig_cache() as cache,
    ):
        slow: Final = scenario.model(model="openai/slow-endpoint", **RATES)
        key: Final = scenario.key(models=[slow], max_budget=LEAK_KEY_BUDGET)
        counter: Final = key_counter(key)
        before: Final = _workers(owned)
        assert len(before) == 2, before
        with ThreadPoolExecutor(max_workers=1) as pool:
            in_flight: Final = pool.submit(_attempt, owned.gateway, slow, key, 0)
            reserved: Final = eventually(
                lambda: counter_value(cache, counter),
                lambda value: value is not None and value >= LEAK_KEY_BUDGET - 1e-9,
                seconds=SLOW_UPSTREAM_SECONDS,
            )
            with owned.replacing_worker(before[0]) as first, owned.replacing_worker(before[1]) as second:
                first.freeze()
                second.freeze()
                first()
                second()
            interrupted: Final = in_flight.result(timeout=60)
        assert interrupted != 200, interrupted
        leaked: Final = (counter_value(cache, counter), counter_ttl(cache, counter))
        assert leaked[0] == pytest.approx(reserved), (reserved, leaked)
        respawned: Final = eventually(
            lambda: _workers(owned),
            lambda pids: len(pids) == 2 and not set(pids) & set(before),
            seconds=RESPAWN_SECONDS,
        )
        assert respawned
        blocked: Final = eventually(
            lambda: _attempt(owned.gateway, slow, key, 1), lambda code: code == 422, seconds=RESPAWN_SECONDS
        )
        assert blocked == 422, (blocked, leaked)
        held: Final = (counter_value(cache, counter), counter_ttl(cache, counter))
        assert held[0] == pytest.approx(reserved), (reserved, held)
        assert outlives(held[1]), held


@pytest.mark.timeout(CELL_TIMEOUT)
def test_a_redis_outage_mid_burst_keeps_serving_and_rebuilds_the_counter_long(gateway: Gateway, tmp_path: Path) -> None:
    with owned_redis(tmp_path) as redis:
        overrides: Final[Mapping[str, str]] = {
            "REDIS_HOST": redis.host,
            "REDIS_PORT": str(redis.port),
            "REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT": "2",
        }
        with (
            owned_proxy_process(gateway, tmp_path, overrides) as owned,
            owned.gateway.scenario() as scenario,
            Redis(host=redis.host, port=redis.port) as cache,
        ):
            model: Final = scenario.model(**RATES)
            key: Final = scenario.key(models=[model], max_budget=UNBOUNDED_BUDGET)
            counter: Final = key_counter(key)

            def burst(start: int) -> list[int]:
                with ThreadPoolExecutor(max_workers=6) as pool:
                    return list(
                        pool.map(lambda index: _attempt(owned.gateway, model, key, index), range(start, start + 6))
                    )

            assert burst(0) == [200] * 6
            _counter_matches(cache, counter, 6 * 0.06)
            redis.stop()
            during: Final = burst(6)
            assert during == [200] * 6, during
            redis.start()
            assert cache.ping()
            probes: Final[list[int]] = []

            def probe_until_the_counter_is_back() -> float | None:
                probes.append(_attempt(owned.gateway, model, key, 100 + len(probes)))
                return counter_value(cache, counter)

            eventually(probe_until_the_counter_is_back, lambda value: value is not None, seconds=30)
            assert set(probes) == {200}, probes
            after: Final = burst(12)
            assert after == [200] * 6, after
            rows: Final = _successes(key, 18 + len(probes))
            total: Final = sum(float(str(row["spend"])) for row in rows)
            assert total == pytest.approx((18 + len(probes)) * 0.06), rows
            rebuilt: Final = eventually(
                lambda: (counter_value(cache, counter), counter_ttl(cache, counter)),
                lambda state: state[0] is not None and 7 * 0.06 - 1e-9 <= state[0] <= total + 1e-9,
                seconds=20,
            )
            assert outlives(rebuilt[1]), rebuilt
