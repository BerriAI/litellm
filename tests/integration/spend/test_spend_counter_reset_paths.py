"""The two spend reset paths that touch the Redis spend counters.

``POST /global/spend/reset`` zeroes every key and team spend in the database. The
counters that enforce those budgets must follow it on every pod, or a key the
operator just reset stays blocked until the counter's TTL runs out. ``PATCH
/organization/update`` with ``spend`` sets one organization's counter to the given
value. Both write a reset fence next to each zeroed counter so a pod still holding
the pre-reset floor cannot repair the counter back up, and both answer 500 when the
counters could not be written to Redis, so the operator retries once it is back.
Every cell drives a real proxy: the cells that reset every key or stop Redis boot
their own proxy on a scratch database and an owned Redis.
"""

import asyncio
import os
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, object_value
from integration._support.database import read_rows, scratch_database
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.redis_process import OwnedRedis, owned_redis
from integration._support.spend_counters import (
    SPEND_COUNTER_TTL_SECONDS,
    counter_ttl,
    counter_value,
    key_counter,
    outlives,
    rig_cache,
)
from pydantic import BaseModel, ConfigDict
from redis import Redis

from litellm.caching.redis_cache import RedisCache

ROOT: Final = Path(os.environ.get("INTEGRATION_PROXY_ROOT") or Path(__file__).resolve().parents[3])
RATES: Final = {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002}
ONE_CALL_COST: Final = 0.06
INVALIDATION_CHANNEL: Final = "litellm_proxy.auth_cache_invalidation"
REDIS_AUTH_CACHE_SETTING: Final = frozenset({"enable_redis_auth_cache"})
REDIS_SETTINGS: Final = frozenset({"cache", "cache_params", *REDIS_AUTH_CACHE_SETTING})
SEEDED_COUNTERS: Final = 1200
SETTLE_SECONDS: Final = 20
BROADCAST_SECONDS: Final = 10
RACE_WINDOW_SECONDS: Final = 2
DATABASE_SECONDS: Final = 70
FENCE_SECONDS: Final = 10
RECOVERY_SECONDS: Final = 30
LOST_COUNTERS: Final = "but the cached spend counters could not be reset in Redis"
RETRY_HINT: Final = "retry this request once Redis accepts writes."
CELL_TIMEOUT: Final = int(2 * graceful_stop_seconds() + 120)


class _BroadcastMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    cache_key: str | None = None
    cache_keys: tuple[str, ...] = ()
    new_value: float | None = None
    ttl: float | None = None

    def names(self) -> tuple[str, ...]:
        return (self.cache_key,) if self.cache_key is not None else self.cache_keys


@dataclass(frozen=True, slots=True)
class Broadcast:
    key: str
    value: float | None
    ttl: float | None


class _Broadcasts:
    def __init__(self, cache: Redis) -> None:
        self._pubsub: Final = cache.pubsub()
        self._pubsub.subscribe(INVALIDATION_CHANNEL)
        confirmation: Final = self._pubsub.get_message(timeout=5)
        assert confirmation is not None and confirmation["type"] == "subscribe", confirmation
        self._condition: Final = threading.Condition()
        self._received: Final[list[Broadcast]] = []
        self._stopped: Final = threading.Event()
        self._thread: Final = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        while not self._stopped.is_set():
            self._record(self._pubsub.get_message(timeout=0.2))

    def _record(self, message: Mapping[str, object] | None) -> None:
        if message is None or message["type"] != "message":
            return
        data: Final = message["data"]
        assert isinstance(data, bytes | str), message
        parsed: Final = _BroadcastMessage.model_validate_json(data)
        with self._condition:
            self._received.extend(Broadcast(key, parsed.new_value, parsed.ttl) for key in parsed.names())
            self._condition.notify_all()

    def wait_for(self, satisfied: Callable[[Sequence[Broadcast]], bool], seconds: float) -> bool:
        with self._condition:
            return self._condition.wait_for(lambda: satisfied(tuple(self._received)), timeout=seconds)

    def received(self) -> tuple[Broadcast, ...]:
        with self._condition:
            return tuple(self._received)

    def close(self) -> None:
        self._stopped.set()
        self._thread.join(timeout=5)
        self._pubsub.close()


@contextmanager
def _broadcasts(cache: Redis) -> Iterator[_Broadcasts]:
    broadcasts: Final = _Broadcasts(cache)
    try:
        yield broadcasts
    finally:
        broadcasts.close()


def _zeroed(counter: str) -> Callable[[Sequence[Broadcast]], bool]:
    return lambda received: any(broadcast.key == counter and broadcast.value == 0.0 for broadcast in received)


def _zeroed_with_its_floor(counter: str) -> Callable[[Sequence[Broadcast]], bool]:
    return lambda received: (
        sum(1 for broadcast in received if broadcast.key.endswith(counter) and broadcast.value == 0.0) >= 2
    )


@dataclass(frozen=True, slots=True)
class Isolated:
    gateway: Gateway
    cache: Redis
    redis: OwnedRedis
    database_url: str
    log: Path


def _pod_environment(database_url: str, redis: OwnedRedis) -> Mapping[str, str]:
    return {
        "DATABASE_URL": database_url,
        "REDIS_HOST": redis.host,
        "REDIS_PORT": str(redis.port),
        "REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT": "2",
    }


@contextmanager
def _isolated(
    gateway: Gateway,
    directory: Path,
    *,
    workers: int = 1,
    settings: tuple[tuple[str, str], ...] = (),
    config: Path | None = None,
) -> Iterator[Isolated]:
    with scratch_database() as database_url, owned_redis(directory) as redis:
        overrides: Final[Mapping[str, str]] = {**_pod_environment(database_url, redis), **dict(settings)}
        with (
            owned_proxy_process(
                gateway,
                directory,
                overrides,
                workers=workers,
                config=config,
                remove_environment=("DATABASE_URL_READ_REPLICA",),
            ) as owned,
            Redis(host=redis.host, port=redis.port) as cache,
        ):
            yield Isolated(owned.gateway, cache, redis, database_url, owned.log)


@contextmanager
def _second_pod(gateway: Gateway, directory: Path, isolated: Isolated, config: Path) -> Iterator[Gateway]:
    with owned_proxy_process(
        gateway,
        directory,
        _pod_environment(isolated.database_url, isolated.redis),
        config=config,
        database_setup=(),
        remove_environment=("DATABASE_URL_READ_REPLICA",),
    ) as owned:
        yield owned.gateway


def _config_without(directory: Path, dropped: frozenset[str]) -> Path:
    shared: Final = object_value(yaml.safe_load((ROOT / "tests/integration/proxy_config.yaml").read_text()))
    litellm_settings: Final = {
        name: value for name, value in object_value(shared["litellm_settings"]).items() if name not in dropped
    }
    path: Final = directory / f"proxy_config_{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump({**shared, "litellm_settings": litellm_settings}, sort_keys=False))
    return path


@contextmanager
def _isolated_without_redis(gateway: Gateway, directory: Path) -> Iterator[tuple[Gateway, str]]:
    with (
        scratch_database() as database_url,
        owned_proxy_process(
            gateway,
            directory,
            {"DATABASE_URL": database_url},
            config=_config_without(directory, REDIS_SETTINGS),
            remove_environment=("REDIS_HOST", "REDIS_PORT", "DATABASE_URL_READ_REPLICA"),
        ) as owned,
    ):
        yield owned.gateway, database_url


def _one_spend(rows: Sequence[Mapping[str, object]]) -> float:
    assert len(rows) == 1, rows
    return float(str(rows[0]["spend"]))


def _key_spend(key: str, database_url: str | None = None) -> float:
    return _one_spend(
        read_rows(
            'SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s',
            (sha256(key.encode()).hexdigest(),),
            database_url=database_url,
        )
    )


def _team_spend(team: str, database_url: str | None = None) -> float:
    return _one_spend(
        read_rows('SELECT spend FROM "LiteLLM_TeamTable" WHERE team_id=%s', (team,), database_url=database_url)
    )


def _user_spend(user: str, database_url: str | None = None) -> float:
    return _one_spend(
        read_rows('SELECT spend FROM "LiteLLM_UserTable" WHERE user_id=%s', (user,), database_url=database_url)
    )


def _organization_spend(organization: str, database_url: str | None = None) -> float:
    return _one_spend(
        read_rows(
            'SELECT spend FROM "LiteLLM_OrganizationTable" WHERE organization_id=%s',
            (organization,),
            database_url=database_url,
        )
    )


def _landed(read: Callable[[], float], least: float) -> float:
    return eventually(read, lambda spend: spend >= least - 1e-9, seconds=DATABASE_SECONDS)


def _chat(client: httpx.Client, model: str, key: str, *, end_user: str | None = None) -> httpx.Response:
    return client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": f"reset path {uuid.uuid4().hex}"}],
            **({} if end_user is None else {"user": end_user}),
        },
        headers={"Authorization": f"Bearer {key}"},
    )


def _assert_served(response: httpx.Response) -> str:
    assert response.status_code == 200, response.text
    body: Final = response.json()
    assert body["usage"]["total_tokens"] == 40, body
    return str(body["id"])


def _assert_denied(response: httpx.Response) -> None:
    assert response.status_code == 422, response.text
    assert response.json()["error"]["type"] == "budget_exceeded", response.text


def _assert_counters_unwritten(response: httpx.Response, saved: str) -> None:
    assert response.status_code == 500, response.text
    error: Final = response.json()["error"]
    assert error["type"] == "internal_server_error", response.text
    message: Final = str(error["message"])
    assert message.startswith(f"{saved}, {LOST_COUNTERS}"), message
    assert message.endswith(RETRY_HINT), message


def _accepted_once_redis_is_back(request: Callable[[], httpx.Response]) -> None:
    def attempt() -> int:
        response: Final = request()
        assert response.status_code in {200, 500}, response.text
        return response.status_code

    eventually(attempt, lambda code: code == 200, seconds=RECOVERY_SECONDS)


def _fence(counter: str) -> str:
    return f"{{{counter}}}:reset_fence"


async def _repair_as_a_pod_would(redis: OwnedRedis, counter: str, value: float) -> float | None:
    cache: Final = RedisCache(host=redis.host, port=redis.port)
    assert cache.reset_fence_key(counter) == _fence(counter)
    return await cache.async_set_max(
        key=counter, value=value, ttl=SPEND_COUNTER_TTL_SECONDS, fence_key=cache.reset_fence_key(counter)
    )


def _state(cache: Redis, counter: str) -> tuple[float | None, int]:
    return counter_value(cache, counter), counter_ttl(cache, counter)


def _settled(cache: Redis, counter: str, spend: float) -> int:
    _, ttl = eventually(
        lambda: _state(cache, counter),
        lambda state: state[0] is not None and state[0] == pytest.approx(spend, abs=1e-9),
        seconds=SETTLE_SECONDS,
    )
    return ttl


def _values(cache: Redis, counters: Sequence[str]) -> tuple[float | None, ...]:
    return tuple(None if raw is None else float(raw) for raw in cache.mget(counters))


def _ttls(cache: Redis, counters: Sequence[str]) -> tuple[int, ...]:
    with cache.pipeline() as pipeline:
        for counter in counters:
            pipeline.ttl(counter)
        return tuple(int(ttl) for ttl in pipeline.execute())


def _exhaust(gateway: Gateway, cache: Redis, database_url: str, model: str, key: str) -> str:
    counter: Final = key_counter(key)
    _assert_served(_chat(gateway.client, model, key))
    _settled(cache, counter, ONE_CALL_COST)
    _landed(lambda: _key_spend(key, database_url), ONE_CALL_COST)
    _assert_denied(_chat(gateway.client, model, key))
    return counter


def _global_reset(gateway: Gateway) -> None:
    reset: Final = gateway.request("POST", "/global/spend/reset")
    assert reset.status_code == 200, reset.text


@pytest.mark.timeout(CELL_TIMEOUT)
def test_a_global_spend_reset_admits_a_blocked_key_on_the_next_request(gateway: Gateway, tmp_path: Path) -> None:
    with (
        _isolated(gateway, tmp_path, workers=2) as isolated,
        isolated.gateway.scenario() as scenario,
        httpx.Client(base_url=str(isolated.gateway.client.base_url), timeout=60) as other_connection,
    ):
        model: Final = scenario.model(**RATES)
        key: Final = scenario.key(models=[model], max_budget=ONE_CALL_COST)
        counter: Final = _exhaust(isolated.gateway, isolated.cache, isolated.database_url, model, key)
        _global_reset(isolated.gateway)
        assert outlives(_settled(isolated.cache, counter, 0.0))
        assert _key_spend(key, isolated.database_url) == 0.0
        _assert_served(_chat(other_connection, model, key))
        assert outlives(_settled(isolated.cache, counter, ONE_CALL_COST))
        _assert_denied(_chat(other_connection, model, key))
        _landed(lambda: _key_spend(key, isolated.database_url), ONE_CALL_COST)


@pytest.mark.timeout(CELL_TIMEOUT)
def test_a_global_spend_reset_admits_a_key_blocked_by_its_team_budget(gateway: Gateway, tmp_path: Path) -> None:
    with _isolated(gateway, tmp_path) as isolated, isolated.gateway.scenario() as scenario:
        model: Final = scenario.model(**RATES)
        team: Final = scenario.team(models=[model], max_budget=ONE_CALL_COST)
        key: Final = scenario.key(team_id=team, models=[model])
        counter: Final = f"spend:team:{team}"
        _assert_served(_chat(isolated.gateway.client, model, key))
        _settled(isolated.cache, counter, ONE_CALL_COST)
        _landed(lambda: _team_spend(team, isolated.database_url), ONE_CALL_COST)
        _assert_denied(_chat(isolated.gateway.client, model, key))
        _global_reset(isolated.gateway)
        assert outlives(_settled(isolated.cache, counter, 0.0))
        assert _team_spend(team, isolated.database_url) == 0.0
        _assert_served(_chat(isolated.gateway.client, model, key))
        assert outlives(_settled(isolated.cache, counter, ONE_CALL_COST))
        _assert_denied(_chat(isolated.gateway.client, model, key))


@pytest.mark.timeout(CELL_TIMEOUT)
def test_a_global_spend_reset_leaves_the_other_scopes_and_a_non_master_caller_alone(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _isolated(gateway, tmp_path) as isolated, isolated.gateway.scenario() as scenario:
        model: Final = scenario.model(**RATES)
        organization: Final = scenario.organization(max_budget=50.0)
        team: Final = scenario.team(models=[model], max_budget=20.0, organization_id=organization)
        user: Final = scenario.user(max_budget=20.0)
        added: Final = isolated.gateway.request(
            "POST",
            "/team/member_add",
            {"team_id": team, "member": {"user_id": user, "role": "user"}, "max_budget_in_team": 20.0},
        )
        assert added.status_code == 200, added.text
        tag: Final = f"integration-tag-{uuid.uuid4().hex}"
        key: Final = scenario.key(team_id=team, user_id=user, models=[model], max_budget=5.0, tags=[tag])
        end_user: Final = f"integration-end-user-{uuid.uuid4().hex}"
        reset_counters: Final = (key_counter(key), f"spend:team:{team}")
        kept_counters: Final = (
            f"spend:team_member:{user}:{team}",
            f"spend:user:{user}",
            f"spend:end_user:{end_user}",
            f"spend:tag:{tag}",
            f"spend:org:{organization}",
        )
        _assert_served(_chat(isolated.gateway.client, model, key, end_user=end_user))
        for counter in (*reset_counters, *kept_counters):
            assert outlives(_settled(isolated.cache, counter, ONE_CALL_COST)), counter
        _landed(lambda: _user_spend(user, isolated.database_url), ONE_CALL_COST)
        _landed(lambda: _organization_spend(organization, isolated.database_url), ONE_CALL_COST)

        forbidden: Final = isolated.gateway.request("POST", "/global/spend/reset", key=key)
        assert forbidden.status_code == 401, forbidden.text
        assert forbidden.json()["error"]["type"] == "auth_error", forbidden.text
        assert _values(isolated.cache, (*reset_counters, *kept_counters)) == pytest.approx(
            (ONE_CALL_COST,) * 7, abs=1e-9
        )

        _global_reset(isolated.gateway)
        for counter in reset_counters:
            assert outlives(_settled(isolated.cache, counter, 0.0)), counter
        assert _values(isolated.cache, kept_counters) == pytest.approx((ONE_CALL_COST,) * 5, abs=1e-9)
        assert all(outlives(ttl) for ttl in _ttls(isolated.cache, kept_counters)), _ttls(isolated.cache, kept_counters)
        assert _key_spend(key, isolated.database_url) == 0.0
        assert _team_spend(team, isolated.database_url) == 0.0
        assert _user_spend(user, isolated.database_url) >= ONE_CALL_COST - 1e-9
        assert _organization_spend(organization, isolated.database_url) >= ONE_CALL_COST - 1e-9


@pytest.mark.timeout(CELL_TIMEOUT)
def test_a_global_spend_reset_zeroes_every_key_counter_past_one_scan_page_and_broadcasts_them(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _isolated(gateway, tmp_path) as isolated:
        stem: Final = f"spend:key:audit-{uuid.uuid4().hex}"
        team_named_like_a_window: Final = f"spend:team:audit-{uuid.uuid4().hex}:window:prod"
        counters: Final = (*(f"{stem}-{index}" for index in range(SEEDED_COUNTERS)), team_named_like_a_window)
        windows: Final = tuple(f"{stem}-w{index}:window:1d" for index in range(3))
        with isolated.cache.pipeline() as pipeline:
            for name in (*counters, *windows):
                pipeline.set(name, "1.0", ex=3600)
            pipeline.execute()
        with _broadcasts(isolated.cache) as broadcasts:
            _global_reset(isolated.gateway)
            eventually(
                lambda: _values(isolated.cache, counters),
                lambda values: all(value == 0.0 for value in values),
                seconds=SETTLE_SECONDS,
            )
            ttls: Final = _ttls(isolated.cache, counters)
            assert all(outlives(ttl) for ttl in ttls), sorted(ttls)[:5]
            assert _values(isolated.cache, windows) == (1.0, 1.0, 1.0)
            assert broadcasts.wait_for(
                lambda received: {broadcast.key for broadcast in received if broadcast.value == 0.0} >= set(counters),
                BROADCAST_SECONDS,
            ), len(broadcasts.received())
            assert not any(broadcast.key in windows for broadcast in broadcasts.received())


@pytest.mark.timeout(CELL_TIMEOUT)
def test_a_global_spend_reset_with_a_redis_batch_size_of_one_resets_and_fences_every_counter(
    gateway: Gateway, tmp_path: Path
) -> None:
    with (
        _isolated(gateway, tmp_path, settings=(("DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE", "1"),)) as isolated,
        isolated.gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(**RATES)
        key: Final = scenario.key(models=[model], max_budget=ONE_CALL_COST)
        counter: Final = _exhaust(isolated.gateway, isolated.cache, isolated.database_url, model, key)
        seeded: Final = tuple(f"spend:key:audit-{uuid.uuid4().hex}" for _ in range(2))
        with isolated.cache.pipeline() as pipeline:
            for name in seeded:
                pipeline.set(name, "1.0", ex=3600)
            pipeline.execute()
        counters: Final = (counter, *seeded)
        _global_reset(isolated.gateway)
        assert isolated.cache.exists(*(_fence(name) for name in counters)) == len(counters)
        assert _values(isolated.cache, counters) == (0.0, 0.0, 0.0)
        assert all(outlives(ttl) for ttl in _ttls(isolated.cache, counters)), _ttls(isolated.cache, counters)
        _assert_served(_chat(isolated.gateway.client, model, key))
        assert outlives(_settled(isolated.cache, counter, ONE_CALL_COST))
        _assert_denied(_chat(isolated.gateway.client, model, key))


@pytest.mark.timeout(CELL_TIMEOUT)
def test_a_pod_that_cached_the_pre_reset_floor_admits_the_next_request_after_a_reset_on_another_pod(
    gateway: Gateway, tmp_path: Path
) -> None:
    with (
        _isolated(gateway, tmp_path, config=_config_without(tmp_path, REDIS_AUTH_CACHE_SETTING)) as isolated,
        _second_pod(gateway, tmp_path, isolated, _config_without(tmp_path, REDIS_AUTH_CACHE_SETTING)) as other_pod,
        isolated.gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(**RATES)
        key: Final = scenario.key(models=[model], max_budget=ONE_CALL_COST)
        counter: Final = _exhaust(isolated.gateway, isolated.cache, isolated.database_url, model, key)
        with _broadcasts(isolated.cache) as broadcasts:
            assert isolated.cache.set(counter, "0.0", ex=3600)
            _assert_denied(_chat(other_pod.client, model, key))
            _settled(isolated.cache, counter, ONE_CALL_COST)
            _global_reset(isolated.gateway)
            broadcasts.wait_for(_zeroed_with_its_floor(counter), RACE_WINDOW_SECONDS)
            assert other_pod.client.get("/health/liveliness").status_code == 200
            _assert_served(_chat(other_pod.client, model, key))
            assert outlives(_settled(isolated.cache, counter, ONE_CALL_COST))
            _assert_denied(_chat(other_pod.client, model, key))
            assert broadcasts.wait_for(_zeroed(counter), BROADCAST_SECONDS), broadcasts.received()


@pytest.mark.timeout(CELL_TIMEOUT)
def test_a_global_spend_reset_fences_the_counter_against_a_stale_floor_repair(gateway: Gateway, tmp_path: Path) -> None:
    with _isolated(gateway, tmp_path) as isolated, isolated.gateway.scenario() as scenario:
        model: Final = scenario.model(**RATES)
        key: Final = scenario.key(models=[model], max_budget=ONE_CALL_COST)
        counter: Final = _exhaust(isolated.gateway, isolated.cache, isolated.database_url, model, key)
        _global_reset(isolated.gateway)
        assert outlives(_settled(isolated.cache, counter, 0.0))
        assert isolated.cache.get(_fence(counter)) == b"1"
        assert 0 < counter_ttl(isolated.cache, _fence(counter)) <= FENCE_SECONDS
        assert asyncio.run(_repair_as_a_pod_would(isolated.redis, counter, ONE_CALL_COST)) == 0.0
        value, ttl = _state(isolated.cache, counter)
        assert value == 0.0 and outlives(ttl), (value, ttl)
        _assert_served(_chat(isolated.gateway.client, model, key))
        assert outlives(_settled(isolated.cache, counter, ONE_CALL_COST))
        _assert_denied(_chat(isolated.gateway.client, model, key))
        eventually(lambda: isolated.cache.exists(_fence(counter)), lambda found: found == 0, seconds=FENCE_SECONDS + 5)
        late_total: Final = 2 * ONE_CALL_COST
        assert asyncio.run(_repair_as_a_pod_would(isolated.redis, counter, late_total)) == pytest.approx(late_total)
        value_after, ttl_after = _state(isolated.cache, counter)
        assert value_after == pytest.approx(late_total) and outlives(ttl_after), (value_after, ttl_after)


@pytest.mark.timeout(CELL_TIMEOUT)
def test_a_global_spend_reset_while_redis_is_down_answers_500_and_the_retry_resets_the_counters(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _isolated(gateway, tmp_path) as isolated, isolated.gateway.scenario() as scenario:
        model: Final = scenario.model(**RATES)
        key: Final = scenario.key(models=[model], max_budget=ONE_CALL_COST)
        counter: Final = _exhaust(isolated.gateway, isolated.cache, isolated.database_url, model, key)
        isolated.redis.stop()
        _assert_counters_unwritten(isolated.gateway.request("POST", "/global/spend/reset"), "Spend was reset")
        assert _key_spend(key, isolated.database_url) == 0.0
        _assert_served(_chat(isolated.gateway.client, model, key))
        _landed(lambda: _key_spend(key, isolated.database_url), ONE_CALL_COST)
        isolated.redis.start()
        assert isolated.cache.ping()
        _accepted_once_redis_is_back(lambda: isolated.gateway.request("POST", "/global/spend/reset"))
        assert _key_spend(key, isolated.database_url) == 0.0
        assert outlives(_settled(isolated.cache, counter, 0.0))
        _assert_served(_chat(isolated.gateway.client, model, key))
        assert outlives(_settled(isolated.cache, counter, ONE_CALL_COST))
        _assert_denied(_chat(isolated.gateway.client, model, key))


@pytest.mark.timeout(CELL_TIMEOUT)
def test_a_global_spend_reset_without_redis_admits_the_next_request(gateway: Gateway, tmp_path: Path) -> None:
    with _isolated_without_redis(gateway, tmp_path) as (owned, database_url), owned.scenario() as scenario:
        model: Final = scenario.model(**RATES)
        key: Final = scenario.key(models=[model], max_budget=ONE_CALL_COST)
        _assert_served(_chat(owned.client, model, key))
        _landed(lambda: _key_spend(key, database_url), ONE_CALL_COST)
        _assert_denied(_chat(owned.client, model, key))
        _global_reset(owned)
        assert _key_spend(key, database_url) == 0.0
        _assert_served(_chat(owned.client, model, key))
        _landed(lambda: _key_spend(key, database_url), ONE_CALL_COST)
        _assert_denied(_chat(owned.client, model, key))


def _organization_entities(
    gateway: Gateway, scenario: Scenario, database_url: str | None = None
) -> tuple[str, str, str]:
    model: Final = scenario.model(**RATES)
    organization: Final = scenario.organization(max_budget=ONE_CALL_COST)
    team: Final = scenario.team(models=[model], organization_id=organization)
    key: Final = scenario.key(team_id=team, models=[model])
    counter: Final = f"spend:org:{organization}"
    _assert_served(_chat(gateway.client, model, key))
    _landed(lambda: _organization_spend(organization, database_url), ONE_CALL_COST)
    _assert_denied(_chat(gateway.client, model, key))
    return model, key, counter


def _organization_update(gateway: Gateway, counter: str, **fields: object) -> httpx.Response:
    return gateway.request(
        "PATCH", "/organization/update", {"organization_id": counter.removeprefix("spend:org:"), **fields}
    )


def _update_organization(gateway: Gateway, counter: str, **fields: object) -> None:
    updated: Final = _organization_update(gateway, counter, **fields)
    assert updated.status_code == 200, updated.text


def test_an_organization_spend_set_to_zero_admits_a_key_blocked_by_the_organization_budget(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, rig_cache() as cache:
        model, key, counter = _organization_entities(gateway, scenario)
        _settled(cache, counter, ONE_CALL_COST)
        with _broadcasts(cache) as broadcasts:
            _update_organization(gateway, counter, spend=0.0)
            assert outlives(_settled(cache, counter, 0.0))
            assert _organization_spend(counter.removeprefix("spend:org:")) == 0.0
            assert broadcasts.wait_for(_zeroed(counter), BROADCAST_SECONDS), broadcasts.received()
        _assert_served(_chat(gateway.client, model, key))
        assert outlives(_settled(cache, counter, ONE_CALL_COST))
        _assert_denied(_chat(gateway.client, model, key))


def test_an_organization_update_without_spend_keeps_the_counter_and_the_block(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, rig_cache() as cache:
        model, key, counter = _organization_entities(gateway, scenario)
        _settled(cache, counter, ONE_CALL_COST)
        _update_organization(gateway, counter, metadata={"note": f"audit {uuid.uuid4().hex}"})
        value, ttl = _state(cache, counter)
        assert value == pytest.approx(ONE_CALL_COST, abs=1e-9) and outlives(ttl), (value, ttl)
        _assert_denied(_chat(gateway.client, model, key))


def test_an_organization_spend_set_above_the_budget_keeps_the_block_until_set_to_zero(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, rig_cache() as cache:
        model, key, counter = _organization_entities(gateway, scenario)
        _settled(cache, counter, ONE_CALL_COST)
        _update_organization(gateway, counter, spend=1.5)
        assert outlives(_settled(cache, counter, 1.5))
        _assert_denied(_chat(gateway.client, model, key))
        _update_organization(gateway, counter, spend=0.0)
        assert outlives(_settled(cache, counter, 0.0))
        _assert_served(_chat(gateway.client, model, key))


@pytest.mark.timeout(CELL_TIMEOUT)
def test_an_organization_spend_set_while_redis_is_down_answers_500_and_the_retry_resets_the_counter(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _isolated(gateway, tmp_path) as isolated, isolated.gateway.scenario() as scenario:
        model, key, counter = _organization_entities(isolated.gateway, scenario, isolated.database_url)
        organization: Final = counter.removeprefix("spend:org:")
        _settled(isolated.cache, counter, ONE_CALL_COST)
        isolated.redis.stop()
        _assert_counters_unwritten(
            _organization_update(isolated.gateway, counter, spend=0.0), "The organization spend was saved"
        )
        assert _organization_spend(organization, isolated.database_url) == 0.0
        _assert_served(_chat(isolated.gateway.client, model, key))
        isolated.redis.start()
        assert isolated.cache.ping()
        _accepted_once_redis_is_back(lambda: _organization_update(isolated.gateway, counter, spend=0.0))
        assert _organization_spend(organization, isolated.database_url) == 0.0
        assert outlives(_settled(isolated.cache, counter, 0.0))
        _assert_served(_chat(isolated.gateway.client, model, key))
        assert outlives(_settled(isolated.cache, counter, ONE_CALL_COST))
        _assert_denied(_chat(isolated.gateway.client, model, key))
