from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.redis_process import OwnedRedis, owned_redis
from integration._support.wire import wire_server
from integration.providers._cache_control_marks_support import anthropic_peer
from integration.routing._prompt_cache_pin_support import (
    FIVE_MINUTE_MARK,
    ONE_HOUR_MARK,
    OVERLOAD_TOKEN,
    Pin,
    anthropic_deployment,
    ask_turn,
    chat_body,
    claude_peer,
    follow_up,
    lives_five_minutes,
    lives_one_hour,
    marked,
    new_marker,
    one_new_pin,
    pin_at,
    pin_config,
    pins_since,
    post_chat,
    read_pins,
    spend_row_lands,
    system_turn,
)
from pydantic import JsonValue
from redis import Redis

pytestmark: Final = [pytest.mark.timeout(2 * graceful_stop_seconds() + 600), pytest.mark.shares_worker_state]

_GROUP: Final = "pin-chaos"
_LEFT: Final = "pin-chaos-left"
_RIGHT: Final = "pin-chaos-right"
_REDIS_ENVIRONMENT: Final = ("REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD", "REDIS_URL", "REDIS_SSL")
_NO_RESPONSE_CACHE: Final[Mapping[str, JsonValue]] = {"cache": False, "enable_redis_auth_cache": False}
_AFTER_GROUP: Final = "pin-chaos-after"
_AFTER_LEFT: Final = "pin-chaos-after-left"
_AFTER_RIGHT: Final = "pin-chaos-after-right"
_AFTER_IDS: Final = frozenset({_AFTER_LEFT, _AFTER_RIGHT})
_PIN_SLOTS: Final = 10_000
_FILLERS: Final = _PIN_SLOTS - 1
_FILLER_BATCH: Final = 250


def _attach_path(path: str, cache: OwnedRedis) -> tuple[Mapping[str, str], Mapping[str, Mapping[str, JsonValue]]]:
    target: Final[dict[str, JsonValue]] = {"host": cache.host, "port": cache.port}
    if path == "router-settings":
        return {}, {"router_settings": {"redis_host": cache.host, "redis_port": cache.port}}
    if path == "coordination-redis":
        return {}, {"general_settings": {"coordination_redis": target}}
    return {"REDIS_HOST": cache.host, "REDIS_PORT": str(cache.port)}, {}


@pytest.mark.parametrize("path", ["router-settings", "coordination-redis", "env-only"])
def test_one_hour_pin_lands_in_the_redis_of_every_attach_path(gateway: Gateway, tmp_path: Path, path: str) -> None:
    with (
        owned_redis(tmp_path) as cache,
        wire_server(claude_peer) as left,
        wire_server(claude_peer) as right,
        Redis(host=cache.host, port=cache.port, socket_timeout=2) as redis,
    ):
        environment, settings = _attach_path(path, cache)
        config: Final = pin_config(
            tmp_path,
            [anthropic_deployment(_GROUP, _LEFT, left.url), anthropic_deployment(_GROUP, _RIGHT, right.url)],
            litellm_settings=_NO_RESPONSE_CACHE,
            router_settings=settings.get("router_settings", {}),
            general_settings=settings.get("general_settings", {}),
        )
        with owned_proxy_process(
            gateway, tmp_path, environment, config=config, remove_environment=_REDIS_ENVIRONMENT, workers=2
        ) as owned:
            before: Final = frozenset(read_pins(redis))
            response_id, served = post_chat(
                owned.gateway,
                chat_body(_GROUP, [system_turn(new_marker(), ONE_HOUR_MARK), ask_turn(new_marker())]),
            )
            _, pin = one_new_pin(redis, before)
            assert pin.model_id == served, (pin, served)
            assert lives_one_hour(pin), pin
            spend_row_lands(response_id)


def _short_turn(session: str, mark: JsonValue, ask: str) -> list[JsonValue]:
    return [{"role": "system", "content": [marked(f"Reference {session}", mark)]}, {"role": "user", "content": ask}]


def _served(gateway: Gateway, session: str, mark: JsonValue, *, overload: bool = False) -> str:
    ask: Final = f"Reply in one word. marker-{new_marker()}{f' {OVERLOAD_TOKEN}' if overload else ''}"
    return post_chat(gateway, chat_body(_GROUP, _short_turn(session, mark, ask)))[1]


def _pinned_to_right(
    gateway: Gateway, redis: Redis, session: str, mark: JsonValue, *, overload: bool, lives: Callable[[Pin], bool]
) -> None:
    before: Final = frozenset(read_pins(redis))
    assert _served(gateway, session, mark, overload=overload) == _RIGHT
    pin: Final = eventually(
        lambda: tuple(pins_since(redis, before).values()),
        lambda found: len(found) == 1 and lives(found[0]),
        seconds=30,
    )[0]
    assert pin.model_id == _RIGHT, pin


def _pin_keys(redis: Redis) -> tuple[bytes, ...]:
    return tuple(redis.scan_iter(match="*:prompt_caching", count=1000))


def _fill_batch(gateway: Gateway, redis: Redis, size: int) -> None:
    before: Final = frozenset(_pin_keys(redis))
    with ThreadPoolExecutor(max_workers=8) as pool:
        served: Final = frozenset(pool.map(lambda _: _served(gateway, new_marker(), ONE_HOUR_MARK), range(size)))
    assert served == {_LEFT}, served
    eventually(lambda: len(frozenset(_pin_keys(redis)) - before), lambda count: count >= size, seconds=120)


def test_in_memory_pins_keep_the_latest_lifetime_when_the_slots_fill(gateway: Gateway, tmp_path: Path) -> None:
    upgraded: Final = new_marker()
    downgraded: Final = new_marker()
    with (
        owned_redis(tmp_path) as cache,
        wire_server(claude_peer) as left,
        wire_server(anthropic_peer) as right,
        Redis(host=cache.host, port=cache.port, socket_timeout=5) as redis,
    ):
        config: Final = pin_config(
            tmp_path,
            [
                anthropic_deployment(_GROUP, _LEFT, left.url, order=1),
                anthropic_deployment(_GROUP, _RIGHT, right.url, order=2),
            ],
            litellm_settings=_NO_RESPONSE_CACHE,
            router_settings={"redis_host": cache.host, "redis_port": cache.port},
            general_settings={"disable_spend_logs": True},
        )
        with owned_proxy_process(
            gateway,
            tmp_path,
            {"MINIMUM_PROMPT_CACHE_TOKEN_COUNT": "1"},
            config=config,
            remove_environment=_REDIS_ENVIRONMENT,
            workers=1,
        ) as owned:
            _pinned_to_right(owned.gateway, redis, upgraded, FIVE_MINUTE_MARK, overload=True, lives=lives_five_minutes)
            upgraded_key: Final = next(iter(read_pins(redis)))
            assert _served(owned.gateway, upgraded, ONE_HOUR_MARK) == _RIGHT
            eventually(
                lambda: pin_at(redis, upgraded_key),
                lambda found: found is not None and found.model_id == _RIGHT and lives_one_hour(found),
                seconds=30,
            )
            settled: Final = frozenset(read_pins(redis))
            _pinned_to_right(owned.gateway, redis, downgraded, ONE_HOUR_MARK, overload=True, lives=lives_one_hour)
            downgraded_key: Final = next(iter(pins_since(redis, settled)))
            assert _served(owned.gateway, downgraded, FIVE_MINUTE_MARK) == _RIGHT
            eventually(
                lambda: pin_at(redis, downgraded_key),
                lambda found: found is not None and found.model_id == _RIGHT and lives_five_minutes(found),
                seconds=30,
            )
            for start in range(0, _FILLERS, _FILLER_BATCH):
                _fill_batch(owned.gateway, redis, min(_FILLER_BATCH, _FILLERS - start))
            written: Final = _pin_keys(redis)
            assert redis.delete(*written) == len(written)
            assert _served(owned.gateway, upgraded, ONE_HOUR_MARK) == _RIGHT
            assert _served(owned.gateway, downgraded, FIVE_MINUTE_MARK) == _LEFT


def _first_turns(gateway: Gateway, count: int, group: str = _GROUP) -> tuple[tuple[str, str], ...]:
    bodies: Final = tuple(
        chat_body(group, [system_turn(new_marker(), ONE_HOUR_MARK), ask_turn(new_marker())]) for _ in range(count)
    )
    with ThreadPoolExecutor(max_workers=count) as pool:
        return tuple(pool.map(lambda body: post_chat(gateway, body), bodies))


def _probe_pin(gateway: Gateway, redis: Redis) -> Mapping[str, Pin]:
    before: Final = frozenset(read_pins(redis))
    _first_turns(gateway, 1)
    return eventually(
        lambda: pins_since(redis, before), lambda found: len(found) >= 1, seconds=10, return_last_on_timeout=True
    )


def _after_group_pins(redis: Redis) -> Mapping[str, Pin]:
    return MappingProxyType({key: pin for key, pin in read_pins(redis).items() if pin.model_id in _AFTER_IDS})


def test_pins_resume_after_a_redis_outage_and_survive_a_killed_worker(gateway: Gateway, tmp_path: Path) -> None:
    with (
        owned_redis(tmp_path) as cache,
        wire_server(claude_peer) as left,
        wire_server(claude_peer) as right,
        Redis(host=cache.host, port=cache.port, socket_timeout=2) as redis,
    ):
        config: Final = pin_config(
            tmp_path,
            [
                anthropic_deployment(_GROUP, _LEFT, left.url),
                anthropic_deployment(_GROUP, _RIGHT, right.url),
                anthropic_deployment(_AFTER_GROUP, _AFTER_LEFT, left.url),
                anthropic_deployment(_AFTER_GROUP, _AFTER_RIGHT, right.url),
            ],
        )
        with owned_proxy_process(
            gateway,
            tmp_path,
            {"REDIS_HOST": cache.host, "REDIS_PORT": str(cache.port), "REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT": "1"},
            config=config,
            workers=2,
        ) as owned:
            healthy: Final = _first_turns(owned.gateway, 8)
            cache.stop()
            during: Final = _first_turns(owned.gateway, 16)
            cache.start()
            eventually(lambda: _probe_pin(owned.gateway, redis), lambda found: len(found) >= 1, seconds=90)
            recovered: Final = _first_turns(owned.gateway, 8, _AFTER_GROUP)
            for response_id, _ in (*healthy, *during, *recovered):
                spend_row_lands(response_id)
            resumed: Final = eventually(lambda: _after_group_pins(redis), lambda found: len(found) >= 8, seconds=30)
            assert len(resumed) == 8, resumed
            assert all(lives_one_hour(pin) for pin in resumed.values()), resumed
            assert sorted(pin.model_id for pin in resumed.values()) == sorted(served for _, served in recovered)
            turn: Final = [system_turn(new_marker(), ONE_HOUR_MARK), ask_turn(new_marker())]
            first_id, first_served = post_chat(owned.gateway, chat_body(_AFTER_GROUP, turn))
            grown: Final = eventually(lambda: _after_group_pins(redis), lambda found: len(found) >= 9, seconds=30)
            assert len(grown) == 9, grown
            ((key, pin),) = tuple((key, pin) for key, pin in grown.items() if key not in resumed)
            assert pin.model_id == first_served and lives_one_hour(pin), pin
            spend_row_lands(first_id)
            owned.kill_worker(sorted(owned.worker_pids())[0])
            follow_ups: Final = tuple(
                post_chat(owned.gateway, chat_body(_AFTER_GROUP, follow_up(turn, new_marker()))) for _ in range(6)
            )
            assert [served for _, served in follow_ups] == [first_served] * 6
            after: Final = pin_at(redis, key)
            assert after is not None and after.model_id == first_served and lives_one_hour(after), after
