import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.redis_process import owned_redis
from integration._support.wire import wire_server
from integration.routing._prompt_cache_pin_support import (
    FIVE_MINUTE_MARK,
    FIVE_MINUTES,
    ONE_HOUR,
    ONE_HOUR_MARK,
    Pin,
    anthropic_deployment,
    ask_turn,
    chat_body,
    claude_peer,
    lives_one_hour,
    new_marker,
    pin_config,
    pins_since,
    post_chat,
    read_pins,
    system_turn,
)
from pydantic import JsonValue
from redis import Redis

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 300)

_GROUP: Final = "pin-memory"
_LEFT: Final = "pin-memory-left"
_RIGHT: Final = "pin-memory-right"
_ITEMS: Final = "/memory-usage-in-mem-cache-items"
_MASTER_KEY_ONLY: Final = ("/memory-usage-in-mem-cache", _ITEMS)
_POLLED_ROUTES: Final = (*_MASTER_KEY_ONLY, "/debug/memory/summary")
_NO_RESPONSE_CACHE: Final[Mapping[str, JsonValue]] = {"cache": False, "enable_redis_auth_cache": False}
_BURST: Final = 24


@dataclass(frozen=True, slots=True)
class _Rig:
    owned: OwnedProxy
    redis: Redis


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("prompt-cache-pin-memory-routes")
    with (
        gateway_from_environment() as gateway,
        owned_redis(directory) as cache,
        wire_server(claude_peer) as left,
        wire_server(claude_peer) as right,
    ):
        config: Final = pin_config(
            directory,
            [anthropic_deployment(_GROUP, _LEFT, left.url), anthropic_deployment(_GROUP, _RIGHT, right.url)],
            litellm_settings=_NO_RESPONSE_CACHE,
        )
        with (
            owned_proxy_process(
                gateway, directory, {"REDIS_HOST": cache.host, "REDIS_PORT": str(cache.port)}, config=config, workers=1
            ) as owned,
            Redis(host=cache.host, port=cache.port, socket_timeout=2) as redis,
        ):
            yield _Rig(owned, redis)


def _field(body: JsonValue, *path: str) -> JsonValue:
    if not path:
        return body
    return _field(object_value(body)[path[0]], *path[1:])


def _read(gateway: Gateway, path: str) -> dict[str, JsonValue]:
    response: Final = gateway.request("GET", path)
    assert response.status_code == 200, (path, response.status_code, response.text)
    return object_value(response.json())


def _write_pins(rig: _Rig, marks: Sequence[JsonValue]) -> Mapping[str, Pin]:
    before: Final = frozenset(read_pins(rig.redis))
    for mark in marks:
        post_chat(rig.owned.gateway, chat_body(_GROUP, [system_turn(new_marker(), mark), ask_turn(new_marker())]))
    pins: Final = eventually(lambda: pins_since(rig.redis, before), lambda found: len(found) >= len(marks), seconds=30)
    assert len(pins) == len(marks), pins
    return pins


def test_items_route_lists_every_pin_with_its_own_lifetime_and_keeps_it_out_of_the_router_cache(rig: _Rig) -> None:
    pins: Final = _write_pins(rig, (ONE_HOUR_MARK, FIVE_MINUTE_MARK))
    body: Final = _read(rig.owned.gateway, _ITEMS)
    read_at: Final = time.time()
    entries: Final = object_value(body["prompt_caching_pin_cache"])
    expiries: Final = object_value(body["prompt_caching_pin_ttl"])
    router_entries: Final = object_value(body["llm_router_cache"])
    assert sorted(lives_one_hour(pin) for pin in pins.values()) == [False, True], pins
    for key, pin in pins.items():
        assert entries[key] == {"model_id": pin.model_id}, (key, entries.get(key), pin)
        remaining: Final = float(str(expiries[key])) - read_at
        assert 0 < remaining <= ONE_HOUR, (key, remaining)
        assert (remaining > FIVE_MINUTES) is lives_one_hour(pin), (key, remaining, pin)
        assert key not in router_entries, key


_COUNT_ROUTES: Final[Mapping[str, tuple[str, Callable[[dict[str, JsonValue]], JsonValue], int]]] = {
    "mem-cache": ("/memory-usage-in-mem-cache", lambda body: body["num_items_in_prompt_caching_pin_cache"], 2),
    "summary": (
        "/debug/memory/summary",
        lambda body: _field(body, "caches", "breakdown", "prompt_caching_pins", "count"),
        1,
    ),
    "details": (
        "/debug/memory/details?include_process_info=false",
        lambda body: _field(body, "cache_memory", "prompt_caching_pin_cache", "num_items"),
        1,
    ),
}


@pytest.mark.parametrize("route", sorted(_COUNT_ROUTES))
def test_count_routes_count_exactly_the_pins_the_pin_cache_holds(rig: _Rig, route: str) -> None:
    path, count_of, per_pin = _COUNT_ROUTES[route]
    pins: Final = _write_pins(rig, (ONE_HOUR_MARK, ONE_HOUR_MARK, ONE_HOUR_MARK))
    counted: Final = _read(rig.owned.gateway, path)
    held_pins: Final = object_value(_read(rig.owned.gateway, _ITEMS)["prompt_caching_pin_cache"])
    assert frozenset(pins) <= frozenset(held_pins), frozenset(pins) - frozenset(held_pins)
    assert count_of(counted) == per_pin * len(held_pins), (count_of(counted), len(held_pins))


def test_summary_total_adds_the_pins_to_every_other_cache(rig: _Rig) -> None:
    _write_pins(rig, (ONE_HOUR_MARK,))
    caches: Final = object_value(_read(rig.owned.gateway, "/debug/memory/summary")["caches"])
    breakdown: Final = object_value(caches["breakdown"])
    assert "prompt_caching_pins" in breakdown, breakdown
    assert caches["total_items"] == sum(int(str(_field(entry, "count"))) for entry in breakdown.values()), caches


def _poll_routes(gateway: Gateway, rounds: int) -> tuple[int, ...]:
    return tuple(gateway.request("GET", path).status_code for _ in range(rounds) for path in _POLLED_ROUTES)


def test_memory_routes_answer_while_a_burst_writes_pins(rig: _Rig) -> None:
    before: Final = frozenset(read_pins(rig.redis))
    bodies: Final = tuple(
        chat_body(_GROUP, [system_turn(new_marker(), ONE_HOUR_MARK), ask_turn(new_marker())]) for _ in range(_BURST)
    )
    with ThreadPoolExecutor(max_workers=_BURST + 4) as pool:
        polls: Final = tuple(pool.submit(_poll_routes, rig.owned.gateway, 8) for _ in range(4))
        answers: Final = tuple(pool.map(lambda body: post_chat(rig.owned.gateway, body), bodies))
        statuses: Final = tuple(status for poll in polls for status in poll.result())
    assert frozenset(statuses) == {200}, statuses
    pins: Final = eventually(lambda: pins_since(rig.redis, before), lambda found: len(found) >= _BURST, seconds=60)
    assert len(pins) == _BURST, pins
    assert sorted(pin.model_id for pin in pins.values()) == sorted(served for _, served in answers)
    held_pins: Final = object_value(_read(rig.owned.gateway, _ITEMS)["prompt_caching_pin_cache"])
    assert {key: held_pins.get(key) for key in pins} == {key: {"model_id": pin.model_id} for key, pin in pins.items()}


def test_master_key_only_memory_routes_refuse_a_virtual_key_and_show_it_no_pin(rig: _Rig) -> None:
    pins: Final = _write_pins(rig, (ONE_HOUR_MARK,))
    with rig.owned.gateway.scenario() as scenario:
        virtual_key: Final = scenario.key()
        refused: Final = tuple(rig.owned.gateway.request("GET", path, key=virtual_key) for path in _MASTER_KEY_ONLY)
    assert [response.status_code for response in refused] == [401, 401], [response.text for response in refused]
    assert not any(key in response.text for key in pins for response in refused)


def test_memory_routes_report_zero_pins_on_a_proxy_with_no_deployments(tmp_path: Path) -> None:
    config: Final = pin_config(
        tmp_path, [], litellm_settings=_NO_RESPONSE_CACHE, general_settings={"store_model_in_db": False}
    )
    with gateway_from_environment() as gateway, owned_proxy_process(gateway, tmp_path, {}, config=config) as owned:
        counts: Final = _read(owned.gateway, "/memory-usage-in-mem-cache")
        items: Final = _read(owned.gateway, _ITEMS)
        summary: Final = _read(owned.gateway, "/debug/memory/summary")
        details: Final = _read(owned.gateway, "/debug/memory/details?include_process_info=false")
    assert counts["num_items_in_prompt_caching_pin_cache"] == 0, counts
    assert (items["prompt_caching_pin_cache"], items["prompt_caching_pin_ttl"]) == ({}, {}), items
    assert _field(summary, "caches", "breakdown", "prompt_caching_pins", "count") == 0, summary
    assert _field(details, "cache_memory", "prompt_caching_pin_cache", "num_items") == 0, details


def test_memory_routes_report_zero_pins_on_a_proxy_that_loaded_no_config(tmp_path: Path) -> None:
    unloaded: Final = {"CONFIG_FILE_PATH": str(tmp_path / "absent.yaml"), "STORE_MODEL_IN_DB": "False"}
    with gateway_from_environment() as gateway, owned_proxy_process(gateway, tmp_path, unloaded) as owned:
        counts: Final = _read(owned.gateway, "/memory-usage-in-mem-cache")
        items: Final = _read(owned.gateway, _ITEMS)
        breakdown: Final = object_value(_field(_read(owned.gateway, "/debug/memory/summary"), "caches", "breakdown"))
        details: Final = object_value(
            _read(owned.gateway, "/debug/memory/details?include_process_info=false")["cache_memory"]
        )
    assert (counts["num_items_in_prompt_caching_pin_cache"], counts["num_items_in_llm_router_cache"]) == (0, 0), counts
    assert [items[name] for name in ("prompt_caching_pin_cache", "prompt_caching_pin_ttl", "llm_router_cache")] == [
        {}
    ] * 3, items
    assert "prompt_caching_pins" not in breakdown and "llm_responses" not in breakdown, breakdown
    assert "prompt_caching_pin_cache" not in details and "llm_router_cache" not in details, details
