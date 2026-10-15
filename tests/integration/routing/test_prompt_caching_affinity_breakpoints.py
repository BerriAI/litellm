from collections.abc import Callable
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows, scratch_database
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.redis_process import owned_redis
from integration._support.wire import Wire, wire_server
from integration.providers._cache_control_marks_support import (
    anthropic_deployment,
    anthropic_peer,
    final_text,
    marker_of,
    new_marker,
    owned_config,
    post_chat,
)
from integration.routing._prompt_cache_pin_support import (
    Pin,
    claude_peer,
    lives_five_minutes,
    lives_one_hour,
    one_new_pin,
    pin_config,
    read_pins,
    reference,
)
from pydantic import JsonValue
from redis import Redis

_MODEL: Final = "affinity-breakpoint-claude"
_FOLLOW_UPS: Final = 16
_LONG_SYSTEM: Final = "Answer from the tool results below. " + "lorem ipsum " * 1500
_REDIS_ENVIRONMENT: Final = ("REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD", "REDIS_URL", "REDIS_SSL")
_NO_RESPONSE_CACHE: Final[dict[str, JsonValue]] = {"cache": False, "enable_redis_auth_cache": False}


def _first_turn(session: str, marker: str) -> list[JsonValue]:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": f"{_LONG_SYSTEM} session {session}",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        },
        {"role": "user", "content": final_text(marker)},
    ]


def _follow_up(first_turn: list[JsonValue], marker: str) -> list[JsonValue]:
    return [*first_turn, {"role": "assistant", "content": "sunny"}, {"role": "user", "content": final_text(marker)}]


def _served(wire: Wire) -> frozenset[str]:
    return frozenset(marker_of(request) for request in wire.drain())


def _body(messages: list[JsonValue]) -> dict[str, JsonValue]:
    return {"model": _MODEL, "messages": messages, "max_tokens": 64}


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_follow_ups_return_to_the_deployment_that_cached_the_breakpoint_prefix(
    gateway: Gateway, tmp_path: Path
) -> None:
    session: Final = new_marker()
    first_marker: Final = new_marker()
    follow_up_markers: Final = frozenset(new_marker() for _ in range(_FOLLOW_UPS))
    first_turn: Final = _first_turn(session, first_marker)
    with scratch_database() as database_url, wire_server(anthropic_peer) as left, wire_server(anthropic_peer) as right:
        config: Final = owned_config(
            tmp_path,
            [
                {**anthropic_deployment(_MODEL, left.url), "model_info": {"id": f"breakpoint-left-{session}"}},
                {**anthropic_deployment(_MODEL, right.url), "model_info": {"id": f"breakpoint-right-{session}"}},
            ],
            router_settings={"optional_pre_call_checks": ["prompt_caching"]},
        )
        with owned_proxy_process(
            gateway,
            tmp_path,
            {"DATABASE_URL": database_url},
            config=config,
            remove_environment=("DATABASE_URL_READ_REPLICA",),
            workers=2,
        ) as owned:
            status, response_id, text = post_chat(owned.gateway, _body(first_turn))
            assert status == 200, text
            eventually(
                lambda: read_rows(
                    'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                    (response_id,),
                    database_url=database_url,
                ),
                lambda rows: len(rows) == 1,
                seconds=70,
            )
            follow_ups: Final = tuple(
                post_chat(owned.gateway, _body(_follow_up(first_turn, marker))) for marker in follow_up_markers
            )
        served: Final = (_served(left), _served(right))
    assert [status for status, _, _ in follow_ups] == [200] * _FOLLOW_UPS, [text for _, _, text in follow_ups]
    (pinned,) = tuple(side for side in served if first_marker in side)
    assert pinned == {first_marker, *follow_up_markers}, served


@pytest.mark.parametrize(
    ("ttl", "lives"),
    ((None, lives_five_minutes), ("1h", lives_one_hour)),
    ids=("default_ttl", "one_hour"),
)
@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_a_breakpoint_pin_lives_as_long_as_the_ttl_the_breakpoint_asks_for(
    gateway: Gateway, tmp_path: Path, ttl: str | None, lives: Callable[[Pin], bool]
) -> None:
    session: Final = new_marker()
    marker: Final = new_marker()
    identities: Final = (f"breakpoint-pin-left-{session}", f"breakpoint-pin-right-{session}")
    breakpoint: Final[dict[str, JsonValue]] = {"mode": "explicit", **({"ttl": ttl} if ttl else {})}
    turns: Final[list[JsonValue]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": reference(session), "prompt_cache_breakpoint": breakpoint}],
        },
        {"role": "user", "content": final_text(marker)},
    ]
    with (
        owned_redis(tmp_path) as cache,
        wire_server(claude_peer) as left,
        wire_server(claude_peer) as right,
        Redis(host=cache.host, port=cache.port, socket_timeout=2) as redis,
    ):
        config: Final = pin_config(
            tmp_path,
            [
                {**anthropic_deployment(_MODEL, left.url), "model_info": {"id": identities[0]}},
                {**anthropic_deployment(_MODEL, right.url), "model_info": {"id": identities[1]}},
            ],
            litellm_settings=_NO_RESPONSE_CACHE,
            router_settings={"redis_host": cache.host, "redis_port": cache.port},
        )
        with owned_proxy_process(
            gateway, tmp_path, {}, config=config, remove_environment=_REDIS_ENVIRONMENT, workers=2
        ) as owned:
            before: Final = frozenset(read_pins(redis))
            status, _, text = post_chat(owned.gateway, _body(turns))
            assert status == 200, text
            _, pin = one_new_pin(redis, before)
        served: Final = (_served(left), _served(right))
    assert served[identities.index(pin.model_id)] == {marker}, (pin, served)
    assert lives(pin), pin
