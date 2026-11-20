import socket
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import pytest

from litellm.caching.redis_cache import RedisCache
from litellm.proxy.hooks.priority_class_share import MEMBER_SEPARATOR, team_member
from litellm.redis.stores.active_teams import ActiveTeamStore
from litellm.redis.stores.connection import connect
from litellm.rust_bridge._native import NativeRedisStoreError


@dataclass(frozen=True, slots=True)
class _WireTouch:
    """One touch as main's ``ActiveTeamRegistry`` sent it through redis-py: the sorted set member and the script's
    score arguments, which redis-py encodes with ``repr``. The same literals sit in
    ``litellm/rust/crates/redis-store/tests/active_teams.rs``, where real Redis is checked to hold them after the
    Rust store writes."""

    team: str
    member: str
    now: float
    window: int
    cutoff_arg: str
    now_arg: str


_WIRE_TOUCH: Final = _WireTouch(
    team="team\x1ft1",
    member="team\x1ft1\x1finteractive",
    now=1000.0,
    window=60,
    cutoff_arg="940.0",
    now_arg="1000.0",
)


def test_the_wire_case_is_what_main_sends() -> None:
    assert team_member("t1") == _WIRE_TOUCH.team
    assert f"{_WIRE_TOUCH.team}{MEMBER_SEPARATOR}interactive" == _WIRE_TOUCH.member
    assert repr(_WIRE_TOUCH.now - _WIRE_TOUCH.window) == _WIRE_TOUCH.cutoff_arg
    assert repr(_WIRE_TOUCH.now) == _WIRE_TOUCH.now_arg


@pytest.fixture
def down_teams() -> Iterator[ActiveTeamStore]:
    """A real native client to a port nothing listens on, so every Redis call is refused at once."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: Final = probe.getsockname()[1]
    yield ActiveTeamStore(connect(RedisCache(host="127.0.0.1", port=port)).active_teams())


async def test_a_touch_fails_while_redis_is_down_and_never_answers_from_memory(down_teams: ActiveTeamStore) -> None:
    with pytest.raises(NativeRedisStoreError):
        await down_teams.touch("chat", _WIRE_TOUCH.team, "interactive", _WIRE_TOUCH.window)
