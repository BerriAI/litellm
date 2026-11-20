from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import pytest

from litellm.proxy.hooks.priority_class_share import MEMBER_SEPARATOR, team_member
from litellm.redis.stores.active_teams import ActiveTeamStore
from litellm.redis.stores.active_teams.handle import ActiveMemberRecord
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


class _DownRedis:
    """A handle on a Redis that refuses every call, raising what the Rust store raises for one."""

    has_redis: Final = True

    async def touch(
        self, model: str, member: str, priority_class: str, window_seconds: int
    ) -> Sequence[ActiveMemberRecord]:
        raise NativeRedisStoreError("transport", "redis is down")

    def touch_local(
        self, model: str, member: str, priority_class: str, window_seconds: int
    ) -> Sequence[ActiveMemberRecord]:
        raise AssertionError("memory answered for a store that has Redis")


async def test_a_touch_fails_while_redis_is_down_and_never_answers_from_memory() -> None:
    with pytest.raises(NativeRedisStoreError):
        await ActiveTeamStore(_DownRedis()).touch("chat", _WIRE_TOUCH.team, "interactive", _WIRE_TOUCH.window)
