import asyncio
from collections.abc import Sequence

import pytest

from litellm.router_utils.prompt_cache_observations.streaks import (
    MISSES_TO_OPEN_INCIDENT,
    Counting,
    InMemoryStreakStore,
    RedisStreakStore,
    StreakEvent,
    StreakKey,
    StreakOutcome,
    StreakUpdate,
    Triggered,
    Unproven,
)

_KEY = StreakKey(deployment_id="claude-east", config_fingerprint="config-1", prefix_fingerprint="prefix-1")
_TTL_MS = 300_000
_START_MS = 1_700_000_000_000


def _update(
    event: StreakEvent, call_id: str, at_ms: int = _START_MS, key: StreakKey = _KEY, write_tokens: int = 0
) -> StreakUpdate:
    return StreakUpdate(key=key, event=event, call_id=call_id, now_ms=at_ms, ttl_ms=_TTL_MS, write_tokens=write_tokens)


async def _apply(store: InMemoryStreakStore, *updates: StreakUpdate) -> tuple[StreakOutcome, ...]:
    return tuple([await store.apply(update) for update in updates])


def _proof(at_ms: int = _START_MS) -> StreakUpdate:
    return _update(StreakEvent.PROOF, "proof", at_ms=at_ms, write_tokens=2048)


def _miss(call_id: str, at_ms: int = _START_MS + 1_000) -> StreakUpdate:
    return _update(StreakEvent.MISS, call_id, at_ms=at_ms)


async def test_three_distinct_misses_after_a_proving_write_trigger_once() -> None:
    outcomes = await _apply(
        InMemoryStreakStore(), _proof(), _miss("a", _START_MS + 1), _miss("b", _START_MS + 2), _miss("c", _START_MS + 3)
    )
    assert outcomes == (
        Unproven(),
        Counting(1),
        Counting(2),
        Triggered(misses=MISSES_TO_OPEN_INCIDENT, proven_write_tokens=2048, first_miss_ms=_START_MS + 1),
    )


async def test_a_request_logged_twice_counts_once() -> None:
    outcomes = await _apply(InMemoryStreakStore(), _proof(), _miss("a"), _miss("a"), _miss("b"), _miss("b"))
    assert outcomes[1:] == (Counting(1), Counting(1), Counting(2), Counting(2))


@pytest.mark.parametrize("breaker", [StreakEvent.HIT, StreakEvent.UNKNOWN])
async def test_a_hit_or_unknown_ends_the_streak_without_losing_the_proof(breaker: StreakEvent) -> None:
    outcomes = await _apply(
        InMemoryStreakStore(),
        _proof(),
        _miss("a"),
        _miss("b"),
        _update(breaker, "x"),
        _miss("c"),
        _miss("d"),
        _miss("e"),
    )
    assert outcomes[3:] == (Unproven(), Counting(1), Counting(2), Triggered(3, 2048, _START_MS + 1_000))


async def test_misses_on_a_never_proven_prefix_are_unproven() -> None:
    outcomes = await _apply(InMemoryStreakStore(), _miss("a"), _miss("b"), _miss("c"), _miss("d"))
    assert outcomes == (Unproven(),) * 4


async def test_proof_expires_with_the_cache_entry_it_proved() -> None:
    expired = _START_MS + _TTL_MS
    outcomes = await _apply(InMemoryStreakStore(), _proof(), _miss("a", expired), _miss("b", expired + 1))
    assert outcomes[1:] == (Unproven(), Unproven())


async def test_a_hit_keeps_the_entry_alive_for_another_ttl() -> None:
    refreshed = _START_MS + _TTL_MS - 1
    outcomes = await _apply(
        InMemoryStreakStore(),
        _proof(),
        _update(StreakEvent.HIT, "read", at_ms=refreshed),
        _miss("a", refreshed + _TTL_MS - 3),
        _miss("b", refreshed + _TTL_MS - 2),
        _miss("c", refreshed + _TTL_MS - 1),
    )
    assert outcomes[-1] == Triggered(3, 2048, refreshed + _TTL_MS - 3)


async def test_triggering_starts_a_new_streak_rather_than_firing_on_every_later_miss() -> None:
    outcomes = await _apply(InMemoryStreakStore(), _proof(), _miss("a"), _miss("b"), _miss("c"), _miss("d"))
    assert outcomes[-1] == Counting(1)


async def test_a_new_configuration_never_borrows_another_configurations_proof() -> None:
    upgraded = StreakKey(deployment_id="claude-east", config_fingerprint="config-2", prefix_fingerprint="prefix-1")
    outcomes = await _apply(
        InMemoryStreakStore(),
        _proof(),
        *(_update(StreakEvent.MISS, call_id, at_ms=_START_MS + 1, key=upgraded) for call_id in ("a", "b", "c")),
    )
    assert outcomes[1:] == (Unproven(),) * 3


async def test_concurrent_misses_from_different_requests_trigger_exactly_once() -> None:
    store = InMemoryStreakStore()
    await store.apply(_proof())
    outcomes = await asyncio.gather(*(store.apply(_miss(call_id)) for call_id in ("a", "b", "c", "a", "b")))
    assert sum(isinstance(outcome, Triggered) for outcome in outcomes) == 1


class _ScriptCall:
    def __init__(self, reply: object) -> None:
        self.reply = reply
        self.calls: tuple[tuple[tuple[str, ...], tuple[str | int, ...]], ...] = ()

    async def __call__(self, keys: Sequence[str], args: Sequence[str | int]) -> object:
        self.calls = (*self.calls, (tuple(keys), tuple(args)))
        return self.reply


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ([b"unproven"], Unproven()),
        ([b"counting", 2], Counting(2)),
        ([b"triggered", 3, b"2048", b"1700000000001"], Triggered(3, 2048, 1_700_000_000_001)),
        (None, Unproven()),
        ([b"triggered", 3], Unproven()),
    ],
)
async def test_redis_replies_are_read_back_and_anything_unexpected_is_unproven(
    reply: object, expected: StreakOutcome
) -> None:
    assert await RedisStreakStore(_ScriptCall(reply)).apply(_miss("a")) == expected


async def test_redis_keys_share_one_cluster_slot_and_carry_the_whole_update() -> None:
    script = _ScriptCall([b"unproven"])
    await RedisStreakStore(script).apply(_miss("a"))
    ((keys, args),) = script.calls
    slots = {key[key.index("{") : key.index("}") + 1] for key in keys}
    assert len(keys) == 2 and len(slots) == 1
    assert args == ("miss", _START_MS + 1_000, _TTL_MS, "a", 0, MISSES_TO_OPEN_INCIDENT)
