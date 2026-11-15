from collections.abc import Sequence

import pytest

from litellm.router_utils.prompt_cache_observations.streaks import (
    MISSES_TO_OPEN_INCIDENT,
    SCOPE_BREAK_TTL_MS,
    Counting,
    PrefixEvent,
    RedisStreakStore,
    ScopeBreak,
    StreakEvent,
    StreakOutcome,
    StreakScope,
    StreakUpdate,
    Triggered,
    Unproven,
)

_SCOPE = StreakScope(deployment_id="claude-east", config_fingerprint="config-1")
_NOW_MS = 1_700_000_005_000


def _miss(prefix: str = "prefix-1", scope: StreakScope = _SCOPE) -> PrefixEvent:
    return PrefixEvent(
        scope=scope,
        prefix_fingerprint=prefix,
        event=StreakEvent.MISS,
        call_id="a",
        started_ms=1_700_000_001_000,
        ttl_ms=300_000,
    )


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
        ([b"triggered", 3, b"2048", b"1.700000000001e+12"], Triggered(3, 2048, 1_700_000_000_001)),
        (None, Unproven()),
        ([b"triggered", 3], Unproven()),
        ([b"triggered", 3, b"", b"1"], Unproven()),
    ],
)
async def test_redis_replies_are_read_back_and_anything_unexpected_is_unproven(
    reply: object, expected: StreakOutcome
) -> None:
    assert await RedisStreakStore(_ScriptCall(reply)).apply(_miss(), _NOW_MS) == expected


async def _keys(*updates: StreakUpdate) -> tuple[tuple[str, ...], ...]:
    script = _ScriptCall([b"unproven"])
    store = RedisStreakStore(script)
    for update in updates:
        await store.apply(update, _NOW_MS)
    return tuple(keys for keys, _ in script.calls)


def _slots(keys: Sequence[str]) -> set[str]:
    return {key[key.index("{") : key.index("}") + 1] for key in keys}


async def test_a_scope_break_reaches_the_cluster_slot_of_every_prefix_in_its_scope() -> None:
    first, second, broken = await _keys(_miss("prefix-1"), _miss("prefix-2"), ScopeBreak(_SCOPE, 1_700_000_002_000))
    assert len(_slots((*first, *second, *broken))) == 1
    assert first[0] == second[0] == broken[0]
    assert first[1:] != second[1:]


async def test_each_deployment_configuration_has_its_own_streaks() -> None:
    upgraded = StreakScope(deployment_id="claude-east", config_fingerprint="config-2")
    ours, theirs = await _keys(_miss(), _miss(scope=upgraded))
    assert _slots(ours).isdisjoint(_slots(theirs))


@pytest.mark.parametrize(
    ("update", "args"),
    [
        (_miss(), ("miss", 1_700_000_001_000, _NOW_MS, 300_000, "a", 0, MISSES_TO_OPEN_INCIDENT, SCOPE_BREAK_TTL_MS)),
        (
            ScopeBreak(_SCOPE, 1_700_000_002_000),
            ("break", 1_700_000_002_000, _NOW_MS, 0, "", 0, MISSES_TO_OPEN_INCIDENT, SCOPE_BREAK_TTL_MS),
        ),
    ],
)
async def test_the_script_gets_the_requests_own_start_and_the_workers_clock(
    update: StreakUpdate, args: tuple[str | int, ...]
) -> None:
    script = _ScriptCall([b"unproven"])
    await RedisStreakStore(script).apply(update, _NOW_MS)
    ((_, sent),) = script.calls
    assert sent == args
