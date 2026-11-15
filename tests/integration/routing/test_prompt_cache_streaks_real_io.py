import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from typing import Final

import pytest

from litellm.caching.redis_cache import RedisCache
from litellm.router_utils.prompt_cache_observations.streaks import (
    MISSES_TO_OPEN_INCIDENT,
    STREAK_SCRIPT,
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

_TTL_MS: Final = 300_000
_T0: Final = 1_700_000_000_000


class _Streaks:
    def __init__(self, store: RedisStreakStore, scope: StreakScope) -> None:
        self.store = store
        self.scope = scope

    def event(
        self,
        event: StreakEvent,
        call_id: str,
        started_ms: int,
        prefix: str = "prefix-1",
        scope: StreakScope | None = None,
    ) -> PrefixEvent:
        return PrefixEvent(
            scope=scope or self.scope,
            prefix_fingerprint=prefix,
            event=event,
            call_id=call_id,
            started_ms=started_ms,
            ttl_ms=_TTL_MS,
            write_tokens=2048 if event is StreakEvent.PROOF else 0,
        )

    async def apply(self, *updates: StreakUpdate, now_ms: int | None = None) -> tuple[StreakOutcome, ...]:
        return tuple(
            [await self.store.apply(update, update.started_ms if now_ms is None else now_ms) for update in updates]
        )

    async def proof(self, started_ms: int = _T0) -> None:
        await self.apply(self.event(StreakEvent.PROOF, "proof", started_ms))

    def misses(self, *call_ids: str, start: int = _T0 + 1_000) -> tuple[PrefixEvent, ...]:
        return tuple(self.event(StreakEvent.MISS, call_id, start + index) for index, call_id in enumerate(call_ids))


@pytest.fixture
async def streaks() -> AsyncIterator[_Streaks]:
    cache: Final = RedisCache(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))
    scope: Final = StreakScope(deployment_id=f"claude-{uuid.uuid4()}", config_fingerprint="config-1")
    yield _Streaks(RedisStreakStore(cache.async_register_script(STREAK_SCRIPT)), scope)
    client: Final = cache.init_async_client()
    stale: Final = [key async for key in client.scan_iter(match=f"*{scope.slot()[len('prompt_cache_streak:') :]}*")]
    if stale:
        await client.delete(*stale)


async def test_three_distinct_misses_after_a_proving_write_trigger_once(streaks: _Streaks) -> None:
    await streaks.proof()
    outcomes = await streaks.apply(*streaks.misses("a", "b", "c"))
    assert outcomes == (Counting(1), Counting(2), Triggered(MISSES_TO_OPEN_INCIDENT, 2048, _T0 + 1_000))


async def test_a_request_logged_twice_counts_once(streaks: _Streaks) -> None:
    await streaks.proof()
    a, b = streaks.misses("a", "b")
    assert await streaks.apply(a, a, b, b) == (Counting(1), Counting(1), Counting(2), Counting(2))


async def test_misses_replayed_after_an_incident_never_count_toward_another(streaks: _Streaks) -> None:
    await streaks.proof()
    misses = streaks.misses("a", "b", "c")
    await streaks.apply(*misses)
    assert await streaks.apply(*misses) == (Unproven(),) * 3
    assert await streaks.apply(*streaks.misses("d", start=_T0 + 2_000)) == (Counting(1),)


@pytest.mark.parametrize("breaker", [StreakEvent.HIT, StreakEvent.UNKNOWN])
async def test_misses_replayed_after_a_reset_never_count_again(streaks: _Streaks, breaker: StreakEvent) -> None:
    await streaks.proof()
    a, b = streaks.misses("a", "b")
    await streaks.apply(a, b, streaks.event(breaker, "x", _T0 + 1_500))
    assert await streaks.apply(a, b, *streaks.misses("c", start=_T0 + 1_600)) == (Unproven(), Unproven(), Counting(1))


async def test_a_reset_delivered_late_only_ends_misses_that_started_before_it(streaks: _Streaks) -> None:
    await streaks.proof()
    early, a = streaks.misses("early", "a")
    await streaks.apply(early, a)
    await streaks.apply(streaks.event(StreakEvent.HIT, "x", early.started_ms), now_ms=_T0 + 5_000)
    assert await streaks.apply(*streaks.misses("c", "d", start=_T0 + 3_000)) == (
        Counting(2),
        Triggered(MISSES_TO_OPEN_INCIDENT, 2048, a.started_ms),
    )


async def test_a_proof_replayed_after_its_entry_expired_does_not_prove_the_prefix_again(streaks: _Streaks) -> None:
    after_expiry = _T0 + _TTL_MS + 10
    replayed = streaks.event(StreakEvent.PROOF, "proof", _T0)
    misses = streaks.misses("a", "b", "c", start=after_expiry)
    assert await streaks.apply(replayed, *misses, now_ms=after_expiry + 5) == (Unproven(),) * 4


async def test_a_miss_that_started_before_the_proof_does_not_count(streaks: _Streaks) -> None:
    await streaks.proof(_T0 + 1_000)
    assert await streaks.apply(*streaks.misses("a", start=_T0), now_ms=_T0 + 2_000) == (Unproven(),)


async def test_proof_expires_with_the_cache_entry_it_proved(streaks: _Streaks) -> None:
    await streaks.proof()
    assert await streaks.apply(*streaks.misses("a", "b", "c", start=_T0 + _TTL_MS)) == (Unproven(),) * 3


async def test_a_hit_keeps_the_entry_alive_for_another_ttl(streaks: _Streaks) -> None:
    await streaks.proof()
    refreshed = _T0 + _TTL_MS - 1
    await streaks.apply(streaks.event(StreakEvent.HIT, "read", refreshed))
    outcomes = await streaks.apply(*streaks.misses("a", "b", "c", start=refreshed + _TTL_MS - 4))
    assert outcomes[-1] == Triggered(MISSES_TO_OPEN_INCIDENT, 2048, refreshed + _TTL_MS - 4)


async def test_an_unreadable_caching_request_ends_every_streak_of_its_scope(streaks: _Streaks) -> None:
    await streaks.proof()
    a, b = streaks.misses("a", "b")
    await streaks.apply(a, b, ScopeBreak(scope=streaks.scope, started_ms=_T0 + 1_500))
    assert await streaks.apply(*streaks.misses("c", start=_T0 + 1_600)) == (Counting(1),)


async def test_a_scope_break_before_the_misses_leaves_them_counting(streaks: _Streaks) -> None:
    await streaks.proof()
    await streaks.apply(ScopeBreak(scope=streaks.scope, started_ms=_T0 + 500))
    outcomes = await streaks.apply(*streaks.misses("a", "b", "c"))
    assert outcomes[-1] == Triggered(MISSES_TO_OPEN_INCIDENT, 2048, _T0 + 1_000)


async def test_a_scope_break_stays_inside_its_deployment_configuration(streaks: _Streaks) -> None:
    await streaks.proof()
    other = StreakScope(deployment_id=streaks.scope.deployment_id, config_fingerprint="config-2")
    await streaks.apply(ScopeBreak(scope=other, started_ms=_T0 + 1_500))
    outcomes = await streaks.apply(*streaks.misses("a", "b", "c", start=_T0 + 1_000))
    assert outcomes[-1] == Triggered(MISSES_TO_OPEN_INCIDENT, 2048, _T0 + 1_000)


async def test_a_new_configuration_never_borrows_another_configurations_proof(streaks: _Streaks) -> None:
    await streaks.proof()
    upgraded = StreakScope(deployment_id=streaks.scope.deployment_id, config_fingerprint="config-3")
    misses = tuple(streaks.event(StreakEvent.MISS, call_id, _T0 + 1_000, scope=upgraded) for call_id in "abc")
    assert await streaks.apply(*misses) == (Unproven(),) * 3


async def test_concurrent_misses_from_different_requests_trigger_exactly_once(streaks: _Streaks) -> None:
    await streaks.proof()
    a, b, c = streaks.misses("a", "b", "c")
    outcomes = await asyncio.gather(*(streaks.store.apply(miss, _T0 + 2_000) for miss in (a, b, c, a, b)))
    assert sum(isinstance(outcome, Triggered) for outcome in outcomes) == 1
