import asyncio
import hashlib
import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Final, Literal

import pytest

from litellm.caching.redis_cache import RedisCache
from litellm.integrations.prompt_cache_alerts.detector import (
    MISSES_TO_ALERT,
    STREAK_SCRIPT,
    TTL_MS,
    Assessed,
    DeploymentReset,
    RunScript,
    Triggered,
    Update,
    _keys,
    apply,
    quiet,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")

_T0: Final = 1_700_000_000_000
_DEADLINES: Final = "return {redis.call('PEXPIRETIME', KEYS[1]), redis.call('PEXPIRETIME', KEYS[2])}"


@dataclass(frozen=True, slots=True)
class _Streaks:
    run: RunScript
    deployment_id: str
    deadlines: RunScript

    def event(
        self,
        event: Literal["proof", "hit", "miss"],
        call_id: str,
        started_ms: int,
        ended_ms: int | None = None,
        config: str = "config-1",
        deployment_id: str | None = None,
    ) -> Assessed:
        return Assessed(
            deployment_id=deployment_id or self.deployment_id,
            config_fingerprint=config,
            model="claude-sonnet-5-5",
            prefix_fingerprint="prefix-1",
            event=event,
            call_id=call_id,
            started_ms=started_ms,
            ended_ms=started_ms if ended_ms is None else ended_ms,
            write_tokens=2048 if event == "proof" else 0,
        )

    async def apply(self, *updates: Update, now_ms: int | None = None) -> tuple[Triggered | None, ...]:
        return tuple(
            [await apply(self.run, update, update.ended_ms if now_ms is None else now_ms) for update in updates]
        )

    async def proof(self, started_ms: int = _T0) -> None:
        await self.apply(self.event("proof", "proof", started_ms))

    def misses(self, *call_ids: str, start: int = _T0 + 1_000, config: str = "config-1") -> tuple[Assessed, ...]:
        return tuple(
            self.event("miss", call_id, start + index, config=config) for index, call_id in enumerate(call_ids)
        )


@pytest.fixture
async def streaks() -> AsyncIterator[_Streaks]:
    """Each test uses fresh deployment ids, so teardown removes exactly the keys it created and closes its client."""
    cache: Final = RedisCache(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))
    deployment_id: Final = f"claude-{uuid.uuid4()}"
    yield _Streaks(cache.async_register_script(STREAK_SCRIPT), deployment_id, cache.async_register_script(_DEADLINES))
    client: Final = cache.init_async_client()
    tags: Final = tuple(
        hashlib.sha256(deployment.encode()).hexdigest() for deployment in (deployment_id, f"{deployment_id}-other")
    )
    for tag in tags:
        owned: Final = [key async for key in client.scan_iter(match=f"*{tag}*")]
        if owned:
            await client.delete(*owned)
    await cache.disconnect()


async def test_three_distinct_misses_after_a_proving_write_trigger_once(streaks: _Streaks) -> None:
    await streaks.proof()
    outcomes: Final = await streaks.apply(*streaks.misses("a", "b", "c"))
    assert outcomes == (None, None, Triggered(MISSES_TO_ALERT, 2048, _T0 + 1_000))


async def test_requests_already_in_flight_while_the_proving_write_was_made_never_count(streaks: _Streaks) -> None:
    """The proving request starts at 0 and returns its write at 10. Requests that started at 1, 2 and 3 may have
    reached the provider before the entry existed, so their zero counters say nothing about caching stopping."""
    await streaks.apply(streaks.event("proof", "proof", _T0, ended_ms=_T0 + 10_000))
    in_flight: Final = tuple(
        streaks.event("miss", call_id, _T0 + index, ended_ms=_T0 + 11_000 + index)
        for index, call_id in enumerate(("a", "b", "c"), start=1)
    )
    assert await streaks.apply(*in_flight) == (None,) * 3
    outcomes: Final = await streaks.apply(*streaks.misses("d", "e", "f", start=_T0 + 10_001))
    assert outcomes[-1] == Triggered(MISSES_TO_ALERT, 2048, _T0 + 10_001)


async def test_a_hit_ends_every_miss_that_started_before_it_returned(streaks: _Streaks) -> None:
    await streaks.proof()
    a, b = streaks.misses("a", "b", start=_T0 + 2_000)
    await streaks.apply(a, b, streaks.event("hit", "x", _T0 + 1_000, ended_ms=_T0 + 3_000))
    assert await streaks.apply(*streaks.misses("c", "d", start=_T0 + 3_001)) == (None, None)


async def test_a_request_logged_twice_counts_once(streaks: _Streaks) -> None:
    await streaks.proof()
    a, b = streaks.misses("a", "b")
    assert await streaks.apply(a, a, b, b) == (None,) * 4


async def test_misses_replayed_after_an_incident_never_count_toward_another(streaks: _Streaks) -> None:
    await streaks.proof()
    misses: Final = streaks.misses("a", "b", "c")
    await streaks.apply(*misses)
    assert await streaks.apply(*misses, *streaks.misses("d", "e", start=_T0 + 2_000)) == (None,) * 5


async def test_a_deployment_reset_ends_every_configurations_streak_and_replays_never_count_again(
    streaks: _Streaks,
) -> None:
    await streaks.proof()
    await streaks.apply(streaks.event("proof", "proof-2", _T0, config="config-2"))
    a, b = streaks.misses("a", "b")
    other: Final = streaks.misses("x", "y", config="config-2")
    await streaks.apply(a, b, *other, DeploymentReset(streaks.deployment_id, _T0 + 1_500))
    late: Final = streaks.misses("c", start=_T0 + 1_600)
    assert (
        await streaks.apply(a, b, *other, *late, *streaks.misses("z", start=_T0 + 1_600, config="config-2"))
        == (None,) * 6
    )


async def test_a_deployment_reset_stays_inside_its_deployment(streaks: _Streaks) -> None:
    await streaks.proof()
    await streaks.apply(DeploymentReset(f"{streaks.deployment_id}-other", _T0 + 1_500))
    outcomes: Final = await streaks.apply(*streaks.misses("a", "b", "c"))
    assert outcomes[-1] == Triggered(MISSES_TO_ALERT, 2048, _T0 + 1_000)


async def test_a_reset_before_the_misses_leaves_them_counting(streaks: _Streaks) -> None:
    await streaks.proof()
    await streaks.apply(DeploymentReset(streaks.deployment_id, _T0 + 500))
    outcomes: Final = await streaks.apply(*streaks.misses("a", "b", "c"))
    assert outcomes[-1] == Triggered(MISSES_TO_ALERT, 2048, _T0 + 1_000)


async def test_a_hit_delivered_late_only_ends_misses_that_started_before_it_returned(streaks: _Streaks) -> None:
    await streaks.proof()
    early, a = streaks.misses("early", "a")
    await streaks.apply(early, a)
    await streaks.apply(streaks.event("hit", "x", early.started_ms), now_ms=_T0 + 5_000)
    assert await streaks.apply(*streaks.misses("c", "d", start=_T0 + 3_000)) == (
        None,
        Triggered(MISSES_TO_ALERT, 2048, a.started_ms),
    )


async def test_a_proof_replayed_after_its_entry_expired_does_not_prove_the_prefix_again(streaks: _Streaks) -> None:
    after_expiry: Final = _T0 + TTL_MS + 10
    replayed: Final = streaks.event("proof", "proof", _T0)
    misses: Final = streaks.misses("a", "b", "c", start=after_expiry)
    assert await streaks.apply(replayed, *misses, now_ms=after_expiry + 5) == (None,) * 4


async def test_a_miss_that_started_before_the_proof_does_not_count(streaks: _Streaks) -> None:
    await streaks.proof(_T0 + 1_000)
    assert await streaks.apply(*streaks.misses("a", start=_T0), now_ms=_T0 + 2_000) == (None,)


async def test_proof_expires_with_the_cache_entry_it_proved_and_misses_never_extend_it(streaks: _Streaks) -> None:
    await streaks.proof()
    await streaks.apply(*streaks.misses("a", "b", start=_T0 + TTL_MS - 10))
    assert await streaks.apply(*streaks.misses("c", start=_T0 + TTL_MS)) == (None,)


async def test_a_hit_keeps_the_entry_alive_for_another_ttl(streaks: _Streaks) -> None:
    await streaks.proof()
    refreshed: Final = _T0 + TTL_MS - 1
    await streaks.apply(streaks.event("hit", "read", refreshed))
    outcomes: Final = await streaks.apply(*streaks.misses("a", "b", "c", start=refreshed + TTL_MS - 4))
    assert outcomes[-1] == Triggered(MISSES_TO_ALERT, 2048, refreshed + TTL_MS - 4)


async def test_a_hit_delivered_after_the_stored_proof_expired_leaves_the_prefix_unproven(streaks: _Streaks) -> None:
    """The hit started while the entry was live, but its callback arrived after the stored expiry. The detector does
    not revive the proof from a late callback, so later misses stay unassessed until a new write proves the prefix."""
    await streaks.proof()
    expired: Final = _T0 + TTL_MS
    await streaks.apply(streaks.event("hit", "late-read", expired - 1), now_ms=expired + 5)
    assert await streaks.apply(*streaks.misses("a", "b", "c", start=expired + 10)) == (None,) * 3
    await streaks.proof(expired + 20)
    outcomes: Final = await streaks.apply(*streaks.misses("d", "e", "f", start=expired + 21))
    assert outcomes[-1] == Triggered(MISSES_TO_ALERT, 2048, expired + 21)


@pytest.mark.parametrize("refresh", ["hit", "proof"])
async def test_a_late_refresh_keeps_the_misses_after_it_alive_as_long_as_the_proof(
    streaks: _Streaks, refresh: Literal["hit", "proof"]
) -> None:
    """Misses stored 20s before the proof expired expire with it. A refresh that returned before those misses but
    arrived after them extends the proof, so the misses it keeps must expire at the same moment as the proof."""
    await streaks.proof()
    expiry: Final = _T0 + TTL_MS
    a, b, c = streaks.misses("a", "b", "c", start=expiry - 20_000)
    await streaks.apply(a, b)
    late: Final = streaks.event(refresh, "late", expiry - 30_000, ended_ms=expiry - 25_000)
    await streaks.apply(late, now_ms=expiry - 15_000)
    _, state, misses, _ = _keys(streaks.deployment_id, "config-1", "prefix-1")
    match await streaks.deadlines((state, misses), ()):
        case [int(state_deadline), int(misses_deadline)]:
            assert misses_deadline == state_deadline > 0
        case reply:
            pytest.fail(f"unexpected deadlines {reply!r}")
    assert await streaks.apply(c, now_ms=expiry - 10_000) == (Triggered(MISSES_TO_ALERT, 2048, a.started_ms),)


async def test_a_new_configuration_never_borrows_another_configurations_proof(streaks: _Streaks) -> None:
    await streaks.proof()
    assert await streaks.apply(*streaks.misses("a", "b", "c", config="config-3")) == (None,) * 3


async def test_a_saved_alert_keeps_its_configuration_quiet_and_an_unsaved_one_triggers_again(
    streaks: _Streaks,
) -> None:
    await streaks.proof()
    (*_, first) = await streaks.apply(*streaks.misses("a", "b", "c"))
    assert first is not None
    (*_, unsaved_retry) = await streaks.apply(*streaks.misses("d", "e", "f", start=_T0 + 2_000))
    assert unsaved_retry is not None
    await quiet(streaks.run, streaks.event("miss", "f", _T0 + 2_002), _T0 + 2_002)
    assert await streaks.apply(*streaks.misses("g", "h", "i", start=_T0 + 3_000)) == (None,) * 3


async def test_concurrent_misses_from_different_requests_trigger_exactly_once(streaks: _Streaks) -> None:
    await streaks.proof()
    a, b, c = streaks.misses("a", "b", "c")
    outcomes: Final = await asyncio.gather(*(apply(streaks.run, miss, _T0 + 2_000) for miss in (a, b, c, a, b)))
    assert sum(isinstance(outcome, Triggered) for outcome in outcomes) == 1
