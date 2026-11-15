from datetime import datetime, timedelta, timezone

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from litellm.models.alert import Alert, AlertDetection, PromptCacheEvidence, RecordedAlert
from litellm.router_utils.prompt_cache_observations.incidents import (
    IncidentSlot,
    PromptCacheIncidentDetector,
    PromptCacheIncidents,
)
from litellm.router_utils.prompt_cache_observations.streaks import (
    Counting,
    PrefixEvent,
    ScopeBreak,
    StreakEvent,
    StreakOutcome,
    StreakScope,
    StreakUpdate,
    Triggered,
    Unproven,
)
from litellm.router_utils.prompt_cache_observations.types import (
    Assessed,
    CacheablePrefix,
    CacheCounters,
    CacheIntent,
    ClientSurface,
    DeploymentCacheIdentity,
    ObservationPath,
    PromptCacheObservation,
    ProviderFamily,
    Runtime,
    Unassessed,
    UnassessedReason,
    UpstreamCacheUsage,
    UsageUnknown,
    UsageUnknownReason,
    WireShape,
)

_START = 1_700_000_000.0
_DURATION = 0.5
_PATH = ObservationPath(
    provider=ProviderFamily.ANTHROPIC,
    wire=WireShape.ANTHROPIC_MESSAGES,
    surface=ClientSurface.CHAT,
    stream=False,
    runtime=Runtime.PYTHON,
)
_PREFIX = CacheablePrefix(fingerprint="prefix-1", ttl_seconds=300, breakpoints=1)


def _deployment(config: str = "config-1") -> DeploymentCacheIdentity:
    return DeploymentCacheIdentity(
        deployment_id="claude-east",
        config_fingerprint=config,
        provider="anthropic",
        provider_model="claude-test",
        min_prefix_tokens=1024,
    )


def _observation(
    call_id: str,
    usage: UpstreamCacheUsage,
    at: float,
    prefix: CacheablePrefix = _PREFIX,
    config: str = "config-1",
) -> PromptCacheObservation:
    return PromptCacheObservation(
        call_id=call_id,
        deployment=_deployment(config),
        intent=CacheIntent.CLIENT,
        started_at=at,
        ended_at=at + _DURATION,
        outcome=Assessed(path=_PATH, prefix=prefix, usage=usage),
    )


_WRITE = CacheCounters(read_tokens=0, write_tokens=2048)
_READ = CacheCounters(read_tokens=2048, write_tokens=0)
_MISS = CacheCounters(read_tokens=0, write_tokens=0)
_SCOPE = StreakScope(deployment_id="claude-east", config_fingerprint="config-1")
_CLOCK_MS = int(_START * 1000) + 60_000


def _unassessed(call_id: str, at: float, prefix: CacheablePrefix | None) -> PromptCacheObservation:
    return PromptCacheObservation(
        call_id=call_id,
        deployment=_deployment(),
        intent=CacheIntent.CLIENT,
        started_at=at,
        ended_at=at + _DURATION,
        outcome=Unassessed(UnassessedReason.REQUEST_BODY_UNAVAILABLE, path=_PATH, prefix=prefix),
    )


def _event(event: StreakEvent, call_id: str, at: float, write_tokens: int = 0) -> PrefixEvent:
    return PrefixEvent(
        scope=_SCOPE,
        prefix_fingerprint="prefix-1",
        event=event,
        call_id=call_id,
        started_ms=int(at * 1000),
        ended_ms=int((at + _DURATION) * 1000),
        ttl_ms=300_000,
        write_tokens=write_tokens,
    )


class _Recorder:
    def __init__(self) -> None:
        self.detections: tuple[AlertDetection, ...] = ()

    async def open(self, detection: AlertDetection) -> None:
        self.detections = (*self.detections, detection)


class _ScriptedStore:
    """Answers with scripted outcomes, and fails the calls listed in `failing` by their position."""

    def __init__(self, outcomes: tuple[StreakOutcome, ...] = (), failing: frozenset[int] = frozenset()) -> None:
        self.outcomes = outcomes
        self.failing = failing
        self.calls = 0
        self.applied: tuple[tuple[StreakUpdate, int], ...] = ()

    async def apply(self, update: StreakUpdate, now_ms: int) -> StreakOutcome:
        position = self.calls
        self.calls += 1
        if position in self.failing:
            raise RedisConnectionError("redis is down")
        self.applied = (*self.applied, (update, now_ms))
        return self.outcomes[len(self.applied) - 1] if len(self.applied) <= len(self.outcomes) else Unproven()


def _detector(recorder: _Recorder | None, store: _ScriptedStore) -> PromptCacheIncidentDetector:
    slot = IncidentSlot()
    slot.attach(recorder)
    return PromptCacheIncidentDetector(store, slot.current, clock=lambda: _CLOCK_MS / 1000)


async def _feed(detector: PromptCacheIncidentDetector, *observations: PromptCacheObservation) -> None:
    for observation in observations:
        await detector.record(observation)


@pytest.mark.parametrize(
    ("observation", "update"),
    [
        (_observation("w", _WRITE, _START), _event(StreakEvent.PROOF, "w", _START, 2048)),
        (_observation("r", _READ, _START), _event(StreakEvent.HIT, "r", _START)),
        (_observation("m", _MISS, _START), _event(StreakEvent.MISS, "m", _START)),
        (_observation("u", UsageUnknown(UsageUnknownReason.MISSING), _START), _event(StreakEvent.UNKNOWN, "u", _START)),
        (
            PromptCacheObservation(
                call_id="f",
                deployment=_deployment(),
                intent=CacheIntent.CLIENT,
                started_at=_START,
                ended_at=_START + _DURATION,
                outcome=Unassessed(UnassessedReason.UPSTREAM_FAILED, path=_PATH, prefix=_PREFIX),
            ),
            _event(StreakEvent.UNKNOWN, "f", _START),
        ),
        (_unassessed("x", _START + 0.25, None), ScopeBreak(_SCOPE, int((_START + 0.25 + _DURATION) * 1000))),
        (
            _observation("w2", _WRITE, _START, prefix=CacheablePrefix("prefix-1", 300, breakpoints=2)),
            _event(StreakEvent.HIT, "w2", _START),
        ),
    ],
    ids=["proving-write", "read", "miss", "unknown-usage", "failed-attempt", "unreadable-prefix", "multi-breakpoint"],
)
async def test_each_observation_reaches_its_streak_with_the_requests_own_start_and_end(
    observation: PromptCacheObservation, update: StreakUpdate
) -> None:
    store = _ScriptedStore()
    await _detector(_Recorder(), store).record(observation)
    assert store.applied == ((update, _CLOCK_MS),)


_TRIGGERED = Triggered(misses=3, proven_write_tokens=2048, first_miss_ms=int((_START + 10) * 1000))


async def test_a_triggered_streak_opens_one_incident_for_the_deployment_configuration() -> None:
    recorder = _Recorder()
    store = _ScriptedStore((Unproven(), Counting(1), Counting(2), _TRIGGERED))
    await _feed(_detector(recorder, store), _observation("w", _WRITE, _START), *_misses("a", "b", "c"))
    assert recorder.detections == (
        AlertDetection(
            type="prompt_cache",
            resource_type="deployment",
            resource_id="claude-east",
            window_start=datetime.fromtimestamp(_START + 10, tz=timezone.utc),
            window_end=datetime.fromtimestamp(_START + 12 + _DURATION, tz=timezone.utc) + timedelta(milliseconds=1),
            evidence=PromptCacheEvidence(
                config_fingerprint="config-1",
                prefix_fingerprint="prefix-1",
                provider="anthropic",
                provider_model="claude-test",
                consecutive_misses=3,
                proven_write_tokens=2048,
                ttl_seconds=300,
            ),
        ),
    )


def _misses(*call_ids: str, start: float = _START + 10) -> tuple[PromptCacheObservation, ...]:
    return tuple(_observation(call_id, _MISS, start + index) for index, call_id in enumerate(call_ids))


async def test_without_an_attached_recorder_a_trigger_saves_nothing() -> None:
    store = _ScriptedStore((_TRIGGERED,))
    await _feed(_detector(None, store), *_misses("c"))
    assert store.calls == 1


async def test_evidence_redis_lost_ends_its_scopes_streaks_before_anything_else_counts() -> None:
    """A hit lost to a Redis error may be what ended the streak, so the next update first breaks the scope as of
    that hit's end. Without it, miss, miss, lost hit, miss would open an incident."""
    store = _ScriptedStore(failing=frozenset({1}))
    hit = _observation("r", _READ, _START + 11)
    await _feed(_detector(_Recorder(), store), *_misses("a"), hit, *_misses("c", start=_START + 12))
    assert [update for update, _ in store.applied] == [
        _event(StreakEvent.MISS, "a", _START + 10),
        ScopeBreak(_SCOPE, int((_START + 11 + _DURATION) * 1000)),
        _event(StreakEvent.MISS, "c", _START + 12),
    ]


async def test_while_the_scope_break_cannot_be_delivered_no_later_evidence_counts() -> None:
    store = _ScriptedStore(failing=frozenset({0, 1}))
    await _feed(_detector(_Recorder(), store), _observation("r", _READ, _START + 11), *_misses("c", "d"))
    assert [update for update, _ in store.applied] == [
        ScopeBreak(_SCOPE, int((_START + 11 + _DURATION) * 1000)),
        _event(StreakEvent.MISS, "d", _START + 11),
    ]


async def test_redis_trouble_skips_the_evidence_instead_of_failing_the_request() -> None:
    recorder = _Recorder()
    store = _ScriptedStore(failing=frozenset({0, 1, 2}))
    await _feed(_detector(recorder, store), *_misses("a", "b", "c"))
    assert (recorder.detections, store.applied) == ((), ())


def _alert(detection: AlertDetection, status: str = "open") -> Alert:
    return Alert.model_validate(
        {
            **detection.model_dump(),
            "id": "7",
            "status": status,
            "created_at": detection.window_start,
            "updated_at": detection.window_end,
        }
    )


class _Store:
    def __init__(self, created: bool) -> None:
        self.created = created
        self.recorded: tuple[AlertDetection, ...] = ()

    async def record_once(self, detection: AlertDetection) -> RecordedAlert:
        self.recorded = (*self.recorded, detection)
        return RecordedAlert(alert=_alert(detection), created=self.created)


class _Notifier:
    def __init__(self) -> None:
        self.alerts: tuple[Alert, ...] = ()

    async def notify(self, alert: Alert) -> None:
        self.alerts = (*self.alerts, alert)


async def _opened_detection() -> AlertDetection:
    recorder = _Recorder()
    await _feed(_detector(recorder, _ScriptedStore((_TRIGGERED,))), *_misses("c"))
    (detection,) = recorder.detections
    return detection


@pytest.mark.parametrize(("created", "notified"), [(True, 1), (False, 0)])
async def test_only_the_insert_that_created_the_incident_notifies(created: bool, notified: int) -> None:
    detection = await _opened_detection()
    store, notifier = _Store(created), _Notifier()
    await PromptCacheIncidents(store, notifier).open(detection)
    assert (store.recorded, len(notifier.alerts)) == ((detection,), notified)


class _FailingStore:
    async def record_once(self, detection: AlertDetection) -> RecordedAlert:
        raise OSError("database unreachable")


async def test_database_trouble_is_logged_and_never_notifies() -> None:
    notifier = _Notifier()
    await PromptCacheIncidents(_FailingStore(), notifier).open(await _opened_detection())
    assert notifier.alerts == ()
