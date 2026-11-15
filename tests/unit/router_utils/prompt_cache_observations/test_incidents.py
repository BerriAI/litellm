from datetime import datetime, timedelta, timezone

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from litellm.models.alert import Alert, AlertDetection, PromptCacheEvidence, RecordedAlert
from litellm.router_utils.prompt_cache_observations.incidents import (
    IncidentSlot,
    PromptCacheIncidentDetector,
    PromptCacheIncidents,
)
from litellm.router_utils.prompt_cache_observations.streaks import InMemoryStreakStore, StreakOutcome, StreakUpdate
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
        observed_at=at,
        outcome=Assessed(path=_PATH, prefix=prefix, usage=usage),
    )


_WRITE = CacheCounters(read_tokens=0, write_tokens=2048)
_READ = CacheCounters(read_tokens=2048, write_tokens=0)
_MISS = CacheCounters(read_tokens=0, write_tokens=0)


class _Recorder:
    def __init__(self) -> None:
        self.detections: tuple[AlertDetection, ...] = ()

    async def open(self, detection: AlertDetection) -> None:
        self.detections = (*self.detections, detection)


def _detector(recorder: _Recorder | None) -> PromptCacheIncidentDetector:
    slot = IncidentSlot()
    slot.attach(recorder)
    return PromptCacheIncidentDetector(InMemoryStreakStore(), slot.current)


async def _feed(detector: PromptCacheIncidentDetector, *observations: PromptCacheObservation) -> None:
    for observation in observations:
        await detector.record(observation)


def _misses(*call_ids: str, start: float = _START + 10, config: str = "config-1") -> tuple[PromptCacheObservation, ...]:
    return tuple(_observation(call_id, _MISS, start + index, config=config) for index, call_id in enumerate(call_ids))


async def test_three_misses_after_a_proving_write_open_one_incident_for_the_deployment_configuration() -> None:
    recorder = _Recorder()
    await _feed(_detector(recorder), _observation("w", _WRITE, _START), *_misses("a", "b", "c"))
    assert recorder.detections == (
        AlertDetection(
            type="prompt_cache",
            resource_type="deployment",
            resource_id="claude-east",
            window_start=datetime.fromtimestamp(_START + 10, tz=timezone.utc),
            window_end=datetime.fromtimestamp(_START + 12, tz=timezone.utc) + timedelta(milliseconds=1),
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


@pytest.mark.parametrize(
    "between",
    [
        _observation("r", _READ, _START + 10.5),
        _observation("u", UsageUnknown(UsageUnknownReason.MISSING), _START + 10.5),
        PromptCacheObservation(
            call_id="f",
            deployment=_deployment(),
            intent=CacheIntent.CLIENT,
            observed_at=_START + 10.5,
            outcome=Unassessed(UnassessedReason.UPSTREAM_FAILED, path=_PATH, prefix=_PREFIX),
        ),
    ],
    ids=["positive-read", "unknown-usage", "failed-attempt"],
)
async def test_any_other_evidence_about_the_prefix_breaks_the_streak(between: PromptCacheObservation) -> None:
    recorder = _Recorder()
    await _feed(_detector(recorder), _observation("w", _WRITE, _START), *_misses("a", "b"), between, *_misses("c"))
    assert recorder.detections == ()


async def test_a_write_on_several_breakpoints_does_not_prove_the_prefix() -> None:
    recorder = _Recorder()
    several = CacheablePrefix(fingerprint="prefix-1", ttl_seconds=300, breakpoints=2)
    await _feed(
        _detector(recorder),
        _observation("w", _WRITE, _START, prefix=several),
        *(_observation(call_id, _MISS, _START + 10, prefix=several) for call_id in ("a", "b", "c")),
    )
    assert recorder.detections == ()


async def test_a_model_or_config_upgrade_needs_its_own_proof() -> None:
    recorder = _Recorder()
    await _feed(_detector(recorder), _observation("w", _WRITE, _START), *_misses("a", "b", "c", config="config-2"))
    assert recorder.detections == ()


async def test_a_never_proven_prefix_never_opens_an_incident() -> None:
    recorder = _Recorder()
    await _feed(_detector(recorder), *_misses("a", "b", "c", "d", "e"))
    assert recorder.detections == ()


async def test_without_an_attached_recorder_streaks_are_kept_and_nothing_is_saved_retroactively() -> None:
    slot = IncidentSlot()
    detector = PromptCacheIncidentDetector(InMemoryStreakStore(), slot.current)
    await _feed(detector, _observation("w", _WRITE, _START), *_misses("a", "b", "c"))
    recorder = _Recorder()
    slot.attach(recorder)
    await _feed(detector, *_misses("d", "e", start=_START + 20))
    assert recorder.detections == ()
    await _feed(detector, *_misses("f", start=_START + 30))
    assert len(recorder.detections) == 1


class _UnavailableStreaks:
    async def apply(self, update: StreakUpdate) -> StreakOutcome:
        raise RedisConnectionError("redis is down")


async def test_redis_trouble_skips_the_evidence_instead_of_failing_the_request() -> None:
    recorder = _Recorder()
    slot = IncidentSlot()
    slot.attach(recorder)
    await _feed(PromptCacheIncidentDetector(_UnavailableStreaks(), slot.current), *_misses("a", "b", "c"))
    assert recorder.detections == ()


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
    await _feed(_detector(recorder), _observation("w", _WRITE, _START), *_misses("a", "b", "c"))
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
