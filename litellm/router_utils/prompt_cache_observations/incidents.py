from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Final, Protocol

from redis.exceptions import RedisError

from litellm._logging import verbose_router_logger
from litellm.caching.redis_cache import RedisCircuitBreakerOpenError
from litellm.models.alert import Alert, AlertDetection, IncidentAlertStore, PromptCacheEvidence

from .eligibility import write_proves_prefix_eligible
from .streaks import StreakEvent, StreakKey, StreakStore, StreakUpdate, Triggered
from .types import Assessed, CacheablePrefix, CacheCounters, PromptCacheObservation, Unassessed, UsageUnknown

_STORE_FAILURES: Final = (RedisError, RedisCircuitBreakerOpenError, OSError, TimeoutError)


class IncidentNotifier(Protocol):
    async def notify(self, alert: Alert) -> None: ...


class IncidentRecorder(Protocol):
    async def open(self, detection: AlertDetection) -> None: ...


class PromptCacheIncidents:
    """Notifies only for the insert that created the incident. Every later detection of the same deployment and
    configuration finds the saved row, whatever its status, so a dismissed or resolved incident stays quiet."""

    def __init__(self, store: IncidentAlertStore, notifier: IncidentNotifier | None) -> None:
        self._store: Final = store
        self._notifier: Final = notifier

    async def open(self, detection: AlertDetection) -> None:
        from prisma.errors import PrismaError

        try:
            recorded: Final = await self._store.record_once(detection)
        except (PrismaError, OSError, TimeoutError) as error:
            verbose_router_logger.warning("prompt cache incident not saved (%s)", type(error).__name__)
            return
        if recorded.created and self._notifier is not None:
            await self._notifier.notify(recorded.alert)


class IncidentSlot:
    """The recorder the proxy attaches once its database is available. Without one, streaks are still tracked and
    nothing is saved, which is observation mode."""

    def __init__(self) -> None:
        self._recorder: IncidentRecorder | None = None

    def attach(self, recorder: IncidentRecorder | None) -> None:
        self._recorder = recorder

    def current(self) -> IncidentRecorder | None:
        return self._recorder


PROMPT_CACHE_INCIDENTS: Final = IncidentSlot()


def _event(counters: CacheCounters, prefix: CacheablePrefix) -> StreakEvent:
    if write_proves_prefix_eligible(prefix, counters):
        return StreakEvent.PROOF
    return StreakEvent.HIT if counters.active else StreakEvent.MISS


def _classified(observation: PromptCacheObservation) -> tuple[CacheablePrefix, StreakEvent, int] | None:
    """A failed, unverified or unreadable attempt on a prefix is unknown evidence, which ends its streak."""
    match observation.outcome:
        case Assessed(prefix=prefix, usage=CacheCounters() as counters):
            return prefix, _event(counters, prefix), counters.write_tokens
        case Assessed(prefix=prefix, usage=UsageUnknown()) | Unassessed(prefix=CacheablePrefix() as prefix):
            return prefix, StreakEvent.UNKNOWN, 0
        case _:
            return None


def streak_update(observation: PromptCacheObservation) -> StreakUpdate | None:
    classified: Final = _classified(observation)
    if classified is None:
        return None
    prefix, event, write_tokens = classified
    return StreakUpdate(
        key=StreakKey(
            deployment_id=observation.deployment.deployment_id,
            config_fingerprint=observation.deployment.config_fingerprint,
            prefix_fingerprint=prefix.fingerprint,
        ),
        event=event,
        call_id=observation.call_id,
        now_ms=int(observation.observed_at * 1000),
        ttl_ms=prefix.ttl_seconds * 1000,
        write_tokens=write_tokens,
    )


def _moment(epoch_ms: int) -> datetime:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)


def incident_detection(observation: PromptCacheObservation, triggered: Triggered) -> AlertDetection | None:
    outcome: Final = observation.outcome
    if not isinstance(outcome, Assessed):
        return None
    prefix: Final = outcome.prefix
    deployment: Final = observation.deployment
    start: Final = _moment(triggered.first_miss_ms)
    return AlertDetection(
        type="prompt_cache",
        resource_type="deployment",
        resource_id=deployment.deployment_id,
        window_start=start,
        window_end=max(_moment(int(observation.observed_at * 1000)), start) + timedelta(milliseconds=1),
        evidence=PromptCacheEvidence(
            config_fingerprint=deployment.config_fingerprint,
            prefix_fingerprint=prefix.fingerprint,
            provider=deployment.provider,
            provider_model=deployment.provider_model,
            consecutive_misses=triggered.misses,
            proven_write_tokens=triggered.proven_write_tokens,
            ttl_seconds=prefix.ttl_seconds,
        ),
    )


class PromptCacheIncidentDetector:
    """Feeds each observation into its prefix's streak and opens an incident when a proven prefix misses three
    times in a row. Redis or database trouble skips the evidence instead of failing the request it came from."""

    def __init__(
        self,
        streaks: StreakStore,
        incidents: Callable[[], IncidentRecorder | None],
        failures: Sequence[type[BaseException]] = _STORE_FAILURES,
    ) -> None:
        self._streaks: Final = streaks
        self._incidents: Final = incidents
        self._failures: Final = tuple(failures)

    async def record(self, observation: PromptCacheObservation) -> None:
        update: Final = streak_update(observation)
        if update is None:
            return
        try:
            outcome: Final = await self._streaks.apply(update)
        except self._failures as error:
            verbose_router_logger.warning("prompt cache streak unavailable (%s)", type(error).__name__)
            return
        if not isinstance(outcome, Triggered):
            return
        recorder: Final = self._incidents()
        detection: Final = incident_detection(observation, outcome)
        if recorder is None or detection is None:
            return
        await recorder.open(detection)
