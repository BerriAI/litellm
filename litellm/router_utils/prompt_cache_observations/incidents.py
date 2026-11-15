import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Final, Protocol

from redis.exceptions import RedisError

from litellm._logging import verbose_router_logger
from litellm.caching.redis_cache import RedisCircuitBreakerOpenError
from litellm.models.alert import Alert, AlertDetection, IncidentAlertStore, PromptCacheEvidence

from .eligibility import write_proves_prefix_eligible
from .streaks import (
    PrefixEvent,
    ScopeBreak,
    StreakEvent,
    StreakOutcome,
    StreakScope,
    StreakStore,
    StreakUpdate,
    Triggered,
)
from .types import Assessed, CacheablePrefix, CacheCounters, PromptCacheObservation

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


def _prefix_event(
    observation: PromptCacheObservation, scope: StreakScope, prefix: CacheablePrefix, event: StreakEvent, tokens: int
) -> PrefixEvent:
    return PrefixEvent(
        scope=scope,
        prefix_fingerprint=prefix.fingerprint,
        event=event,
        call_id=observation.call_id,
        started_ms=_epoch_ms(observation.started_at),
        ended_ms=_epoch_ms(observation.ended_at),
        ttl_ms=prefix.ttl_seconds * 1000,
        write_tokens=tokens,
    )


def streak_update(observation: PromptCacheObservation) -> StreakUpdate:
    """A failed, unverified or unreadable attempt on a prefix is unknown evidence, which ends its streak. A
    caching attempt whose prefix could not be read at all ends every streak of its deployment and configuration."""
    scope: Final = StreakScope(
        deployment_id=observation.deployment.deployment_id,
        config_fingerprint=observation.deployment.config_fingerprint,
    )
    outcome: Final = observation.outcome
    prefix: Final = outcome.prefix
    if prefix is None:
        return ScopeBreak(scope=scope, ended_ms=_epoch_ms(observation.ended_at))
    usage: Final = outcome.usage if isinstance(outcome, Assessed) else None
    if isinstance(usage, CacheCounters):
        return _prefix_event(observation, scope, prefix, _event(usage, prefix), usage.write_tokens)
    return _prefix_event(observation, scope, prefix, StreakEvent.UNKNOWN, 0)


def _epoch_ms(epoch_seconds: float) -> int:
    return int(epoch_seconds * 1000)


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
        window_end=max(_moment(_epoch_ms(observation.ended_at)), start) + timedelta(milliseconds=1),
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
    times in a row. Redis or database trouble skips the evidence instead of failing the request it came from.

    Evidence this worker could not deliver may have been the hit that ended a streak, so before anything else
    reaches Redis again the lost evidence's scope is broken as of the lost request's end. Lost callbacks on
    other workers, or ones that never ran, stay invisible."""

    def __init__(
        self,
        streaks: StreakStore,
        incidents: Callable[[], IncidentRecorder | None],
        clock: Callable[[], float] = time.time,
        failures: Sequence[type[BaseException]] = _STORE_FAILURES,
    ) -> None:
        self._streaks: Final = streaks
        self._incidents: Final = incidents
        self._clock: Final = clock
        self._failures: Final = tuple(failures)
        self._lost: Mapping[StreakScope, int] = MappingProxyType({})

    def _remember(self, update: StreakUpdate) -> None:
        ended: Final = max(self._lost.get(update.scope, update.ended_ms), update.ended_ms)
        self._lost = MappingProxyType({**self._lost, update.scope: ended})

    async def _break_lost_scopes(self, now_ms: int) -> bool:
        for scope, ended in tuple(self._lost.items()):
            try:
                await self._streaks.apply(ScopeBreak(scope=scope, ended_ms=ended), now_ms)
            except self._failures as error:
                verbose_router_logger.warning("prompt cache streak unavailable (%s)", type(error).__name__)
                return False
            self._lost = MappingProxyType({key: value for key, value in self._lost.items() if key != scope})
        return True

    async def _apply(self, update: StreakUpdate) -> StreakOutcome | None:
        now_ms: Final = _epoch_ms(self._clock())
        if not await self._break_lost_scopes(now_ms):
            self._remember(update)
            return None
        try:
            return await self._streaks.apply(update, now_ms)
        except self._failures as error:
            verbose_router_logger.warning("prompt cache streak unavailable (%s)", type(error).__name__)
            self._remember(update)
            return None

    async def record(self, observation: PromptCacheObservation) -> None:
        outcome: Final = await self._apply(streak_update(observation))
        if not isinstance(outcome, Triggered):
            return
        recorder: Final = self._incidents()
        detection: Final = incident_detection(observation, outcome)
        if recorder is None or detection is None:
            return
        await recorder.open(detection)
