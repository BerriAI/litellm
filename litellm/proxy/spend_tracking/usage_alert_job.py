"""Once-daily evaluation, polled hourly by one database-elected worker."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from itertools import chain, groupby
from typing import (
    TYPE_CHECKING,
    Final,
    Protocol,
    cast,  # noqa: TID251  # generated Prisma delegate supplies the repository protocol
)
from uuid import uuid4

from litellm._logging import verbose_proxy_logger
from litellm.integrations.SlackAlerting.usage_alerts import UsageAlertDelivery, configured_destinations
from litellm.proxy.spend_tracking.daily_usage_alerts import compare_daily_usage
from litellm.repositories.usage_alert_health import UsageAlertHealthRepository
from litellm.repositories.usage_alert_metrics import DailyAlertMetrics, MetricsDatabase, UsageAlertMetricsRepository
from litellm.repositories.usage_alerts import UsageAlertsRepository
from litellm.types.integrations.slack_alerting import AlertType
from litellm.types.proxy.usage_alert_defaults import USAGE_ALERT_SCOPES, default_usage_rules, is_default_usage_rule
from litellm.types.proxy.usage_alerts import (
    UsageAlert,
    UsageAlertChannel,
    UsageAlertEvaluationStatus,
    UsageAlertNotification,
    UsageAlertObservation,
    UsageAlertRule,
    UsageAlertRuleInput,
    UsageAlertScope,
)

if TYPE_CHECKING:
    from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
    from litellm.proxy.utils import PrismaClient

USAGE_ALERT_JOB_ID: Final = "daily_usage_alerts"


def completed_day(now: datetime) -> date:
    """A UTC day gets an hour for normal buffered writes before it is eligible."""
    return (now.astimezone(timezone.utc) - timedelta(hours=1)).date() - timedelta(days=1)


def _has_activity_page(scope: UsageAlertScope, entity: str) -> bool:
    if not entity.strip():
        return False
    if scope == "key":
        return entity != "unattributed" and not entity.removeprefix("raw:").startswith("sk-")
    if scope == "user":
        return entity != "default_user_id"
    if scope == "team":
        return entity != "litellm-dashboard"
    return True


@dataclass(frozen=True, slots=True)
class _Observations:
    evaluated: bool
    changed: tuple[UsageAlertObservation, ...]


def _observations(
    rule: UsageAlertRule,
    rows: Sequence[DailyAlertMetrics],
    target: date,
    suppressed: frozenset[str | None] = frozenset(),
    *,
    active: frozenset[str | None] = frozenset(),
) -> _Observations:
    changed: Final[list[UsageAlertObservation]] = []  # mutable-ok: collect only persisted changes for this page
    evaluated = False  # rebind-ok: retain whether any entity had sufficient evidence, including healthy entities
    for entity, group in groupby(rows, lambda row: row.entity_id):
        if None in suppressed or entity in suppressed:
            continue
        if not _has_activity_page(rule.scope_type, entity):
            continue
        if rule.scope_id is not None and rule.scope_id != entity:
            continue
        comparison: Final = compare_daily_usage(rule, tuple(group), target)
        if comparison is None:
            continue
        evaluated = True
        identity: Final = None if rule.scope_type == "deployment" else entity
        if not comparison.breached and identity not in active:
            continue
        changed.append(
            UsageAlertObservation(
                entity_id=identity,
                baseline_sample_days=comparison.baseline_sample_days,
                day=target,
                observed=comparison.observed,
                baseline=comparison.baseline,
                threshold=comparison.threshold,
                breached=comparison.breached,
            )
        )
    return _Observations(evaluated, tuple(changed))


def _suppressed_entities(rule: UsageAlertRule, enabled: tuple[UsageAlertRule, ...]) -> frozenset[str | None]:
    if not is_default_usage_rule(rule):
        return frozenset()
    matching: Final = tuple(
        other
        for other in enabled
        if (other.scope_type, other.metric, other.direction) == (rule.scope_type, rule.metric, rule.direction)
    )
    return frozenset(other.scope_id for other in matching if not is_default_usage_rule(other))


class IncidentDelivery(Protocol):
    async def send(self, alert: UsageAlert, rule: UsageAlertRule, channel: UsageAlertChannel) -> bool: ...


class UsageAlertJob:
    def __init__(
        self,
        store: UsageAlertsRepository,
        metrics: UsageAlertMetricsRepository,
        health: UsageAlertHealthRepository,
        delivery: IncidentDelivery,
        defaults: tuple[UsageAlertRuleInput, ...] | None = None,
        *,
        deprecation_projection: Callable[[datetime], Awaitable[None]] | None = None,
    ) -> None:
        self._store = store
        self._metrics = metrics
        self._health = health
        self._delivery = delivery
        self._defaults = default_usage_rules() if defaults is None else defaults
        self._defaults_registered = False
        self._deprecation_projection = deprecation_projection

    @classmethod
    def from_prisma(
        cls,
        prisma: "PrismaClient",
        slack: "SlackAlerting",
        *,
        deprecation_projection: Callable[[datetime], Awaitable[None]] | None = None,
    ) -> "UsageAlertJob":
        database: Final = cast(MetricsDatabase, prisma.writer_db)  # cast-ok: generated Prisma delegate
        return cls(
            UsageAlertsRepository.from_prisma(prisma),
            UsageAlertMetricsRepository(database),
            UsageAlertHealthRepository(database),
            UsageAlertDelivery(slack),
            default_usage_rules(
                ("slack",)
                if "slack" in configured_destinations(slack) and AlertType.usage_alerts in slack.alert_types
                else (),
                legacy_user_spend_alerts=AlertType.user_spend_anomalies in slack.alert_types,
            ),
            deprecation_projection=deprecation_projection,
        )

    async def run(self, now: datetime | None = None) -> None:
        owner: Final = str(uuid4())
        if not await self._store.claim_job(owner):
            return
        try:
            await asyncio.wait_for(self._run(now or datetime.now(timezone.utc)), timeout=840)
        except asyncio.TimeoutError:
            verbose_proxy_logger.warning(
                "Usage alert evaluation exceeded its 14-minute lease budget; retrying next hour"
            )
        except Exception:  # noqa: BLE001  # the scheduler retries; completed rule/day checkpoints survive a failed pass
            verbose_proxy_logger.exception("Daily usage alert evaluation failed")
        finally:
            await self._store.release_job(owner)

    async def _rules(self) -> tuple[UsageAlertRule, ...]:
        pages: Final[list[UsageAlertRule]] = []  # mutable-ok: collect configuration pages
        offset = 0  # rebind-ok: configuration pagination cursor
        while page := await self._store.list_enabled_rules(limit=500, offset=offset):
            pages.extend(page)
            if len(page) < 500:
                return tuple(pages)
            offset += len(page)
        return tuple(pages)

    async def _run(self, now: datetime) -> None:
        if self._deprecation_projection is not None:
            await self._deprecation_projection(now)
        if not self._defaults_registered:
            await self._store.ensure_default_rules(self._defaults)
            self._defaults_registered = True
        target: Final = completed_day(now)
        rules: Final = await self._rules()
        complete: Final[set[str]] = set()  # mutable-ok: batched completion lookup, bounded by configuration size
        for offset in range(0, len(rules), 500):
            complete.update(await self._store.completed_rule_ids(rules[offset : offset + 500], target))
        due: Final = tuple(rule for rule in rules if rule.id not in complete)
        for scoped in (tuple(rule for rule in due if rule.scope_type == scope) for scope in USAGE_ALERT_SCOPES):
            if scoped:
                await self._scope(scoped, target, rules)
        await self._notify()
        await self._store.prune()

    async def _scope(
        self, rules: tuple[UsageAlertRule, ...], target: date, enabled: tuple[UsageAlertRule, ...]
    ) -> None:
        scope: Final = rules[0].scope_type
        start: Final = datetime.combine(target, time(), tzinfo=timezone.utc)
        coverage: Final = await self._health.coverage("team" if scope == "team" else "user", start)
        eligible: Final = tuple(
            rule for rule in rules if coverage.after_grace and (rule.direction == "increase" or coverage.hours == 24)
        )
        suppressed: Final = {rule.id: _suppressed_entities(rule, enabled) for rule in eligible}
        eligible_ids: Final = frozenset(rule.id for rule in eligible)
        await self._record_evaluations(
            tuple((rule, "insufficient_coverage") for rule in rules if rule.id not in eligible_ids),
            target,
            completed=False,
        )
        if not eligible:
            return
        by_entity: Final = {
            entity: tuple(group)
            for entity, group in groupby(
                sorted(eligible, key=lambda rule: rule.scope_id or ""), lambda rule: rule.scope_id
            )
        }
        evaluated: Final[set[str]] = set()  # mutable-ok: at most one entry per configured rule across streaming pages
        async for rows in self._metrics.batches(scope, target, max(rule.baseline_days for rule in eligible)):
            evaluated.update(await self._page(by_entity, rows, target, suppressed))
        await self._record_evaluations(
            tuple(
                (rule, "complete" if rule.id in evaluated or suppressed[rule.id] else "insufficient_history")
                for rule in eligible
            ),
            target,
            completed=True,
        )

    async def _record_evaluations(
        self,
        evaluations: tuple[tuple[UsageAlertRule, UsageAlertEvaluationStatus], ...],
        target: date,
        *,
        completed: bool,
    ) -> None:
        for offset in range(0, len(evaluations), 500):
            await self._store.record_evaluations(evaluations[offset : offset + 500], target, completed=completed)

    async def _page(
        self,
        rules: Mapping[str | None, tuple[UsageAlertRule, ...]],
        rows: tuple[DailyAlertMetrics, ...],
        target: date,
        suppressed: Mapping[str, frozenset[str | None]],
    ) -> tuple[str, ...]:
        by_entity: Final = {entity: tuple(group) for entity, group in groupby(rows, lambda row: row.entity_id)}
        applicable: Final = tuple(chain.from_iterable(rules.get(entity, ()) for entity in (None, *by_entity)))
        if not applicable:
            return ()
        entities: Final = (None,) if applicable[0].scope_type == "deployment" else tuple(by_entity)
        active: Final[dict[str, frozenset[str | None]]] = {}  # mutable-ok: bounded active results for this entity page
        for offset in range(0, len(applicable), 500):
            active.update(
                await self._store.breached_entities(
                    tuple(rule.id for rule in applicable[offset : offset + 500]), entities
                )
            )
        return tuple(
            [
                rule.id
                for rule in applicable
                if await self._rule_page(
                    rule,
                    rows if rule.scope_id is None else by_entity[rule.scope_id],
                    target,
                    active.get(rule.id, frozenset()),
                    suppressed[rule.id],
                )
            ]
        )

    async def _rule_page(
        self,
        rule: UsageAlertRule,
        rows: tuple[DailyAlertMetrics, ...],
        target: date,
        active: frozenset[str | None],
        suppressed: frozenset[str | None],
    ) -> bool:
        observations: Final = _observations(rule, rows, target, suppressed, active=active)
        if observations.changed:
            await self._store.record_observations(rule, observations.changed)
        return observations.evaluated

    async def _deliver_channel(self, alert: UsageAlert, rule: UsageAlertRule, channel: UsageAlertChannel) -> str | None:
        try:
            delivered: Final = await asyncio.wait_for(self._delivery.send(alert, rule, channel), timeout=30)
            if not delivered:
                return "transport_not_configured"
            await self._store.mark_notified(alert.id, channel)
        except Exception:  # noqa: BLE001  # retry only failed channels; persisted acknowledgements survive restarts
            verbose_proxy_logger.warning("Usage alert delivery failed for incident %s via %s", alert.id, channel)
            return "delivery_failed"
        return None

    async def _notify(self) -> None:
        while page := await self._store.pending_notifications(limit=100):
            for pending in page:
                await self._notify_incident(pending)

    async def _notify_incident(self, pending: UsageAlertNotification) -> None:
        current: Final = await self._store.get_rule(pending.rule.id)
        if current is None or not current.enabled or current.updated_at != pending.rule.updated_at:
            return
        failures: Final = tuple(
            [
                await self._deliver_channel(pending.incident, pending.rule, channel)
                for channel in pending.pending_channels
            ]
        )
        if any(failures):
            error: Final = "delivery_failed" if "delivery_failed" in failures else "transport_not_configured"
            await self._store.defer_notification(pending.incident.id, error)
