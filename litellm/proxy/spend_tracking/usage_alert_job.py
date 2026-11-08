"""Once-daily evaluation, polled hourly by one database-elected worker."""

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from itertools import chain, groupby, takewhile
from typing import (
    TYPE_CHECKING,
    Final,
    cast,  # noqa: TID251  # generated Prisma delegate supplies the repository protocol
)
from uuid import uuid4

from litellm._logging import verbose_proxy_logger
from litellm.integrations.SlackAlerting.usage_alerts import UsageAlertDelivery, configured_destinations
from litellm.proxy.spend_tracking.daily_usage_alerts import compare_daily_usage
from litellm.proxy.spend_tracking.performance_issues import PerformanceIssueProjector
from litellm.repositories.analytics_budget import (
    AnalyticsCapacityError,
    AnalyticsDatabase,
    BoundedAnalyticsDatabase,
    analytics_deadline,
)
from litellm.repositories.issues import IssuesRepository
from litellm.repositories.provider_latency import ProviderLatencyRepository
from litellm.repositories.usage_alert_health import UsageAlertHealthRepository
from litellm.repositories.usage_alert_metrics import DailyAlertMetrics, UsageAlertMetricsRepository
from litellm.repositories.usage_alerts import UsageAlertsRepository
from litellm.types.integrations.slack_alerting import AlertType
from litellm.types.proxy.usage_alert_defaults import USAGE_ALERT_SCOPES, default_usage_rules
from litellm.types.proxy.usage_alerts import (
    UsageAlertEvaluationStatus,
    UsageAlertObservation,
    UsageAlertProgress,
    UsageAlertRule,
    UsageAlertRuleInput,
    UsageAlertScope,
    UsageAlertWorkCursor,
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
        key: Final = re.sub(r"^(?:raw:)+", "", entity)
        return key not in ("", "unattributed") and not key.startswith("sk-")
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
        if (is_suppressed := None in suppressed or entity in suppressed) and (
            None if rule.scope_type == "deployment" else entity
        ) not in active:
            continue
        if not _has_activity_page(rule.scope_type, entity):
            continue
        if rule.scope_id is not None and rule.scope_id != entity:
            continue
        if (comparison := compare_daily_usage(rule, tuple(group), target)) is None:
            continue
        evaluated = True
        if is_suppressed and comparison.breached:
            continue
        if not comparison.breached and (None if rule.scope_type == "deployment" else entity) not in active:
            continue
        changed.append(
            UsageAlertObservation(
                entity_id=None if rule.scope_type == "deployment" else entity,
                baseline_sample_days=comparison.baseline_sample_days,
                day=target,
                observed=comparison.observed,
                baseline=comparison.baseline,
                threshold=comparison.threshold,
                breached=comparison.breached,
            )
        )
    return _Observations(evaluated, tuple(changed))


class UsageAlertJob:
    def __init__(
        self,
        store: UsageAlertsRepository,
        metrics: UsageAlertMetricsRepository,
        health: UsageAlertHealthRepository,
        defaults: tuple[UsageAlertRuleInput, ...] | None = None,
        *,
        deprecation_projection: Callable[[datetime], Awaitable[None]] | None = None,
        performance_projection: PerformanceIssueProjector,
        work_budget_seconds: float = 60,
    ) -> None:
        self._store = store
        self._metrics = metrics
        self._health = health
        self._defaults = default_usage_rules() if defaults is None else defaults
        self._defaults_registered = False
        self._deprecation_projection = deprecation_projection
        self._performance_projection = performance_projection
        self._work_budget_seconds = work_budget_seconds

    @classmethod
    def from_prisma(
        cls,
        prisma: "PrismaClient",
        slack: "SlackAlerting",
        *,
        deprecation_projection: Callable[[datetime], Awaitable[None]] | None = None,
        enterprise_access: Callable[[], bool] = lambda: False,
    ) -> "UsageAlertJob":
        database: Final = BoundedAnalyticsDatabase(
            cast(AnalyticsDatabase, prisma.writer_db),  # cast-ok: generated Prisma transaction delegate
            background=True,
        )
        metrics: Final = UsageAlertMetricsRepository(database)
        delivery: Final = UsageAlertDelivery(slack)
        return cls(
            UsageAlertsRepository(database),
            metrics,
            UsageAlertHealthRepository(database),
            default_usage_rules(
                ("slack",)
                if "slack" in configured_destinations(slack) and AlertType.usage_alerts in slack.alert_types
                else (),
                legacy_user_spend_alerts=AlertType.user_spend_anomalies in slack.alert_types,
            ),
            deprecation_projection=deprecation_projection,
            performance_projection=PerformanceIssueProjector(
                IssuesRepository(database),
                metrics,
                ProviderLatencyRepository(database),
                delivery,
                enterprise_access,
            ),
        )

    async def run(self, now: datetime | None = None) -> None:
        owner: Final = str(uuid4())
        finish_at: Final = asyncio.get_running_loop().time() + self._work_budget_seconds
        deadline: Final = finish_at - 2
        try:
            with analytics_deadline(deadline):
                claimed: Final = await self._store.claim_job(owner, lease_seconds=int(self._work_budget_seconds) + 15)
        except AnalyticsCapacityError:
            verbose_proxy_logger.info("Usage alert evaluation deferred while analytics capacity is busy")
            return
        if not claimed:
            return
        try:
            await asyncio.wait_for(
                self._run(now or datetime.now(timezone.utc), deadline),
                timeout=max(0, deadline - asyncio.get_running_loop().time()),
            )
        except AnalyticsCapacityError:
            verbose_proxy_logger.info("Usage alert work deferred within its database budget")
        except asyncio.TimeoutError:
            verbose_proxy_logger.warning("Usage alert evaluation reached its work budget; resuming next run")
        except Exception:  # noqa: BLE001  # the scheduler retries; completed rule/day checkpoints survive a failed pass
            verbose_proxy_logger.exception("Daily usage alert evaluation failed")
        finally:
            try:
                with analytics_deadline(finish_at - 1):
                    await self._store.release_job(owner)
            except AnalyticsCapacityError:
                verbose_proxy_logger.info("Usage alert lease will expire after bounded cleanup")

    async def _run(self, now: datetime, stop_at: float | None = None) -> None:
        deadline: Final = (
            stop_at if stop_at is not None else asyncio.get_running_loop().time() + self._work_budget_seconds - 2
        )
        with analytics_deadline(deadline):
            await self._work(now, deadline)

    async def _work(self, now: datetime, deadline: float) -> None:
        cleanup_until: Final = min(deadline, asyncio.get_running_loop().time() + self._work_budget_seconds / 10)
        while asyncio.get_running_loop().time() < cleanup_until:
            if not any((await self._performance_projection.prune(), await self._store.prune(limit=10))):
                break
        if self._deprecation_projection is not None:
            await self._deprecation_projection(now)
        monitoring: Final = await self._performance_projection.monitoring()
        if not monitoring.enabled:
            return
        if monitoring.can_detect:
            if not self._defaults_registered:
                await self._store.ensure_default_rules(self._defaults)
                self._defaults_registered = True
            target: Final = completed_day(now)
            await self._store.ensure_projection_work(target)
            await self._evaluate(
                target, asyncio.get_running_loop().time() + max(0, deadline - asyncio.get_running_loop().time()) * 0.65
            )
        if (work := await self._store.pending_projection_work()) is None:
            await self._performance_projection.notify(deadline)
        elif await self._performance_projection.run_pending(now, work, self._store, deadline):
            await self._store.finish_projection_work(work)

    @staticmethod
    def _position(rule: UsageAlertRule, target: date) -> UsageAlertProgress:
        if rule.evaluation_progress is not None and rule.evaluation_progress.revision == rule.updated_at:
            return rule.evaluation_progress
        return UsageAlertProgress(day=target, revision=rule.updated_at)

    async def _evaluate(self, target: date, deadline: float) -> None:
        cursor = await self._store.work_cursor()  # rebind-ok: durable round-robin position
        exhausted: Final[set[int]] = set()  # mutable-ok: at most the 10 scope/config phases in this invocation
        phase_progress = False  # rebind-ok: whether a page advanced in the current phase
        while asyncio.get_running_loop().time() < deadline and len(exhausted) < 10:
            if cursor.phase in exhausted:
                cursor = UsageAlertWorkCursor(phase=(cursor.phase + 1) % 10)
                continue
            phase: int = cursor.phase
            cursor, advanced, finished = await self._evaluation_page(cursor, target)
            phase_progress = phase_progress or advanced
            if finished:
                if not phase_progress:
                    exhausted.add(phase)
                phase_progress = False
            await self._store.save_work_cursor(cursor)
            await asyncio.sleep(0.01)

    async def _evaluation_page(
        self, cursor: UsageAlertWorkCursor, target: date
    ) -> tuple[UsageAlertWorkCursor, bool, bool]:
        scope: Final = USAGE_ALERT_SCOPES[cursor.phase // 2]
        wildcard: Final = cursor.phase % 2 == 0
        limit: Final = 8 if wildcard else 10
        rules: Final = await self._store.enabled_rule_page(scope, wildcard, cursor.after_id, limit=limit)
        if not rules:
            return UsageAlertWorkCursor(phase=(cursor.phase + 1) % 10), False, True
        # Keep one date/cursor per page. A newly edited rule cannot rewind another rule.
        position: Final = self._position(rules[0], target)
        selected: Final = tuple(
            takewhile(
                lambda rule: (
                    (
                        self._position(rule, target).day,
                        self._position(rule, target).cursor if wildcard else None,
                    )
                    == (position.day, position.cursor if wildcard else None)
                ),
                rules,
            )
        )
        complete: Final = await self._store.completed_rule_ids(selected, position.day)
        due: Final = tuple(rule for rule in selected if rule.id not in complete)
        advanced: Final = bool(due) and await self._attempt_scope_page(due, position.day, wildcard)
        if len(rules) < limit and len(selected) == len(rules):
            return UsageAlertWorkCursor(phase=(cursor.phase + 1) % 10), advanced, True
        return UsageAlertWorkCursor(phase=cursor.phase, after_id=selected[-1].id), advanced, False

    async def _attempt_scope_page(self, rules: tuple[UsageAlertRule, ...], target: date, wildcard: bool) -> bool:
        try:
            return await self._scope_page(rules, target, wildcard)
        except AnalyticsCapacityError:
            await self._store.save_progress(tuple((rule, self._position(rule, target)) for rule in rules))
            verbose_proxy_logger.info("Usage alert page deferred within its database budget")
            return False

    async def _scope_page(self, rules: tuple[UsageAlertRule, ...], target: date, wildcard: bool) -> bool:
        scope: Final = rules[0].scope_type
        start: Final = datetime.combine(target, time(), tzinfo=timezone.utc)
        coverage: Final = await self._health.coverage("team" if scope == "team" else "user", start)
        eligible: Final = tuple(
            rule for rule in rules if coverage.after_grace and (rule.direction == "increase" or coverage.hours == 24)
        )
        eligible_ids: Final = frozenset(rule.id for rule in eligible)
        await self._record_evaluations(
            tuple((rule, "insufficient_coverage") for rule in rules if rule.id not in eligible_ids),
            target,
            completed=False,
        )
        if not eligible:
            return False
        await self._store.save_progress(
            tuple((rule, self._position(rule, target)) for rule in eligible if rule.evaluation_progress is None)
        )
        page: Final = await self._metrics.page(
            scope,
            target,
            max(rule.baseline_days for rule in eligible),
            cursor=self._position(eligible[0], target).cursor if wildcard else None,
            entity_ids=None
            if wildcard
            else tuple(dict.fromkeys(rule.scope_id for rule in eligible if rule.scope_id is not None)),
        )
        by_entity: Final = {
            entity: tuple(group)
            for entity, group in groupby(
                sorted(eligible, key=lambda rule: rule.scope_id or ""), lambda rule: rule.scope_id
            )
        }
        entities: Final = (None,) if scope == "deployment" else tuple(dict.fromkeys(row.entity_id for row in page.rows))
        suppressed: Final = await self._store.suppressed_entities(eligible, entities)
        evaluated: Final = frozenset(await self._page(by_entity, page.rows, target, suppressed))
        if page.complete:
            await self._record_evaluations(
                tuple(
                    (
                        rule,
                        "complete"
                        if rule.id in evaluated or self._position(rule, target).evaluated or suppressed.get(rule.id)
                        else "insufficient_history",
                    )
                    for rule in eligible
                ),
                target,
                completed=True,
            )
        else:
            await self._store.save_progress(
                tuple(
                    (
                        rule,
                        UsageAlertProgress(
                            day=target,
                            revision=rule.updated_at,
                            cursor=page.cursor,
                            evaluated=rule.id in evaluated
                            or self._position(rule, target).evaluated
                            or bool(suppressed.get(rule.id)),
                        ),
                    )
                    for rule in eligible
                )
            )
        return True

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
        evaluated: Final[list[str]] = []  # mutable-ok: at most 500 rule IDs in the current config page
        chunk_size: Final = 8 if None in rules else 500
        for offset in range(0, len(applicable), chunk_size):
            evaluated.extend(
                await self._rule_chunk(
                    applicable[offset : offset + chunk_size], entities, rows, by_entity, target, suppressed
                )
            )
        return tuple(evaluated)

    async def _rule_chunk(
        self,
        selected: tuple[UsageAlertRule, ...],
        entities: tuple[str | None, ...],
        rows: tuple[DailyAlertMetrics, ...],
        by_entity: Mapping[str, tuple[DailyAlertMetrics, ...]],
        target: date,
        suppressed: Mapping[str, frozenset[str | None]],
    ) -> tuple[str, ...]:
        active: Final = await self._store.breached_entities(tuple(rule.id for rule in selected), entities)
        return tuple(
            [
                rule.id
                for rule in selected
                if await self._rule_page(
                    rule,
                    rows if rule.scope_id is None else by_entity[rule.scope_id],
                    target,
                    active.get(rule.id, frozenset()),
                    suppressed.get(rule.id, frozenset()),
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
