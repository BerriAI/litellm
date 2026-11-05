"""Project daily observations into scoped, persisted investigations."""

import asyncio
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from itertools import chain, groupby, islice
from typing import Final, Protocol
from uuid import NAMESPACE_URL, uuid5

from litellm._logging import verbose_proxy_logger
from litellm.models.issues import IssueStore
from litellm.proxy.spend_tracking.daily_usage_alerts import DailyComparison, compare_daily_usage
from litellm.repositories.provider_latency import ProviderLatencyEvidence, ProviderLatencyRepository
from litellm.repositories.usage_alert_metrics import (
    DailyAlertMetrics,
    IssueKeyAttribution,
    IssueMetricDay,
    IssueMetricWindow,
    IssueSourceProvider,
    UsageAlertMetricsRepository,
)
from litellm.repositories.usage_alerts import UsageAlertsRepository
from litellm.types.proxy.issues import (
    Issue,
    IssueCohort,
    IssueComparison,
    IssueDeliveryBinding,
    IssueDeliveryInput,
    IssueImpact,
    IssueInput,
    IssuePendingDelivery,
    IssueProjectionSignal,
    IssueSignal,
    IssueSignalGroupingReason,
)
from litellm.types.proxy.usage_alert_defaults import (
    PerformanceIssueFamily,
    default_usage_rule_id,
    is_default_usage_rule,
)
from litellm.types.proxy.usage_alerts import (
    UsageAlertDirection,
    UsageAlertMetric,
    UsageAlertProjectionCursor,
    UsageAlertProjectionWork,
    UsageAlertRule,
    UsageAlertScope,
)

_SCOPES: Final[tuple[UsageAlertScope, ...]] = ("key", "user", "team", "model", "deployment")
_METRICS: Final[tuple[UsageAlertMetric, ...]] = ("spend", "latency", "error_rate", "errors", "requests")


class IssueDelivery(Protocol):
    async def send_issue(self, pending: IssuePendingDelivery) -> bool: ...


@dataclass(frozen=True, slots=True)
class _Candidate:
    source: IssueProjectionSignal
    comparison: DailyComparison
    rows: tuple[DailyAlertMetrics, ...]
    impact: IssueImpact
    window: IssueMetricWindow


@dataclass(frozen=True, slots=True)
class _Target:
    issue: IssueInput
    reason: IssueSignalGroupingReason = "same_resource"
    explanation: str = ""


@dataclass(frozen=True, slots=True)
class _ProviderGroup:
    evidence: ProviderLatencyEvidence
    rule: UsageAlertRule


def _family(metric: UsageAlertMetric) -> PerformanceIssueFamily:
    if metric in ("errors", "error_rate"):
        return "reliability"
    if metric == "requests":
        return "traffic"
    return "spend" if metric == "spend" else "latency"


def _cohort_key(family: str, scope: str, entity: str | None, direction: UsageAlertDirection) -> str:
    return str(uuid5(NAMESPACE_URL, f"litellm:issue:{family}:{scope}:{entity}:{direction}"))


def _window(source: IssueProjectionSignal, day: date) -> IssueMetricWindow:
    return IssueMetricWindow(
        source.alert.id, source.alert.scope_id or "deployment", day - timedelta(days=source.rule.baseline_days), day
    )


def _comparison(candidate: _Candidate) -> IssueComparison:
    return IssueComparison(
        metric=candidate.source.rule.metric,
        direction=candidate.source.rule.direction,
        observed_value=candidate.comparison.observed,
        baseline_value=candidate.comparison.baseline,
        threshold=candidate.comparison.threshold,
        baseline_start=candidate.window.start,
        baseline_end=candidate.window.day - timedelta(days=1),
        baseline_sample_days=candidate.comparison.baseline_sample_days,
    )


def _request_context(candidate: _Candidate) -> IssueComparison:
    history: Final = tuple(row for row in candidate.rows if row.day < candidate.window.day)
    baseline: Final = sum(row.requests for row in history) / len({row.day for row in history})
    observed: Final = sum(row.requests for row in candidate.rows if row.day == candidate.window.day)
    return IssueComparison(
        metric="requests",
        direction="increase" if observed >= baseline else "decrease",
        observed_value=observed,
        baseline_value=baseline,
        baseline_start=candidate.window.start,
        baseline_end=candidate.window.day - timedelta(days=1),
        baseline_sample_days=len({row.day for row in history}),
    )


def _resource_issue(candidate: _Candidate, now: datetime) -> IssueInput:
    rule: Final = candidate.source.rule
    family: Final = _family(rule.metric)
    cohort_key: Final = _cohort_key(family, rule.scope_type, candidate.source.alert.scope_id, rule.direction)
    title: Final = {
        "spend": "Spend",
        "latency": "Latency",
        "error_rate": "Error rate",
        "errors": "Failed requests",
        "requests": "Traffic",
    }[rule.metric]
    return IssueInput(
        id=str(
            uuid5(NAMESPACE_URL, f"{cohort_key}:{candidate.source.alert.id}:{candidate.window.day}:{rule.updated_at}")
        ),
        cohort_key=cohort_key,
        kind="performance",
        family=family,
        title=f"{title} {'increased' if rule.direction == 'increase' else 'decreased'}",
        cohort=IssueCohort(scope_type=rule.scope_type, scope_id=candidate.source.alert.scope_id),
        observed_on=candidate.window.day,
        observed_at=now,
        range_start=candidate.window.start,
        range_end=candidate.window.day,
        grouping_reason="same_resource",
        explanation=f"Compared with {candidate.comparison.baseline_sample_days} recorded reference days.",
        comparisons=(_comparison(candidate), _request_context(candidate))
        if rule.metric == "spend"
        else (_comparison(candidate),),
        impact=candidate.impact,
    )


def _comparison_window(candidate: _Candidate) -> tuple[str | None, date, date]:
    return candidate.source.alert.scope_id, candidate.window.start, candidate.window.day


def _volume_explains_spend(traffic: _Candidate, spend: _Candidate) -> bool:
    if spend.comparison.baseline <= 0 or traffic.comparison.baseline <= 0:
        return False
    if (spend.window.start, spend.window.day) != (traffic.window.start, traffic.window.day):
        return False
    cost_ratio: Final = spend.comparison.observed / spend.comparison.baseline
    request_ratio: Final = traffic.comparison.observed / traffic.comparison.baseline
    return 0.8 <= cost_ratio / request_ratio <= 1.2


def _bindings(candidate: _Candidate, issue_id: str) -> tuple[IssueDeliveryBinding, ...]:
    rule: Final = candidate.source.rule
    destinations: Final = (
        tuple(("email", address) for address in rule.email_recipients) if "email" in rule.notification_channels else ()
    )
    channels: Final = (
        *destinations,
        *((("slack", "configured-slack-route"),) if "slack" in rule.notification_channels else ()),
    )
    return tuple(
        IssueDeliveryBinding(
            delivery=IssueDeliveryInput(
                id=str(uuid5(NAMESPACE_URL, f"{issue_id}:{channel}:{destination}")),
                issue_id=issue_id,
                channel="email" if channel == "email" else "slack",
                email_recipients=(destination,) if channel == "email" else (),
            ),
            rule_id=rule.id,
            rule_revision=rule.updated_at.isoformat(),
            source_id=candidate.source.alert.id,
        )
        for channel, destination in channels
    )


class PerformanceIssueProjector:
    def __init__(
        self,
        store: IssueStore,
        metrics: UsageAlertMetricsRepository,
        providers: ProviderLatencyRepository,
        delivery: IssueDelivery,
    ) -> None:
        self._store = store
        self._metrics = metrics
        self._providers = providers
        self._delivery = delivery
        self._provider_rule: UsageAlertRule | None = None
        self._provider_cache: OrderedDict[str, _ProviderGroup | None] = OrderedDict()  # mutable-ok: max 32 providers

    async def run_pending(
        self, now: datetime, work: UsageAlertProjectionWork, rules: UsageAlertsRepository, deadline: float
    ) -> bool:
        self._provider_cache.clear()
        self._provider_rule = None
        for scope in _SCOPES:
            if (
                rule := await rules.get_rule(default_usage_rule_id(scope, "latency", "increase"))
            ) is not None and rule.enabled:
                self._provider_rule = rule
                break
        position = await rules.projection_cursor(work)  # rebind-ok: persisted bounded source cursor
        observed_at: Final = position.observed_at or now.replace(microsecond=now.microsecond // 1000 * 1000)
        projection_deadline: Final = deadline - min(2, max(0, deadline - asyncio.get_running_loop().time()) / 4)
        while asyncio.get_running_loop().time() < projection_deadline:
            if (next_position := await self._projection_page(position, work, observed_at)) is None:
                await self._store.complete_projection(work.day, work.version)
                await self._notify(deadline)
                await self._store.prune()
                return True
            position = next_position
            await rules.save_projection_cursor(position)
            await asyncio.sleep(0.01)
        await self._notify(deadline)
        return False

    async def _projection_page(
        self, position: UsageAlertProjectionCursor, work: UsageAlertProjectionWork, observed_at: datetime
    ) -> UsageAlertProjectionCursor | None:
        if position.step == 0:
            prior: Final = await self._store.active_provider_issues(limit=1, cursor=position.after_id)
            if prior:
                if self._provider_rule is not None:
                    await self._prior_provider_page(prior, work.day, self._provider_rule, observed_at)
                return UsageAlertProjectionCursor(version=work.version, after_id=prior[-1].id, observed_at=observed_at)
            return UsageAlertProjectionCursor(version=work.version, step=1, observed_at=observed_at)
        if position.step > len(_SCOPES) * len(_METRICS):
            identities: Final = await self._store.active_resource_issue_ids(position.after_id)
            if identities:
                await self._store.recover_issues(work.day, issue_ids=identities)
                return UsageAlertProjectionCursor(
                    version=work.version, step=position.step, after_id=identities[-1], observed_at=observed_at
                )
            return None
        metric: Final = _METRICS[(position.step - 1) // len(_SCOPES)]
        scope: Final = _SCOPES[(position.step - 1) % len(_SCOPES)]
        sources: Final = await self._store.active_signals(
            work.day, scope_type=scope, metric=metric, cursor=position.after_id, limit=10, completed_only=True
        )
        if sources:
            await self._project_page(sources, scope, work.day, observed_at)
            return UsageAlertProjectionCursor(
                version=work.version, step=position.step, after_id=sources[-1].alert.id, observed_at=observed_at
            )
        return UsageAlertProjectionCursor(version=work.version, step=position.step + 1, observed_at=observed_at)

    async def _candidates(
        self, sources: tuple[IssueProjectionSignal, ...], scope: UsageAlertScope, day: date
    ) -> tuple[_Candidate, ...]:
        eligible: Final = sources
        windows: Final = tuple(_window(source, day) for source in eligible)
        rows: Final = await self._metrics.issue_windows(scope, windows)
        by_reference: Final = {
            reference: tuple(group)
            for reference, group in groupby(
                sorted(rows, key=lambda row: row.reference_id), lambda row: row.reference_id
            )
        }
        return tuple(
            candidate
            for source in eligible
            if (candidate := self._candidate(source, by_reference.get(source.alert.id, ()), day)) is not None
        )

    @staticmethod
    def _candidate(source: IssueProjectionSignal, days: tuple[IssueMetricDay, ...], day: date) -> _Candidate | None:
        rows: Final = tuple(item.metrics for item in days)
        comparison: Final = compare_daily_usage(source.rule, rows, day)
        if comparison is None or not comparison.breached:
            return None
        return _Candidate(
            source,
            comparison,
            rows,
            IssueImpact(
                keys=days[0].keys if days else None,
                users=days[0].users if days else None,
                teams=1 if source.rule.scope_type == "team" else None,
                models=1 if source.rule.scope_type == "model" else None,
                requests=sum(row.requests for row in rows if row.day == day),
            ),
            _window(source, day),
        )

    async def _project_page(
        self,
        sources: tuple[IssueProjectionSignal, ...],
        scope: UsageAlertScope,
        day: date,
        now: datetime,
    ) -> None:
        candidates: Final = await self._candidates(sources, scope, day)
        provider_targets: Final = await self._provider_targets(candidates, now)
        resource_targets: Final = await self._resource_targets(
            tuple(item for item in candidates if item.source.alert.id not in provider_targets), now
        )
        targets: Final = {
            item.source.alert.id: provider_targets.get(item.source.alert.id)
            or resource_targets.get(item.source.alert.id)
            or _Target(_resource_issue(item, now))
            for item in candidates
        }
        await self._save(candidates, targets, now)

    async def _spend_pages(
        self, scope: UsageAlertScope, entities: tuple[str, ...] | None, day: date
    ) -> AsyncIterator[tuple[_Candidate, ...]]:
        cursor: str | None = None  # rebind-ok: stream configured rules without retaining earlier candidate pages
        while sources := await self._store.active_signals(
            day,
            scope_type=scope,
            metric="spend",
            scope_ids=entities,
            cursor=cursor,
            limit=10,
            completed_only=True,
        ):
            yield await self._candidates(sources, scope, day)
            cursor = sources[-1].alert.id

    async def _volume_targets(self, candidates: tuple[_Candidate, ...], now: datetime) -> Mapping[str, _Target]:
        traffic: Final = tuple(
            item
            for item in candidates
            if item.source.rule.metric == "requests" and item.source.rule.direction == "increase"
        )
        if not traffic:
            return {}
        scope: Final = traffic[0].source.rule.scope_type
        keys: Final = tuple(_cohort_key("spend", scope, item.source.alert.scope_id, "increase") for item in traffic)
        issues: Final = {
            issue.cohort_key: issue for issue in await self._store.find_active_issues(keys) if issue.last_seen == now
        }
        targets: Final = {
            _comparison_window(item): item
            for item in traffic
            if _cohort_key("spend", scope, item.source.alert.scope_id, "increase") in issues
        }
        if not targets:
            return {}
        entities: Final = tuple(dict.fromkeys(identity for identity, _, _ in targets if identity is not None))
        matched: Final[set[tuple[str | None, date, date]]] = set()  # mutable-ok: max 10 flags
        async for page in self._spend_pages(scope, entities if scope != "deployment" else None, traffic[0].window.day):
            for spend in page:
                if (identity := _comparison_window(spend)) in targets and _volume_explains_spend(
                    targets[identity], spend
                ):
                    matched.add(identity)
            if len(matched) == len(targets):
                break
        return {
            item.source.alert.id: _Target(
                issues[_cohort_key("spend", scope, item.source.alert.scope_id, "increase")].model_copy(
                    update={"observed_at": now}
                ),
                "volume_explains_spend",
                "Request volume and spend increased by similar proportions in the same comparison window.",
            )
            for item in traffic
            if _comparison_window(item) in matched
        }

    async def _resource_targets(self, candidates: tuple[_Candidate, ...], now: datetime) -> Mapping[str, _Target]:
        volume: Final = await self._volume_targets(candidates, now)
        broader: Final = tuple(
            item
            for item in candidates
            if is_default_usage_rule(item.source.rule) and item.source.rule.scope_type != "key"
        )
        if not broader:
            return volume
        attributions: Final = {
            row.reference_id: row
            for row in await self._metrics.issue_key_attributions(
                broader[0].source.rule.scope_type,
                tuple(item.window for item in broader),
                "requests" if broader[0].source.rule.metric == "requests" else "spend",
            )
        }
        eligible: Final = tuple(
            item
            for item in broader
            if (attribution := attributions.get(item.source.alert.id)) is not None
            and (
                attribution.equivalent
                or (
                    item.source.rule.metric in ("spend", "requests")
                    and item.source.rule.direction == "increase"
                    and attribution.positive_change_share >= 0.8
                )
            )
        )
        keys: Final = tuple(
            _cohort_key(
                _family(item.source.rule.metric),
                "key",
                attributions[item.source.alert.id].key_id,
                item.source.rule.direction,
            )
            for item in eligible
        )
        issues: Final = {
            issue.cohort_key: issue for issue in await self._store.find_active_issues(keys) if issue.last_seen == now
        }
        borrowed: Final = {
            item.source.alert.id: _Target(
                issue.model_copy(update={"observed_at": now}),
                "same_key_traffic" if attributions[item.source.alert.id].equivalent else "dominant_key",
                "This source contains the same key traffic."
                if attributions[item.source.alert.id].equivalent
                else f"This key accounts for {attributions[item.source.alert.id].positive_change_share:.0%} of positive {item.source.rule.metric} growth in this source.",
            )
            for item in eligible
            if (
                issue := issues.get(
                    _cohort_key(
                        _family(item.source.rule.metric),
                        "key",
                        attributions[item.source.alert.id].key_id,
                        item.source.rule.direction,
                    )
                )
            )
            is not None
        }
        key_volume: Final = await self._key_volume_targets(eligible, attributions, now)
        return {**volume, **borrowed, **key_volume}

    async def _key_volume_targets(
        self,
        candidates: tuple[_Candidate, ...],
        attributions: Mapping[str, IssueKeyAttribution],
        now: datetime,
    ) -> Mapping[str, _Target]:
        traffic: Final = tuple(
            item
            for item in candidates
            if item.source.rule.metric == "requests" and item.source.rule.direction == "increase"
        )
        if not traffic:
            return {}
        key_ids: Final = tuple(dict.fromkeys(attributions[item.source.alert.id].key_id for item in traffic))
        keys: Final = tuple(_cohort_key("spend", "key", key, "increase") for key in key_ids)
        issues: Final = {
            issue.cohort_key: issue for issue in await self._store.find_active_issues(keys) if issue.last_seen == now
        }
        needed: Final = frozenset(
            (attributions[item.source.alert.id].key_id, item.window.start, item.window.day)
            for item in traffic
            if _cohort_key("spend", "key", attributions[item.source.alert.id].key_id, "increase") in issues
        )
        if not needed:
            return {}
        matched: Final[set[tuple[str | None, date, date]]] = set()  # mutable-ok: max 10 flags
        async for page in self._spend_pages("key", key_ids, traffic[0].window.day):
            for spend in page:
                if (identity := _comparison_window(spend)) in needed and self._volume_matches_key_spend(spend):
                    matched.add(identity)
            if len(matched) == len(needed):
                break
        return {
            item.source.alert.id: _Target(
                issue.model_copy(update={"observed_at": now}),
                "volume_explains_spend",
                "The same key explains this source's request growth and has a matching increase in spend.",
            )
            for item in traffic
            if (issue := issues.get(_cohort_key("spend", "key", attributions[item.source.alert.id].key_id, "increase")))
            is not None
            and (attributions[item.source.alert.id].key_id, item.window.start, item.window.day) in matched
        }

    @staticmethod
    def _volume_matches_key_spend(spend: _Candidate) -> bool:
        requests: Final = _request_context(spend)
        if (
            requests.baseline_value <= 0
            or requests.observed_value <= requests.baseline_value
            or spend.comparison.baseline <= 0
        ):
            return False
        return (
            0.8
            <= (spend.comparison.observed / spend.comparison.baseline)
            / (requests.observed_value / requests.baseline_value)
            <= 1.2
        )

    @staticmethod
    def _same_policy(left: UsageAlertRule, right: UsageAlertRule) -> bool:
        return (left.baseline_days, left.min_history_days, left.min_requests, left.multiplier, left.min_value) == (
            right.baseline_days,
            right.min_history_days,
            right.min_requests,
            right.multiplier,
            right.min_value,
        )

    async def _provider_group(self, provider: str, day: date, now: datetime) -> _ProviderGroup | None:
        if provider in self._provider_cache:
            self._provider_cache.move_to_end(provider)
            return self._provider_cache[provider]
        rule: Final = self._provider_rule
        if rule is None:
            return None
        existing: Final = await self._store.find_active_issues(
            (_cohort_key("latency", "provider", provider, "increase"),)
        )
        evidence: ProviderLatencyEvidence | None = None  # rebind-ok: fresh or saved cohort evidence
        if existing:
            prior: Final = existing[0]
            _, recovered, evidence = await self._prior_provider(prior, day, rule)  # rebind-ok: saved cohort
            if recovered:
                await self._store.recover_issue_ids((prior.id,), now)
        else:
            pairs: Final = await self._providers.provider_pairs(provider, day)
            if 2 <= len(pairs) <= 100:
                evidence = await self._providers.latency_cohort(pairs, day, rule)  # rebind-ok: new cohort
        result: Final = _ProviderGroup(evidence, rule) if evidence is not None else None
        self._provider_cache[provider] = result
        if len(self._provider_cache) > 32:
            self._provider_cache.popitem(last=False)
        return result

    async def _prior_provider_page(
        self,
        existing: tuple[Issue, ...],
        day: date,
        rule: UsageAlertRule,
        now: datetime,
    ) -> tuple[tuple[Issue, bool, ProviderLatencyEvidence | None], ...]:
        states: Final = tuple([await self._prior_provider(issue, day, rule) for issue in existing])
        await self._store.recover_issue_ids(tuple(issue.id for issue, recovered, _ in states if recovered), now)
        return states

    async def _prior_provider(
        self,
        issue: Issue,
        day: date,
        rule: UsageAlertRule,
    ) -> tuple[Issue, bool, ProviderLatencyEvidence | None]:
        if await self._providers.cohort_recovered(issue.cohort.model_pairs, day, rule):
            return issue, True, None
        evidence: Final = await self._providers.latency_cohort(issue.cohort.model_pairs, day, rule)
        return (
            issue,
            False,
            evidence if evidence is not None and evidence.cohort.model_pairs == issue.cohort.model_pairs else None,
        )

    async def _provider_targets(self, candidates: tuple[_Candidate, ...], now: datetime) -> Mapping[str, _Target]:
        eligible: Final = tuple(
            item
            for item in candidates
            if item.source.rule.metric == "latency" and is_default_usage_rule(item.source.rule)
        )
        if not eligible or self._provider_rule is None:
            return {}
        scope: Final = eligible[0].source.rule.scope_type
        by_reference: Final = {item.source.alert.id: item for item in eligible}
        hints: Final = await self._metrics.issue_source_providers(scope, tuple(item.window for item in eligible))
        targets: Final[dict[str, _Target]] = {}  # mutable-ok: each source belongs to at most one 80%-coverage provider
        for provider, matches in groupby(sorted(hints, key=lambda hint: hint.provider), lambda hint: hint.provider):
            if (group := await self._provider_group(provider, eligible[0].window.day, now)) is not None:
                targets.update(
                    await self._provider_matches(provider, tuple(matches), by_reference, {provider: group}, scope, now)
                )
        return targets

    async def _provider_matches(
        self,
        provider: str,
        matches: tuple[IssueSourceProvider, ...],
        by_reference: Mapping[str, _Candidate],
        providers: Mapping[str, _ProviderGroup],
        scope: UsageAlertScope,
        now: datetime,
    ) -> Mapping[str, _Target]:
        group: Final = providers.get(provider)
        if group is None:
            return {}
        selected: Final = tuple(
            by_reference[hint.reference_id]
            for hint in matches
            if self._same_policy(by_reference[hint.reference_id].source.rule, group.rule)
        )
        if not selected:
            return {}
        members: Final = await self._providers.source_members(
            group.evidence.cohort.model_pairs,
            selected[0].window.day,
            group.rule,
            scope,
            tuple(item.source.alert.scope_id or "deployment" for item in selected),
        )
        return {
            item.source.alert.id: target
            for item in selected
            if (item.source.alert.scope_id or "deployment") in members
            and (target := self._provider_target(item, group.evidence, now)) is not None
        }

    @staticmethod
    def _provider_target(item: _Candidate, evidence: ProviderLatencyEvidence, now: datetime) -> _Target | None:
        cohort: Final = evidence.comparison
        if (
            item.comparison.baseline <= 0
            or item.comparison.observed / item.comparison.baseline > 2 * cohort.observed_value / cohort.baseline_value
        ):
            return None
        key: Final = _cohort_key("latency", "provider", evidence.cohort.scope_id, "increase")
        return _Target(
            IssueInput(
                id=str(
                    uuid5(
                        NAMESPACE_URL, f"{key}:{item.source.alert.id}:{item.window.day}:{item.source.rule.updated_at}"
                    )
                ),
                cohort_key=key,
                kind="performance",
                family="latency",
                title=f"Higher latency across models through {evidence.cohort.scope_id}",
                cohort=evidence.cohort,
                observed_on=item.window.day,
                observed_at=now,
                range_start=cohort.baseline_start,
                range_end=item.window.day,
                grouping_reason="provider_cohort",
                explanation="Several keys and models slowed down together. The comparison holds the observed day's key and model mix constant and excludes keys still collecting a baseline. This is a shared pattern, not proof of a common cause.",
                comparisons=(cohort,),
                impact=evidence.impact,
            ),
            "provider_cohort",
            "At least 80% of this source's timed requests belong to independently degraded key/model pairs in the saved cohort.",
        )

    async def _save(self, candidates: tuple[_Candidate, ...], targets: Mapping[str, _Target], now: datetime) -> None:
        if not candidates:
            return
        distinct: Final = {target.issue.cohort_key: target.issue for target in targets.values()}
        existing: Final = {item.cohort_key: item for item in await self._store.find_active_issues(tuple(distinct))}
        inputs: Final = tuple(
            item.model_copy(update={"id": existing[key].id}) if key in existing else item
            for key, item in distinct.items()
        )
        saved: Final = {item.cohort_key: item for item in await self._store.upsert_issues(inputs)}
        await self._store.add_signal_batch(
            tuple(
                IssueSignal(
                    issue_id=saved[targets[candidate.source.alert.id].issue.cohort_key].id,
                    source="usage_alert",
                    source_id=candidate.source.alert.id,
                    usage_alert=candidate.source.alert,
                    grouping_reason=targets[candidate.source.alert.id].reason,
                    explanation=targets[candidate.source.alert.id].explanation,
                )
                for candidate in candidates
            )
        )
        bindings: Final = chain.from_iterable(
            _bindings(candidate, saved[targets[candidate.source.alert.id].issue.cohort_key].id)
            for candidate in candidates
        )
        while batch := tuple(islice(bindings, 10)):
            await self._store.queue_deliveries(batch)

    async def _notify(self, stop_at: float | None = None) -> None:
        deadline: Final = stop_at if stop_at is not None else asyncio.get_running_loop().time() + 60
        while asyncio.get_running_loop().time() + 2 < deadline and (
            pending := await self._store.pending_deliveries(limit=10)
        ):
            for item in pending:
                if asyncio.get_running_loop().time() + 2 >= deadline:
                    return
                await self._notify_one(item, deadline)

    async def _notify_one(self, item: IssuePendingDelivery, deadline: float) -> None:
        current: Final = await self._store.pending_delivery(item.delivery.id)
        if current is None:
            return
        remaining: Final = deadline - asyncio.get_running_loop().time()
        if remaining <= 2 or not await self._store.reserve_delivery(item.delivery.id):
            return
        try:
            delivered: Final = await asyncio.wait_for(
                self._delivery.send_issue(current), timeout=min(30, remaining - 1)
            )
        except Exception:  # noqa: BLE001  # persisted retry avoids losing a notification on transport failure
            await self._store.defer_delivery(item.delivery.id, "delivery_failed")
            verbose_proxy_logger.warning("Performance issue notification failed; queued for retry")
        else:
            if delivered:
                await self._store.mark_delivery_sent(item.delivery.id)
            else:
                await self._store.defer_delivery(item.delivery.id, "transport_not_configured")
