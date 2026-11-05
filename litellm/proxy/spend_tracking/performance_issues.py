"""Project daily observations into scoped, persisted investigations."""

import asyncio
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from itertools import chain, groupby
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
from litellm.types.proxy.issues import (
    Issue,
    IssueCohort,
    IssueComparison,
    IssueDeliveryBinding,
    IssueDeliveryInput,
    IssueImpact,
    IssueInput,
    IssueModelPair,
    IssuePendingDelivery,
    IssueProjectionSignal,
    IssueSignal,
    IssueSignalGroupingReason,
)
from litellm.types.proxy.usage_alert_defaults import PerformanceIssueFamily, is_default_usage_rule
from litellm.types.proxy.usage_alerts import UsageAlertDirection, UsageAlertMetric, UsageAlertRule, UsageAlertScope

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
        self._overrides: frozenset[tuple[UsageAlertScope, UsageAlertMetric, UsageAlertDirection, str | None]] = (
            frozenset()
        )

    async def run(self, now: datetime, day: date, rules: tuple[UsageAlertRule, ...], *, ready: bool) -> None:
        self._overrides = frozenset(
            (rule.scope_type, rule.metric, rule.direction, rule.scope_id)
            for rule in rules
            if rule.enabled and not is_default_usage_rule(rule)
        )
        observed_at: Final = now.replace(microsecond=now.microsecond // 1000 * 1000)
        version: Final = hashlib.sha256(
            ("performance-v1:" + ",".join(f"{rule.id}:{rule.updated_at.isoformat()}" for rule in rules)).encode()
        ).hexdigest()
        if ready and not await self._store.projection_completed(day, version):
            await self._store.recover_issues(day)
            providers: Final = await self._provider_evidence(day, rules, observed_at)
            for metric in _METRICS:
                for scope in _SCOPES:
                    await self._scope(observed_at, day, scope, metric, providers)
            await self._store.complete_projection(day, version)
        await self._notify()
        await self._store.prune()

    async def _candidates(
        self, sources: tuple[IssueProjectionSignal, ...], scope: UsageAlertScope, day: date
    ) -> tuple[_Candidate, ...]:
        eligible: Final = tuple(
            source
            for source in sources
            if not is_default_usage_rule(source.rule)
            or not any(
                (scope, source.rule.metric, source.rule.direction, entity) in self._overrides
                for entity in (None, source.alert.scope_id)
            )
        )
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

    async def _scope(
        self,
        now: datetime,
        day: date,
        scope: UsageAlertScope,
        metric: UsageAlertMetric,
        providers: Mapping[str, _ProviderGroup],
    ) -> None:
        cursor: str | None = None  # rebind-ok: bounded source pagination
        while sources := await self._store.active_signals(day, scope_type=scope, metric=metric, cursor=cursor):
            await self._project_page(sources, scope, day, providers, now)
            cursor = sources[-1].alert.id
            await asyncio.sleep(0)

    async def _project_page(
        self,
        sources: tuple[IssueProjectionSignal, ...],
        scope: UsageAlertScope,
        day: date,
        providers: Mapping[str, _ProviderGroup],
        now: datetime,
    ) -> None:
        candidates: Final = await self._candidates(sources, scope, day)
        provider_targets: Final = await self._provider_targets(candidates, providers, now)
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

    async def _spend_candidates(
        self, scope: UsageAlertScope, entities: tuple[str, ...] | None, day: date
    ) -> tuple[_Candidate, ...]:
        found: Final[list[_Candidate]] = []  # mutable-ok: bounded entity batch may match multiple configured rules
        cursor: str | None = None  # rebind-ok: page matching rules without one query per entity
        while sources := await self._store.active_signals(
            day, scope_type=scope, metric="spend", scope_ids=entities, cursor=cursor
        ):
            found.extend(await self._candidates(sources, scope, day))
            cursor = sources[-1].alert.id
        return tuple(found)

    async def _volume_targets(self, candidates: tuple[_Candidate, ...], now: datetime) -> Mapping[str, _Target]:
        traffic: Final = tuple(
            item
            for item in candidates
            if item.source.rule.metric == "requests" and item.source.rule.direction == "increase"
        )
        if not traffic:
            return {}
        scope: Final = traffic[0].source.rule.scope_type
        entities: Final = tuple(
            dict.fromkeys(item.source.alert.scope_id for item in traffic if item.source.alert.scope_id is not None)
        )
        spending: Final = await self._spend_candidates(
            scope, entities if scope != "deployment" else None, traffic[0].window.day
        )
        by_entity: Final = {
            entity: tuple(group)
            for entity, group in groupby(
                sorted(spending, key=lambda item: item.source.alert.scope_id or ""),
                lambda item: item.source.alert.scope_id,
            )
        }
        keys: Final = tuple(_cohort_key("spend", scope, item.source.alert.scope_id, "increase") for item in traffic)
        issues: Final = {
            issue.cohort_key: issue for issue in await self._store.find_active_issues(keys) if issue.last_seen == now
        }
        return {
            item.source.alert.id: _Target(
                issue.model_copy(update={"observed_at": now}),
                "volume_explains_spend",
                "Request volume and spend increased by similar proportions in the same comparison window.",
            )
            for item in traffic
            if (issue := issues.get(_cohort_key("spend", scope, item.source.alert.scope_id, "increase"))) is not None
            and any(_volume_explains_spend(item, spend) for spend in by_entity.get(item.source.alert.scope_id, ()))
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
        spending: Final = await self._spend_candidates("key", key_ids, traffic[0].window.day)
        matching: Final = tuple(item for item in spending if self._volume_matches_key_spend(item))
        by_key: Final = {
            identity: tuple(group)
            for identity, group in groupby(
                sorted(matching, key=lambda item: item.source.alert.scope_id or ""),
                lambda item: item.source.alert.scope_id,
            )
        }
        keys: Final = tuple(_cohort_key("spend", "key", key, "increase") for key in key_ids)
        issues: Final = {
            issue.cohort_key: issue for issue in await self._store.find_active_issues(keys) if issue.last_seen == now
        }
        return {
            item.source.alert.id: _Target(
                issue.model_copy(update={"observed_at": now}),
                "volume_explains_spend",
                "The same key explains this source's request growth and has a matching increase in spend.",
            )
            for item in traffic
            if (issue := issues.get(_cohort_key("spend", "key", attributions[item.source.alert.id].key_id, "increase")))
            is not None
            and any(
                (spend.window.start, spend.window.day) == (item.window.start, item.window.day)
                for spend in by_key.get(attributions[item.source.alert.id].key_id, ())
            )
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

    async def _provider_evidence(
        self, day: date, rules: tuple[UsageAlertRule, ...], now: datetime
    ) -> Mapping[str, _ProviderGroup]:
        rule: Final = next(
            (
                rule
                for rule in rules
                if is_default_usage_rule(rule) and rule.metric == "latency" and rule.direction == "increase"
            ),
            None,
        )
        if rule is None:
            return {}
        groups: Final[dict[str, _ProviderGroup]] = {}  # mutable-ok: retain one confirmed cohort per provider
        reserved: Final[set[str]] = set()  # mutable-ok: an active frozen cohort prevents a competing subset episode
        cursor: str | None = None  # rebind-ok: bounded existing-cohort pagination
        while existing := await self._store.active_provider_issues(cursor=cursor):
            for prior, recovered, evidence in await self._prior_provider_page(existing, day, rule, now):
                if recovered or prior.cohort.scope_id is None:
                    continue
                reserved.add(prior.cohort.scope_id)
                if evidence is not None:
                    groups[prior.cohort.scope_id] = _ProviderGroup(evidence, rule)
            cursor = existing[-1].id
        source_cursor: str | None = None  # rebind-ok: stop discovery when no compatible active source exists
        compatible = False  # rebind-ok: early-exit source scan
        while sources := await self._store.active_signals(day, metric="latency", cursor=source_cursor):
            if any(is_default_usage_rule(source.rule) and self._same_policy(source.rule, rule) for source in sources):
                compatible = True
                break
            source_cursor = sources[-1].alert.id
        if not compatible:
            return groups
        pair_cursor: IssueModelPair | None = None  # rebind-ok: keyset pagination of model pairs
        while pairs := await self._providers.candidate_pairs(day, pair_cursor):
            for provider, selected in groupby(pairs, lambda pair: pair.provider):
                if provider in reserved or provider in groups:
                    continue
                if (candidate_evidence := await self._providers.latency_cohort(tuple(selected), day, rule)) is not None:
                    groups[provider] = _ProviderGroup(candidate_evidence, rule)
            pair_cursor = pairs[-1]
        return groups

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

    async def _provider_targets(
        self, candidates: tuple[_Candidate, ...], providers: Mapping[str, _ProviderGroup], now: datetime
    ) -> Mapping[str, _Target]:
        eligible: Final = tuple(
            item
            for item in candidates
            if item.source.rule.metric == "latency" and is_default_usage_rule(item.source.rule)
        )
        if not eligible or not providers:
            return {}
        scope: Final = eligible[0].source.rule.scope_type
        by_reference: Final = {item.source.alert.id: item for item in eligible}
        hints: Final = await self._metrics.issue_source_providers(scope, tuple(item.window for item in eligible))
        targets: Final[dict[str, _Target]] = {}  # mutable-ok: each source belongs to at most one 80%-coverage provider
        for provider, matches in groupby(sorted(hints, key=lambda hint: hint.provider), lambda hint: hint.provider):
            targets.update(await self._provider_matches(provider, tuple(matches), by_reference, providers, scope, now))
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
                explanation="Several keys and models slowed down together. The comparison holds today's key and model mix constant and excludes keys still collecting a baseline. This is a shared pattern, not proof of a common cause.",
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
        bindings: Final = tuple(
            chain.from_iterable(
                _bindings(candidate, saved[targets[candidate.source.alert.id].issue.cohort_key].id)
                for candidate in candidates
            )
        )
        for offset in range(0, len(bindings), 500):
            await self._store.queue_deliveries(bindings[offset : offset + 500])

    async def _notify(self) -> None:
        deadline: Final = asyncio.get_running_loop().time() + 60
        while asyncio.get_running_loop().time() < deadline and (pending := await self._store.pending_deliveries()):
            for item in pending:
                if asyncio.get_running_loop().time() >= deadline:
                    return
                await self._notify_one(item)

    async def _notify_one(self, item: IssuePendingDelivery) -> None:
        current: Final = await self._store.pending_delivery(item.delivery.id)
        if current is None:
            return
        try:
            delivered: Final = await asyncio.wait_for(self._delivery.send_issue(current), timeout=30)
        except Exception:  # noqa: BLE001  # persisted retry avoids losing a notification on transport failure
            await self._store.defer_delivery(item.delivery.id, "delivery_failed")
            verbose_proxy_logger.warning("Performance issue notification failed; queued for retry")
        else:
            if delivered:
                await self._store.mark_delivery_sent(item.delivery.id)
            else:
                await self._store.defer_delivery(item.delivery.id, "transport_not_configured")
