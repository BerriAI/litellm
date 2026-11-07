import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from itertools import chain, groupby
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast  # noqa: TID251  # Prisma delegates the query protocol dynamically

from pydantic import BaseModel, Field

from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils.model_deprecation import (
    DeprecationScanCursor,
    RouterDeprecationSource,
    deprecation_alert_state_key,
)
from litellm.repositories.analytics_budget import (
    AnalyticsCapacityError,
    AnalyticsDatabase,
    BoundedAnalyticsDatabase,
    analytics_deadline,
)
from litellm.repositories.config_repository import BoundedConfigRepository
from litellm.repositories.usage_alert_metrics import UsageAlertMetricsRepository
from litellm.types.proxy.issues import IssueCohort, IssueDeprecation, IssueImpact, IssueInput, IssueSignal
from litellm.types.proxy.model_deprecation import (
    DeprecationAlertState,
    ModelDeprecationInfo,
    ModelDeprecationUsage,
)

if TYPE_CHECKING:
    from litellm.models.issues import IssueStore
    from litellm.proxy.utils import PrismaClient

_PROJECTION_CHECKPOINT: Final = "issues:deprecation_projection"
_PAGE_SIZE: Final = 10
_WORK_BUDGET_SECONDS: Final = 5


class _ProjectionCheckpoint(BaseModel, frozen=True):
    hour: datetime
    observed_at: datetime | None = None
    source_revision: str = ""
    cursor: DeprecationScanCursor = Field(default_factory=DeprecationScanCursor)
    complete: bool = True


@dataclass(frozen=True, slots=True)
class DeprecationCandidate:
    usage_model: str
    model: ModelDeprecationInfo

    @property
    def state_key(self) -> str:
        return deprecation_alert_state_key(self.usage_model, self.model.deprecation_date)


def _candidates(model: ModelDeprecationInfo) -> tuple[DeprecationCandidate, ...]:
    base: Final = model.litellm_model if model.litellm_model is not None else model.model_name
    primary: Final = DeprecationCandidate(base, model)
    provider: Final = model.litellm_provider
    if not provider or base.startswith(f"{provider}/"):
        return (primary,)
    return (primary, DeprecationCandidate(f"{provider}/{base}", model))


def _identity(candidate: DeprecationCandidate) -> tuple[str, date]:
    return candidate.usage_model, candidate.model.deprecation_date


def deprecation_candidates(models: Sequence[ModelDeprecationInfo]) -> tuple[DeprecationCandidate, ...]:
    if len(models) > _PAGE_SIZE:
        raise ValueError("Read at most 10 retirement models per page")
    ordered: Final = sorted(
        chain.from_iterable(_candidates(model) for model in models if model.status != "upcoming"),
        key=lambda candidate: (*_identity(candidate), candidate.model.model_name),
    )
    return tuple(next(group) for _, group in groupby(ordered, _identity))


def deprecation_issue(
    candidate: DeprecationCandidate,
    usage: ModelDeprecationUsage,
    state: DeprecationAlertState,
    now: datetime,
) -> IssueInput:
    today: Final = now.astimezone(timezone.utc).date()
    model: Final = candidate.model
    name: Final = model.model_name or usage.model
    title: Final = (
        f"{name} retired on {model.deprecation_date.isoformat()}"
        if model.deprecation_date < today
        else f"{name} retires on {model.deprecation_date.isoformat()}"
    )
    return IssueInput(
        id=f"deprecation:{candidate.state_key.removeprefix('model_deprecation_alert:')}",
        cohort_key=candidate.state_key,
        kind="model_deprecation",
        family="retirement",
        title=title,
        cohort=IssueCohort(scope_type="model", scope_id=usage.model),
        observed_on=usage.last_used_date,
        observed_at=now,
        range_start=today - timedelta(days=29),
        range_end=today,
        grouping_reason="model_deprecation",
        explanation="This model has recorded requests in the last 30 UTC dates and is retiring within 30 days or has retired.",
        impact=IssueImpact(keys=usage.keys, users=usage.users, teams=usage.teams, models=1, requests=usage.requests),
        deprecation=IssueDeprecation(
            model=model,
            usage_model=usage.model,
            requests=usage.requests,
            last_used_date=usage.last_used_date,
            last_day_requests=usage.last_day_requests,
            state=state,
        ),
    )


def used_deprecation_issues(
    candidates: Sequence[DeprecationCandidate],
    usage: Mapping[str, ModelDeprecationUsage],
    states: Mapping[str, DeprecationAlertState],
    now: datetime,
) -> tuple[IssueInput, ...]:
    def build(candidate: DeprecationCandidate) -> IssueInput:
        state: Final = states.get(candidate.state_key) or DeprecationAlertState(
            model=candidate.usage_model, deprecation_date=candidate.model.deprecation_date
        )
        return deprecation_issue(candidate, usage[candidate.usage_model], state, now)

    return tuple(build(candidate) for candidate in candidates if candidate.usage_model in usage)


class DeprecationIssueProjector:
    def __init__(
        self,
        config: BoundedConfigRepository,
        metrics: UsageAlertMetricsRepository,
        issues: "IssueStore",
        source: RouterDeprecationSource,
    ) -> None:
        self._config = config
        self._metrics = metrics
        self._issues = issues
        self._source = source

    @classmethod
    def from_prisma(cls, prisma: "PrismaClient", source: RouterDeprecationSource) -> "DeprecationIssueProjector":
        from litellm.repositories.issues import IssuesRepository

        database: Final = BoundedAnalyticsDatabase(
            cast(AnalyticsDatabase, prisma.writer_db),  # cast-ok: generated Prisma transaction delegate
            background=True,
        )
        return cls(
            BoundedConfigRepository(database),
            UsageAlertMetricsRepository(database),
            IssuesRepository(database),
            source,
        )

    async def _page(self, candidates: tuple[DeprecationCandidate, ...], now: datetime) -> None:
        today: Final = now.astimezone(timezone.utc).date()
        models: Final = tuple(sorted(frozenset(candidate.usage_model for candidate in candidates)))
        usage: Final = await self._metrics.deprecation_usage(models, today - timedelta(days=29), today)
        records: Final = await self._config.get_params(tuple(candidate.state_key for candidate in candidates))
        states: Final = MappingProxyType(
            {record.param_name: DeprecationAlertState.model_validate(record.param_value) for record in records}
        )
        by_model: Final = MappingProxyType({row.model: row for row in usage})
        issues: Final = used_deprecation_issues(candidates, by_model, states, now)
        saved: Final = await self._issues.upsert_issues(issues)
        await self._issues.add_signal_batch(
            tuple(
                IssueSignal(
                    issue_id=issue.id, source="model_deprecation", source_id=issue.id, deprecation=issue.deprecation
                )
                for issue in saved
            )
        )

    async def run(self, now: datetime) -> None:
        deadline: Final = asyncio.get_running_loop().time() + _WORK_BUDGET_SECONDS
        try:
            with analytics_deadline(deadline):
                await asyncio.wait_for(self._run(now), timeout=_WORK_BUDGET_SECONDS)
        except (AnalyticsCapacityError, asyncio.TimeoutError):
            verbose_proxy_logger.debug("Model retirement scan deferred within its work budget")

    async def _run(self, now: datetime) -> None:
        hour: Final = now.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        previous: Final = await self._config.get_param(_PROJECTION_CHECKPOINT)
        stored: Final = _ProjectionCheckpoint.model_validate(previous.param_value) if previous is not None else None
        if stored is not None and stored.complete and stored.hour >= hour:
            return
        revision: Final = await self._source.prepare_revision(hour.date())
        if revision is None:
            return
        state = (  # rebind-ok: durable continuation after each completed page
            stored
            if stored is not None
            and not stored.complete
            and stored.source_revision == revision
            and stored.hour.date() == hour.date()
            else _ProjectionCheckpoint(hour=hour, observed_at=now, source_revision=revision, complete=False)
        )
        while self._source.revision == state.source_revision:
            page = self._source.page(state.hour.date(), state.cursor)
            if not page.complete and not page.models and page.cursor == state.cursor:
                return
            observed_at = state.observed_at or now
            candidates = deprecation_candidates(page.models)
            for offset in range(0, len(candidates), _PAGE_SIZE):
                await self._page(candidates[offset : offset + _PAGE_SIZE], observed_at)
            state = state.model_copy(update={"cursor": page.cursor})
            await self._config.set_param(_PROJECTION_CHECKPOINT, state.model_dump(mode="json"))
            if self._source.revision != state.source_revision:
                return
            if page.complete:
                await self._issues.finish_deprecation_projection(observed_at)
                finished = state.model_copy(update={"hour": hour, "complete": True})
                await self._config.set_param(_PROJECTION_CHECKPOINT, finished.model_dump(mode="json"))
                return
            await asyncio.sleep(0)
