import asyncio
from collections.abc import Awaitable, Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from itertools import chain
from types import MappingProxyType
from typing import Final, Literal, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from typing_extensions import ReadOnly, TypedDict, Unpack

from litellm.proxy.roi_calculator.estimator import CompletionCaller, Estimator, cache_context
from litellm.proxy.roi_calculator.github import GitHub, GitHubPullListItem, SourceError
from litellm.proxy.roi_calculator.pull_cache import cache_key, settings_fingerprint
from litellm.types.roi_calculator import (
    ROIEstimate,
    ROIPullEvidence,
    ROIPullRecord,
    ROIReport,
    ROISettings,
    ROISpendRecord,
    ROISyncStatus,
)

PR_CONCURRENCY: Final = 3
_ESTIMATE_ADAPTER: Final = TypeAdapter(ROIEstimate)
_REPORT_ADAPTER: Final = TypeAdapter(ROIReport)
_JSON_OBJECT_ADAPTER: Final = TypeAdapter(dict[str, object])


class _ConfigParam(Protocol):
    @property
    def param_value(self) -> object: ...


class _ReportRepository(Protocol):
    async def get_param(self, param_name: str) -> _ConfigParam | None: ...

    async def set_param(self, param_name: str, param_value: object) -> object: ...


class _DailySpendTable(Protocol):
    async def group_by(
        self,
        *,
        by: list[Literal["user_id", "date"]],
        sum: dict[str, object],
        where: dict[str, object],
        order: dict[str, object],
    ) -> Sequence[Mapping[str, object]]: ...


class _UserTable(Protocol):
    async def find_many(
        self,
        *,
        where: dict[str, object],
    ) -> Sequence[Mapping[str, object]]: ...


class _PrismaDatabase(Protocol):
    @property
    def litellm_dailyuserspend(self) -> _DailySpendTable: ...

    @property
    def litellm_usertable(self) -> _UserTable: ...


@runtime_checkable
class _SpendPrismaClient(Protocol):
    @property
    def db(self) -> _PrismaDatabase: ...


def spend_prisma_client(prisma_client: object) -> _SpendPrismaClient:
    if not isinstance(prisma_client, _SpendPrismaClient):
        raise TypeError("The database client does not support spend queries.")
    return prisma_client


class _DailySpendSums(BaseModel):
    spend: float = 0.0
    api_requests: int = 0


class _DailySpendGroup(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user_id: str | None
    date: str
    sums: _DailySpendSums = Field(alias="_sum")


class _UserEmail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user_id: str
    user_email: str | None


_DAILY_SPEND_GROUPS: Final = TypeAdapter(tuple[_DailySpendGroup, ...])
_USER_EMAILS: Final = TypeAdapter(tuple[_UserEmail, ...])


async def read_spend(
    prisma_client: _SpendPrismaClient,
    start: date,
    end: date,
) -> tuple[ROISpendRecord, ...]:
    from litellm.proxy.roi_calculator.analytics import normalize_email

    database: Final = prisma_client.db
    daily_table: Final = database.litellm_dailyuserspend
    group_by: Final[list[Literal["user_id", "date"]]] = [
        "user_id",
        "date",
    ]  # mutable-ok: prisma client serializer only accepts builtin dict/list
    sums: Final[dict[str, object]] = {
        "spend": True,
        "api_requests": True,
    }  # mutable-ok: prisma client serializer only accepts builtin dict/list
    date_filter: Final[dict[str, object]] = {
        "date": {
            "gte": start.isoformat(),
            "lte": end.isoformat(),
        }
    }  # mutable-ok: prisma client serializer only accepts builtin dict/list
    order: Final[dict[str, object]] = {
        "date": "asc"
    }  # mutable-ok: prisma client serializer only accepts builtin dict/list
    groups: Final = _DAILY_SPEND_GROUPS.validate_python(
        await daily_table.group_by(
            by=group_by,
            sum=sums,
            where=date_filter,
            order=order,
        )
    )
    user_ids: Final = tuple(sorted(frozenset(group.user_id for group in groups if group.user_id)))
    user_table: Final = database.litellm_usertable
    user_filter: Final[dict[str, object]] = {
        "user_id": {"in": list(user_ids)}
    }  # mutable-ok: prisma client serializer only accepts builtin dict/list
    users: Final = _USER_EMAILS.validate_python(
        await user_table.find_many(
            where=user_filter,
        )
        if user_ids
        else ()
    )
    emails: Final[Mapping[str, str]] = MappingProxyType(
        {user.user_id: normalize_email(user.user_email) for user in users if normalize_email(user.user_email)}
    )
    return tuple(
        ROISpendRecord(
            date=group.date,
            user_id=group.user_id or "",
            email=emails.get(group.user_id or "", "") or normalize_email(group.user_id),
            spend=group.sums.spend,
            requests=group.sums.api_requests,
        )
        for group in groups
    )


class GitHubFactory(Protocol):
    def __call__(
        self,
        settings: ROISettings,
        transport: httpx.AsyncBaseTransport | None,
    ) -> GitHub: ...


class SpendReader(Protocol):
    def __call__(
        self,
        start: date,
        end: date,
    ) -> Awaitable[tuple[ROISpendRecord, ...]]: ...


class SyncClock(Protocol):
    def __call__(self) -> datetime: ...


class _StatusUpdate(TypedDict, total=False):
    running: ReadOnly[bool]
    phase: ReadOnly[Literal["idle", "spend", "repositories", "estimates", "complete", "cancelled", "error"]]
    stage: ReadOnly[str]
    done: ReadOnly[int]
    total: ReadOnly[int]
    estimated: ReadOnly[int]
    reused: ReadOnly[int]
    needs_attention: ReadOnly[int]
    error: ReadOnly[str | None]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


async def _estimate_with_fallback(
    estimator: Estimator,
    evidence: ROIPullEvidence,
) -> ROIEstimate:
    try:
        return await estimator.estimate(evidence)
    except SourceError as exc:
        estimate: Final[ROIEstimate] = {
            "status": "error",
            "hours": None,
            "reasoning": str(exc),
        }
        return estimate


class SyncManager:
    def __init__(
        self,
        github_factory: GitHubFactory = GitHub,
        clock: SyncClock = _utc_now,
    ) -> None:
        self._github_factory: Final = github_factory
        self._clock: Final = clock
        self._status: ROISyncStatus = ROISyncStatus(
            running=False,
            phase="idle",
            stage="Idle",
            done=0,
            total=0,
            estimated=0,
            reused=0,
            needs_attention=0,
            error=None,
        )
        self._task: asyncio.Task[None] | None = None

    @property
    def status(self) -> ROISyncStatus:
        return self._status

    def start(
        self,
        settings: ROISettings,
        repository: _ReportRepository,
        spend_reader: SpendReader,
        complete: CompletionCaller,
        github_transport: httpx.AsyncBaseTransport | None = None,
    ) -> bool:
        if self._status.running or not settings.repos or not settings.estimator_model:
            return False
        self._status = ROISyncStatus(
            running=True,
            phase="spend",
            stage="Reading gateway spend",
            done=0,
            total=0,
            estimated=0,
            reused=0,
            needs_attention=0,
            error=None,
        )
        self._task = asyncio.create_task(self._run(settings, repository, spend_reader, complete, github_transport))
        return True

    async def cancel(self) -> bool:
        task: Final = self._task
        if task is None or task.done():
            return False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self._update_status(running=False, phase="cancelled", stage="Sync cancelled")
        return True

    async def _run(
        self,
        settings: ROISettings,
        repository: _ReportRepository,
        spend_reader: SpendReader,
        complete: CompletionCaller,
        github_transport: httpx.AsyncBaseTransport | None,
    ) -> None:
        github: Final = self._github_factory(settings, github_transport)
        try:
            end: Final = self._clock().date()
            start: Final = end - timedelta(days=settings.backfill_days - 1)
            spend: Final = await spend_reader(start, end)
            self._update_status(phase="repositories", stage="Reading configured repositories")
            pull_groups: Final = await asyncio.gather(*(github.pulls(repo, start, end) for repo in settings.repos))
            queue: Final = tuple(
                chain.from_iterable(
                    ((repo, pull) for pull in pulls) for repo, pulls in zip(settings.repos, pull_groups, strict=True)
                )
            )
            context: Final = cache_context(settings)
            previous: Final = await self._previous_report(repository)
            previous_pulls: Final[Mapping[str, ROIPullRecord]] = MappingProxyType(
                {
                    pull["cache_key"]: pull
                    for pull in (previous["pulls"] if previous else ())
                    if pull["cache_key"] is not None
                }
            )
            indexed_queue: Final = tuple(
                (index, repo, pull, cache_key(settings, context, repo, pull))
                for index, (repo, pull) in enumerate(queue)
            )
            cached: Final = tuple(
                (index, previous_pulls[key])
                for index, _, _, key in indexed_queue
                if key is not None
                and (previous_pull := previous_pulls.get(key)) is not None
                and previous_pull["estimate"]["status"] == "estimated"
            )
            cached_by_index: Final[Mapping[int, ROIPullRecord]] = MappingProxyType(
                {index: self._cached_record(pull) for index, pull in cached}
            )
            pending: Final = tuple(item for item in indexed_queue if item[0] not in cached_by_index)
            reused_count: Final = len(cached)
            self._update_status(
                phase="estimates",
                stage="Estimating new or changed pull requests",
                done=reused_count,
                total=len(queue),
                estimated=reused_count,
                reused=reused_count,
            )
            semaphore: Final = asyncio.Semaphore(PR_CONCURRENCY)
            estimator: Final = Estimator(settings, complete)

            async def process(
                item: tuple[int, str, GitHubPullListItem, str | None],
            ) -> tuple[int, ROIPullRecord]:
                async with semaphore:
                    index, repo, pull, key = item
                    evidence: Final = await github.evidence(repo, pull)
                    estimate: Final = await _estimate_with_fallback(estimator, evidence)
                    record: Final = self._report_record(evidence, estimate, key)
                    self._update_estimate_progress(estimate)
                    return index, record

            workers: Final = tuple(asyncio.create_task(process(item)) for item in pending)
            try:
                processed: Final = await asyncio.gather(*workers)
            finally:
                for worker in workers:
                    if not worker.done():
                        worker.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
            processed_by_index: Final[Mapping[int, ROIPullRecord]] = MappingProxyType(
                {index: pull for index, pull in processed}
            )
            report_pulls: Final[Mapping[int, ROIPullRecord]] = MappingProxyType(
                {**cached_by_index, **processed_by_index}
            )
            report: Final = ROIReport(
                mode="live",
                start=start.isoformat(),
                end=end.isoformat(),
                synced_at=self._clock().isoformat(),
                repos=settings.repos,
                estimator_model=settings.estimator_model,
                estimator_prompt=settings.estimator_prompt,
                effort_basis="without_ai",
                spend=spend,
                pulls=tuple(report_pulls[index] for index in range(len(queue))),
                settings_fingerprint=settings_fingerprint(settings),
                warnings=(),
            )
            await github.close()
            report_json: Final[dict[str, object]] = _JSON_OBJECT_ADAPTER.validate_python(
                _REPORT_ADAPTER.dump_python(report, mode="json")
            )
            await repository.set_param("roi_calculator_report", report_json)
            self._update_status(phase="complete", stage="Up to date")
        except asyncio.CancelledError:
            self._update_status(phase="cancelled", stage="Sync cancelled")
            raise
        except SourceError as exc:
            self._update_status(phase="error", stage="Sync failed", error=str(exc))
        except Exception:
            self._update_status(
                phase="error",
                stage="Sync failed",
                error=(
                    "Unexpected source response. No partial report was saved. "
                    "Check service compatibility and try again."
                ),
            )
        finally:
            try:
                if self._status.phase != "complete":
                    await github.close()
            finally:
                self._update_status(running=False)

    def _update_status(self, **update: Unpack[_StatusUpdate]) -> None:
        status: Final = ROISyncStatus.model_validate(MappingProxyType({**self._status.model_dump(), **update}))
        self._status = status

    async def _previous_report(self, repository: _ReportRepository) -> ROIReport | None:
        parameter: Final = await repository.get_param("roi_calculator_report")
        if parameter is None:
            return None
        try:
            return _REPORT_ADAPTER.validate_python(parameter.param_value)
        except Exception:
            return None

    def _cached_record(self, pull: ROIPullRecord) -> ROIPullRecord:
        estimate: Final = _ESTIMATE_ADAPTER.validate_python(MappingProxyType({**pull["estimate"], "cached": True}))
        return ROIPullRecord(
            repo=pull["repo"],
            number=pull["number"],
            title=pull["title"],
            url=pull["url"],
            login=pull["login"],
            emails=pull["emails"],
            profile_email=pull["profile_email"],
            merged_at=pull["merged_at"],
            head_sha=pull["head_sha"],
            additions=pull["additions"],
            deletions=pull["deletions"],
            changed_files=pull["changed_files"],
            commit_count=pull["commit_count"],
            incomplete_metadata=pull["incomplete_metadata"],
            estimate=estimate,
            cache_key=pull.get("cache_key"),
        )

    def _report_record(
        self,
        evidence: ROIPullEvidence,
        estimate: ROIEstimate,
        key: str | None,
    ) -> ROIPullRecord:
        return ROIPullRecord(
            repo=evidence["repo"],
            number=evidence["number"],
            title=evidence["title"],
            url=evidence["url"],
            login=evidence["login"],
            emails=evidence["emails"],
            profile_email=evidence["profile_email"],
            merged_at=evidence["merged_at"],
            head_sha=evidence["head_sha"],
            additions=evidence["additions"],
            deletions=evidence["deletions"],
            changed_files=evidence["changed_files"],
            commit_count=evidence["commit_count"],
            incomplete_metadata=evidence["incomplete_metadata"],
            estimate=estimate,
            cache_key=key,
        )

    def _update_estimate_progress(self, estimate: ROIEstimate) -> None:
        estimated: Final = estimate["status"] == "estimated"
        reused: Final = estimate.get("cached", False)
        self._update_status(
            done=self._status.done + 1,
            estimated=self._status.estimated + int(estimated),
            reused=self._status.reused + int(reused),
            needs_attention=self._status.needs_attention + int(not estimated),
        )
