import asyncio
from collections.abc import Awaitable, Mapping, Sequence
from contextlib import suppress
from datetime import date, datetime, timedelta, timezone
from itertools import chain
from types import MappingProxyType
from typing import Final, Literal, NamedTuple, Protocol, runtime_checkable
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from typing_extensions import ReadOnly, TypedDict, Unpack

from litellm.proxy.roi_calculator.estimator import CompletionCaller, Estimator, EstimatorModel, cache_context
from litellm.proxy.roi_calculator.github import GitHub, GitHubPullListItem, SourceError
from litellm.proxy.roi_calculator.pull_cache import cache_key, settings_fingerprint
from litellm.repositories.chunked_in import find_many_in
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


class SyncCoordinator(Protocol):
    async def status(self) -> ROISyncStatus | None: ...
    async def acquire(self, owner: str, status: ROISyncStatus, scheduled_interval: float = 0) -> bool: ...
    async def heartbeat(self, owner: str, status: ROISyncStatus) -> bool: ...
    async def finish(self, owner: str, status: ROISyncStatus, report: ROIReport | None = None) -> bool: ...


class _DailySpendTable(Protocol):
    async def group_by(
        self,
        *,
        by: Sequence[Literal["user_id", "date"]],
        sum: Mapping[str, object],
        where: Mapping[str, object],
        order: Mapping[str, object],
    ) -> Sequence[Mapping[str, object]]: ...


class _UserTable(Protocol):
    async def find_many(
        self,
        *,
        where: Mapping[str, object],
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
    group_by: Final = TypeAdapter(list[Literal["user_id", "date"]]).validate_python(("user_id", "date"))
    sums: Final = _JSON_OBJECT_ADAPTER.validate_python(MappingProxyType({"spend": True, "api_requests": True}))
    date_filter: Final = _JSON_OBJECT_ADAPTER.validate_python(
        MappingProxyType(
            {
                "date": _JSON_OBJECT_ADAPTER.validate_python(
                    MappingProxyType({"gte": start.isoformat(), "lte": end.isoformat()})
                )
            }
        )
    )
    order: Final = _JSON_OBJECT_ADAPTER.validate_python(MappingProxyType({"date": "asc"}))
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
    users: Final = _USER_EMAILS.validate_python(await find_many_in(user_table, "user_id", user_ids))
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


async def _unavailable_record(github: GitHub, repo: str, pull: GitHubPullListItem, error: SourceError) -> ROIPullRecord:
    login: Final = pull.user.login if pull.user and pull.user.login else "deleted-user"
    profile: Final = await github.profile_email(login)
    estimate: Final[ROIEstimate] = {
        "status": "needs_review",
        "hours": None,
        "reasoning": f"PR metadata could not be read: {error} Run analysis again to retry this PR.",
    }
    return ROIPullRecord(
        repo=repo,
        number=pull.number,
        title=pull.title,
        url=pull.html_url,
        login=login,
        emails=(profile,) if profile else (),
        profile_email=profile,
        commit_emails=(),
        merged_at=pull.merged_at or pull.updated_at,
        head_sha=pull.head.sha if pull.head else "",
        additions=0,
        deletions=0,
        changed_files=0,
        commit_count=0,
        incomplete_metadata=True,
        estimate=estimate,
        cache_key=None,
    )


class _ProcessedPull(NamedTuple):
    position: int
    record: ROIPullRecord
    metadata_unavailable: bool = False


class _RepositoryPulls(NamedTuple):
    repo: str
    pulls: tuple[GitHubPullListItem, ...]
    unavailable: bool = False


class _RepositoryBatch(NamedTuple):
    queue: tuple[tuple[str, GitHubPullListItem], ...]
    unavailable_repos: tuple[str, ...]
    warnings: tuple[str, ...]
    stage: str


async def _read_repository(github: GitHub, repo: str, start: date, end: date) -> _RepositoryPulls:
    try:
        return _RepositoryPulls(repo, await github.pulls(repo, start, end))
    except SourceError:
        return _RepositoryPulls(repo, (), unavailable=True)


async def _read_repositories(github: GitHub, repos: tuple[str, ...], start: date, end: date) -> _RepositoryBatch:
    groups: Final = await asyncio.gather(*(_read_repository(github, repo, start, end) for repo in repos))
    unavailable: Final = tuple(group.repo for group in groups if group.unavailable)
    if len(unavailable) == len(repos):
        raise SourceError(
            "GitHub could not read any selected repository. No new report was published; "
            "check repository access or try analysis again later."
        )
    queue: Final = tuple(chain.from_iterable(((group.repo, pull) for pull in group.pulls) for group in groups))
    if unavailable and not queue:
        raise SourceError(
            f"GitHub could not read {', '.join(unavailable)}, and the accessible repositories returned no pull requests. "
            "No new report was published; check repository access or try analysis again later."
        )
    warnings: Final = (
        (
            (
                f"Incomplete report: could not read {', '.join(unavailable)}. "
                "Results include only accessible repositories. Spend-per-hour figures are unavailable until "
                "all selected repositories can be read. Check repository access or run analysis again to retry."
            ),
        )
        if unavailable
        else ()
    )
    return _RepositoryBatch(
        queue,
        unavailable,
        warnings,
        "Analysis complete with unavailable repositories" if unavailable else "Analysis complete",
    )


def _processed_records(processed: tuple[_ProcessedPull, ...]) -> Mapping[int, ROIPullRecord]:
    if processed and all(item.metadata_unavailable for item in processed):
        raise SourceError(
            "GitHub could not provide PR metadata. No new report was published; try analysis again later."
        )
    if any(item.record["estimate"]["status"] == "error" for item in processed) and not any(
        item.record["estimate"]["status"] == "estimated" for item in processed
    ):
        raise SourceError(
            "The estimator could not score any pull requests. No new report was published; "
            "check the estimator connection or try analysis again later."
        )
    return MappingProxyType({item.position: item.record for item in processed})


async def _cache_estimated_pull(repository: _ReportRepository, key: str | None, record: ROIPullRecord) -> None:
    if key is None or record["estimate"]["status"] != "estimated":
        return
    await repository.set_param(
        "roi_calculator_pull_" + key,
        _JSON_OBJECT_ADAPTER.validate_python(TypeAdapter(ROIPullRecord).dump_python(record, mode="json")),
    )


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
        self._coordinator: SyncCoordinator | None = None
        self._owner: str = ""
        self._start_lock: Final = asyncio.Lock()

    @property
    def status(self) -> ROISyncStatus:
        if self._status.started_at is None:
            return self._status
        start: Final = datetime.fromisoformat(self._status.started_at)
        finish: Final = datetime.fromisoformat(self._status.finished_at) if self._status.finished_at else self._clock()
        elapsed: Final = max(0, int((finish - start).total_seconds()))
        remaining: Final = (
            max(0, round(elapsed / self._status.done * (self._status.total - self._status.done)))
            if self._status.running and self._status.done >= PR_CONCURRENCY
            else None
        )
        return self._status.model_copy(
            update=MappingProxyType({"elapsed_seconds": elapsed, "remaining_seconds": remaining})
        )

    async def start(
        self,
        settings: ROISettings,
        repository: _ReportRepository,
        spend_reader: SpendReader,
        complete: CompletionCaller,
        github_transport: httpx.AsyncBaseTransport | None = None,
        estimator_models: tuple[EstimatorModel, ...] | None = None,
        coordinator: SyncCoordinator | None = None,
        scheduled_interval: float = 0,
    ) -> bool:
        async with self._start_lock:
            if not settings.repos or not settings.estimator_model:
                return False
            if self._status.running:
                if coordinator is None:
                    return False
                shared: Final = await coordinator.status()
                if shared is not None and shared.running:
                    return False
                await self.cancel()
            initial_status: Final = ROISyncStatus(
                running=True,
                started_at=self._clock().isoformat(),
                phase="spend",
                stage="Reading gateway spend",
                done=0,
                total=0,
                estimated=0,
                reused=0,
                needs_attention=0,
                error=None,
            )
            owner: Final = str(uuid4())
            if coordinator is not None and not await coordinator.acquire(owner, initial_status, scheduled_interval):
                return False
            self._status = initial_status
            self._coordinator = coordinator
            self._owner = owner
            self._task = asyncio.create_task(
                self._run(
                    settings, repository, spend_reader, complete, github_transport, estimator_models, coordinator, owner
                )
            )
            return True

    async def cancel(self) -> bool:
        task: Final = self._task
        if task is None or task.done():
            return False
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        self._update_status(running=False, phase="cancelled", stage="Sync cancelled")
        self._status = self.status.model_copy(update=MappingProxyType({"finished_at": self._clock().isoformat()}))
        if self._coordinator is not None:
            await self._coordinator.finish(self._owner, self.status)
        return True

    async def _heartbeat(
        self, task: asyncio.Task[object] | None, coordinator: SyncCoordinator | None, owner: str
    ) -> None:
        if coordinator is None or task is None:
            return
        try:
            while True:
                await asyncio.sleep(1)
                if not await coordinator.heartbeat(owner, self.status):
                    task.cancel()
                    return
        except Exception:  # noqa: BLE001 - any coordination failure must stop a worker before its lease expires
            task.cancel()

    async def _run(
        self,
        settings: ROISettings,
        repository: _ReportRepository,
        spend_reader: SpendReader,
        complete: CompletionCaller,
        github_transport: httpx.AsyncBaseTransport | None,
        estimator_models: tuple[EstimatorModel, ...] | None,
        coordinator: SyncCoordinator | None,
        owner: str,
    ) -> None:
        monitor: Final = asyncio.create_task(self._heartbeat(asyncio.current_task(), coordinator, owner))
        github: Final = self._github_factory(settings, github_transport)
        try:
            end: Final = self._clock().date()
            start: Final = end - timedelta(days=settings.backfill_days - 1)
            spend: Final = await spend_reader(start, end)
            self._update_status(phase="repositories", stage="Reading configured repositories")
            repositories: Final = await _read_repositories(github, settings.repos, start, end)
            queue: Final = repositories.queue
            context: Final = cache_context(settings, estimator_models)
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
            self._update_status(
                phase="estimates",
                stage="Estimating new or changed pull requests",
                total=len(queue),
            )
            estimator: Final = Estimator(settings, complete, estimator_models)

            async def process(
                item: tuple[int, str, GitHubPullListItem, str | None],
            ) -> _ProcessedPull:
                index, repo, pull, key = item
                saved: Final = await repository.get_param("roi_calculator_pull_" + key) if key is not None else None
                cached_pull: Final = (
                    TypeAdapter(ROIPullRecord).validate_python(saved.param_value)
                    if saved is not None
                    else previous_pulls.get(key or "")
                )
                if (
                    cached_pull is not None
                    and cached_pull["estimate"]["status"] == "estimated"
                    and "commit_emails" in cached_pull
                ):
                    profile: Final = await github.profile_email(
                        cached_pull["login"], fallback=cached_pull.get("profile_email", "")
                    )
                    cached_record: Final = TypeAdapter(ROIPullRecord).validate_python(
                        MappingProxyType(
                            {
                                **self._cached_record(cached_pull),
                                "profile_email": profile,
                                "emails": tuple(
                                    sorted(
                                        frozenset(email for email in (*cached_pull["commit_emails"], profile) if email)
                                    )
                                ),
                            }
                        )
                    )
                    await _cache_estimated_pull(repository, key, cached_record)
                    self._update_estimate_progress(cached_record["estimate"])
                    return _ProcessedPull(index, cached_record)
                try:
                    evidence: Final = await github.evidence(repo, pull)
                except SourceError as exc:
                    unavailable: Final = await _unavailable_record(github, repo, pull, exc)
                    self._update_estimate_progress(unavailable["estimate"])
                    return _ProcessedPull(index, unavailable, metadata_unavailable=True)
                estimate: Final = await _estimate_with_fallback(estimator, evidence)
                evidence_item: Final = GitHubPullListItem.model_validate(
                    MappingProxyType(
                        {
                            "number": evidence["number"],
                            "title": evidence["title"],
                            "body": evidence["body"],
                            "head": MappingProxyType({"sha": evidence["head_sha"]}),
                            "user": MappingProxyType({"login": evidence["login"]}),
                            "merged_at": evidence["merged_at"],
                            "updated_at": evidence["merged_at"],
                        }
                    )
                )
                fetched_key: Final = cache_key(settings, context, repo, evidence_item)
                record: Final = self._report_record(evidence, estimate, fetched_key)
                await _cache_estimated_pull(repository, fetched_key, record)
                self._update_estimate_progress(estimate)
                return _ProcessedPull(index, record)

            async def worker(offset: int) -> tuple[_ProcessedPull, ...]:
                return tuple(
                    [await process(indexed_queue[index]) for index in range(offset, len(indexed_queue), PR_CONCURRENCY)]
                )

            workers: Final = tuple(asyncio.create_task(worker(offset)) for offset in range(PR_CONCURRENCY))
            try:
                groups: Final = await asyncio.gather(*workers)
                processed: Final = tuple(chain.from_iterable(groups))
            finally:
                for worker_task in workers:
                    if not worker_task.done():
                        worker_task.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
            processed_by_index: Final = _processed_records(processed)
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
                pulls=tuple(processed_by_index[index] for index in range(len(queue))),
                settings_fingerprint=settings_fingerprint(settings),
                warnings=repositories.warnings,
                unavailable_repos=repositories.unavailable_repos,
            )
            await github.close()
            report_json: Final[Mapping[str, object]] = _JSON_OBJECT_ADAPTER.validate_python(
                _REPORT_ADAPTER.dump_python(report, mode="json")
            )
            monitor.cancel()
            with suppress(asyncio.CancelledError):
                await monitor
            completed_status: Final = self.status.model_copy(
                update=MappingProxyType(
                    {
                        "running": False,
                        "phase": "complete",
                        "stage": repositories.stage,
                        "finished_at": self._clock().isoformat(),
                    }
                )
            )
            if coordinator is not None:
                if not await coordinator.finish(owner, completed_status, report):
                    raise SourceError(
                        "This sync was cancelled or replaced. Run analysis again to resume saved estimates."
                    )
            else:
                await repository.set_param("roi_calculator_report", report_json)
            self._status = completed_status
        except asyncio.CancelledError:
            self._update_status(phase="cancelled", stage="Sync cancelled")
            raise
        except SourceError as exc:
            self._update_status(phase="error", stage="Sync failed", error=str(exc))
        except Exception:  # noqa: BLE001 - background job boundary records a safe failure for every source error
            self._update_status(
                phase="error",
                stage="Sync failed",
                error=(
                    "Unexpected source response. No partial report was saved. "
                    "Check service compatibility and try again."
                ),
            )
        finally:
            monitor.cancel()
            with suppress(asyncio.CancelledError):
                await monitor
            try:
                if self._status.phase != "complete":
                    await github.close()
            finally:
                self._status = self._status.model_copy(
                    update=MappingProxyType({"running": False, "finished_at": self._clock().isoformat()})
                )
                if coordinator is not None and self._status.phase != "complete":
                    await coordinator.finish(owner, self.status)

    def _update_status(
        self,
        **update: Unpack[_StatusUpdate],  # kwargs-ok: Unpack preserves the typed status update contract
    ) -> None:
        status: Final = ROISyncStatus.model_validate(MappingProxyType({**self._status.model_dump(), **update}))
        self._status = status

    async def _previous_report(self, repository: _ReportRepository) -> ROIReport | None:
        parameter: Final = await repository.get_param("roi_calculator_report")
        if parameter is None:
            return None
        try:
            return _REPORT_ADAPTER.validate_python(parameter.param_value)
        except ValueError:
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
            commit_emails=pull.get("commit_emails", ()),
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
            commit_emails=evidence.get("commit_emails", ()),
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
