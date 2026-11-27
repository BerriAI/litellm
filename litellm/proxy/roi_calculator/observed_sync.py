import asyncio
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from datetime import datetime, timezone
from itertools import chain
from types import MappingProxyType
from typing import Final, TypeAlias
from uuid import uuid4

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.proxy.roi_calculator.analytics import normalize_email
from litellm.proxy.roi_calculator.github import GitHub, GitHubPullListItem, SourceError
from litellm.proxy.roi_calculator.github_observed import GitHubObserved
from litellm.proxy.roi_calculator.gitlab import GitLab
from litellm.proxy.roi_calculator.observed_analytics import declared_requester, reporting_windows
from litellm.proxy.roi_calculator.source import repository_tag
from litellm.proxy.roi_calculator.sync import BranchSpendReader, GatewayUserReader, SpendReader
from litellm.proxy.roi_calculator.sync_store import SyncStore
from litellm.types.roi_calculator import ROISettings, ROISyncStatus
from litellm.types.roi_observed import ObservedData, ObservedIssue, ObservedPeriodData, ObservedPull, ObservedWindow


def _author(pull: GitHubPullListItem) -> str:
    return pull.user.login or "" if pull.user else ""


def _agent(pull: GitHubPullListItem) -> bool:
    login: Final = _author(pull)
    return (
        bool(pull.user and pull.user.type == "Bot")
        or login.endswith("[bot]")
        or bool(login.startswith(("project_", "group_")) and "_bot_" in login)
    )


def _owner(pull: GitHubPullListItem) -> str:
    login: Final = _author(pull)
    return declared_requester(login, pull.body or "") if _agent(pull) else login


def _public_email(pull: GitHubPullListItem) -> str:
    return normalize_email(pull.user.email) if pull.user and not _agent(pull) else ""


def _pull(settings: ROISettings, repo: str, pull: GitHubPullListItem, profiles: Mapping[str, str]) -> ObservedPull:
    if not pull.merged_at:
        raise SourceError("The source returned an unmerged change. No partial report was saved.")
    return ObservedPull(
        repo=repo,
        number=pull.number,
        title=pull.title,
        url=pull.html_url,
        author=_author(pull),
        agent=_agent(pull),
        requester=declared_requester(_author(pull), pull.body or "") if _agent(pull) else "",
        profile_email=profiles.get(_owner(pull).casefold(), "") or _public_email(pull),
        created_at=pull.created_at,
        merged_at=datetime.fromisoformat(pull.merged_at.replace("Z", "+00:00")),
        source_repo=repository_tag(settings, pull.head.repo.full_name) if pull.head and pull.head.repo else "",
        source_branch=pull.head.ref if pull.head else "",
    )


async def collect_observed(
    settings: ROISettings,
    spend_reader: SpendReader,
    gateway_user_reader: GatewayUserReader,
    branch_spend_reader: BranchSpendReader,
    now: datetime,
    progress: Callable[[str, int, int], None],
    transport: httpx.AsyncBaseTransport | None = None,
    days: int = 28,
) -> ObservedData:
    source: Final = GitLab(settings, transport) if settings.source_provider == "gitlab" else GitHub(settings, transport)
    activity: Final = (
        GitHubObserved(settings, source.client)
        if settings.source_provider == "github" and settings.github_token.get_secret_value()
        else source
    )
    windows: Final = reporting_windows(now.astimezone(timezone.utc), days)
    slots: Final = asyncio.Semaphore(4)
    total: Final = len(settings.repos) * 3

    async def profile(login: str) -> tuple[str, str]:
        async with slots:
            return login.casefold(), await source.profile_email(login)

    async def repository(
        repo: str, window: ObservedWindow
    ) -> tuple[tuple[ObservedPull, ...], tuple[ObservedIssue, ...] | None]:
        raw: Final = await activity.pulls(repo, window.start, window.end)
        unique: Final = {(repo, item.number): item for item in raw}
        if len(unique) != len(raw):
            raise SourceError("The source returned duplicate changes. Retry to get a complete report.")
        owners: Final = (
            frozenset(_owner(pull) for pull in raw if not _public_email(pull))
            - {""}
            - settings.identity_map.keys()
            - frozenset(settings.ignored_logins)
        )
        profiles: Final = MappingProxyType(dict(await asyncio.gather(*(profile(login) for login in owners))))
        pulls: Final = tuple(_pull(settings, repo, item, profiles) for item in raw)
        issues: Final = await activity.issues(repo, window.start, window.end)
        return pulls, issues

    async def period(window: ObservedWindow, offset: int) -> ObservedPeriodData:
        async def read(index: int, repo: str) -> tuple[tuple[ObservedPull, ...], tuple[ObservedIssue, ...] | None]:
            progress(f"Reading {repo} ({window.start} to {window.end})", offset + index, total)
            return await repository(repo, window)

        results: Final = tuple([await read(index, repo) for index, repo in enumerate(settings.repos)])
        pulls: Final = tuple(chain.from_iterable(result[0] for result in results))
        issues: Final = (
            None
            if results and all(result[1] is None for result in results)
            else tuple(chain.from_iterable(result[1] or () for result in results))
        )
        branches: Final = tuple(
            sorted(
                frozenset(
                    (
                        *(repository_tag(settings, repo) for repo in settings.repos),
                        *(pull.source_repo for pull in pulls),
                    )
                )
                - {""}
            )
        )
        return ObservedPeriodData(
            window=window,
            pulls=tuple(sorted(pulls, key=lambda pull: (pull.merged_at, pull.repo, pull.number), reverse=True)),
            issues=issues,
            spend=await spend_reader(window.start, window.end),
            branch_spend=await branch_spend_reader(window.start, window.end, branches),
        )

    try:
        gateway_emails: Final = await gateway_user_reader()
        current: Final = await period(windows[0], 0)
        previous: Final = await period(windows[1], len(settings.repos))
        last_year: Final = await period(windows[2], len(settings.repos) * 2)
        progress("Saving report", total, total)
        return ObservedData(
            source_provider=settings.source_provider,
            source_api_url=settings.source_api_url,
            repos=settings.repos,
            captured_at=now,
            gateway_emails=tuple(sorted(gateway_emails)),
            current=current,
            previous=previous,
            last_year=last_year,
        )
    finally:
        await source.close()


Progress: TypeAlias = Callable[[str, int, int], None]
BuildReport: TypeAlias = Callable[[Progress], Awaitable[ObservedData]]


class ObservedSyncManager:
    def __init__(self) -> None:
        self.status: ROISyncStatus = ROISyncStatus(
            running=False,
            phase="idle",
            stage="Not synced",
            done=0,
            total=0,
            estimated=0,
            reused=0,
            needs_attention=0,
            error=None,
        )
        self._task: asyncio.Task[None] | None = None
        self._lock: Final = asyncio.Lock()

    def _progress(self, stage: str, done: int, total: int) -> None:
        self.status = self.status.model_copy(
            update={"phase": "repositories", "stage": stage, "done": done, "total": total}
        )

    async def start(self, build: BuildReport, store: SyncStore, scheduled_interval: float = 0) -> bool:
        async with self._lock:
            if self._task is not None and not self._task.done():
                return False
            status: Final = ROISyncStatus(
                running=True,
                phase="repositories",
                stage="Reading repository activity",
                done=0,
                total=0,
                estimated=0,
                reused=0,
                needs_attention=0,
                error=None,
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            owner: Final = str(uuid4())
            if not await store.acquire(owner, status, scheduled_interval):
                return False
            self.status = status
            self._task = asyncio.create_task(self._run(build, store, owner))
            return True

    async def cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task

    async def _heartbeat(self, store: SyncStore, owner: str, task: asyncio.Task[object] | None) -> None:
        if task is None:
            return
        try:
            while True:
                await asyncio.sleep(5)
                if not await store.heartbeat(owner, self.status):
                    task.cancel()
                    return
        except Exception:  # noqa: BLE001  # loss of the database lease must stop publication
            task.cancel()

    async def _run(self, build: BuildReport, store: SyncStore, owner: str) -> None:
        monitor: Final = asyncio.create_task(self._heartbeat(store, owner, asyncio.current_task()))
        try:
            report: Final = await build(self._progress)
            monitor.cancel()
            with suppress(asyncio.CancelledError):
                await monitor
            complete: Final = self.status.model_copy(
                update={
                    "running": False,
                    "phase": "complete",
                    "stage": "Up to date",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            if not await store.finish(owner, complete, report):
                raise SourceError("The sync was cancelled or replaced. The previous report was kept.")
            self.status = complete
        except asyncio.CancelledError:
            self.status = self.status.model_copy(update={"phase": "cancelled", "stage": "Sync cancelled"})
            raise
        except SourceError as exc:
            self.status = self.status.model_copy(update={"phase": "error", "stage": "Sync failed", "error": str(exc)})
        except Exception:  # noqa: BLE001  # background tasks must persist a safe error without exposing credentials
            verbose_proxy_logger.exception("Observed ROI sync failed")
            self.status = self.status.model_copy(
                update={
                    "phase": "error",
                    "stage": "Sync failed",
                    "error": "Could not finish syncing. The previous report was kept. Retry after checking the connection.",
                }
            )
        finally:
            monitor.cancel()
            with suppress(asyncio.CancelledError):
                await monitor
            self.status = self.status.model_copy(
                update={"running": False, "finished_at": datetime.now(timezone.utc).isoformat()}
            )
            if self.status.phase != "complete":
                await store.finish(owner, self.status)
