import asyncio
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from types import MappingProxyType
from typing import Final, TypeVar
from urllib.parse import quote

import httpx
from pydantic import BaseModel, TypeAdapter

from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared client factory has untyped params
)
from litellm.proxy.roi_calculator.analytics import normalize_email
from litellm.proxy.roi_calculator.github import GitHubPullListItem, SourceError
from litellm.proxy.roi_calculator.source import repository_tag
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.roi_calculator import ROIPullCommit, ROIPullEvidence, ROIPullFile, ROISettings
from litellm.types.roi_observed import ObservedIssue

_T: Final = TypeVar("_T", bound=BaseModel)


class _User(LiteLLMBaseModel):
    username: str
    public_email: str | None = None
    bot: bool = False


class _Project(LiteLLMBaseModel):
    id: int
    path_with_namespace: str
    visibility: str = "private"
    archived: bool = False
    issues_enabled: bool = True
    issues_access_level: str = "enabled"


class _MergeRequest(LiteLLMBaseModel):
    iid: int
    title: str
    description: str | None = None
    web_url: str
    author: _User
    merged_at: str | None
    updated_at: str
    created_at: datetime | None = None
    sha: str | None = None
    source_branch: str
    source_project_id: int | None
    changes_count: str | None = None

    def pull(self, source: _Project | None) -> GitHubPullListItem:
        return GitHubPullListItem.model_validate(
            {
                "number": self.iid,
                "title": self.title,
                "body": self.description or "",
                "html_url": self.web_url,
                "user": {"login": self.author.username, "type": "Bot" if self.author.bot else "User"},
                "created_at": self.created_at,
                "merged_at": self.merged_at,
                "updated_at": self.updated_at,
                "head": {
                    "sha": self.sha or "",
                    "ref": self.source_branch,
                    "repo": {"full_name": source.path_with_namespace} if source else None,
                },
            }
        )


class _Diff(LiteLLMBaseModel):
    new_path: str
    old_path: str
    diff: str = ""
    new_file: bool = False
    deleted_file: bool = False
    renamed_file: bool = False
    collapsed: bool = False
    too_large: bool = False

    def file(self) -> ROIPullFile:
        return ROIPullFile(
            filename=self.new_path,
            status="added"
            if self.new_file
            else "removed"
            if self.deleted_file
            else "renamed"
            if self.renamed_file
            else "modified",
            additions=sum(line.startswith("+") for line in self.diff.splitlines()),
            deletions=sum(line.startswith("-") for line in self.diff.splitlines()),
        )


class _Commit(LiteLLMBaseModel):
    id: str
    message: str


class _Issue(LiteLLMBaseModel):
    iid: int
    created_at: datetime
    labels: tuple[str, ...] = ()


class GitLab:
    def __init__(self, settings: ROISettings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings: Final = settings
        token: Final = settings.gitlab_token.get_secret_value()
        authorization: Final = (
            {"Authorization": f"Bearer {token}"} if settings.connection_type == "app" else {"PRIVATE-TOKEN": token}
        )
        self.headers: Final = {"Accept": "application/json", **(authorization if token else {})}
        self.client: Final = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.ROICalculator,
            params={"timeout": 45, "follow_redirects": False, "transport": transport},
        ).client
        self.close_client: Final = transport is not None
        self.profiles: Mapping[str, str] = MappingProxyType({})
        self.projects: Mapping[int, _Project] = MappingProxyType({})
        self.source_project_slots: Final = asyncio.Semaphore(8)

    async def close(self) -> None:
        if self.close_client:
            await self.client.aclose()

    async def _request(
        self, path: str, params: Mapping[str, str | int] | None = None, attempt: int = 0
    ) -> httpx.Response:
        try:
            response: Final = await self.client.get(
                self.settings.gitlab_api_url + "/" + path, params=params, headers=self.headers
            )
        except httpx.RequestError:
            raise SourceError("Could not reach GitLab. Check the API URL and network connection.") from None
        if response.status_code in (429, 502, 503, 504) and attempt < 2:
            await asyncio.sleep(0.5 * (attempt + 1))
            return await self._request(path, params, attempt + 1)
        if response.status_code != 200:
            raise SourceError(
                f"GitLab could not read this resource (HTTP {response.status_code}). "
                "Check the project, token read_api scope, and project membership."
            )
        return response

    async def _page(
        self, path: str, model: type[_T], params: Mapping[str, str | int], page: int
    ) -> tuple[tuple[_T, ...], bool]:
        response: Final = await self._request(path, {**params, "per_page": 100, "page": page})
        try:
            values: Final = TypeAdapter(tuple[object, ...]).validate_python(response.json())
            items: Final = tuple(model.model_validate(value) for value in values)
        except ValueError:
            raise SourceError("GitLab returned an invalid page of results.") from None
        has_more: Final = response.headers.get("x-next-page", "") != "" or 'rel="next"' in response.headers.get(
            "link", ""
        )
        return items, has_more

    async def _all(self, path: str, model: type[_T], params: Mapping[str, str | int] | None = None) -> tuple[_T, ...]:
        async def collect(page: int, previous: tuple[_T, ...]) -> tuple[_T, ...]:
            items, more = await self._page(path, model, params or {}, page)
            if not more:
                return previous + items
            if page >= 100:
                raise SourceError("GitLab's pagination limit was reached. Narrow the reporting window.")
            return await collect(page + 1, previous + items)

        return await collect(1, ())

    async def _project(self, project: str | int) -> _Project:
        if isinstance(project, int) and project in self.projects:
            return self.projects[project]
        response: Final = await self._request("projects/" + quote(str(project), safe=""))
        try:
            result: Final = _Project.model_validate(response.json())
        except ValueError:
            raise SourceError("GitLab returned invalid project details.") from None
        self.projects = MappingProxyType({**self.projects, result.id: result})
        return result

    async def repositories(self, query: str = "", page: int = 1) -> tuple[tuple[tuple[str, str, bool], ...], bool]:
        params: Final = {
            "simple": "true",
            "search": query,
            **(
                {"membership": "true"} if self.headers.get("PRIVATE-TOKEN") or self.headers.get("Authorization") else {}
            ),
        }
        items, more = await self._page("projects", _Project, params, page)
        return tuple((item.path_with_namespace, item.visibility, item.archived) for item in items), more

    async def test_repositories(self, repos: tuple[str, ...]) -> None:
        async def test(repo: str) -> None:
            project: Final = await self._project(repo)
            await self._request(f"projects/{project.id}/merge_requests", {"state": "merged", "per_page": 1})

        for repo in repos:
            await test(repo)

    async def pulls(self, repo: str, start: date, end: date) -> tuple[GitHubPullListItem, ...]:
        project: Final = await self._project(repo)
        items: Final = await self._all(
            f"projects/{project.id}/merge_requests",
            _MergeRequest,
            {
                "state": "merged",
                "scope": "all",
                "updated_after": start.isoformat() + "T00:00:00Z",
                "merged_after": start.isoformat() + "T00:00:00Z",
                "merged_before": (end + timedelta(days=1)).isoformat() + "T00:00:00Z",
                "order_by": "updated_at",
                "sort": "desc",
            },
        )
        merged: Final = tuple(
            item for item in items if item.merged_at and start.isoformat() <= item.merged_at[:10] <= end.isoformat()
        )
        source_ids: Final = tuple(frozenset(item.source_project_id for item in merged))
        projects: Final = await asyncio.gather(*(self._source_project(source_id) for source_id in source_ids))
        sources: Final = MappingProxyType(dict(zip(source_ids, projects, strict=True)))
        return tuple(item.pull(sources[item.source_project_id]) for item in merged)

    async def profile_email(self, login: str, *, fallback: str = "") -> str:
        if login.casefold() in self.profiles:
            return self.profiles[login.casefold()]
        try:
            users: Final = await self._all("users", _User, {"username": login})
        except SourceError:
            return fallback
        email: Final = next(
            (normalize_email(user.public_email) for user in users if user.username.casefold() == login.casefold()), ""
        )
        self.profiles = MappingProxyType({**self.profiles, login.casefold(): email})
        return email

    async def issues(self, repo: str, start: date, end: date) -> tuple[ObservedIssue, ...] | None:
        project: Final = await self._project(repo)
        if not project.issues_enabled or project.issues_access_level == "disabled":
            return None
        issues: Final = await self._all(
            f"projects/{project.id}/issues",
            _Issue,
            {
                "scope": "all",
                "state": "all",
                "created_after": f"{start}T00:00:00Z",
                "created_before": f"{end + timedelta(days=1)}T00:00:00Z",
            },
        )
        return tuple(
            ObservedIssue(repo=repo, number=issue.iid, created_at=issue.created_at, labels=issue.labels)
            for issue in issues
            if start <= issue.created_at.date() <= end
        )

    async def evidence(self, repo: str, pull: GitHubPullListItem) -> ROIPullEvidence:
        project: Final = await self._project(repo)
        path: Final = f"projects/{project.id}/merge_requests/{pull.number}"
        response: Final = await self._request(path)
        try:
            detail: Final = _MergeRequest.model_validate(response.json())
        except ValueError:
            raise SourceError("GitLab returned invalid merge request details.") from None
        diffs: Final = await self._all(path + "/diffs", _Diff)
        commits: Final = await self._all(path + "/commits", _Commit)
        profile: Final = await self.profile_email(detail.author.username)
        source: Final = await self._source_project(detail.source_project_id)
        files: Final = tuple(diff.file() for diff in diffs)
        return ROIPullEvidence(
            repo=repo,
            number=detail.iid,
            title=detail.title,
            body=detail.description or "",
            url=detail.web_url,
            login=detail.author.username,
            emails=(profile,) if profile else (),
            profile_email=profile,
            commit_emails=(),
            merged_at=detail.merged_at or "",
            head_sha=detail.sha or "",
            source_repo=repository_tag(self.settings, source.path_with_namespace) if source else "",
            source_branch=detail.source_branch,
            additions=sum(file["additions"] or 0 for file in files),
            deletions=sum(file["deletions"] or 0 for file in files),
            changed_files=len(files),
            files=files,
            commits=tuple(ROIPullCommit(sha=commit.id, message=commit.message) for commit in commits),
            commit_count=len(commits),
            incomplete_metadata=any(diff.collapsed or diff.too_large for diff in diffs)
            or detail.changes_count is None
            or not detail.changes_count.isdigit()
            or int(detail.changes_count) != len(files),
        )

    async def _source_project(self, project_id: int | None) -> _Project | None:
        if project_id is None:
            return None
        try:
            async with self.source_project_slots:
                return await self._project(project_id)
        except SourceError:
            return None
