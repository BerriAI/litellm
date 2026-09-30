import asyncio
from collections.abc import AsyncIterator, Mapping
from datetime import date
from types import MappingProxyType
from typing import Final, TypeVar
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared client factory has untyped params
)
from litellm.proxy.roi_calculator.analytics import normalize_email
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.roi_calculator import ROIPullCommit, ROIPullEvidence, ROIPullFile, ROISettings

_T: Final = TypeVar("_T")


class SourceError(Exception):
    pass


class _GitHubModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _GitHubUser(_GitHubModel):
    login: str | None = None


class _GitHubHead(_GitHubModel):
    sha: str = ""


class GitHubPullListItem(_GitHubModel):
    number: int
    html_url: str = ""
    merged_at: str | None = None
    updated_at: str
    title: str
    body: str | None = None
    head: _GitHubHead | None = None
    user: _GitHubUser | None = None


class _RepositoryItem(_GitHubModel):
    full_name: str
    visibility: str | None = None
    private: bool = False
    archived: bool = False


def _repository_values(repositories: tuple[_RepositoryItem, ...]) -> tuple[tuple[str, str, bool], ...]:
    return tuple(
        (
            repository.full_name,
            repository.visibility or ("private" if repository.private else "public"),
            repository.archived,
        )
        for repository in repositories
    )


class _PullDetail(_GitHubModel):
    number: int
    title: str
    body: str | None = None
    html_url: str
    user: _GitHubUser | None = None
    merged_at: str
    head: _GitHubHead
    additions: int = 0
    deletions: int = 0
    changed_files: int | None = None
    commits: int | None = None


class _PullFile(_GitHubModel):
    filename: str | None = None
    status: str | None = None
    additions: int | None = None
    deletions: int | None = None

    def evidence(self) -> ROIPullFile:
        evidence: Final[ROIPullFile] = {
            "filename": self.filename,
            "status": self.status,
            "additions": self.additions,
            "deletions": self.deletions,
        }
        return evidence


class _RestAuthor(_GitHubModel):
    email: str = ""


class _RestCommitContent(_GitHubModel):
    message: str = ""
    author: _RestAuthor | None = None


class _RestCommit(_GitHubModel):
    sha: str = ""
    author: _GitHubUser | None = None
    commit: _RestCommitContent = Field(default_factory=_RestCommitContent)


class _GraphQLAuthor(_GitHubModel):
    email: str = ""
    user: _GitHubUser | None = None


class _GraphQLCommit(_GitHubModel):
    oid: str
    message: str
    additions: int
    deletions: int
    changedFilesIfAvailable: int | None = None
    author: _GraphQLAuthor | None = None


class _GraphQLNode(_GitHubModel):
    commit: _GraphQLCommit


def _rest_commit_evidence(commit: _RestCommit) -> ROIPullCommit:
    evidence: Final[ROIPullCommit] = {
        "sha": commit.sha,
        "message": commit.commit.message,
    }
    return evidence


def _graphql_commit_evidence(node: _GraphQLNode) -> ROIPullCommit:
    commit: Final = node.commit
    evidence: Final[ROIPullCommit] = {
        "sha": commit.oid,
        "message": commit.message,
        "additions": commit.additions,
        "deletions": commit.deletions,
        "changed_files": commit.changedFilesIfAvailable,
    }
    return evidence


class _GraphQLPageInfo(_GitHubModel):
    hasNextPage: bool
    endCursor: str | None = None


class _GraphQLConnection(_GitHubModel):
    totalCount: int
    pageInfo: _GraphQLPageInfo
    nodes: tuple[_GraphQLNode, ...]


class _GraphQLPullRequest(_GitHubModel):
    commits: _GraphQLConnection


class _GraphQLRepository(_GitHubModel):
    pullRequest: _GraphQLPullRequest | None = None


class _GraphQLData(_GitHubModel):
    repository: _GraphQLRepository | None = None


class _GraphQLError(_GitHubModel):
    message: str = ""


class _GraphQLResponse(_GitHubModel):
    data: _GraphQLData | None = None
    errors: tuple[_GraphQLError, ...] = ()


class _GraphQLVariables(TypedDict):
    owner: ReadOnly[str]
    name: ReadOnly[str]
    number: ReadOnly[int]
    cursor: ReadOnly[str | None]


class _GraphQLPayload(TypedDict):
    query: ReadOnly[str]
    variables: ReadOnly[_GraphQLVariables]


_REPOSITORIES: Final[TypeAdapter[tuple[_RepositoryItem, ...]]] = TypeAdapter(tuple[_RepositoryItem, ...])
_REPOSITORY_SEARCH_PAGES: Final[int] = 10
_REPOSITORY_PAGE_ERROR: Final[str] = "GitHub returned an unexpected repository list."
_PULLS: Final[TypeAdapter[tuple[GitHubPullListItem, ...]]] = TypeAdapter(tuple[GitHubPullListItem, ...])
_PULL_FILES: Final[TypeAdapter[tuple[_PullFile, ...]]] = TypeAdapter(tuple[_PullFile, ...])
_REST_COMMITS: Final[TypeAdapter[tuple[_RestCommit, ...]]] = TypeAdapter(tuple[_RestCommit, ...])
_GRAPHQL_RESPONSE: Final = TypeAdapter(_GraphQLResponse)
_GRAPHQL_QUERY: Final = """query($owner:String!, $name:String!, $number:Int!, $cursor:String) {
  repository(owner:$owner, name:$name) { pullRequest(number:$number) {
    commits(first:100, after:$cursor) {
      totalCount pageInfo { hasNextPage endCursor }
      nodes { commit { oid message additions deletions changedFilesIfAvailable
        author { email user { login } } } }
    }
  } }
}"""


async def _request(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    params: Mapping[str, str | int] | None = None,
    json_body: object | None = None,
    headers: Mapping[str, str] | None = None,
) -> httpx.Response:
    async def send(attempt: int) -> httpx.Response:
        try:
            response: Final = await client.request(
                method,
                path,
                params=params,
                json=json_body,
                headers=headers,
            )
        except httpx.RequestError:
            raise SourceError("Could not reach GitHub. Check the API URL and network connection.") from None
        if response.status_code in (429, 502, 503, 504) and method == "GET" and attempt < 2:
            await asyncio.sleep(0.5 * (attempt + 1))
            return await send(attempt + 1)
        if response.status_code >= 400:
            labels: Final[Mapping[int, str]] = MappingProxyType(
                {
                    401: "Authentication failed. Check the configured GitHub token.",
                    403: "GitHub denied access or reached a rate limit. Check token permissions and organization approval.",
                    404: "GitHub repository or organization not found. Check its name, token access, and API URL.",
                    429: "GitHub rate limit reached. Wait before syncing again.",
                }
            )
            raise SourceError(
                labels.get(
                    response.status_code,
                    "GitHub returned an error.",
                )
                + f" (HTTP {response.status_code})"
            )
        return response

    return await send(0)


async def _fetch_page(
    client: httpx.AsyncClient,
    path: str,
    adapter: TypeAdapter[tuple[_T, ...]],
    params: Mapping[str, str | int] | None,
    page: int,
    headers: Mapping[str, str] | None = None,
    error_message: str = "GitHub returned an unexpected pagination response.",
) -> tuple[tuple[_T, ...], bool]:
    response: Final = await _request(
        client,
        "GET",
        path,
        params=MappingProxyType(
            {
                **(params if params is not None else MappingProxyType({})),
                "per_page": 100,
                "page": page,
            }
        ),
        headers=headers,
    )
    try:
        parsed: Final[tuple[_T, ...]] = adapter.validate_python(response.json())
    except ValueError:
        raise SourceError(error_message) from None
    return parsed, 'rel="next"' in response.headers.get("link", "")


async def _pages(
    client: httpx.AsyncClient,
    path: str,
    adapter: TypeAdapter[tuple[_T, ...]],
    params: Mapping[str, str | int] | None = None,
    limit: int = 10000,
    headers: Mapping[str, str] | None = None,
) -> AsyncIterator[tuple[_T, ...]]:
    for page in range(1, limit + 1):
        result = await _fetch_page(client, path, adapter, params, page, headers)
        yield result[0]
        if not result[1]:
            return
    raise SourceError("GitHub's pagination limit was reached. Narrow the date range.")


async def _collect(items: AsyncIterator[_T]) -> tuple[_T, ...]:
    collected: Final = [item async for item in items]  # mutable-ok: async iterables require an intermediate buffer
    return tuple(collected)


class _GitHubUserProfile(_GitHubModel):
    email: str | None = None


class GitHub:
    def __init__(
        self,
        settings: ROISettings,
        transport: httpx.AsyncBaseTransport | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if client is not None and transport is not None:
            raise ValueError("Pass either an injected GitHub client or a transport.")
        self._profiles: Mapping[str, str | None] = MappingProxyType({})
        token: Final = settings.github_token.get_secret_value()
        self._headers: Final[Mapping[str, str]] = (
            MappingProxyType(
                {
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {token}",
                }
            )
            if token
            else MappingProxyType({"Accept": "application/vnd.github+json"})
        )
        self._api_url: Final = settings.github_api_url.rstrip("/")
        client_params: Final = TypeAdapter(dict[str, object]).validate_python(
            MappingProxyType({"timeout": 45, "follow_redirects": False, "transport": transport})
        )
        self.client: Final[httpx.AsyncClient] = (
            client
            if client is not None
            else get_async_httpx_client(
                llm_provider=httpxSpecialProvider.ROICalculator,
                params=client_params,
            ).client
        )
        self._close_client: Final = client is not None or transport is not None

    async def close(self) -> None:
        if self._close_client:
            await self.client.aclose()

    def _url(self, path: str) -> str:
        return f"{self._api_url}/{path.lstrip('/')}"

    async def repositories(
        self,
        query: str = "",
        page: int = 1,
    ) -> tuple[tuple[tuple[str, str, bool], ...], bool]:
        params: Final = MappingProxyType(
            {
                "sort": "updated",
                "direction": "desc",
                "affiliation": "owner,collaborator,organization_member",
            }
        )
        if not query:
            repositories, has_more = await _fetch_page(
                self.client,
                self._url("user/repos"),
                _REPOSITORIES,
                params,
                page,
                self._headers,
                error_message=_REPOSITORY_PAGE_ERROR,
            )
            return _repository_values(repositories), has_more

        normalized_query: Final = query.casefold()
        first_github_page: Final = (page - 1) * _REPOSITORY_SEARCH_PAGES + 1

        async def search_pages(
            github_page: int,
            pages_remaining: int,
        ) -> tuple[tuple[_RepositoryItem, ...], bool]:
            repositories, has_more = await _fetch_page(
                self.client,
                self._url("user/repos"),
                _REPOSITORIES,
                params,
                github_page,
                self._headers,
                error_message=_REPOSITORY_PAGE_ERROR,
            )
            matches: Final = tuple(
                repository for repository in repositories if normalized_query in repository.full_name.casefold()
            )
            if pages_remaining == 1 or not has_more:
                return matches, has_more
            later_matches, later_has_more = await search_pages(github_page + 1, pages_remaining - 1)
            return (*matches, *later_matches), later_has_more

        matches, search_has_more = await search_pages(first_github_page, _REPOSITORY_SEARCH_PAGES)
        return _repository_values(matches), search_has_more

    async def test_repositories(self, repos: tuple[str, ...]) -> None:
        for repo in repos:
            await _request(self.client, "GET", self._url(f"repos/{repo}"), headers=self._headers)
            await _request(
                self.client,
                "GET",
                self._url(f"repos/{repo}/pulls"),
                params=MappingProxyType({"per_page": 1, "state": "closed"}),
                headers=self._headers,
            )

    async def pulls(self, repo: str, start: date, end: date) -> tuple[GitHubPullListItem, ...]:
        async def pull_pages() -> AsyncIterator[GitHubPullListItem]:
            async for page in _pages(
                self.client,
                self._url(f"repos/{repo}/pulls"),
                _PULLS,
                MappingProxyType({"state": "closed", "sort": "updated", "direction": "desc"}),
                headers=self._headers,
            ):
                for pull in page:
                    yield pull
                if page and page[-1].updated_at[:10] < start.isoformat():
                    return

        async def matching_pulls() -> AsyncIterator[GitHubPullListItem]:
            async for pull in pull_pages():
                if pull.merged_at is not None and start.isoformat() <= pull.merged_at[:10] <= end.isoformat():
                    yield pull

        return await _collect(matching_pulls())

    async def evidence(self, repo: str, pull: GitHubPullListItem) -> ROIPullEvidence:
        detail_response: Final = await _request(
            self.client,
            "GET",
            self._url(f"repos/{repo}/pulls/{pull.number}"),
            headers=self._headers,
        )
        try:
            detail: Final = _PullDetail.model_validate(detail_response.json())
        except ValueError:
            raise SourceError("GitHub returned unexpected pull request details.") from None
        login: Final = detail.user.login if detail.user and detail.user.login else "deleted-user"

        async def file_pages() -> AsyncIterator[_PullFile]:
            async for page in _pages(
                self.client,
                self._url(f"repos/{repo}/pulls/{pull.number}/files"),
                _PULL_FILES,
                limit=30,
                headers=self._headers,
            ):
                for item in page:
                    yield item

        files: Final = tuple(item.evidence() for item in await _collect(file_pages()))
        profile_email: Final = await self.profile_email(login)
        commits, authors, commit_count = await self._commit_metadata(repo, pull.number, detail)
        commit_emails: Final = tuple(
            sorted(
                frozenset(normalize_email(author[1]) for author in authors if author[0].casefold() == login.casefold())
            )
        )
        email_candidates: Final = frozenset(
            address
            for address in (
                profile_email,
                *commit_emails,
            )
            if address
        )
        changed_files: Final = detail.changed_files if detail.changed_files is not None else len(files)
        evidence: Final[ROIPullEvidence] = {
            "repo": repo,
            "number": detail.number,
            "title": detail.title,
            "body": detail.body or "",
            "url": detail.html_url,
            "login": login,
            "emails": tuple(sorted(email_candidates)),
            "profile_email": profile_email,
            "commit_emails": commit_emails,
            "merged_at": detail.merged_at,
            "head_sha": detail.head.sha,
            "additions": detail.additions,
            "deletions": detail.deletions,
            "changed_files": changed_files,
            "files": files,
            "commits": commits,
            "commit_count": commit_count,
            "incomplete_metadata": len(files) != changed_files or len(commits) != commit_count,
        }
        return evidence

    async def profile_email(self, login: str, *, fallback: str = "") -> str:
        if login.casefold() in self._profiles:
            cached: Final = self._profiles[login.casefold()]
            return cached if cached is not None else fallback
        address: Final = await self._load_profile_email(login)
        self._profiles = MappingProxyType({**self._profiles, login.casefold(): address})
        return address if address is not None else fallback

    async def _load_profile_email(self, login: str) -> str | None:
        try:
            response: Final = await self.client.get(
                self._url(f"users/{quote(login, safe='')}"),
                headers=self._headers,
            )
            if response.status_code != 200:
                return None
            profile: Final = _GitHubUserProfile.model_validate(response.json())
            return normalize_email(profile.email)
        except (httpx.HTTPError, ValueError):
            return None

    async def _commit_metadata(
        self, repo: str, number: int, detail: _PullDetail
    ) -> tuple[tuple[ROIPullCommit, ...], tuple[tuple[str, str], ...], int]:
        if not self._headers.get("Authorization"):

            async def commit_pages() -> AsyncIterator[_RestCommit]:
                async for page in _pages(
                    self.client,
                    self._url(f"repos/{repo}/pulls/{number}/commits"),
                    _REST_COMMITS,
                    limit=3,
                    headers=self._headers,
                ):
                    for item in page:
                        yield item

            rest_commits: Final = await _collect(commit_pages())
            commits: Final[tuple[ROIPullCommit, ...]] = tuple(_rest_commit_evidence(item) for item in rest_commits)
            authors: Final = tuple(
                (
                    item.author.login if item.author and item.author.login else "",
                    item.commit.author.email if item.commit.author else "",
                )
                for item in rest_commits
            )
            count: Final = detail.commits if detail.commits is not None else len(commits)
            return commits, authors, count
        base: Final = self._api_url
        endpoint: Final = (
            base.removesuffix("/api/v3") + "/api/graphql" if base.endswith("/api/v3") else base + "/graphql"
        )
        owner, name = repo.split("/", maxsplit=1)
        return await self._graphql_commits(repo, number, endpoint, owner, name, None, 100)

    async def _graphql_commits(
        self,
        repo: str,
        number: int,
        endpoint: str,
        owner: str,
        name: str,
        cursor: str | None,
        remaining_pages: int,
        accumulated_commits: tuple[ROIPullCommit, ...] = (),
        accumulated_authors: tuple[tuple[str, str], ...] = (),
    ) -> tuple[tuple[ROIPullCommit, ...], tuple[tuple[str, str], ...], int]:
        if remaining_pages == 0:
            raise SourceError("GitHub commit pagination limit was reached.")
        response: Final = await _request(
            self.client,
            "POST",
            endpoint,
            headers=self._headers,
            json_body=_GraphQLPayload(
                query=_GRAPHQL_QUERY,
                variables=_GraphQLVariables(owner=owner, name=name, number=number, cursor=cursor),
            ),
        )
        try:
            parsed: Final = _GRAPHQL_RESPONSE.validate_python(response.json())
            if parsed.errors or parsed.data is None or parsed.data.repository is None:
                raise SourceError(
                    "GitHub could not read commit metadata. Check repository permissions and API compatibility."
                )
            pull_request: Final = parsed.data.repository.pullRequest
            if pull_request is None:
                raise SourceError(
                    "GitHub could not read commit metadata. Check repository permissions and API compatibility."
                )
            connection: Final = pull_request.commits
        except SourceError:
            raise
        except ValueError:
            raise SourceError("GitHub returned unexpected commit metadata.") from None
        new_commits: Final[tuple[ROIPullCommit, ...]] = tuple(
            _graphql_commit_evidence(node) for node in connection.nodes
        )
        new_authors: Final = tuple(
            (
                author.user.login if author and author.user and author.user.login else "",
                author.email if author else "",
            )
            for author in (node.commit.author for node in connection.nodes)
        )
        commits: Final = accumulated_commits + new_commits
        authors: Final = accumulated_authors + new_authors
        if not connection.pageInfo.hasNextPage:
            return commits, authors, connection.totalCount
        return await self._graphql_commits(
            repo,
            number,
            endpoint,
            owner,
            name,
            connection.pageInfo.endCursor,
            remaining_pages - 1,
            commits,
            authors,
        )
