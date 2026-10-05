from datetime import date, datetime, time, timedelta, timezone
from typing import Final, Literal

import httpx
from pydantic import BaseModel, Field

from litellm.proxy.roi_calculator.github import GitHubIssueSettings, GitHubPullListItem, SourceError, request_github
from litellm.types.roi_calculator import ROISettings
from litellm.types.roi_observed import ObservedIssue


class _PageInfo(BaseModel):
    hasNextPage: bool = False
    endCursor: str | None = None


class _Author(BaseModel):
    login: str
    kind: str = Field(alias="__typename")
    email: str | None = None


class _Repository(BaseModel):
    nameWithOwner: str


class _Label(BaseModel):
    name: str


class _Labels(BaseModel):
    nodes: tuple[_Label, ...] = ()
    pageInfo: _PageInfo = Field(default_factory=_PageInfo)


class _Node(BaseModel):
    number: int
    url: str
    title: str
    createdAt: datetime
    updatedAt: str
    mergedAt: str | None = None
    author: _Author | None = None
    body: str = ""
    headRefName: str = ""
    headRefOid: str = ""
    headRepository: _Repository | None = None
    labels: _Labels = Field(default_factory=_Labels)

    def pull(self) -> GitHubPullListItem:
        return GitHubPullListItem.model_validate(
            {
                "number": self.number,
                "html_url": self.url,
                "title": self.title,
                "body": self.body,
                "created_at": self.createdAt,
                "merged_at": self.mergedAt,
                "updated_at": self.updatedAt,
                "user": {"login": self.author.login, "type": self.author.kind, "email": self.author.email}
                if self.author
                else None,
                "head": {
                    "ref": self.headRefName,
                    "sha": self.headRefOid,
                    "repo": {"full_name": self.headRepository.nameWithOwner} if self.headRepository else None,
                },
            }
        )


class _Search(BaseModel):
    issueCount: int
    pageInfo: _PageInfo
    nodes: tuple[_Node, ...]


class _Data(BaseModel):
    search: _Search


class _Response(BaseModel):
    data: _Data | None = None
    errors: tuple[object, ...] = ()


_QUERY: Final = """query($q:String!, $after:String) {
  search(query:$q, type:ISSUE, first:100, after:$after) {
    issueCount pageInfo { hasNextPage endCursor }
    nodes {
      ... on PullRequest {
        number url title body createdAt updatedAt mergedAt headRefName headRefOid
        author { __typename login ... on User { email } } headRepository { nameWithOwner }
      }
      ... on Issue {
        number url title createdAt updatedAt
        labels(first:100) { nodes { name } pageInfo { hasNextPage } }
      }
    }
  }
}"""


class GitHubObserved:
    def __init__(self, settings: ROISettings, client: httpx.AsyncClient) -> None:
        self._client: Final = client
        self._api_url: Final = settings.github_api_url
        self._url: Final = (
            "https://api.github.com/graphql"
            if settings.github_api_url == "https://api.github.com"
            else settings.github_api_url.removesuffix("/api/v3") + "/api/graphql"
        )
        self._headers: Final = {"Authorization": "Bearer " + settings.github_token.get_secret_value()}

    async def _page(self, query: str, cursor: str | None = None) -> _Search:
        response: Final = await request_github(
            self._client,
            "POST",
            self._url,
            headers=self._headers,
            json_body={"query": _QUERY, "variables": {"q": query, "after": cursor}},
            read_only=True,
        )
        try:
            result: Final = _Response.model_validate(response.json())
        except ValueError:
            raise SourceError("GitHub returned invalid activity data. Try syncing again.") from None
        if result.errors or result.data is None:
            raise SourceError("GitHub could not read all activity. Check app permissions and rate limits, then retry.")
        return result.data.search

    async def _range(
        self, repo: str, start: datetime, end: datetime, kind: Literal["pull", "issue"]
    ) -> tuple[_Node, ...]:
        qualifier: Final = "merged" if kind == "pull" else "created"
        source: Final = "is:pr is:merged" if kind == "pull" else "is:issue"
        lower: Final = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        upper: Final = (end - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        query: Final = f"repo:{repo} {source} {qualifier}:{lower}..{upper} sort:created-asc"
        first: Final = await self._page(query)
        if first.issueCount > 1000:
            seconds: Final = int((end - start).total_seconds())
            if seconds < 2:
                raise SourceError("GitHub has more than 1,000 results in one second. The report was not truncated.")
            middle: Final = start + timedelta(seconds=seconds // 2)
            left: Final = await self._range(repo, start, middle, kind)
            return left + await self._range(repo, middle, end, kind)

        async def remaining(page: _Search, seen: frozenset[str]) -> tuple[_Node, ...]:
            if not page.pageInfo.hasNextPage:
                return page.nodes
            cursor: Final = page.pageInfo.endCursor
            if not cursor or cursor in seen or len(seen) >= 10:
                raise SourceError("GitHub returned incomplete pagination. The previous report was kept.")
            following: Final = await self._page(query, cursor)
            return page.nodes + await remaining(following, seen | {cursor})

        nodes: Final = await remaining(first, frozenset())
        if len(nodes) != first.issueCount or len(frozenset(node.url for node in nodes)) != len(nodes):
            raise SourceError("GitHub activity changed during collection. Retry to get a complete report.")
        return nodes

    async def _read(self, repo: str, start: date, end: date, kind: Literal["pull", "issue"]) -> tuple[_Node, ...]:
        return await self._range(
            repo,
            datetime.combine(start, time.min, timezone.utc),
            datetime.combine(end + timedelta(days=1), time.min, timezone.utc),
            kind,
        )

    async def pulls(self, repo: str, start: date, end: date) -> tuple[GitHubPullListItem, ...]:
        nodes: Final = await self._read(repo, start, end, "pull")
        return tuple(node.pull() for node in nodes)

    async def issues(self, repo: str, start: date, end: date) -> tuple[ObservedIssue, ...] | None:
        response: Final = await request_github(
            self._client, "GET", f"{self._api_url}/repos/{repo}", headers=self._headers
        )
        try:
            settings: Final = GitHubIssueSettings.model_validate(response.json())
        except ValueError:
            raise SourceError("GitHub returned invalid repository settings.") from None
        if not settings.has_issues:
            return None
        nodes: Final = await self._read(repo, start, end, "issue")
        if any(node.labels.pageInfo.hasNextPage for node in nodes):
            raise SourceError("GitHub returned incomplete issue labels. The previous report was kept.")
        return tuple(
            ObservedIssue(
                repo=repo,
                number=node.number,
                created_at=node.createdAt,
                labels=tuple(label.name for label in node.labels.nodes),
            )
            for node in nodes
        )
