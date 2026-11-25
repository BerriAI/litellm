import itertools
from dataclasses import dataclass
from typing import Final, Protocol

import httpx
from pydantic import BaseModel, ConfigDict

from sdk_watch.triage import Finding


class GitHub(Protocol):
    def count_prs(self, query: str) -> int: ...

    def branch_exists(self, branch: str) -> bool: ...

    def create_pr(self, title: str, head: str, base: str, body: str) -> str: ...


class _SearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_count: int


class _PullRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    html_url: str


@dataclass(frozen=True, slots=True)
class RestGitHub:
    client: httpx.Client
    repo: str

    def count_prs(self, query: str) -> int:
        response: Final = self.client.get(
            "/search/issues", params={"q": f"repo:{self.repo} is:pr {query}", "per_page": 1}
        )
        response.raise_for_status()
        return _SearchResult.model_validate_json(response.content).total_count

    def branch_exists(self, branch: str) -> bool:
        response: Final = self.client.get(f"/repos/{self.repo}/branches/{branch}")
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return True

    def create_pr(self, title: str, head: str, base: str, body: str) -> str:
        response: Final = self.client.post(
            f"/repos/{self.repo}/pulls", json={"title": title, "head": head, "base": base, "body": body}
        )
        response.raise_for_status()
        return _PullRequest.model_validate_json(response.content).html_url


def github_client(token: str | None) -> httpx.Client:
    auth: Final = {} if token is None else {"Authorization": f"Bearer {token}"}
    return httpx.Client(
        base_url="https://api.github.com",
        headers={"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", **auth},
        timeout=30,
    )


def already_handled(github: GitHub, finding: Finding) -> bool:
    if github.branch_exists(finding.branch):
        return True
    merged_since: Final = finding.candidate.release.released.isoformat()
    queries: Final = itertools.chain.from_iterable(
        _pr_queries(term, merged_since) for term in finding.change.search_terms
    )
    return any(github.count_prs(query) > 0 for query in queries)


def _pr_queries(term: str, merged_since: str) -> tuple[str, str]:
    return f'is:open "{term}"', f'is:merged merged:>={merged_since} "{term}"'
