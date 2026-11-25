import itertools
import time
from collections.abc import Callable, Mapping
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


MAX_RATE_LIMIT_WAIT_SECONDS: Final = 90.0
RATE_LIMIT_RETRIES: Final = 3


def rate_limit_wait(response: httpx.Response, now: float) -> float | None:
    if response.status_code not in (403, 429):
        return None
    if "retry-after" in response.headers:
        return _bounded(float(response.headers["retry-after"]))
    if response.headers.get("x-ratelimit-remaining") != "0":
        return None
    return _bounded(float(response.headers["x-ratelimit-reset"]) - now + 1)


def _bounded(wait: float) -> float | None:
    return max(wait, 0.0) if wait <= MAX_RATE_LIMIT_WAIT_SECONDS else None


@dataclass(frozen=True, slots=True)
class RestGitHub:
    client: httpx.Client
    repo: str
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.time

    def count_prs(self, query: str) -> int:
        response: Final = self._get("/search/issues", {"q": f"repo:{self.repo} is:pr {query}", "per_page": 1})
        response.raise_for_status()
        return _SearchResult.model_validate_json(response.content).total_count

    def branch_exists(self, branch: str) -> bool:
        response: Final = self._get(f"/repos/{self.repo}/branches/{branch}")
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

    def _get(self, path: str, params: Mapping[str, str | int] | None = None) -> httpx.Response:
        for _ in range(RATE_LIMIT_RETRIES):
            response = self.client.get(path, params=params)
            wait = rate_limit_wait(response, self.clock())
            if wait is None:
                return response
            self.sleep(wait)
        return self.client.get(path, params=params)


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
