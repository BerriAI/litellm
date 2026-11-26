import itertools
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol

import httpx
from pydantic import BaseModel, ConfigDict

from sdk_watch.triage import Finding


class GitHub(Protocol):
    def first_pr(self, query: str) -> str | None: ...

    def branch_exists(self, branch: str) -> bool: ...

    def create_pr(self, title: str, head: str, base: str, body: str) -> str: ...


class _PullRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    html_url: str


class _SearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: tuple[_PullRequest, ...]


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

    def first_pr(self, query: str) -> str | None:
        response: Final = self._get("/search/issues", {"q": f"repo:{self.repo} is:pr {query}", "per_page": 1})
        response.raise_for_status()
        items: Final = _SearchResult.model_validate_json(response.content).items
        return items[0].html_url if items else None

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

    def _get(self, path: str, params: Mapping[str, str | int] | None = None, retries: int = 0) -> httpx.Response:
        response: Final = self.client.get(path, params=params)
        wait: Final = rate_limit_wait(response, self.clock())
        if wait is None or retries == RATE_LIMIT_RETRIES:
            return response
        self.sleep(wait)
        return self._get(path, params, retries + 1)


def github_client(token: str | None) -> httpx.Client:
    auth: Final = {} if token is None else {"Authorization": f"Bearer {token}"}
    return httpx.Client(
        base_url="https://api.github.com",
        headers={"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", **auth},
        timeout=30,
    )


def already_handled(github: GitHub, finding: Finding) -> str | None:
    if github.branch_exists(finding.branch):
        return f"branch {finding.branch}"
    merged_since: Final = finding.candidate.release.released.isoformat()
    queries: Final = itertools.chain.from_iterable(
        _pr_queries(term, merged_since) for term in finding.change.search_terms
    )
    return next(filter(None, map(github.first_pr, queries)), None)


def _pr_queries(term: str, merged_since: str) -> tuple[str, str]:
    return f'is:open in:title "{term}"', f'is:merged merged:>={merged_since} in:title "{term}"'
