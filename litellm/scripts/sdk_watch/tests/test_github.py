from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

import httpx
import pytest
from sdk_watch.changelog import Release
from sdk_watch.github import RestGitHub, already_handled
from sdk_watch.sources import Candidate, Sdk
from sdk_watch.triage import ChangeKind, Finding, NeededChange

FINDING = Finding(
    candidate=Candidate(
        sdk=Sdk(name="anthropic", repo="anthropics/anthropic-sdk-python"),
        release=Release(version="0.110.0", released=date(2026, 11, 20), entries=()),
    ),
    change=NeededChange(
        kind=ChangeKind.NEW_REQUEST_PARAM,
        slug="billing_policy",
        sdk_change="**api:** add billing_policy",
        litellm_change="Pass billing_policy through",
        search_terms=("billing_policy", "BillingPolicy"),
    ),
)


PR = "https://github.com/BerriAI/litellm/pull/2"


@dataclass(frozen=True)
class FakeGitHub:
    prs: Mapping[str, str] = field(default_factory=dict)
    branches: frozenset[str] = frozenset()

    def first_pr(self, query: str) -> str | None:
        return self.prs.get(query)

    def branch_exists(self, branch: str) -> bool:
        return branch in self.branches

    def create_pr(self, title: str, head: str, base: str, body: str) -> str:
        raise AssertionError("dedupe never opens a PR")


@pytest.mark.parametrize(
    ("github", "handled"),
    [
        (FakeGitHub(), None),
        (FakeGitHub(prs={'is:open in:title "BillingPolicy"': PR}), PR),
        (FakeGitHub(prs={'is:merged merged:>=2026-11-20 in:title "billing_policy"': PR}), PR),
        (FakeGitHub(branches=frozenset({FINDING.branch})), f"branch {FINDING.branch}"),
    ],
    ids=["nothing-yet", "open-pr-title-names-it", "merged-since-the-release", "agent-branch-exists"],
)
def test_a_change_is_handled_once_an_open_pr_a_recent_merge_or_the_agent_branch_covers_it(
    github: FakeGitHub, handled: str | None
) -> None:
    assert already_handled(github, FINDING) == handled


def github_answering(responses: tuple[httpx.Response, ...], waits: list[float]) -> RestGitHub:
    replies = iter(responses)
    client = httpx.Client(base_url="https://api.github.com", transport=httpx.MockTransport(lambda _: next(replies)))
    return RestGitHub(client=client, repo="BerriAI/litellm", sleep=waits.append, clock=lambda: 1000.0)


@pytest.mark.parametrize(
    ("limit_headers", "wait"),
    [
        ({"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1030"}, 31.0),
        ({"retry-after": "20"}, 20.0),
    ],
    ids=["search-quota-used-up", "secondary-limit"],
)
def test_a_rate_limited_lookup_waits_for_github_and_retries(limit_headers: dict[str, str], wait: float) -> None:
    waits: list[float] = []
    limited = httpx.Response(403, headers=limit_headers)
    github = github_answering((limited, httpx.Response(200, json={"items": [{"html_url": PR}]})), waits)

    assert github.first_pr('is:open in:title "billing_policy"') == PR
    assert waits == [wait]


def test_a_search_with_no_matching_pr_finds_none() -> None:
    github = github_answering((httpx.Response(200, json={"total_count": 0, "items": []}),), [])

    assert github.first_pr('is:open in:title "billing_policy"') is None


def test_a_limit_that_resets_much_later_fails_at_once_instead_of_stalling_the_run() -> None:
    waits: list[float] = []
    limited = httpx.Response(403, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "4600"})
    github = github_answering((limited,) * 4, waits)

    with pytest.raises(httpx.HTTPStatusError):
        github.branch_exists("litellm_sdk_watch_anthropic_0_110_0_billing_policy")
    assert waits == []
