from dataclasses import dataclass
from datetime import date

import pytest
from sdk_watch.changelog import Release
from sdk_watch.github import already_handled
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


@dataclass(frozen=True)
class FakeGitHub:
    hits: frozenset[str] = frozenset()
    branches: frozenset[str] = frozenset()

    def count_prs(self, query: str) -> int:
        return int(query in self.hits)

    def branch_exists(self, branch: str) -> bool:
        return branch in self.branches

    def create_pr(self, title: str, head: str, base: str, body: str) -> str:
        raise AssertionError("dedupe never opens a PR")


@pytest.mark.parametrize(
    ("github", "handled"),
    [
        (FakeGitHub(), False),
        (FakeGitHub(hits=frozenset({'is:open "BillingPolicy"'})), True),
        (FakeGitHub(hits=frozenset({'is:merged merged:>=2026-11-20 "billing_policy"'})), True),
        (FakeGitHub(branches=frozenset({FINDING.branch})), True),
    ],
    ids=["nothing-yet", "open-pr-names-it", "merged-since-the-release", "agent-branch-exists"],
)
def test_a_change_is_handled_once_an_open_pr_a_recent_merge_or_the_agent_branch_covers_it(
    github: FakeGitHub, handled: bool
) -> None:
    assert already_handled(github, FINDING) is handled
