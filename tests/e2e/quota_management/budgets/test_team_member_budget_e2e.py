"""Live e2e: a team member's per-team budget attributes spend and enforces a cap.

The team carries a large budget while the one enrolled member is capped at a tiny
per-team budget, so any block is member-level, not team-level. Two scenarios share
that single member:
- attribution: the member's calls land in the spend logs tagged with both the team_id
  and the member's user_id, so per-member spend can be billed back
- enforcement: once the member's spend passes the per-team budget, calls are blocked
  with budget_exceeded while the team's own budget is nowhere near exhausted

Per-member budgets enforce off batch-written spend (~60s), so a quick burst all goes
through; the block only lands once that spend flushes.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from budget_client import BudgetClient, is_budget_block
from e2e_config import CHEAP_ANTHROPIC_MODEL, unique_marker
from e2e_http import Success, require_successful_call
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage

pytestmark = pytest.mark.e2e

MODEL = CHEAP_ANTHROPIC_MODEL
TEAM_BUDGET = 100.0
MEMBER_BUDGET = 3e-6
BURST = 6


@dataclass(frozen=True, slots=True)
class _Member:
    team_id: str
    user_id: str
    key: str


@pytest.fixture(scope="class")
def member(client: BudgetClient) -> Iterator[_Member]:
    """A team with a large budget plus one member capped at a tiny per-team budget,
    and that member's key. Shared across the class; torn down when it finishes.
    Cleanups register progressively and run LIFO best-effort through ResourceManager,
    so a partial-setup failure still releases what came before and one failed delete
    never strands the rest on the shared proxy."""
    resources = ResourceManager(client=client.gateway)
    try:
        marker = unique_marker()
        team_id = client.create_team(alias=f"e2e-team-member-{marker}", max_budget=TEAM_BUDGET)
        resources.defer(lambda: client.delete_team(team_id))
        user_id = client.create_user(max_budget=TEAM_BUDGET)
        resources.defer(lambda: client.delete_user(user_id))
        client.add_team_member(team_id, user_id, max_budget_in_team=MEMBER_BUDGET)
        key = client.generate_key(team_id=team_id, user_id=user_id)
        resources.defer(lambda: client.delete_key(key))
        yield _Member(team_id=team_id, user_id=user_id, key=key)
    finally:
        resources.teardown()


def _send(client: BudgetClient, key: str) -> str | None:
    """One member call; its response id (== the spend-log request_id) if it went
    through, else None."""
    match client.gateway.chat(
        key,
        ChatBody(
            model=MODEL,
            messages=[ChatMessage(role="user", content=f"hi {unique_marker()}")],
            max_tokens=16,
        ),
    ):
        case Success(data=response):
            return response.id
        case _:
            return None


class TestTeamMemberBudget:
    def test_member_spend_attributed_to_team_and_user(self, client: BudgetClient, member: _Member) -> None:
        sent = frozenset(rid for rid in (_send(client, member.key) for _ in range(BURST)) if rid)
        assert sent, "no member call went through; cannot check attribution"

        rows = client.gateway.poll_logs_for_key(
            member.key, predicate=lambda rs: bool(sent & {r.request_id for r in rs})
        )
        logged = [row for row in rows if row.request_id in sent]
        assert logged, f"none of the member's {len(sent)} calls reached the spend logs"

        for row in logged:
            assert row.team_id == member.team_id, (
                f"call {row.request_id} logged under team {row.team_id}, not the member's team {member.team_id}"
            )
            assert row.user == member.user_id, (
                f"call {row.request_id} logged under user {row.user}, not member {member.user_id}"
            )

    @pytest.mark.covers("quota_management.budget.team_member.blocks_over_limit")
    def test_member_spend_over_budget_is_blocked(self, client: BudgetClient, member: _Member) -> None:
        for _ in range(40):
            result = client.chat(member.key, MODEL, f"spend {unique_marker()}", max_tokens=16)
            if is_budget_block(result):
                return
            require_successful_call(result)
            time.sleep(2)
        pytest.fail("per-member budget never enforced within the call budget")
