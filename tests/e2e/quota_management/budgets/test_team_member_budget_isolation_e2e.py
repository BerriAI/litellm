"""Live e2e: per-team-member budgets are enforced independently between members.

Two members share one team that has a large team budget. The tight member is capped
at a tiny per-team budget and spends past it; the roomy member has plenty of room.
Once the tight member is blocked with budget_exceeded, the roomy member still serves
on the same team, its calls land in the spend logs under its own user id, and the
tight member stays blocked. A shared or leaky member counter would either block the
roomy member too or let the tight member back through once its peer spent.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from budget_client import BudgetClient, is_budget_block
from e2e_config import unique_marker
from e2e_http import Success, require_successful_call
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage

pytestmark = pytest.mark.e2e

MODEL = "gpt-5.5"
TEAM_BUDGET = 100.0
TIGHT_MEMBER_BUDGET = 3e-6
ROOMY_MEMBER_BUDGET = 100.0
ROOMY_BURST = 3


@dataclass(frozen=True, slots=True)
class _Pair:
    team_id: str
    tight_user_id: str
    roomy_user_id: str
    tight_key: str
    roomy_key: str


@pytest.fixture(scope="class")
def pair(client: BudgetClient) -> Iterator[_Pair]:
    """One team with a large budget and two members on it: a tight member capped at
    a tiny per-team budget and a roomy member with headroom, each with their own key.
    Shared across the class and torn down LIFO best-effort when it finishes."""
    resources = ResourceManager(client=client.proxy)
    try:
        marker = unique_marker()
        team_id = client.create_team(alias=f"e2e-member-iso-{marker}", max_budget=TEAM_BUDGET)
        resources.defer(lambda: client.delete_team(team_id))
        tight_user = client.create_user(max_budget=TEAM_BUDGET)
        resources.defer(lambda: client.delete_user(tight_user))
        roomy_user = client.create_user(max_budget=TEAM_BUDGET)
        resources.defer(lambda: client.delete_user(roomy_user))
        client.add_team_member(team_id, tight_user, max_budget_in_team=TIGHT_MEMBER_BUDGET)
        client.add_team_member(team_id, roomy_user, max_budget_in_team=ROOMY_MEMBER_BUDGET)
        tight_key = client.generate_key(team_id=team_id, user_id=tight_user)
        resources.defer(lambda: client.delete_key(tight_key))
        roomy_key = client.generate_key(team_id=team_id, user_id=roomy_user)
        resources.defer(lambda: client.delete_key(roomy_key))
        yield _Pair(
            team_id=team_id,
            tight_user_id=tight_user,
            roomy_user_id=roomy_user,
            tight_key=tight_key,
            roomy_key=roomy_key,
        )
    finally:
        resources.teardown()


def _roomy_send(client: BudgetClient, key: str) -> str:
    """One roomy-member call that must go through; returns its request id."""
    match client.proxy.chat(
        key,
        ChatBody(
            model=MODEL,
            messages=[ChatMessage(role="user", content=f"roomy {unique_marker()}")],
            max_tokens=16,
        ),
    ):
        case Success(data=response):
            assert response.id is not None, "roomy member call returned no id"
            return response.id
        case other:
            pytest.fail(f"roomy member call failed while a peer was over budget: {other}")


class TestTeamMemberBudgetIsolation:
    @pytest.mark.covers("quota_management.budget.team_member.isolates_per_member")
    def test_blocked_member_does_not_block_peer(self, client: BudgetClient, pair: _Pair) -> None:
        blocked = False
        for _ in range(40):
            result = client.chat(pair.tight_key, MODEL, f"tight {unique_marker()}", max_tokens=16)
            if is_budget_block(result):
                blocked = True
                break
            require_successful_call(result)
            time.sleep(2)
        assert blocked, "tight member's per-team budget never enforced"

        sent = frozenset(_roomy_send(client, pair.roomy_key) for _ in range(ROOMY_BURST))

        assert is_budget_block(
            client.chat(pair.tight_key, MODEL, f"tight {unique_marker()}", max_tokens=16)
        ), "tight member stopped being blocked once the peer spent"

        rows = client.proxy.poll_logs_for_key(
            pair.roomy_key, predicate=lambda rs: bool(sent & {r.request_id for r in rs})
        )
        logged = [row for row in rows if row.request_id in sent]
        assert logged, "none of the roomy member's calls reached the spend logs"
        for row in logged:
            assert row.user == pair.roomy_user_id, (
                f"roomy call {row.request_id} logged under user {row.user}, not {pair.roomy_user_id}"
            )
            assert row.team_id == pair.team_id, (
                f"roomy call {row.request_id} logged under team {row.team_id}, not {pair.team_id}"
            )
