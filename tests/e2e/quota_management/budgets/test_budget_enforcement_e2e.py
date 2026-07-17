"""Live e2e: a tiny max_budget on an entity actually blocks requests.

Each entity is an E2ECase (lifecycle.E2ECase) driven by run_case: init() creates
the budgeted entity + a key, run() drives spend until a `budget_exceeded` block,
teardown() deletes everything init() created (always runs, even on failure/skip).
Covers the entities with no prior live coverage - internal user, end-user,
organization, team member. See BUDGET_TEST_COVERAGE_MATRIX.md.

A non-budget error fails hard (never a skip); if calls never get blocked, budget
enforcement is broken -> fail.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, List, Type

import pytest

from budget_client import BudgetClient, is_budget_block
from e2e_config import unique_marker
from e2e_http import StreamingResponse, require_successful_call
from lifecycle import run_case

pytestmark = pytest.mark.e2e

def _assert_budget_blocks(client: BudgetClient, key: str, *, user: str = "") -> StreamingResponse:
    """Send paid calls until the entity's budget blocks one; return the blocked
    response so callers can assert on its shape. Key/user/org/member block within
    a couple calls off real-time reservation counters; the end-user budget
    enforces off table spend that lands on the batch write, so it takes a few
    more. A non-budget error fails hard (never a skip)."""
    for _ in range(40):
        result = client.chat(
            key,
            "claude-haiku-4-5",
            f"spend {unique_marker()}",
            max_tokens=16,
            user=user or None,
        )
        if is_budget_block(result):
            return result
        require_successful_call(result)
        time.sleep(2)
    pytest.fail("budget never enforced within the call budget")


@dataclass
class _BudgetCase:
    """Base E2ECase: a key under some budgeted entity must get blocked.

    Subclasses set up the budgeted entity in init() and register every created id
    in `_undo` (run LIFO in teardown so a key is deleted before its team/org).
    """

    client: BudgetClient
    key: str = ""
    _undo: List[Callable[[], None]] = field(
        default_factory=list
    )  # mutable-ok: per-case teardown registry

    def init(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        _assert_budget_blocks(self.client, self.key)

    def teardown(self) -> None:
        for undo in reversed(self._undo):
            undo()


class KeyBudgetCase(_BudgetCase):
    """A bare key (no team_id / user_id) carrying its own max_budget, so only the
    key-level budget can be the thing that blocks. The refusal must be a 429
    budget_exceeded; any other error already fails via _assert_budget_blocks."""

    def init(self) -> None:
        self.key = self.client.generate_key(max_budget=3e-6)
        self._undo.append(lambda: self.client.delete_key(self.key))

    def run(self) -> None:
        blocked = _assert_budget_blocks(self.client, self.key)
        assert blocked.status_code == 429, (
            f"budget refusal must be 429, got {blocked.status_code}: {blocked.body[:200]}"
        )


class InternalUserBudgetCase(_BudgetCase):
    def init(self) -> None:
        user_id = self.client.create_user(max_budget=3e-6)
        self._undo.append(lambda: self.client.delete_user(user_id))
        # personal key (no team) -> the user budget governs
        self.key = self.client.generate_key(user_id=user_id)
        self._undo.append(lambda: self.client.delete_key(self.key))


class EndUserBudgetCase(_BudgetCase):
    def init(self) -> None:
        customer = f"e2e-budget-cust-{unique_marker()}"
        self.client.create_customer(customer, max_budget=3e-6)
        self._undo.append(lambda: self.client.delete_customers([customer]))
        self.key = self.client.generate_key(models=["claude-haiku-4-5"])
        self._undo.append(lambda: self.client.delete_key(self.key))
        self._customer = customer

    def run(self) -> None:
        _assert_budget_blocks(self.client, self.key, user=self._customer)


class OrganizationBudgetCase(_BudgetCase):
    def init(self) -> None:
        # Org carries the tiny budget; the team under it has none, so a block here
        # is org-level enforcement (the historically weak link).
        org_id = self.client.create_org(
            max_budget=3e-6, alias=f"e2e-budget-org-{unique_marker()}"
        )
        self._undo.append(lambda: self.client.delete_org(org_id))
        team_id = self.client.create_team(
            alias=f"e2e-budget-team-{unique_marker()}", organization_id=org_id
        )
        self._undo.append(lambda: self.client.delete_team(team_id))
        self.key = self.client.generate_key(team_id=team_id)
        self._undo.append(lambda: self.client.delete_key(self.key))


class TeamMemberBudgetCase(_BudgetCase):
    def init(self) -> None:
        # Member's per-team budget is tiny while the team has a large budget, so a
        # block proves member-level (not team-level) enforcement.
        team_id = self.client.create_team(
            alias=f"e2e-budget-team-{unique_marker()}", max_budget=100.0
        )
        self._undo.append(lambda: self.client.delete_team(team_id))
        user_id = self.client.create_user(max_budget=100.0)
        self._undo.append(lambda: self.client.delete_user(user_id))
        self.client.add_team_member(team_id, user_id, max_budget_in_team=3e-6)
        self.key = self.client.generate_key(team_id=team_id, user_id=user_id)
        self._undo.append(lambda: self.client.delete_key(self.key))


def _case_id(case_cls: Type[_BudgetCase]) -> str:
    return case_cls.__name__


@pytest.mark.parametrize(
    "case_cls",
    [
        pytest.param(
            KeyBudgetCase,
            marks=pytest.mark.covers("quota_management.budget.key.blocks_over_limit"),
        ),
        pytest.param(
            InternalUserBudgetCase,
            marks=pytest.mark.covers("quota_management.budget.internal_user.blocks_over_limit"),
        ),
        pytest.param(
            EndUserBudgetCase,
            marks=pytest.mark.covers("quota_management.budget.end_user.blocks_over_limit"),
        ),
        pytest.param(
            OrganizationBudgetCase,
            marks=pytest.mark.covers("quota_management.budget.organization.blocks_over_limit"),
        ),
        pytest.param(
            TeamMemberBudgetCase,
            marks=pytest.mark.covers("quota_management.budget.team_member.blocks_over_limit"),
        ),
    ],
    ids=_case_id,
)
def test_budget_enforcement(
    client: BudgetClient, case_cls: Type[_BudgetCase]
) -> None:
    run_case(case_cls(client))
