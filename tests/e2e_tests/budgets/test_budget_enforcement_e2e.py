"""Live e2e: a tiny max_budget on an entity actually blocks requests.

Mirrors tests/otel_tests/test_e2e_budgeting.py (call until `budget_exceeded`) but
covers the entities that had NO live coverage - internal user, end-user,
organization, team member - and runs in this suite so the shared `resources`
teardown deletes every entity created. See BUDGET_TEST_COVERAGE_MATRIX.md.

Skip on environment, fail on behavior: a non-budget error (provider down) skips;
if calls never get blocked, budget enforcement is broken -> fail.
"""

import time

import pytest

from budget_client import BudgetClient, is_budget_block
from lifecycle import ResourceManager
from proxy_client import require_successful_call, unique_marker

pytestmark = pytest.mark.e2e

TINY_BUDGET = 1e-6  # one real call's spend exceeds this, so the block lands fast


def _assert_budget_blocks(
    client: BudgetClient, key: str, model: str, *, user: str = ""
) -> None:
    """Drive spend over the entity's budget and assert a call is blocked.

    Two-phase so it's robust across entities: key/user/org enforce off real-time
    reservation counters (block within a couple calls), but end-user enforcement
    reads the table spend that only updates on the ~60s batch write - so after a
    warmup we poll for the block across that window. Skip on a non-budget error;
    fail if the budget is never enforced.
    """
    extra: dict = {"max_tokens": 16}
    if user:
        extra["user"] = user

    def _call(label: str):
        result = client.chat(key, model, f"{label} {unique_marker()}", extra_body=extra)
        if not result.ok and not is_budget_block(result):
            require_successful_call(result)  # non-budget error -> skip
        return result

    for _ in range(2):  # warmup: incur spend (fast entities block here)
        if is_budget_block(_call("warmup")):
            return
        time.sleep(0.5)

    deadline = time.monotonic() + 100  # cover the batch-write propagation window
    while time.monotonic() < deadline:
        if is_budget_block(_call("probe")):
            return
        time.sleep(5)
    pytest.fail("budget not enforced within 100s")


def test_key_budget_blocks(client: BudgetClient, resources: ResourceManager) -> None:
    key = client.generate_key(max_budget=TINY_BUDGET)
    resources.defer(lambda: client.delete_key(key))
    _assert_budget_blocks(client, key, "gpt-5.5")


def test_internal_user_budget_blocks(
    client: BudgetClient, resources: ResourceManager
) -> None:
    user_id = client.create_user(max_budget=TINY_BUDGET)
    resources.defer(lambda: client.delete_user(user_id))
    # personal key (no team) -> the user budget governs
    key = client.generate_key(extra_params={"user_id": user_id})
    resources.defer(lambda: client.delete_key(key))
    _assert_budget_blocks(client, key, "gpt-5.5")


def test_end_user_budget_blocks(
    client: BudgetClient, scoped_key: str, resources: ResourceManager
) -> None:
    customer = f"e2e-budget-cust-{unique_marker()}"
    client.create_customer(customer, max_budget=TINY_BUDGET)
    resources.defer(lambda: client.delete_customers([customer]))
    _assert_budget_blocks(client, scoped_key, "gpt-5.5", user=customer)


def test_organization_budget_blocks(
    client: BudgetClient, resources: ResourceManager
) -> None:
    # Org carries the tiny budget; the team under it has none, so a block here is
    # org-level enforcement (the historically weak link).
    org_id = client.create_org(
        max_budget=TINY_BUDGET, alias=f"e2e-budget-org-{unique_marker()}"
    )
    resources.defer(lambda: client.delete_org(org_id))
    team_id = client.create_team(
        alias=f"e2e-budget-team-{unique_marker()}", organization_id=org_id
    )
    resources.defer(lambda: client.delete_team(team_id))
    key = client.generate_key(extra_params={"team_id": team_id})
    resources.defer(lambda: client.delete_key(key))
    _assert_budget_blocks(client, key, "gpt-5.5")


def test_team_member_budget_blocks(
    client: BudgetClient, resources: ResourceManager
) -> None:
    # Member's per-team budget is tiny while the team itself has a large budget,
    # so a block proves member-level (not team-level) enforcement.
    team_id = client.create_team(
        alias=f"e2e-budget-team-{unique_marker()}", max_budget=100.0
    )
    resources.defer(lambda: client.delete_team(team_id))
    user_id = client.create_user(max_budget=100.0)
    resources.defer(lambda: client.delete_user(user_id))
    client.add_team_member(team_id, user_id, max_budget_in_team=TINY_BUDGET)
    key = client.generate_key(
        extra_params={"team_id": team_id, "user_id": user_id}
    )
    resources.defer(lambda: client.delete_key(key))
    _assert_budget_blocks(client, key, "gpt-5.5")
