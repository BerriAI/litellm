"""Live e2e: proxy-level tag budgets block tagged requests.

A tag with a tiny budget: requests carrying that tag get blocked once the tag's
spend is exceeded, while a request with a different tag (no budget) still works.
Closes the proxy-level tag-budget gap in BUDGET_TEST_COVERAGE_MATRIX.md (today
only router-level tag budgets are tested).
"""

import time

import pytest

from budget_client import BudgetClient, is_budget_block
from e2e_config import unique_marker
from e2e_http import require_successful_call
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e

TINY_BUDGET = 1e-6


def _tagged_call(client: BudgetClient, key: str, tag: str):
    result = client.chat(
        key,
        "claude-haiku-4-5",
        f"hi {unique_marker()}",
        tags=[tag],
        max_tokens=64,
    )
    if not result.ok and not is_budget_block(result):
        require_successful_call(result)
    return result


@pytest.mark.covers("quota_management.budget.tag.blocks_over_limit")
def test_tag_budget_blocks_tagged_requests(
    client: BudgetClient, scoped_key: str, resources: ResourceManager
) -> None:
    budgeted_tag = f"e2e-budget-tag-{unique_marker()}"
    client.create_tag(budgeted_tag, max_budget=TINY_BUDGET)
    resources.defer(lambda: client.delete_tag(budgeted_tag))

    first = _tagged_call(client, scoped_key, budgeted_tag)
    if is_budget_block(first):
        blocked = True
    else:
        require_successful_call(first)
        blocked = False
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if is_budget_block(_tagged_call(client, scoped_key, budgeted_tag)):
                blocked = True
                break
            time.sleep(1)
    assert blocked, f"tag budget for {budgeted_tag!r} never enforced"

    free_tag = f"e2e-free-tag-{unique_marker()}"
    other = _tagged_call(client, scoped_key, free_tag)
    assert not is_budget_block(other), (
        f"unbudgeted tag {free_tag!r} was blocked by {budgeted_tag!r}'s budget"
    )
    require_successful_call(other)
