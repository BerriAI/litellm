"""Live e2e: soft_budget alerts but does NOT block.

A key with a tiny `soft_budget` well under a large `max_budget`: spend crosses the
soft threshold within a couple calls, but requests keep succeeding (soft budget is
advisory). Closes the soft_budget gap in BUDGET_TEST_COVERAGE_MATRIX.md. The alert
side-effect (Slack/email) is not observable from the proxy API, so we assert the
load-bearing behavior: soft != block.
"""

import pytest

from budget_client import BudgetClient, is_budget_block
from e2e_config import unique_marker
from e2e_http import require_successful_call
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e


@pytest.mark.covers("quota_management.budget.soft.alerts_without_blocking")
def test_soft_budget_does_not_block(
    client: BudgetClient, resources: ResourceManager
) -> None:
    # soft far below max: spend crosses soft immediately, stays under max.
    key = client.generate_key(max_budget=1000.0, soft_budget=1e-9)
    resources.defer(lambda: client.delete_key(key))

    for _ in range(3):
        result = client.chat(
            key, "claude-haiku-4-5", f"hi {unique_marker()}", max_tokens=16
        )
        assert not is_budget_block(result), (
            "soft_budget blocked a request; it must alert only, not block "
            f"(body={result.body[:200]})"
        )
        require_successful_call(result)  # any other non-2xx (e.g. provider down) is a hard fail
