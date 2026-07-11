"""Live e2e: a key budget resets (zeroes spend) after its budget_duration.

Short budget_duration (30s) + the fast-rescheduled reset job: a key blocked for
exceeding its max_budget starts succeeding again once the duration elapses and the
reset job zeroes key.spend. Closes the reset-zeroing gap in
BUDGET_TEST_COVERAGE_MATRIX.md (reset_budget_for_litellm_keys), which the unit
suite covers but no live test did - distinct from the per-window reset in
test_multi_window_budget_e2e.py.
"""

import time

import pytest

from budget_client import BudgetClient, is_budget_block
from e2e_config import unique_marker
from e2e_http import require_successful_call
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e


def _call(client: BudgetClient, key: str):
    return client.chat(
        key, "claude-haiku-4-5", f"reset {unique_marker()}", max_tokens=16
    )


@pytest.mark.covers("quota_management.budget.key.resets_after_window")
def test_key_budget_resets_after_duration(
    client: BudgetClient, resources: ResourceManager
) -> None:
    key = client.generate_key(max_budget=3e-6, budget_duration="30s")
    resources.defer(lambda: client.delete_key(key))

    # 1. exceed the budget -> litellm returns budget_exceeded
    blocked = False
    for _ in range(20):
        result = _call(client, key)
        if is_budget_block(result):
            blocked = True
            break
        require_successful_call(result)
        time.sleep(2)
    assert blocked, "key budget never enforced"

    # 2. once the 30s duration elapses + the reset job runs, key.spend zeroes and
    #    calls flow again. The window is wall-clock-aligned, so the reset lands up to
    #    a window later, then the rescheduler (~15-20s) zeroes the spend; allow
    #    generous headroom over that. A stuck rescheduler is caught by the wait-loop
    #    timeout, not this elapsed bound.
    start = time.monotonic()
    while time.monotonic() < start + 150:
        time.sleep(5)
        result = _call(client, key)
        if result.ok:
            assert time.monotonic() - start < 120, "reset too slow for a 30s budget"
            return
        assert is_budget_block(result), f"non-budget error: {result.body[:200]}"
    pytest.fail("key budget never reset within 150s")
