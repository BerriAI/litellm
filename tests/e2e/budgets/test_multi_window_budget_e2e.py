"""Live e2e: multi-window budgets (budget_limits) enforce AND reset per window.

Short windows make the time limit reachable inside a test: a tight 30s window and
a roomy 1m window. The 30s window blocks once its tiny cap is exceeded, then - once
its 30s elapses and the reset job runs (rescheduled fast via
PROXY_BUDGET_RESCHEDULER_* in docker-compose) - the window resets and calls flow
again. Closes the multi-window gap (enforcement + per-window reset) in
BUDGET_TEST_COVERAGE_MATRIX.md, which the unit suite covered but no live test did.
"""

import time

import pytest

from budget_client import BudgetClient, is_budget_block
from lifecycle import ResourceManager
from proxy_client import require_successful_call, unique_marker

pytestmark = pytest.mark.e2e

WINDOW_SECONDS = 30  # the tight window; calls succeed again only after it elapses


def _call(client: BudgetClient, key: str):
    return client.chat(
        key, "claude-haiku-4-5", f"window {unique_marker()}", extra_body={"max_tokens": 16}
    )


def test_short_window_blocks_then_resets(
    client: BudgetClient, resources: ResourceManager
) -> None:
    key = client.generate_key(
        extra_params={
            "budget_limits": [
                {"budget_duration": f"{WINDOW_SECONDS}s", "max_budget": 3e-6},
                {"budget_duration": "1m", "max_budget": 1.0},  # roomy: never blocks
            ]
        }
    )
    resources.defer(lambda: client.delete_key(key))

    # 1. exhaust the tight window -> litellm returns budget_exceeded
    start = time.monotonic()
    blocked = False
    for _ in range(20):
        result = _call(client, key)
        if is_budget_block(result):
            blocked = True
            break
        require_successful_call(result)
        time.sleep(2)
    assert blocked, f"{WINDOW_SECONDS}s window never enforced"

    # 2. the window resets at the next wall-clock-aligned boundary (so it can land
    #    a little under WINDOW_SECONDS from creation) + the reset job. When a call
    #    flows again the window has reset; the elapsed clock must be short enough
    #    that this is the 30s window resetting, not the roomy 1m one.
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        time.sleep(5)
        result = _call(client, key)
        if result.ok:
            elapsed = time.monotonic() - start
            assert elapsed < WINDOW_SECONDS + 45, (
                f"reset took {elapsed:.0f}s - too long for a {WINDOW_SECONDS}s window"
            )
            return
        assert is_budget_block(result), f"non-budget error during reset wait: {result.body[:200]}"
    pytest.fail(f"{WINDOW_SECONDS}s window never reset within 90s")
