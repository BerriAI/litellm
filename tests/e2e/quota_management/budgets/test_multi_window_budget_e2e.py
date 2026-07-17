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
from e2e_config import CHEAP_OPENAI_MODEL, unique_marker
from e2e_http import require_successful_call
from lifecycle import ResourceManager
from models import BudgetWindow

pytestmark = pytest.mark.e2e

WINDOW_SECONDS = 30  # the tight window; calls succeed again only after it elapses
# Prefer the OpenAI cheap model for this polling test: under the full stage suite
# Claude chat latency + ALB target idle timeout (~60s) can surface as awselb 502
# HTML mid-wait, which is not a budget signal. gpt-5.5 + 1 token stays well under
# that ceiling so the wait loop measures window reset, not provider/ALB timeout.
MODEL = CHEAP_OPENAI_MODEL


def _call(client: BudgetClient, key: str):
    return client.chat(
        key, MODEL, f"window {unique_marker()}", max_tokens=1
    )


@pytest.mark.covers("quota_management.budget.key_multi_window.blocks_then_resets")
def test_short_window_blocks_then_resets(
    client: BudgetClient, resources: ResourceManager
) -> None:
    key = client.generate_key(
        models=[MODEL],
        budget_limits=[
            BudgetWindow(budget_duration=f"{WINDOW_SECONDS}s", max_budget=3e-6),
            BudgetWindow(budget_duration="1m", max_budget=1.0),  # roomy: never blocks
        ],
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

    # 2. the window resets at the next wall-clock-aligned boundary (up to a window
    #    after start), then the reset job (~15-20s rescheduler) zeroes the spend.
    #    Allow generous headroom for that alignment + rescheduler latency; a stuck
    #    rescheduler is caught by the wait-loop timeout, not this elapsed bound.
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        time.sleep(5)
        result = _call(client, key)
        if result.ok:
            elapsed = time.monotonic() - start
            assert elapsed < WINDOW_SECONDS + 90, (
                f"reset took {elapsed:.0f}s - too long for a {WINDOW_SECONDS}s window"
            )
            return
        assert is_budget_block(result), (
            f"non-budget error during reset wait: status={result.status_code} "
            f"body={result.body[:200]}"
        )
    pytest.fail(f"{WINDOW_SECONDS}s window never reset within 150s")
