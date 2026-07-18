"""Live e2e: multi-window budgets (budget_limits) enforce AND reset per window.

Short windows make the time limit reachable inside a test: a tight 30s window and
a roomy 1m window. The 30s window blocks once its tiny cap is exceeded, then - once
its 30s elapses and the reset job runs (rescheduled fast via
PROXY_BUDGET_RESCHEDULER_* in docker-compose) - the window resets and calls flow
again. Closes the multi-window gap (enforcement + per-window reset) in
BUDGET_TEST_COVERAGE_MATRIX.md, which the unit suite covered but no live test did.

The second test is the long-window direction: one burn crosses
both caps; after the 30s window's reset_at strictly advances (read post-block since
a mint-time read races the boundary; the reset job zeroes the counter in the same
pass), the key must still be refused with "over 1d budget". That check polls because
enforcement's cached auth view lags the DB write; any 200 or non-budget error fails
immediately.
"""

import time

import pytest

from budget_client import BudgetClient, drive_to_block, is_budget_block, window_reset_at
from e2e_config import CHEAP_OPENAI_MODEL, unique_marker
from lifecycle import ResourceManager
from models import BudgetWindow

pytestmark = pytest.mark.e2e

WINDOW_SECONDS = 30  # the tight window; calls succeed again only after it elapses
# Prefer the OpenAI cheap model for this polling test: under the full stage suite
# Claude chat latency + ALB target idle timeout (~60s) can surface as awselb 502
# HTML mid-wait, which is not a budget signal. gpt-5.5 stays well under that
# ceiling so the wait loop measures window reset, not provider/ALB timeout.
# max_tokens must be >1: gpt-5.5 refuses completions that hit the output limit
# mid-message when capped at 1 token.
MODEL = CHEAP_OPENAI_MODEL
SHORT_WINDOW = f"{WINDOW_SECONDS}s"
LONG_WINDOW = "1d"
TINY_CAP = 1e-9
LONG_CAP = 5e-7
RESET_DEADLINE_SECONDS = 150


def _call(client: BudgetClient, key: str):
    return client.chat(key, MODEL, f"window {unique_marker()}", max_tokens=16)


def _drive_to_block(client: BudgetClient, key: str) -> None:
    drive_to_block(
        lambda: _call(client, key),
        attempts=20,
        pause_seconds=2,
        fail_message="budget never enforced before block",
    )


@pytest.mark.covers("quota_management.budget.key_multi_window.blocks_then_resets")
def test_short_window_blocks_then_resets(client: BudgetClient, resources: ResourceManager) -> None:
    key = client.generate_key(
        models=[MODEL],
        budget_limits=[
            BudgetWindow(budget_duration=SHORT_WINDOW, max_budget=TINY_CAP),
            BudgetWindow(budget_duration="1m", max_budget=1.0),  # roomy: never blocks
        ],
    )
    resources.defer(lambda: client.delete_key(key))

    # 1. exhaust the tight window -> litellm returns budget_exceeded
    start = time.monotonic()
    _drive_to_block(client, key)

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
            assert elapsed < WINDOW_SECONDS + 90, f"reset took {elapsed:.0f}s - too long for a {WINDOW_SECONDS}s window"
            return
        assert is_budget_block(result), (
            f"non-budget error during reset wait: status={result.status_code} body={result.body[:200]}"
        )
    pytest.fail(f"{WINDOW_SECONDS}s window never reset within 150s")


@pytest.mark.covers("quota_management.budget.key_multi_window.blocks_then_resets")
def test_long_window_blocks_after_short_window_resets(client: BudgetClient, resources: ResourceManager) -> None:
    key = client.generate_key(
        models=[MODEL],
        budget_limits=[
            BudgetWindow(budget_duration=SHORT_WINDOW, max_budget=TINY_CAP),
            BudgetWindow(budget_duration=LONG_WINDOW, max_budget=LONG_CAP),
        ],
    )
    resources.defer(lambda: client.delete_key(key))

    _drive_to_block(client, key)

    blocked_reset_at = window_reset_at(client.key_budget_windows(key), SHORT_WINDOW)
    assert blocked_reset_at is not None, "short window missing from /key/info budget_limits"
    blocked_long_reset_at = window_reset_at(client.key_budget_windows(key), LONG_WINDOW)
    assert blocked_long_reset_at is not None, "long window missing from /key/info budget_limits"

    # wait for the reset job to roll the short (30s) window: reset_at advancing past
    # the value recorded at block time proves that window's spend counter was zeroed
    deadline = time.monotonic() + RESET_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        time.sleep(5)
        current = window_reset_at(client.key_budget_windows(key), SHORT_WINDOW)
        if current is not None and current > blocked_reset_at:
            break
    else:
        pytest.fail(
            f"{SHORT_WINDOW} window's reset_at never advanced past {blocked_reset_at} within {RESET_DEADLINE_SECONDS}s"
        )

    # the long (1d) window must keep blocking now that the short window is clean:
    # every response must stay a budget 429, and we pass only once the error names
    # the 1d window (a 200 here means the long cap failed to hold)
    deadline = time.monotonic() + RESET_DEADLINE_SECONDS
    last_body = ""
    while time.monotonic() < deadline:
        result = _call(client, key)
        if result.ok:
            rolled = window_reset_at(client.key_budget_windows(key), LONG_WINDOW) != blocked_long_reset_at
            pytest.fail(
                f"{LONG_WINDOW} window failed to block after the {SHORT_WINDOW} window reset"
                + (f" (the {LONG_WINDOW} window itself rolled mid-test - boundary crossed; rerun)" if rolled else "")
            )
        assert is_budget_block(result), (
            f"non-budget error while waiting for {LONG_WINDOW} attribution: "
            f"status={result.status_code} body={result.body[:200]}"
        )
        if f"over {LONG_WINDOW} budget" in result.body:
            return
        last_body = result.body
        time.sleep(5)
    pytest.fail(
        f"block never attributed to the {LONG_WINDOW} window within {RESET_DEADLINE_SECONDS}s: {last_body[:200]}"
    )
