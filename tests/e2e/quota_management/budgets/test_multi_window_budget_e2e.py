"""Live e2e: multi-window budgets (budget_limits) enforce AND reset per window.

Short windows make the time limit reachable inside a test: a tight 30s window and
a roomy 1m window. The 30s window blocks once its tiny cap is exceeded, then - once
its 30s elapses and the reset job runs (rescheduled fast via
PROXY_BUDGET_RESCHEDULER_* in docker-compose) - the window resets and calls flow
again. Closes the multi-window gap (enforcement + per-window reset) in
BUDGET_TEST_COVERAGE_MATRIX.md, which the unit suite covered but no live test did.

The long-window direction (E2E-12 steps 4-5) is the second test: a 1d window whose
cap the first burn already crossed (LONG_CAP sits ~20-100x below one real call's
cost) must keep blocking after the 30s window provably reset. The short window's reset_at is recorded AFTER the block and the test waits
for it to advance strictly past that value - the reset job advances reset_at in the
same pass that zeroes the window's spend counter, so once it moved (and no further
spend landed) the short window cannot be the blocker; the 1d window's persisted
spend is. Recording reset_at at mint time instead would race a boundary that rolls
before the burn's spend lands, letting the short window block again and pass the
test for the wrong reason. Enforcement reads a cached auth object, so its view of
the rolled window lags the DB reset_at write; the final phase therefore polls until
the refusal is attributed to the 1d window ("over 1d budget"), failing immediately
if any call succeeds (the 1d cap failed to hold) or a non-budget error leaks.
"""

import time
from datetime import datetime

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
# HTML mid-wait, which is not a budget signal. gpt-5.5 stays well under that
# ceiling so the wait loop measures window reset, not provider/ALB timeout.
# max_tokens must be >1: gpt-5.5 refuses completions that hit the output limit
# mid-message when capped at 1 token.
MODEL = CHEAP_OPENAI_MODEL
SHORT_WINDOW = f"{WINDOW_SECONDS}s"
LONG_WINDOW = "1d"
LONG_CAP = 5e-7
RESET_DEADLINE_SECONDS = 150


def _call(client: BudgetClient, key: str):
    return client.chat(key, MODEL, f"window {unique_marker()}", max_tokens=16)


def _as_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _drive_to_block(client: BudgetClient, key: str) -> None:
    for _ in range(20):
        result = _call(client, key)
        if is_budget_block(result):
            return
        require_successful_call(result)
        time.sleep(2)
    pytest.fail("budget never enforced before block")


@pytest.mark.covers("quota_management.budget.key_multi_window.blocks_then_resets")
def test_short_window_blocks_then_resets(client: BudgetClient, resources: ResourceManager) -> None:
    key = client.generate_key(
        models=[MODEL],
        budget_limits=[
            BudgetWindow(budget_duration=SHORT_WINDOW, max_budget=1e-9),
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
            BudgetWindow(budget_duration=SHORT_WINDOW, max_budget=1e-9),
            BudgetWindow(budget_duration=LONG_WINDOW, max_budget=LONG_CAP),
        ],
    )
    resources.defer(lambda: client.delete_key(key))

    _drive_to_block(client, key)

    blocked_reset_at = client.key_window_reset_at(key, SHORT_WINDOW)
    assert blocked_reset_at is not None, "short window missing from /key/info budget_limits"
    blocked_long_reset_at = client.key_window_reset_at(key, LONG_WINDOW)

    deadline = time.monotonic() + RESET_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        time.sleep(5)
        current = client.key_window_reset_at(key, SHORT_WINDOW)
        if current is not None and _as_datetime(current) > _as_datetime(blocked_reset_at):
            break
    else:
        pytest.fail(
            f"{SHORT_WINDOW} window's reset_at never advanced past {blocked_reset_at} within {RESET_DEADLINE_SECONDS}s"
        )

    deadline = time.monotonic() + RESET_DEADLINE_SECONDS
    last_body = ""
    while time.monotonic() < deadline:
        result = _call(client, key)
        if result.ok:
            rolled = client.key_window_reset_at(key, LONG_WINDOW) != blocked_long_reset_at
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
