"""Live e2e: a user's multi-window budgets (budget_limits) enforce AND reset per window.

The user analog of test_team_multi_window_budget_e2e.py. A user is created with a
tight 30s window and a roomy 1m window; a personal key (no team) blocks once the tight
window's cap is exceeded. After the 30s elapses and the reset job runs, the window
resets and calls flow again.

User budget_limits enforcement only fires for personal keys (team_id is None), matching
the single-budget user check.
"""

import time

import pytest

from budget_client import BudgetClient, is_budget_block
from e2e_config import unique_marker
from e2e_http import require_successful_call
from lifecycle import ResourceManager
from models import BudgetWindow

pytestmark = pytest.mark.e2e

WINDOW_SECONDS = 30


def _call(client: BudgetClient, key: str):
    return client.chat(key, "claude-haiku-4-5", f"user-window {unique_marker()}", max_tokens=16)


def test_user_short_window_blocks_then_resets(client: BudgetClient, resources: ResourceManager) -> None:
    user_id = client.create_user(
        budget_limits=[
            BudgetWindow(budget_duration=f"{WINDOW_SECONDS}s", max_budget=3e-6),
            BudgetWindow(budget_duration="1m", max_budget=1.0),
        ],
    )
    resources.defer(lambda: client.delete_user(user_id))
    key = client.generate_key(user_id=user_id)
    resources.defer(lambda: client.delete_key(key))

    start = time.monotonic()
    blocked = False
    for _ in range(20):
        result = _call(client, key)
        if is_budget_block(result):
            blocked = True
            break
        require_successful_call(result)
        time.sleep(2)
    assert blocked, f"user {WINDOW_SECONDS}s window never enforced"

    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        time.sleep(5)
        result = _call(client, key)
        if result.ok:
            elapsed = time.monotonic() - start
            assert elapsed < WINDOW_SECONDS + 90, f"reset took {elapsed:.0f}s - too long for a {WINDOW_SECONDS}s window"
            return
        assert is_budget_block(result), f"non-budget error during reset wait: {result.body[:200]}"
    pytest.fail(f"user {WINDOW_SECONDS}s window never reset within 150s")
