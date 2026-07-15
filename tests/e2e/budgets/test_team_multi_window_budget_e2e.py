"""Live e2e: a team's multi-window budgets (budget_limits) enforce AND reset per window.

The team analog of test_multi_window_budget_e2e.py (which covers keys). A team is
created with a tight 30s window and a roomy 1m window; a key on that team blocks once
the tight window's cap is exceeded, then - once the 30s elapses and the reset job runs
(rescheduled fast via PROXY_BUDGET_RESCHEDULER_* in docker-compose) - the window resets
and calls flow again. This exercises the reset_budget_windows TEAM branch (raw SQL over
LiteLLM_TeamTable.budget_limits, the literal #25109 path), which had no live coverage.

This also guards the /team/new write path: it must json.dumps the window list into
the Json? column. A raw list there made Prisma reject the create with a 500 (the key
path and /team/update already json.dumps first); a regression would fail team creation
here.
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
    return client.chat(key, "claude-haiku-4-5", f"team-window {unique_marker()}", max_tokens=16)


def test_team_short_window_blocks_then_resets(client: BudgetClient, resources: ResourceManager) -> None:
    team_id = client.create_team(
        alias=f"e2e-team-window-{unique_marker()}",
        budget_limits=[
            BudgetWindow(budget_duration=f"{WINDOW_SECONDS}s", max_budget=3e-6),
            BudgetWindow(budget_duration="1m", max_budget=1.0),  # roomy: never blocks
        ],
    )
    resources.defer(lambda: client.delete_team(team_id))
    key = client.generate_key(team_id=team_id)
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
    assert blocked, f"team {WINDOW_SECONDS}s window never enforced"

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
        assert is_budget_block(result), f"non-budget error during reset wait: {result.body[:200]}"
    pytest.fail(f"team {WINDOW_SECONDS}s window never reset within 150s")
