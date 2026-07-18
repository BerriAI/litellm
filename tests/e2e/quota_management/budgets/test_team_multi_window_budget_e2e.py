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

The second test is the long-window direction, mirroring the
key-side test (see its docstring): after the team's 30s window provably resets, the
1d window the same burn crossed must still block the team key with "over 1d budget".
"""

import time

import pytest

from budget_client import BudgetClient, drive_to_block, is_budget_block, window_reset_at
from e2e_http import StreamingResponse
from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import BudgetWindow

pytestmark = pytest.mark.e2e

WINDOW_SECONDS = 30
SHORT_WINDOW = f"{WINDOW_SECONDS}s"
LONG_WINDOW = "1d"
TINY_CAP = 1e-9
LONG_CAP = 5e-7
RESET_DEADLINE_SECONDS = 150


def _call(client: BudgetClient, key: str):
    return client.chat(key, "claude-haiku-4-5", f"team-window {unique_marker()}", max_tokens=16)


def _drive_to_block(client: BudgetClient, key: str) -> StreamingResponse:
    return drive_to_block(
        lambda: _call(client, key),
        attempts=30,
        pause_seconds=1,
        fail_message="team budget never enforced before block",
    )


@pytest.mark.covers("quota_management.budget.team_multi_window.blocks_then_resets")
def test_team_short_window_blocks_then_resets(client: BudgetClient, resources: ResourceManager) -> None:
    team_id = client.create_team(
        alias=f"e2e-team-window-{unique_marker()}",
        budget_limits=[
            BudgetWindow(budget_duration=SHORT_WINDOW, max_budget=3e-6),
            BudgetWindow(budget_duration="1m", max_budget=1.0),  # roomy: never blocks
        ],
    )
    resources.defer(lambda: client.delete_team(team_id))
    key = client.generate_key(team_id=team_id, models=["claude-haiku-4-5"])
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
        assert is_budget_block(result), f"non-budget error during reset wait: {result.body[:200]}"
    pytest.fail(f"team {WINDOW_SECONDS}s window never reset within 150s")


@pytest.mark.covers("quota_management.budget.team_multi_window.blocks_then_resets")
def test_team_long_window_blocks_after_short_window_resets(client: BudgetClient, resources: ResourceManager) -> None:
    team_id = client.create_team(
        alias=f"e2e-team-long-window-{unique_marker()}",
        budget_limits=[
            BudgetWindow(budget_duration=SHORT_WINDOW, max_budget=TINY_CAP),
            BudgetWindow(budget_duration=LONG_WINDOW, max_budget=LONG_CAP),
        ],
    )
    resources.defer(lambda: client.delete_team(team_id))
    key = client.generate_key(team_id=team_id, models=["claude-haiku-4-5"])
    resources.defer(lambda: client.delete_key(key))

    blocked = _drive_to_block(client, key)
    assert blocked.status_code == 429, f"budget block was not a 429: {blocked.status_code} {blocked.body[:200]}"

    blocked_reset_at = window_reset_at(client.team_budget_windows(team_id), SHORT_WINDOW)
    assert blocked_reset_at is not None, "short window missing from /team/info budget_limits"
    blocked_long_reset_at = window_reset_at(client.team_budget_windows(team_id), LONG_WINDOW)
    assert blocked_long_reset_at is not None, "long window missing from /team/info budget_limits"

    # wait for the reset job to roll the short (30s) window: reset_at advancing past
    # the value recorded at block time proves that window's spend counter was zeroed
    deadline = time.monotonic() + RESET_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        time.sleep(5)
        current = window_reset_at(client.team_budget_windows(team_id), SHORT_WINDOW)
        if current is not None and current > blocked_reset_at:
            break
    else:
        pytest.fail(
            f"team {SHORT_WINDOW} window's reset_at never advanced past "
            f"{blocked_reset_at} within {RESET_DEADLINE_SECONDS}s"
        )

    # the long (1d) window must keep blocking now that the short window is clean:
    # every response must stay a budget 429, and we pass only once the error names
    # the 1d window (a 200 here means the long cap failed to hold)
    deadline = time.monotonic() + RESET_DEADLINE_SECONDS
    last_body = ""
    while time.monotonic() < deadline:
        result = _call(client, key)
        if result.ok:
            rolled = window_reset_at(client.team_budget_windows(team_id), LONG_WINDOW) != blocked_long_reset_at
            pytest.fail(
                f"team {LONG_WINDOW} window failed to block after the {SHORT_WINDOW} window reset"
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
        f"team block never attributed to the {LONG_WINDOW} window within {RESET_DEADLINE_SECONDS}s: {last_body[:200]}"
    )
