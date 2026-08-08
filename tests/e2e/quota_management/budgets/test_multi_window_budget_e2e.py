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

The blocks-then-resets check also sweeps the personal / team / team-member mint
shapes: the same budget_limits pair rides a key minted to roomy (100.0)
surroundings, so only the key's own windows can block regardless of who holds it.
"""

import time

import pytest

from budget_client import BudgetClient, is_budget_block, window_reset_at
from e2e_http import StreamingResponse, require_successful_call
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


def _drive_to_block(client: BudgetClient, key: str) -> StreamingResponse:
    for _ in range(20):
        result = _call(client, key)
        if is_budget_block(result):
            return result
        require_successful_call(result)
        time.sleep(2)
    pytest.fail("budget never enforced before block")


def _short_roomy_limits() -> list[BudgetWindow]:
    return [
        BudgetWindow(budget_duration=SHORT_WINDOW, max_budget=TINY_CAP),
        BudgetWindow(budget_duration="1m", max_budget=1.0),  # roomy: never blocks
    ]


def _assert_short_window_blocks_then_resets(client: BudgetClient, key: str) -> None:
    # 1. exhaust the tight window -> litellm returns budget_exceeded
    start = time.monotonic()
    blocked = _drive_to_block(client, key)
    assert f"over {SHORT_WINDOW} budget" in blocked.body, (
        f"block must be attributed to the {SHORT_WINDOW} window, got: {blocked.body[:200]}"
    )

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


def _mint_key_of_kind(client: BudgetClient, resources: ResourceManager, kind: str) -> str:
    """A key carrying the tight+roomy budget_limits pair, minted to roomy (100.0)
    surroundings so only the key's own windows can block."""
    match kind:
        case "personal":
            user_id = client.create_user(max_budget=100.0)
            resources.defer(lambda: client.delete_user(user_id))
            key = client.generate_key(models=[MODEL], user_id=user_id, budget_limits=_short_roomy_limits())
        case "team":
            team_id = client.create_team(alias=f"e2e-mw-team-{unique_marker()}", max_budget=100.0)
            resources.defer(lambda: client.delete_team(team_id))
            key = client.generate_key(models=[MODEL], team_id=team_id, budget_limits=_short_roomy_limits())
        case "team_member":
            team_id = client.create_team(alias=f"e2e-mw-team-{unique_marker()}", max_budget=100.0)
            resources.defer(lambda: client.delete_team(team_id))
            member_id = client.create_user(max_budget=100.0)
            resources.defer(lambda: client.delete_user(member_id))
            client.add_team_member(team_id, member_id, max_budget_in_team=100.0)
            key = client.generate_key(
                models=[MODEL], team_id=team_id, user_id=member_id, budget_limits=_short_roomy_limits()
            )
        case _:
            pytest.fail(f"unknown key kind: {kind}")
    resources.defer(lambda: client.delete_key(key))
    return key


class TestKeyMultiWindowBudget:
    @pytest.mark.covers("quota_management.budget.key_multi_window.blocks_then_resets")
    def test_short_window_blocks_then_resets(self, client: BudgetClient, resources: ResourceManager) -> None:
        key = client.generate_key(models=[MODEL], budget_limits=_short_roomy_limits())
        resources.defer(lambda: client.delete_key(key))

        _assert_short_window_blocks_then_resets(client, key)


    @pytest.mark.covers("quota_management.budget.key_multi_window.blocks_then_resets")
    @pytest.mark.parametrize("kind", ["personal", "team", "team_member"])
    def test_short_window_blocks_then_resets_across_key_kinds(
        self, client: BudgetClient, resources: ResourceManager, kind: str
    ) -> None:
        key = _mint_key_of_kind(client, resources, kind)

        _assert_short_window_blocks_then_resets(client, key)


    @pytest.mark.covers("quota_management.budget.key_multi_window.blocks_then_resets")
    def test_long_window_blocks_after_short_window_resets(
        self, client: BudgetClient, resources: ResourceManager
    ) -> None:
        key = client.generate_key(
            models=[MODEL],
            budget_limits=[
                BudgetWindow(budget_duration=SHORT_WINDOW, max_budget=TINY_CAP),
                BudgetWindow(budget_duration=LONG_WINDOW, max_budget=LONG_CAP),
            ],
        )
        resources.defer(lambda: client.delete_key(key))

        # 1. drive the key to get blocked by SHORT_WINDOW, assert it's budget error
        blocked = _drive_to_block(client, key)
        assert blocked.status_code == 429, f"budget block was not a 429: {blocked.status_code} {blocked.body[:200]}"

        # 2. check the reset times of both budget windows after we drove to being blocked
        blocked_reset_at = window_reset_at(client.key_budget_windows(key), SHORT_WINDOW)
        assert blocked_reset_at is not None, "short window missing from /key/info budget_limits"
        blocked_long_reset_at = window_reset_at(client.key_budget_windows(key), LONG_WINDOW)
        assert blocked_long_reset_at is not None, "long window missing from /key/info budget_limits"

        # 3. poll every 5s for the SHORT_WINDOW reset time until it is past it, fails if it doesnt reset 
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

        # 4. short window just reset in 3, so now make a call, check that its blocked (should be blocked by LONG_WINDOW because short window reset), also make sure its budget error
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
