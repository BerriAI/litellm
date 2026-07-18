"""Live e2e: regression guards for #25109 (budget resets stopped working).

The existing test_budget_reset_e2e.py / test_multi_window_budget_e2e.py prove a
blocked key flows again after its window. #25109 stored multi-budget-window data
in nullable JSON columns and filtered eligible rows with a `not: None`-style Prisma
filter that misbehaves on a nullable JSON column, so due rows were either skipped
(budget_reset_at stayed pinned, spend never cleared) or the reset path errored
(a non-budget 5xx leaked to callers). These tests assert the precise invariants
that bug broke, built up START-SLOW from scheduling -> enforcement -> the reset
strictly advancing -> the JSON-backed multi-window / team-member edges -> the
error path. They EXTEND the happy-path modules rather than duplicate them: each
asserts a delta (before<after timestamp, independent windows, block-not-error)
that the happy-path "calls flow again" check alone does not pin down.
"""

import time
from datetime import datetime

import pytest

from budget_client import BudgetClient, is_budget_block
from e2e_config import unique_marker
from e2e_http import require_successful_call
from lifecycle import ResourceManager
from models import BudgetWindow

pytestmark = pytest.mark.e2e

WINDOW_SECONDS = 30
RESET_DEADLINE_SECONDS = 150
TINY_CAP = 3e-6


def _call(client: BudgetClient, key: str):
    return client.chat(key, "claude-haiku-4-5", f"advance {unique_marker()}", max_tokens=16)


def _as_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _drive_to_block(client: BudgetClient, key: str) -> None:
    """Spend until the cap blocks; fails loudly if enforcement never trips."""
    for _ in range(20):
        result = _call(client, key)
        if is_budget_block(result):
            return
        require_successful_call(result)
        time.sleep(2)
    pytest.fail("budget never enforced before block")


# ---- Rung 1: scheduling exists at creation -----------------------------------


def test_key_with_budget_duration_schedules_reset_at_creation(
    client: BudgetClient, resources: ResourceManager
) -> None:
    """Baseline: a key created with a budget_duration has budget_reset_at populated
    immediately. The reset job can only advance a timestamp that was scheduled in
    the first place; everything below depends on this."""
    key = client.generate_key(max_budget=TINY_CAP, budget_duration=f"{WINDOW_SECONDS}s")
    resources.defer(lambda: client.delete_key(key))

    info = client.proxy.key_info(key)
    assert info.budget_reset_at is not None, "budget_duration set no budget_reset_at"
    assert _as_datetime(info.budget_reset_at) > _as_datetime("1970-01-01T00:00:00Z")


# ---- Rung 2: enforcement trips at the cap ------------------------------------


@pytest.mark.covers("quota_management.budget.key.blocks_over_limit")
def test_key_spend_blocks_at_cap(client: BudgetClient, resources: ResourceManager) -> None:
    """Sanity that the tiny cap is enforced before we test that it resets: spend
    accrues across calls and eventually returns budget_exceeded, never a 5xx."""
    key = client.generate_key(max_budget=TINY_CAP, budget_duration=f"{WINDOW_SECONDS}s")
    resources.defer(lambda: client.delete_key(key))

    # _drive_to_block is the enforcement proof: it fails unless a budget_exceeded
    # block follows successful (non-5xx) calls. key_info.spend is deliberately not
    # asserted - it is the DB-persisted field that flushes ~60s later
    # (proxy_batch_write_at), so reading it right after the block races to 0.0.
    _drive_to_block(client, key)


# ---- Rung 3: the core regression - reset_at strictly advances + spend zeroes --


@pytest.mark.covers("quota_management.budget.key.resets_after_window")
def test_key_budget_reset_at_advances_after_window(
    client: BudgetClient, resources: ResourceManager
) -> None:
    """The core #25109 guard: after the window elapses the reset job must move
    budget_reset_at strictly forward AND zero key.spend. The broken nullable-JSON
    filter left eligible rows untouched, so the timestamp stayed pinned and spend
    never cleared. Asserting before<after (not merely "a call succeeded") kills a
    mutation that no-ops the reset while leaving enforcement intact."""
    key = client.generate_key(max_budget=TINY_CAP, budget_duration=f"{WINDOW_SECONDS}s")
    resources.defer(lambda: client.delete_key(key))

    before_raw = client.proxy.key_info(key).budget_reset_at
    assert before_raw is not None, "no budget_reset_at scheduled at creation"
    before = _as_datetime(before_raw)

    _drive_to_block(client, key)

    deadline = time.monotonic() + RESET_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        time.sleep(5)
        result = _call(client, key)
        if not result.ok:
            assert is_budget_block(result), f"non-budget error during reset wait: {result.body[:200]}"
            continue
        info = client.proxy.key_info(key)
        assert info.budget_reset_at is not None, "budget_reset_at cleared by reset"
        assert _as_datetime(info.budget_reset_at) > before, (
            "budget_reset_at did not advance past the pre-reset value"
        )
        assert (info.spend or 0.0) < TINY_CAP, f"spend not cleared after reset: {info.spend}"
        return
    pytest.fail(f"key budget never reset within {RESET_DEADLINE_SECONDS}s")


# ---- Rung 4: multi-window - tight window resets, roomy window keeps spend -----


@pytest.mark.covers("quota_management.budget.key_multi_window.resets_windows_independently")
def test_multi_window_key_resets_each_window_independently(
    client: BudgetClient, resources: ResourceManager
) -> None:
    """The JSON-backed path #25109 specifically touched. A tight 30s window and a
    roomy 1m window: the tight window must reset on its own boundary while the roomy
    window keeps its accumulated spend (independent per-window reset). The
    nullable-JSON filter bug skipped these JSON-backed rows entirely, so the tight
    window never came back; a job that ERRORS on the JSON column would surface here
    as a non-budget 5xx, which we reject throughout the wait."""
    key = client.generate_key(
        budget_limits=[
            BudgetWindow(budget_duration=f"{WINDOW_SECONDS}s", max_budget=TINY_CAP),
            BudgetWindow(budget_duration="1m", max_budget=1.0),
        ]
    )
    resources.defer(lambda: client.delete_key(key))

    start = time.monotonic()
    _drive_to_block(client, key)
    spend_at_block = client.proxy.key_info(key).spend or 0.0

    deadline = time.monotonic() + RESET_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        time.sleep(5)
        result = _call(client, key)
        if result.ok:
            elapsed = time.monotonic() - start
            assert elapsed < WINDOW_SECONDS + 90, (
                f"tight window reset took {elapsed:.0f}s - too long for {WINDOW_SECONDS}s"
            )
            assert (client.proxy.key_info(key).spend or 0.0) >= spend_at_block, (
                "roomy window spend was wiped when only the tight window should reset"
            )
            return
        assert is_budget_block(result), f"non-budget error during reset wait: {result.body[:200]}"
    pytest.fail(f"tight window never reset within {RESET_DEADLINE_SECONDS}s")


# ---- Rung 5: team-member window advances (JSON-backed per-team budget) --------


@pytest.mark.covers("quota_management.budget.team_member.resets_after_window")
def test_team_member_budget_reset_at_advances(
    client: BudgetClient, resources: ResourceManager
) -> None:
    """Per-team member windows are also JSON-backed. member_budget_reset_at must
    advance after the window; the explicit before<after assertion is the #25109
    regression guard (the existing reset test only checks "it eventually moved",
    this pins it strictly past the value recorded before the window)."""
    team_id = client.create_team(alias=f"e2e-member-advance-{unique_marker()}", max_budget=100.0)
    resources.defer(lambda: client.delete_team(team_id))
    user_id = client.create_user(max_budget=100.0)
    resources.defer(lambda: client.delete_user(user_id))

    client.add_team_member(team_id, user_id, max_budget_in_team=1.0)
    client.update_team_member(
        team_id, user_id, max_budget_in_team=1.0, budget_duration=f"{WINDOW_SECONDS}s"
    )

    before_raw = client.member_budget_reset_at(team_id, user_id)
    assert before_raw, "updating the member with a budget_duration set no budget_reset_at"
    before = _as_datetime(before_raw)

    key = client.generate_key(team_id=team_id, user_id=user_id)
    resources.defer(lambda: client.delete_key(key))
    require_successful_call(_call(client, key))

    deadline = time.monotonic() + RESET_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        time.sleep(5)
        current = client.member_budget_reset_at(team_id, user_id)
        if current and _as_datetime(current) > before:
            return
    pytest.fail(f"member budget_reset_at never advanced past {before.isoformat()} in {RESET_DEADLINE_SECONDS}s")


# ---- Rung 6: error-path edge - resets surface as blocks, never 5xx -----------


def test_reset_wait_never_yields_non_budget_error(
    client: BudgetClient, resources: ResourceManager
) -> None:
    """The other #25109 failure mode: a reset job that ERRORS on the nullable-JSON
    column surfaces to the caller as a non-budget 5xx. Across the whole reset wait
    every non-ok response must be a budget block (is_budget_block) and never a
    server error; this guards the error path independently of whether the reset
    eventually fires."""
    key = client.generate_key(max_budget=TINY_CAP, budget_duration=f"{WINDOW_SECONDS}s")
    resources.defer(lambda: client.delete_key(key))

    _drive_to_block(client, key)

    saw_reset = False
    deadline = time.monotonic() + RESET_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        time.sleep(5)
        result = _call(client, key)
        if result.ok:
            saw_reset = True
            break
        assert is_budget_block(result), (
            f"reset wait yielded a non-budget error (likely a JSON-column reset crash): {result.body[:200]}"
        )
    assert saw_reset, f"key budget never reset within {RESET_DEADLINE_SECONDS}s"
