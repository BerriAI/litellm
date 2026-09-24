"""Live e2e: a per-user max_budget is enforced across ALL of that user's keys.

An internal user's budget governs every personal key it owns, not only the one
that happened to spend it down. One user with a tiny max_budget owns two keys:
driving the first key to a budget_exceeded block then makes a fresh, untouched
second key of the same user (which carries no budget of its own, so nothing but the
shared user budget can block it) reject the same way, and the user's recorded spend
has crossed the cap. A key-scoped-only budget would leave the second key serving.
"""

import time

import pytest

from budget_client import BudgetClient, is_budget_block
from e2e_config import unique_marker
from e2e_http import StreamingResponse, require_successful_call
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e

MODEL = "gpt-5.5"
TINY_CAP = 3e-6
RECORDED_SPEND_DEADLINE_SECONDS = 90
SECOND_KEY_BLOCK_ATTEMPTS = 6


def _call(client: BudgetClient, key: str) -> StreamingResponse:
    return client.chat(key, MODEL, f"across {unique_marker()}", max_tokens=16)


def _drive_to_block(client: BudgetClient, key: str, subject: str) -> None:
    for _ in range(40):
        result = _call(client, key)
        if is_budget_block(result):
            return
        require_successful_call(result)
        time.sleep(2)
    pytest.fail(f"user budget never enforced on {subject} within the call budget")


def _expect_prompt_block(client: BudgetClient, key: str, subject: str) -> None:
    """The shared user budget is already exhausted before this key makes a single
    call, so a key with no budget of its own must be rejected promptly. The small
    bounded retry only absorbs spend-propagation lag between the two keys; it is far
    below the spend a key-scoped budget would need to accumulate to block itself, so
    a block here can only come from the shared user budget."""
    for _ in range(SECOND_KEY_BLOCK_ATTEMPTS):
        result = _call(client, key)
        if is_budget_block(result):
            return
        require_successful_call(result)
        time.sleep(2)
    pytest.fail(
        f"{subject} was not blocked by the shared user budget within {SECOND_KEY_BLOCK_ATTEMPTS} calls"
    )


class TestUserBudgetAcrossKeys:
    @pytest.mark.covers("quota_management.budget.internal_user.enforced_across_keys")
    def test_user_budget_blocks_a_second_key(self, client: BudgetClient, resources: ResourceManager) -> None:
        user_id = client.create_user(max_budget=TINY_CAP)
        resources.defer(lambda: client.delete_user(user_id))

        first_key = client.generate_key(user_id=user_id)
        resources.defer(lambda: client.delete_key(first_key))
        second_key = client.generate_key(user_id=user_id)
        resources.defer(lambda: client.delete_key(second_key))

        _drive_to_block(client, first_key, "the first key")
        _expect_prompt_block(client, second_key, "the second key")

        deadline = time.monotonic() + RECORDED_SPEND_DEADLINE_SECONDS
        while time.monotonic() < deadline:
            info = client.user_info(user_id)
            if info is not None and (info.spend or 0.0) >= TINY_CAP:
                return
            time.sleep(5)
        pytest.fail(f"user spend never reached the {TINY_CAP} cap in the recorded state")
