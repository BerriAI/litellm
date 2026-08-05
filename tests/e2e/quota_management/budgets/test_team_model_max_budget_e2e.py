"""Live e2e: a team's `model_max_budget` is one shared per-model cap for every
team key, and a key carrying its own entry for that model is exempt from it."""

import time

import pytest

from budget_client import BudgetClient, is_budget_block, model_budget
from e2e_config import unique_marker
from e2e_http import StreamingResponse, require_successful_call
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e

CAPPED_MODEL = "claude-haiku-4-5"
FREE_MODEL = "gemini-2.5-flash"
BLOCK_DEADLINE_SECONDS = 60
SPEND_FLUSH_SECONDS = 2


def _call(client: BudgetClient, key: str, model: str) -> StreamingResponse:
    result = client.chat(key, model, f"hi {unique_marker()}", max_tokens=16)
    if not result.ok and not is_budget_block(result):
        require_successful_call(result)
    return result


def _assert_team_model_block(result: StreamingResponse, team_id: str) -> None:
    assert result.status_code == 429, f"HTTP {result.status_code} {result.body}"
    assert team_id in result.body, result.body
    assert f"exceeded budget for model={CAPPED_MODEL}" in result.body, result.body


def _drive_to_team_block(client: BudgetClient, key: str, team_id: str) -> None:
    deadline = time.monotonic() + BLOCK_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        result = _call(client, key, CAPPED_MODEL)
        if is_budget_block(result):
            _assert_team_model_block(result, team_id)
            return
        time.sleep(1)
    pytest.fail(f"team model_max_budget on {CAPPED_MODEL} never enforced")


@pytest.mark.covers("quota_management.budget.team_model_max.enforced_across_keys")
def test_team_model_cap_blocks_sibling_key(client: BudgetClient, resources: ResourceManager) -> None:
    # Step 0: admin creates the team (claude capped tiny, gemini roomy) and two plain team keys
    team_id = client.create_team(
        alias=f"e2e-team-model-max-{unique_marker()}",
        model_max_budget={
            **model_budget(CAPPED_MODEL, 1e-6),
            **model_budget(FREE_MODEL, 1000.0),
        },
    )
    resources.defer(lambda: client.delete_team(team_id))
    spender = client.generate_key(team_id=team_id)
    resources.defer(lambda: client.delete_key(spender))
    bystander = client.generate_key(team_id=team_id)
    resources.defer(lambda: client.delete_key(bystander))

    # Step 1: Alice's key burns through the claude cap
    _drive_to_team_block(client, spender, team_id)

    # Step 2: Bob's untouched key is refused claude on its first try
    first = _call(client, bystander, CAPPED_MODEL)
    assert is_budget_block(first), (
        f"sibling team key served {CAPPED_MODEL} after the team cap was exhausted: "
        f"HTTP {first.status_code} {first.body}"
    )
    _assert_team_model_block(first, team_id)

    # Step 3: Bob's same key still gets gemini answers
    other = _call(client, bystander, FREE_MODEL)
    assert not is_budget_block(other), f"{FREE_MODEL} was blocked by {CAPPED_MODEL}'s team cap: {other.body}"
    require_successful_call(other)


@pytest.mark.covers("quota_management.budget.team_model_max.key_override_wins")
def test_key_override_exempts_from_team_cap(client: BudgetClient, resources: ResourceManager) -> None:
    # Step 0: admin creates the capped team, Carol's key with its own big claude budget, Dave's plain key
    team_id = client.create_team(
        alias=f"e2e-team-model-max-override-{unique_marker()}",
        model_max_budget=model_budget(CAPPED_MODEL, 1e-6),
    )
    resources.defer(lambda: client.delete_team(team_id))
    override = client.generate_key(team_id=team_id, model_max_budget=model_budget(CAPPED_MODEL, 1000.0))
    resources.defer(lambda: client.delete_key(override))
    plain = client.generate_key(team_id=team_id)
    resources.defer(lambda: client.delete_key(plain))

    # Step 1: Carol keeps getting claude answers past the team cap
    require_successful_call(_call(client, override, CAPPED_MODEL))
    time.sleep(SPEND_FLUSH_SECONDS)
    second = _call(client, override, CAPPED_MODEL)
    assert not is_budget_block(second), f"override key hit the team cap it should be exempt from: {second.body}"
    require_successful_call(second)
    time.sleep(SPEND_FLUSH_SECONDS)

    # Step 2: Dave's first claude call still succeeds; Carol spent none of the team cap
    fresh = _call(client, plain, CAPPED_MODEL)
    assert not is_budget_block(fresh), f"override key's spend drove the shared team counter: {fresh.body}"
    require_successful_call(fresh)

    # Step 3: Dave's own claude use exhausts the team cap and is refused
    _drive_to_team_block(client, plain, team_id)
