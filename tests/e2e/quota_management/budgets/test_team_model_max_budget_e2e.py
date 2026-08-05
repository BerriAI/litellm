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


def _call(client: BudgetClient, key: str, model: str) -> StreamingResponse:
    result = client.chat(key, model, f"hi {unique_marker()}", max_tokens=16)
    if not result.ok and not is_budget_block(result):
        require_successful_call(result)
    return result


def _drive_to_team_block(client: BudgetClient, key: str, team_id: str) -> None:
    deadline = time.monotonic() + BLOCK_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        result = _call(client, key, CAPPED_MODEL)
        if is_budget_block(result):
            assert team_id in result.body, result.body
            return
        time.sleep(1)
    pytest.fail(f"team model_max_budget on {CAPPED_MODEL} never enforced")


# User flow (TLDR^2)
# 1. Admin caps claude-haiku tiny on the team, leaves gemini roomy
# 2. Alice's team key burns through the claude cap
# 3. Bob's untouched team key is refused claude on its first try
# 4. Bob's same key still gets gemini answers
@pytest.mark.covers("quota_management.budget.team_model_max.enforced_across_keys")
def test_team_model_cap_blocks_sibling_key(client: BudgetClient, resources: ResourceManager) -> None:
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

    _drive_to_team_block(client, spender, team_id)

    first = _call(client, bystander, CAPPED_MODEL)
    assert is_budget_block(first), (
        f"sibling team key served {CAPPED_MODEL} after the team cap was exhausted: "
        f"HTTP {first.status_code} {first.body}"
    )
    assert team_id in first.body, first.body

    require_successful_call(_call(client, bystander, FREE_MODEL))


# User flow (TLDR^2)
# 1. Admin caps claude-haiku tiny on the team
# 2. Admin issues Carol a team key with its own big claude budget
# 3. Dave's plain team key exhausts the team cap and is refused
# 4. Carol keeps getting claude answers on the same team
@pytest.mark.covers("quota_management.budget.team_model_max.key_override_wins")
def test_key_override_exempts_from_team_cap(client: BudgetClient, resources: ResourceManager) -> None:
    team_id = client.create_team(
        alias=f"e2e-team-model-max-override-{unique_marker()}",
        model_max_budget=model_budget(CAPPED_MODEL, 1e-6),
    )
    resources.defer(lambda: client.delete_team(team_id))
    inherit = client.generate_key(team_id=team_id)
    resources.defer(lambda: client.delete_key(inherit))
    override = client.generate_key(team_id=team_id, model_max_budget=model_budget(CAPPED_MODEL, 1000.0))
    resources.defer(lambda: client.delete_key(override))

    _drive_to_team_block(client, inherit, team_id)

    require_successful_call(_call(client, override, CAPPED_MODEL))
