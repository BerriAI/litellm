import time
from datetime import datetime

import pytest

from budget_client import BudgetClient
from e2e_config import unique_marker
from e2e_http import require_successful_call
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e

MEMBER_BUDGET = 1.0 # default member budget is $50, we're testing with a smaller value

def _as_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_team_member_budget_reset_keeps_advancing(client: BudgetClient, resources: ResourceManager) -> None:
    team_id = client.create_team(alias=f"e2e-member-reset-{unique_marker()}", max_budget=100.0)
    resources.defer(lambda: client.delete_team(team_id))
    user_id = client.create_user(max_budget=100.0)
    resources.defer(lambda: client.delete_user(user_id))

    # add the member, then update them onto a short per-team budget window
    client.add_team_member(team_id, user_id, max_budget_in_team=MEMBER_BUDGET)
    client.update_team_member(team_id, user_id, max_budget_in_team=MEMBER_BUDGET, budget_duration="30s")

    scheduled = client.member_budget_reset_at(team_id, user_id)
    assert scheduled, "updating the member with a budget_duration set no budget_reset_at"
    first_reset = _as_datetime(scheduled)

    # the member can spend within the team while the window is live
    key = client.generate_key(team_id=team_id, user_id=user_id)
    resources.defer(lambda: client.delete_key(key))
    require_successful_call(client.chat(key, "claude-haiku-4-5", f"reset {unique_marker()}", max_tokens=16))

    # once the window elapses the reset job must move budget_reset_at forward; a job
    # that skips the member's budget row (the #25109 regression) leaves it pinned at
    # first_reset forever
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        time.sleep(5)
        current = client.member_budget_reset_at(team_id, user_id)
        if current and _as_datetime(current) > first_reset:
            return
    pytest.fail(f"member budget_reset_at never advanced past {first_reset.isoformat()} in 150s")
