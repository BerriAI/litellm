from __future__ import annotations

from datetime import datetime, timezone
from typing import Final

import pytest

from litellm.proxy._types import TeamMemberBudgetSource
from litellm.proxy.management.teams.repository import TeamMemberRow
from litellm.proxy.management.teams.service import TEAM_MEMBERS_LIST_SPEC, member_budget_source


@pytest.mark.parametrize(
    ("budget_id", "team_default_budget_id", "expected"),
    [
        pytest.param("own", "default", "custom", id="own-row-beside-a-default"),
        pytest.param("own", None, "custom", id="own-row-without-a-default"),
        pytest.param("default", "default", "team_default", id="linked-to-the-default-row"),
        pytest.param(None, "default", "team_default", id="no-row-follows-the-default"),
        pytest.param(None, None, "none", id="no-row-no-default"),
    ],
)
def test_member_budget_source(
    budget_id: str | None, team_default_budget_id: str | None, expected: TeamMemberBudgetSource
) -> None:
    assert member_budget_source(budget_id, team_default_budget_id) == expected


def test_list_item_carries_the_member_row_and_derives_its_budget_source() -> None:
    reset_at: Final = datetime(2026, 11, 1, tzinfo=timezone.utc)
    row: Final = TeamMemberRow(
        user_id="u-1",
        user_email="u1@example.com",
        user_alias="Una",
        role="admin",
        spend=1.5,
        total_spend=7.25,
        budget_id="own",
        team_default_budget_id="default",
        max_budget_in_team=20.0,
        budget_duration="30d",
        budget_reset_at=reset_at,
        tpm_limit=1000,
        rpm_limit=10,
        allowed_models=("gpt-a", "gpt-b"),
    )

    item: Final = TEAM_MEMBERS_LIST_SPEC.serialize(row)

    assert item.model_dump() == {
        "user_id": "u-1",
        "user_email": "u1@example.com",
        "user_alias": "Una",
        "role": "admin",
        "spend": 1.5,
        "total_spend": 7.25,
        "budget_id": "own",
        "budget_source": "custom",
        "max_budget_in_team": 20.0,
        "budget_duration": "30d",
        "budget_reset_at": reset_at,
        "tpm_limit": 1000,
        "rpm_limit": 10,
        "allowed_models": ("gpt-a", "gpt-b"),
    }
