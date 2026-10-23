import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from litellm.proxy._types import UpdateTeamRequest
from litellm.proxy.management_helpers.team_member_default_budget import plan_team_member_default_budget_write


def _plan(existing_budget_id: str | None = None, team_budget_duration: str | None = None, **fields: object):
    return plan_team_member_default_budget_write(
        data=UpdateTeamRequest(team_id="team-1", **fields),
        existing_budget_id=existing_budget_id,
        team_alias="My Team",
        team_budget_duration=team_budget_duration,
    )


def test_a_request_without_member_defaults_writes_nothing():
    assert _plan(existing_budget_id="default-1", max_budget=10.0) is None


def test_clearing_a_default_the_team_never_had_writes_nothing():
    assert _plan(team_member_budget=None) is None


def test_clearing_an_existing_default_updates_it_without_relinking_members():
    write = _plan(existing_budget_id="default-1", team_member_budget=None, team_member_model_max_budget=None)

    assert write is not None
    assert write.budget_id == "default-1"
    assert dict(write.update) == {"max_budget": None, "model_max_budget": {}}
    assert write.links_members is False


def test_a_first_default_gets_a_new_row_that_inherits_the_team_reset_period():
    before = datetime.now(timezone.utc)

    write = _plan(team_budget_duration="30d", team_member_budget=25.0)

    assert write is not None
    assert write.budget_id.startswith("team-My-Team-budget-")
    assert write.create["max_budget"] == 25.0
    assert write.create["budget_duration"] == "30d"
    reset_at = write.create["budget_reset_at"]
    assert isinstance(reset_at, datetime) and reset_at > before
    assert write.links_members is True


def test_an_explicit_null_reset_period_is_not_replaced_by_the_team_period():
    write = _plan(team_budget_duration="30d", team_member_budget=25.0, team_member_budget_duration=None)

    assert write is not None
    assert "budget_duration" not in write.create
    assert write.update["budget_duration"] is None
    assert write.update["budget_reset_at"] is None


def test_changing_the_reset_period_restarts_the_window():
    before = datetime.now(timezone.utc)

    write = _plan(existing_budget_id="default-1", team_member_budget_duration="7d")

    assert write is not None
    assert write.update["budget_duration"] == "7d"
    reset_at = write.update["budget_reset_at"]
    assert isinstance(reset_at, datetime) and reset_at > before


@pytest.mark.parametrize("team_member_budget", [None, 0.0])
def test_backfill_without_a_positive_default_in_the_same_request_is_rejected(team_member_budget):
    with pytest.raises(ValidationError, match="team_member_budget_backfill needs a positive team_member_budget"):
        UpdateTeamRequest(
            team_id="team-1", team_member_budget=team_member_budget, team_member_budget_backfill="raise_to_default"
        )


@pytest.mark.parametrize("backfill", [None, "none"])
def test_keeping_custom_budgets_needs_no_default(backfill):
    assert UpdateTeamRequest(team_id="team-1", team_member_budget_backfill=backfill).team_member_budget is None


def test_per_model_member_budgets_are_stored_as_plain_json():
    caps = {"gpt-4o": {"max_budget": 5.0, "budget_duration": "1d"}}

    write = _plan(existing_budget_id="default-1", team_member_model_max_budget=caps)

    assert write is not None
    assert json.loads(json.dumps(write.update["model_max_budget"]))["gpt-4o"]["max_budget"] == 5.0
