from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from litellm.models.team import LiteLLM_TeamTable, TeamMemberBudgetDuration, TeamMemberBudgetWindow
from litellm.models.team_membership import LiteLLM_TeamMembership
from litellm.proxy._types import NewTeamRequest, TeamMemberAddRequest, TeamMemberUpdateRequest, UpdateTeamRequest
from litellm.proxy.common_utils.team_member_budget import (
    MemberBudgetCounter,
    get_team_member_budget_windows,
    member_window_entity_id,
    parse_member_window_entity_id,
)


@pytest.mark.parametrize("amount", [-1, float("inf"), float("nan")])
def test_rejects_invalid_caps(amount: float) -> None:
    with pytest.raises(ValidationError):
        TeamMemberBudgetWindow(budget_duration=TeamMemberBudgetDuration.DAILY, max_budget=amount)


@pytest.mark.parametrize(
    "request_type, fields, field",
    [
        (NewTeamRequest, {}, "team_member_budget_limits"),
        (UpdateTeamRequest, {"team_id": "team"}, "team_member_budget_limits"),
        (TeamMemberAddRequest, {"team_id": "team", "member": {"user_id": "member", "role": "user"}}, "budget_limits"),
        (TeamMemberUpdateRequest, {"team_id": "team", "user_id": "member"}, "budget_limits"),
    ],
)
def test_rejects_duplicate_windows(request_type: type, fields: dict, field: str) -> None:
    duplicate = [{"budget_duration": "24h", "max_budget": 20}] * 2
    with pytest.raises(ValidationError, match="unique"):
        request_type(**fields, **{field: duplicate})


def test_override_and_clear_preserve_team_inheritance() -> None:
    defaults = [TeamMemberBudgetWindow(budget_duration=TeamMemberBudgetDuration.DAILY, max_budget=20)]
    custom = [TeamMemberBudgetWindow(budget_duration=TeamMemberBudgetDuration.DAILY, max_budget=0)]
    team = LiteLLM_TeamTable(team_id="team", team_member_budget_limits=defaults)
    member = LiteLLM_TeamMembership(user_id="member", team_id="team", budget_limits=custom)
    assert get_team_member_budget_windows(team, member) == tuple(custom)
    member.budget_limits = []
    assert get_team_member_budget_windows(team, member) == tuple(defaults)
    assert get_team_member_budget_windows(team, None) == tuple(defaults)
    assert get_team_member_budget_windows(LiteLLM_TeamTable(team_id="off"), None) == ()


def test_reset_boundaries_are_independent() -> None:
    before = datetime(2026, 10, 10, 23, 59, tzinfo=timezone.utc)
    after = datetime(2026, 10, 11, 0, 1, tzinfo=timezone.utc)
    daily = MemberBudgetCounter.for_member("member", "team", TeamMemberBudgetDuration.DAILY, before)
    monthly = MemberBudgetCounter.for_member("member", "team", TeamMemberBudgetDuration.MONTHLY, before)
    assert MemberBudgetCounter.for_member("member", "team", daily.duration, after).key != daily.key
    assert MemberBudgetCounter.for_member("member", "team", monthly.duration, after).key == monthly.key
    assert MemberBudgetCounter.for_member("other", "team", daily.duration, before).key != daily.key
    assert MemberBudgetCounter.for_member("member", "other", daily.duration, before).key != daily.key


def test_member_identity_cannot_collide_at_a_separator() -> None:
    first = member_window_entity_id("user:other", "team")
    second = member_window_entity_id("user", "other:team")
    assert first != second
    assert parse_member_window_entity_id(first) == ("user:other", "team")


@pytest.mark.parametrize("request_type", [NewTeamRequest, UpdateTeamRequest])
def test_default_windows_round_trip_through_prisma_json(request_type: type) -> None:
    import json
    from litellm.proxy.utils import PrismaClient

    data = request_type(
        team_id="team",
        team_member_budget_limits=[
            {"budget_duration": "24h", "max_budget": 20},
            {"budget_duration": "30d", "max_budget": 200},
        ],
    )
    client = object.__new__(PrismaClient)
    serialized = client.jsonify_team_object(data.model_dump(exclude_none=True))
    windows = json.loads(serialized["team_member_budget_limits"])
    assert windows == [{"budget_duration": "24h", "max_budget": 20}, {"budget_duration": "30d", "max_budget": 200}]
    restored = LiteLLM_TeamTable(team_id="team", team_member_budget_limits=serialized["team_member_budget_limits"])
    assert restored.team_member_budget_limits == data.team_member_budget_limits
