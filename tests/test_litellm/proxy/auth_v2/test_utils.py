from __future__ import annotations

from litellm.proxy._types import LiteLLM_TeamTable, Member
from litellm.proxy.auth_v2.utils import db_team_to_scim


def test_db_team_to_scim_carries_members():
    team = LiteLLM_TeamTable(
        team_id="team-eng",
        team_alias="eng",
        members_with_roles=[
            Member(user_id="u-1", role="user"),
            Member(user_id="u-2", role="admin"),
        ],
    )

    group = db_team_to_scim(team)

    assert group.id == "team-eng"
    assert group.display_name == "eng"
    assert [member.value for member in group.members] == ["u-1", "u-2"]


def test_db_team_to_scim_without_members_omits_member_list():
    team = LiteLLM_TeamTable(team_id="team-solo", team_alias="solo")

    group = db_team_to_scim(team)

    assert group.id == "team-solo"
    assert group.members is None
