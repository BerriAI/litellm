import pytest

from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_VerificationTokenView,
    Member,
    UserAPIKeyAuth,
)
from litellm.models.team import LiteLLM_ModelTable
from litellm.proxy.auth.team_grants import team_grants, team_model_aliases

TEAM_ID = "team-grants"
USER_ID = "user-in-team"
ALIASES = {"fast": "gpt-4o-mini", "smart": "gpt-4o"}


def _alias_table(model_aliases) -> LiteLLM_ModelTable:
    return LiteLLM_ModelTable(model_aliases=model_aliases, created_by="admin", updated_by="admin")


def _full_team(model_aliases=ALIASES) -> LiteLLM_TeamTable:
    return LiteLLM_TeamTable(
        team_id=TEAM_ID,
        team_alias="grants-team",
        tpm_limit=1000,
        rpm_limit=10,
        max_budget=50.0,
        soft_budget=25.0,
        spend=12.5,
        models=["gpt-4o", "gpt-4o-mini"],
        blocked=True,
        metadata={"tier": "gold"},
        litellm_model_table=_alias_table(model_aliases),
        object_permission_id="op-1",
        object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="op-1", mcp_servers=["mcp-a"]),
        members_with_roles=[
            Member(user_id="someone-else", role="user"),
            Member(user_id=USER_ID, role="admin"),
        ],
    )


def _membership() -> LiteLLM_TeamMembership:
    return LiteLLM_TeamMembership(
        user_id=USER_ID,
        team_id=TEAM_ID,
        spend=3.25,
        litellm_budget_table=LiteLLM_BudgetTable(tpm_limit=500, rpm_limit=5),
    )


def test_team_grants_cover_every_team_field_the_key_path_gets():
    """Class guard for LIT-5858 and its siblings: every ``team_*`` column the combined-view SQL hands the
    virtual-key path must come out of the projection too, with the team's actual value, so adding a column
    to ``LiteLLM_VerificationTokenView`` without teaching ``team_grants`` fails here instead of in prod."""
    team = _full_team()
    grants = team_grants(team_object=team, team_membership=_membership(), user_id=USER_ID)
    token = UserAPIKeyAuth(team_id=TEAM_ID, **grants)

    view_team_fields = {name for name in LiteLLM_VerificationTokenView.model_fields if name.startswith("team_")}
    assert view_team_fields - {"team_id"} <= set(grants)
    assert all(grants[name] is not None for name in view_team_fields - {"team_id"})

    assert token.team_alias == "grants-team"
    assert token.team_tpm_limit == 1000
    assert token.team_rpm_limit == 10
    assert token.team_max_budget == 50.0
    assert token.team_soft_budget == 25.0
    assert token.team_spend == 12.5
    assert token.team_models == ["gpt-4o", "gpt-4o-mini"]
    assert token.team_blocked is True
    assert token.team_metadata == {"tier": "gold"}
    assert token.team_model_aliases == ALIASES
    assert token.team_object_permission_id == "op-1"
    assert token.team_object_permission is not None
    assert token.team_object_permission.mcp_servers == ["mcp-a"]
    assert token.team_member == Member(user_id=USER_ID, role="admin")
    assert token.team_member_spend == 3.25
    assert token.team_member_tpm_limit == 500
    assert token.team_member_rpm_limit == 5


def test_team_grants_without_team_leave_token_defaults():
    token = UserAPIKeyAuth(**team_grants(team_object=None, team_membership=None, user_id=USER_ID))
    assert token == UserAPIKeyAuth()


@pytest.mark.parametrize(
    "stored_aliases",
    [ALIASES, '{"fast": "gpt-4o-mini", "smart": "gpt-4o"}'],
    ids=["json-object", "json-string-as-written-by-team-new"],
)
def test_team_model_aliases_decode_both_storage_shapes(stored_aliases):
    team = _full_team(model_aliases=stored_aliases)
    assert team_model_aliases(team) == ALIASES
    assert team_grants(team_object=team, team_membership=None, user_id=None)["team_model_aliases"] == ALIASES


@pytest.mark.parametrize("stored_aliases", [None, "not json", '["a", "b"]', {"fast": 3}], ids=str)
def test_team_model_aliases_treat_unusable_column_as_no_aliases(stored_aliases):
    team = _full_team(model_aliases=stored_aliases)
    assert team_model_aliases(team) is None
    assert team_grants(team_object=team, team_membership=None, user_id=None)["team_model_aliases"] is None


def test_team_model_aliases_none_without_relation_loaded():
    team = _full_team()
    team.litellm_model_table = None
    assert team_model_aliases(team) is None
    assert team_model_aliases(None) is None


def test_team_member_is_the_callers_row_only():
    team = _full_team()
    assert team_grants(team_object=team, team_membership=None, user_id="someone-else")["team_member"] == Member(
        user_id="someone-else", role="user"
    )
    assert team_grants(team_object=team, team_membership=None, user_id="stranger")["team_member"] is None
    assert team_grants(team_object=team, team_membership=None, user_id=None)["team_member"] is None


def test_membership_limits_absent_without_membership_row():
    grants = team_grants(team_object=_full_team(), team_membership=None, user_id=USER_ID)
    assert grants["team_member_spend"] is None
    assert grants["team_member_tpm_limit"] is None
    assert grants["team_member_rpm_limit"] is None
