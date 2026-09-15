"""Auth-resolved budget state rides on ``UserAPIKeyAuth`` and round-trips through request metadata."""

from datetime import datetime, timezone

from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_OrganizationTable,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.spend_tracking.carried_budget_state import (
    carried_budget_metadata,
    carry_organization_budget_state,
    carry_team_and_user_budget_state,
)
from litellm.types.proxy.carried_budget_state import (
    KeyBudgetSnapshot,
    OrgBudgetSnapshot,
    TeamBudgetSnapshot,
    UserBudgetSnapshot,
)

RESET_AT = datetime(2026, 10, 1, 12, 30, tzinfo=timezone.utc)


def test_team_and_user_state_round_trips_through_metadata():
    token = UserAPIKeyAuth(token="hashed", team_id="t1", user_id="u1")
    carry_team_and_user_budget_state(
        valid_token=token,
        team_object=LiteLLM_TeamTable(team_id="t1", budget_reset_at=RESET_AT, max_budget=300.0),
        user_object=LiteLLM_UserTable(user_id="u1", budget_reset_at=None, max_budget=None, user_alias="Alice"),
    )

    metadata = dict(carried_budget_metadata(token))

    assert metadata == {
        "user_api_key_team_budget_reset_at": RESET_AT.isoformat().replace("+00:00", "Z"),
        "user_api_key_team_table_max_budget": 300.0,
        "user_api_key_user_budget_reset_at": None,
        "user_api_key_user_table_max_budget": None,
        "user_api_key_user_alias": "Alice",
    }
    assert TeamBudgetSnapshot.from_metadata(metadata) == TeamBudgetSnapshot(budget_reset_at=RESET_AT, max_budget=300.0)
    assert UserBudgetSnapshot.from_metadata(metadata) == UserBudgetSnapshot(
        budget_reset_at=None, max_budget=None, user_alias="Alice"
    )


def test_missing_objects_leave_no_metadata_and_no_snapshot():
    token = UserAPIKeyAuth(token="hashed", team_id="t1", user_id="u1")
    carry_team_and_user_budget_state(valid_token=token, team_object=None, user_object=None)

    assert dict(carried_budget_metadata(token)) == {}
    assert TeamBudgetSnapshot.from_metadata({}) is None
    assert UserBudgetSnapshot.from_metadata({"user_api_key_user_alias": "Alice"}) is None
    assert OrgBudgetSnapshot.from_metadata({"user_api_key_org_spend": 1.0}) is None
    assert KeyBudgetSnapshot.from_metadata({}) is None


def test_organization_state_carries_alias_spend_and_max_budget():
    token = UserAPIKeyAuth(token="hashed", org_id="o1")
    org = LiteLLM_OrganizationTable(
        organization_id="o1",
        organization_alias="platform-org",
        budget_id="b1",
        created_by="admin",
        updated_by="admin",
        spend=12.5,
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=100.0),
    )

    carry_organization_budget_state(valid_token=token, org_table=org)

    assert token.organization_alias == "platform-org"
    assert OrgBudgetSnapshot.from_metadata(carried_budget_metadata(token)) == OrgBudgetSnapshot(
        spend=12.5, max_budget=100.0
    )


def test_organization_without_budget_table_carries_no_cap():
    token = UserAPIKeyAuth(token="hashed", org_id="o1")
    org = LiteLLM_OrganizationTable(
        organization_id="o1",
        budget_id="b1",
        created_by="admin",
        updated_by="admin",
        spend=3.0,
    )

    carry_organization_budget_state(valid_token=token, org_table=org)

    assert token.org_budget_snapshot == OrgBudgetSnapshot(spend=3.0, max_budget=None)


def test_key_snapshot_parses_the_iso_string_auth_metadata_writes():
    assert KeyBudgetSnapshot.from_metadata({"user_api_key_budget_reset_at": RESET_AT.isoformat()}) == KeyBudgetSnapshot(
        budget_reset_at=RESET_AT
    )
    assert KeyBudgetSnapshot.from_metadata({"user_api_key_budget_reset_at": None}) == KeyBudgetSnapshot(
        budget_reset_at=None
    )


def test_snapshots_never_reach_the_serialized_token():
    token = UserAPIKeyAuth(token="hashed", team_id="t1", user_id="u1", org_id="o1")
    carry_team_and_user_budget_state(
        valid_token=token,
        team_object=LiteLLM_TeamTable(team_id="t1", budget_reset_at=RESET_AT),
        user_object=LiteLLM_UserTable(user_id="u1", user_alias="Alice"),
    )
    token.org_budget_snapshot = OrgBudgetSnapshot(spend=1.0, max_budget=2.0)

    dumped = token.model_dump()

    assert "team_budget_snapshot" not in dumped
    assert "user_budget_snapshot" not in dumped
    assert "org_budget_snapshot" not in dumped
    assert UserAPIKeyAuth(**dumped).team_budget_snapshot is None
