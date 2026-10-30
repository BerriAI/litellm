"""Auth-resolved budget state rides on ``UserAPIKeyAuth`` and round-trips through request metadata."""

from datetime import datetime, timezone

import pytest

from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_OrganizationTable,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.spend_tracking.carried_budget_state import (
    carried_budget_metadata,
    carry_live_budget_spend,
    carry_organization_budget_state,
    carry_team_and_user_budget_state,
)
from litellm.types.proxy.carried_budget_state import (
    KeyBudgetSnapshot,
    LiveBudgetSpend,
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


def test_team_model_max_budget_rides_on_the_token():
    """The team's per-model caps must reach the token, or the auth check and the spend hook never see them."""
    token = UserAPIKeyAuth(token="hashed", team_id="t1")
    team_model_max_budget = {"gpt-4o": {"max_budget": 5.0, "budget_duration": "1d"}}
    carry_team_and_user_budget_state(
        valid_token=token,
        team_object=LiteLLM_TeamTable(team_id="t1", model_max_budget=team_model_max_budget),
        user_object=None,
    )

    assert token.team_model_max_budget == team_model_max_budget


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
    token.live_budget_spend = LiveBudgetSpend(key=1.0)

    dumped = token.model_dump()

    assert "team_budget_snapshot" not in dumped
    assert "user_budget_snapshot" not in dumped
    assert "org_budget_snapshot" not in dumped
    assert "live_budget_spend" not in dumped
    assert UserAPIKeyAuth(**dumped).team_budget_snapshot is None


def _counter_reader(counters: dict[str, float]):
    reads: list[str] = []

    async def read_spend(counter_key: str, fallback_spend: float) -> float:
        reads.append(counter_key)
        return counters.get(counter_key, fallback_spend)

    return read_spend, reads


def _budgeted_token() -> UserAPIKeyAuth:
    token = UserAPIKeyAuth(token="hashed", team_id="t1", user_id="u1", org_id="o1", spend=1.0, max_budget=10.0)
    token.org_budget_snapshot = OrgBudgetSnapshot(spend=40.0, max_budget=500.0)
    return token


@pytest.mark.asyncio
async def test_live_budget_spend_reads_every_budgeted_counter_and_leaves_the_header_spend_alone():
    """Each counter holds committed spend plus another request's 0.33 reservation, which only the gauges see."""
    token = _budgeted_token()
    read_spend, _ = _counter_reader(
        {
            "spend:key:hashed": 1.63,
            "spend:team:t1": 20.63,
            "spend:user:u1": 5.63,
            "spend:org:o1": 40.63,
            "spend:end_user:cust-1": 0.83,
        }
    )

    await carry_live_budget_spend(
        valid_token=token,
        team_object=LiteLLM_TeamTable(team_id="t1", max_budget=300.0, spend=20.0),
        user_object=LiteLLM_UserTable(user_id="u1", max_budget=50.0, spend=5.0),
        end_user_object=LiteLLM_EndUserTable(
            user_id="cust-1", blocked=False, spend=0.5, litellm_budget_table=LiteLLM_BudgetTable(max_budget=3.0)
        ),
        read_spend=read_spend,
    )

    expected = LiveBudgetSpend(key=1.63, team=20.63, user=5.63, org=40.63, end_user=0.83)
    assert token.live_budget_spend == expected
    assert LiveBudgetSpend.from_metadata(carried_budget_metadata(token)) == expected
    assert (token.spend, token.team_spend) == (1.0, None)


@pytest.mark.asyncio
async def test_live_budget_spend_never_reads_below_the_spend_auth_loaded():
    token = _budgeted_token()
    read_spend, _ = _counter_reader({"spend:key:hashed": 0.2, "spend:org:o1": 12.0})

    await carry_live_budget_spend(
        valid_token=token, team_object=None, user_object=None, end_user_object=None, read_spend=read_spend
    )

    assert token.live_budget_spend == LiveBudgetSpend(key=1.0, org=40.0)


@pytest.mark.asyncio
async def test_live_budget_spend_skips_entities_without_a_max_budget():
    token = UserAPIKeyAuth(token="hashed", team_id="t1", user_id="u1", org_id="o1", spend=1.0)
    token.org_budget_snapshot = OrgBudgetSnapshot(spend=40.0, max_budget=None)
    read_spend, reads = _counter_reader({})

    await carry_live_budget_spend(
        valid_token=token,
        team_object=LiteLLM_TeamTable(team_id="t1", spend=20.0),
        user_object=LiteLLM_UserTable(user_id="u1", spend=5.0),
        end_user_object=LiteLLM_EndUserTable(user_id="cust-1", blocked=False, spend=0.5),
        read_spend=read_spend,
    )

    assert reads == []
    assert token.live_budget_spend == LiveBudgetSpend()


@pytest.mark.asyncio
async def test_live_budget_spend_reads_the_team_org_when_the_key_has_none():
    token = UserAPIKeyAuth(token="hashed", team_id="t1")
    token.org_budget_snapshot = OrgBudgetSnapshot(spend=40.0, max_budget=500.0)
    read_spend, _ = _counter_reader({"spend:org:team-org": 41.0})

    await carry_live_budget_spend(
        valid_token=token,
        team_object=LiteLLM_TeamTable(team_id="t1", organization_id="team-org"),
        user_object=None,
        end_user_object=None,
        read_spend=read_spend,
    )

    assert token.live_budget_spend == LiveBudgetSpend(org=41.0)
