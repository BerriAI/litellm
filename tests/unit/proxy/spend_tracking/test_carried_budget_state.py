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
    carry_committed_budget_spend,
    carry_organization_budget_state,
    carry_team_and_user_budget_state,
)
from litellm.types.proxy.carried_budget_state import (
    CommittedBudgetSpend,
    EntitySpendCounterKeys,
    KeyBudgetSnapshot,
    OrgBudgetSnapshot,
    TeamBudgetSnapshot,
    UserBudgetSnapshot,
)

RESET_AT = datetime(2026, 10, 1, 12, 30, tzinfo=timezone.utc)


@pytest.mark.parametrize("entity_type", ["user", "customer_member"])
def test_personal_counter_identity_keeps_the_exact_reset_instant(entity_type):
    from litellm.types.proxy.carried_budget_state import PersonBudgetWindow

    window = PersonBudgetWindow(
        entity_type=entity_type, entity_id="person:one/é", budget_duration="1mo",
        window_start=datetime.fromisoformat("2026-11-01T00:30:00-05:00"),
        reset_at=datetime.fromisoformat("2026-12-01T00:30:00-05:00"), max_budget=10,
    )
    same_instant = window.model_copy(update={"window_start": datetime.fromisoformat("2026-11-01T05:30:00+00:00")})
    midnight = window.model_copy(update={"window_start": datetime(2026, 11, 1, tzinfo=timezone.utc)})

    assert window.counter_key == (
        f"spend:person:{entity_type}:person%3Aone%2F%C3%A9:window:1mo@2026-11-01T05:30:00+00:00"
    )
    assert window.counter_key == same_instant.counter_key
    assert window.counter_key != midnight.counter_key
    assert window.accounting_window() == {
        "entity_type": entity_type, "entity_id": "person:one/é",
        "window_duration": "1mo@2026-11-01T05:30:00+00:00",
        "window_start": "2026-11-01T05:30:00+00:00",
    }


def test_deferred_creator_windows_and_customer_usage_survive_json_and_auth_reconstruction():
    import json

    from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth
    from litellm.proxy.spend_tracking.carried_budget_state import (
        creator_budget_metadata, person_budget_windows_from_kwargs,
        restore_creator_budget_metadata, stored_budget_metadata,
    )
    from litellm.types.proxy.carried_budget_state import PersonBudgetWindow

    creator = UserAPIKeyAuth(user_id="creator")
    creator.person_budget_windows = (PersonBudgetWindow(
        entity_type="customer_member", entity_id="member", budget_duration="12mo",
        window_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        reset_at=datetime(2027, 1, 1, tzinfo=timezone.utc), max_budget=100,
    ),)
    creator.customer_hierarchy = CustomerHierarchyAuth(
        customer_org_id="org", customer_team_id="team", customer_member_id="member", scopes=(),
    )
    persisted = json.loads(json.dumps(creator_budget_metadata(creator)))
    restored = UserAPIKeyAuth(person_budget_windows=creator.person_budget_windows)
    assert restored.person_budget_windows == ()
    restore_creator_budget_metadata(restored, stored_budget_metadata(persisted))

    assert restored.person_budget_windows == creator.person_budget_windows
    assert restored.customer_hierarchy == creator.customer_hierarchy
    assert person_budget_windows_from_kwargs({"standard_logging_object": {"metadata": persisted}}) == (
        creator.person_budget_windows
    )
    assert person_budget_windows_from_kwargs({"litellm_params": {"litellm_metadata": persisted}}) == (
        creator.person_budget_windows
    )
    assert stored_budget_metadata(None) == {}


@pytest.mark.parametrize("missing", ["user_api_key_customer_member_id", "user_api_key_customer_team_id"])
def test_deferred_snapshot_drops_malformed_customer_ids(missing):
    from litellm.proxy.spend_tracking.carried_budget_state import restore_creator_budget_metadata, stored_budget_metadata

    metadata = {"user_api_key_customer_org_id": "org", "user_api_key_customer_team_id": "team",
                "user_api_key_customer_member_id": "member", missing: None}
    persisted = stored_budget_metadata(metadata)
    auth = UserAPIKeyAuth()
    restore_creator_budget_metadata(auth, persisted)

    assert missing not in persisted
    assert persisted["user_api_key_customer_org_id"] == "org"
    assert auth.customer_hierarchy is None


def test_team_and_user_state_round_trips_through_metadata():
    token = UserAPIKeyAuth(token="hashed", team_id="t1", user_id="u1")
    carry_team_and_user_budget_state(
        valid_token=token,
        team_object=LiteLLM_TeamTable(team_id="t1", budget_reset_at=RESET_AT, max_budget=300.0),
        user_object=LiteLLM_UserTable(user_id="u1", budget_reset_at=None, max_budget=None, user_alias="Alice"),
    )

    metadata = dict(carried_budget_metadata(token))

    assert metadata == {
        "user_api_key_person_budget_windows": [],
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


def test_missing_objects_clear_personal_metadata_and_leave_no_snapshot():
    token = UserAPIKeyAuth(token="hashed", team_id="t1", user_id="u1")
    carry_team_and_user_budget_state(valid_token=token, team_object=None, user_object=None)

    assert dict(carried_budget_metadata(token)) == {"user_api_key_person_budget_windows": []}
    assert TeamBudgetSnapshot.from_metadata({}) is None
    assert UserBudgetSnapshot.from_metadata({"user_api_key_user_alias": "Alice"}) is None
    assert OrgBudgetSnapshot.from_metadata({"user_api_key_org_spend": 1.0}) is None
    assert KeyBudgetSnapshot.from_metadata({}) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["/v1/chat/completions", "/v1/responses"])
@pytest.mark.parametrize("serialized", [False, True])
async def test_request_enrichment_clears_forged_personal_windows_for_unassigned_auth(monkeypatch, route, serialized):
    import json
    from starlette.requests import Request
    from litellm.proxy import proxy_server
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request, metadata_variable_name_for_request
    from litellm.proxy.spend_tracking.carried_budget_state import creator_budget_metadata, person_budget_windows_from_kwargs
    from litellm.types.proxy.carried_budget_state import PERSON_BUDGET_WINDOWS_METADATA_KEY, PersonBudgetWindow

    fake = PersonBudgetWindow(
        entity_type="user", entity_id="victim", budget_duration="1mo",
        window_start=datetime(2026, 11, 1, tzinfo=timezone.utc),
        reset_at=datetime(2026, 12, 1, tzinfo=timezone.utc), max_budget=100,
    )
    metadata = {PERSON_BUDGET_WINDOWS_METADATA_KEY: [fake.model_dump(mode="json")]}
    data = {
        "model": "gpt-4o",
        "metadata": json.dumps(metadata) if serialized else dict(metadata),
        "litellm_metadata": json.dumps(metadata) if serialized else dict(metadata),
        **({"input": "hello"} if route.endswith("responses") else {"messages": [{"role": "user", "content": "hello"}]}),
    }
    request = Request({
        "type": "http", "method": "POST", "path": route, "scheme": "http",
        "server": ("localhost", 80), "client": ("127.0.0.1", 1000),
        "headers": [(b"content-type", b"application/json")], "query_string": b"",
    })
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "premium_user", False)
    auth = UserAPIKeyAuth(user_id="requester")
    enriched = await add_litellm_data_to_request(
        data=data, request=request, user_api_key_dict=auth,
        proxy_config=proxy_server.ProxyConfig(), general_settings={},
    )

    bucket = metadata_variable_name_for_request(route, enriched)
    assert enriched[bucket][PERSON_BUDGET_WINDOWS_METADATA_KEY] == []
    assert person_budget_windows_from_kwargs({"litellm_params": enriched}) == ()
    assert creator_budget_metadata(auth) == {}


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
    metadata_without_committed_spend = dict(carried_budget_metadata(token))
    token.committed_budget_spend = CommittedBudgetSpend(key=1.0, team=2.0, user=3.0, org=4.0, end_user=5.0)

    dumped = token.model_dump()

    assert "team_budget_snapshot" not in dumped
    assert "user_budget_snapshot" not in dumped
    assert "org_budget_snapshot" not in dumped
    assert "committed_budget_spend" not in dumped
    assert dict(carried_budget_metadata(token)) == metadata_without_committed_spend
    assert UserAPIKeyAuth(**dumped).team_budget_snapshot is None


def _committed_reader(committed: CommittedBudgetSpend):
    reads: list[EntitySpendCounterKeys] = []

    async def read_committed(counter_keys: EntitySpendCounterKeys) -> CommittedBudgetSpend:
        reads.append(counter_keys)
        return committed

    return read_committed, reads


@pytest.mark.asyncio
async def test_committed_spend_is_read_once_for_every_budgeted_entity():
    token = UserAPIKeyAuth(token="hashed", team_id="t1", user_id="u1", org_id="o1", spend=1.0, max_budget=10.0)
    token.org_budget_snapshot = OrgBudgetSnapshot(spend=40.0, max_budget=500.0)
    committed = CommittedBudgetSpend(key=1.3, team=20.3, user=5.3, org=40.3, end_user=0.8)
    read_committed, reads = _committed_reader(committed)

    await carry_committed_budget_spend(
        valid_token=token,
        team_object=LiteLLM_TeamTable(team_id="t1", max_budget=300.0, spend=20.0),
        user_object=LiteLLM_UserTable(user_id="u1", max_budget=50.0, spend=5.0),
        end_user_object=LiteLLM_EndUserTable(
            user_id="cust-1", blocked=False, spend=0.5, litellm_budget_table=LiteLLM_BudgetTable(max_budget=3.0)
        ),
        read_committed=read_committed,
    )

    assert reads == [
        EntitySpendCounterKeys(
            key="spend:key:hashed",
            team="spend:team:t1",
            user="spend:user:u1",
            org="spend:org:o1",
            end_user="spend:end_user:cust-1",
        )
    ]
    assert token.committed_budget_spend == committed
    assert (token.spend, token.team_spend) == (1.0, None)


@pytest.mark.asyncio
async def test_route_without_a_budget_check_carries_the_counters_it_charges_and_reads_nothing():
    token = UserAPIKeyAuth(token="hashed", max_budget=10.0)

    await carry_committed_budget_spend(
        valid_token=token, team_object=None, user_object=None, end_user_object=None, read_committed=None
    )

    assert token.committed_budget_spend == CommittedBudgetSpend(
        counter_keys=EntitySpendCounterKeys(key="spend:key:hashed")
    )


@pytest.mark.asyncio
async def test_committed_spend_skips_entities_without_a_max_budget():
    token = UserAPIKeyAuth(token="hashed", team_id="t1", user_id="u1", org_id="o1", spend=1.0)
    token.org_budget_snapshot = OrgBudgetSnapshot(spend=40.0, max_budget=None)
    read_committed, reads = _committed_reader(CommittedBudgetSpend())

    await carry_committed_budget_spend(
        valid_token=token,
        team_object=LiteLLM_TeamTable(team_id="t1", spend=20.0),
        user_object=LiteLLM_UserTable(user_id="u1", spend=5.0),
        end_user_object=LiteLLM_EndUserTable(
            user_id="cust-1", blocked=False, spend=0.5, litellm_budget_table=LiteLLM_BudgetTable(max_budget=None)
        ),
        read_committed=read_committed,
    )

    assert reads == [EntitySpendCounterKeys()]
    assert token.committed_budget_spend == CommittedBudgetSpend()


@pytest.mark.asyncio
async def test_committed_spend_reads_the_team_org_when_the_key_has_none():
    token = UserAPIKeyAuth(token="hashed", team_id="t1")
    token.org_budget_snapshot = OrgBudgetSnapshot(spend=40.0, max_budget=500.0)
    read_committed, reads = _committed_reader(CommittedBudgetSpend(org=41.0))

    await carry_committed_budget_spend(
        valid_token=token,
        team_object=LiteLLM_TeamTable(team_id="t1", organization_id="team-org"),
        user_object=None,
        end_user_object=None,
        read_committed=read_committed,
    )

    assert reads == [EntitySpendCounterKeys(org="spend:org:team-org")]
    assert token.committed_budget_spend == CommittedBudgetSpend(org=41.0)
