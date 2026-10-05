import os
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from pydantic import BaseModel, JsonValue
from redis import Redis


class _ObservedRequest(BaseModel):
    body: dict[str, JsonValue]


class _Observations(BaseModel):
    requests: list[_ObservedRequest]


class _ProxyError(BaseModel):
    type: str


class _ErrorResponse(BaseModel):
    error: _ProxyError


class _BudgetRecord(BaseModel):
    budget_id: str
    max_budget: float | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    budget_duration: str | None = None
    allowed_models: list[str] | None = None
    temp_budget_increase: float | None = None
    temp_budget_expiry: datetime | None = None


class _MemberBudgetState(BaseModel):
    user_id: str
    team_id: str
    budget_id: str | None = None
    spend: float | None = None
    max_budget: float | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    budget_duration: str | None = None
    allowed_models: list[str] | None = None
    temp_budget_increase: float | None = None
    temp_budget_expiry: datetime | None = None


class _TeamRosterMember(BaseModel):
    role: str
    user_id: str | None = None
    user_email: str | None = None


class _TeamState(BaseModel):
    team_id: str
    models: list[str] | None = None
    members_with_roles: tuple[_TeamRosterMember, ...] = ()
    metadata: dict[str, JsonValue] | None = None


class _TeamMembershipInfo(BaseModel):
    user_id: str
    team_id: str
    budget_id: str | None = None
    spend: float | None = None
    budget_source: str | None = None
    litellm_budget_table: _BudgetRecord | None = None


class _TeamInfo(BaseModel):
    team_info: _TeamState
    team_memberships: list[_TeamMembershipInfo]


class _MemberMe(BaseModel):
    user_id: str
    team_id: str
    budget_id: str | None = None
    role: str | None = None
    litellm_budget_table: _BudgetRecord | None = None


class _MemberUpdateResponse(BaseModel):
    team_id: str
    user_id: str
    user_email: str | None = None
    max_budget_in_team: float | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    budget_duration: str | None = None
    allowed_models: list[str] | None = None
    temp_budget_increase: float | None = None
    temp_budget_expiry: datetime | None = None


class _BulkBudgetResult(BaseModel):
    user_id: str | None = None
    user_email: str | None = None
    success: bool
    error: str | None = None
    budget_id: str | None = None
    max_budget: float | None = None
    max_budget_source: str | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    budget_duration: str | None = None
    allowed_models: tuple[str, ...] | None = None


class _BulkBudgetUpdateResponse(BaseModel):
    data: tuple[_BulkBudgetResult, ...]


@pytest.fixture
def upstream(gateway: Gateway) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=gateway.upstream_url, timeout=15, trust_env=False) as client:
        response: Final = client.get("/__observations")
        assert response.status_code == 200, response.text
        assert _Observations.model_validate_json(response.text).requests == [], response.text
        yield client


def _priced_model(scenario: Scenario) -> tuple[str, str]:
    provider_model: Final = f"team-member-spend-{uuid.uuid4().hex}"
    model: Final = scenario.model(
        model=f"openai/{provider_model}",
        input_cost_per_token=0.001,
        output_cost_per_token=0.002,
    )
    return model, provider_model


def _free_model(scenario: Scenario) -> tuple[str, str]:
    provider_model: Final = f"team-member-free-{uuid.uuid4().hex}"
    model: Final = scenario.model(
        model=f"openai/{provider_model}",
        input_cost_per_token=0,
        output_cost_per_token=0,
    )
    return model, provider_model


def _observed_models(upstream: httpx.Client) -> tuple[str, ...]:
    response: Final = upstream.get("/__observations")
    assert response.status_code == 200, response.text
    observations: Final = _Observations.model_validate_json(response.text)
    return tuple(string_value(request.body["model"]) for request in observations.requests)


def _chat(gateway: Gateway, model: str, key: str, text: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": text}]},
        key=key,
    )


def _team_info(gateway: Gateway, team_id: str) -> _TeamInfo:
    response: Final = gateway.request("GET", "/team/info", params={"team_id": team_id})
    assert response.status_code == 200, response.text
    return _TeamInfo.model_validate_json(response.text)


def _member_me(gateway: Gateway, team_id: str, key: str) -> _MemberMe:
    response: Final = gateway.request("GET", f"/team/{team_id}/members/me", key=key)
    assert response.status_code == 200, response.text
    return _MemberMe.model_validate_json(response.text)


def _budget_state(team_id: str, user_id: str) -> tuple[_MemberBudgetState, ...]:
    return tuple(
        _MemberBudgetState.model_validate(row)
        for row in read_rows(
            "SELECT tm.user_id, tm.team_id, tm.budget_id, tm.spend, b.max_budget, b.tpm_limit, b.rpm_limit, "
            "b.budget_duration, b.allowed_models, b.temp_budget_increase, "
            "b.temp_budget_expiry::text AS temp_budget_expiry "
            'FROM "LiteLLM_TeamMembership" tm LEFT JOIN "LiteLLM_BudgetTable" b ON b.budget_id = tm.budget_id '
            "WHERE tm.team_id = %s AND tm.user_id = %s",
            (team_id, user_id),
        )
    )


def _budget_record(budget_id: str) -> tuple[_BudgetRecord, ...]:
    return tuple(
        _BudgetRecord.model_validate(row)
        for row in read_rows(
            "SELECT budget_id, max_budget, tpm_limit, rpm_limit, budget_duration, allowed_models, "
            'temp_budget_increase, temp_budget_expiry FROM "LiteLLM_BudgetTable" WHERE budget_id = %s',
            (budget_id,),
        )
    )


def _team_default_budget_id(team_id: str) -> str:
    rows: Final = read_rows('SELECT metadata FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team_id,))
    assert len(rows) == 1, rows
    metadata: Final = object_value(rows[0]["metadata"])
    return string_value(metadata["team_member_budget_id"])


def _member_update(gateway: Gateway, body: dict[str, JsonValue]) -> _MemberUpdateResponse:
    response: Final = gateway.request("POST", "/team/member_update", body)
    assert response.status_code == 200, response.text
    return _MemberUpdateResponse.model_validate_json(response.text)


def _budget_refused(response: httpx.Response) -> None:
    assert response.status_code == 422, response.text
    assert _ErrorResponse.model_validate_json(response.text).error.type == "budget_exceeded", response.text


@pytest.mark.covers("spend.team_member.member_without_budget_gets_membership_row_and_spend")
def test_member_added_without_any_budget_is_charged_on_its_membership_row(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team: Final = scenario.team(models=[model])
        user: Final = scenario.user()
        added: Final = gateway.request(
            "POST", "/team/member_add", {"team_id": team, "member": {"user_id": user, "role": "user"}}
        )
        assert added.status_code == 200, added.text
        memberships: Final = added.json()["updated_team_memberships"]
        assert [
            {"user_id": row["user_id"], "team_id": row["team_id"], "budget_id": row["budget_id"], "spend": row["spend"]}
            for row in memberships
        ] == [{"user_id": user, "team_id": team, "budget_id": None, "spend": 0}], added.text
        assert read_rows(
            'SELECT budget_id, spend, total_spend FROM "LiteLLM_TeamMembership" WHERE team_id=%s AND user_id=%s',
            (team, user),
        ) == [{"budget_id": None, "spend": 0.0, "total_spend": 0.0}]
        key: Final = scenario.key(team_id=team, user_id=user, models=[model])
        assert gateway.chat(model, key=key, text=f"member spend {uuid.uuid4().hex}")["usage"]["total_tokens"] == 40
        charged: Final = eventually(
            lambda: read_rows(
                'SELECT spend, total_spend FROM "LiteLLM_TeamMembership" WHERE team_id=%s AND user_id=%s',
                (team, user),
            ),
            lambda values: len(values) == 1 and float(values[0]["spend"]) >= 0.06,
            seconds=70,
        )
        assert float(charged[0]["spend"]) == pytest.approx(0.06)
        assert float(charged[0]["total_spend"]) == pytest.approx(0.06)
        team_rows: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_TeamTable" WHERE team_id=%s', (team,)),
            lambda values: len(values) == 1 and float(values[0]["spend"]) >= 0.06,
            seconds=70,
        )
        assert float(team_rows[0]["spend"]) == pytest.approx(0.06)
        info: Final = gateway.get("/team/info", {"team_id": team})
        listed: Final = info["team_memberships"]
        assert isinstance(listed, list)
        exposed: Final = [
            (object_value(row)["user_id"], object_value(row)["spend"])
            for row in listed
            if object_value(row)["user_id"] == user
        ]
        assert len(exposed) == 1 and exposed[0][1] == pytest.approx(0.06), info


@pytest.mark.covers("spend.team_member.stale_low_redis_counter_still_blocks_member_over_budget")
def test_member_over_budget_is_blocked_when_redis_counter_reads_stale_low(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
        Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as cache,
    ):
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team: Final = scenario.team(models=[model])
        user: Final = scenario.user()
        added: Final = gateway.request(
            "POST",
            "/team/member_add",
            {"team_id": team, "member": {"user_id": user, "role": "user"}, "max_budget_in_team": 0.05},
        )
        assert added.status_code == 200, added.text
        key: Final = scenario.key(team_id=team, user_id=user, models=[model])
        assert gateway.chat(model, key=key, text=f"member budget {uuid.uuid4().hex}")["usage"]["total_tokens"] == 40
        charged: Final = eventually(
            lambda: read_rows(
                'SELECT spend FROM "LiteLLM_TeamMembership" WHERE team_id=%s AND user_id=%s', (team, user)
            ),
            lambda values: len(values) == 1 and float(values[0]["spend"]) >= 0.06,
            seconds=70,
        )
        assert float(charged[0]["spend"]) == pytest.approx(0.06)
        counter_key: Final = f"spend:team_member:{user}:{team}"
        counted: Final = eventually(lambda: cache.get(counter_key), lambda value: value is not None, seconds=10)
        assert float(counted) == pytest.approx(0.06), counted
        cache.set(counter_key, "0.01")
        upstream.get("/__observations").raise_for_status()
        denied: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": f"stale counter {uuid.uuid4().hex}"}]},
            key=key,
        )
        assert denied.status_code == 422 and denied.json()["error"]["type"] == "budget_exceeded", denied.text
        assert upstream.get("/__observations").json()["requests"] == []
        assert float(cache.get(counter_key)) == pytest.approx(0.06), denied.text


def test_member_update_lowering_the_budget_below_spend_blocks_the_next_call(
    gateway: Gateway, upstream: httpx.Client
) -> None:
    with gateway.scenario() as scenario:
        model, provider_model = _priced_model(scenario)
        team: Final = scenario.team(models=[model])
        user: Final = scenario.user()
        added: Final = gateway.request(
            "POST", "/team/member_add", {"team_id": team, "member": {"user_id": user, "role": "user"}}
        )
        assert added.status_code == 200, added.text
        initial_budget: Final = _member_update(
            gateway,
            {"team_id": team, "user_id": user, "max_budget_in_team": 1},
        )
        assert initial_budget == _MemberUpdateResponse(team_id=team, user_id=user, max_budget_in_team=1), initial_budget
        key: Final = scenario.key(team_id=team, user_id=user)
        personal_key: Final = scenario.key(user_id=user)

        first: Final = _chat(gateway, model, key, f"accrued member spend {uuid.uuid4().hex}")
        assert first.status_code == 200, first.text
        observed_before_update: Final = _observed_models(upstream)
        assert observed_before_update == (provider_model,), repr(observed_before_update)
        accrued: Final = eventually(
            lambda: _budget_state(team, user),
            lambda values: len(values) == 1 and values[0].spend is not None and values[0].spend >= 0.06,
            seconds=70,
        )
        assert accrued[0].spend == pytest.approx(0.06), accrued
        cached_membership: Final = _member_me(gateway, team, key)
        assert cached_membership.litellm_budget_table is not None, repr(cached_membership)
        assert cached_membership.litellm_budget_table.max_budget == 1, repr(cached_membership)

        updated: Final = _member_update(
            gateway,
            {
                "team_id": team,
                "user_id": user,
                "max_budget_in_team": 0.0001,
                "budget_duration": "30d",
                "tpm_limit": 1000,
                "rpm_limit": 100,
            },
        )
        assert updated == _MemberUpdateResponse(
            team_id=team,
            user_id=user,
            max_budget_in_team=0.0001,
            budget_duration="30d",
            tpm_limit=1000,
            rpm_limit=100,
        ), repr(updated)
        denied: Final = _chat(gateway, model, key, f"lowered member budget {uuid.uuid4().hex}")
        _budget_refused(denied)
        observed_after_denial: Final = _observed_models(upstream)
        assert observed_after_denial == (), repr(observed_after_denial)
        assert observed_before_update + observed_after_denial == (provider_model,), observed_after_denial

        state: Final = _budget_state(team, user)
        assert len(state) == 1, state
        assert state[0].user_id == user, state
        assert state[0].team_id == team, state
        assert state[0].spend == pytest.approx(0.06), state
        assert state[0].max_budget == 0.0001, state
        assert state[0].tpm_limit == 1000, state
        assert state[0].rpm_limit == 100, state
        assert state[0].budget_duration == "30d", state
        team_info: Final = _team_info(gateway, team)
        listed: Final = next(member for member in team_info.team_memberships if member.user_id == user)
        assert listed.budget_id == state[0].budget_id, repr(listed)
        assert listed.litellm_budget_table is not None, repr(listed)
        assert listed.litellm_budget_table.max_budget == 0.0001, repr(listed)
        assert listed.litellm_budget_table.tpm_limit == 1000, repr(listed)
        assert listed.litellm_budget_table.rpm_limit == 100, repr(listed)
        assert listed.litellm_budget_table.budget_duration == "30d", repr(listed)
        member_info: Final = _member_me(gateway, team, personal_key)
        assert member_info.budget_id == state[0].budget_id, repr(member_info)
        assert member_info.litellm_budget_table is not None, repr(member_info)
        assert member_info.litellm_budget_table.max_budget == 0.0001, repr(member_info)
        assert member_info.litellm_budget_table.tpm_limit == 1000, repr(member_info)
        assert member_info.litellm_budget_table.rpm_limit == 100, repr(member_info)
        assert member_info.litellm_budget_table.budget_duration == "30d", repr(member_info)


def test_member_update_temp_budget_increase_readmits_a_blocked_member(gateway: Gateway, upstream: httpx.Client) -> None:
    with gateway.scenario() as scenario:
        model, provider_model = _priced_model(scenario)
        team: Final = scenario.team(models=[model])
        user: Final = scenario.user()
        added: Final = gateway.request(
            "POST",
            "/team/member_add",
            {
                "team_id": team,
                "member": {"user_id": user, "role": "user"},
                "max_budget_in_team": 0.05,
            },
        )
        assert added.status_code == 200, added.text
        key: Final = scenario.key(team_id=team, user_id=user)

        first: Final = _chat(gateway, model, key, f"initial member budget {uuid.uuid4().hex}")
        assert first.status_code == 200, first.text
        observed_before_block: Final = _observed_models(upstream)
        assert observed_before_block == (provider_model,), repr(observed_before_block)
        accrued: Final = eventually(
            lambda: _budget_state(team, user),
            lambda values: len(values) == 1 and values[0].spend is not None and values[0].spend >= 0.06,
            seconds=70,
        )
        assert accrued[0].spend == pytest.approx(0.06), accrued
        blocked: Final = _chat(gateway, model, key, f"exceeded member budget {uuid.uuid4().hex}")
        _budget_refused(blocked)
        observed_after_block: Final = _observed_models(upstream)
        assert observed_after_block == (), repr(observed_after_block)

        expiry: Final = (datetime.now(timezone.utc) + timedelta(days=1)).replace(microsecond=0)
        updated: Final = _member_update(
            gateway,
            {
                "team_id": team,
                "user_id": user,
                "temp_budget_increase": 1.0,
                "temp_budget_expiry": expiry.isoformat(),
            },
        )
        assert updated.temp_budget_expiry is not None, repr(updated)
        assert updated.model_copy(
            update={"temp_budget_expiry": updated.temp_budget_expiry.replace(tzinfo=timezone.utc)}
        ) == _MemberUpdateResponse(
            team_id=team,
            user_id=user,
            temp_budget_increase=1.0,
            temp_budget_expiry=expiry,
        ), repr(updated)
        state: Final = _budget_state(team, user)
        assert len(state) == 1, state
        assert state[0].max_budget == 0.05, state
        assert state[0].temp_budget_increase == 1.0, state
        assert state[0].temp_budget_expiry is not None, state
        assert state[0].temp_budget_expiry.replace(tzinfo=timezone.utc) == expiry, state
        team_info: Final = _team_info(gateway, team)
        listed: Final = next(member for member in team_info.team_memberships if member.user_id == user)
        assert listed.litellm_budget_table is not None, repr(listed)
        assert listed.litellm_budget_table.temp_budget_increase == 1.0, repr(listed)
        assert listed.litellm_budget_table.temp_budget_expiry is not None, repr(listed)
        assert listed.litellm_budget_table.temp_budget_expiry.replace(tzinfo=timezone.utc) == expiry, repr(listed)
        member_info: Final = _member_me(gateway, team, key)
        assert member_info.litellm_budget_table is not None, repr(member_info)
        assert member_info.litellm_budget_table.temp_budget_increase == 1.0, repr(member_info)
        assert member_info.litellm_budget_table.temp_budget_expiry is not None, repr(member_info)
        assert member_info.litellm_budget_table.temp_budget_expiry.replace(tzinfo=timezone.utc) == expiry, repr(
            member_info
        )

        readmitted: Final = _chat(gateway, model, key, f"temporary member budget {uuid.uuid4().hex}")
        assert readmitted.status_code == 200, readmitted.text
        observed_after_readmission: Final = _observed_models(upstream)
        assert observed_after_readmission == (provider_model,), repr(observed_after_readmission)
        assert observed_before_block + observed_after_readmission == (provider_model, provider_model), repr(
            (observed_before_block, observed_after_readmission)
        )


def test_member_update_by_email_sets_models_role_and_forks_the_team_default_budget(
    gateway: Gateway, upstream: httpx.Client
) -> None:
    with gateway.scenario() as scenario:
        model_one, provider_one = _free_model(scenario)
        model_two, provider_two = _free_model(scenario)
        team: Final = scenario.team(models=[model_one, model_two], team_member_budget=25)
        email_one: Final = f"member-{uuid.uuid4().hex}@example.com"
        user_one: Final = scenario.user(user_email=email_one, user_role="internal_user")
        user_two: Final = scenario.user(user_role="internal_user")
        added: Final = gateway.request(
            "POST",
            "/team/member_add",
            {
                "team_id": team,
                "member": [
                    {"user_email": email_one, "role": "user"},
                    {"user_id": user_two, "role": "user"},
                ],
            },
        )
        assert added.status_code == 200, added.text
        default_budget_id: Final = _team_default_budget_id(team)
        default_budget_before: Final = _budget_record(default_budget_id)
        assert len(default_budget_before) == 1, repr(default_budget_before)
        initial_one: Final = _budget_state(team, user_one)
        initial_two: Final = _budget_state(team, user_two)
        assert len(initial_one) == 1 and initial_one[0].budget_id == default_budget_id, initial_one
        assert len(initial_two) == 1 and initial_two[0].budget_id == default_budget_id, initial_two

        user_one_key: Final = scenario.key(team_id=team, user_id=user_one)
        model_scope: Final = _member_update(
            gateway,
            {"team_id": team, "user_email": email_one, "allowed_models": [model_one]},
        )
        assert model_scope.user_id == user_one, repr(model_scope)
        assert model_scope.allowed_models == [model_one], repr(model_scope)
        scoped_state: Final = _budget_state(team, user_one)
        assert len(scoped_state) == 1, scoped_state
        assert scoped_state[0].allowed_models == [model_one], scoped_state
        assert scoped_state[0].budget_id != default_budget_id, scoped_state
        other_member_state: Final = _budget_state(team, user_two)
        assert other_member_state[0].budget_id == default_budget_id, other_member_state
        assert _budget_record(default_budget_id) == default_budget_before, repr(default_budget_before)

        allowed: Final = _chat(gateway, model_one, user_one_key, f"email allowed model {uuid.uuid4().hex}")
        assert allowed.status_code == 200, allowed.text
        denied: Final = _chat(gateway, model_two, user_one_key, f"email restricted model {uuid.uuid4().hex}")
        assert denied.status_code == 403, denied.text
        assert _ErrorResponse.model_validate_json(denied.text).error.type == "team_model_access_denied", denied.text
        observed_before_clear: Final = _observed_models(upstream)
        assert observed_before_clear == (provider_one,), repr(observed_before_clear)

        cleared_models: Final = _member_update(
            gateway,
            {"team_id": team, "user_email": email_one, "allowed_models": []},
        )
        assert cleared_models.user_id == user_one, repr(cleared_models)
        assert cleared_models.allowed_models == [], repr(cleared_models)
        cleared_state: Final = _budget_state(team, user_one)
        assert len(cleared_state) == 1, cleared_state
        assert cleared_state[0].allowed_models == [], cleared_state
        unrestricted: Final = _chat(gateway, model_two, user_one_key, f"email cleared model scope {uuid.uuid4().hex}")
        assert unrestricted.status_code == 200, unrestricted.text
        observed_after_clear: Final = _observed_models(upstream)
        assert observed_after_clear == (provider_two,), repr(observed_after_clear)

        budget_update: Final = _member_update(
            gateway,
            {"team_id": team, "user_email": email_one, "max_budget_in_team": 12},
        )
        assert budget_update.user_id == user_one, repr(budget_update)
        assert budget_update.max_budget_in_team == 12, repr(budget_update)
        user_one_state: Final = _budget_state(team, user_one)
        user_two_state: Final = _budget_state(team, user_two)
        assert len(user_one_state) == 1 and user_one_state[0].budget_id != default_budget_id, user_one_state
        assert user_one_state[0].max_budget == 12, user_one_state
        assert user_two_state == initial_two, user_two_state
        assert _budget_record(default_budget_id) == default_budget_before, repr(default_budget_before)
        team_info: Final = _team_info(gateway, team)
        listed_one: Final = next(member for member in team_info.team_memberships if member.user_id == user_one)
        assert listed_one.litellm_budget_table is not None, repr(listed_one)
        assert listed_one.litellm_budget_table.max_budget == 12, repr(listed_one)
        assert listed_one.litellm_budget_table.allowed_models == [], repr(listed_one)
        member_info: Final = _member_me(gateway, team, user_one_key)
        assert member_info.litellm_budget_table is not None, repr(member_info)
        assert member_info.litellm_budget_table.max_budget == 12, repr(member_info)
        assert member_info.litellm_budget_table.allowed_models == [], repr(member_info)
        roster_member: Final = next(
            member for member in team_info.team_info.members_with_roles if member.user_id == user_one
        )
        assert roster_member.role == "user", repr(roster_member)

        new_member: Final = scenario.user()
        user_key: Final = scenario.key(user_id=user_one)
        denied_admin_action: Final = gateway.request(
            "POST",
            "/team/member_add",
            {"team_id": team, "member": {"user_id": new_member, "role": "user"}},
            key=user_key,
        )
        assert denied_admin_action.status_code == 403, denied_admin_action.text
        role_update: Final = _member_update(
            gateway,
            {"team_id": team, "user_email": email_one, "role": "admin"},
        )
        assert role_update.user_id == user_one, repr(role_update)
        role_info: Final = _team_info(gateway, team)
        admin_roster: Final = next(
            member for member in role_info.team_info.members_with_roles if member.user_id == user_one
        )
        assert admin_roster.role == "admin", repr(admin_roster)
        roster_rows: Final = read_rows(
            'SELECT members_with_roles FROM "LiteLLM_TeamTable" WHERE team_id = %s',
            (team,),
        )
        assert len(roster_rows) == 1, roster_rows
        database_roster: Final = _TeamState.model_validate(
            {"team_id": team, "members_with_roles": roster_rows[0]["members_with_roles"]}
        ).members_with_roles
        assert next(member for member in database_roster if member.user_id == user_one).role == "admin", repr(
            database_roster
        )
        allowed_admin_action: Final = gateway.request(
            "POST",
            "/team/member_add",
            {"team_id": team, "member": {"user_id": new_member, "role": "user"}},
            key=user_key,
        )
        assert allowed_admin_action.status_code == 200, allowed_admin_action.text
        assert observed_before_clear + observed_after_clear == (provider_one, provider_two), repr(
            (observed_before_clear, observed_after_clear)
        )


def test_bulk_update_is_a_merge_patch_per_member(gateway: Gateway, upstream: httpx.Client) -> None:
    with gateway.scenario() as scenario:
        model_one, provider_one = _free_model(scenario)
        model_two, _ = _free_model(scenario)
        spend_model, provider_spend = _priced_model(scenario)
        team: Final = scenario.team(models=[model_one, model_two, spend_model], team_member_budget=25)
        user_one: Final = scenario.user()
        email_two: Final = f"bulk-member-{uuid.uuid4().hex}@example.com"
        user_two: Final = scenario.user(user_email=email_two, user_role="internal_user")
        added: Final = gateway.request(
            "POST",
            "/team/member_add",
            {
                "team_id": team,
                "member": [
                    {"user_id": user_one, "role": "user"},
                    {"user_email": email_two, "role": "user"},
                ],
            },
        )
        assert added.status_code == 200, added.text
        default_budget_id: Final = _team_default_budget_id(team)
        default_budget_before: Final = _budget_record(default_budget_id)
        key_one: Final = scenario.key(team_id=team, user_id=user_one)
        key_two: Final = scenario.key(team_id=team, user_id=user_two)
        personal_key_one: Final = scenario.key(user_id=user_one)

        first_chat: Final = _chat(gateway, spend_model, key_one, f"bulk member spend {uuid.uuid4().hex}")
        assert first_chat.status_code == 200, first_chat.text
        observed_before_update: Final = _observed_models(upstream)
        assert observed_before_update == (provider_spend,), repr(observed_before_update)
        accrued: Final = eventually(
            lambda: _budget_state(team, user_one),
            lambda values: len(values) == 1 and values[0].spend is not None and values[0].spend >= 0.06,
            seconds=70,
        )
        assert accrued[0].spend == pytest.approx(0.06), accrued

        update: Final = gateway.request(
            "POST",
            f"/management/v1/teams/{team}/members/bulk_update",
            {
                "members": [
                    {"user_id": user_one, "max_budget_in_team": 0.0001, "rpm_limit": 7},
                    {"user_email": email_two, "rpm_limit": 5, "allowed_models": [model_one]},
                ]
            },
        )
        assert update.status_code == 200, update.text
        update_response: Final = _BulkBudgetUpdateResponse.model_validate_json(update.text)
        state_one: Final = _budget_state(team, user_one)
        state_two: Final = _budget_state(team, user_two)
        assert len(state_one) == 1 and len(state_two) == 1, (state_one, state_two)
        assert state_one[0].budget_id is not None, state_one
        assert state_two[0].budget_id is not None, state_two
        assert state_one[0].budget_id != default_budget_id, state_one
        assert state_two[0].budget_id != default_budget_id, state_two
        assert update_response.data == (
            _BulkBudgetResult(
                user_id=user_one,
                success=True,
                budget_id=state_one[0].budget_id,
                max_budget=0.0001,
                max_budget_source="member",
                rpm_limit=7,
                allowed_models=tuple(state_one[0].allowed_models or ()),
            ),
            _BulkBudgetResult(
                user_id=user_two,
                user_email=email_two,
                success=True,
                budget_id=state_two[0].budget_id,
                max_budget=25,
                max_budget_source="member",
                rpm_limit=5,
                allowed_models=(model_one,),
            ),
        ), update.text
        assert state_one[0].max_budget == 0.0001 and state_one[0].rpm_limit == 7, state_one
        assert state_two[0].max_budget == 25 and state_two[0].rpm_limit == 5, state_two
        assert state_two[0].allowed_models == [model_one], state_two
        assert _budget_record(default_budget_id) == default_budget_before, repr(default_budget_before)
        info: Final = _team_info(gateway, team)
        info_one: Final = next(member for member in info.team_memberships if member.user_id == user_one)
        assert info_one.litellm_budget_table is not None, repr(info_one)
        assert info_one.litellm_budget_table.max_budget == 0.0001, repr(info_one)
        assert info_one.litellm_budget_table.rpm_limit == 7, repr(info_one)
        me_one: Final = _member_me(gateway, team, personal_key_one)
        assert me_one.litellm_budget_table is not None, repr(me_one)
        assert me_one.litellm_budget_table.max_budget == 0.0001, repr(me_one)
        assert me_one.litellm_budget_table.rpm_limit == 7, repr(me_one)

        over_budget: Final = _chat(gateway, spend_model, key_one, f"bulk member blocked {uuid.uuid4().hex}")
        _budget_refused(over_budget)
        observed_after_over_budget: Final = _observed_models(upstream)
        assert observed_after_over_budget == (), repr(observed_after_over_budget)
        allowed_two: Final = _chat(gateway, model_one, key_two, f"bulk member scoped allowed {uuid.uuid4().hex}")
        assert allowed_two.status_code == 200, allowed_two.text
        denied_two: Final = _chat(gateway, model_two, key_two, f"bulk member scoped denied {uuid.uuid4().hex}")
        assert denied_two.status_code == 403, denied_two.text
        assert _ErrorResponse.model_validate_json(denied_two.text).error.type == "team_model_access_denied", (
            denied_two.text
        )
        observed_after_first_patch: Final = _observed_models(upstream)
        assert observed_after_first_patch == (provider_one,), repr(observed_after_first_patch)

        cleared_max: Final = gateway.request(
            "POST",
            f"/management/v1/teams/{team}/members/bulk_update",
            {"members": [{"user_id": user_one, "max_budget_in_team": None}]},
        )
        assert cleared_max.status_code == 200, cleared_max.text
        cleared_max_response: Final = _BulkBudgetUpdateResponse.model_validate_json(cleared_max.text)
        cleared_max_state: Final = _budget_state(team, user_one)
        assert len(cleared_max_state) == 1, cleared_max_state
        assert cleared_max_response.data == (
            _BulkBudgetResult(
                user_id=user_one,
                success=True,
                budget_id=cleared_max_state[0].budget_id,
                max_budget=25,
                max_budget_source="team_default",
                rpm_limit=7,
                allowed_models=tuple(cleared_max_state[0].allowed_models or ()),
            ),
        ), cleared_max.text
        assert cleared_max_state[0].budget_id == state_one[0].budget_id, cleared_max_state
        assert cleared_max_state[0].max_budget is None, cleared_max_state
        assert cleared_max_state[0].rpm_limit == 7, cleared_max_state
        assert _budget_record(default_budget_id) == default_budget_before, repr(default_budget_before)
        me_after_max_clear: Final = _member_me(gateway, team, personal_key_one)
        assert me_after_max_clear.litellm_budget_table is not None, repr(me_after_max_clear)
        assert me_after_max_clear.litellm_budget_table.max_budget is None, repr(me_after_max_clear)
        assert me_after_max_clear.litellm_budget_table.rpm_limit == 7, repr(me_after_max_clear)
        admitted_again: Final = _chat(gateway, spend_model, key_one, f"bulk member readmitted {uuid.uuid4().hex}")
        assert admitted_again.status_code == 200, admitted_again.text
        observed_after_max_clear: Final = _observed_models(upstream)
        assert observed_after_max_clear == (provider_spend,), repr(observed_after_max_clear)

        assert observed_before_update + observed_after_first_patch + observed_after_max_clear == (
            provider_spend,
            provider_one,
            provider_spend,
        ), repr((observed_before_update, observed_after_first_patch, observed_after_max_clear))


def test_bulk_update_restores_team_default_when_last_member_limit_is_cleared(
    gateway: Gateway, upstream: httpx.Client
) -> None:
    with gateway.scenario() as scenario:
        model, provider_model = _priced_model(scenario)
        team: Final = scenario.team(models=[model], team_member_budget=0.0001)
        user: Final = scenario.user()
        added: Final = gateway.request(
            "POST",
            "/team/member_add",
            {"team_id": team, "member": {"user_id": user, "role": "user"}},
        )
        assert added.status_code == 200, added.text
        default_budget_id: Final = _team_default_budget_id(team)
        default_budget_before: Final = _budget_record(default_budget_id)
        initial_state: Final = _budget_state(team, user)
        assert len(initial_state) == 1 and initial_state[0].budget_id == default_budget_id, initial_state
        key: Final = scenario.key(team_id=team, user_id=user, models=[model])
        increased: Final = gateway.request(
            "POST",
            f"/management/v1/teams/{team}/members/bulk_update",
            {"members": [{"user_id": user, "max_budget_in_team": 5}]},
        )
        assert increased.status_code == 200, increased.text
        increased_response: Final = _BulkBudgetUpdateResponse.model_validate_json(increased.text)
        increased_state: Final = _budget_state(team, user)
        assert len(increased_state) == 1, increased_state
        assert increased_state[0].budget_id is not None and increased_state[0].budget_id != default_budget_id, (
            increased_state
        )
        assert increased_response.data == (
            _BulkBudgetResult(
                user_id=user,
                success=True,
                budget_id=increased_state[0].budget_id,
                max_budget=5,
                max_budget_source="member",
                allowed_models=tuple(increased_state[0].allowed_models or ()),
            ),
        ), increased.text
        first_chat: Final = _chat(gateway, model, key, f"member raised limit {uuid.uuid4().hex}")
        assert first_chat.status_code == 200, first_chat.text
        first_observations: Final = _observed_models(upstream)
        assert first_observations == (provider_model,), repr(first_observations)
        accrued: Final = eventually(
            lambda: _budget_state(team, user),
            lambda values: len(values) == 1 and values[0].spend is not None and values[0].spend >= 0.06,
            seconds=70,
        )
        assert accrued[0].spend == pytest.approx(0.06), accrued
        second_chat: Final = _chat(gateway, model, key, f"member still under limit {uuid.uuid4().hex}")
        assert second_chat.status_code == 200, second_chat.text
        second_observations: Final = _observed_models(upstream)
        assert second_observations == (provider_model,), repr(second_observations)
        cleared: Final = gateway.request(
            "POST",
            f"/management/v1/teams/{team}/members/bulk_update",
            {"members": [{"user_id": user, "max_budget_in_team": None}]},
        )
        assert cleared.status_code == 200, cleared.text
        cleared_response: Final = _BulkBudgetUpdateResponse.model_validate_json(cleared.text)
        final_state: Final = _budget_state(team, user)
        assert len(final_state) == 1, final_state
        assert final_state[0].budget_id is None, final_state
        assert cleared_response.data == (
            _BulkBudgetResult(
                user_id=user,
                success=True,
                budget_id=None,
                max_budget=0.0001,
                max_budget_source="team_default",
                tpm_limit=None,
                rpm_limit=None,
                budget_duration=None,
                allowed_models=None,
            ),
        ), cleared.text
        assert read_rows(
            'SELECT budget_id FROM "LiteLLM_TeamMembership" WHERE team_id=%s AND user_id=%s',
            (team, user),
        ) == [{"budget_id": None}], cleared.text
        assert _budget_record(default_budget_id) == default_budget_before, repr(default_budget_before)
        refused: Final = _chat(gateway, model, key, f"member default cap reached {uuid.uuid4().hex}")
        _budget_refused(refused)
        observed_after_refusal: Final = _observed_models(upstream)
        assert observed_after_refusal == (), repr(observed_after_refusal)
        assert first_observations + second_observations == (provider_model, provider_model), repr(
            (first_observations, second_observations)
        )


@pytest.fixture(autouse=True)
def _drain_unobserved_requests(gateway: Gateway) -> Iterator[None]:
    yield
    with httpx.Client(base_url=gateway.upstream_url, timeout=15, trust_env=False) as client:
        response: Final = client.get("/__observations")
        assert response.status_code == 200, response.text
