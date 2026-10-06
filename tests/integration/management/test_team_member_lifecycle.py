import json
import uuid
from collections.abc import Iterator
from hashlib import sha256
from typing import Final

import httpx
import pytest
from pydantic import BaseModel, JsonValue

from tests.integration._support.client import (
    Gateway,
    Scenario,
    delete_key_if_present,
    string_value,
)
from tests.integration._support.database import read_rows, write_rows


class _ObservedRequest(BaseModel):
    body: dict[str, JsonValue]


class _Observations(BaseModel):
    requests: list[_ObservedRequest]


class _RosterMember(BaseModel):
    role: str
    user_id: str | None = None
    user_email: str | None = None


class _TeamRecord(BaseModel):
    team_id: str
    models: list[str] | None = None
    members_with_roles: list[_RosterMember]


class _BudgetDetails(BaseModel):
    max_budget: float | None = None
    allowed_models: list[str] | None = None


class _TeamMembership(BaseModel):
    team_id: str
    user_id: str
    budget_id: str | None = None
    spend: float | None = None
    total_spend: float | None = None
    litellm_budget_table: _BudgetDetails | None = None


class _TeamInfo(BaseModel):
    team_info: _TeamRecord
    team_memberships: list[_TeamMembership] | None = None


class _UserRecord(BaseModel):
    user_id: str
    user_email: str | None = None
    teams: list[str] | None = None


class _MemberAddResponse(BaseModel):
    team_id: str
    updated_users: list[_UserRecord]
    updated_team_memberships: list[_TeamMembership]


class _GeneratedKey(BaseModel):
    key: str


class _ProxyError(BaseModel):
    type: str


class _ErrorResponse(BaseModel):
    error: _ProxyError


class _HTTPErrorDetail(BaseModel):
    error: str


class _HTTPErrorResponse(BaseModel):
    detail: _HTTPErrorDetail


class _BulkAddResult(BaseModel):
    user_id: str | None = None
    user_email: str | None = None
    success: bool
    error: str | None = None


class _BulkAddResponse(BaseModel):
    team_id: str
    results: list[_BulkAddResult]
    total_requested: int
    successful_additions: int
    failed_additions: int


class _BulkDeleteResult(BaseModel):
    user_id: str | None = None
    user_email: str | None = None
    success: bool
    error: str | None = None


class _BulkDeleteResponse(BaseModel):
    data: tuple[_BulkDeleteResult, ...]


@pytest.fixture
def upstream(gateway: Gateway) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=gateway.upstream_url, timeout=15, trust_env=False) as client:
        response: Final = client.get("/__observations")
        assert response.status_code == 200, response.text
        yield client


def _model(scenario: Scenario) -> tuple[str, str]:
    provider_model: Final = f"team-member-{uuid.uuid4().hex}"
    return scenario.model(model=f"openai/{provider_model}"), provider_model


def _key(scenario: Scenario, **fields: JsonValue) -> str:
    response: Final = scenario.gateway.request("POST", "/key/generate", fields)
    assert response.status_code == 200, response.text
    token: Final = _GeneratedKey.model_validate_json(response.text).key
    scenario.cleanups.callback(delete_key_if_present, scenario.gateway, token)
    return token


def _team_info(gateway: Gateway, team_id: str) -> _TeamInfo:
    response: Final = gateway.request("GET", "/team/info", params={"team_id": team_id})
    assert response.status_code == 200, response.text
    return _TeamInfo.model_validate_json(response.text)


def _models(upstream: httpx.Client) -> tuple[str, ...]:
    response: Final = upstream.get("/__observations")
    assert response.status_code == 200, response.text
    observations: Final = _Observations.model_validate_json(response.text)
    return tuple(string_value(request.body["model"]) for request in observations.requests)


def _chat(proxy: Gateway, model: str, key: str, text: str) -> httpx.Response:
    return proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": text}]},
        key=key,
    )


def _refused(response: httpx.Response, status: int, error_type: str) -> None:
    assert response.status_code == status, response.text
    assert _ErrorResponse.model_validate_json(response.text).error.type == error_type, response.text


def _remove_default_user(gateway: Gateway, team_id: str) -> None:
    response: Final = gateway.request("POST", "/team/member_delete", {"team_id": team_id, "user_id": "default_user_id"})
    assert response.status_code == 200, response.text


def _roster_member(user_id: str, user_email: str | None, role: str = "user") -> dict[str, JsonValue]:
    return {"role": role, "user_id": user_id, "user_email": user_email}


@pytest.mark.parametrize("identifier", ("user_id", "user_email"), ids=("by-id", "by-email"))
def test_member_delete_revokes_the_removed_members_warmed_key(
    gateway: Gateway, upstream: httpx.Client, identifier: str
) -> None:
    with gateway.scenario() as scenario:
        model, provider_model = _model(scenario)
        team: Final = scenario.team(models=[model])
        leaver_email: Final = f"member-leaver-{uuid.uuid4().hex}@integration.test"
        leaver: Final = scenario.user(user_email=leaver_email)
        _remove_default_user(gateway, team)
        stayer: Final = scenario.member(team)
        added: Final = gateway.request(
            "POST",
            "/team/member_add",
            {
                "team_id": team,
                "member": (
                    {"role": "user", "user_id": leaver}
                    if identifier == "user_id"
                    else {"role": "user", "user_email": leaver_email}
                ),
            },
        )
        assert added.status_code == 200, added.text
        leaver_key: Final = _key(scenario, team_id=team, user_id=leaver)
        stayer_key: Final = _key(scenario, team_id=team, user_id=stayer)
        leaver_warm: Final = _chat(gateway, model, leaver_key, "warm leaver " + uuid.uuid4().hex)
        assert leaver_warm.status_code == 200, leaver_warm.text
        stayer_warm: Final = _chat(gateway, model, stayer_key, "warm stayer " + uuid.uuid4().hex)
        assert stayer_warm.status_code == 200, stayer_warm.text
        observed_before_delete: Final = _models(upstream)
        assert observed_before_delete == (provider_model, provider_model), repr(observed_before_delete)

        delete_body: Final = (
            {"team_id": team, "user_id": leaver}
            if identifier == "user_id"
            else {"team_id": team, "user_email": leaver_email}
        )
        deleted: Final = gateway.request("POST", "/team/member_delete", delete_body)
        assert deleted.status_code == 200, deleted.text

        info: Final = _team_info(gateway, team)
        stayer_roster: Final = [_roster_member(stayer, None)]
        assert info.team_info.members_with_roles == [_RosterMember.model_validate(stayer_roster[0])], repr(info)
        assert read_rows('SELECT members_with_roles FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team,)) == [
            {"members_with_roles": stayer_roster}
        ], deleted.text
        assert (
            read_rows(
                'SELECT user_id, team_id FROM "LiteLLM_TeamMembership" WHERE team_id = %s AND user_id = %s',
                (team, leaver),
            )
            == []
        ), deleted.text
        assert read_rows(
            'SELECT user_id, team_id FROM "LiteLLM_TeamMembership" WHERE team_id = %s AND user_id = %s',
            (team, stayer),
        ) == [{"user_id": stayer, "team_id": team}], deleted.text
        leaver_hash: Final = sha256(leaver_key.encode()).hexdigest()
        assert (
            read_rows(
                'SELECT token FROM "LiteLLM_VerificationToken" WHERE token = %s',
                (leaver_hash,),
            )
            == []
        ), deleted.text
        assert read_rows(
            'SELECT token FROM "LiteLLM_DeletedVerificationToken" WHERE token = %s',
            (leaver_hash,),
        ) == [{"token": leaver_hash}], deleted.text

        refused: Final = _chat(gateway, model, leaver_key, "deleted leaver " + uuid.uuid4().hex)
        _refused(refused, 401, "token_not_found_in_db")
        after_refused: Final = _models(upstream)
        assert after_refused == (), repr(after_refused)
        stayer_after: Final = _chat(gateway, model, stayer_key, "stayer after delete " + uuid.uuid4().hex)
        assert stayer_after.status_code == 200, stayer_after.text
        observed_after_delete: Final = _models(upstream)
        assert observed_before_delete + after_refused + observed_after_delete == (
            provider_model,
            provider_model,
            provider_model,
        ), repr((observed_before_delete, after_refused, observed_after_delete))


def test_member_delete_by_email_removes_a_legacy_email_only_roster_entry(
    gateway: Gateway, upstream: httpx.Client
) -> None:
    with gateway.scenario() as scenario:
        model, provider_model = _model(scenario)
        team: Final = scenario.team(models=[model])
        email: Final = f"legacy-member-{uuid.uuid4().hex}@integration.test"
        user: Final = scenario.user(user_email=email)
        stayer: Final = scenario.member(team)
        stayer_roster: Final = _roster_member(stayer, None)
        legacy_roster: Final = {"role": "user", "user_id": None, "user_email": email}
        seeded_roster: Final = [legacy_roster, stayer_roster]
        write_rows(
            'UPDATE "LiteLLM_TeamTable" SET members_with_roles = %s::jsonb WHERE team_id = %s',
            (json.dumps(seeded_roster), team),
        )
        write_rows(
            'UPDATE "LiteLLM_UserTable" SET teams = array_append(teams, %s) WHERE user_id = %s',
            (team, user),
        )

        deleted: Final = gateway.request("POST", "/team/member_delete", {"team_id": team, "user_email": email})
        assert deleted.status_code == 200, deleted.text
        info: Final = _team_info(gateway, team)
        expected_stayer: Final = [_RosterMember.model_validate(stayer_roster)]
        assert info.team_info.members_with_roles == expected_stayer, repr(info)
        assert read_rows('SELECT members_with_roles FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team,)) == [
            {"members_with_roles": [stayer_roster]}
        ], deleted.text
        stayer_key: Final = _key(scenario, team_id=team, user_id=stayer)
        served: Final = _chat(gateway, model, stayer_key, "stayer after legacy delete " + uuid.uuid4().hex)
        assert served.status_code == 200, served.text
        observed: Final = _models(upstream)
        assert observed == (provider_model,), repr(observed)


def test_member_add_by_email_provisions_a_new_user_and_resolves_an_existing_one(
    gateway: Gateway, upstream: httpx.Client
) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        _remove_default_user(gateway, team)
        existing_email: Final = f"existing-member-{uuid.uuid4().hex}@integration.test"
        existing_user: Final = scenario.user(user_email=existing_email)
        existing_email_variant: Final = existing_email.upper()
        existing_added: Final = gateway.request(
            "POST",
            "/team/member_add",
            {"team_id": team, "member": {"role": "user", "user_email": existing_email_variant}},
        )
        assert existing_added.status_code == 200, existing_added.text
        existing_response: Final = _MemberAddResponse.model_validate_json(existing_added.text)
        assert tuple(user.user_id for user in existing_response.updated_users) == (existing_user,), existing_added.text
        assert read_rows(
            'SELECT user_id, user_email FROM "LiteLLM_UserTable" WHERE LOWER(user_email) = LOWER(%s)',
            (existing_email,),
        ) == [{"user_id": existing_user, "user_email": existing_email}], existing_added.text

        new_email: Final = f"provisioned-member-{uuid.uuid4().hex}@integration.test"
        new_added: Final = gateway.request(
            "POST",
            "/team/member_add",
            {"team_id": team, "member": {"role": "user", "user_email": new_email}},
        )
        assert new_added.status_code == 200, new_added.text
        new_response: Final = _MemberAddResponse.model_validate_json(new_added.text)
        assert len(new_response.updated_users) == 1, new_added.text
        provisioned: Final = new_response.updated_users[0]
        assert provisioned.user_email == new_email, new_added.text
        scenario.cleanups.callback(scenario.delete_user, provisioned.user_id)
        assert read_rows(
            'SELECT user_id, user_email, teams FROM "LiteLLM_UserTable" WHERE user_email = %s',
            (new_email,),
        ) == [{"user_id": provisioned.user_id, "user_email": new_email, "teams": [team]}], new_added.text
        info: Final = _team_info(gateway, team)
        assert info.team_info.members_with_roles == [
            _RosterMember(role="user", user_id=existing_user, user_email=existing_email_variant),
            _RosterMember(role="user", user_id=provisioned.user_id, user_email=new_email),
        ], repr(info)
        observed: Final = _models(upstream)
        assert observed == (), repr(observed)


def test_member_add_list_mixing_id_and_email_applies_budget_and_allowed_models(
    gateway: Gateway, upstream: httpx.Client
) -> None:
    with gateway.scenario() as scenario:
        model_one, provider_one = _model(scenario)
        model_two, provider_two = _model(scenario)
        team: Final = scenario.team(models=[model_one, model_two])
        _remove_default_user(gateway, team)
        user_one: Final = scenario.user(user_role="internal_user")
        email_two: Final = f"mixed-member-{uuid.uuid4().hex}@integration.test"
        user_two: Final = scenario.user(user_email=email_two, user_role="internal_user")
        added: Final = gateway.request(
            "POST",
            "/team/member_add",
            {
                "team_id": team,
                "member": [
                    {"role": "user", "user_id": user_one},
                    {"role": "user", "user_email": email_two},
                ],
                "max_budget_in_team": 1,
                "allowed_models": [model_one],
            },
        )
        assert added.status_code == 200, added.text
        response: Final = _MemberAddResponse.model_validate_json(added.text)
        assert tuple(user.user_id for user in response.updated_users) == (user_one, user_two), added.text
        assert tuple(member.user_id for member in response.updated_team_memberships) == (user_one, user_two), added.text
        membership_rows: Final = read_rows(
            "SELECT tm.user_id, tm.budget_id, b.max_budget, b.allowed_models "
            'FROM "LiteLLM_TeamMembership" tm '
            'JOIN "LiteLLM_BudgetTable" b ON b.budget_id = tm.budget_id '
            "WHERE tm.team_id = %s ORDER BY tm.user_id",
            (team,),
        )
        assert tuple(row["user_id"] for row in membership_rows) == tuple(
            sorted((user_one, user_two)),
        ), membership_rows
        assert all(float(row["max_budget"]) == 1 for row in membership_rows), membership_rows
        assert all(row["allowed_models"] == [model_one] for row in membership_rows), membership_rows
        info: Final = _team_info(gateway, team)
        membership_limits: Final = tuple(
            sorted(
                (
                    (
                        membership.user_id,
                        membership.litellm_budget_table.max_budget
                        if membership.litellm_budget_table is not None
                        else None,
                        membership.litellm_budget_table.allowed_models
                        if membership.litellm_budget_table is not None
                        else None,
                    )
                    for membership in info.team_memberships or ()
                ),
                key=lambda member: member[0],
            )
        )
        assert membership_limits == tuple(
            sorted(((user_one, 1, [model_one]), (user_two, 1, [model_one])), key=lambda member: member[0])
        ), added.text

        key_one: Final = _key(scenario, team_id=team, user_id=user_one)
        key_two: Final = _key(scenario, team_id=team, user_id=user_two)
        allowed_one: Final = _chat(gateway, model_one, key_one, "mixed member one allowed")
        assert allowed_one.status_code == 200, allowed_one.text
        denied_one: Final = _chat(gateway, model_two, key_one, "mixed member one denied")
        _refused(denied_one, 403, "team_model_access_denied")
        after_denied_one: Final = _models(upstream)
        allowed_two: Final = _chat(gateway, model_one, key_two, "mixed member two allowed")
        assert allowed_two.status_code == 200, allowed_two.text
        denied_two: Final = _chat(gateway, model_two, key_two, "mixed member two denied")
        _refused(denied_two, 403, "team_model_access_denied")
        after_denied_two: Final = _models(upstream)
        assert after_denied_one == (provider_one,), repr(after_denied_one)
        assert after_denied_two == (provider_one,), repr(after_denied_two)
        assert provider_two not in after_denied_two, repr(after_denied_two)
        assert after_denied_one + after_denied_two == (provider_one, provider_one), repr(
            (after_denied_one, after_denied_two)
        )


def test_bulk_member_add_adds_every_listed_member_and_reports_failures(
    gateway: Gateway, upstream: httpx.Client
) -> None:
    with gateway.scenario() as scenario:
        model, provider_model = _model(scenario)
        team: Final = scenario.team(models=[model])
        _remove_default_user(gateway, team)
        user_one: Final = scenario.user(user_role="internal_user")
        user_two: Final = scenario.user(user_role="internal_user")
        email_three: Final = f"bulk-member-{uuid.uuid4().hex}@integration.test"
        user_three: Final = scenario.user(user_email=email_three, user_role="internal_user")
        members: Final = [
            {"role": "user", "user_id": user_one},
            {"role": "user", "user_id": user_two},
            {"role": "user", "user_email": email_three},
        ]
        added: Final = gateway.request(
            "POST",
            "/team/bulk_member_add",
            {"team_id": team, "members": members, "max_budget_in_team": 1},
        )
        assert added.status_code == 200, added.text
        response: Final = _BulkAddResponse.model_validate_json(added.text)
        assert response.team_id == team, added.text
        assert response.total_requested == 3, added.text
        assert response.successful_additions == 3, added.text
        assert response.failed_additions == 0, added.text
        assert response.results == [
            _BulkAddResult(user_id=user_one, success=True),
            _BulkAddResult(user_id=user_two, success=True),
            _BulkAddResult(user_id=user_three, user_email=email_three, success=True),
        ], added.text
        assert read_rows(
            'SELECT user_id, team_id FROM "LiteLLM_TeamMembership" WHERE team_id = %s ORDER BY user_id',
            (team,),
        ) == [{"user_id": user_id, "team_id": team} for user_id in sorted((user_one, user_two, user_three))], added.text
        assert read_rows(
            'SELECT members_with_roles FROM "LiteLLM_TeamTable" WHERE team_id = %s',
            (team,),
        ) == [
            {
                "members_with_roles": [
                    _roster_member(user_one, None),
                    _roster_member(user_two, None),
                    _roster_member(user_three, email_three),
                ]
            }
        ], added.text

        keys: Final = tuple(_key(scenario, team_id=team, user_id=user) for user in (user_one, user_two, user_three))
        successes: Final = tuple(_chat(gateway, model, key, f"bulk member {index}") for index, key in enumerate(keys))
        assert tuple(response.status_code for response in successes) == (200, 200, 200), tuple(
            response.text for response in successes
        )
        duplicate: Final = gateway.request(
            "POST",
            "/team/bulk_member_add",
            {"team_id": team, "members": members, "max_budget_in_team": 1},
        )
        assert duplicate.status_code == 200, duplicate.text
        failed: Final = _BulkAddResponse.model_validate_json(duplicate.text)
        assert failed.total_requested == 3, duplicate.text
        assert failed.successful_additions == 0, duplicate.text
        assert failed.failed_additions == 3, duplicate.text
        assert tuple(result.success for result in failed.results) == (False, False, False), duplicate.text
        assert all(result.error is not None for result in failed.results), duplicate.text
        assert read_rows(
            'SELECT members_with_roles FROM "LiteLLM_TeamTable" WHERE team_id = %s',
            (team,),
        ) == [
            {
                "members_with_roles": [
                    _roster_member(user_one, None),
                    _roster_member(user_two, None),
                    _roster_member(user_three, email_three),
                ]
            }
        ], duplicate.text
        observed: Final = _models(upstream)
        assert observed == (provider_model, provider_model, provider_model), repr(observed)


def test_bulk_delete_removes_exactly_the_named_members_and_revokes_their_access(
    gateway: Gateway, upstream: httpx.Client
) -> None:
    with gateway.scenario() as scenario:
        model, provider_model = _model(scenario)
        team: Final = scenario.team(models=[model])
        _remove_default_user(gateway, team)
        admin: Final = scenario.member(team, role="admin")
        email_member: Final = f"bulk-delete-member-{uuid.uuid4().hex}@integration.test"
        member: Final = scenario.user(user_email=email_member, user_role="internal_user")
        added: Final = gateway.request(
            "POST",
            "/team/member_add",
            {"team_id": team, "member": {"role": "user", "user_email": email_member}},
        )
        assert added.status_code == 200, added.text
        stayer: Final = scenario.member(team)
        admin_key: Final = _key(scenario, team_id=team, user_id=admin)
        member_key: Final = _key(scenario, team_id=team, user_id=member)
        stayer_key: Final = _key(scenario, team_id=team, user_id=stayer)
        personal_admin_key: Final = _key(scenario, user_id=admin)
        admin_warm: Final = _chat(gateway, model, admin_key, "bulk delete warm admin")
        assert admin_warm.status_code == 200, admin_warm.text
        member_warm: Final = _chat(gateway, model, member_key, "bulk delete warm member")
        assert member_warm.status_code == 200, member_warm.text
        stayer_warm: Final = _chat(gateway, model, stayer_key, "bulk delete warm stayer")
        assert stayer_warm.status_code == 200, stayer_warm.text
        observed_before_delete: Final = _models(upstream)
        assert observed_before_delete == (provider_model, provider_model, provider_model), repr(observed_before_delete)

        probe_user: Final = scenario.user()
        admin_route_before: Final = gateway.request(
            "POST",
            "/team/member_add",
            {"team_id": team, "member": {"role": "user", "user_id": probe_user}},
            key=personal_admin_key,
        )
        assert admin_route_before.status_code == 200, admin_route_before.text
        probe_removed: Final = gateway.request("POST", "/team/member_delete", {"team_id": team, "user_id": probe_user})
        assert probe_removed.status_code == 200, probe_removed.text
        deleted: Final = gateway.request(
            "POST",
            f"/management/v1/teams/{team}/members/bulk_delete",
            {
                "members": [
                    {"user_id": admin},
                    {"user_email": email_member},
                    {"user_id": admin},
                ]
            },
        )
        assert deleted.status_code == 200, deleted.text
        results: Final = _BulkDeleteResponse.model_validate_json(deleted.text).data
        assert results == (
            _BulkDeleteResult(user_id=admin, success=True),
            _BulkDeleteResult(user_email=email_member, success=True),
            _BulkDeleteResult(user_id=admin, success=False, error="Duplicate member in request"),
        ), deleted.text
        team_info: Final = _team_info(gateway, team)
        assert team_info.team_info.members_with_roles == [
            _RosterMember(role="user", user_id=stayer, user_email=None)
        ], repr(team_info)
        assert read_rows(
            'SELECT user_id, team_id FROM "LiteLLM_TeamMembership" WHERE team_id = %s ORDER BY user_id',
            (team,),
        ) == [{"user_id": stayer, "team_id": team}], deleted.text
        for removed_key in (admin_key, member_key):
            removed_hash: Final = sha256(removed_key.encode()).hexdigest()
            assert (
                read_rows(
                    'SELECT token FROM "LiteLLM_VerificationToken" WHERE token = %s',
                    (removed_hash,),
                )
                == []
            ), deleted.text

        denied_admin: Final = _chat(gateway, model, admin_key, "deleted admin key")
        _refused(denied_admin, 401, "token_not_found_in_db")
        observed_after_admin_denied: Final = _models(upstream)
        denied_member: Final = _chat(gateway, model, member_key, "deleted member key")
        _refused(denied_member, 401, "token_not_found_in_db")
        observed_after_member_denied: Final = _models(upstream)
        denied_admin_route: Final = gateway.request(
            "POST",
            "/team/member_add",
            {"team_id": team, "member": {"role": "user", "user_id": probe_user}},
            key=personal_admin_key,
        )
        assert denied_admin_route.status_code == 403, denied_admin_route.text
        denied_admin_response: Final = _HTTPErrorResponse.model_validate_json(denied_admin_route.text)
        assert "User not proxy admin OR team admin" in denied_admin_response.detail.error, denied_admin_route.text
        stayer_after: Final = _chat(gateway, model, stayer_key, "stayer after bulk delete")
        assert stayer_after.status_code == 200, stayer_after.text
        observed_after_stayer: Final = _models(upstream)
        assert observed_after_admin_denied == (), repr(observed_after_admin_denied)
        assert observed_after_member_denied == (), repr(observed_after_member_denied)
        assert (
            observed_before_delete + observed_after_admin_denied + observed_after_member_denied + observed_after_stayer
            == (
                provider_model,
                provider_model,
                provider_model,
                provider_model,
            )
        )
