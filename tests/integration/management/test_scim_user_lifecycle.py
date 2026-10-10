import uuid
from datetime import datetime, timezone
from hashlib import sha256
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario, object_value, string_value
from integration._support.database import read_rows
from pydantic import BaseModel, ConfigDict, Field, JsonValue

SCIM_HEADERS: Final = {"Content-Type": "application/scim+json"}
SCIM_CORE_USER_SCHEMA: Final = "urn:ietf:params:scim:schemas:core:2.0:User"
SCIM_ENTERPRISE_USER_SCHEMA: Final = "urn:ietf:params:scim:schemas:extension:enterprise:2.0:User"
SCIM_LIST_RESPONSE_SCHEMA: Final = "urn:ietf:params:scim:api:messages:2.0:ListResponse"


class ScimResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ScimName(ScimResponseModel):
    familyName: str | None = None
    givenName: str | None = None
    formatted: str | None = None
    middleName: str | None = None
    honorificPrefix: str | None = None
    honorificSuffix: str | None = None


class ScimEmail(ScimResponseModel):
    value: str
    type: str | None = None
    primary: bool | None = None


class ScimGroupRef(ScimResponseModel):
    value: str
    display: str | None = None
    type: str | None = None


class ScimUserMeta(ScimResponseModel):
    resourceType: str
    created: str | None = None
    lastModified: str | None = None


class ScimManager(ScimResponseModel):
    value: str | None = None
    displayName: str | None = None
    ref: str | None = Field(default=None, alias="$ref")


class ScimEnterpriseUser(ScimResponseModel):
    employeeNumber: str | None = None
    costCenter: str | None = None
    organization: str | None = None
    division: str | None = None
    department: str | None = None
    manager: ScimManager | None = None


class ScimMultiValuedAttribute(ScimResponseModel):
    value: str
    display: str | None = None
    type: str | None = None
    primary: bool | None = None


class ScimUserResponse(ScimResponseModel):
    schemas: list[str]
    id: str
    externalId: str | None = None
    userName: str
    displayName: str | None = None
    name: ScimName | None = None
    emails: list[ScimEmail] | None = None
    groups: list[ScimGroupRef] | None = None
    entitlements: list[ScimMultiValuedAttribute] | None = None
    roles: list[ScimMultiValuedAttribute] | None = None
    active: bool
    meta: ScimUserMeta
    enterprise: ScimEnterpriseUser | None = Field(default=None, alias=SCIM_ENTERPRISE_USER_SCHEMA)


class ScimListResponse(ScimResponseModel):
    schemas: list[str]
    totalResults: int
    startIndex: int
    itemsPerPage: int
    Resources: list[ScimUserResponse]


def _utc_datetime(value: str) -> datetime:
    parsed: Final = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _scim_request(
    gateway: Gateway,
    method: str,
    path: str,
    body: dict[str, JsonValue] | None = None,
) -> httpx.Response:
    return gateway.request(method, path, body, headers=SCIM_HEADERS)


def _delete_user_if_present(gateway: Gateway, user_id: str) -> None:
    if read_rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,)):
        deleted: Final = gateway.request("POST", "/user/delete", {"user_ids": [user_id]})
        assert deleted.status_code == 200 and deleted.json() == 1, deleted.text
    assert read_rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,)) == []


def _team_members(gateway: Gateway, team_id: str) -> list[dict[str, JsonValue]]:
    team_info: Final = object_value(gateway.get("/team/info", {"team_id": team_id})["team_info"])
    members: Final = team_info.get("members_with_roles") or []
    assert isinstance(members, list), members
    return [object_value(member) for member in members]


def _has_team_member(gateway: Gateway, team_id: str, user_id: str) -> bool:
    return any(
        member.get("user_id") == user_id and member.get("role") == "user" for member in _team_members(gateway, team_id)
    )


def _delete_key_if_present(gateway: Gateway, key: str) -> None:
    digest: Final = sha256(key.encode()).hexdigest()
    if read_rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE token = %s', (digest,)):
        deleted: Final = gateway.request("POST", "/key/delete", {"keys": [key]})
        assert deleted.status_code == 200, deleted.text
    assert read_rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE token = %s', (digest,)) == []


def _key_with_cleanup(scenario: Scenario, **fields: JsonValue) -> str:
    created: Final = scenario.gateway.post("/key/generate", fields)
    key: Final = string_value(created["key"])
    scenario.cleanups.callback(_delete_key_if_present, scenario.gateway, key)
    return key


def _assert_serving(gateway: Gateway, model: str, key: str) -> None:
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": "SCIM lifecycle integration"}]},
        key=key,
    )
    assert response.status_code == 200, response.text
    assert response.json()["usage"]["total_tokens"] == 40, response.text
    assert response.json()["choices"][0]["message"]["content"] == (
        "Hello! This is a mock response from the fake OpenAI endpoint."
    ), response.text


def test_okta_create_provisions_the_user_into_its_group_and_rejects_a_duplicate(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        team_info: Final = object_value(gateway.get("/team/info", {"team_id": team})["team_info"])
        team_alias: Final = string_value(team_info["team_alias"])
        suffix: Final = uuid.uuid4().hex
        user_name: Final = f"scim-{suffix}@example.com"
        user_email: Final = user_name
        external_id: Final = f"external-{suffix}"
        manager_id: Final = f"manager-{suffix}"
        schemas: Final = [SCIM_CORE_USER_SCHEMA, SCIM_ENTERPRISE_USER_SCHEMA]
        enterprise: Final = {
            "employeeNumber": f"employee-{suffix}",
            "department": "Research",
            "manager": {"value": manager_id},
        }
        requested: Final[dict[str, JsonValue]] = {
            "schemas": schemas,
            "userName": user_name,
            "name": {"givenName": "SCIM", "familyName": "Provisioned"},
            "emails": [{"primary": True, "type": "work", "value": user_email}],
            "externalId": external_id,
            "active": True,
            "groups": [{"value": team}],
            SCIM_ENTERPRISE_USER_SCHEMA: enterprise,
        }
        created: Final = _scim_request(gateway, "POST", "/scim/v2/Users", requested)
        scenario.cleanups.callback(_delete_user_if_present, gateway, user_name)
        assert created.status_code == 201, created.text
        expected_user: Final = ScimUserResponse(
            schemas=schemas,
            id=user_name,
            externalId=None,
            userName=user_name,
            displayName=user_name,
            active=True,
            meta=ScimUserMeta(resourceType="User"),
            name=ScimName(givenName="SCIM", familyName="Provisioned"),
            emails=[ScimEmail(value=user_email, type=None, primary=True)],
            groups=[ScimGroupRef(value=team, display=team_alias, type="direct")],
            enterprise=ScimEnterpriseUser(
                employeeNumber=f"employee-{suffix}",
                department="Research",
                manager=ScimManager(value=manager_id),
            ),
        )
        created_user: Final = ScimUserResponse.model_validate(created.json())
        create_exclude: Final = {"externalId": True, "meta": {"created": True, "lastModified": True}}
        assert created_user.model_dump(by_alias=True, exclude=create_exclude) == expected_user.model_dump(
            by_alias=True,
            exclude=create_exclude,
        ), created.text
        readback_response: Final = _scim_request(gateway, "GET", f"/scim/v2/Users/{user_name}")
        assert readback_response.status_code == 200, readback_response.text
        readback_user: Final = ScimUserResponse.model_validate(readback_response.json())
        metadata_rows: Final = read_rows(
            'SELECT created_at::text AS created_at, updated_at::text AS updated_at '
            'FROM "LiteLLM_UserTable" WHERE user_id = %s',
            (user_name,),
        )
        assert len(metadata_rows) == 1, (metadata_rows, readback_response.text)
        created_at: Final = string_value(metadata_rows[0]["created_at"])
        updated_at: Final = string_value(metadata_rows[0]["updated_at"])
        created_at_utc: Final = _utc_datetime(created_at)
        updated_at_utc: Final = _utc_datetime(updated_at)
        assert readback_user.meta.created is not None, readback_response.text
        assert readback_user.meta.lastModified is not None, readback_response.text
        assert _utc_datetime(readback_user.meta.created) == created_at_utc, (
            created_at,
            readback_user.meta.created,
            readback_response.text,
        )
        assert _utc_datetime(readback_user.meta.lastModified) == updated_at_utc, (
            updated_at,
            readback_user.meta.lastModified,
            readback_response.text,
        )
        expected_readback_user: Final = expected_user.model_copy(
            update={
                "meta": ScimUserMeta(
                    resourceType="User",
                    created=created_at_utc.isoformat(),
                    lastModified=updated_at_utc.isoformat(),
                )
            }
        )
        assert readback_user.model_dump(by_alias=True, exclude={"externalId": True}) == (
            expected_readback_user.model_dump(by_alias=True, exclude={"externalId": True})
        ), readback_response.text

        user_rows: Final = read_rows(
            'SELECT user_id, user_email, user_role, teams, metadata FROM "LiteLLM_UserTable" WHERE user_id = %s',
            (user_name,),
        )
        assert len(user_rows) == 1, user_rows
        assert user_rows[0]["user_id"] == user_name, user_rows
        assert user_rows[0]["user_email"] == user_email, user_rows
        assert user_rows[0]["user_role"] == "internal_user_viewer", user_rows
        assert user_rows[0]["teams"] == [team], user_rows
        metadata: Final = object_value(user_rows[0]["metadata"])
        assert metadata.get("scim_metadata") == {
            "givenName": "SCIM",
            "familyName": "Provisioned",
        }, metadata
        assert metadata.get("scim_enterprise") == enterprise, metadata
        assert _has_team_member(gateway, team, user_name), _team_members(gateway, team)
        key: Final = scenario.key(user_id=user_name, team_id=team, models=[model])
        _assert_serving(gateway, model, key)

        duplicate: Final = _scim_request(gateway, "POST", "/scim/v2/Users", requested)
        assert duplicate.status_code == 409, duplicate.text
        assert object_value(duplicate.json()) == {
            "detail": {"error": f"User already exists with username: {user_name}"}
        }, duplicate.text
        assert read_rows(
            'SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s',
            (user_name,),
        ) == [{"user_id": user_name}]


def test_scim_create_adopts_an_existing_user_by_email_and_keeps_its_key_and_teams(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        suffix: Final = uuid.uuid4().hex
        user_email: Final = f"existing-{suffix}@example.com"
        existing_user: Final = scenario.user(user_email=user_email, user_role="internal_user")
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        team_info: Final = object_value(gateway.get("/team/info", {"team_id": team})["team_info"])
        team_alias: Final = string_value(team_info["team_alias"])
        added_member: Final = gateway.request(
            "POST",
            "/team/member_add",
            {"team_id": team, "member": {"role": "user", "user_id": existing_user}},
        )
        assert added_member.status_code == 200, added_member.text
        key: Final = _key_with_cleanup(scenario, user_id=existing_user, team_id=team, models=[model])
        _assert_serving(gateway, model, key)
        team_before: Final = _team_members(gateway, team)
        user_before: Final = read_rows(
            'SELECT user_id, user_email, user_role, teams FROM "LiteLLM_UserTable" WHERE user_id = %s',
            (existing_user,),
        )
        idp_user_name: Final = f"idp-login-{suffix}"
        created: Final = _scim_request(
            gateway,
            "POST",
            "/scim/v2/Users",
            {
                "schemas": [SCIM_CORE_USER_SCHEMA],
                "userName": idp_user_name,
                "emails": [{"value": user_email, "primary": True, "type": "work"}],
            },
        )
        assert created.status_code == 201, created.text
        created_user: Final = ScimUserResponse.model_validate(created.json())
        assert created_user.id == existing_user, created.text
        readback: Final = _scim_request(gateway, "GET", f"/scim/v2/Users/{existing_user}")
        assert readback.status_code == 200, readback.text
        readback_user: Final = ScimUserResponse.model_validate(readback.json())
        assert readback_user.id == existing_user, readback.text
        assert readback_user.groups == [
            ScimGroupRef(value=team, display=team_alias, type="direct")
        ], readback.text
        assert len(read_rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_email = %s', (user_email,))) == 1
        assert (
            read_rows(
                'SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s',
                (idp_user_name,),
            )
            == []
        )
        assert (
            read_rows(
                'SELECT user_id, user_email, user_role, teams FROM "LiteLLM_UserTable" WHERE user_id = %s',
                (existing_user,),
            )
            == user_before
        )
        assert _team_members(gateway, team) == team_before
        _assert_serving(gateway, model, key)


def test_profile_put_without_groups_keeps_memberships_and_a_new_groups_list_moves_them(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model_a: Final = scenario.model()
        model_b: Final = scenario.model()
        team_a: Final = scenario.team(models=[model_a])
        team_b: Final = scenario.team(models=[model_b])
        team_a_info: Final = object_value(gateway.get("/team/info", {"team_id": team_a})["team_info"])
        team_b_info: Final = object_value(gateway.get("/team/info", {"team_id": team_b})["team_info"])
        alias_a: Final = string_value(team_a_info["team_alias"])
        alias_b: Final = string_value(team_b_info["team_alias"])
        suffix: Final = uuid.uuid4().hex
        user_name: Final = f"scim-move-{suffix}@example.com"
        external_id: Final = f"entra-object-{suffix}"
        profile: Final[dict[str, JsonValue]] = {
            "schemas": [SCIM_CORE_USER_SCHEMA],
            "userName": user_name,
            "name": {"givenName": "Group", "familyName": "Member"},
            "emails": [{"value": user_name, "primary": True, "type": "work"}],
            "externalId": external_id,
            "active": True,
        }
        created: Final = _scim_request(
            gateway,
            "POST",
            "/scim/v2/Users",
            {**profile, "groups": [{"value": team_a}]},
        )
        scenario.cleanups.callback(_delete_user_if_present, gateway, user_name)
        assert created.status_code == 201, created.text
        created_user: Final = ScimUserResponse.model_validate(created.json())
        assert created_user.groups == [
            ScimGroupRef(value=team_a, display=alias_a, type="direct")
        ], created.text
        key_a: Final = _key_with_cleanup(scenario, user_id=user_name, team_id=team_a, models=[model_a])
        _assert_serving(gateway, model_a, key_a)

        without_groups: Final = _scim_request(gateway, "PUT", f"/scim/v2/Users/{user_name}", profile)
        assert without_groups.status_code == 200, without_groups.text
        assert read_rows(
            'SELECT sso_user_id FROM "LiteLLM_UserTable" WHERE user_id = %s',
            (user_name,),
        ) == [{"sso_user_id": external_id}], without_groups.text
        assert object_value(gateway.get("/user/info", {"user_id": user_name})["user_info"])["sso_user_id"] == (
            external_id
        ), without_groups.text
        without_groups_user: Final = ScimUserResponse.model_validate(without_groups.json())
        assert without_groups_user.groups == [
            ScimGroupRef(value=team_a, display=alias_a, type="direct")
        ], without_groups.text
        without_groups_readback: Final = _scim_request(
            gateway,
            "GET",
            f"/scim/v2/Users/{user_name}",
        )
        assert without_groups_readback.status_code == 200, without_groups_readback.text
        without_groups_readback_user: Final = ScimUserResponse.model_validate(without_groups_readback.json())
        assert without_groups_readback_user.groups == [
            ScimGroupRef(value=team_a, display=alias_a, type="direct")
        ], (
            without_groups_readback.text
        )
        assert _has_team_member(gateway, team_a, user_name), _team_members(gateway, team_a)
        _assert_serving(gateway, model_a, key_a)

        empty_groups: Final = _scim_request(
            gateway,
            "PUT",
            f"/scim/v2/Users/{user_name}",
            {**profile, "groups": []},
        )
        assert empty_groups.status_code == 200, empty_groups.text
        empty_groups_user: Final = ScimUserResponse.model_validate(empty_groups.json())
        assert empty_groups_user.groups == [
            ScimGroupRef(value=team_a, display=alias_a, type="direct")
        ], empty_groups.text
        empty_groups_readback: Final = _scim_request(gateway, "GET", f"/scim/v2/Users/{user_name}")
        assert empty_groups_readback.status_code == 200, empty_groups_readback.text
        empty_groups_readback_user: Final = ScimUserResponse.model_validate(empty_groups_readback.json())
        assert empty_groups_readback_user.groups == [
            ScimGroupRef(value=team_a, display=alias_a, type="direct")
        ], (
            empty_groups_readback.text
        )
        assert _has_team_member(gateway, team_a, user_name), _team_members(gateway, team_a)
        _assert_serving(gateway, model_a, key_a)

        moved: Final = _scim_request(
            gateway,
            "PUT",
            f"/scim/v2/Users/{user_name}",
            {**profile, "groups": [{"value": team_b}]},
        )
        assert moved.status_code == 200, moved.text
        moved_user: Final = ScimUserResponse.model_validate(moved.json())
        assert moved_user.groups == [
            ScimGroupRef(value=team_b, display=alias_b, type="direct")
        ], moved.text
        moved_readback: Final = _scim_request(gateway, "GET", f"/scim/v2/Users/{user_name}")
        assert moved_readback.status_code == 200, moved_readback.text
        moved_readback_user: Final = ScimUserResponse.model_validate(moved_readback.json())
        assert moved_readback_user.groups == [
            ScimGroupRef(value=team_b, display=alias_b, type="direct")
        ], moved_readback.text
        assert not _has_team_member(gateway, team_a, user_name), _team_members(gateway, team_a)
        assert _has_team_member(gateway, team_b, user_name), _team_members(gateway, team_b)
        refused: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model_a, "messages": [{"role": "user", "content": "SCIM moved membership"}]},
            key=key_a,
        )
        expected_refusal: Final = {
            "error": {
                "message": (
                    "Authentication Error, Invalid proxy server token passed. "
                    f"Received API Key = sk-...{key_a[-4:]}, Key Hash (Token) ={sha256(key_a.encode()).hexdigest()}. "
                    "Unable to find token in cache or `LiteLLM_VerificationTokenTable`"
                ),
                "type": "token_not_found_in_db",
                "param": "key",
                "code": "401",
            }
        }
        assert refused.status_code == 401, refused.text
        assert object_value(refused.json()) == expected_refusal, refused.text
        key_b: Final = _key_with_cleanup(scenario, user_id=user_name, team_id=team_b, models=[model_b])
        _assert_serving(gateway, model_b, key_b)


def test_scim_create_persists_external_id_as_the_sso_user_id(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: POST /scim/v2/Users drops externalId, so the 201 resource and GET readback return externalId null and "
        "the user's sso_user_id stays NULL"
    )
    with gateway.scenario() as scenario:
        suffix: Final = uuid.uuid4().hex
        user_name: Final = f"scim-ext-{suffix}@example.com"
        external_id: Final = f"okta-{suffix}"
        created: Final = _scim_request(
            gateway,
            "POST",
            "/scim/v2/Users",
            {
                "schemas": [SCIM_CORE_USER_SCHEMA],
                "userName": user_name,
                "emails": [{"value": user_name, "primary": True, "type": "work"}],
                "externalId": external_id,
                "active": True,
            },
        )
        scenario.cleanups.callback(_delete_user_if_present, gateway, user_name)
        assert created.status_code == 201, created.text
        created_user: Final = ScimUserResponse.model_validate(created.json())
        assert created_user.externalId == external_id, created.text
        readback: Final = _scim_request(gateway, "GET", f"/scim/v2/Users/{user_name}")
        assert readback.status_code == 200, readback.text
        readback_user: Final = ScimUserResponse.model_validate(readback.json())
        assert readback_user.externalId == external_id, readback.text
        assert read_rows(
            'SELECT sso_user_id FROM "LiteLLM_UserTable" WHERE user_id = %s',
            (user_name,),
        ) == [{"sso_user_id": external_id}], created.text
        assert object_value(gateway.get("/user/info", {"user_id": user_name})["user_info"])["sso_user_id"] == (
            external_id
        ), created.text


def test_okta_username_filter_finds_users_by_email_and_by_id(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        suffix: Final = uuid.uuid4().hex
        email_a: Final = f"filter-existing-{suffix}@example.com"
        email_b: Final = f"filter-scim-{suffix}@example.com"
        existing: Final = scenario.user(user_email=email_a, user_role="internal_user")
        created: Final = _scim_request(
            gateway,
            "POST",
            "/scim/v2/Users",
            {
                "schemas": [SCIM_CORE_USER_SCHEMA],
                "userName": email_b,
                "emails": [{"value": email_b, "primary": True, "type": "work"}],
            },
        )
        scenario.cleanups.callback(_delete_user_if_present, gateway, email_b)
        assert created.status_code == 201, created.text

        def lookup(value: str) -> tuple[ScimListResponse, str]:
            response: Final = gateway.request(
                "GET",
                "/scim/v2/Users",
                params={"filter": f'userName eq "{value}"', "startIndex": "1", "count": "100"},
                headers={"Accept": "application/scim+json"},
            )
            assert response.status_code == 200, response.text
            return ScimListResponse.model_validate(response.json()), response.text

        existing_results, existing_text = lookup(email_a)
        assert (
            existing_results.schemas,
            existing_results.totalResults,
            existing_results.startIndex,
            existing_results.itemsPerPage,
            [(resource.id, resource.userName) for resource in existing_results.Resources],
        ) == (
            [SCIM_LIST_RESPONSE_SCHEMA],
            1,
            1,
            1,
            [(existing, email_a)],
        ), existing_text

        existing_id_results, existing_id_text = lookup(existing)
        assert (
            existing_id_results.schemas,
            existing_id_results.totalResults,
            existing_id_results.startIndex,
            existing_id_results.itemsPerPage,
            [(resource.id, resource.userName) for resource in existing_id_results.Resources],
        ) == (
            ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
            1,
            1,
            1,
            [(existing, email_a)],
        ), existing_id_text

        scim_results, scim_text = lookup(email_b)
        assert (
            scim_results.schemas,
            scim_results.totalResults,
            scim_results.startIndex,
            scim_results.itemsPerPage,
            [(resource.id, resource.userName) for resource in scim_results.Resources],
        ) == (
            ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
            1,
            1,
            1,
            [(email_b, email_b)],
        ), scim_text

        absent_results, absent_text = lookup(f"absent-{suffix}@example.com")
        assert (
            absent_results.schemas,
            absent_results.totalResults,
            absent_results.startIndex,
            absent_results.itemsPerPage,
            absent_results.Resources,
        ) == (
            ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
            0,
            1,
            0,
            [],
        ), absent_text
