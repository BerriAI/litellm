import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Mapping

import httpx
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

SCIM_HEADERS: Final = {"Content-Type": "application/scim+json"}
SCIM_CORE_USER_SCHEMA: Final = "urn:ietf:params:scim:schemas:core:2.0:User"
SCIM_CORE_GROUP_SCHEMA: Final = "urn:ietf:params:scim:schemas:core:2.0:Group"
SCIM_PATCH_SCHEMA: Final = "urn:ietf:params:scim:api:messages:2.0:PatchOp"
SCIM_LIST_RESPONSE_SCHEMA: Final = "urn:ietf:params:scim:api:messages:2.0:ListResponse"
SCIM_CONFIG: Final = TypeAdapter(dict[str, JsonValue])


class ScimGroupMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resourceType: str
    created: str
    lastModified: str


class ScimGroupMember(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    display: str
    type: str


class ScimGroupResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schemas: list[str]
    id: str
    externalId: str | None = None
    meta: ScimGroupMeta
    displayName: str
    members: list[ScimGroupMember]


class ScimGroupListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schemas: list[str]
    totalResults: int
    startIndex: int
    itemsPerPage: int
    Resources: list[ScimGroupResponse]


class ScimUserResponse(BaseModel):
    schemas: list[str]
    id: str
    userName: str | None = None


@dataclass(frozen=True, slots=True)
class ScimAdminGroupFixture:
    gateway: Gateway
    scenario: Scenario
    group_id: str
    group_name: str
    created_group: ScimGroupResponse
    user_id: str
    member_display: str
    user_key: str
    other_user: str
    default_role: str


def _owned_scim_config(path: Path, settings: Mapping[str, JsonValue]) -> Path:
    config: Final = object_value(
        SCIM_CONFIG.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    )
    litellm_settings: Final = object_value(config["litellm_settings"])
    path.write_text(
        yaml.safe_dump(
            {
                **config,
                "litellm_settings": {**litellm_settings, **settings},
            }
        )
    )
    return path


def _delete_user_if_present(gateway: Gateway, user_id: str) -> None:
    if read_rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,)):
        deleted: Final = gateway.request("POST", "/user/delete", {"user_ids": [user_id]})
        assert deleted.status_code == 200 and deleted.json() == 1, deleted.text
    assert read_rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,)) == []


def _delete_group_if_present(gateway: Gateway, group_id: str) -> None:
    if read_rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = %s', (group_id,)):
        deleted: Final = gateway.request(
            "DELETE",
            f"/scim/v2/Groups/{group_id}",
            headers=SCIM_HEADERS,
        )
        assert deleted.status_code == 204, deleted.text
    assert read_rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = %s', (group_id,)) == []


def _user_role(gateway: Gateway, user_id: str) -> str:
    info: Final = object_value(gateway.get("/user/info", {"user_id": user_id})["user_info"])
    return string_value(info["user_role"])


def _database_user_role(user_id: str) -> str:
    rows: Final = read_rows('SELECT user_role FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,))
    return string_value(rows[0]["user_role"])


def _expected_scim_write_denial(route: str, user_id: str, user_role: str) -> dict[str, JsonValue]:
    masked_user_id: Final = f"{user_id[:6]}{'*' * (len(user_id) - 8)}{user_id[-2:]}"
    return {
        "error": {
            "message": (
                "Authentication Error, Only proxy admin can be used to generate, delete, update info for new "
                f"keys/users/teams. Route={route}. Your role={user_role}. Your user_id={masked_user_id}"
            ),
            "type": "auth_error",
            "param": "None",
            "code": "401",
        }
    }


def _assert_scim_group_resource(
    response: httpx.Response,
    group_id: str,
    display_name: str,
    members: list[ScimGroupMember],
    expected_created: str | None,
) -> ScimGroupResponse:
    actual: Final = ScimGroupResponse.model_validate(response.json())
    created: Final = actual.meta.created if expected_created is None else expected_created
    assert isinstance(created, str), response.text
    assert isinstance(actual.meta.lastModified, str), response.text
    expected: Final = ScimGroupResponse(
        schemas=[SCIM_CORE_GROUP_SCHEMA],
        id=group_id,
        externalId=None,
        meta=ScimGroupMeta(
            resourceType="Group",
            created=created,
            lastModified=actual.meta.lastModified,
        ),
        displayName=display_name,
        members=members,
    )
    assert actual == expected, response.text
    return actual


def _assert_scim_group_resource_except_members(
    response: httpx.Response,
    group_id: str,
    display_name: str,
) -> ScimGroupResponse:
    actual: Final = ScimGroupResponse.model_validate(response.json())
    expected: Final = ScimGroupResponse(
        schemas=[SCIM_CORE_GROUP_SCHEMA],
        id=group_id,
        externalId=None,
        meta=ScimGroupMeta(
            resourceType="Group",
            created=actual.meta.created,
            lastModified=actual.meta.lastModified,
        ),
        displayName=display_name,
        members=actual.members,
    )
    assert actual.model_dump(exclude={"members"}) == expected.model_dump(exclude={"members"}), response.text
    return actual


def _group_filter_response(gateway: Gateway, group_name: str) -> httpx.Response:
    return gateway.request(
        "GET",
        "/scim/v2/Groups",
        params={"filter": f'displayName eq "{group_name}"', "startIndex": "1", "count": "100"},
        headers={"Accept": "application/scim+json"},
    )


def _assert_scim_group_list_response(
    response: httpx.Response,
    group_id: str | None,
    group_name: str,
    members: list[ScimGroupMember],
) -> ScimGroupListResponse:
    actual: Final = ScimGroupListResponse.model_validate(response.json())
    expected_results: Final = 0 if group_id is None else 1
    assert len(actual.Resources) == expected_results, response.text
    expected_resources: Final = (
        []
        if group_id is None
        else [
            ScimGroupResponse(
                schemas=[SCIM_CORE_GROUP_SCHEMA],
                id=group_id,
                externalId=None,
                meta=ScimGroupMeta(
                    resourceType="Group",
                    created=actual.Resources[0].meta.created,
                    lastModified=actual.Resources[0].meta.lastModified,
                ),
                displayName=group_name,
                members=members,
            )
        ]
    )
    expected: Final = ScimGroupListResponse(
        schemas=[SCIM_LIST_RESPONSE_SCHEMA],
        totalResults=expected_results,
        startIndex=1,
        itemsPerPage=expected_results,
        Resources=expected_resources,
    )
    assert actual == expected, response.text
    return actual


def _group_membership_operations(
    user_id: str,
    spelling: Literal["okta", "entra"],
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    if spelling == "okta":
        return (
            {"op": "add", "path": "members", "value": [{"value": user_id}]},
            {"op": "remove", "path": f'members[value eq "{user_id}"]'},
        )
    return (
        {"op": "Add", "path": "members", "value": [{"value": user_id}]},
        {"op": "Remove", "path": "members", "value": [{"value": user_id}]},
    )


def _change_group_membership(
    gateway: Gateway,
    group_id: str,
    group_name: str,
    user_id: str,
    member_display: str,
    action: Literal["add", "remove"],
    method: Literal["PATCH", "PUT"],
    spelling: Literal["okta", "entra"] = "okta",
) -> httpx.Response:
    if method == "PUT":
        members: Final = [{"value": user_id, "display": member_display}] if action == "add" else []
        return gateway.request(
            "PUT",
            f"/scim/v2/Groups/{group_id}",
            {
                "schemas": [SCIM_CORE_GROUP_SCHEMA],
                "id": group_id,
                "displayName": group_name,
                "members": members,
            },
            headers=SCIM_HEADERS,
        )
    add_operation, remove_operation = _group_membership_operations(user_id, spelling)
    operation: Final = add_operation if action == "add" else remove_operation
    return gateway.request(
        "PATCH",
        f"/scim/v2/Groups/{group_id}",
        {
            "schemas": [SCIM_PATCH_SCHEMA],
            "Operations": [operation],
        },
        headers=SCIM_HEADERS,
    )


def _generate_key_for_user(scenario: Scenario, user_key: str, target_user: str) -> httpx.Response:
    response: Final = scenario.gateway.request(
        "POST",
        "/key/generate",
        {"user_id": target_user},
        key=user_key,
    )
    if response.status_code == 200:
        generated_key: Final = string_value(object_value(response.json())["key"])
        scenario.cleanups.callback(scenario.delete_key, generated_key)
    return response


@contextmanager
def _scim_admin_group_fixture(
    gateway: Gateway,
    tmp_path: Path,
    group_name: str,
) -> Iterator[ScimAdminGroupFixture]:
    config: Final = _owned_scim_config(tmp_path / "scim-admin-group.yaml", {"scim_admin_group": group_name})
    with owned_proxy_process(gateway, tmp_path, {}, config=config) as owned, owned.gateway.scenario() as scenario:
        group_id: Final = str(uuid.uuid4())
        unrelated_group_alias: Final = f"{group_name}-unrelated"
        unrelated_group_id: Final = scenario.team(team_alias=unrelated_group_alias)
        assert read_rows(
            'SELECT team_alias FROM "LiteLLM_TeamTable" WHERE team_id = %s',
            (unrelated_group_id,),
        ) == [{"team_alias": unrelated_group_alias}]
        before_filter: Final = _group_filter_response(owned.gateway, group_name)
        assert before_filter.status_code == 200, before_filter.text
        _assert_scim_group_list_response(before_filter, None, group_name, [])
        group_response: Final = owned.gateway.request(
            "POST",
            "/scim/v2/Groups",
            {
                "schemas": [SCIM_CORE_GROUP_SCHEMA],
                "id": group_id,
                "displayName": group_name,
                "members": [],
            },
            headers=SCIM_HEADERS,
        )
        assert group_response.status_code == 201, group_response.text
        created_group: Final = _assert_scim_group_resource(
            group_response,
            group_id,
            group_name,
            [],
            None,
        )
        scenario.cleanups.callback(_delete_group_if_present, owned.gateway, group_id)
        after_create_filter: Final = _group_filter_response(owned.gateway, group_name)
        assert after_create_filter.status_code == 200, after_create_filter.text
        _assert_scim_group_list_response(after_create_filter, group_id, group_name, [])
        user: Final = f"scim-admin-user-{uuid.uuid4().hex}"
        created_user: Final = owned.gateway.request(
            "POST",
            "/scim/v2/Users",
            {
                "schemas": [SCIM_CORE_USER_SCHEMA],
                "userName": user,
                "name": {"givenName": "Group", "familyName": "Member"},
            },
            headers=SCIM_HEADERS,
        )
        scenario.cleanups.callback(_delete_user_if_present, owned.gateway, user)
        assert created_user.status_code == 201, created_user.text
        created_user_response: Final = ScimUserResponse.model_validate(created_user.json())
        assert created_user_response.id == user, created_user.text
        default_role: Final = _database_user_role(user)
        assert _user_role(owned.gateway, user) == default_role
        member_rows: Final = read_rows(
            'SELECT user_email FROM "LiteLLM_UserTable" WHERE user_id = %s',
            (user,),
        )
        member_email: Final = member_rows[0]["user_email"]
        member_display: Final = member_email if isinstance(member_email, str) and member_email else user
        user_key: Final = scenario.key(user_id=user)
        other_user: Final = scenario.user(user_role="internal_user")
        yield ScimAdminGroupFixture(
            gateway=owned.gateway,
            scenario=scenario,
            group_id=group_id,
            group_name=group_name,
            created_group=created_group,
            user_id=user,
            member_display=member_display,
            user_key=user_key,
            other_user=other_user,
            default_role=default_role,
        )


@pytest.mark.timeout(300)
@pytest.mark.parametrize("spelling", ["okta", "entra"], ids=["okta", "entra"])
def test_scim_admin_group_promotes_and_demotes_group_members(
    gateway: Gateway,
    tmp_path: Path,
    spelling: Literal["okta", "entra"],
) -> None:
    group_name: Final = f"scim-admin-{uuid.uuid4().hex}"
    with _scim_admin_group_fixture(gateway, tmp_path, group_name) as fixture:
        owned: Final = fixture
        scenario: Final = fixture.scenario
        group_id: Final = fixture.group_id
        created_group: Final = fixture.created_group
        user: Final = fixture.user_id
        member_display: Final = fixture.member_display
        user_key: Final = fixture.user_key
        other_user: Final = fixture.other_user
        default_role: Final = fixture.default_role
        initial_denial: Final = _generate_key_for_user(scenario, user_key, other_user)
        assert initial_denial.status_code == 401, initial_denial.text
        assert object_value(initial_denial.json()) == _expected_scim_write_denial(
            "/key/generate",
            user,
            default_role,
        ), initial_denial.text

        expected_member: Final = ScimGroupMember(value=user, display=member_display, type="User")
        added: Final = _change_group_membership(
            owned.gateway,
            group_id,
            group_name,
            user,
            member_display,
            "add",
            "PATCH",
            spelling,
        )
        assert added.status_code == 200, added.text
        after_add_filter: Final = _group_filter_response(owned.gateway, group_name)
        assert after_add_filter.status_code == 200, after_add_filter.text
        _assert_scim_group_list_response(after_add_filter, group_id, group_name, [expected_member])
        added_group: Final = _assert_scim_group_resource(
            added,
            group_id,
            group_name,
            [expected_member],
            created_group.meta.created,
        )
        assert added_group.members == [expected_member], added.text
        added_readback: Final = owned.gateway.request(
            "GET",
            f"/scim/v2/Groups/{group_id}",
            headers={"Accept": "application/scim+json"},
        )
        assert added_readback.status_code == 200, added_readback.text
        _assert_scim_group_resource(
            added_readback,
            group_id,
            group_name,
            [expected_member],
            created_group.meta.created,
        )
        assert eventually(lambda: _user_role(owned.gateway, user), lambda role: role == "proxy_admin") == "proxy_admin"
        assert eventually(lambda: _database_user_role(user), lambda role: role == "proxy_admin") == "proxy_admin"
        allowed: Final = eventually(
            lambda: _generate_key_for_user(scenario, user_key, other_user),
            lambda response: response.status_code == 200,
        )
        assert allowed.status_code == 200, allowed.text

        removed: Final = _change_group_membership(
            owned.gateway,
            group_id,
            group_name,
            user,
            member_display,
            "remove",
            "PATCH",
            spelling,
        )
        assert removed.status_code == 200, removed.text
        _assert_scim_group_resource(
            removed,
            group_id,
            group_name,
            [],
            created_group.meta.created,
        )
        removed_readback: Final = owned.gateway.request(
            "GET",
            f"/scim/v2/Groups/{group_id}",
            headers={"Accept": "application/scim+json"},
        )
        assert removed_readback.status_code == 200, removed_readback.text
        _assert_scim_group_resource(
            removed_readback,
            group_id,
            group_name,
            [],
            created_group.meta.created,
        )
        assert eventually(lambda: _user_role(owned.gateway, user), lambda role: role == default_role) == default_role
        assert eventually(lambda: _database_user_role(user), lambda role: role == default_role) == default_role
        refused_again: Final = eventually(
            lambda: _generate_key_for_user(scenario, user_key, other_user),
            lambda response: response.status_code == 401,
        )
        assert refused_again.status_code == 401, refused_again.text
        assert object_value(refused_again.json()) == _expected_scim_write_denial(
            "/key/generate",
            user,
            default_role,
        ), refused_again.text

        second_user: Final = f"scim-admin-user-{uuid.uuid4().hex}"
        created_second_user: Final = owned.gateway.request(
            "POST",
            "/scim/v2/Users",
            {
                "schemas": [SCIM_CORE_USER_SCHEMA],
                "userName": second_user,
                "name": {"givenName": "Display", "familyName": "Member"},
                "groups": [{"value": group_id, "display": group_name}],
            },
            headers=SCIM_HEADERS,
        )
        scenario.cleanups.callback(_delete_user_if_present, owned.gateway, second_user)
        assert created_second_user.status_code == 201, created_second_user.text
        created_second_user_response: Final = ScimUserResponse.model_validate(created_second_user.json())
        assert created_second_user_response.id == second_user, created_second_user.text
        second_key: Final = scenario.key(user_id=second_user)
        assert eventually(lambda: _user_role(owned.gateway, second_user), lambda role: role == "proxy_admin") == (
            "proxy_admin"
        )
        assert eventually(lambda: _database_user_role(second_user), lambda role: role == "proxy_admin") == (
            "proxy_admin"
        )
        second_allowed: Final = eventually(
            lambda: _generate_key_for_user(scenario, second_key, other_user),
            lambda response: response.status_code == 200,
        )
        assert second_allowed.status_code == 200, second_allowed.text

        deleted_group: Final = owned.gateway.request(
            "DELETE",
            f"/scim/v2/Groups/{group_id}",
            headers=SCIM_HEADERS,
        )
        assert deleted_group.status_code == 204, deleted_group.text
        assert eventually(lambda: _user_role(owned.gateway, second_user), lambda role: role == default_role) == (
            default_role
        )
        assert eventually(lambda: _database_user_role(second_user), lambda role: role == default_role) == default_role


@pytest.mark.timeout(300)
def test_scim_admin_group_put_members_promotes_and_demotes(gateway: Gateway, tmp_path: Path) -> None:
    group_name: Final = f"scim-admin-put-{uuid.uuid4().hex}"
    with _scim_admin_group_fixture(gateway, tmp_path, group_name) as fixture:
        expected_member: Final = ScimGroupMember(
            value=fixture.user_id,
            display=fixture.member_display,
            type="User",
        )
        initial_denial: Final = _generate_key_for_user(
            fixture.scenario,
            fixture.user_key,
            fixture.other_user,
        )
        assert initial_denial.status_code == 401, initial_denial.text
        assert object_value(initial_denial.json()) == _expected_scim_write_denial(
            "/key/generate",
            fixture.user_id,
            fixture.default_role,
        ), initial_denial.text

        added: Final = _change_group_membership(
            fixture.gateway,
            fixture.group_id,
            fixture.group_name,
            fixture.user_id,
            fixture.member_display,
            "add",
            "PUT",
        )
        assert added.status_code == 200, added.text
        _assert_scim_group_resource_except_members(
            added,
            fixture.group_id,
            fixture.group_name,
        )
        added_readback: Final = fixture.gateway.request(
            "GET",
            f"/scim/v2/Groups/{fixture.group_id}",
            headers={"Accept": "application/scim+json"},
        )
        assert added_readback.status_code == 200, added_readback.text
        _assert_scim_group_resource(
            added_readback,
            fixture.group_id,
            fixture.group_name,
            [expected_member],
            fixture.created_group.meta.created,
        )
        added_filter: Final = _group_filter_response(fixture.gateway, fixture.group_name)
        assert added_filter.status_code == 200, added_filter.text
        _assert_scim_group_list_response(
            added_filter,
            fixture.group_id,
            fixture.group_name,
            [expected_member],
        )
        assert eventually(
            lambda: _user_role(fixture.gateway, fixture.user_id),
            lambda role: role == "proxy_admin",
        ) == "proxy_admin"
        assert eventually(
            lambda: _database_user_role(fixture.user_id),
            lambda role: role == "proxy_admin",
        ) == "proxy_admin"
        allowed: Final = eventually(
            lambda: _generate_key_for_user(fixture.scenario, fixture.user_key, fixture.other_user),
            lambda response: response.status_code == 200,
        )
        assert allowed.status_code == 200, allowed.text

        removed: Final = _change_group_membership(
            fixture.gateway,
            fixture.group_id,
            fixture.group_name,
            fixture.user_id,
            fixture.member_display,
            "remove",
            "PUT",
        )
        assert removed.status_code == 200, removed.text
        _assert_scim_group_resource_except_members(
            removed,
            fixture.group_id,
            fixture.group_name,
        )
        removed_readback: Final = fixture.gateway.request(
            "GET",
            f"/scim/v2/Groups/{fixture.group_id}",
            headers={"Accept": "application/scim+json"},
        )
        assert removed_readback.status_code == 200, removed_readback.text
        _assert_scim_group_resource(
            removed_readback,
            fixture.group_id,
            fixture.group_name,
            [],
            fixture.created_group.meta.created,
        )
        removed_filter: Final = _group_filter_response(fixture.gateway, fixture.group_name)
        assert removed_filter.status_code == 200, removed_filter.text
        _assert_scim_group_list_response(
            removed_filter,
            fixture.group_id,
            fixture.group_name,
            [],
        )
        assert eventually(
            lambda: _user_role(fixture.gateway, fixture.user_id),
            lambda role: role == fixture.default_role,
        ) == fixture.default_role
        assert eventually(
            lambda: _database_user_role(fixture.user_id),
            lambda role: role == fixture.default_role,
        ) == fixture.default_role
        refused: Final = eventually(
            lambda: _generate_key_for_user(fixture.scenario, fixture.user_key, fixture.other_user),
            lambda response: response.status_code == 401,
        )
        assert refused.status_code == 401, refused.text
        assert object_value(refused.json()) == _expected_scim_write_denial(
            "/key/generate",
            fixture.user_id,
            fixture.default_role,
        ), refused.text


@pytest.mark.timeout(300)
def test_scim_group_put_returns_the_updated_members(gateway: Gateway, tmp_path: Path) -> None:
    pytest.skip(
        "BUG: PUT /scim/v2/Groups/{id} returns the members list from before the update, so the 200 body omits "
        "the members it just added"
    )
    group_name: Final = f"scim-admin-put-response-{uuid.uuid4().hex}"
    with _scim_admin_group_fixture(gateway, tmp_path, group_name) as fixture:
        expected_member: Final = ScimGroupMember(
            value=fixture.user_id,
            display=fixture.member_display,
            type="User",
        )
        added: Final = _change_group_membership(
            fixture.gateway,
            fixture.group_id,
            fixture.group_name,
            fixture.user_id,
            fixture.member_display,
            "add",
            "PUT",
        )
        assert added.status_code == 200, added.text
        _assert_scim_group_resource(
            added,
            fixture.group_id,
            fixture.group_name,
            [expected_member],
            fixture.created_group.meta.created,
        )


@pytest.mark.timeout(300)
def test_scim_admin_group_delete_revokes_the_members_admin_key(gateway: Gateway, tmp_path: Path) -> None:
    pytest.skip(
        "BUG: DELETE /scim/v2/Groups/{id} demotes the admin group's members in the DB but their existing keys keep proxy_admin access on /key/generate"
    )
    group_name: Final = f"scim-admin-delete-{uuid.uuid4().hex}"
    config: Final = _owned_scim_config(tmp_path / "scim-admin-group-delete.yaml", {"scim_admin_group": group_name})
    with owned_proxy_process(gateway, tmp_path, {}, config=config) as owned, owned.gateway.scenario() as scenario:
        group_id: Final = str(uuid.uuid4())
        group_response: Final = owned.gateway.request(
            "POST",
            "/scim/v2/Groups",
            {
                "schemas": [SCIM_CORE_GROUP_SCHEMA],
                "id": group_id,
                "displayName": group_name,
                "members": [],
            },
            headers=SCIM_HEADERS,
        )
        scenario.cleanups.callback(_delete_group_if_present, owned.gateway, group_id)
        assert group_response.status_code == 201, group_response.text
        created_group: Final = ScimGroupResponse.model_validate(group_response.json())
        assert created_group.id == group_id and created_group.displayName == group_name, group_response.text

        other_user: Final = f"scim-admin-target-{uuid.uuid4().hex}"
        created_other_user: Final = owned.gateway.request(
            "POST",
            "/scim/v2/Users",
            {
                "schemas": [SCIM_CORE_USER_SCHEMA],
                "userName": other_user,
                "name": {"givenName": "Key", "familyName": "Target"},
            },
            headers=SCIM_HEADERS,
        )
        scenario.cleanups.callback(_delete_user_if_present, owned.gateway, other_user)
        assert created_other_user.status_code == 201, created_other_user.text
        other_user_response: Final = ScimUserResponse.model_validate(created_other_user.json())
        assert other_user_response.id == other_user, created_other_user.text
        default_role: Final = _database_user_role(other_user)

        user: Final = f"scim-admin-delete-user-{uuid.uuid4().hex}"
        created_user: Final = owned.gateway.request(
            "POST",
            "/scim/v2/Users",
            {
                "schemas": [SCIM_CORE_USER_SCHEMA],
                "userName": user,
                "name": {"givenName": "Delete", "familyName": "Member"},
                "groups": [{"value": group_id, "display": group_name}],
            },
            headers=SCIM_HEADERS,
        )
        scenario.cleanups.callback(_delete_user_if_present, owned.gateway, user)
        assert created_user.status_code == 201, created_user.text
        created_user_response: Final = ScimUserResponse.model_validate(created_user.json())
        assert created_user_response.id == user, created_user.text
        user_key: Final = scenario.key(user_id=user)
        assert eventually(lambda: _user_role(owned.gateway, user), lambda role: role == "proxy_admin") == "proxy_admin"
        assert eventually(lambda: _database_user_role(user), lambda role: role == "proxy_admin") == "proxy_admin"
        allowed: Final = eventually(
            lambda: _generate_key_for_user(scenario, user_key, other_user),
            lambda response: response.status_code == 200,
        )
        assert allowed.status_code == 200, allowed.text

        deleted_group: Final = owned.gateway.request(
            "DELETE",
            f"/scim/v2/Groups/{group_id}",
            headers=SCIM_HEADERS,
        )
        assert deleted_group.status_code == 204, deleted_group.text
        assert eventually(lambda: _user_role(owned.gateway, user), lambda role: role == default_role) == default_role
        assert eventually(lambda: _database_user_role(user), lambda role: role == default_role) == default_role
        revoked: Final = eventually(
            lambda: _generate_key_for_user(scenario, user_key, other_user),
            lambda response: response.status_code == 401,
            return_last_on_timeout=True,
        )
        assert revoked.status_code == 401, (
            revoked.text if revoked.status_code != 200 else "Key generation returned 200; response body redacted"
        )
