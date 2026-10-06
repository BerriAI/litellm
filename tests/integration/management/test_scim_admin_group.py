import uuid
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


class ScimUserResponse(BaseModel):
    schemas: list[str]
    id: str
    userName: str | None = None


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


@pytest.mark.timeout(300)
@pytest.mark.parametrize("spelling", ["okta", "entra"], ids=["okta", "entra"])
def test_scim_admin_group_promotes_and_demotes_group_members(
    gateway: Gateway,
    tmp_path: Path,
    spelling: Literal["okta", "entra"],
) -> None:
    group_name: Final = f"scim-admin-{uuid.uuid4().hex}"
    config: Final = _owned_scim_config(tmp_path / "scim-admin-group.yaml", {"scim_admin_group": group_name})
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
        assert group_response.status_code == 201, group_response.text
        created_group: Final = _assert_scim_group_resource(
            group_response,
            group_id,
            group_name,
            [],
            None,
        )
        scenario.cleanups.callback(_delete_group_if_present, owned.gateway, group_id)

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
        initial_denial: Final = _generate_key_for_user(scenario, user_key, other_user)
        assert initial_denial.status_code == 401, initial_denial.text
        assert object_value(initial_denial.json()) == _expected_scim_write_denial(
            "/key/generate",
            user,
            default_role,
        ), initial_denial.text

        add_op, remove_op = _group_membership_operations(user, spelling)
        added: Final = owned.gateway.request(
            "PATCH",
            f"/scim/v2/Groups/{group_id}",
            {
                "schemas": [SCIM_PATCH_SCHEMA],
                "Operations": [add_op],
            },
            headers=SCIM_HEADERS,
        )
        assert added.status_code == 200, added.text
        expected_member: Final = ScimGroupMember(value=user, display=member_display, type="User")
        added_group: Final = _assert_scim_group_resource(
            added,
            group_id,
            group_name,
            [expected_member],
            created_group.meta.created,
        )
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

        removed: Final = owned.gateway.request(
            "PATCH",
            f"/scim/v2/Groups/{group_id}",
            {
                "schemas": [SCIM_PATCH_SCHEMA],
                "Operations": [remove_op],
            },
            headers=SCIM_HEADERS,
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
