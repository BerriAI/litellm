import os
import uuid
from contextlib import ExitStack
from hashlib import sha256
from typing import Final

import httpx
import psycopg
import pytest
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule, run_state_machine_as_test
from pydantic import BaseModel, ConfigDict, JsonValue

from tests.integration._support.client import Gateway, eventually, object_value, string_value, team_admin_permissions
from tests.integration._support.database import read_rows
from tests.integration._support.generation import LIFECYCLE_SETTINGS, bounded_http_requests


SCIM_CORE_USER_SCHEMA: Final = "urn:ietf:params:scim:schemas:core:2.0:User"


class ScimNameResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    givenName: str | None = None
    familyName: str | None = None
    formatted: str | None = None
    middleName: str | None = None
    honorificPrefix: str | None = None
    honorificSuffix: str | None = None


class ScimMetaResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resourceType: str
    created: str
    lastModified: str


class ScimEmailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    type: str | None = None
    primary: bool | None = None


class ScimGroupRefResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    display: str | None = None
    type: str | None = None


class ScimMultiValueResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str | None = None
    display: str | None = None
    type: str | None = None
    primary: bool | None = None


class ScimUserResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schemas: list[str]
    id: str
    externalId: str | None = None
    meta: ScimMetaResponse
    userName: str
    displayName: str
    active: bool
    name: ScimNameResponse
    emails: list[ScimEmailResponse]
    groups: list[ScimGroupRefResponse]
    entitlements: list[ScimMultiValueResponse] | None = None
    roles: list[ScimMultiValueResponse] | None = None


def _assert_scim_user_resource(
    response: httpx.Response,
    user_id: str,
    active: bool,
    expected_created: str | None,
    user_name: str,
    name: ScimNameResponse,
    emails: list[ScimEmailResponse],
) -> ScimUserResponse:
    actual: Final = ScimUserResponse.model_validate(response.json())
    created: Final = actual.meta.created if expected_created is None else expected_created
    assert isinstance(created, str), response.text
    assert isinstance(actual.meta.lastModified, str), response.text
    expected: Final = ScimUserResponse(
        schemas=[SCIM_CORE_USER_SCHEMA],
        id=user_id,
        externalId=None,
        meta=ScimMetaResponse(
            resourceType="User",
            created=created,
            lastModified=actual.meta.lastModified,
        ),
        userName=user_name,
        displayName=user_name,
        active=active,
        name=name,
        emails=emails,
        groups=[],
        entitlements=None,
        roles=None,
    )
    assert actual == expected, response.text
    return actual


def assert_serving(gateway: Gateway, model: str, key: str, status: int, error_type: str = "auth_error") -> None:
    response: Final = eventually(
        lambda: gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "warmed policy control"}]},
            key=key,
        ),
        lambda value: value.status_code == status,
        seconds=3,
    )
    if status == 200:
        assert response.json()["usage"]["total_tokens"] == 40
        assert response.json()["choices"][0]["message"]["content"] == (
            "Hello! This is a mock response from the fake OpenAI endpoint."
        )
    else:
        assert response.json()["error"]["type"] == error_type


@pytest.mark.covers("mgmt.key.update.two_workers_enforce_warmed_policy")
def test_generated_policy_changes_reach_both_warmed_workers(gateway: Gateway, peer: Gateway) -> None:
    class Policies(RuleBasedStateMachine):
        def __init__(self) -> None:
            super().__init__()
            self.resources = ExitStack()
            try:
                scenario = self.resources.enter_context(gateway.scenario())
                self.models = (scenario.model(), scenario.model())
                self.allowed = 0
                self.blocked = False
                self.key = scenario.key(models=[self.models[0]], blocked=False)
                self.control = scenario.key(models=list(self.models))
                for worker in (gateway, peer):
                    assert_serving(worker, self.models[0], self.key, 200)
                    assert_serving(worker, self.models[1], self.control, 200)
            except BaseException:
                with budget.cleanup():
                    self.resources.close()
                raise

        @rule(index=st.integers(min_value=0, max_value=1))
        def model_grant(self, index: int) -> None:
            gateway.post("/key/update", {"key": self.key, "models": [self.models[index]]})
            self.allowed = index

        @rule(blocked=st.booleans())
        def block(self, blocked: bool) -> None:
            gateway.post("/key/update", {"key": self.key, "blocked": blocked})
            self.blocked = blocked

        @invariant()
        def both_workers_enforce_policy(self) -> None:
            rows: Final = read_rows(
                'SELECT models, blocked FROM "LiteLLM_VerificationToken" WHERE token = %s',
                (sha256(self.key.encode()).hexdigest(),),
            )
            assert rows == [{"models": [self.models[self.allowed]], "blocked": self.blocked}]
            for worker in (gateway, peer):
                for index, model in enumerate(self.models):
                    status: Final = 401 if self.blocked else 200 if index == self.allowed else 403
                    kind: Final = "auth_error" if self.blocked else "key_model_access_denied"
                    assert_serving(worker, model, self.key, status, kind)
                assert_serving(worker, self.models[1], self.control, 200)

        def teardown(self) -> None:
            with budget.cleanup():
                self.resources.close()

    with bounded_http_requests((gateway, peer), limit=3000) as budget:
        run_state_machine_as_test(Policies, settings=LIFECYCLE_SETTINGS)


@pytest.mark.covers("mgmt.user.scim.deactivation_includes_nullable_blocked_keys")
def test_scim_deactivation_blocks_null_and_false_keys_but_preserves_other_owners(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(user_role="internal_user")
        other: Final = scenario.user(user_role="internal_user")
        null_key: Final = scenario.key(user_id=user, models=[model])
        false_key: Final = scenario.key(user_id=user, models=[model], blocked=False)
        manual: Final = scenario.key(user_id=user, models=[model], blocked=True)
        control: Final = scenario.key(user_id=other, models=[model])
        team: Final = scenario.team(models=[model])
        service: Final = gateway.post("/key/service-account/generate", {"team_id": team, "models": [model]})
        service_key: Final = service["key"]
        assert isinstance(service_key, str)
        scenario.cleanups.callback(scenario.delete_key, service_key)
        with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
            connection.execute(
                'UPDATE "LiteLLM_VerificationToken" SET blocked = NULL WHERE token = %s',
                (sha256(null_key.encode()).hexdigest(),),
            )
        assert read_rows(
            'SELECT user_id, blocked FROM "LiteLLM_VerificationToken" WHERE token = %s',
            (sha256(null_key.encode()).hexdigest(),),
        ) == [{"user_id": user, "blocked": None}]
        assert read_rows(
            'SELECT user_id FROM "LiteLLM_VerificationToken" WHERE token = %s',
            (sha256(service_key.encode()).hexdigest(),),
        ) == [{"user_id": None}]
        for token in (null_key, false_key, control, service_key):
            assert_serving(gateway, model, token, 200)
        for active in (False, True):
            response: Final = gateway.request(
                "PATCH",
                f"/scim/v2/Users/{user}",
                {
                    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                    "Operations": [{"op": "replace", "path": "active", "value": active}],
                },
            )
            assert response.status_code == 200, response.text
            for token in (null_key, false_key):
                rows: Final = read_rows(
                    'SELECT blocked, metadata FROM "LiteLLM_VerificationToken" WHERE token = %s',
                    (sha256(token.encode()).hexdigest(),),
                )
                assert rows[0]["blocked"] is not active
                assert object_value(rows[0]["metadata"]).get("scim_blocked") is (None if active else True)
                assert_serving(gateway, model, token, 200 if active else 401)
            assert_serving(gateway, model, manual, 401)
            for token in (control, service_key):
                assert_serving(gateway, model, token, 200)


@pytest.mark.parametrize(
    "deactivate_operations,reactivate_operations",
    [
        pytest.param(
            [{"op": "replace", "value": {"active": False}}],
            [{"op": "replace", "value": {"active": True}}],
            id="okta_pathless",
        ),
        pytest.param(
            [{"op": "Replace", "path": "active", "value": "False"}],
            [{"op": "Replace", "path": "active", "value": "True"}],
            id="entra_string_boolean",
        ),
    ],
)
@pytest.mark.parametrize(
    "warm_before_deactivation",
    [True, False],
    ids=["warmed", "first_used_while_deactivated"],
)
def test_scim_idp_deactivation_spelling_blocks_and_restores_the_users_key(
    gateway: Gateway,
    peer: Gateway,
    deactivate_operations: list[dict[str, JsonValue]],
    reactivate_operations: list[dict[str, JsonValue]],
    warm_before_deactivation: bool,
) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(user_role="internal_user")
        key: Final = scenario.key(user_id=user, models=[model])
        original_name: Final = ScimNameResponse(
            givenName="Unknown User",
            familyName="Unknown Family Name",
        )
        initial_response: Final = gateway.request("GET", f"/scim/v2/Users/{user}")
        assert initial_response.status_code == 200, initial_response.text
        initial_user: Final = _assert_scim_user_resource(
            initial_response,
            user,
            True,
            None,
            "Unknown Display Name",
            original_name,
            [],
        )
        created: Final = initial_user.meta.created
        if warm_before_deactivation:
            for worker in (gateway, peer):
                assert_serving(worker, model, key, 200)

        deactivated: Final = gateway.request(
            "PATCH",
            f"/scim/v2/Users/{user}",
            {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                "Operations": deactivate_operations,
            },
            headers={"Content-Type": "application/scim+json"},
        )
        assert deactivated.status_code == 200, deactivated.text
        _assert_scim_user_resource(
            deactivated,
            user,
            False,
            created,
            "Unknown Display Name",
            original_name,
            [],
        )
        deactivated_readback_response: Final = gateway.request("GET", f"/scim/v2/Users/{user}")
        assert deactivated_readback_response.status_code == 200, deactivated_readback_response.text
        _assert_scim_user_resource(
            deactivated_readback_response,
            user,
            False,
            created,
            "Unknown Display Name",
            original_name,
            [],
        )
        user_rows: Final = read_rows(
            'SELECT metadata FROM "LiteLLM_UserTable" WHERE user_id = %s',
            (user,),
        )
        user_metadata: Final = object_value(user_rows[0]["metadata"])
        assert user_metadata.get("scim_active") is False and "active" not in user_metadata, user_metadata
        key_rows: Final = read_rows(
            'SELECT blocked, metadata FROM "LiteLLM_VerificationToken" WHERE token = %s',
            (sha256(key.encode()).hexdigest(),),
        )
        assert key_rows[0]["blocked"] is True, key_rows
        assert object_value(key_rows[0]["metadata"]).get("scim_blocked") is True, key_rows
        for worker in (gateway, peer):
            assert_serving(worker, model, key, 401)

        reactivated: Final = gateway.request(
            "PATCH",
            f"/scim/v2/Users/{user}",
            {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                "Operations": reactivate_operations,
            },
            headers={"Content-Type": "application/scim+json"},
        )
        assert reactivated.status_code == 200, reactivated.text
        _assert_scim_user_resource(
            reactivated,
            user,
            True,
            created,
            "Unknown Display Name",
            original_name,
            [],
        )
        reactivated_readback_response: Final = gateway.request("GET", f"/scim/v2/Users/{user}")
        assert reactivated_readback_response.status_code == 200, reactivated_readback_response.text
        _assert_scim_user_resource(
            reactivated_readback_response,
            user,
            True,
            created,
            "Unknown Display Name",
            original_name,
            [],
        )
        reactivated_user_rows: Final = read_rows(
            'SELECT metadata FROM "LiteLLM_UserTable" WHERE user_id = %s',
            (user,),
        )
        reactivated_metadata: Final = object_value(reactivated_user_rows[0]["metadata"])
        assert reactivated_metadata.get("scim_active") is True, reactivated_user_rows
        assert "active" not in reactivated_metadata, reactivated_user_rows
        restored_key_rows: Final = read_rows(
            'SELECT blocked, metadata FROM "LiteLLM_VerificationToken" WHERE token = %s',
            (sha256(key.encode()).hexdigest(),),
        )
        assert restored_key_rows[0]["blocked"] is False, restored_key_rows
        assert "scim_blocked" not in object_value(restored_key_rows[0]["metadata"]), restored_key_rows
        for worker in (gateway, peer):
            assert_serving(worker, model, key, 200)


def test_scim_put_without_active_keeps_a_deactivated_user_blocked(gateway: Gateway, peer: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(user_role="internal_user")
        key: Final = scenario.key(user_id=user, models=[model])
        original_name: Final = ScimNameResponse(
            givenName="Unknown User",
            familyName="Unknown Family Name",
        )
        initial_response: Final = gateway.request("GET", f"/scim/v2/Users/{user}")
        assert initial_response.status_code == 200, initial_response.text
        initial_user: Final = _assert_scim_user_resource(
            initial_response,
            user,
            True,
            None,
            "Unknown Display Name",
            original_name,
            [],
        )
        created: Final = initial_user.meta.created
        email: Final = f"{user}@example.com"
        resource: Final[dict[str, JsonValue]] = {
            "schemas": [SCIM_CORE_USER_SCHEMA],
            "userName": user,
            "name": {"givenName": "Inactive", "familyName": "User"},
            "emails": [{"value": email, "primary": True, "type": "work"}],
            "active": False,
        }
        deactivated: Final = gateway.request(
            "PUT",
            f"/scim/v2/Users/{user}",
            resource,
            headers={"Content-Type": "application/scim+json"},
        )
        assert deactivated.status_code == 200, deactivated.text
        deactivated_name: Final = ScimNameResponse(givenName="Inactive", familyName="User")
        deactivated_emails: Final = [ScimEmailResponse(value=email, type=None, primary=True)]
        _assert_scim_user_resource(
            deactivated,
            user,
            False,
            created,
            email,
            deactivated_name,
            deactivated_emails,
        )
        deactivated_readback_response: Final = gateway.request("GET", f"/scim/v2/Users/{user}")
        assert deactivated_readback_response.status_code == 200, deactivated_readback_response.text
        _assert_scim_user_resource(
            deactivated_readback_response,
            user,
            False,
            created,
            email,
            deactivated_name,
            deactivated_emails,
        )
        blocked_key_rows: Final = read_rows(
            'SELECT blocked, metadata FROM "LiteLLM_VerificationToken" WHERE token = %s',
            (sha256(key.encode()).hexdigest(),),
        )
        assert blocked_key_rows[0]["blocked"] is True, blocked_key_rows
        assert object_value(blocked_key_rows[0]["metadata"]).get("scim_blocked") is True, blocked_key_rows
        for worker in (gateway, peer):
            assert_serving(worker, model, key, 401)

        without_active: Final[dict[str, JsonValue]] = {
            field: value for field, value in resource.items() if field != "active"
        }
        without_active["name"] = {"givenName": "Inactive", "familyName": "Updated"}
        updated: Final = gateway.request(
            "PUT",
            f"/scim/v2/Users/{user}",
            without_active,
            headers={"Content-Type": "application/scim+json"},
        )
        assert updated.status_code == 200, updated.text
        updated_name: Final = ScimNameResponse(givenName="Inactive", familyName="Updated")
        _assert_scim_user_resource(
            updated,
            user,
            False,
            created,
            email,
            updated_name,
            deactivated_emails,
        )
        updated_readback_response: Final = gateway.request("GET", f"/scim/v2/Users/{user}")
        assert updated_readback_response.status_code == 200, updated_readback_response.text
        _assert_scim_user_resource(
            updated_readback_response,
            user,
            False,
            created,
            email,
            updated_name,
            deactivated_emails,
        )
        still_blocked_rows: Final = read_rows(
            'SELECT blocked, metadata FROM "LiteLLM_VerificationToken" WHERE token = %s',
            (sha256(key.encode()).hexdigest(),),
        )
        assert still_blocked_rows[0]["blocked"] is True, still_blocked_rows
        assert object_value(still_blocked_rows[0]["metadata"]).get("scim_blocked") is True, still_blocked_rows
        for worker in (gateway, peer):
            assert_serving(worker, model, key, 401)


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


def _delete_user_if_present(gateway: Gateway, user_id: str) -> None:
    if read_rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,)):
        response: Final = gateway.request("POST", "/user/delete", {"user_ids": [user_id]})
        assert response.status_code == 200 and response.json() == 1, response.text
    assert read_rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,)) == []


def _delete_group_if_present(gateway: Gateway, group_id: str) -> None:
    if read_rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = %s', (group_id,)):
        response: Final = gateway.request(
            "DELETE",
            f"/scim/v2/Groups/{group_id}",
            headers={"Content-Type": "application/scim+json"},
        )
        assert response.status_code == 204, response.text
    assert read_rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = %s', (group_id,)) == []


def _delete_team_if_present(gateway: Gateway, team_id: str) -> None:
    if read_rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team_id,)):
        gateway.post("/team/delete", {"team_ids": [team_id]})
    assert read_rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team_id,)) == []


@pytest.mark.parametrize(
    "write",
    [
        "post_user",
        "put_user",
        "patch_user",
        "post_group",
        "put_group",
        "patch_group",
        "delete_group",
    ],
)
@pytest.mark.parametrize("team_admin", [False, True], ids=["internal_user", "team_admin"])
def test_non_admin_keys_cannot_write_scim_users_or_groups(gateway: Gateway, team_admin: bool, write: str) -> None:
    with gateway.scenario() as scenario:
        caller: Final = scenario.user(user_role="internal_user")
        created_team: Final = gateway.post(
            "/team/new",
            {
                "team_alias": f"integration-{uuid.uuid4().hex}",
                "members_with_roles": [{"user_id": caller, "role": "admin"}] if team_admin else [],
            },
        )
        team: Final = string_value(created_team["team_id"])
        scenario.cleanups.callback(_delete_team_if_present, gateway, team)
        key: Final = scenario.key(user_id=caller, team_id=team if team_admin else None)
        user_before: Final = read_rows(
            'SELECT user_role, metadata, teams FROM "LiteLLM_UserTable" WHERE user_id = %s',
            (caller,),
        )
        caller_role: Final = string_value(user_before[0]["user_role"])
        team_before: Final = read_rows(
            'SELECT team_alias, members_with_roles FROM "LiteLLM_TeamTable" WHERE team_id = %s',
            (team,),
        )
        new_user: Final = f"scim-denied-{uuid.uuid4().hex}"
        scenario.cleanups.callback(_delete_user_if_present, gateway, new_user)
        new_group: Final = str(uuid.uuid4())
        scenario.cleanups.callback(_delete_group_if_present, gateway, new_group)
        request_details: Final = {
            "post_user": (
                "POST",
                "/scim/v2/Users",
                {
                    "schemas": [SCIM_CORE_USER_SCHEMA],
                    "userName": new_user,
                    "name": {"givenName": "Denied", "familyName": "Create"},
                    "emails": [{"value": f"{new_user}@example.com", "primary": True, "type": "work"}],
                    "active": True,
                    "roles": [{"value": "proxy_admin"}],
                },
            ),
            "put_user": (
                "PUT",
                f"/scim/v2/Users/{caller}",
                {
                    "schemas": [SCIM_CORE_USER_SCHEMA],
                    "userName": caller,
                    "name": {"givenName": "Denied", "familyName": "Update"},
                    "emails": [{"value": f"{caller}@example.com", "primary": True, "type": "work"}],
                    "active": False,
                    "roles": [{"value": "proxy_admin"}],
                },
            ),
            "patch_user": (
                "PATCH",
                f"/scim/v2/Users/{caller}",
                {
                    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                    "Operations": [
                        {"op": "replace", "path": "roles", "value": [{"value": "proxy_admin"}]},
                        {"op": "replace", "path": "active", "value": False},
                    ],
                },
            ),
            "post_group": (
                "POST",
                "/scim/v2/Groups",
                {
                    "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                    "id": new_group,
                    "displayName": f"denied-{new_group}",
                    "members": [{"value": caller}],
                },
            ),
            "put_group": (
                "PUT",
                f"/scim/v2/Groups/{team}",
                {
                    "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                    "displayName": f"updated-{team}",
                    "members": [{"value": caller}],
                },
            ),
            "patch_group": (
                "PATCH",
                f"/scim/v2/Groups/{team}",
                {
                    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                    "Operations": [{"op": "add", "path": "members", "value": [{"value": caller}]}],
                },
            ),
            "delete_group": ("DELETE", f"/scim/v2/Groups/{team}", None),
        }
        method, route, body = request_details[write]
        response: Final = gateway.request(
            method,
            route,
            body,
            key=key,
            headers={"Content-Type": "application/scim+json"},
        )
        assert response.status_code == 401, response.text
        assert object_value(response.json()) == _expected_scim_write_denial(route, caller, caller_role), response.text
        user_after: Final = read_rows(
            'SELECT user_role, metadata, teams FROM "LiteLLM_UserTable" WHERE user_id = %s',
            (caller,),
        )
        team_after: Final = read_rows(
            'SELECT team_alias, members_with_roles FROM "LiteLLM_TeamTable" WHERE team_id = %s',
            (team,),
        )
        new_user_rows: Final = read_rows(
            'SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s',
            (new_user,),
        )
        new_group_rows: Final = read_rows(
            'SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = %s',
            (new_group,),
        )
        assert user_after == user_before, (user_after, response.text)
        assert team_after == team_before, (team_after, response.text)
        assert new_user_rows == [], (new_user_rows, response.text)
        assert new_group_rows == [], (new_group_rows, response.text)


@pytest.mark.covers("mgmt.team.member_update.demoted_role_cannot_write")
def test_warmed_team_role_demotion_prevents_later_management_writes(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, team_admin_permissions(gateway, ["tpm_limit"]):
        model: Final = scenario.model()
        user: Final = scenario.user(user_role="internal_user")
        team: Final = scenario.team(
            models=[model], tpm_limit=1000, members_with_roles=[{"user_id": user, "role": "admin"}]
        )
        control_team: Final = scenario.team(models=[model], tpm_limit=1000)
        caller: Final = scenario.key(
            user_id=user, team_id=team, models=[model], allowed_routes=["/team/update", "/v1/chat/completions"]
        )
        gateway.chat(model, key=caller)
        changed: Final = gateway.request("POST", "/team/update", {"team_id": team, "tpm_limit": 5000}, key=caller)
        assert changed.status_code == 200, changed.text
        assert read_rows('SELECT tpm_limit FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team,)) == [
            {"tpm_limit": 5000}
        ]
        unrelated_before: Final = read_rows(
            'SELECT tpm_limit FROM "LiteLLM_TeamTable" WHERE team_id = %s', (control_team,)
        )
        unrelated: Final = gateway.request(
            "POST", "/team/update", {"team_id": control_team, "tpm_limit": 7000}, key=caller
        )
        assert unrelated.status_code == 403, unrelated.text
        assert (
            read_rows('SELECT tpm_limit FROM "LiteLLM_TeamTable" WHERE team_id = %s', (control_team,))
            == unrelated_before
        )
        gateway.post("/team/member_update", {"team_id": team, "user_id": user, "role": "user"})
        for target in (team, control_team):
            before: Final = read_rows('SELECT tpm_limit FROM "LiteLLM_TeamTable" WHERE team_id = %s', (target,))
            denied: Final = gateway.request("POST", "/team/update", {"team_id": target, "tpm_limit": 9000}, key=caller)
            assert denied.status_code == 403, denied.text
            after: Final = read_rows('SELECT tpm_limit FROM "LiteLLM_TeamTable" WHERE team_id = %s', (target,))
            assert after == before
        roster: Final = read_rows('SELECT members_with_roles FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team,))
        members: Final = roster[0]["members_with_roles"]
        assert isinstance(members, list)
        assert (
            next(object_value(member)["role"] for member in members if object_value(member)["user_id"] == user)
            == "user"
        )
        assert_serving(gateway, model, caller, 200)


def _key_row(key: str) -> dict[str, JsonValue]:
    rows: Final = read_rows(
        'SELECT max_budget, key_alias FROM "LiteLLM_VerificationToken" WHERE token = %s',
        (sha256(key.encode()).hexdigest(),),
    )
    assert len(rows) == 1
    return rows[0]


@pytest.mark.covers("mgmt.key.update.team_admin_member_key_budget_requires_opt_in")
def test_team_admin_changes_member_key_budget_only_when_opted_in(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        admin: Final = scenario.user(user_role="internal_user")
        member: Final = scenario.user(user_role="internal_user")
        team: Final = scenario.team(
            models=[model],
            members_with_roles=[{"user_id": admin, "role": "admin"}, {"user_id": member, "role": "user"}],
        )
        other_team: Final = scenario.team(models=[model], members_with_roles=[{"user_id": member, "role": "user"}])
        member_key: Final = scenario.key(
            user_id=member, team_id=team, models=[model], max_budget=10, key_alias="member"
        )
        personal_key: Final = scenario.key(user_id=member, models=[model], max_budget=10)
        foreign_key: Final = scenario.key(user_id=member, team_id=other_team, models=[model], max_budget=10)
        admin_key: Final = scenario.key(
            user_id=admin, team_id=team, models=[model], allowed_routes=["/key/update", "/v1/chat/completions"]
        )
        member_caller: Final = scenario.key(
            user_id=member, team_id=team, models=[model], allowed_routes=["/key/update", "/v1/chat/completions"]
        )
        assert_serving(gateway, model, member_key, 200)
        with team_admin_permissions(gateway, []):
            denied: Final = gateway.request("POST", "/key/update", {"key": member_key, "max_budget": 0}, key=admin_key)
            assert denied.status_code == 403, denied.text
            assert _key_row(member_key) == {"max_budget": 10.0, "key_alias": "member"}
        with team_admin_permissions(gateway, ["member_key_budgets"]):
            for target in (personal_key, foreign_key):
                out_of_scope: Final = gateway.request(
                    "POST", "/key/update", {"key": target, "max_budget": 0}, key=admin_key
                )
                assert out_of_scope.status_code == 403, out_of_scope.text
                assert _key_row(target)["max_budget"] == 10.0
            by_member: Final = gateway.request(
                "POST", "/key/update", {"key": admin_key, "max_budget": 0}, key=member_caller
            )
            assert by_member.status_code == 403, by_member.text
            not_budget: Final = gateway.request(
                "POST", "/key/update", {"key": member_key, "key_alias": "renamed"}, key=admin_key
            )
            assert not_budget.status_code == 403, not_budget.text
            assert _key_row(member_key) == {"max_budget": 10.0, "key_alias": "member"}
            changed: Final = gateway.request(
                "POST", "/key/update", {"key": member_key, "max_budget": 0, "budget_duration": "30d"}, key=admin_key
            )
            assert changed.status_code == 200, changed.text
            assert _key_row(member_key) == {"max_budget": 0.0, "key_alias": "member"}
            assert_serving(gateway, model, member_key, 422, "budget_exceeded")
            restored: Final = gateway.request(
                "POST", "/key/update", {"key": member_key, "max_budget": 10}, key=admin_key
            )
            assert restored.status_code == 200, restored.text
            assert_serving(gateway, model, member_key, 200)


@pytest.mark.covers("mgmt.key.update.expiry_changes_reach_warmed_workers")
def test_expiry_and_explicit_clear_reach_both_warmed_workers(gateway: Gateway, peer: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model], duration="1h")
        control: Final = scenario.key(models=[model])
        for worker in (gateway, peer):
            assert_serving(worker, model, key, 200)
        gateway.post("/key/update", {"key": key, "duration": "0s"})
        assert read_rows(
            "SELECT expires <= timezone('UTC', now()) AS expired FROM \"LiteLLM_VerificationToken\" WHERE token = %s",
            (sha256(key.encode()).hexdigest(),),
        ) == [{"expired": True}]
        for worker in (gateway, peer):
            assert_serving(worker, model, key, 401, "expired_key")
            assert_serving(worker, model, control, 200)
        gateway.post("/key/update", {"key": key, "duration": None})
        assert read_rows(
            'SELECT expires IS NULL AS cleared FROM "LiteLLM_VerificationToken" WHERE token = %s',
            (sha256(key.encode()).hexdigest(),),
        ) == [{"cleared": True}]
        for worker in (gateway, peer):
            assert_serving(worker, model, key, 200)
