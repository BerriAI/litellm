import os
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.database import read_rows
from pydantic import BaseModel, JsonValue

from litellm.models.user import LiteLLM_UserTable
from litellm.proxy.auth.auth_checks import ExperimentalUIJWTToken

_SELF_SERVICE_ROUTES: Final = ("/user/update", "/user/bulk_update", "/user/new", "/user/info")


class _UserInfo(BaseModel):
    user_id: str
    user_alias: str | None
    user_role: str
    max_budget: float | None


def _user_row(user_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        "SELECT user_id, user_alias, user_role, max_budget, model_max_budget, spend, object_permission_id, user_email"
        ' FROM "LiteLLM_UserTable" WHERE user_id=%s',
        (user_id,),
    )


def _user_info(gateway: Gateway, user_id: str) -> _UserInfo:
    payload: Final = object_value(gateway.get("/user/info", {"user_id": user_id}))
    return _UserInfo.model_validate(payload["user_info"])


def _escalation_error(field: str) -> dict[str, JsonValue]:
    return {
        "error": {
            "message": f"{{'error': \"Non-admin users cannot modify '{field}' on their own record. Contact your proxy admin.\"}}",
            "type": "auth_error",
            "param": "None",
            "code": "403",
        }
    }


def _role_change_error() -> dict[str, JsonValue]:
    return {
        "error": {
            "message": "Only proxy admins can modify user roles.",
            "type": "auth_error",
            "param": "None",
            "code": "403",
        }
    }


def _ui_session_route_error(user_id: str) -> dict[str, JsonValue]:
    masked_user_id: Final = f"{user_id[:6]}{'*' * (len(user_id) - 8)}{user_id[-2:]}"
    return {
        "error": {
            "message": (
                "Authentication Error, Only proxy admin can be used to generate, delete, update info for new "
                "keys/users/teams. Route=/user/update. Your role=internal_user. "
                f"Your user_id={masked_user_id}"
            ),
            "type": "auth_error",
            "param": "None",
            "code": "401",
        }
    }


def test_internal_user_can_update_allowed_fields_on_own_record(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        user_id: Final = scenario.user(user_role="internal_user", max_budget=10.0)
        key: Final = scenario.key(user_id=user_id, allowed_routes=list(_SELF_SERVICE_ROUTES))
        alias: Final = f"self-alias-{uuid.uuid4().hex}"
        updated: Final = gateway.request("POST", "/user/update", {"user_id": user_id, "user_alias": alias}, key=key)
        assert updated.status_code == 200, updated.text
        assert object_value(object_value(updated.json())["data"])["user_alias"] == alias
        assert _user_row(user_id)[0]["user_alias"] == alias
        assert _user_info(gateway, user_id).user_alias == alias


@pytest.mark.parametrize("user_role", ("proxy_admin", "proxy_admin_viewer"))
def test_internal_user_cannot_promote_own_role(gateway: Gateway, user_role: str) -> None:
    with gateway.scenario() as scenario:
        user_id: Final = scenario.user(user_role="internal_user", max_budget=10.0)
        key: Final = scenario.key(user_id=user_id, allowed_routes=list(_SELF_SERVICE_ROUTES))
        before_row: Final = _user_row(user_id)
        before_info: Final = _user_info(gateway, user_id)

        refused: Final = gateway.request(
            "POST",
            "/user/update",
            {"user_id": user_id, "user_role": user_role},
            key=key,
        )
        assert refused.status_code == 403, refused.text
        assert refused.json() == _role_change_error(), refused.text
        assert _user_row(user_id) == before_row
        assert _user_info(gateway, user_id) == before_info


def test_internal_user_can_update_allowed_fields_by_own_email(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        user_email: Final = f"integration-{uuid.uuid4().hex}@example.com"
        user_id: Final = scenario.user(
            user_email=user_email,
            user_role="internal_user",
            max_budget=10.0,
        )
        key: Final = scenario.key(user_id=user_id, allowed_routes=list(_SELF_SERVICE_ROUTES))
        alias: Final = f"self-email-alias-{uuid.uuid4().hex}"

        updated: Final = gateway.request(
            "POST",
            "/user/update",
            {"user_email": user_email, "user_alias": alias},
            key=key,
        )
        assert updated.status_code == 200, updated.text
        updated_data: Final = object_value(object_value(updated.json())["data"])
        assert updated_data["user_alias"] == alias
        assert updated_data["user_id"] == user_id
        assert _user_row(user_id)[0]["user_alias"] == alias
        assert _user_info(gateway, user_id).user_alias == alias


def test_internal_user_cannot_escalate_protected_fields_by_own_email(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        user_email: Final = f"integration-{uuid.uuid4().hex}@example.com"
        user_id: Final = scenario.user(
            user_email=user_email,
            user_role="internal_user",
            max_budget=10.0,
        )
        key: Final = scenario.key(user_id=user_id, allowed_routes=list(_SELF_SERVICE_ROUTES))
        before_row: Final = _user_row(user_id)
        before_info: Final = _user_info(gateway, user_id)

        refused: Final = gateway.request(
            "POST",
            "/user/update",
            {"user_email": user_email, "max_budget": 1000.0},
            key=key,
        )
        assert refused.status_code == 403, refused.text
        assert refused.json() == _escalation_error("max_budget"), refused.text
        assert _user_row(user_id) == before_row
        assert _user_info(gateway, user_id) == before_info


@pytest.mark.parametrize(
    ("field", "value"),
    (("max_budget", 1000), ("object_permission", {})),
    ids=("max_budget", "object_permission"),
)
def test_ui_session_token_cannot_update_own_protected_fields(
    gateway: Gateway,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: JsonValue,
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", os.environ.get("LITELLM_SALT_KEY", "sk-integration-salt"))
    with gateway.scenario() as scenario:
        user_id: Final = scenario.user(user_role="internal_user", max_budget=10.0)
        token: Final = ExperimentalUIJWTToken.get_experimental_ui_login_jwt_auth_token(
            LiteLLM_UserTable(user_id=user_id, user_role="internal_user", models=[])
        )
        self_info: Final = gateway.request("GET", "/user/info", params={"user_id": user_id}, key=token)
        assert self_info.status_code == 200, self_info.text
        assert object_value(object_value(self_info.json())["user_info"])["user_id"] == user_id, self_info.text
        before_row: Final = _user_row(user_id)
        before_info: Final = _user_info(gateway, user_id)

        refused: Final = gateway.request("POST", "/user/update", {"user_id": user_id, field: value}, key=token)
        assert refused.status_code == 401, refused.text
        assert refused.json() == _ui_session_route_error(user_id), refused.text
        assert _user_row(user_id) == before_row
        assert _user_info(gateway, user_id) == before_info


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("max_budget", 1000),
        ("max_budget", "1000"),
        ("model_max_budget", {"gpt-4o": 5.0}),
        ("spend", 0),
        ("object_permission", {"vector_stores": ["vs-1"]}),
        ("object_permission", {}),
    ),
    ids=(
        "max_budget_number",
        "max_budget_string",
        "model_max_budget",
        "spend",
        "object_permission",
        "object_permission_empty",
    ),
)
def test_internal_user_cannot_escalate_protected_fields(gateway: Gateway, field: str, value: JsonValue) -> None:
    with gateway.scenario() as scenario:
        user_id: Final = scenario.user(user_role="internal_user", max_budget=10.0)
        key: Final = scenario.key(user_id=user_id, allowed_routes=list(_SELF_SERVICE_ROUTES))
        before_row: Final = _user_row(user_id)
        before_info: Final = _user_info(gateway, user_id)

        refused: Final = gateway.request("POST", "/user/update", {"user_id": user_id, field: value}, key=key)
        assert refused.status_code == 403, refused.text
        assert refused.json() == _escalation_error(field), refused.text
        assert _user_row(user_id) == before_row
        assert _user_info(gateway, user_id) == before_info


def test_bulk_update_self_escalation_fails_per_user(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        user_id: Final = scenario.user(user_role="internal_user", max_budget=10.0)
        key: Final = scenario.key(user_id=user_id, allowed_routes=list(_SELF_SERVICE_ROUTES))
        before_row: Final = _user_row(user_id)

        refused: Final = gateway.request(
            "POST",
            "/user/bulk_update",
            {"users": [{"user_id": user_id, "max_budget": 1000}]},
            key=key,
        )
        assert refused.status_code == 200, refused.text
        assert refused.json() == {
            "results": [
                {
                    "user_id": user_id,
                    "user_email": None,
                    "success": False,
                    "error": "403: {'error': \"Non-admin users cannot modify 'max_budget' on their own record. Contact your proxy admin.\"}",
                    "updated_user": None,
                }
            ],
            "total_requested": 1,
            "successful_updates": 0,
            "failed_updates": 1,
        }, refused.text
        assert _user_row(user_id) == before_row


def test_bulk_update_user_role_requires_proxy_admin(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        user_id: Final = scenario.user(user_role="internal_user", max_budget=10.0)
        key: Final = scenario.key(user_id=user_id, allowed_routes=list(_SELF_SERVICE_ROUTES))
        before_row: Final = _user_row(user_id)

        refused: Final = gateway.request(
            "POST",
            "/user/bulk_update",
            {"users": [{"user_id": user_id, "user_role": "proxy_admin"}]},
            key=key,
        )
        assert refused.status_code == 403, refused.text
        assert refused.json() == {"detail": "Only proxy admins can modify user roles."}, refused.text
        assert _user_row(user_id) == before_row


@pytest.mark.parametrize("role", ("proxy_admin", "proxy_admin_viewer"))
def test_new_user_with_admin_role_requires_proxy_admin(gateway: Gateway, role: str) -> None:
    with gateway.scenario() as scenario:
        user_id: Final = scenario.user(user_role="internal_user", max_budget=10.0)
        key: Final = scenario.key(user_id=user_id, allowed_routes=list(_SELF_SERVICE_ROUTES))
        attempted_id: Final = f"integration-{uuid.uuid4().hex}"

        refused: Final = gateway.request(
            "POST",
            "/user/new",
            {"user_id": attempted_id, "user_role": role, "auto_create_key": False},
            key=key,
        )
        assert refused.status_code == 403, refused.text
        assert refused.json() == {
            "error": {
                "message": f"Only proxy admins can create administrative users (proxy_admin, proxy_admin_viewer). Attempted to create user with role: {role}. Your role: internal_user",
                "type": "internal_server_error",
                "param": None,
                "code": "403",
            }
        }, refused.text
        assert _user_row(attempted_id) == []
