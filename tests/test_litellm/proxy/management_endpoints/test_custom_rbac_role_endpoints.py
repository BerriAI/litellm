from unittest.mock import patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.custom_rbac_role_endpoints import (
    delete_custom_role,
    new_custom_role,
    update_custom_role,
)
from litellm.types.custom_rbac import (
    CustomRBACRoleCreateRequest,
    CustomRBACRoleDeleteRequest,
    CustomRBACRoleUpdateRequest,
)

_TABLE_PATH = "litellm.proxy.management_endpoints.custom_rbac_role_endpoints._role_table"


class _FakeRecord:
    def __init__(self, values: dict):
        self._values = values

    def dict(self) -> dict:
        return self._values


class _FakeTable:
    def __init__(self, rows: dict[str, dict] | None = None):
        self.rows = dict(rows or {})
        self.deleted: list[str] = []

    async def find_unique(self, where):
        row = self.rows.get(where["role_name"])
        return None if row is None else _FakeRecord(row)

    async def find_many(self, order):
        return [_FakeRecord(row) for row in self.rows.values()]

    async def create(self, data):
        self.rows[data["role_name"]] = dict(data)
        return _FakeRecord(dict(data))

    async def update(self, where, data):
        merged = {**self.rows[where["role_name"]], **data}
        self.rows[where["role_name"]] = merged
        return _FakeRecord(merged)

    async def delete(self, where):
        self.deleted.append(where["role_name"])
        return self.rows.pop(where["role_name"])


def _admin() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)


def _internal_user() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id="u1", user_role=LitellmUserRoles.INTERNAL_USER)


@pytest.mark.asyncio
async def test_create_persists_role_and_returns_it():
    table = _FakeTable()
    with patch(_TABLE_PATH, return_value=table):
        response = await new_custom_role(
            data=CustomRBACRoleCreateRequest(
                role_name="data-scientist",
                allowed_routes=("llm_api_routes", "/key/info"),
            ),
            user_api_key_dict=_admin(),
        )
    assert response.role_name == "data-scientist"
    assert table.rows["data-scientist"]["allowed_routes"] == ["llm_api_routes", "/key/info"]


@pytest.mark.asyncio
async def test_non_admin_cannot_create_role():
    table = _FakeTable()
    with patch(_TABLE_PATH, return_value=table):
        with pytest.raises(HTTPException) as exc:
            await new_custom_role(
                data=CustomRBACRoleCreateRequest(role_name="data-scientist"),
                user_api_key_dict=_internal_user(),
            )
    assert exc.value.status_code == 403
    assert table.rows == {}


@pytest.mark.asyncio
async def test_builtin_role_name_is_rejected():
    table = _FakeTable()
    with patch(_TABLE_PATH, return_value=table):
        with pytest.raises(HTTPException) as exc:
            await new_custom_role(
                data=CustomRBACRoleCreateRequest(role_name="internal_user"),
                user_api_key_dict=_admin(),
            )
    assert exc.value.status_code == 400
    assert table.rows == {}


@pytest.mark.asyncio
async def test_invalid_route_permission_is_rejected():
    table = _FakeTable()
    with patch(_TABLE_PATH, return_value=table):
        with pytest.raises(HTTPException) as exc:
            await new_custom_role(
                data=CustomRBACRoleCreateRequest(role_name="viewer", allowed_routes=("key/info",)),
                user_api_key_dict=_admin(),
            )
    assert exc.value.status_code == 400
    assert table.rows == {}


@pytest.mark.asyncio
async def test_duplicate_role_name_conflicts():
    table = _FakeTable(rows={"viewer": {"role_name": "viewer", "allowed_routes": [], "inherits": []}})
    with patch(_TABLE_PATH, return_value=table):
        with pytest.raises(HTTPException) as exc:
            await new_custom_role(
                data=CustomRBACRoleCreateRequest(role_name="viewer"),
                user_api_key_dict=_admin(),
            )
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_update_only_changes_provided_fields():
    table = _FakeTable(
        rows={
            "viewer": {
                "role_name": "viewer",
                "description": "read only",
                "allowed_routes": ["/key/info"],
                "inherits": [],
            }
        }
    )
    with patch(_TABLE_PATH, return_value=table):
        response = await update_custom_role(
            data=CustomRBACRoleUpdateRequest(role_name="viewer", allowed_routes=("/team/info",)),
            user_api_key_dict=_admin(),
        )
    assert response.allowed_routes == ("/team/info",)
    assert table.rows["viewer"]["description"] == "read only"


@pytest.mark.asyncio
async def test_update_missing_role_is_404():
    with patch(_TABLE_PATH, return_value=_FakeTable()):
        with pytest.raises(HTTPException) as exc:
            await update_custom_role(
                data=CustomRBACRoleUpdateRequest(role_name="ghost"),
                user_api_key_dict=_admin(),
            )
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_delete_removes_role():
    table = _FakeTable(rows={"viewer": {"role_name": "viewer", "allowed_routes": [], "inherits": []}})
    with patch(_TABLE_PATH, return_value=table):
        response = await delete_custom_role(
            data=CustomRBACRoleDeleteRequest(role_name="viewer"),
            user_api_key_dict=_admin(),
        )
    assert (response.role_name, response.status) == ("viewer", "deleted")
    assert table.rows == {}


@pytest.mark.asyncio
async def test_inheriting_unknown_role_is_rejected():
    table = _FakeTable()
    with patch(_TABLE_PATH, return_value=table):
        with pytest.raises(HTTPException) as exc:
            await new_custom_role(
                data=CustomRBACRoleCreateRequest(role_name="viewer", inherits=("ghost",)),
                user_api_key_dict=_admin(),
            )
    assert exc.value.status_code == 400
    assert table.rows == {}


@pytest.mark.asyncio
async def test_inheriting_existing_role_is_allowed():
    table = _FakeTable(rows={"base": {"role_name": "base", "allowed_routes": ["/key/info"], "inherits": []}})
    with patch(_TABLE_PATH, return_value=table):
        response = await new_custom_role(
            data=CustomRBACRoleCreateRequest(role_name="viewer", inherits=("base",)),
            user_api_key_dict=_admin(),
        )
    assert response.inherits == ("base",)
