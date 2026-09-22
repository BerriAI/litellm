from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LiteLLM_UserTable, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.liteask.auth import AdminRefusal, assert_fresh_admin, check_fresh_admin
from litellm.types.proxy.auth.auth_checks import UserNotFoundError


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "caller",
    [
        UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY),
        UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.INTERNAL_USER),
        UserAPIKeyAuth(user_id="admin"),
        UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    ],
)
async def test_non_admin_and_identityless_credentials_are_denied_before_lookup(caller: UserAPIKeyAuth) -> None:
    lookup = AsyncMock()
    result = await check_fresh_admin(caller, lookup)
    assert isinstance(result, AdminRefusal)
    assert result.status_code == 403
    lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_same_session_is_rechecked_after_role_revocation() -> None:
    caller = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)
    lookup = AsyncMock(
        side_effect=[
            LiteLLM_UserTable(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN),
            LiteLLM_UserTable(user_id="admin", user_role=LitellmUserRoles.INTERNAL_USER),
        ]
    )
    assert await assert_fresh_admin(caller, lookup=lookup) is caller
    with pytest.raises(HTTPException) as error:
        await assert_fresh_admin(caller, lookup=lookup)
    assert error.value.status_code == 403
    assert lookup.await_count == 2
    lookup.assert_awaited_with("admin")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row",
    [
        None,
        LiteLLM_UserTable(user_id="another-admin", user_role=LitellmUserRoles.PROXY_ADMIN),
        LiteLLM_UserTable(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN, metadata={"scim_active": False}),
    ],
)
async def test_missing_or_different_live_identity_is_denied(row: LiteLLM_UserTable | None) -> None:
    caller = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)
    result = await check_fresh_admin(caller, AsyncMock(return_value=row))
    assert isinstance(result, AdminRefusal)
    assert result.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,status", [(UserNotFoundError(user_id="admin"), 403), (OSError("private DB URL"), 503)]
)
async def test_lookup_failures_cannot_restore_minted_admin_role(failure: Exception, status: int) -> None:
    caller = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)
    result = await check_fresh_admin(caller, AsyncMock(side_effect=failure))
    assert isinstance(result, AdminRefusal)
    assert result.status_code == status
    assert "private DB URL" not in result.message
