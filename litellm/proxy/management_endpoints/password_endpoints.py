"""
Self-service password management.

/user/password/change

Deliberately NOT wrapped in `management_endpoint_wrapper`: the wrapper emits
request kwargs to OTEL spans, which would log plaintext passwords. The audit
signal is emitted by hand below, with field names only, never values.
"""

from typing import TYPE_CHECKING, Annotated, Final

from fastapi import APIRouter, Depends, HTTPException
from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import (
    UI_TEAM_ID,
    ChangePasswordRequest,
    ChangePasswordResponse,
    CommonProxyErrors,
    HTTPExceptionErrorDetail,
    LitellmTableNames,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.login_utils import PASSWORD_SESSION_METADATA
from litellm.proxy.auth.password_policy import (
    get_hibp_client,
    validate_password_not_breached,
    validate_password_policy,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.session_endpoints import revoke_ui_session_keys
from litellm.proxy.management_helpers.audit_logs import create_object_audit_log
from litellm.proxy.utils import hash_password, verify_password
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.user_repository import UserRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models
    from prisma import types as prisma_types

    from litellm.proxy.utils import PrismaClient

router: Final = APIRouter()

_PASSWORD_CHANGED_AUDIT_VALUES: Final = '{"fields_changed": ["password"]}'
_KEY_METADATA: Final = TypeAdapter(dict[str, object])


def _error_detail(message: str) -> HTTPExceptionErrorDetail:
    detail: Final[HTTPExceptionErrorDetail] = {"error": message}
    return detail


def _is_password_login_session(user_api_key_dict: UserAPIKeyAuth) -> bool:
    if user_api_key_dict.team_id != UI_TEAM_ID:
        return False
    key_metadata: Final = _KEY_METADATA.validate_python(user_api_key_dict.metadata)
    return all(key_metadata.get(k) == v for k, v in PASSWORD_SESSION_METADATA.items())


def _user_table(
    prisma_client: "PrismaClient | None",
) -> "TableActions[prisma_models.LiteLLM_UserTable]":
    user_table: Final[TableActions[prisma_models.LiteLLM_UserTable]] = UserRepository(prisma_client).table
    return user_table


@router.post(
    "/user/password/change",
    tags=("Internal User management",),
    dependencies=(Depends(user_api_key_auth),),
)
async def change_password(
    data: ChangePasswordRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    hibp_client: Annotated[AsyncHTTPHandler, Depends(get_hibp_client)],
) -> ChangePasswordResponse:
    """
    Change the calling user's own password.

    Only callable with the dashboard session issued by a username/password
    login; SSO sessions and virtual keys are rejected with 403. Requires the
    current password. The new password must differ from the
    current one and satisfy the configured password policy
    (`general_settings.password_policy_*`: minimum length, character classes,
    and, when enabled, breached-password screening via haveibeenpwned.com).
    A successful change lifts any pending forced password reset
    (`password_reset_required`) on the account.

    Parameters:
    - current_password: str - The user's current password.
    - new_password: str - The password to change to.
    """
    from litellm.proxy.proxy_server import general_settings, litellm_proxy_admin_name, prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail=_error_detail(CommonProxyErrors.db_not_connected_error.value),
        )

    if not _is_password_login_session(user_api_key_dict):
        raise HTTPException(
            status_code=403,
            detail=_error_detail(
                "Passwords can only be changed from a dashboard session created by logging in with a password."
            ),
        )

    user_id: Final = user_api_key_dict.user_id
    if user_id is None:
        raise HTTPException(
            status_code=400,
            detail=_error_detail("No user is associated with this session, so there is no password to change."),
        )

    find_user: Final[prisma_types.LiteLLM_UserTableWhereInput] = {"user_id": user_id}
    user_row: Final = await _user_table(prisma_client).find_first(where=find_user)
    stored_password: Final = user_row.password if user_row is not None else None
    if stored_password is None:
        raise HTTPException(
            status_code=400,
            detail=_error_detail(
                "This account has no password set, so there is no password to change. "
                "Passwords are set through an invitation link (POST /invitation/new)."
            ),
        )

    if not verify_password(data.current_password, stored_password):
        raise HTTPException(status_code=400, detail=_error_detail("Current password is incorrect."))

    if data.new_password == data.current_password:
        raise HTTPException(
            status_code=400,
            detail=_error_detail("New password must be different from the current password."),
        )

    validate_password_policy(data.new_password, general_settings)
    await validate_password_not_breached(data.new_password, general_settings, hibp_client)

    password_update: Final[prisma_types.LiteLLM_UserTableUpdateInput] = {
        "password": hash_password(data.new_password),
        "password_reset_required": False,
        "last_breach_check_at": None,
    }
    await _user_table(prisma_client).update(where=find_user, data=password_update)

    # The old password may have been compromised; revoke every other UI session
    # so a holder of a stolen session token is cut off. The caller's own session
    # is kept — they just proved they hold the current password.
    await revoke_ui_session_keys(
        user_id=user_id,
        user_api_key_dict=user_api_key_dict,
        keep_hashed_token=user_api_key_dict.token,
    )

    verbose_proxy_logger.info("Password changed via /user/password/change for user_id=%s", user_id)
    await create_object_audit_log(
        object_id=user_id,
        action="updated",
        litellm_changed_by=None,
        user_api_key_dict=user_api_key_dict,
        litellm_proxy_admin_name=litellm_proxy_admin_name,
        table_name=LitellmTableNames.USER_TABLE_NAME,
        after_value=_PASSWORD_CHANGED_AUDIT_VALUES,
    )
    return ChangePasswordResponse(user_id=user_id, message="Password updated successfully.")
