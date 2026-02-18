"""
Login utilities for handling user authentication in the proxy server.

This module contains the core login logic that can be reused across different
login endpoints (e.g., /login and /v2/login).
"""

import os
import secrets
from typing import Literal, Optional, cast

from fastapi import HTTPException

import litellm
from litellm.constants import LITELLM_PROXY_ADMIN_NAME
from litellm.proxy._types import (
    LiteLLM_UserTable,
    LitellmUserRoles,
    ProxyErrorTypes,
    ProxyException,
    UpdateUserRequest,
    UserAPIKeyAuth,
    hash_token,
)
from litellm.proxy.management_endpoints.internal_user_endpoints import user_update
from litellm.proxy.management_endpoints.key_management_endpoints import (
    generate_key_helper_fn,
)
from litellm.proxy.management_endpoints.ui_sso import (
    get_disabled_non_admin_personal_key_creation,
)
from litellm.proxy.utils import PrismaClient, get_server_root_path
from litellm.secret_managers.main import get_secret_bool
from litellm.types.proxy.ui_sso import ReturnedUITokenObject


def get_ui_credentials(master_key: Optional[str]) -> tuple[str, str]:
    """
    Get UI username and password from environment variables or master key.

    Args:
        master_key: Master key for the proxy (used as fallback for password)

    Returns:
        tuple[str, str]: A tuple containing (ui_username, ui_password)

    Raises:
        ProxyException: If neither UI_PASSWORD nor master_key is available
    """
    ui_username = os.getenv("UI_USERNAME", "admin")
    ui_password = os.getenv("UI_PASSWORD", None)
    if ui_password is None:
        ui_password = str(master_key) if master_key is not None else None
    if ui_password is None:
        raise ProxyException(
            message="set Proxy master key to use UI. https://docs.litellm.ai/docs/proxy/virtual_keys. If set, use `--detailed_debug` to debug issue.",
            type=ProxyErrorTypes.auth_error,
            param="UI_PASSWORD",
            code=500,
        )
    return ui_username, ui_password


class LoginResult:
    """Result object containing authentication data from login."""

    user_id: str
    key: str
    user_email: Optional[str]
    user_role: str
    login_method: Literal["sso", "username_password"]

    def __init__(
        self,
        user_id: str,
        key: str,
        user_email: Optional[str],
        user_role: str,
        login_method: Literal["sso", "username_password"] = "username_password",
    ):
        self.user_id = user_id
        self.key = key
        self.user_email = user_email
        self.user_role = user_role
        self.login_method = login_method


async def authenticate_user(  # noqa: PLR0915
    username: str,
    password: str,
    master_key: Optional[str],
    prisma_client: Optional[PrismaClient],
) -> LoginResult:
    """
    Authenticate a user and generate an API key for UI access.

    This function handles two login scenarios:
    1. Admin login using UI_USERNAME and UI_PASSWORD
    2. User login using email and password from database

    Args:
        username: Username or email from the login form
        password: Password from the login form
        master_key: Master key for the proxy (required)
        prisma_client: Prisma database client (optional)

    Returns:
        LoginResult: Object containing authentication data

    Raises:
        ProxyException: If authentication fails or required configuration is missing
    """
    if master_key is None:
        raise ProxyException(
            message="Master Key not set for Proxy. Please set Master Key to use Admin UI. Set `LITELLM_MASTER_KEY` in .env or set general_settings:master_key in config.yaml.  https://docs.litellm.ai/docs/proxy/virtual_keys. If set, use `--detailed_debug` to debug issue.",
            type=ProxyErrorTypes.auth_error,
            param="master_key",
            code=500,
        )

    ui_username, ui_password = get_ui_credentials(master_key)

    # Check if we can find the `username` in the db. On the UI, users can enter username=their email
    _user_row: Optional[LiteLLM_UserTable] = None
    user_role: Optional[
        Literal[
            LitellmUserRoles.PROXY_ADMIN,
            LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
            LitellmUserRoles.INTERNAL_USER,
            LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        ]
    ] = None

    if prisma_client is not None:
        _user_row = cast(
            Optional[LiteLLM_UserTable],
            await prisma_client.db.litellm_usertable.find_first(
                where={"user_email": {"equals": username, "mode": "insensitive"}}
            ),
        )

    """
    To login to Admin UI, we support the following
    - Login with UI_USERNAME and UI_PASSWORD
    - Login with Invite Link `user_email` and `password` combination
    """
    if secrets.compare_digest(
        username.encode("utf-8"), ui_username.encode("utf-8")
    ) and secrets.compare_digest(password.encode("utf-8"), ui_password.encode("utf-8")):
        # Non SSO -> If user is using UI_USERNAME and UI_PASSWORD they are Proxy admin
        user_role = LitellmUserRoles.PROXY_ADMIN
        user_id = LITELLM_PROXY_ADMIN_NAME

        # we want the key created to have PROXY_ADMIN_PERMISSIONS
        key_user_id = LITELLM_PROXY_ADMIN_NAME
        if (
            os.getenv("PROXY_ADMIN_ID", None) is not None
            and os.environ["PROXY_ADMIN_ID"] == user_id
        ) or user_id == LITELLM_PROXY_ADMIN_NAME:
            # checks if user is admin
            key_user_id = os.getenv("PROXY_ADMIN_ID", LITELLM_PROXY_ADMIN_NAME)

        # Admin is Authe'd in - generate key for the UI to access Proxy

        # ensure this user is set as the proxy admin, in this route there is no sso, we can assume this user is only the admin
        await user_update(
            data=UpdateUserRequest(
                user_id=key_user_id,
                user_role=user_role,
            ),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
            ),
        )

        if os.getenv("DATABASE_URL") is not None:
            response = await generate_key_helper_fn(
                request_type="key",
                **{
                    "user_role": LitellmUserRoles.PROXY_ADMIN,
                    "duration": "24hr",
                    "key_max_budget": litellm.max_ui_session_budget,
                    "models": [],
                    "aliases": {},
                    "config": {},
                    "spend": 0,
                    "user_id": key_user_id,
                    "team_id": "litellm-dashboard",
                },  # type: ignore
            )
        else:
            raise ProxyException(
                message="No Database connected. Set DATABASE_URL in .env. If set, use `--detailed_debug` to debug issue.",
                type=ProxyErrorTypes.auth_error,
                param="DATABASE_URL",
                code=500,
            )

        key = response["token"]  # type: ignore

        if get_secret_bool("EXPERIMENTAL_UI_LOGIN"):
            from litellm.proxy.auth.auth_checks import ExperimentalUIJWTToken

            user_info: Optional[LiteLLM_UserTable] = None
            if _user_row is not None:
                user_info = _user_row
            elif (
                user_id is not None
            ):  # if user_id is not None, we are using the UI_USERNAME and UI_PASSWORD
                user_info = LiteLLM_UserTable(
                    user_id=user_id,
                    user_role=user_role,
                    models=[],
                    max_budget=litellm.max_ui_session_budget,
                )
            if user_info is None:
                raise HTTPException(
                    status_code=401,
                    detail={
                        "error": "User Information is required for experimental UI login"
                    },
                )

            key = ExperimentalUIJWTToken.get_experimental_ui_login_jwt_auth_token(
                user_info
            )

        return LoginResult(
            user_id=user_id,
            key=key,
            user_email=None,
            user_role=user_role,
            login_method="username_password",
        )

    elif _user_row is not None:
        """
        When sharing invite links

        -> if the user has no role in the DB assume they are only a viewer
        """
        user_id = getattr(_user_row, "user_id", "unknown")
        user_role = getattr(
            _user_row, "user_role", LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
        )
        user_email = getattr(_user_row, "user_email", "unknown")
        _password = getattr(_user_row, "password", "unknown")

        if _password is None:
            raise ProxyException(
                message="User has no password set. Please set a password for the user via `/user/update`.",
                type=ProxyErrorTypes.auth_error,
                param="password",
                code=401,
            )

        # check if password == _user_row.password
        hash_password = hash_token(token=password)
        if secrets.compare_digest(
            password.encode("utf-8"), _password.encode("utf-8")
        ) or secrets.compare_digest(hash_password.encode("utf-8"), _password.encode("utf-8")):
            if os.getenv("DATABASE_URL") is not None:
                response = await generate_key_helper_fn(
                    request_type="key",
                    **{  # type: ignore
                        "user_role": user_role,
                        "duration": "24hr",
                        "key_max_budget": litellm.max_ui_session_budget,
                        "models": [],
                        "aliases": {},
                        "config": {},
                        "spend": 0,
                        "user_id": user_id,
                        "team_id": "litellm-dashboard",
                    },
                )
            else:
                raise ProxyException(
                    message="No Database connected. Set DATABASE_URL in .env. If set, use `--detailed_debug` to debug issue.",
                    type=ProxyErrorTypes.auth_error,
                    param="DATABASE_URL",
                    code=500,
                )

            key = response["token"]  # type: ignore

            return LoginResult(
                user_id=user_id,
                key=key,
                user_email=user_email,
                user_role=cast(str, user_role),
                login_method="username_password",
            )
        else:
            raise ProxyException(
                message=f"Invalid credentials used to access UI.\nNot valid credentials for {username}",
                type=ProxyErrorTypes.auth_error,
                param="invalid_credentials",
                code=401,
            )
    else:
        raise ProxyException(
            message="Invalid credentials used to access UI.\nCheck 'UI_USERNAME', 'UI_PASSWORD' in .env file",
            type=ProxyErrorTypes.auth_error,
            param="invalid_credentials",
            code=401,
        )


def create_ui_token_object(
    login_result: LoginResult,
    general_settings: dict,
    premium_user: bool,
) -> ReturnedUITokenObject:
    """
    Create a ReturnedUITokenObject from a LoginResult.

    Args:
        login_result: The result from authenticate_user
        general_settings: General proxy settings dictionary
        premium_user: Whether premium features are enabled

    Returns:
        ReturnedUITokenObject: Token object ready for JWT encoding
    """
    disabled_non_admin_personal_key_creation = (
        get_disabled_non_admin_personal_key_creation()
    )

    return ReturnedUITokenObject(
        user_id=login_result.user_id,
        key=login_result.key,
        user_email=login_result.user_email,
        user_role=login_result.user_role,
        login_method=login_result.login_method,
        premium_user=premium_user,
        auth_header_name=general_settings.get(
            "litellm_key_header_name", "Authorization"
        ),
        disabled_non_admin_personal_key_creation=disabled_non_admin_personal_key_creation,
        server_root_path=get_server_root_path(),
    )

