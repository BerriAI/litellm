"""
Has all /sso/* routes

/sso/key/generate - handles user signing in with SSO and redirects to /sso/callback
/sso/callback - returns JWT Redirect Response that redirects to LiteLLM UI
"""

import asyncio
import os
import uuid
from typing import TYPE_CHECKING, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    LitellmUserRoles,
    NewUserRequest,
    ProxyErrorTypes,
    ProxyException,
    SSOUserDefinedValues,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.admin_ui_utils import (
    admin_ui_disabled,
    html_form,
    show_missing_vars_in_env,
)
from litellm.proxy.management_endpoints.internal_user_endpoints import new_user
from litellm.secret_managers.main import str_to_bool

if TYPE_CHECKING:
    from fastapi_sso.sso.base import OpenID
else:
    from typing import Any as OpenID

router = APIRouter()


@router.get("/sso/key/generate", tags=["experimental"], include_in_schema=False)
async def google_login(request: Request):
    """
    Create Proxy API Keys using Google Workspace SSO. Requires setting PROXY_BASE_URL in .env
    PROXY_BASE_URL should be the your deployed proxy endpoint, e.g. PROXY_BASE_URL="https://litellm-production-7002.up.railway.app/"
    Example:
    """
    from litellm.proxy.proxy_server import master_key, premium_user, prisma_client

    microsoft_client_id = os.getenv("MICROSOFT_CLIENT_ID", None)
    google_client_id = os.getenv("GOOGLE_CLIENT_ID", None)
    generic_client_id = os.getenv("GENERIC_CLIENT_ID", None)

    ####### Check if UI is disabled #######
    _disable_ui_flag = os.getenv("DISABLE_ADMIN_UI")
    if _disable_ui_flag is not None:
        is_disabled = str_to_bool(value=_disable_ui_flag)
        if is_disabled:
            return admin_ui_disabled()

    ####### Check if user is a Enterprise / Premium User #######
    if (
        microsoft_client_id is not None
        or google_client_id is not None
        or generic_client_id is not None
    ):
        if premium_user is not True:
            raise ProxyException(
                message="You must be a LiteLLM Enterprise user to use SSO. If you have a license please set `LITELLM_LICENSE` in your env. If you want to obtain a license meet with us here: https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat You are seeing this error message because You set one of `MICROSOFT_CLIENT_ID`, `GOOGLE_CLIENT_ID`, or `GENERIC_CLIENT_ID` in your env. Please unset this",
                type=ProxyErrorTypes.auth_error,
                param="premium_user",
                code=status.HTTP_403_FORBIDDEN,
            )

    ####### Detect DB + MASTER KEY in .env #######
    missing_env_vars = show_missing_vars_in_env()
    if missing_env_vars is not None:
        return missing_env_vars

    # get url from request
    redirect_url = os.getenv("PROXY_BASE_URL", str(request.base_url))
    ui_username = os.getenv("UI_USERNAME")
    if redirect_url.endswith("/"):
        redirect_url += "sso/callback"
    else:
        redirect_url += "/sso/callback"
    # Google SSO Auth
    if google_client_id is not None:
        from fastapi_sso.sso.google import GoogleSSO

        google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET", None)
        if google_client_secret is None:
            raise ProxyException(
                message="GOOGLE_CLIENT_SECRET not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="GOOGLE_CLIENT_SECRET",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        google_sso = GoogleSSO(
            client_id=google_client_id,
            client_secret=google_client_secret,
            redirect_uri=redirect_url,
        )
        verbose_proxy_logger.info(
            f"In /google-login/key/generate, \nGOOGLE_REDIRECT_URI: {redirect_url}\nGOOGLE_CLIENT_ID: {google_client_id}"
        )
        with google_sso:
            return await google_sso.get_login_redirect()
    # Microsoft SSO Auth
    elif microsoft_client_id is not None:
        from fastapi_sso.sso.microsoft import MicrosoftSSO

        microsoft_client_secret = os.getenv("MICROSOFT_CLIENT_SECRET", None)
        microsoft_tenant = os.getenv("MICROSOFT_TENANT", None)
        if microsoft_client_secret is None:
            raise ProxyException(
                message="MICROSOFT_CLIENT_SECRET not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="MICROSOFT_CLIENT_SECRET",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        microsoft_sso = MicrosoftSSO(
            client_id=microsoft_client_id,
            client_secret=microsoft_client_secret,
            tenant=microsoft_tenant,
            redirect_uri=redirect_url,
            allow_insecure_http=True,
        )
        with microsoft_sso:
            return await microsoft_sso.get_login_redirect()
    elif generic_client_id is not None:
        from fastapi_sso.sso.base import DiscoveryDocument
        from fastapi_sso.sso.generic import create_provider

        generic_client_secret = os.getenv("GENERIC_CLIENT_SECRET", None)
        generic_scope = os.getenv("GENERIC_SCOPE", "openid email profile").split(" ")
        generic_authorization_endpoint = os.getenv(
            "GENERIC_AUTHORIZATION_ENDPOINT", None
        )
        generic_token_endpoint = os.getenv("GENERIC_TOKEN_ENDPOINT", None)
        generic_userinfo_endpoint = os.getenv("GENERIC_USERINFO_ENDPOINT", None)
        if generic_client_secret is None:
            raise ProxyException(
                message="GENERIC_CLIENT_SECRET not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="GENERIC_CLIENT_SECRET",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        if generic_authorization_endpoint is None:
            raise ProxyException(
                message="GENERIC_AUTHORIZATION_ENDPOINT not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="GENERIC_AUTHORIZATION_ENDPOINT",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        if generic_token_endpoint is None:
            raise ProxyException(
                message="GENERIC_TOKEN_ENDPOINT not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="GENERIC_TOKEN_ENDPOINT",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        if generic_userinfo_endpoint is None:
            raise ProxyException(
                message="GENERIC_USERINFO_ENDPOINT not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="GENERIC_USERINFO_ENDPOINT",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        verbose_proxy_logger.debug(
            f"authorization_endpoint: {generic_authorization_endpoint}\ntoken_endpoint: {generic_token_endpoint}\nuserinfo_endpoint: {generic_userinfo_endpoint}"
        )
        verbose_proxy_logger.debug(
            f"GENERIC_REDIRECT_URI: {redirect_url}\nGENERIC_CLIENT_ID: {generic_client_id}\n"
        )
        discovery = DiscoveryDocument(
            authorization_endpoint=generic_authorization_endpoint,
            token_endpoint=generic_token_endpoint,
            userinfo_endpoint=generic_userinfo_endpoint,
        )
        SSOProvider = create_provider(name="oidc", discovery_document=discovery)
        generic_sso = SSOProvider(
            client_id=generic_client_id,
            client_secret=generic_client_secret,
            redirect_uri=redirect_url,
            allow_insecure_http=True,
            scope=generic_scope,
        )
        with generic_sso:
            # TODO: state should be a random string and added to the user session with cookie
            # or a cryptographicly signed state that we can verify stateless
            # For simplification we are using a static state, this is not perfect but some
            # SSO providers do not allow stateless verification
            redirect_params = {}
            state = os.getenv("GENERIC_CLIENT_STATE", None)

            if state:
                redirect_params["state"] = state
            elif "okta" in generic_authorization_endpoint:
                redirect_params["state"] = (
                    uuid.uuid4().hex
                )  # set state param for okta - required
            return await generic_sso.get_login_redirect(**redirect_params)  # type: ignore
    elif ui_username is not None:
        # No Google, Microsoft SSO
        # Use UI Credentials set in .env
        from fastapi.responses import HTMLResponse

        return HTMLResponse(content=html_form, status_code=200)
    else:
        from fastapi.responses import HTMLResponse

        return HTMLResponse(content=html_form, status_code=200)


@router.get("/sso/callback", tags=["experimental"], include_in_schema=False)
async def auth_callback(request: Request):
    """Verify login"""
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        generate_key_helper_fn,
    )
    from litellm.proxy.proxy_server import (
        general_settings,
        master_key,
        premium_user,
        prisma_client,
        ui_access_mode,
        user_custom_sso,
    )

    microsoft_client_id = os.getenv("MICROSOFT_CLIENT_ID", None)
    google_client_id = os.getenv("GOOGLE_CLIENT_ID", None)
    generic_client_id = os.getenv("GENERIC_CLIENT_ID", None)
    # get url from request
    if master_key is None:
        raise ProxyException(
            message="Master Key not set for Proxy. Please set Master Key to use Admin UI. Set `LITELLM_MASTER_KEY` in .env or set general_settings:master_key in config.yaml.  https://docs.litellm.ai/docs/proxy/virtual_keys. If set, use `--detailed_debug` to debug issue.",
            type=ProxyErrorTypes.auth_error,
            param="master_key",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    redirect_url = os.getenv("PROXY_BASE_URL", str(request.base_url))
    if redirect_url.endswith("/"):
        redirect_url += "sso/callback"
    else:
        redirect_url += "/sso/callback"

    result = None
    if google_client_id is not None:
        from fastapi_sso.sso.google import GoogleSSO

        google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET", None)
        if google_client_secret is None:
            raise ProxyException(
                message="GOOGLE_CLIENT_SECRET not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="GOOGLE_CLIENT_SECRET",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        google_sso = GoogleSSO(
            client_id=google_client_id,
            redirect_uri=redirect_url,
            client_secret=google_client_secret,
        )
        result = await google_sso.verify_and_process(request)
    elif microsoft_client_id is not None:
        from fastapi_sso.sso.microsoft import MicrosoftSSO

        microsoft_client_secret = os.getenv("MICROSOFT_CLIENT_SECRET", None)
        microsoft_tenant = os.getenv("MICROSOFT_TENANT", None)
        if microsoft_client_secret is None:
            raise ProxyException(
                message="MICROSOFT_CLIENT_SECRET not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="MICROSOFT_CLIENT_SECRET",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        if microsoft_tenant is None:
            raise ProxyException(
                message="MICROSOFT_TENANT not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="MICROSOFT_TENANT",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        microsoft_sso = MicrosoftSSO(
            client_id=microsoft_client_id,
            client_secret=microsoft_client_secret,
            tenant=microsoft_tenant,
            redirect_uri=redirect_url,
            allow_insecure_http=True,
        )
        result = await microsoft_sso.verify_and_process(request)
    elif generic_client_id is not None:
        # make generic sso provider
        from fastapi_sso.sso.base import DiscoveryDocument, OpenID
        from fastapi_sso.sso.generic import create_provider

        generic_client_secret = os.getenv("GENERIC_CLIENT_SECRET", None)
        generic_scope = os.getenv("GENERIC_SCOPE", "openid email profile").split(" ")
        generic_authorization_endpoint = os.getenv(
            "GENERIC_AUTHORIZATION_ENDPOINT", None
        )
        generic_token_endpoint = os.getenv("GENERIC_TOKEN_ENDPOINT", None)
        generic_userinfo_endpoint = os.getenv("GENERIC_USERINFO_ENDPOINT", None)
        generic_include_client_id = (
            os.getenv("GENERIC_INCLUDE_CLIENT_ID", "false").lower() == "true"
        )
        if generic_client_secret is None:
            raise ProxyException(
                message="GENERIC_CLIENT_SECRET not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="GENERIC_CLIENT_SECRET",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        if generic_authorization_endpoint is None:
            raise ProxyException(
                message="GENERIC_AUTHORIZATION_ENDPOINT not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="GENERIC_AUTHORIZATION_ENDPOINT",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        if generic_token_endpoint is None:
            raise ProxyException(
                message="GENERIC_TOKEN_ENDPOINT not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="GENERIC_TOKEN_ENDPOINT",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        if generic_userinfo_endpoint is None:
            raise ProxyException(
                message="GENERIC_USERINFO_ENDPOINT not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="GENERIC_USERINFO_ENDPOINT",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        verbose_proxy_logger.debug(
            f"authorization_endpoint: {generic_authorization_endpoint}\ntoken_endpoint: {generic_token_endpoint}\nuserinfo_endpoint: {generic_userinfo_endpoint}"
        )
        verbose_proxy_logger.debug(
            f"GENERIC_REDIRECT_URI: {redirect_url}\nGENERIC_CLIENT_ID: {generic_client_id}\n"
        )

        generic_user_id_attribute_name = os.getenv(
            "GENERIC_USER_ID_ATTRIBUTE", "preferred_username"
        )
        generic_user_display_name_attribute_name = os.getenv(
            "GENERIC_USER_DISPLAY_NAME_ATTRIBUTE", "sub"
        )
        generic_user_email_attribute_name = os.getenv(
            "GENERIC_USER_EMAIL_ATTRIBUTE", "email"
        )
        generic_user_role_attribute_name = os.getenv(
            "GENERIC_USER_ROLE_ATTRIBUTE", "role"
        )
        generic_user_first_name_attribute_name = os.getenv(
            "GENERIC_USER_FIRST_NAME_ATTRIBUTE", "first_name"
        )
        generic_user_last_name_attribute_name = os.getenv(
            "GENERIC_USER_LAST_NAME_ATTRIBUTE", "last_name"
        )

        generic_provider_attribute_name = os.getenv(
            "GENERIC_USER_PROVIDER_ATTRIBUTE", "provider"
        )

        verbose_proxy_logger.debug(
            f" generic_user_id_attribute_name: {generic_user_id_attribute_name}\n generic_user_email_attribute_name: {generic_user_email_attribute_name}\n generic_user_role_attribute_name: {generic_user_role_attribute_name}"
        )

        discovery = DiscoveryDocument(
            authorization_endpoint=generic_authorization_endpoint,
            token_endpoint=generic_token_endpoint,
            userinfo_endpoint=generic_userinfo_endpoint,
        )

        def response_convertor(response, client):
            return OpenID(
                id=response.get(generic_user_id_attribute_name),
                display_name=response.get(generic_user_display_name_attribute_name),
                email=response.get(generic_user_email_attribute_name),
                first_name=response.get(generic_user_first_name_attribute_name),
                last_name=response.get(generic_user_last_name_attribute_name),
                provider=response.get(generic_provider_attribute_name),
            )

        SSOProvider = create_provider(
            name="oidc",
            discovery_document=discovery,
            response_convertor=response_convertor,
        )
        generic_sso = SSOProvider(
            client_id=generic_client_id,
            client_secret=generic_client_secret,
            redirect_uri=redirect_url,
            allow_insecure_http=True,
            scope=generic_scope,
        )
        verbose_proxy_logger.debug("calling generic_sso.verify_and_process")
        result = await generic_sso.verify_and_process(
            request, params={"include_client_id": generic_include_client_id}
        )
        verbose_proxy_logger.debug("generic result: %s", result)

    # User is Authe'd in - generate key for the UI to access Proxy
    user_email: Optional[str] = getattr(result, "email", None)
    user_id: Optional[str] = getattr(result, "id", None) if result is not None else None

    if user_email is not None and os.getenv("ALLOWED_EMAIL_DOMAINS") is not None:
        email_domain = user_email.split("@")[1]
        allowed_domains = os.getenv("ALLOWED_EMAIL_DOMAINS").split(",")  # type: ignore
        if email_domain not in allowed_domains:
            raise HTTPException(
                status_code=401,
                detail={
                    "message": "The email domain={}, is not an allowed email domain={}. Contact your admin to change this.".format(
                        email_domain, allowed_domains
                    )
                },
            )

    # generic client id
    if generic_client_id is not None and result is not None:
        user_id = getattr(result, "id", None)
        user_email = getattr(result, "email", None)
        user_role = getattr(result, generic_user_role_attribute_name, None)  # type: ignore

    if user_id is None and result is not None:
        _first_name = getattr(result, "first_name", "") or ""
        _last_name = getattr(result, "last_name", "") or ""
        user_id = _first_name + _last_name

    if user_email is not None and (user_id is None or len(user_id) == 0):
        user_id = user_email

    user_info = None
    user_id_models: List = []
    max_internal_user_budget = litellm.max_internal_user_budget
    internal_user_budget_duration = litellm.internal_user_budget_duration

    # User might not be already created on first generation of key
    # But if it is, we want their models preferences
    default_ui_key_values = {
        "duration": "24hr",
        "key_max_budget": 0.01,
        "aliases": {},
        "config": {},
        "spend": 0,
        "team_id": "litellm-dashboard",
    }
    user_defined_values: Optional[SSOUserDefinedValues] = None

    if user_custom_sso is not None:
        if asyncio.iscoroutinefunction(user_custom_sso):
            user_defined_values = await user_custom_sso(result)  # type: ignore
        else:
            raise ValueError("user_custom_sso must be a coroutine function")
    elif user_id is not None:
        user_defined_values = SSOUserDefinedValues(
            models=user_id_models,
            user_id=user_id,
            user_email=user_email,
            max_budget=max_internal_user_budget,
            user_role=None,
            budget_duration=internal_user_budget_duration,
        )

    _user_id_from_sso = user_id
    user_role = None
    try:
        if prisma_client is not None:
            user_info = await prisma_client.get_data(user_id=user_id, table_name="user")
            verbose_proxy_logger.debug(
                f"user_info: {user_info}; litellm.default_internal_user_params: {litellm.default_internal_user_params}"
            )
            if user_info is None:
                ## check if user-email in db ##
                user_info = await prisma_client.db.litellm_usertable.find_first(
                    where={"user_email": user_email}
                )

            if user_info is not None and user_id is not None:
                user_defined_values = SSOUserDefinedValues(
                    models=getattr(user_info, "models", user_id_models),
                    user_id=user_id,
                    user_email=getattr(user_info, "user_email", user_email),
                    user_role=getattr(user_info, "user_role", None),
                    max_budget=getattr(
                        user_info, "max_budget", max_internal_user_budget
                    ),
                    budget_duration=getattr(
                        user_info, "budget_duration", internal_user_budget_duration
                    ),
                )

                user_role = getattr(user_info, "user_role", None)

                # update id
                await prisma_client.db.litellm_usertable.update_many(
                    where={"user_email": user_email}, data={"user_id": user_id}  # type: ignore
                )
            else:
                # user not in DB, insert User into LiteLLM DB
                user_role = await insert_sso_user(
                    result_openid=result,
                    user_defined_values=user_defined_values,
                )
    except Exception:
        pass

    if user_defined_values is None:
        raise Exception(
            "Unable to map user identity to known values. 'user_defined_values' is None. File an issue - https://github.com/BerriAI/litellm/issues"
        )

    verbose_proxy_logger.info(
        f"user_defined_values for creating ui key: {user_defined_values}"
    )

    default_ui_key_values.update(user_defined_values)
    default_ui_key_values["request_type"] = "key"
    response = await generate_key_helper_fn(
        **default_ui_key_values,  # type: ignore
        table_name="key",
    )

    key = response["token"]  # type: ignore
    user_id = response["user_id"]  # type: ignore

    # This should always be true
    # User_id on SSO == user_id in the LiteLLM_VerificationToken Table
    assert user_id == _user_id_from_sso
    litellm_dashboard_ui = "/ui/"
    user_role = user_role or LitellmUserRoles.INTERNAL_USER_VIEW_ONLY.value
    if (
        os.getenv("PROXY_ADMIN_ID", None) is not None
        and os.environ["PROXY_ADMIN_ID"] == user_id
    ):
        # checks if user is admin
        user_role = LitellmUserRoles.PROXY_ADMIN.value

    verbose_proxy_logger.debug(
        f"user_role: {user_role}; ui_access_mode: {ui_access_mode}"
    )
    ## CHECK IF ROLE ALLOWED TO USE PROXY ##
    if ui_access_mode == "admin_only" and (
        user_role != LitellmUserRoles.PROXY_ADMIN.value
        or user_role != LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value
    ):
        verbose_proxy_logger.debug("EXCEPTION RAISED")
        raise HTTPException(
            status_code=401,
            detail={
                "error": f"User not allowed to access proxy. User role={user_role}, proxy mode={ui_access_mode}"
            },
        )

    import jwt

    jwt_token = jwt.encode(  # type: ignore
        {
            "user_id": user_id,
            "key": key,
            "user_email": user_email,
            "user_role": user_role,
            "login_method": "sso",
            "premium_user": premium_user,
            "auth_header_name": general_settings.get(
                "litellm_key_header_name", "Authorization"
            ),
        },
        master_key,
        algorithm="HS256",
    )
    if user_id is not None and isinstance(user_id, str):
        litellm_dashboard_ui += "?userID=" + user_id
    redirect_response = RedirectResponse(url=litellm_dashboard_ui, status_code=303)
    redirect_response.set_cookie(key="token", value=jwt_token, secure=True)
    return redirect_response


async def insert_sso_user(
    result_openid: Optional[OpenID],
    user_defined_values: Optional[SSOUserDefinedValues] = None,
) -> str:
    """
    Helper function to create a New User in LiteLLM DB after a successful SSO login

    Args:
        result_openid (OpenID): User information in OpenID format if the login was successful.
        user_defined_values (Optional[SSOUserDefinedValues], optional): LiteLLM SSOValues / fields that were read
    """
    verbose_proxy_logger.debug(
        f"Inserting SSO user into DB. User values: {user_defined_values}"
    )

    if user_defined_values is None:
        raise ValueError("user_defined_values is None")

    if litellm.default_internal_user_params:
        user_defined_values.update(litellm.default_internal_user_params)  # type: ignore

    # Set budget for internal users
    if user_defined_values.get("user_role") == LitellmUserRoles.INTERNAL_USER.value:
        if user_defined_values.get("max_budget") is None:
            user_defined_values["max_budget"] = litellm.max_internal_user_budget
        if user_defined_values.get("budget_duration") is None:
            user_defined_values["budget_duration"] = (
                litellm.internal_user_budget_duration
            )

    if user_defined_values["user_role"] is None:
        user_defined_values["user_role"] = LitellmUserRoles.INTERNAL_USER_VIEW_ONLY

    new_user_request = NewUserRequest(
        user_id=user_defined_values["user_id"],
        user_email=user_defined_values["user_email"],
        user_role=user_defined_values["user_role"],  # type: ignore
        max_budget=user_defined_values["max_budget"],
        budget_duration=user_defined_values["budget_duration"],
    )

    if result_openid:
        new_user_request.metadata = {"auth_provider": result_openid.provider}

    await new_user(data=new_user_request, user_api_key_dict=UserAPIKeyAuth())

    return user_defined_values["user_role"] or LitellmUserRoles.INTERNAL_USER_VIEW_ONLY


@router.get(
    "/sso/get/ui_settings",
    tags=["experimental"],
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
)
async def get_ui_settings(request: Request):
    from litellm.proxy.proxy_server import general_settings

    _proxy_base_url = os.getenv("PROXY_BASE_URL", None)
    _logout_url = os.getenv("PROXY_LOGOUT_URL", None)

    default_team_disabled = general_settings.get("default_team_disabled", False)
    if "PROXY_DEFAULT_TEAM_DISABLED" in os.environ:
        if os.environ["PROXY_DEFAULT_TEAM_DISABLED"].lower() == "true":
            default_team_disabled = True

    return {
        "PROXY_BASE_URL": _proxy_base_url,
        "PROXY_LOGOUT_URL": _logout_url,
        "DEFAULT_TEAM_DISABLED": default_team_disabled,
    }
