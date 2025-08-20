"""
Has all /sso/* routes

/sso/key/generate - handles user signing in with SSO and redirects to /sso/callback
/sso/callback - returns JWT Redirect Response that redirects to LiteLLM UI

/sso/debug/login - handles user signing in with SSO and redirects to /sso/debug/callback
/sso/debug/callback - returns the OpenID object returned by the SSO provider
"""

import asyncio
import os
import uuid
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching import DualCache
from litellm.constants import MAX_SPENDLOG_ROWS_TO_QUERY
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import (
    CommonProxyErrors,
    LiteLLM_UserTable,
    LitellmUserRoles,
    Member,
    NewTeamRequest,
    NewUserRequest,
    NewUserResponse,
    ProxyErrorTypes,
    ProxyException,
    SSOUserDefinedValues,
    TeamMemberAddRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import ExperimentalUIJWTToken, get_user_object
from litellm.proxy.auth.auth_utils import _has_user_setup_sso
from litellm.proxy.auth.handle_jwt import JWTHandler
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.admin_ui_utils import (
    admin_ui_disabled,
    show_missing_vars_in_env,
)
from litellm.proxy.common_utils.html_forms.jwt_display_template import (
    jwt_display_template,
)
from litellm.proxy.common_utils.html_forms.ui_login import html_form
from litellm.proxy.management_endpoints.internal_user_endpoints import new_user
from litellm.proxy.management_endpoints.sso_helper_utils import (
    check_is_admin_only_access,
    has_admin_ui_access,
)
from litellm.proxy.management_endpoints.team_endpoints import new_team, team_member_add
from litellm.proxy.management_endpoints.types import CustomOpenID
from litellm.proxy.utils import (
    PrismaClient,
    ProxyLogging,
    get_custom_url,
    get_server_root_path,
)
from litellm.secret_managers.main import get_secret_bool, str_to_bool
from litellm.types.proxy.management_endpoints.ui_sso import *

if TYPE_CHECKING:
    from fastapi_sso.sso.base import OpenID
else:
    from typing import Any as OpenID

router = APIRouter()


@router.get("/sso/key/generate", tags=["experimental"], include_in_schema=False)
async def google_login(request: Request, source: Optional[str] = None, key: Optional[str] = None):  # noqa: PLR0915
    """
    Create Proxy API Keys using Google Workspace SSO. Requires setting PROXY_BASE_URL in .env
    PROXY_BASE_URL should be the your deployed proxy endpoint, e.g. PROXY_BASE_URL="https://litellm-production-7002.up.railway.app/"
    Example:
    """
    from litellm.proxy.proxy_server import (
        premium_user,
        user_custom_ui_sso_sign_in_handler,
    )

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
    ui_username = os.getenv("UI_USERNAME")

    # get url from request - always use regular callback, but set state for CLI
    redirect_url = SSOAuthenticationHandler.get_redirect_url_for_sso(
        request=request,
        sso_callback_route="sso/callback",
    )
    
    # Store CLI key in state for OAuth flow
    cli_state: Optional[str] = SSOAuthenticationHandler._get_cli_state(
        source=source,
        key=key,
    )

    # check if user defined a custom auth sso sign in handler, if yes, use it
    if user_custom_ui_sso_sign_in_handler is not None:
        try:
            from litellm_enterprise.proxy.auth.custom_sso_handler import (
                EnterpriseCustomSSOHandler,
            )
            return await EnterpriseCustomSSOHandler.handle_custom_ui_sso_sign_in(
                request=request,
            )
        except ImportError:
            raise ValueError("Enterprise features are not available. Custom UI SSO sign-in requires LiteLLM Enterprise.")

    # Check if we should use SSO handler
    if (
        SSOAuthenticationHandler.should_use_sso_handler(
            microsoft_client_id=microsoft_client_id,
            google_client_id=google_client_id,
            generic_client_id=generic_client_id,
        )
        is True
    ):
        verbose_proxy_logger.info(f"Redirecting to SSO login for {redirect_url}")
        return await SSOAuthenticationHandler.get_sso_login_redirect(
            redirect_url=redirect_url,
            microsoft_client_id=microsoft_client_id,
            google_client_id=google_client_id,
            generic_client_id=generic_client_id,
            state=cli_state,
        )
    elif ui_username is not None:
        # No Google, Microsoft SSO
        # Use UI Credentials set in .env
        from fastapi.responses import HTMLResponse

        return HTMLResponse(content=html_form, status_code=200)
    else:
        from fastapi.responses import HTMLResponse

        return HTMLResponse(content=html_form, status_code=200)


def generic_response_convertor(
    response,
    jwt_handler: JWTHandler,
    sso_jwt_handler: Optional[JWTHandler] = None,
):
    generic_user_id_attribute_name = os.getenv(
        "GENERIC_USER_ID_ATTRIBUTE", "preferred_username"
    )
    generic_user_display_name_attribute_name = os.getenv(
        "GENERIC_USER_DISPLAY_NAME_ATTRIBUTE", "sub"
    )
    generic_user_email_attribute_name = os.getenv(
        "GENERIC_USER_EMAIL_ATTRIBUTE", "email"
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
        f" generic_user_id_attribute_name: {generic_user_id_attribute_name}\n generic_user_email_attribute_name: {generic_user_email_attribute_name}"
    )

    all_teams = []
    if sso_jwt_handler is not None:
        team_ids = sso_jwt_handler.get_team_ids_from_jwt(cast(dict, response))
        all_teams.extend(team_ids)

    team_ids = jwt_handler.get_team_ids_from_jwt(cast(dict, response))
    all_teams.extend(team_ids)
    return CustomOpenID(
        id=response.get(generic_user_id_attribute_name),
        display_name=response.get(generic_user_display_name_attribute_name),
        email=response.get(generic_user_email_attribute_name),
        first_name=response.get(generic_user_first_name_attribute_name),
        last_name=response.get(generic_user_last_name_attribute_name),
        provider=response.get(generic_provider_attribute_name),
        team_ids=all_teams,
    )


async def get_generic_sso_response(
    request: Request,
    jwt_handler: JWTHandler,
    sso_jwt_handler: Optional[
        JWTHandler
    ],  # sso specific jwt handler - used for restricted sso group access control
    generic_client_id: str,
    redirect_url: str,
) -> Tuple[Union[OpenID, dict], Optional[dict]]:  # return received response
    # make generic sso provider
    from fastapi_sso.sso.base import DiscoveryDocument
    from fastapi_sso.sso.generic import create_provider

    received_response: Optional[dict] = None
    generic_client_secret = os.getenv("GENERIC_CLIENT_SECRET", None)
    generic_scope = os.getenv("GENERIC_SCOPE", "openid email profile").split(" ")
    generic_authorization_endpoint = os.getenv("GENERIC_AUTHORIZATION_ENDPOINT", None)
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

    discovery = DiscoveryDocument(
        authorization_endpoint=generic_authorization_endpoint,
        token_endpoint=generic_token_endpoint,
        userinfo_endpoint=generic_userinfo_endpoint,
    )

    def response_convertor(response, client):
        nonlocal received_response  # return for user debugging
        received_response = response
        return generic_response_convertor(
            response=response,
            jwt_handler=jwt_handler,
            sso_jwt_handler=sso_jwt_handler,
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
    additional_generic_sso_headers = os.getenv(
        "GENERIC_SSO_HEADERS", None
    )  # Comma-separated list of headers to add to the request - e.g. Authorization=Bearer <token>, Content-Type=application/json, etc.
    additional_generic_sso_headers_dict = {}
    if additional_generic_sso_headers is not None:
        additional_generic_sso_headers_split = additional_generic_sso_headers.split(",")
        for header in additional_generic_sso_headers_split:
            header = header.strip()
            if header:
                key, value = header.split("=")
                additional_generic_sso_headers_dict[key] = value

    try:
        result = await generic_sso.verify_and_process(
            request,
            params={"include_client_id": generic_include_client_id},
            headers=additional_generic_sso_headers_dict,
        )
    except Exception as e:
        verbose_proxy_logger.exception(
            f"Error verifying and processing generic SSO: {e}. Passed in headers: {additional_generic_sso_headers_dict}"
        )
        raise e
    verbose_proxy_logger.debug("generic result: %s", result)
    return result or {}, received_response


async def create_team_member_add_task(team_id, user_info):
    """Create a task for adding a member to a team."""
    try:
        member = Member(user_id=user_info.user_id, role="user")
        team_member_add_request = TeamMemberAddRequest(
            member=member,
            team_id=team_id,
        )
        return await team_member_add(
            data=team_member_add_request,
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )
    except Exception as e:
        verbose_proxy_logger.debug(
            f"[Non-Blocking] Error trying to add sso user to db: {e}"
        )


async def add_missing_team_member(
    user_info: Union[NewUserResponse, LiteLLM_UserTable], sso_teams: List[str]
):
    """
    - Get missing teams (diff b/w user_info.team_ids and sso_teams)
    - Add missing user to missing teams
    """
    if user_info.teams is None:
        return
    missing_teams = set(sso_teams) - set(user_info.teams)
    missing_teams_list = list(missing_teams)
    tasks = []
    tasks = [
        create_team_member_add_task(team_id, user_info)
        for team_id in missing_teams_list
    ]

    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        verbose_proxy_logger.debug(
            f"[Non-Blocking] Error trying to add sso user to db: {e}"
        )


def get_disabled_non_admin_personal_key_creation():
    key_generation_settings = litellm.key_generation_settings
    if key_generation_settings is None:
        return False
    personal_key_generation = (
        key_generation_settings.get("personal_key_generation") or {}
    )
    allowed_user_roles = personal_key_generation.get("allowed_user_roles") or []
    return bool("proxy_admin" in allowed_user_roles)


async def get_existing_user_info_from_db(
    user_id: Optional[str],
    user_email: Optional[str],
    prisma_client: PrismaClient,
    user_api_key_cache: DualCache,
    proxy_logging_obj: ProxyLogging,
) -> Optional[LiteLLM_UserTable]:
    try:
        user_info = await get_user_object(
            user_id=user_id,
            user_email=user_email,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            user_id_upsert=False,
            parent_otel_span=None,
            proxy_logging_obj=proxy_logging_obj,
            sso_user_id=user_id,
        )
    except Exception as e:
        verbose_proxy_logger.debug(f"Error getting user object: {e}")
        user_info = None

    return user_info


async def get_user_info_from_db(
    result: Union[CustomOpenID, OpenID, dict],
    prisma_client: PrismaClient,
    user_api_key_cache: DualCache,
    proxy_logging_obj: ProxyLogging,
    user_email: Optional[str],
    user_defined_values: Optional[SSOUserDefinedValues],
    alternate_user_id: Optional[str] = None,
) -> Optional[Union[LiteLLM_UserTable, NewUserResponse]]:
    try:
        potential_user_ids = []
        if alternate_user_id is not None:
            potential_user_ids.append(alternate_user_id)
        if not isinstance(result, dict):
            _id = getattr(result, "id", None)
            if _id is not None and isinstance(_id, str):
                potential_user_ids.append(_id)
        else:
            _id = result.get("id", None)
            if _id is not None and isinstance(_id, str):
                potential_user_ids.append(_id)

        user_email = (
            getattr(result, "email", None)
            if not isinstance(result, dict)
            else result.get("email", None)
        )

        user_info: Optional[Union[LiteLLM_UserTable, NewUserResponse]] = None

        for user_id in potential_user_ids:
            user_info = await get_existing_user_info_from_db(
                user_id=user_id,
                user_email=user_email,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )
            if user_info is not None:
                break

        verbose_proxy_logger.debug(
            f"user_info: {user_info}; litellm.default_internal_user_params: {litellm.default_internal_user_params}"
        )

        # Upsert SSO User to LiteLLM DB

        if user_info is None:
            user_info = await SSOAuthenticationHandler.upsert_sso_user(
                result=result,
                user_info=user_info,
                user_email=user_email,
                user_defined_values=user_defined_values,
                prisma_client=prisma_client,
            )

        await SSOAuthenticationHandler.add_user_to_teams_from_sso_response(
            result=result,
            user_info=user_info,
        )

        return user_info
    except Exception as e:
        verbose_proxy_logger.exception(
            f"[Non-Blocking] Error trying to add sso user to db: {e}"
        )

    return None


def apply_user_info_values_to_sso_user_defined_values(
    user_info: Optional[Union[LiteLLM_UserTable, NewUserResponse]],
    user_defined_values: Optional[SSOUserDefinedValues],
) -> Optional[SSOUserDefinedValues]:
    if user_defined_values is None:
        return None
    if user_info is not None and user_info.user_id is not None:
        user_defined_values["user_id"] = user_info.user_id

    if user_info is None or user_info.user_role is None:
        user_defined_values["user_role"] = LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
    else:
        user_defined_values["user_role"] = user_info.user_role

    return user_defined_values


async def check_and_update_if_proxy_admin_id(
    user_role: str, user_id: str, prisma_client: Optional[PrismaClient]
):
    """
    - Check if user role in DB is admin
    - If not, update user role in DB to admin role
    """
    proxy_admin_id = os.getenv("PROXY_ADMIN_ID")
    if proxy_admin_id is not None and proxy_admin_id == user_id:
        if user_role and user_role == LitellmUserRoles.PROXY_ADMIN.value:
            return user_role

        if prisma_client:
            await prisma_client.db.litellm_usertable.update(
                where={"user_id": user_id},
                data={"user_role": LitellmUserRoles.PROXY_ADMIN.value},
            )

        user_role = LitellmUserRoles.PROXY_ADMIN.value

    return user_role


@router.get("/sso/callback", tags=["experimental"], include_in_schema=False)
async def auth_callback(request: Request, state: Optional[str] = None):  # noqa: PLR0915
    """Verify login"""
    verbose_proxy_logger.info(f"Starting SSO callback with state: {state}")
    
    # Check if this is a CLI login (state starts with our CLI prefix)
    from litellm.constants import LITELLM_CLI_SESSION_TOKEN_PREFIX
    if state and state.startswith(f"{LITELLM_CLI_SESSION_TOKEN_PREFIX}:"):
        # Extract the key ID from the state
        key_id = state.split(":", 1)[1]
        verbose_proxy_logger.info(f"CLI SSO callback detected for key: {key_id}")
        return await cli_sso_callback(request, key=key_id)
    
    from litellm.proxy._types import LiteLLM_JWTAuth
    from litellm.proxy.auth.handle_jwt import JWTHandler
    from litellm.proxy.proxy_server import (
        general_settings,
        jwt_handler,
        master_key,
        prisma_client,
        user_api_key_cache,
    )

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )

    sso_jwt_handler: Optional[JWTHandler] = None
    ui_access_mode = general_settings.get("ui_access_mode", None)
    if ui_access_mode is not None and isinstance(ui_access_mode, dict):
        sso_jwt_handler = JWTHandler()
        sso_jwt_handler.update_environment(
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            litellm_jwtauth=LiteLLM_JWTAuth(
                team_ids_jwt_field=general_settings.get("ui_access_mode", {}).get(
                    "sso_group_jwt_field", None
                ),
            ),
            leeway=0,
        )

    microsoft_client_id = os.getenv("MICROSOFT_CLIENT_ID", None)
    google_client_id = os.getenv("GOOGLE_CLIENT_ID", None)
    generic_client_id = os.getenv("GENERIC_CLIENT_ID", None)
    received_response: Optional[dict] = None
    # get url from request
    if master_key is None:
        raise ProxyException(
            message="Master Key not set for Proxy. Please set Master Key to use Admin UI. Set `LITELLM_MASTER_KEY` in .env or set general_settings:master_key in config.yaml.  https://docs.litellm.ai/docs/proxy/virtual_keys. If set, use `--detailed_debug` to debug issue.",
            type=ProxyErrorTypes.auth_error,
            param="master_key",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    redirect_url = SSOAuthenticationHandler.get_redirect_url_for_sso(
        request=request, sso_callback_route="sso/callback"
    )

    verbose_proxy_logger.info(f"Redirecting to {redirect_url}")
    result = None
    if google_client_id is not None:
        result = await GoogleSSOHandler.get_google_callback_response(
            request=request,
            google_client_id=google_client_id,
            redirect_url=redirect_url,
        )
    elif microsoft_client_id is not None:
        result = await MicrosoftSSOHandler.get_microsoft_callback_response(
            request=request,
            microsoft_client_id=microsoft_client_id,
            redirect_url=redirect_url,
        )
    elif generic_client_id is not None:
        result, received_response = await get_generic_sso_response(
            request=request,
            jwt_handler=jwt_handler,
            generic_client_id=generic_client_id,
            redirect_url=redirect_url,
            sso_jwt_handler=sso_jwt_handler,
        )

    if result is None:
        raise HTTPException(
            status_code=401,
            detail="Result not returned by SSO provider.",
        )
    
    return await SSOAuthenticationHandler.get_redirect_response_from_openid(
        result=result,
        request=request,
        received_response=received_response,
        generic_client_id=generic_client_id,
        ui_access_mode=ui_access_mode,
    )


    

async def cli_sso_callback(request: Request, key: Optional[str] = None):
    """CLI SSO callback - generates the key with pre-specified ID"""
    verbose_proxy_logger.info(f"CLI SSO callback for key: {key}")
    
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        generate_key_helper_fn,
    )
    from litellm.proxy.proxy_server import prisma_client
    
    if not key or not key.startswith('sk-'):
        raise HTTPException(
            status_code=400,
            detail="Invalid key parameter. Must be a valid key ID starting with 'sk-'"
        )
    
    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )
    
    # Generate a simple key for CLI usage with the pre-specified key ID
    try:
        await generate_key_helper_fn(
            request_type="key",
            duration="24hr",
            key_max_budget=litellm.max_ui_session_budget,
            aliases={},
            config={},
            spend=0,
            team_id="litellm-cli",
            table_name="key",
            token=key,  # Use the pre-specified key ID
        )
        
        verbose_proxy_logger.info(f"Generated CLI key: {key}")
        
        # Return success page
        from fastapi.responses import HTMLResponse

        from litellm.proxy.common_utils.html_forms.cli_sso_success import (
            render_cli_sso_success_page,
        )
        
        html_content = render_cli_sso_success_page()
        return HTMLResponse(content=html_content, status_code=200)
        
    except Exception as e:
        verbose_proxy_logger.error(f"Error generating CLI key: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate key: {str(e)}"
        )


@router.get("/sso/cli/poll/{key_id}", tags=["experimental"], include_in_schema=False)
async def cli_poll_key(key_id: str):
    """CLI polling endpoint - checks if key exists in DB"""
    from litellm.proxy.proxy_server import prisma_client
    
    if not key_id.startswith('sk-'):
        raise HTTPException(
            status_code=400,
            detail="Invalid key ID format"
        )
    
    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )
    
    try:
        # Check if key exists in database
        from litellm.proxy.utils import hash_token
        hashed_token = hash_token(key_id)
        
        key_obj = await prisma_client.db.litellm_verificationtoken.find_unique(
            where={"token": hashed_token}
        )
        
        if key_obj:
            verbose_proxy_logger.info(f"CLI key found: {key_id}")
            return {"status": "ready", "key": key_id}
        else:
            return {"status": "pending"}
            
    except Exception as e:
        verbose_proxy_logger.error(f"Error polling for CLI key: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error checking key status: {str(e)}"
        )


async def insert_sso_user(
    result_openid: Optional[Union[OpenID, dict]],
    user_defined_values: Optional[SSOUserDefinedValues] = None,
) -> NewUserResponse:
    """
    Helper function to create a New User in LiteLLM DB after a successful SSO login

    Args:
        result_openid (OpenID): User information in OpenID format if the login was successful.
        user_defined_values (Optional[SSOUserDefinedValues], optional): LiteLLM SSOValues / fields that were read

    Returns:
        Tuple[str, str]: User ID and User Role
    """
    verbose_proxy_logger.debug(
        f"Inserting SSO user into DB. User values: {user_defined_values}"
    )
    if result_openid is None:
        raise ValueError("result_openid is None")
    if isinstance(result_openid, dict):
        result_openid = OpenID(**result_openid)

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
        sso_user_id=user_defined_values["user_id"],
        auto_create_key=False,
    )

    if result_openid:
        new_user_request.metadata = {"auth_provider": result_openid.provider}

    response = await new_user(
        data=new_user_request,
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    )

    return response


@router.get(
    "/sso/get/ui_settings",
    tags=["experimental"],
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
)
async def get_ui_settings(request: Request):
    from litellm.proxy.proxy_server import general_settings, proxy_state

    _proxy_base_url = os.getenv("PROXY_BASE_URL", None)
    _logout_url = os.getenv("PROXY_LOGOUT_URL", None)
    _is_sso_enabled = _has_user_setup_sso()
    disable_expensive_db_queries = (
        proxy_state.get_proxy_state_variable("spend_logs_row_count")
        > MAX_SPENDLOG_ROWS_TO_QUERY
    )
    default_team_disabled = general_settings.get("default_team_disabled", False)
    if "PROXY_DEFAULT_TEAM_DISABLED" in os.environ:
        if os.environ["PROXY_DEFAULT_TEAM_DISABLED"].lower() == "true":
            default_team_disabled = True

    return {
        "PROXY_BASE_URL": _proxy_base_url,
        "PROXY_LOGOUT_URL": _logout_url,
        "DEFAULT_TEAM_DISABLED": default_team_disabled,
        "SSO_ENABLED": _is_sso_enabled,
        "NUM_SPEND_LOGS_ROWS": proxy_state.get_proxy_state_variable(
            "spend_logs_row_count"
        ),
        "DISABLE_EXPENSIVE_DB_QUERIES": disable_expensive_db_queries,
    }


class SSOAuthenticationHandler:
    """
    Handler for SSO Authentication across all SSO providers
    """
    @staticmethod
    async def get_sso_login_redirect(
        redirect_url: str,
        google_client_id: Optional[str] = None,
        microsoft_client_id: Optional[str] = None,
        generic_client_id: Optional[str] = None,
        state: Optional[str] = None,
    ) -> Optional[RedirectResponse]:
        """
        Step 1. Call Get Login Redirect for the SSO provider. Send the redirect response to `redirect_url`

        Args:
            redirect_url (str): The URL to redirect the user to after login
            google_client_id (Optional[str], optional): The Google Client ID. Defaults to None.
            microsoft_client_id (Optional[str], optional): The Microsoft Client ID. Defaults to None.
            generic_client_id (Optional[str], optional): The Generic Client ID. Defaults to None.

        Returns:
            RedirectResponse: The redirect response from the SSO provider
        """
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
                return await google_sso.get_login_redirect(state=state)
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
                return await microsoft_sso.get_login_redirect(state=state)
        elif generic_client_id is not None:
            from fastapi_sso.sso.base import DiscoveryDocument
            from fastapi_sso.sso.generic import create_provider

            generic_client_secret = os.getenv("GENERIC_CLIENT_SECRET", None)
            generic_scope = os.getenv("GENERIC_SCOPE", "openid email profile").split(
                " "
            )
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
        raise ValueError(
            "Unknown SSO provider. Please setup SSO with client IDs https://docs.litellm.ai/docs/proxy/admin_ui_sso"
        )

    @staticmethod
    def should_use_sso_handler(
        google_client_id: Optional[str] = None,
        microsoft_client_id: Optional[str] = None,
        generic_client_id: Optional[str] = None,
    ) -> bool:
        if (
            google_client_id is not None
            or microsoft_client_id is not None
            or generic_client_id is not None
        ):
            return True
        return False

    @staticmethod
    def get_redirect_url_for_sso(
        request: Request,
        sso_callback_route: str,
    ) -> str:
        """
        Get the redirect URL for SSO
        """
        from litellm.proxy.utils import get_custom_url

        redirect_url = get_custom_url(request_base_url=str(request.base_url))
        if redirect_url.endswith("/"):
            redirect_url += sso_callback_route
        else:
            redirect_url += "/" + sso_callback_route
        return redirect_url

    @staticmethod
    async def upsert_sso_user(
        result: Optional[Union[CustomOpenID, OpenID, dict]],
        user_info: Optional[Union[NewUserResponse, LiteLLM_UserTable]],
        user_email: Optional[str],
        user_defined_values: Optional[SSOUserDefinedValues],
        prisma_client: PrismaClient,
    ):
        """
        Connects the SSO Users to the User Table in LiteLLM DB

        - If user on LiteLLM DB, update the user_email with the SSO user_email
        - If user not on LiteLLM DB, insert the user into LiteLLM DB
        """
        try:
            if user_info is not None:
                user_id = user_info.user_id
                await prisma_client.db.litellm_usertable.update_many(
                    where={"user_id": user_id}, data={"user_email": user_email}
                )
            else:
                verbose_proxy_logger.info(
                    "user not in DB, inserting user into LiteLLM DB"
                )
                # user not in DB, insert User into LiteLLM DB
                user_info = await insert_sso_user(
                    result_openid=result,
                    user_defined_values=user_defined_values,
                )
            return user_info
        except Exception as e:
            verbose_proxy_logger.error(f"Error upserting SSO user into LiteLLM DB: {e}")
            return user_info

    @staticmethod
    async def add_user_to_teams_from_sso_response(
        result: Optional[Union[CustomOpenID, OpenID, dict]],
        user_info: Optional[Union[NewUserResponse, LiteLLM_UserTable]],
    ):
        """
        Adds the user as a team member to the teams specified in the SSO responses `team_ids` field


        The `team_ids` field is populated by litellm after processing the SSO response
        """
        if user_info is None:
            verbose_proxy_logger.debug(
                "User not found in LiteLLM DB, skipping team member addition"
            )
            return
        sso_teams = getattr(result, "team_ids", [])
        await add_missing_team_member(user_info=user_info, sso_teams=sso_teams)

    @staticmethod
    def verify_user_in_restricted_sso_group(
        general_settings: Dict,
        result: Optional[Union[CustomOpenID, OpenID, dict]],
        received_response: Optional[dict],
    ) -> Literal[True]:
        """
        when ui_access_mode.type == "restricted_sso_group":

        - result.team_ids should contain the restricted_sso_group
        - if not, raise a ProxyException
        - if so, return True
        - if result.team_ids is None, return False
        - if result.team_ids is an empty list, return False
        - if result.team_ids is a list, return True if the restricted_sso_group is in the list, otherwise return False
        """

        ui_access_mode = cast(
            Optional[Union[Dict, str]], general_settings.get("ui_access_mode")
        )

        if ui_access_mode is None:
            return True
        if isinstance(ui_access_mode, str):
            return True
        team_ids = getattr(result, "team_ids", [])

        if ui_access_mode.get("type") == "restricted_sso_group":
            restricted_sso_group = ui_access_mode.get("restricted_sso_group")
            if restricted_sso_group not in team_ids:
                raise ProxyException(
                    message=f"User is not in the restricted SSO group: {restricted_sso_group}. User groups: {team_ids}. Received SSO response: {received_response}",
                    type=ProxyErrorTypes.auth_error,
                    param="restricted_sso_group",
                    code=status.HTTP_403_FORBIDDEN,
                )
        return True

    @staticmethod
    async def create_litellm_team_from_sso_group(
        litellm_team_id: str,
        litellm_team_name: Optional[str] = None,
    ):
        """
        Creates a Litellm Team from a SSO Group ID

        Your SSO provider might have groups that should be created on LiteLLM

        Use this helper to create a Litellm Team from a SSO Group ID

        Args:
            litellm_team_id (str): The ID of the Litellm Team
            litellm_team_name (Optional[str]): The name of the Litellm Team
        """
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise ProxyException(
                message="Prisma client not found. Set it in the proxy_server.py file",
                type=ProxyErrorTypes.auth_error,
                param="prisma_client",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        try:
            team_obj = await prisma_client.db.litellm_teamtable.find_first(
                where={"team_id": litellm_team_id}
            )
            verbose_proxy_logger.debug(f"Team object: {team_obj}")

            # only create a new team if it doesn't exist
            if team_obj:
                verbose_proxy_logger.debug(
                    f"Team already exists: {litellm_team_id} - {litellm_team_name}"
                )
                return

            team_request: NewTeamRequest = NewTeamRequest(
                team_id=litellm_team_id,
                team_alias=litellm_team_name,
            )
            if litellm.default_team_params:
                team_request = SSOAuthenticationHandler._cast_and_deepcopy_litellm_default_team_params(
                    default_team_params=litellm.default_team_params,
                    litellm_team_id=litellm_team_id,
                    litellm_team_name=litellm_team_name,
                    team_request=team_request,
                )

            await new_team(
                data=team_request,
                # params used for Audit Logging
                http_request=Request(scope={"type": "http", "method": "POST"}),
                user_api_key_dict=UserAPIKeyAuth(
                    token="",
                    key_alias=f"litellm.{MicrosoftSSOHandler.__name__}",
                ),
            )
        except Exception as e:
            verbose_proxy_logger.exception(f"Error creating Litellm Team: {e}")

    @staticmethod
    def _cast_and_deepcopy_litellm_default_team_params(
        default_team_params: Union[DefaultTeamSSOParams, Dict],
        team_request: NewTeamRequest,
        litellm_team_id: str,
        litellm_team_name: Optional[str] = None,
    ) -> NewTeamRequest:
        """
        Casts and deepcopies the litellm.default_team_params to a NewTeamRequest object

        - Ensures we create a new DefaultTeamSSOParams object
        - Handle the case where litellm.default_team_params is a dict or a DefaultTeamSSOParams object
        - Adds the litellm_team_id and litellm_team_name to the DefaultTeamSSOParams object
        """
        if isinstance(default_team_params, dict):
            _team_request = deepcopy(default_team_params)
            _team_request["team_id"] = litellm_team_id
            _team_request["team_alias"] = litellm_team_name
            team_request = NewTeamRequest(**_team_request)
        elif isinstance(litellm.default_team_params, DefaultTeamSSOParams):
            _default_team_params = deepcopy(litellm.default_team_params)
            _new_team_request = team_request.model_dump()
            _new_team_request.update(_default_team_params)
            team_request = NewTeamRequest(**_new_team_request)
        return team_request
    

    @staticmethod
    def _get_cli_state(source: Optional[str], key: Optional[str]) -> Optional[str]:
        """
        Checks the request 'source' if a cli state token was passed in

        This is used to authenticate through the CLI login flow
        """
        from litellm.constants import (
            LITELLM_CLI_SESSION_TOKEN_PREFIX,
            LITELLM_CLI_SOURCE_IDENTIFIER,
        )
        return f"{LITELLM_CLI_SESSION_TOKEN_PREFIX}:{key}" if source == LITELLM_CLI_SOURCE_IDENTIFIER and key else None
    



    @staticmethod
    async def get_redirect_response_from_openid( # noqa: PLR0915
        result: Union[OpenID, dict, CustomOpenID],
        request: Request,
        received_response: Optional[dict] = None,
        generic_client_id: Optional[str] = None,
        ui_access_mode: Optional[Dict] = None,
    ) -> RedirectResponse:
        import jwt

        from litellm.proxy.proxy_server import (
            general_settings,
            generate_key_helper_fn,
            master_key,
            premium_user,
            proxy_logging_obj,
            user_api_key_cache,
            user_custom_sso,
        )
        from litellm.proxy.utils import get_prisma_client_or_throw
        from litellm.types.proxy.ui_sso import ReturnedUITokenObject
        prisma_client = get_prisma_client_or_throw("Prisma client is None, connect a database to your proxy")


        # User is Authe'd in - generate key for the UI to access Proxy
        verbose_proxy_logger.info(f"SSO callback result: {result}")

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
            generic_user_role_attribute_name = os.getenv(
                "GENERIC_USER_ROLE_ATTRIBUTE", "role"
            )
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
        default_ui_key_values: Dict[str, Any] = {
            "duration": "24hr",
            "key_max_budget": litellm.max_ui_session_budget,
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

        # (IF SET) Verify user is in restricted SSO group
        SSOAuthenticationHandler.verify_user_in_restricted_sso_group(
            general_settings=general_settings,
            result=result,
            received_response=received_response,
        )

        user_info = await get_user_info_from_db(
            result=result,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
            user_email=user_email,
            user_defined_values=user_defined_values,
            alternate_user_id=user_id,
        )

        user_defined_values = apply_user_info_values_to_sso_user_defined_values(
            user_info=user_info, user_defined_values=user_defined_values
        )

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

        user_role = (
            user_defined_values["user_role"]
            or LitellmUserRoles.INTERNAL_USER_VIEW_ONLY.value
        )
        if user_id and isinstance(user_id, str):
            user_role = await check_and_update_if_proxy_admin_id(
                user_role=user_role, user_id=user_id, prisma_client=prisma_client
            )

        verbose_proxy_logger.debug(
            f"user_role: {user_role}; ui_access_mode: {ui_access_mode}"
        )
        ## CHECK IF ROLE ALLOWED TO USE PROXY ##
        is_admin_only_access = check_is_admin_only_access(ui_access_mode or {})
        if is_admin_only_access:
            has_access = has_admin_ui_access(user_role)
            if not has_access:
                raise HTTPException(
                    status_code=401,
                    detail={
                        "error": f"User not allowed to access proxy. User role={user_role}, proxy mode={ui_access_mode}"
                    },
                )

        disabled_non_admin_personal_key_creation = (
            get_disabled_non_admin_personal_key_creation()
        )
        litellm_dashboard_ui = get_custom_url(
            request_base_url=str(request.base_url), route="ui/"
        )

        if get_secret_bool("EXPERIMENTAL_UI_LOGIN"):
            _user_info: Optional[LiteLLM_UserTable] = None
            if (
                user_defined_values is not None
                and user_defined_values["user_id"] is not None
            ):
                _user_info = LiteLLM_UserTable(
                    user_id=user_defined_values["user_id"],
                    user_role=user_defined_values["user_role"] or user_role,
                    models=[],
                    max_budget=litellm.max_ui_session_budget,
                )
            if _user_info is None:
                raise HTTPException(
                    status_code=401,
                    detail={
                        "error": "User Information is required for experimental UI login"
                    },
                )

            key = ExperimentalUIJWTToken.get_experimental_ui_login_jwt_auth_token(
                _user_info
            )

        returned_ui_token_object = ReturnedUITokenObject(
            user_id=cast(str, user_id),
            key=key,
            user_email=user_email,
            user_role=user_role,
            login_method="sso",
            premium_user=premium_user,
            auth_header_name=general_settings.get(
                "litellm_key_header_name", "Authorization"
            ),
            disabled_non_admin_personal_key_creation=disabled_non_admin_personal_key_creation,
            server_root_path=get_server_root_path(),
        )

        jwt_token = jwt.encode(
            cast(dict, returned_ui_token_object),
            master_key or "",
            algorithm="HS256",
        )
        verbose_proxy_logger.info(f"user_id: {user_id}; jwt_token: {jwt_token}")
        if user_id is not None and isinstance(user_id, str):
            litellm_dashboard_ui += "?login=success"
        verbose_proxy_logger.info(f"Redirecting to {litellm_dashboard_ui}")
        redirect_response = RedirectResponse(url=litellm_dashboard_ui, status_code=303)
        redirect_response.set_cookie(key="token", value=jwt_token)
        return redirect_response
        
class MicrosoftSSOHandler:
    """
    Handles Microsoft SSO callback response and returns a CustomOpenID object
    """

    graph_api_base_url = "https://graph.microsoft.com/v1.0"
    graph_api_user_groups_endpoint = f"{graph_api_base_url}/me/memberOf"

    """
    Constants
    """
    MAX_GRAPH_API_PAGES = 200

    # used for debugging to show the user groups litellm found from Graph API
    GRAPH_API_RESPONSE_KEY = "graph_api_user_groups"

    @staticmethod
    async def get_microsoft_callback_response(
        request: Request,
        microsoft_client_id: str,
        redirect_url: str,
        return_raw_sso_response: bool = False,
    ) -> Union[CustomOpenID, OpenID, dict]:
        """
        Get the Microsoft SSO callback response

        Args:
            return_raw_sso_response: If True, return the raw SSO response
        """
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
        original_msft_result = (
            await microsoft_sso.verify_and_process(
                request=request,
                convert_response=False,  # type: ignore
            )
            or {}
        )

        user_team_ids = await MicrosoftSSOHandler.get_user_groups_from_graph_api(
            access_token=microsoft_sso.access_token
        )

        # if user is trying to get the raw sso response for debugging, return the raw sso response
        if return_raw_sso_response:
            original_msft_result[MicrosoftSSOHandler.GRAPH_API_RESPONSE_KEY] = (
                user_team_ids
            )
            return original_msft_result or {}

        result = MicrosoftSSOHandler.openid_from_response(
            response=original_msft_result,
            team_ids=user_team_ids,
        )
        return result

    @staticmethod
    def openid_from_response(
        response: Optional[dict], team_ids: List[str]
    ) -> CustomOpenID:
        response = response or {}
        verbose_proxy_logger.debug(f"Microsoft SSO Callback Response: {response}")
        openid_response = CustomOpenID(
            email=response.get("userPrincipalName") or response.get("mail"),
            display_name=response.get("displayName"),
            provider="microsoft",
            id=response.get("id"),
            first_name=response.get("givenName"),
            last_name=response.get("surname"),
            team_ids=team_ids,
        )
        verbose_proxy_logger.debug(f"Microsoft SSO OpenID Response: {openid_response}")
        return openid_response

    @staticmethod
    async def get_user_groups_from_graph_api(
        access_token: Optional[str] = None,
    ) -> List[str]:
        """
        Returns a list of `team_ids` the user belongs to from the Microsoft Graph API

        Args:
            access_token (Optional[str]): Microsoft Graph API access token

        Returns:
            List[str]: List of group IDs the user belongs to
        """
        try:
            async_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.SSO_HANDLER
            )

            # Handle MSFT Enterprise Application Groups
            service_principal_id = os.getenv("MICROSOFT_SERVICE_PRINCIPAL_ID", None)
            service_principal_group_ids: Optional[List[str]] = []
            service_principal_teams: Optional[List[MicrosoftServicePrincipalTeam]] = []
            if service_principal_id:
                (
                    service_principal_group_ids,
                    service_principal_teams,
                ) = await MicrosoftSSOHandler.get_group_ids_from_service_principal(
                    service_principal_id=service_principal_id,
                    async_client=async_client,
                    access_token=access_token,
                )
                verbose_proxy_logger.debug(
                    f"Service principal group IDs: {service_principal_group_ids}"
                )
                if len(service_principal_group_ids) > 0:
                    await MicrosoftSSOHandler.create_litellm_teams_from_service_principal_team_ids(
                        service_principal_teams=service_principal_teams,
                    )

            # Fetch user membership from Microsoft Graph API
            all_group_ids = []
            next_link: Optional[str] = (
                MicrosoftSSOHandler.graph_api_user_groups_endpoint
            )
            auth_headers = {"Authorization": f"Bearer {access_token}"}
            page_count = 0

            while (
                next_link is not None
                and page_count < MicrosoftSSOHandler.MAX_GRAPH_API_PAGES
            ):
                group_ids, next_link = await MicrosoftSSOHandler.fetch_and_parse_groups(
                    url=next_link, headers=auth_headers, async_client=async_client
                )
                all_group_ids.extend(group_ids)
                page_count += 1

            if (
                next_link is not None
                and page_count >= MicrosoftSSOHandler.MAX_GRAPH_API_PAGES
            ):
                verbose_proxy_logger.warning(
                    f"Reached maximum page limit of {MicrosoftSSOHandler.MAX_GRAPH_API_PAGES}. Some groups may not be included."
                )

            # If service_principal_group_ids is not empty, only return group_ids that are in both all_group_ids and service_principal_group_ids
            if service_principal_group_ids and len(service_principal_group_ids) > 0:
                all_group_ids = [
                    group_id
                    for group_id in all_group_ids
                    if group_id in service_principal_group_ids
                ]

            return all_group_ids

        except Exception as e:
            verbose_proxy_logger.error(
                f"Error getting user groups from Microsoft Graph API: {e}"
            )
            return []

    @staticmethod
    async def fetch_and_parse_groups(
        url: str, headers: dict, async_client: AsyncHTTPHandler
    ) -> Tuple[List[str], Optional[str]]:
        """Helper function to fetch and parse group data from a URL"""
        response = await async_client.get(url, headers=headers)
        response_json = response.json()
        response_typed = await MicrosoftSSOHandler._cast_graph_api_response_dict(
            response=response_json
        )
        group_ids = MicrosoftSSOHandler._get_group_ids_from_graph_api_response(
            response=response_typed
        )
        return group_ids, response_typed.get("odata_nextLink")

    @staticmethod
    def _get_group_ids_from_graph_api_response(
        response: MicrosoftGraphAPIUserGroupResponse,
    ) -> List[str]:
        group_ids = []
        for _object in response.get("value", []) or []:
            _group_id = _object.get("id")
            if _group_id is not None:
                group_ids.append(_group_id)
        return group_ids

    @staticmethod
    async def _cast_graph_api_response_dict(
        response: dict,
    ) -> MicrosoftGraphAPIUserGroupResponse:
        directory_objects: List[MicrosoftGraphAPIUserGroupDirectoryObject] = []
        for _object in response.get("value", []):
            directory_objects.append(
                MicrosoftGraphAPIUserGroupDirectoryObject(
                    odata_type=_object.get("@odata.type"),
                    id=_object.get("id"),
                    deletedDateTime=_object.get("deletedDateTime"),
                    description=_object.get("description"),
                    displayName=_object.get("displayName"),
                    roleTemplateId=_object.get("roleTemplateId"),
                )
            )
        return MicrosoftGraphAPIUserGroupResponse(
            odata_context=response.get("@odata.context"),
            odata_nextLink=response.get("@odata.nextLink"),
            value=directory_objects,
        )

    @staticmethod
    async def get_group_ids_from_service_principal(
        service_principal_id: str,
        async_client: AsyncHTTPHandler,
        access_token: Optional[str] = None,
    ) -> Tuple[List[str], List[MicrosoftServicePrincipalTeam]]:
        """
        Gets the groups belonging to the Service Principal Application

        Service Principal Id is an `Enterprise Application` in Azure AD

        Users use Enterprise Applications to manage Groups and Users on Microsoft Entra ID
        """
        base_url = "https://graph.microsoft.com/v1.0"
        # Endpoint to get app role assignments for the given service principal
        endpoint = f"/servicePrincipals/{service_principal_id}/appRoleAssignedTo"
        url = base_url + endpoint

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        response = await async_client.get(url, headers=headers)
        response_json = response.json()
        verbose_proxy_logger.debug(
            f"Response from service principal app role assigned to: {response_json}"
        )
        group_ids: List[str] = []
        service_principal_teams: List[MicrosoftServicePrincipalTeam] = []

        for _object in response_json.get("value", []):
            if _object.get("principalType") == "Group":
                # Append the group ID to the list
                group_ids.append(_object.get("principalId"))
                # Append the service principal team to the list
                service_principal_teams.append(
                    MicrosoftServicePrincipalTeam(
                        principalDisplayName=_object.get("principalDisplayName"),
                        principalId=_object.get("principalId"),
                    )
                )

        return group_ids, service_principal_teams

    @staticmethod
    async def create_litellm_teams_from_service_principal_team_ids(
        service_principal_teams: List[MicrosoftServicePrincipalTeam],
    ):
        """
        Creates Litellm Teams from the Service Principal Group IDs

        When a user sets a `SERVICE_PRINCIPAL_ID` in the env, litellm will fetch groups under that service principal and create Litellm Teams from them
        """
        verbose_proxy_logger.debug(
            f"Creating Litellm Teams from Service Principal Teams: {service_principal_teams}"
        )
        for service_principal_team in service_principal_teams:
            litellm_team_id: Optional[str] = service_principal_team.get("principalId")
            litellm_team_name: Optional[str] = service_principal_team.get(
                "principalDisplayName"
            )
            if not litellm_team_id:
                verbose_proxy_logger.debug(
                    f"Skipping team creation for {litellm_team_name} because it has no principalId"
                )
                continue

            await SSOAuthenticationHandler.create_litellm_team_from_sso_group(
                litellm_team_id=litellm_team_id,
                litellm_team_name=litellm_team_name,
            )


class GoogleSSOHandler:
    """
    Handles Google SSO callback response and returns a CustomOpenID object
    """

    @staticmethod
    async def get_google_callback_response(
        request: Request,
        google_client_id: str,
        redirect_url: str,
        return_raw_sso_response: bool = False,
    ) -> Union[OpenID, dict]:
        """
        Get the Google SSO callback response

        Args:
            return_raw_sso_response: If True, return the raw SSO response
        """
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

        # if user is trying to get the raw sso response for debugging, return the raw sso response
        if return_raw_sso_response:
            return (
                await google_sso.verify_and_process(
                    request=request,
                    convert_response=False,  # type: ignore
                )
                or {}
            )

        result = await google_sso.verify_and_process(request)
        return result or {}


@router.get("/sso/debug/login", tags=["experimental"], include_in_schema=False)
async def debug_sso_login(request: Request):
    """
    Create Proxy API Keys using Google Workspace SSO. Requires setting PROXY_BASE_URL in .env
    PROXY_BASE_URL should be the your deployed proxy endpoint, e.g. PROXY_BASE_URL="https://litellm-production-7002.up.railway.app/"
    Example:
    """
    from litellm.proxy.proxy_server import premium_user

    microsoft_client_id = os.getenv("MICROSOFT_CLIENT_ID", None)
    google_client_id = os.getenv("GOOGLE_CLIENT_ID", None)
    generic_client_id = os.getenv("GENERIC_CLIENT_ID", None)

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

    # get url from request
    redirect_url = SSOAuthenticationHandler.get_redirect_url_for_sso(
        request=request,
        sso_callback_route="sso/debug/callback",
    )

    # Check if we should use SSO handler
    if (
        SSOAuthenticationHandler.should_use_sso_handler(
            microsoft_client_id=microsoft_client_id,
            google_client_id=google_client_id,
            generic_client_id=generic_client_id,
        )
        is True
    ):
        return await SSOAuthenticationHandler.get_sso_login_redirect(
            redirect_url=redirect_url,
            microsoft_client_id=microsoft_client_id,
            google_client_id=google_client_id,
            generic_client_id=generic_client_id,
        )


@router.get("/sso/debug/callback", tags=["experimental"], include_in_schema=False)
async def debug_sso_callback(request: Request):
    """
    Returns the OpenID object returned by the SSO provider
    """
    import json

    from fastapi.responses import HTMLResponse

    from litellm.proxy._types import LiteLLM_JWTAuth
    from litellm.proxy.auth.handle_jwt import JWTHandler
    from litellm.proxy.proxy_server import (
        general_settings,
        jwt_handler,
        prisma_client,
        user_api_key_cache,
    )

    sso_jwt_handler: Optional[JWTHandler] = None
    ui_access_mode = general_settings.get("ui_access_mode", None)
    if ui_access_mode is not None and isinstance(ui_access_mode, dict):
        sso_jwt_handler = JWTHandler()
        sso_jwt_handler.update_environment(
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            litellm_jwtauth=LiteLLM_JWTAuth(
                team_ids_jwt_field=general_settings.get("ui_access_mode", {}).get(
                    "sso_group_jwt_field", None
                ),
            ),
            leeway=0,
        )

    microsoft_client_id = os.getenv("MICROSOFT_CLIENT_ID", None)
    google_client_id = os.getenv("GOOGLE_CLIENT_ID", None)
    generic_client_id = os.getenv("GENERIC_CLIENT_ID", None)

    redirect_url = os.getenv("PROXY_BASE_URL", str(request.base_url))
    if redirect_url.endswith("/"):
        redirect_url += "sso/debug/callback"
    else:
        redirect_url += "/sso/debug/callback"

    result = None
    if google_client_id is not None:
        result = await GoogleSSOHandler.get_google_callback_response(
            request=request,
            google_client_id=google_client_id,
            redirect_url=redirect_url,
            return_raw_sso_response=True,
        )
    elif microsoft_client_id is not None:
        result = await MicrosoftSSOHandler.get_microsoft_callback_response(
            request=request,
            microsoft_client_id=microsoft_client_id,
            redirect_url=redirect_url,
            return_raw_sso_response=True,
        )

    elif generic_client_id is not None:
        result, _ = await get_generic_sso_response(
            request=request,
            jwt_handler=jwt_handler,
            generic_client_id=generic_client_id,
            redirect_url=redirect_url,
            sso_jwt_handler=sso_jwt_handler,
        )

    # If result is None, return a basic error message
    if result is None:
        return HTMLResponse(
            content="<h1>SSO Authentication Failed</h1><p>No data was returned from the SSO provider.</p>",
            status_code=400,
        )

    # Convert the OpenID object to a dictionary
    if hasattr(result, "__dict__"):
        result_dict = result.__dict__
    else:
        result_dict = dict(result)

    # Filter out any None values and convert to JSON serializable format
    filtered_result = {}
    for key, value in result_dict.items():
        if value is not None and not key.startswith("_"):
            if isinstance(value, (str, int, float, bool)) or value is None:
                filtered_result[key] = value
            else:
                try:
                    # Try to convert to string or another JSON serializable format
                    filtered_result[key] = str(value)
                except Exception as e:
                    filtered_result[key] = f"Complex value (not displayable): {str(e)}"

    # Replace the placeholder in the template with the actual data
    html_content = jwt_display_template.replace(
        "const userData = SSO_DATA;",
        f"const userData = {json.dumps(filtered_result, indent=2)};",
    )

    return HTMLResponse(content=html_content)
