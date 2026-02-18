"""
This file handles authentication for the LiteLLM Proxy.

it checks if the user passed a valid API Key to the LiteLLM Proxy

Returns a UserAPIKeyAuth object if the API key is valid

"""

import asyncio
import re
import secrets
from datetime import datetime, timezone
from typing import List, Optional, Tuple, cast

import fastapi
from fastapi import HTTPException, Request, WebSocket, status
from fastapi.security.api_key import APIKeyHeader

import litellm
from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm._service_logger import ServiceLogging
from litellm.caching import DualCache
from litellm.litellm_core_utils.dd_tracing import tracer
from litellm.proxy._types import *
from litellm.proxy.auth.auth_checks import (
    ExperimentalUIJWTToken,
    _cache_key_object,
    _delete_cache_key_object,
    _get_user_role,
    _is_user_proxy_admin,
    _virtual_key_max_budget_alert_check,
    _virtual_key_max_budget_check,
    _virtual_key_soft_budget_check,
    can_key_call_model,
    common_checks,
    get_end_user_object,
    get_key_object,
    get_team_object,
    get_user_object,
    is_valid_fallback_model,
)
from litellm.proxy.auth.auth_exception_handler import UserAPIKeyAuthExceptionHandler
from litellm.proxy.auth.auth_utils import (
    abbreviate_api_key,
    get_end_user_id_from_request_body,
    get_model_from_request,
    get_request_route,
    normalize_request_route,
    pre_db_read_auth_checks,
    route_in_additonal_public_routes,
)
from litellm.proxy.auth.handle_jwt import JWTAuthManager, JWTHandler
from litellm.proxy.auth.oauth2_check import Oauth2Handler
from litellm.proxy.auth.oauth2_proxy_hook import handle_oauth2_proxy_request
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.common_utils.cache_coordinator import EventDrivenCacheCoordinator
from litellm.proxy.common_utils.http_parsing_utils import (
    _read_request_body,
    _safe_get_request_headers,
    populate_request_with_path_params,
)
from litellm.proxy.common_utils.realtime_utils import _realtime_request_body
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.secret_managers.main import get_secret_bool
from litellm.types.services import ServiceTypes

try:
    from litellm_enterprise.proxy.auth.user_api_key_auth import (
        enterprise_custom_auth as _enterprise_custom_auth,
    )

    enterprise_custom_auth: Optional[Callable] = _enterprise_custom_auth
except ImportError as e:
    verbose_proxy_logger.debug(f"Error in enterprise custom auth: {e}")
    enterprise_custom_auth = None

user_api_key_service_logger_obj = ServiceLogging()  # used for tracking latency on OTEL

custom_litellm_key_header = APIKeyHeader(
    name=SpecialHeaders.custom_litellm_api_key.value,
    auto_error=False,
    description="Bearer token",
)
api_key_header = APIKeyHeader(
    name=SpecialHeaders.openai_authorization.value,
    auto_error=False,
    description="Bearer token",
)
azure_api_key_header = APIKeyHeader(
    name=SpecialHeaders.azure_authorization.value,
    auto_error=False,
    description="Some older versions of the openai Python package will send an API-Key header with just the API key ",
)
anthropic_api_key_header = APIKeyHeader(
    name=SpecialHeaders.anthropic_authorization.value,
    auto_error=False,
    description="If anthropic client used.",
)
google_ai_studio_api_key_header = APIKeyHeader(
    name=SpecialHeaders.google_ai_studio_authorization.value,
    auto_error=False,
    description="If google ai studio client used.",
)
azure_apim_header = APIKeyHeader(
    name=SpecialHeaders.azure_apim_authorization.value,
    auto_error=False,
    description="The default name of the subscription key header of Azure",
)


def _get_bearer_token_or_received_api_key(api_key: str) -> str:
    if api_key.startswith("Bearer "):  # ensure Bearer token passed in
        api_key = api_key.replace("Bearer ", "")  # extract the token
    elif api_key.startswith("Basic "):
        api_key = api_key.replace("Basic ", "")  # handle langfuse input
    elif api_key.startswith("bearer "):
        api_key = api_key.replace("bearer ", "")
    elif api_key.startswith("AWS4-HMAC-SHA256"):
        # Handle AWS Signature V4 format from LangChain
        # Format: AWS4-HMAC-SHA256 Credential=Bearer sk-12345/date/region/service/aws4_request, SignedHeaders=..., Signature=...
        # Extract the Bearer token from the Credential field
        match = re.search(r'Credential=Bearer\s+([^/\s,]+)', api_key)
        if match:
            api_key = match.group(1)
        else:
            # If no Bearer token found in Credential, try to extract just the credential value
            match = re.search(r'Credential=([^/\s,]+)', api_key)
            if match:
                api_key = match.group(1)

    return api_key


def _get_bearer_token(
    api_key: str,
):
    if api_key.startswith("Bearer "):  # ensure Bearer token passed in
        api_key = api_key.replace("Bearer ", "")  # extract the token
    elif api_key.startswith("Basic "):
        api_key = api_key.replace("Basic ", "")  # handle langfuse input
    elif api_key.startswith("bearer "):
        api_key = api_key.replace("bearer ", "")
    elif api_key.startswith("AWS4-HMAC-SHA256"):
        # Handle AWS Signature V4 format from LangChain
        # Format: AWS4-HMAC-SHA256 Credential=Bearer sk-12345/date/region/service/aws4_request, SignedHeaders=..., Signature=...
        # Extract the Bearer token from the Credential field
        match = re.search(r'Credential=Bearer\s+([^/\s,]+)', api_key)
        if match:
            api_key = match.group(1)
        else:
            # If no Bearer token found in Credential, try to extract just the credential value
            match = re.search(r'Credential=([^/\s,]+)', api_key)
            if match:
                api_key = match.group(1)
            else:
                api_key = ""
    else:
        api_key = ""
    return api_key


def _apply_budget_limits_to_end_user_params(
    end_user_params: dict,
    budget_info: LiteLLM_BudgetTable,
    end_user_id: str,
) -> None:
    """
    Helper function to apply budget limits to end user parameters.

    Args:
        end_user_params: Dictionary to update with budget parameters
        budget_info: Budget table object containing limits
        end_user_id: ID of the end user for logging
    """
    if budget_info.tpm_limit is not None:
        end_user_params["end_user_tpm_limit"] = budget_info.tpm_limit

    if budget_info.rpm_limit is not None:
        end_user_params["end_user_rpm_limit"] = budget_info.rpm_limit

    if budget_info.max_budget is not None:
        end_user_params["end_user_max_budget"] = budget_info.max_budget

    verbose_proxy_logger.debug(f"Applied budget limits to end user {end_user_id}")


async def user_api_key_auth_websocket(websocket: WebSocket):
    # Accept the WebSocket connection

    scope_headers = list(websocket.scope.get("headers") or [])
    request = Request(scope={"type": "http", "headers": scope_headers})

    request._url = websocket.url

    query_params = websocket.query_params

    model = query_params.get("model")

    async def return_body():
        return _realtime_request_body(model)

    request.body = return_body  # type: ignore

    authorization = websocket.headers.get("authorization")
    # If no Authorization header, try the api-key header
    if not authorization:
        api_key = websocket.headers.get("api-key")
        if not api_key:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            raise HTTPException(status_code=403, detail="No API key provided")
    else:
        # Extract the API key from the Bearer token
        if not authorization.startswith("Bearer "):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            raise HTTPException(
                status_code=403, detail="Invalid Authorization header format"
            )

        api_key = authorization[len("Bearer ") :].strip()

    # Call user_api_key_auth with the extracted API key
    # Note: You'll need to modify this to work with WebSocket context if needed
    try:
        return await user_api_key_auth(request=request, api_key=f"Bearer {api_key}")
    except Exception as e:
        verbose_proxy_logger.exception(e)
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        raise HTTPException(status_code=403, detail=str(e))


def update_valid_token_with_end_user_params(
    valid_token: UserAPIKeyAuth, end_user_params: dict
) -> UserAPIKeyAuth:
    valid_token.end_user_id = end_user_params.get("end_user_id")
    valid_token.end_user_tpm_limit = end_user_params.get("end_user_tpm_limit")
    valid_token.end_user_rpm_limit = end_user_params.get("end_user_rpm_limit")
    valid_token.allowed_model_region = end_user_params.get("allowed_model_region")
    return valid_token


# Reusable coordinator for global spend to prevent cache stampede
_global_spend_coordinator = EventDrivenCacheCoordinator(log_prefix="[GLOBAL SPEND]")


async def _fetch_global_spend_with_event_coordination(
    cache_key: str,
    user_api_key_cache: DualCache,
    prisma_client: PrismaClient,
) -> Optional[float]:
    """
    Fetch global spend with event-driven coordination to prevent cache stampede.
    Uses EventDrivenCacheCoordinator: first request queries DB and signals others when done.
    """

    async def _load_global_spend() -> Optional[float]:
        sql_query = """SELECT SUM(spend) AS total_spend FROM "MonthlyGlobalSpend";"""
        response = await prisma_client.db.query_raw(query=sql_query)
        val = response[0]["total_spend"]
        return float(val) if val is not None else None

    return await _global_spend_coordinator.get_or_load(
        cache_key=cache_key,
        cache=user_api_key_cache,
        load_fn=_load_global_spend,
    )


async def get_global_proxy_spend(
    litellm_proxy_admin_name: str,
    user_api_key_cache: DualCache,
    prisma_client: Optional[PrismaClient],
    token: str,
    proxy_logging_obj: ProxyLogging,
) -> Optional[float]:
    global_proxy_spend = None
    if litellm.max_budget > 0 and prisma_client is not None:  # user set proxy max budget
        # Use event-driven coordination to prevent cache stampede
        cache_key = "{}:spend".format(litellm_proxy_admin_name)
        global_proxy_spend = await _fetch_global_spend_with_event_coordination(
            cache_key=cache_key,
            user_api_key_cache=user_api_key_cache,
            prisma_client=prisma_client,
        )
        if global_proxy_spend is not None:
            user_info = CallInfo(
                user_id=litellm_proxy_admin_name,
                max_budget=litellm.max_budget,
                spend=global_proxy_spend,
                token=token,
                event_group=Litellm_EntityType.PROXY,
            )
            asyncio.create_task(
                proxy_logging_obj.budget_alerts(
                    type="proxy_budget",
                    user_info=user_info,
                )
            )
    return global_proxy_spend


def get_rbac_role(jwt_handler: JWTHandler, scopes: List[str]) -> str:
    is_admin = jwt_handler.is_admin(scopes=scopes)
    if is_admin:
        return LitellmUserRoles.PROXY_ADMIN
    else:
        return LitellmUserRoles.TEAM


def get_api_key(
    custom_litellm_key_header: Optional[str],
    api_key: str,
    azure_api_key_header: Optional[str],
    anthropic_api_key_header: Optional[str],
    google_ai_studio_api_key_header: Optional[str],
    azure_apim_header: Optional[str],
    pass_through_endpoints: Optional[List[dict]],
    route: str,
    request: Request,
) -> Tuple[str, Optional[str]]:
    """
    Returns:
        Tuple[Optional[str], Optional[str]]: Tuple of the api_key and the passed_in_key
    """
    from litellm.proxy.auth.route_checks import RouteChecks
    from litellm.proxy.common_utils.http_parsing_utils import (
        _safe_get_request_query_params,
    )

    api_key = api_key
    passed_in_key: Optional[str] = None
    if isinstance(custom_litellm_key_header, str):
        passed_in_key = custom_litellm_key_header
        api_key = _get_bearer_token_or_received_api_key(custom_litellm_key_header)
    elif isinstance(api_key, str) and len(api_key) > 0:
        passed_in_key = api_key
        api_key = _get_bearer_token(api_key=api_key)
    elif isinstance(azure_api_key_header, str):
        passed_in_key = azure_api_key_header
        api_key = azure_api_key_header
    elif isinstance(anthropic_api_key_header, str):
        passed_in_key = anthropic_api_key_header
        api_key = anthropic_api_key_header
    elif isinstance(google_ai_studio_api_key_header, str):
        passed_in_key = google_ai_studio_api_key_header
        api_key = google_ai_studio_api_key_header
    elif isinstance(azure_apim_header, str):
        passed_in_key = azure_apim_header
        api_key = azure_apim_header
    elif (
        RouteChecks.is_generate_content_route(route=route)
        and request is not None
        and _safe_get_request_query_params(request).get("key")
    ):
        google_auth_key: str = _safe_get_request_query_params(request).get("key") or ""
        passed_in_key = google_auth_key
        api_key = google_auth_key
    elif pass_through_endpoints is not None:
        for endpoint in pass_through_endpoints:
            if endpoint.get("path", "") == route:
                headers: Optional[dict] = endpoint.get("headers", None)
                if headers is not None:
                    header_key: str = headers.get("litellm_user_api_key", "")
                    if request.headers.get(header_key) is not None:
                        api_key = request.headers.get(header_key) or ""
                        passed_in_key = api_key
    return api_key, passed_in_key


async def check_api_key_for_custom_headers_or_pass_through_endpoints(
    request: Request,
    route: str,
    pass_through_endpoints: Optional[List[dict]],
    api_key: str,
) -> Union[UserAPIKeyAuth, str]:
    is_mapped_pass_through_route: bool = False
    for mapped_route in LiteLLMRoutes.mapped_pass_through_routes.value:  # type: ignore
        if route.startswith(mapped_route):
            is_mapped_pass_through_route = True
    if is_mapped_pass_through_route:
        if request.headers.get("litellm_user_api_key") is not None:
            api_key = request.headers.get("litellm_user_api_key") or ""
    if pass_through_endpoints is not None:
        for endpoint in pass_through_endpoints:
            if isinstance(endpoint, dict) and endpoint.get("path", "") == route:
                ## IF AUTH DISABLED
                if endpoint.get("auth") is not True:
                    return UserAPIKeyAuth()
                ## IF AUTH ENABLED
                ### IF CUSTOM PARSER REQUIRED
                if (
                    endpoint.get("custom_auth_parser") is not None
                    and endpoint.get("custom_auth_parser") == "langfuse"
                ):
                    """
                    - langfuse returns {'Authorization': 'Basic YW55dGhpbmc6YW55dGhpbmc'}
                    - check the langfuse public key if it contains the litellm api key
                    """
                    import base64

                    api_key = api_key.replace("Basic ", "").strip()
                    decoded_bytes = base64.b64decode(api_key)
                    decoded_str = decoded_bytes.decode("utf-8")
                    api_key = decoded_str.split(":")[0]
                else:
                    headers = endpoint.get("headers", None)
                    if headers is not None:
                        header_key = headers.get("litellm_user_api_key", "")
                        if (
                            isinstance(request.headers, dict)
                            and request.headers.get(key=header_key) is not None  # type: ignore
                        ):
                            api_key = request.headers.get(key=header_key)  # type: ignore
    return api_key


async def _user_api_key_auth_builder(  # noqa: PLR0915
    request: Request,
    api_key: str,
    azure_api_key_header: str,
    anthropic_api_key_header: Optional[str],
    google_ai_studio_api_key_header: Optional[str],
    azure_apim_header: Optional[str],
    request_data: dict,
    custom_litellm_key_header: Optional[str] = None,
) -> UserAPIKeyAuth:
    from litellm.proxy.proxy_server import (
        general_settings,
        jwt_handler,
        litellm_proxy_admin_name,
        llm_model_list,
        llm_router,
        master_key,
        model_max_budget_limiter,
        open_telemetry_logger,
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
        user_custom_auth,
    )

    parent_otel_span: Optional[Span] = None
    start_time = datetime.now()
    route: str = get_request_route(request=request)
    valid_token: Optional[UserAPIKeyAuth] = None
    custom_auth_api_key: bool = False

    try:
        # get the request body

        await pre_db_read_auth_checks(
            request_data=request_data,
            request=request,
            route=route,
        )
        pass_through_endpoints: Optional[List[dict]] = general_settings.get(
            "pass_through_endpoints", None
        )
        ## CHECK IF X-LITELM-API-KEY IS PASSED IN - supercedes Authorization header
        api_key, passed_in_key = get_api_key(
            custom_litellm_key_header=custom_litellm_key_header,
            api_key=api_key,
            azure_api_key_header=azure_api_key_header,
            anthropic_api_key_header=anthropic_api_key_header,
            google_ai_studio_api_key_header=google_ai_studio_api_key_header,
            azure_apim_header=azure_apim_header,
            pass_through_endpoints=pass_through_endpoints,
            route=route,
            request=request,
        )
        # if user wants to pass LiteLLM_Master_Key as a custom header, example pass litellm keys as X-LiteLLM-Key: Bearer sk-1234
        custom_litellm_key_header_name = general_settings.get("litellm_key_header_name")
        if custom_litellm_key_header_name is not None:
            api_key = get_api_key_from_custom_header(
                request=request,
                custom_litellm_key_header_name=custom_litellm_key_header_name,
            )

        if open_telemetry_logger is not None:
            parent_otel_span = (
                open_telemetry_logger.create_litellm_proxy_request_started_span(
                    start_time=start_time,
                    headers=dict(request.headers),
                )
            )

        ### USER-DEFINED AUTH FUNCTION ###
        if enterprise_custom_auth is not None:
            response = await enterprise_custom_auth(
                request=request, api_key=api_key, user_custom_auth=user_custom_auth
            )
            if response is not None and isinstance(response, UserAPIKeyAuth):
                return UserAPIKeyAuth.model_validate(response)
            elif response is not None and isinstance(response, str):
                api_key = response
                custom_auth_api_key = True
        elif user_custom_auth is not None:
            response = await user_custom_auth(request=request, api_key=api_key)  # type: ignore
            return UserAPIKeyAuth.model_validate(response)

        ### LITELLM-DEFINED AUTH FUNCTION ###
        #### IF JWT ####
        """
        LiteLLM supports using JWTs.

        Enable this in proxy config, by setting
        ```
        general_settings:
            enable_jwt_auth: true
        ```
        """

        ######## Route Checks Before Reading DB / Cache for "token" ################
        if (
            route in LiteLLMRoutes.public_routes.value  # type: ignore
            or route_in_additonal_public_routes(current_route=route)
        ):
            # check if public endpoint
            return UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY)

        ########## End of Route Checks Before Reading DB / Cache for "token" ########

        if general_settings.get("enable_oauth2_auth", False) is True:
            # Only apply OAuth2 M2M authentication to LLM API routes, not UI/management routes
            # This allows UI SSO to work separately from API M2M authentication
            if RouteChecks.is_llm_api_route(route=route):
                # return UserAPIKeyAuth object
                # helper to check if the api_key is a valid oauth2 token
                from litellm.proxy.proxy_server import premium_user

                if premium_user is not True:
                    raise ValueError(
                        "Oauth2 token validation is only available for premium users"
                        + CommonProxyErrors.not_premium_user.value
                    )

                return await Oauth2Handler.check_oauth2_token(token=api_key)

        if general_settings.get("enable_oauth2_proxy_auth", False) is True:
            return await handle_oauth2_proxy_request(request=request)

        if general_settings.get("enable_jwt_auth", False) is True:
            from litellm.proxy.proxy_server import premium_user

            if premium_user is not True:
                raise ValueError(
                    f"JWT Auth is an enterprise only feature. {CommonProxyErrors.not_premium_user.value}"
                )
            is_jwt = jwt_handler.is_jwt(token=api_key)
            verbose_proxy_logger.debug("is_jwt: %s", is_jwt)
            if is_jwt:
                result = await JWTAuthManager.auth_builder(
                    request_data=request_data,
                    general_settings=general_settings,
                    api_key=api_key,
                    jwt_handler=jwt_handler,
                    route=route,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    proxy_logging_obj=proxy_logging_obj,
                    parent_otel_span=parent_otel_span,
                    request_headers=dict(request.headers),
                )

                is_proxy_admin = result["is_proxy_admin"]
                team_id = result["team_id"]
                team_object = result["team_object"]
                user_id = result["user_id"]
                user_object = result["user_object"]
                end_user_id = result["end_user_id"]
                end_user_object = result["end_user_object"]
                org_id = result["org_id"]
                token = result["token"]
                team_membership: Optional[LiteLLM_TeamMembership] = result.get(
                    "team_membership", None
                )

                global_proxy_spend = await get_global_proxy_spend(
                    litellm_proxy_admin_name=litellm_proxy_admin_name,
                    user_api_key_cache=user_api_key_cache,
                    prisma_client=prisma_client,
                    token=token,
                    proxy_logging_obj=proxy_logging_obj,
                )

                if is_proxy_admin:
                    return UserAPIKeyAuth(
                        api_key=None,
                        user_role=LitellmUserRoles.PROXY_ADMIN,
                        user_id=user_id,
                        team_id=team_id,
                        team_alias=(
                            team_object.team_alias
                            if team_object is not None
                            else None
                        ),
                        team_metadata=team_object.metadata
                        if team_object is not None
                        else None,
                        org_id=org_id,
                        end_user_id=end_user_id,
                        parent_otel_span=parent_otel_span,
                    )

                valid_token = UserAPIKeyAuth(
                    api_key=None,
                    team_id=team_id,
                    team_alias=(
                        team_object.team_alias if team_object is not None else None
                    ),
                    team_tpm_limit=(
                        team_object.tpm_limit if team_object is not None else None
                    ),
                    team_rpm_limit=(
                        team_object.rpm_limit if team_object is not None else None
                    ),
                    team_models=team_object.models if team_object is not None else [],
                    user_role=(
                        LitellmUserRoles(user_object.user_role)
                        if user_object is not None and user_object.user_role is not None
                        else LitellmUserRoles.INTERNAL_USER
                    ),
                    user_id=user_id,
                    org_id=org_id,
                    parent_otel_span=parent_otel_span,
                    end_user_id=end_user_id,
                    user_tpm_limit=(
                        user_object.tpm_limit if user_object is not None else None
                    ),
                    user_rpm_limit=(
                        user_object.rpm_limit if user_object is not None else None
                    ),
                    team_member_rpm_limit=(
                        team_membership.safe_get_team_member_rpm_limit()
                        if team_membership is not None
                        else None
                    ),
                    team_member_tpm_limit=(
                        team_membership.safe_get_team_member_tpm_limit()
                        if team_membership is not None
                        else None
                    ),
                    team_metadata=team_object.metadata
                    if team_object is not None
                    else None,
                )
                
                # Check if model has zero cost - if so, skip all budget checks
                model = get_model_from_request(request_data, route)
                skip_budget_checks = False
                if model is not None and llm_router is not None:
                    from litellm.proxy.auth.auth_checks import _is_model_cost_zero
                    
                    skip_budget_checks = _is_model_cost_zero(
                        model=model, llm_router=llm_router
                    )
                    if skip_budget_checks:
                        verbose_proxy_logger.info(
                            f"Skipping all budget checks for zero-cost model: {model}"
                        )
                
                # run through common checks
                _ = await common_checks(
                    request=request,
                    request_body=request_data,
                    team_object=team_object,
                    user_object=user_object,
                    end_user_object=end_user_object,
                    general_settings=general_settings,
                    global_proxy_spend=global_proxy_spend,
                    route=route,
                    llm_router=llm_router,
                    proxy_logging_obj=proxy_logging_obj,
                    valid_token=valid_token,
                    skip_budget_checks=skip_budget_checks,
                )

                # return UserAPIKeyAuth object
                return cast(UserAPIKeyAuth, valid_token)

        #### ELSE ####
        ## CHECK PASS-THROUGH ENDPOINTS ##
        if not custom_auth_api_key:
            response = await check_api_key_for_custom_headers_or_pass_through_endpoints(
                request=request,
                route=route,
                pass_through_endpoints=pass_through_endpoints,
                api_key=api_key,
            )
            if isinstance(response, str):
                api_key = response
            elif isinstance(response, UserAPIKeyAuth):
                return response
        if master_key is None:
            if isinstance(api_key, str):
                return UserAPIKeyAuth(
                    api_key=api_key,
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                    parent_otel_span=parent_otel_span,
                )
            else:
                return UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                    parent_otel_span=parent_otel_span,
                )
        elif api_key is None:  # only require api key if master key is set
            raise Exception("No api key passed in.")
        elif api_key == "":
            # missing 'Bearer ' prefix
            raise Exception(
                "Malformed API Key passed in. Ensure Key has `Bearer ` prefix."
            )

        if route == "/user/auth":
            if general_settings.get("allow_user_auth", False) is True:
                return UserAPIKeyAuth()
            else:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="'allow_user_auth' not set or set to False",
                )

        ## Check END-USER OBJECT
        _end_user_object = None
        end_user_params = {}

        end_user_id = get_end_user_id_from_request_body(
            request_data, _safe_get_request_headers(request)
        )
        if end_user_id:
            try:
                end_user_params["end_user_id"] = end_user_id

                # get end-user object
                _end_user_object = await get_end_user_object(
                    end_user_id=end_user_id,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    parent_otel_span=parent_otel_span,
                    proxy_logging_obj=proxy_logging_obj,
                    route=route,
                )
                if _end_user_object is not None:
                    end_user_params[
                        "allowed_model_region"
                    ] = _end_user_object.allowed_model_region
                    if _end_user_object.litellm_budget_table is not None:
                        _apply_budget_limits_to_end_user_params(
                            end_user_params=end_user_params,
                            budget_info=_end_user_object.litellm_budget_table,
                            end_user_id=end_user_id,
                        )
                elif litellm.max_end_user_budget_id is not None:
                    # End user doesn't exist yet, but apply default budget limits if configured
                    from litellm.proxy.auth.auth_checks import (
                        get_default_end_user_budget,
                    )

                    default_budget = await get_default_end_user_budget(
                        prisma_client=prisma_client,
                        user_api_key_cache=user_api_key_cache,
                        parent_otel_span=parent_otel_span,
                    )
                    if default_budget is not None:
                        _apply_budget_limits_to_end_user_params(
                            end_user_params=end_user_params,
                            budget_info=default_budget,
                            end_user_id=end_user_id,
                        )
            except Exception as e:
                if isinstance(e, litellm.BudgetExceededError):
                    raise e
                verbose_proxy_logger.debug(
                    "Unable to find user in db. Error - {}".format(str(e))
                )
                pass

        ### CHECK IF ADMIN ###
        # note: never string compare api keys, this is vulenerable to a time attack. Use secrets.compare_digest instead
        ### CHECK IF ADMIN ###
        # note: never string compare api keys, this is vulenerable to a time attack. Use secrets.compare_digest instead
        ## Check CACHE
        try:
            valid_token = await get_key_object(
                hashed_token=hash_token(api_key),
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
                check_cache_only=True,
            )
        except Exception:
            verbose_logger.debug("api key not found in cache.")
            valid_token = None

        ## Check UI Hash Key
        if valid_token is None and get_secret_bool("EXPERIMENTAL_UI_LOGIN"):
            valid_token = ExperimentalUIJWTToken.get_key_object_from_ui_hash_key(
                api_key
            )

        if (
            valid_token is not None
            and isinstance(valid_token, UserAPIKeyAuth)
            and valid_token.user_role == LitellmUserRoles.PROXY_ADMIN
        ):
            if valid_token.expires is not None:
                current_time = datetime.now(timezone.utc)
                if isinstance(valid_token.expires, datetime):
                    expiry_time = valid_token.expires
                else:
                    expiry_time = datetime.fromisoformat(valid_token.expires)
                if (
                    expiry_time.tzinfo is None
                    or expiry_time.tzinfo.utcoffset(expiry_time) is None
                ):
                    expiry_time = expiry_time.replace(tzinfo=timezone.utc)
                if expiry_time < current_time:
                    await _delete_cache_key_object(
                        hashed_token=hash_token(api_key),
                        user_api_key_cache=user_api_key_cache,
                        proxy_logging_obj=proxy_logging_obj,
                    )
                    raise ProxyException(
                        message=f"Authentication Error - Expired Key. Key Expiry time {expiry_time} and current time {current_time}",
                        type=ProxyErrorTypes.expired_key,
                        code=400,
                        param=abbreviate_api_key(api_key=api_key),
                    )
            valid_token = update_valid_token_with_end_user_params(
                valid_token=valid_token, end_user_params=end_user_params
            )
            valid_token.parent_otel_span = parent_otel_span

            return valid_token

        if (
            valid_token is not None
            and isinstance(valid_token, UserAPIKeyAuth)
            and valid_token.team_id is not None
        ):
            ## UPDATE TEAM VALUES BASED ON CACHED TEAM OBJECT - allows `/team/update` values to work for cached token
            try:
                team_obj: LiteLLM_TeamTableCachedObj = await get_team_object(
                    team_id=valid_token.team_id,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    parent_otel_span=parent_otel_span,
                    proxy_logging_obj=proxy_logging_obj,
                    check_cache_only=True,
                )

                if (
                    team_obj.last_refreshed_at is not None
                    and valid_token.last_refreshed_at is not None
                    and team_obj.last_refreshed_at > valid_token.last_refreshed_at
                ):
                    team_obj_dict = team_obj.__dict__

                    for k, v in team_obj_dict.items():
                        field_name = f"team_{k}"
                        if field_name in valid_token.__fields__:
                            setattr(valid_token, field_name, v)
            except Exception as e:
                verbose_logger.debug(
                    e
                )  # moving from .warning to .debug as it spams logs when team missing from cache.

        try:
            is_master_key_valid = secrets.compare_digest(api_key, master_key)  # type: ignore
        except Exception:
            is_master_key_valid = False

        ## VALIDATE MASTER KEY ##
        try:
            assert isinstance(master_key, str)
        except Exception:
            raise HTTPException(
                status_code=500,
                detail={
                    "Master key must be a valid string. Current type={}".format(
                        type(master_key)
                    )
                },
            )

        if is_master_key_valid:
            _user_api_key_obj = await _return_user_api_key_auth_obj(
                user_obj=None,
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key=master_key,
                parent_otel_span=parent_otel_span,
                valid_token_dict={
                    **end_user_params,
                    "user_id": litellm_proxy_admin_name,
                },
                route=route,
                start_time=start_time,
            )
            asyncio.create_task(
                _cache_key_object(
                    hashed_token=hash_token(master_key),
                    user_api_key_obj=_user_api_key_obj,
                    user_api_key_cache=user_api_key_cache,
                    proxy_logging_obj=proxy_logging_obj,
                )
            )

            _user_api_key_obj = update_valid_token_with_end_user_params(
                valid_token=_user_api_key_obj, end_user_params=end_user_params
            )

            return _user_api_key_obj

        ## IF it's not a master key
        ## Route should not be in master_key_only_routes
        if route in LiteLLMRoutes.master_key_only_routes.value:  # type: ignore
            raise Exception(
                f"Tried to access route={route}, which is only for MASTER KEY"
            )

        ## Check DB

        if (
            prisma_client is None
        ):  # if both master key + user key submitted, and user key != master key, and no db connected, raise an error
            raise ProxyException(
                message="No connected db.",
                type=ProxyErrorTypes.no_db_connection,
                code=400,
                param=None,
            )

        ## check for cache hit (In-Memory Cache)
        _user_role = None

        if valid_token is None:
            if isinstance(
                api_key, str
            ):  # if generated token, make sure it starts with sk-.
                _masked_key = "{}****{}".format(api_key[:4], api_key[-4:]) if len(api_key) > 8 else "****"
                assert api_key.startswith(
                    "sk-"
                ), "LiteLLM Virtual Key expected. Received={}, expected to start with 'sk-'.".format(
                    _masked_key
                )  # prevent token hashes from being used
            else:
                verbose_logger.warning(
                    "litellm.proxy.proxy_server.user_api_key_auth(): Warning - Key is not a string. Got type={}".format(
                        type(api_key) if api_key is not None else "None"
                    )
                )
            abbreviated_api_key = abbreviate_api_key(api_key=api_key)
            if api_key.startswith("sk-"):
                api_key = hash_token(token=api_key)

            try:
                valid_token = await get_key_object(
                    hashed_token=api_key,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    parent_otel_span=parent_otel_span,
                    proxy_logging_obj=proxy_logging_obj,
                )
            except ProxyException as e:
                if e.code == 401 or e.code == "401":
                    e.message = "Authentication Error, Invalid proxy server token passed. Received API Key = {}, Key Hash (Token) ={}. Unable to find token in cache or `LiteLLM_VerificationTokenTable`".format(
                        abbreviated_api_key, api_key
                    )
                raise e
            # update end-user params on valid token
            # These can change per request - it's important to update them here
            valid_token.end_user_id = end_user_params.get("end_user_id")
            valid_token.end_user_tpm_limit = end_user_params.get("end_user_tpm_limit")
            valid_token.end_user_rpm_limit = end_user_params.get("end_user_rpm_limit")
            valid_token.allowed_model_region = end_user_params.get(
                "allowed_model_region"
            )
            # update key budget with temp budget increase
            valid_token = _update_key_budget_with_temp_budget_increase(
                valid_token
            )  # updating it here, allows all downstream reporting / checks to use the updated budget

        user_obj: Optional[LiteLLM_UserTable] = None
        valid_token_dict: dict = {}
        if valid_token is not None:
            # Got Valid Token from Cache, DB
            # Run checks for
            # 1. If token can call model
            ## 1a. If token can call fallback models (if client-side fallbacks given)
            # 2. If user_id for this token is in budget
            # 3. If the user spend within their own team is within budget
            # 4. If 'user' passed to /chat/completions, /embeddings endpoint is in budget
            # 5. If token is expired
            # 6. If token spend is under Budget for the token
            # 7. If token spend per model is under budget per model
            # 8. If token spend is under team budget
            # 9. If team spend is under team budget

            ## base case ## key is disabled
            if valid_token.blocked is True:
                raise Exception(
                    "Key is blocked. Update via `/key/unblock` if you're admin."
                )
            config = valid_token.config

            if config != {}:
                model_list = config.get("model_list", [])
                new_model_list = model_list
                verbose_proxy_logger.debug(
                    f"\n new llm router model list {new_model_list}"
                )
            elif (
                isinstance(valid_token.models, list)
                and "all-team-models" in valid_token.models
            ):
                # Do not do any validation at this step
                # the validation will occur when checking the team has access to this model
                pass
            else:
                model = get_model_from_request(request_data, route)
                fallback_models = cast(
                    Optional[List[ALL_FALLBACK_MODEL_VALUES]],
                    request_data.get("fallbacks", None),
                )

                if model is not None:
                    await can_key_call_model(
                        model=model,
                        llm_model_list=llm_model_list,
                        valid_token=valid_token,
                        llm_router=llm_router,
                    )

                if fallback_models is not None:
                    for m in fallback_models:
                        await can_key_call_model(
                            model=m["model"] if isinstance(m, dict) else m,
                            llm_model_list=llm_model_list,
                            valid_token=valid_token,
                            llm_router=llm_router,
                        )
                        await is_valid_fallback_model(
                            model=m["model"] if isinstance(m, dict) else m,
                            llm_router=llm_router,
                            user_model=None,
                        )

            # Check 2. If user_id for this token is in budget - done in common_checks()
            if valid_token.user_id is not None:
                try:
                    user_obj = await get_user_object(
                        user_id=valid_token.user_id,
                        prisma_client=prisma_client,
                        user_api_key_cache=user_api_key_cache,
                        user_id_upsert=False,
                        parent_otel_span=parent_otel_span,
                        proxy_logging_obj=proxy_logging_obj,
                    )
                except Exception as e:
                    verbose_logger.debug(
                        "litellm.proxy.auth.user_api_key_auth.py::user_api_key_auth() - Unable to get user from db/cache. Setting user_obj to None. Exception received - {}".format(
                            str(e)
                        )
                    )
                    user_obj = None

            # Check 2a. Check if model has zero cost - if so, skip all budget checks
            model = get_model_from_request(request_data, route)
            skip_budget_checks = False
            if model is not None and llm_router is not None:
                from litellm.proxy.auth.auth_checks import _is_model_cost_zero
                
                skip_budget_checks = _is_model_cost_zero(
                    model=model, llm_router=llm_router
                )
                if skip_budget_checks:
                    verbose_proxy_logger.info(
                        f"Skipping all budget checks for zero-cost model: {model}"
                    )

            # Check 3. Check if user is in their team budget
            if not skip_budget_checks and valid_token.team_member_spend is not None:
                if prisma_client is not None:
                    _cache_key = f"{valid_token.team_id}_{valid_token.user_id}"

                    team_member_info = await user_api_key_cache.async_get_cache(
                        key=_cache_key
                    )
                    if team_member_info is None:
                        # read from DB
                        _user_id = valid_token.user_id
                        _team_id = valid_token.team_id

                        if _user_id is not None and _team_id is not None:
                            team_member_info = await prisma_client.db.litellm_teammembership.find_first(
                                where={
                                    "user_id": _user_id,
                                    "team_id": _team_id,
                                },  # type: ignore
                                include={"litellm_budget_table": True},
                            )
                            await user_api_key_cache.async_set_cache(
                                key=_cache_key,
                                value=team_member_info,
                                ttl=5,
                            )

                    if (
                        team_member_info is not None
                        and team_member_info.litellm_budget_table is not None
                    ):
                        team_member_budget = (
                            team_member_info.litellm_budget_table.max_budget
                        )
                        if team_member_budget is not None and team_member_budget > 0:
                            if valid_token.team_member_spend > team_member_budget:
                                raise litellm.BudgetExceededError(
                                    current_cost=valid_token.team_member_spend,
                                    max_budget=team_member_budget,
                                )

            # Check 3. If token is expired
            if valid_token.expires is not None:
                current_time = datetime.now(timezone.utc)
                if isinstance(valid_token.expires, datetime):
                    expiry_time = valid_token.expires
                else:
                    expiry_time = datetime.fromisoformat(valid_token.expires)
                if (
                    expiry_time.tzinfo is None
                    or expiry_time.tzinfo.utcoffset(expiry_time) is None
                ):
                    expiry_time = expiry_time.replace(tzinfo=timezone.utc)
                verbose_proxy_logger.debug(
                    f"Checking if token expired, expiry time {expiry_time} and current time {current_time}"
                )
                if expiry_time < current_time:
                    # Token exists but is expired.
                    raise ProxyException(
                        message=f"Authentication Error - Expired Key. Key Expiry time {expiry_time} and current time {current_time}",
                        type=ProxyErrorTypes.expired_key,
                        code=400,
                        param=abbreviate_api_key(api_key=api_key),
                    )

            if not skip_budget_checks:
                # Check 4. Token Spend is under budget
                if RouteChecks.is_llm_api_route(route=route):
                    await _virtual_key_max_budget_check(
                        valid_token=valid_token,
                        proxy_logging_obj=proxy_logging_obj,
                        user_obj=user_obj,
                    )

                # Check 5. Max Budget Alert Check
                await _virtual_key_max_budget_alert_check(
                    valid_token=valid_token,
                    proxy_logging_obj=proxy_logging_obj,
                    user_obj=user_obj,
                )

                # Check 6. Soft Budget Check
                await _virtual_key_soft_budget_check(
                    valid_token=valid_token,
                    proxy_logging_obj=proxy_logging_obj,
                    user_obj=user_obj,
                )

                # Check 5. Token Model Spend is under Model budget
                max_budget_per_model = valid_token.model_max_budget
                current_model = request_data.get("model", None)

                if (
                    max_budget_per_model is not None
                    and isinstance(max_budget_per_model, dict)
                    and len(max_budget_per_model) > 0
                    and prisma_client is not None
                    and current_model is not None
                    and valid_token.token is not None
                ):
                    ## GET THE SPEND FOR THIS MODEL
                    await model_max_budget_limiter.is_key_within_model_budget(
                        user_api_key_dict=valid_token,
                        model=current_model,
                    )

            # Check 6: Additional Common Checks across jwt + key auth
            if valid_token.team_id is not None:
                try:
                    _team_obj = await get_team_object(
                        team_id=valid_token.team_id,
                        prisma_client=prisma_client,
                        user_api_key_cache=user_api_key_cache,
                        parent_otel_span=parent_otel_span,
                        proxy_logging_obj=proxy_logging_obj,
                    )
                except HTTPException:
                    _team_obj = LiteLLM_TeamTableCachedObj(
                        team_id=valid_token.team_id,
                        max_budget=valid_token.team_max_budget,
                        soft_budget=valid_token.team_soft_budget,
                        spend=valid_token.team_spend,
                        tpm_limit=valid_token.team_tpm_limit,
                        rpm_limit=valid_token.team_rpm_limit,
                        blocked=valid_token.team_blocked,
                        models=valid_token.team_models,
                        metadata=valid_token.team_metadata,
                        object_permission_id=valid_token.team_object_permission_id,
                    )
            else:
                _team_obj = None

            user_api_key_cache.set_cache(
                key=valid_token.team_id, value=_team_obj
            )  # save team table in cache - used for tpm/rpm limiting - tpm_rpm_limiter.py

            global_proxy_spend = None
            if (
                litellm.max_budget > 0 and prisma_client is not None
            ):  # user set proxy max budget
                cache_key = "{}:spend".format(litellm_proxy_admin_name)
                global_proxy_spend = await _fetch_global_spend_with_event_coordination(
                    cache_key=cache_key,
                    user_api_key_cache=user_api_key_cache,
                    prisma_client=prisma_client,
                )

                if global_proxy_spend is not None:
                    call_info = CallInfo(
                        token=valid_token.token,
                        spend=global_proxy_spend,
                        max_budget=litellm.max_budget,
                        user_id=litellm_proxy_admin_name,
                        team_id=valid_token.team_id,
                        event_group=Litellm_EntityType.PROXY,
                    )
                    asyncio.create_task(
                        proxy_logging_obj.budget_alerts(
                            type="proxy_budget",
                            user_info=call_info,
                        )
                    )
            _ = await common_checks(
                request=request,
                request_body=request_data,
                team_object=_team_obj,
                user_object=user_obj,
                end_user_object=_end_user_object,
                general_settings=general_settings,
                global_proxy_spend=global_proxy_spend,
                route=route,
                llm_router=llm_router,
                proxy_logging_obj=proxy_logging_obj,
                valid_token=valid_token,
                skip_budget_checks=skip_budget_checks,
            )
            # Token passed all checks
            if valid_token is None:
                raise HTTPException(401, detail="Invalid API key")
            if valid_token.token is None:
                raise HTTPException(401, detail="Invalid API key, no token associated")
            api_key = valid_token.token

            # Add hashed token to cache
            asyncio.create_task(
                _cache_key_object(
                    hashed_token=api_key,
                    user_api_key_obj=valid_token,
                    user_api_key_cache=user_api_key_cache,
                    proxy_logging_obj=proxy_logging_obj,
                )
            )

            valid_token_dict = valid_token.model_dump(exclude_none=True)
            valid_token_dict.pop("token", None)

            if _end_user_object is not None:
                valid_token_dict.update(end_user_params)

        # check if token is from litellm-ui, litellm ui makes keys to allow users to login with sso. These keys can only be used for LiteLLM UI functions
        # sso/login, ui/login, /key functions and /user functions
        # this will never be allowed to call /chat/completions

        if valid_token is None:
            # No token was found when looking up in the DB
            raise Exception("Invalid proxy server token passed")
        if valid_token_dict is not None:
            return await _return_user_api_key_auth_obj(
                user_obj=user_obj,
                api_key=api_key,
                parent_otel_span=parent_otel_span,
                valid_token_dict=valid_token_dict,
                route=route,
                start_time=start_time,
            )
    except Exception as e:
        return await UserAPIKeyAuthExceptionHandler._handle_authentication_error(
            e=e,
            request=request,
            request_data=request_data,
            route=route,
            parent_otel_span=parent_otel_span,
            api_key=api_key,
        )


@tracer.wrap()
async def user_api_key_auth(
    request: Request,
    api_key: str = fastapi.Security(api_key_header),
    azure_api_key_header: str = fastapi.Security(azure_api_key_header),
    anthropic_api_key_header: Optional[str] = fastapi.Security(
        anthropic_api_key_header
    ),
    google_ai_studio_api_key_header: Optional[str] = fastapi.Security(
        google_ai_studio_api_key_header
    ),
    azure_apim_header: Optional[str] = fastapi.Security(azure_apim_header),
    custom_litellm_key_header: Optional[str] = fastapi.Security(
        custom_litellm_key_header
    ),
) -> UserAPIKeyAuth:
    """
    Parent function to authenticate user api key / jwt token.
    """

    request_data = await _read_request_body(request=request)
    request_data = populate_request_with_path_params(
        request_data=request_data, request=request
    )
    route: str = get_request_route(request=request)
    ## CHECK IF ROUTE IS ALLOWED

    user_api_key_auth_obj = await _user_api_key_auth_builder(
        request=request,
        api_key=api_key,
        azure_api_key_header=azure_api_key_header,
        anthropic_api_key_header=anthropic_api_key_header,
        google_ai_studio_api_key_header=google_ai_studio_api_key_header,
        azure_apim_header=azure_apim_header,
        request_data=request_data,
        custom_litellm_key_header=custom_litellm_key_header,
    )

    ## ENSURE DISABLE ROUTE WORKS ACROSS ALL USER AUTH FLOWS ##
    RouteChecks.should_call_route(route=route, valid_token=user_api_key_auth_obj)

    end_user_id = get_end_user_id_from_request_body(
        request_data, _safe_get_request_headers(request)
    )
    if end_user_id is not None:
        user_api_key_auth_obj.end_user_id = end_user_id

    user_api_key_auth_obj.request_route = normalize_request_route(route)
    return user_api_key_auth_obj


async def _return_user_api_key_auth_obj(
    user_obj: Optional[LiteLLM_UserTable],
    api_key: str,
    parent_otel_span: Optional[Span],
    valid_token_dict: dict,
    route: str,
    start_time: datetime,
    user_role: Optional[LitellmUserRoles] = None,
) -> UserAPIKeyAuth:
    end_time = datetime.now()

    asyncio.create_task(
        user_api_key_service_logger_obj.async_service_success_hook(
            service=ServiceTypes.AUTH,
            call_type=route,
            start_time=start_time,
            end_time=end_time,
            duration=end_time.timestamp() - start_time.timestamp(),
            parent_otel_span=parent_otel_span,
        )
    )

    retrieved_user_role = (
        user_role or _get_user_role(user_obj=user_obj) or LitellmUserRoles.INTERNAL_USER
    )

    user_api_key_kwargs = {
        "api_key": api_key,
        "parent_otel_span": parent_otel_span,
        "user_role": retrieved_user_role,
        **valid_token_dict,
    }
    if user_obj is not None:
        user_api_key_kwargs.update(
            user_tpm_limit=user_obj.tpm_limit,
            user_rpm_limit=user_obj.rpm_limit,
            user_email=user_obj.user_email,
            user_spend=getattr(user_obj, "spend", None),
            user_max_budget=getattr(user_obj, "max_budget", None),
        )
    if user_obj is not None and _is_user_proxy_admin(user_obj=user_obj):
        user_api_key_kwargs.update(
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )
        return UserAPIKeyAuth(**user_api_key_kwargs)
    else:
        return UserAPIKeyAuth(**user_api_key_kwargs)


def get_api_key_from_custom_header(
    request: Request, custom_litellm_key_header_name: str
) -> str:
    """
    Get API key from custom header

    Args:
        request (Request): Request object
        custom_litellm_key_header_name (str): Custom header name

    Returns:
        Optional[str]: API key
    """
    api_key: str = ""
    # use this as the virtual key passed to litellm proxy
    custom_litellm_key_header_name = custom_litellm_key_header_name.lower()
    _headers = {k.lower(): v for k, v in request.headers.items()}
    verbose_proxy_logger.debug(
        "searching for custom_litellm_key_header_name= %s, in headers=%s",
        custom_litellm_key_header_name,
        _headers,
    )
    custom_api_key = _headers.get(custom_litellm_key_header_name)
    if custom_api_key:
        api_key = _get_bearer_token(api_key=custom_api_key)
        verbose_proxy_logger.debug(
            "Found custom API key using header: {}, setting api_key={}".format(
                custom_litellm_key_header_name, abbreviate_api_key(api_key)
            )
        )
    else:
        verbose_proxy_logger.exception(
            f"No LiteLLM Virtual Key pass. Please set header={custom_litellm_key_header_name}: Bearer <api_key>"
        )
    return api_key


def _get_temp_budget_increase(valid_token: UserAPIKeyAuth):
    valid_token_metadata = valid_token.metadata
    if (
        "temp_budget_increase" in valid_token_metadata
        and "temp_budget_expiry" in valid_token_metadata
    ):
        expiry = datetime.fromisoformat(valid_token_metadata["temp_budget_expiry"])
        if expiry > datetime.now():
            return valid_token_metadata["temp_budget_increase"]
    return None


def _update_key_budget_with_temp_budget_increase(
    valid_token: UserAPIKeyAuth,
) -> UserAPIKeyAuth:
    if valid_token.max_budget is None:
        return valid_token
    temp_budget_increase = _get_temp_budget_increase(valid_token) or 0.0
    valid_token.max_budget = valid_token.max_budget + temp_budget_increase
    return valid_token
