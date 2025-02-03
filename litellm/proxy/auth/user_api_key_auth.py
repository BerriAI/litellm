"""
This file handles authentication for the LiteLLM Proxy.

it checks if the user passed a valid API Key to the LiteLLM Proxy

Returns a UserAPIKeyAuth object if the API key is valid

"""

import asyncio
import secrets
from datetime import datetime, timezone
from typing import Optional, cast

import fastapi
from fastapi import HTTPException, Request, WebSocket, status
from fastapi.security.api_key import APIKeyHeader

import litellm
from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm._service_logger import ServiceLogging
from litellm.caching import DualCache
from litellm.proxy._types import *
from litellm.proxy.auth.auth_checks import (
    _cache_key_object,
    _handle_failed_db_connection_for_get_key_object,
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
from litellm.proxy.auth.auth_utils import (
    _get_request_ip_address,
    get_end_user_id_from_request_body,
    get_request_route,
    is_pass_through_provider_route,
    pre_db_read_auth_checks,
    route_in_additonal_public_routes,
    should_run_auth_on_pass_through_provider_route,
)
from litellm.proxy.auth.handle_jwt import JWTAuthManager, JWTHandler
from litellm.proxy.auth.oauth2_check import check_oauth2_token
from litellm.proxy.auth.oauth2_proxy_hook import handle_oauth2_proxy_request
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.service_account_checks import service_account_checks
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy.utils import PrismaClient, ProxyLogging, _to_ns
from litellm.types.services import ServiceTypes

user_api_key_service_logger_obj = ServiceLogging()  # used for tracking latency on OTEL


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


def _get_bearer_token(
    api_key: str,
):
    if api_key.startswith("Bearer "):  # ensure Bearer token passed in
        api_key = api_key.replace("Bearer ", "")  # extract the token
    elif api_key.startswith("Basic "):
        api_key = api_key.replace("Basic ", "")  # handle langfuse input
    elif api_key.startswith("bearer "):
        api_key = api_key.replace("bearer ", "")
    else:
        api_key = ""
    return api_key


def _is_ui_route(
    route: str,
    user_obj: Optional[LiteLLM_UserTable] = None,
) -> bool:
    """
    - Check if the route is a UI used route
    """
    # this token is only used for managing the ui
    allowed_routes = LiteLLMRoutes.ui_routes.value
    # check if the current route startswith any of the allowed routes
    if (
        route is not None
        and isinstance(route, str)
        and any(route.startswith(allowed_route) for allowed_route in allowed_routes)
    ):
        # Do something if the current route starts with any of the allowed routes
        return True
    elif any(
        RouteChecks._route_matches_pattern(route=route, pattern=allowed_route)
        for allowed_route in allowed_routes
    ):
        return True
    return False


def _is_api_route_allowed(
    route: str,
    request: Request,
    request_data: dict,
    api_key: str,
    valid_token: Optional[UserAPIKeyAuth],
    user_obj: Optional[LiteLLM_UserTable] = None,
) -> bool:
    """
    - Route b/w api token check and normal token check
    """
    _user_role = _get_user_role(user_obj=user_obj)

    if valid_token is None:
        raise Exception("Invalid proxy server token passed. valid_token=None.")

    if not _is_user_proxy_admin(user_obj=user_obj):  # if non-admin
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=user_obj,
            _user_role=_user_role,
            route=route,
            request=request,
            request_data=request_data,
            api_key=api_key,
            valid_token=valid_token,
        )
    return True


def _is_allowed_route(
    route: str,
    token_type: Literal["ui", "api"],
    request: Request,
    request_data: dict,
    api_key: str,
    valid_token: Optional[UserAPIKeyAuth],
    user_obj: Optional[LiteLLM_UserTable] = None,
) -> bool:
    """
    - Route b/w ui token check and normal token check
    """

    if token_type == "ui" and _is_ui_route(route=route, user_obj=user_obj):
        return True
    else:
        return _is_api_route_allowed(
            route=route,
            request=request,
            request_data=request_data,
            api_key=api_key,
            valid_token=valid_token,
            user_obj=user_obj,
        )


async def user_api_key_auth_websocket(websocket: WebSocket):
    # Accept the WebSocket connection

    request = Request(scope={"type": "http"})
    request._url = websocket.url

    query_params = websocket.query_params

    model = query_params.get("model")

    async def return_body():
        return_string = f'{{"model": "{model}"}}'
        # return string as bytes
        return return_string.encode()

    request.body = return_body  # type: ignore

    # Extract the Authorization header
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


async def get_global_proxy_spend(
    litellm_proxy_admin_name: str,
    user_api_key_cache: DualCache,
    prisma_client: Optional[PrismaClient],
    token: str,
    proxy_logging_obj: ProxyLogging,
) -> Optional[float]:
    global_proxy_spend = None
    if litellm.max_budget > 0:  # user set proxy max budget
        # check cache
        global_proxy_spend = await user_api_key_cache.async_get_cache(
            key="{}:spend".format(litellm_proxy_admin_name)
        )
        if global_proxy_spend is None and prisma_client is not None:
            # get from db
            sql_query = (
                """SELECT SUM(spend) as total_spend FROM "MonthlyGlobalSpend";"""
            )

            response = await prisma_client.db.query_raw(query=sql_query)

            global_proxy_spend = response[0]["total_spend"]

            await user_api_key_cache.async_set_cache(
                key="{}:spend".format(litellm_proxy_admin_name),
                value=global_proxy_spend,
            )
        if global_proxy_spend is not None:
            user_info = CallInfo(
                user_id=litellm_proxy_admin_name,
                max_budget=litellm.max_budget,
                spend=global_proxy_spend,
                token=token,
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


async def _user_api_key_auth_builder(  # noqa: PLR0915
    request: Request,
    api_key: str,
    azure_api_key_header: str,
    anthropic_api_key_header: Optional[str],
    google_ai_studio_api_key_header: Optional[str],
    request_data: dict,
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
        passed_in_key: Optional[str] = None
        if isinstance(api_key, str):
            passed_in_key = api_key
            api_key = _get_bearer_token(api_key=api_key)
        elif isinstance(azure_api_key_header, str):
            api_key = azure_api_key_header
        elif isinstance(anthropic_api_key_header, str):
            api_key = anthropic_api_key_header
        elif isinstance(google_ai_studio_api_key_header, str):
            api_key = google_ai_studio_api_key_header
        elif pass_through_endpoints is not None:
            for endpoint in pass_through_endpoints:
                if endpoint.get("path", "") == route:
                    headers: Optional[dict] = endpoint.get("headers", None)
                    if headers is not None:
                        header_key: str = headers.get("litellm_user_api_key", "")
                        if request.headers.get(key=header_key) is not None:
                            api_key = request.headers.get(key=header_key)

        # if user wants to pass LiteLLM_Master_Key as a custom header, example pass litellm keys as X-LiteLLM-Key: Bearer sk-1234
        custom_litellm_key_header_name = general_settings.get("litellm_key_header_name")
        if custom_litellm_key_header_name is not None:
            api_key = get_api_key_from_custom_header(
                request=request,
                custom_litellm_key_header_name=custom_litellm_key_header_name,
            )

        if open_telemetry_logger is not None:

            parent_otel_span = open_telemetry_logger.tracer.start_span(
                name="Received Proxy Server Request",
                start_time=_to_ns(start_time),
                context=open_telemetry_logger.get_traceparent_from_header(
                    headers=request.headers
                ),
                kind=open_telemetry_logger.span_kind.SERVER,
            )

        ### USER-DEFINED AUTH FUNCTION ###
        if user_custom_auth is not None:
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
        elif is_pass_through_provider_route(route=route):
            if should_run_auth_on_pass_through_provider_route(route=route) is False:
                return UserAPIKeyAuth(
                    user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
                )

        ########## End of Route Checks Before Reading DB / Cache for "token" ########

        if general_settings.get("enable_oauth2_auth", False) is True:
            # return UserAPIKeyAuth object
            # helper to check if the api_key is a valid oauth2 token
            from litellm.proxy.proxy_server import premium_user

            if premium_user is not True:
                raise ValueError(
                    "Oauth2 token validation is only available for premium users"
                    + CommonProxyErrors.not_premium_user.value
                )

            return await check_oauth2_token(token=api_key)

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

                global_proxy_spend = await get_global_proxy_spend(
                    litellm_proxy_admin_name=litellm_proxy_admin_name,
                    user_api_key_cache=user_api_key_cache,
                    prisma_client=prisma_client,
                    token=token,
                    proxy_logging_obj=proxy_logging_obj,
                )

                if is_proxy_admin:
                    return UserAPIKeyAuth(
                        user_role=LitellmUserRoles.PROXY_ADMIN,
                        parent_otel_span=parent_otel_span,
                    )
                # run through common checks
                _ = await common_checks(
                    request_body=request_data,
                    team_object=team_object,
                    user_object=user_object,
                    end_user_object=end_user_object,
                    general_settings=general_settings,
                    global_proxy_spend=global_proxy_spend,
                    route=route,
                    llm_router=llm_router,
                    proxy_logging_obj=proxy_logging_obj,
                    valid_token=None,
                )

                # return UserAPIKeyAuth object
                return UserAPIKeyAuth(
                    api_key=None,
                    team_id=team_id,
                    team_tpm_limit=(
                        team_object.tpm_limit if team_object is not None else None
                    ),
                    team_rpm_limit=(
                        team_object.rpm_limit if team_object is not None else None
                    ),
                    team_models=team_object.models if team_object is not None else [],
                    user_role=LitellmUserRoles.INTERNAL_USER,
                    user_id=user_id,
                    org_id=org_id,
                    parent_otel_span=parent_otel_span,
                    end_user_id=end_user_id,
                )

        #### ELSE ####
        ## CHECK PASS-THROUGH ENDPOINTS ##
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
                f"Malformed API Key passed in. Ensure Key has `Bearer ` prefix. Passed in: {passed_in_key}"
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

        end_user_id = get_end_user_id_from_request_body(request_data)
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
                )
                if _end_user_object is not None:
                    end_user_params["allowed_model_region"] = (
                        _end_user_object.allowed_model_region
                    )
                    if _end_user_object.litellm_budget_table is not None:
                        budget_info = _end_user_object.litellm_budget_table
                        if budget_info.tpm_limit is not None:
                            end_user_params["end_user_tpm_limit"] = (
                                budget_info.tpm_limit
                            )
                        if budget_info.rpm_limit is not None:
                            end_user_params["end_user_rpm_limit"] = (
                                budget_info.rpm_limit
                            )
                        if budget_info.max_budget is not None:
                            end_user_params["end_user_max_budget"] = (
                                budget_info.max_budget
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

        if (
            valid_token is not None
            and isinstance(valid_token, UserAPIKeyAuth)
            and valid_token.user_role == LitellmUserRoles.PROXY_ADMIN
        ):
            # update end-user params on valid token
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
        if isinstance(
            api_key, str
        ):  # if generated token, make sure it starts with sk-.
            assert api_key.startswith(
                "sk-"
            ), "LiteLLM Virtual Key expected. Received={}, expected to start with 'sk-'.".format(
                api_key
            )  # prevent token hashes from being used
        else:
            verbose_logger.warning(
                "litellm.proxy.proxy_server.user_api_key_auth(): Warning - Key={} is not a string.".format(
                    api_key
                )
            )

        if (
            prisma_client is None
        ):  # if both master key + user key submitted, and user key != master key, and no db connected, raise an error
            return await _handle_failed_db_connection_for_get_key_object(
                e=Exception("No connected db.")
            )

        ## check for cache hit (In-Memory Cache)
        _user_role = None
        if api_key.startswith("sk-"):
            api_key = hash_token(token=api_key)

        if valid_token is None:
            try:
                valid_token = await get_key_object(
                    hashed_token=api_key,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    parent_otel_span=parent_otel_span,
                    proxy_logging_obj=proxy_logging_obj,
                )
                # update end-user params on valid token
                # These can change per request - it's important to update them here
                valid_token.end_user_id = end_user_params.get("end_user_id")
                valid_token.end_user_tpm_limit = end_user_params.get(
                    "end_user_tpm_limit"
                )
                valid_token.end_user_rpm_limit = end_user_params.get(
                    "end_user_rpm_limit"
                )
                valid_token.allowed_model_region = end_user_params.get(
                    "allowed_model_region"
                )
                # update key budget with temp budget increase
                valid_token = _update_key_budget_with_temp_budget_increase(
                    valid_token
                )  # updating it here, allows all downstream reporting / checks to use the updated budget
            except Exception:
                verbose_logger.info(
                    "litellm.proxy.auth.user_api_key_auth.py::user_api_key_auth() - Unable to find token={} in cache or `LiteLLM_VerificationTokenTable`. Defaulting 'valid_token' to None'".format(
                        api_key
                    )
                )
                valid_token = None

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

            # Check 1. If token can call model
            _model_alias_map = {}
            model: Optional[str] = None
            if (
                hasattr(valid_token, "team_model_aliases")
                and valid_token.team_model_aliases is not None
            ):
                _model_alias_map = {
                    **valid_token.aliases,
                    **valid_token.team_model_aliases,
                }
            else:
                _model_alias_map = {**valid_token.aliases}
            litellm.model_alias_map = _model_alias_map
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
                model = request_data.get("model", None)
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

            # Check 3. Check if user is in their team budget
            if valid_token.team_member_spend is not None:
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
                        param=api_key,
                    )

            # Check 4. Token Spend is under budget
            await _virtual_key_max_budget_check(
                valid_token=valid_token,
                proxy_logging_obj=proxy_logging_obj,
                user_obj=user_obj,
            )

            # Check 5. Soft Budget Check
            await _virtual_key_soft_budget_check(
                valid_token=valid_token,
                proxy_logging_obj=proxy_logging_obj,
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
                _team_obj: Optional[LiteLLM_TeamTable] = LiteLLM_TeamTable(
                    team_id=valid_token.team_id,
                    max_budget=valid_token.team_max_budget,
                    spend=valid_token.team_spend,
                    tpm_limit=valid_token.team_tpm_limit,
                    rpm_limit=valid_token.team_rpm_limit,
                    blocked=valid_token.team_blocked,
                    models=valid_token.team_models,
                    metadata=valid_token.team_metadata,
                )
            else:
                _team_obj = None

            # Check 7: Check if key is a service account key
            await service_account_checks(
                valid_token=valid_token,
                request_data=request_data,
            )

            user_api_key_cache.set_cache(
                key=valid_token.team_id, value=_team_obj
            )  # save team table in cache - used for tpm/rpm limiting - tpm_rpm_limiter.py

            global_proxy_spend = None
            if (
                litellm.max_budget > 0 and prisma_client is not None
            ):  # user set proxy max budget
                # check cache
                global_proxy_spend = await user_api_key_cache.async_get_cache(
                    key="{}:spend".format(litellm_proxy_admin_name)
                )
                if global_proxy_spend is None:
                    # get from db
                    sql_query = """SELECT SUM(spend) as total_spend FROM "MonthlyGlobalSpend";"""

                    response = await prisma_client.db.query_raw(query=sql_query)

                    global_proxy_spend = response[0]["total_spend"]
                    await user_api_key_cache.async_set_cache(
                        key="{}:spend".format(litellm_proxy_admin_name),
                        value=global_proxy_spend,
                    )

                if global_proxy_spend is not None:
                    call_info = CallInfo(
                        token=valid_token.token,
                        spend=global_proxy_spend,
                        max_budget=litellm.max_budget,
                        user_id=litellm_proxy_admin_name,
                        team_id=valid_token.team_id,
                    )
                    asyncio.create_task(
                        proxy_logging_obj.budget_alerts(
                            type="proxy_budget",
                            user_info=call_info,
                        )
                    )
            _ = await common_checks(
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
        token_team = getattr(valid_token, "team_id", None)
        token_type: Literal["ui", "api"] = (
            "ui"
            if token_team is not None and token_team == "litellm-dashboard"
            else "api"
        )
        _is_route_allowed = _is_allowed_route(
            route=route,
            token_type=token_type,
            user_obj=user_obj,
            request=request,
            request_data=request_data,
            api_key=api_key,
            valid_token=valid_token,
        )
        if not _is_route_allowed:
            raise HTTPException(401, detail="Invalid route for UI token")

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
        else:
            raise Exception()
    except Exception as e:
        requester_ip = _get_request_ip_address(
            request=request,
            use_x_forwarded_for=general_settings.get("use_x_forwarded_for", False),
        )
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.user_api_key_auth(): Exception occured - {}\nRequester IP Address:{}".format(
                str(e),
                requester_ip,
            ),
            extra={"requester_ip": requester_ip},
        )

        # Log this exception to OTEL, Datadog etc
        user_api_key_dict = UserAPIKeyAuth(
            parent_otel_span=parent_otel_span,
            api_key=api_key,
        )
        asyncio.create_task(
            proxy_logging_obj.post_call_failure_hook(
                request_data=request_data,
                original_exception=e,
                user_api_key_dict=user_api_key_dict,
                error_type=ProxyErrorTypes.auth_error,
                route=route,
            )
        )

        if isinstance(e, litellm.BudgetExceededError):
            raise ProxyException(
                message=e.message,
                type=ProxyErrorTypes.budget_exceeded,
                param=None,
                code=400,
            )
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type=ProxyErrorTypes.auth_error,
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_401_UNAUTHORIZED),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type=ProxyErrorTypes.auth_error,
            param=getattr(e, "param", "None"),
            code=status.HTTP_401_UNAUTHORIZED,
        )


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
) -> UserAPIKeyAuth:
    """
    Parent function to authenticate user api key / jwt token.
    """

    request_data = await _read_request_body(request=request)

    user_api_key_auth_obj = await _user_api_key_auth_builder(
        request=request,
        api_key=api_key,
        azure_api_key_header=azure_api_key_header,
        anthropic_api_key_header=anthropic_api_key_header,
        google_ai_studio_api_key_header=google_ai_studio_api_key_header,
        request_data=request_data,
    )

    end_user_id = get_end_user_id_from_request_body(request_data)
    if end_user_id is not None:
        user_api_key_auth_obj.end_user_id = end_user_id

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
        )
    if user_obj is not None and _is_user_proxy_admin(user_obj=user_obj):
        user_api_key_kwargs.update(
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )
        return UserAPIKeyAuth(**user_api_key_kwargs)
    else:
        return UserAPIKeyAuth(**user_api_key_kwargs)


def _is_user_proxy_admin(user_obj: Optional[LiteLLM_UserTable]):
    if user_obj is None:
        return False

    if (
        user_obj.user_role is not None
        and user_obj.user_role == LitellmUserRoles.PROXY_ADMIN.value
    ):
        return True

    if (
        user_obj.user_role is not None
        and user_obj.user_role == LitellmUserRoles.PROXY_ADMIN.value
    ):
        return True

    return False


def _get_user_role(
    user_obj: Optional[LiteLLM_UserTable],
) -> Optional[LitellmUserRoles]:
    if user_obj is None:
        return None

    _user = user_obj

    _user_role = _user.user_role
    try:
        role = LitellmUserRoles(_user_role)
    except ValueError:
        return LitellmUserRoles.INTERNAL_USER

    return role


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
                custom_litellm_key_header_name, api_key
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
