"""
This file handles authentication for the LiteLLM Proxy.

it checks if the user passed a valid API Key to the LiteLLM Proxy

Returns a UserAPIKeyAuth object if the API key is valid

"""

import asyncio
import json
import secrets
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from uuid import uuid4

import fastapi
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Path,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    ORJSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import litellm
from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm._service_logger import ServiceLogging
from litellm.proxy._types import *
from litellm.proxy.auth.auth_checks import (
    _cache_key_object,
    allowed_routes_check,
    can_key_call_model,
    common_checks,
    get_actual_routes,
    get_end_user_object,
    get_key_object,
    get_org_object,
    get_team_object,
    get_user_object,
    log_db_metrics,
)
from litellm.proxy.auth.auth_utils import (
    _get_request_ip_address,
    _has_user_setup_sso,
    get_request_route,
    is_pass_through_provider_route,
    pre_db_read_auth_checks,
    route_in_additonal_public_routes,
    should_run_auth_on_pass_through_provider_route,
)
from litellm.proxy.auth.oauth2_check import check_oauth2_token
from litellm.proxy.auth.oauth2_proxy_hook import handle_oauth2_proxy_request
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.service_account_checks import service_account_checks
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy.utils import _to_ns
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


async def user_api_key_auth(  # noqa: PLR0915
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
    from litellm.proxy.proxy_server import (
        general_settings,
        jwt_handler,
        litellm_proxy_admin_name,
        llm_model_list,
        master_key,
        open_telemetry_logger,
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
        user_custom_auth,
    )

    parent_otel_span: Optional[Span] = None
    start_time = datetime.now()
    try:
        route: str = get_request_route(request=request)
        # get the request body
        request_data = await _read_request_body(request=request)
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
                # check if valid token
                jwt_valid_token: dict = await jwt_handler.auth_jwt(token=api_key)
                # get scopes
                scopes = jwt_handler.get_scopes(token=jwt_valid_token)

                # check if admin
                is_admin = jwt_handler.is_admin(scopes=scopes)
                # if admin return
                if is_admin:
                    # check allowed admin routes
                    is_allowed = allowed_routes_check(
                        user_role=LitellmUserRoles.PROXY_ADMIN,
                        user_route=route,
                        litellm_proxy_roles=jwt_handler.litellm_jwtauth,
                    )
                    if is_allowed:
                        return UserAPIKeyAuth(
                            user_role=LitellmUserRoles.PROXY_ADMIN,
                            parent_otel_span=parent_otel_span,
                        )
                    else:
                        allowed_routes: List[Any] = (
                            jwt_handler.litellm_jwtauth.admin_allowed_routes
                        )
                        actual_routes = get_actual_routes(allowed_routes=allowed_routes)
                        raise Exception(
                            f"Admin not allowed to access this route. Route={route}, Allowed Routes={actual_routes}"
                        )

                # get team id
                team_id = jwt_handler.get_team_id(
                    token=jwt_valid_token, default_value=None
                )

                if team_id is None and jwt_handler.is_required_team_id() is True:
                    raise Exception(
                        f"No team id passed in. Field checked in jwt token - '{jwt_handler.litellm_jwtauth.team_id_jwt_field}'"
                    )

                team_object: Optional[LiteLLM_TeamTable] = None
                if team_id is not None:
                    # check allowed team routes
                    is_allowed = allowed_routes_check(
                        user_role=LitellmUserRoles.TEAM,
                        user_route=route,
                        litellm_proxy_roles=jwt_handler.litellm_jwtauth,
                    )
                    if is_allowed is False:
                        allowed_routes = jwt_handler.litellm_jwtauth.team_allowed_routes  # type: ignore
                        actual_routes = get_actual_routes(allowed_routes=allowed_routes)
                        raise Exception(
                            f"Team not allowed to access this route. Route={route}, Allowed Routes={actual_routes}"
                        )

                    # check if team in db
                    team_object = await get_team_object(
                        team_id=team_id,
                        prisma_client=prisma_client,
                        user_api_key_cache=user_api_key_cache,
                        parent_otel_span=parent_otel_span,
                        proxy_logging_obj=proxy_logging_obj,
                    )

                # [OPTIONAL] track spend for an org id - `LiteLLM_OrganizationTable`
                org_id = jwt_handler.get_org_id(
                    token=jwt_valid_token, default_value=None
                )
                if org_id is not None:
                    _ = await get_org_object(
                        org_id=org_id,
                        prisma_client=prisma_client,
                        user_api_key_cache=user_api_key_cache,
                        parent_otel_span=parent_otel_span,
                        proxy_logging_obj=proxy_logging_obj,
                    )
                # [OPTIONAL] allowed user email domains
                valid_user_email: Optional[bool] = None
                user_email: Optional[str] = None
                if jwt_handler.is_enforced_email_domain():
                    """
                    if 'allowed_email_subdomains' is set,

                    - checks if token contains 'email' field
                    - checks if 'email' is from an allowed domain
                    """
                    user_email = jwt_handler.get_user_email(
                        token=jwt_valid_token, default_value=None
                    )
                    if user_email is None:
                        valid_user_email = False
                    else:
                        valid_user_email = jwt_handler.is_allowed_domain(
                            user_email=user_email
                        )

                # [OPTIONAL] track spend against an internal employee - `LiteLLM_UserTable`
                user_object = None
                user_id = jwt_handler.get_user_id(
                    token=jwt_valid_token, default_value=user_email
                )
                if user_id is not None:
                    # get the user object
                    user_object = await get_user_object(
                        user_id=user_id,
                        prisma_client=prisma_client,
                        user_api_key_cache=user_api_key_cache,
                        user_id_upsert=jwt_handler.is_upsert_user_id(
                            valid_user_email=valid_user_email
                        ),
                        parent_otel_span=parent_otel_span,
                        proxy_logging_obj=proxy_logging_obj,
                    )
                # [OPTIONAL] track spend against an external user - `LiteLLM_EndUserTable`
                end_user_object = None
                end_user_id = jwt_handler.get_end_user_id(
                    token=jwt_valid_token, default_value=None
                )
                if end_user_id is not None:
                    # get the end-user object
                    end_user_object = await get_end_user_object(
                        end_user_id=end_user_id,
                        prisma_client=prisma_client,
                        user_api_key_cache=user_api_key_cache,
                        parent_otel_span=parent_otel_span,
                        proxy_logging_obj=proxy_logging_obj,
                    )

                global_proxy_spend = None
                if litellm.max_budget > 0:  # user set proxy max budget
                    # check cache
                    global_proxy_spend = await user_api_key_cache.async_get_cache(
                        key="{}:spend".format(litellm_proxy_admin_name)
                    )
                    if global_proxy_spend is None and prisma_client is not None:
                        # get from db
                        sql_query = """SELECT SUM(spend) as total_spend FROM "MonthlyGlobalSpend";"""

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
                            token=jwt_valid_token["token"],
                        )
                        asyncio.create_task(
                            proxy_logging_obj.budget_alerts(
                                type="proxy_budget",
                                user_info=user_info,
                            )
                        )
                # run through common checks
                _ = common_checks(
                    request_body=request_data,
                    team_object=team_object,
                    user_object=user_object,
                    end_user_object=end_user_object,
                    general_settings=general_settings,
                    global_proxy_spend=global_proxy_spend,
                    route=route,
                )

                # return UserAPIKeyAuth object
                return UserAPIKeyAuth(
                    api_key=None,
                    team_id=team_object.team_id if team_object is not None else None,
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
        if "user" in request_data:
            try:
                _end_user_object = await get_end_user_object(
                    end_user_id=request_data["user"],
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
                        end_user_params["end_user_id"] = _end_user_object.user_id
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
            valid_token.end_user_id = end_user_params.get("end_user_id")
            valid_token.end_user_tpm_limit = end_user_params.get("end_user_tpm_limit")
            valid_token.end_user_rpm_limit = end_user_params.get("end_user_rpm_limit")
            valid_token.allowed_model_region = end_user_params.get(
                "allowed_model_region"
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
            _user_api_key_obj = _return_user_api_key_auth_obj(
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
            await _cache_key_object(
                hashed_token=hash_token(master_key),
                user_api_key_obj=_user_api_key_obj,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
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
            raise Exception("No connected db.")

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
            if (
                len(valid_token.models) == 0
            ):  # assume an empty model list means all models are allowed to be called
                pass
            elif (
                isinstance(valid_token.models, list)
                and "all-team-models" in valid_token.models
            ):
                # Do not do any validation at this step
                # the validation will occur when checking the team has access to this model
                pass
            else:
                try:
                    data = await request.json()
                except json.JSONDecodeError:
                    data = {}  # Provide a default value, such as an empty dictionary
                model = data.get("model", None)
                fallback_models: Optional[List[str]] = data.get("fallbacks", None)

                if model is not None:
                    await can_key_call_model(
                        model=model,
                        llm_model_list=llm_model_list,
                        valid_token=valid_token,
                    )

                if fallback_models is not None:
                    for m in fallback_models:
                        await can_key_call_model(
                            model=m,
                            llm_model_list=llm_model_list,
                            valid_token=valid_token,
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
            if valid_token.spend is not None and valid_token.max_budget is not None:

                ####################################
                # collect information for alerting #
                ####################################

                user_email = None
                # Check if the token has any user id information
                if user_obj is not None:
                    user_email = user_obj.user_email

                call_info = CallInfo(
                    token=valid_token.token,
                    spend=valid_token.spend,
                    max_budget=valid_token.max_budget,
                    user_id=valid_token.user_id,
                    team_id=valid_token.team_id,
                    user_email=user_email,
                    key_alias=valid_token.key_alias,
                )
                asyncio.create_task(
                    proxy_logging_obj.budget_alerts(
                        type="token_budget",
                        user_info=call_info,
                    )
                )

                ####################################
                # collect information for alerting #
                ####################################

                if valid_token.spend >= valid_token.max_budget:
                    raise litellm.BudgetExceededError(
                        current_cost=valid_token.spend,
                        max_budget=valid_token.max_budget,
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
                twenty_eight_days_ago = datetime.now() - timedelta(days=28)
                model_spend = await prisma_client.db.litellm_spendlogs.group_by(
                    by=["model"],
                    sum={"spend": True},
                    where={
                        "AND": [
                            {"api_key": valid_token.token},
                            {"startTime": {"gt": twenty_eight_days_ago}},
                            {"model": current_model},
                        ]
                    },  # type: ignore
                )
                if (
                    len(model_spend) > 0
                    and max_budget_per_model.get(current_model, None) is not None
                ):
                    if (
                        "model" in model_spend[0]
                        and model_spend[0].get("model") == current_model
                        and "_sum" in model_spend[0]
                        and "spend" in model_spend[0]["_sum"]
                        and model_spend[0]["_sum"]["spend"]
                        >= max_budget_per_model[current_model]
                    ):
                        current_model_spend = model_spend[0]["_sum"]["spend"]
                        current_model_budget = max_budget_per_model[current_model]
                        raise litellm.BudgetExceededError(
                            current_cost=current_model_spend,
                            max_budget=current_model_budget,
                        )

            # Check 6. Team spend is under Team budget
            if (
                hasattr(valid_token, "team_spend")
                and valid_token.team_spend is not None
                and hasattr(valid_token, "team_max_budget")
                and valid_token.team_max_budget is not None
            ):
                call_info = CallInfo(
                    token=valid_token.token,
                    spend=valid_token.team_spend,
                    max_budget=valid_token.team_max_budget,
                    user_id=valid_token.user_id,
                    team_id=valid_token.team_id,
                    team_alias=valid_token.team_alias,
                )
                asyncio.create_task(
                    proxy_logging_obj.budget_alerts(
                        type="team_budget",
                        user_info=call_info,
                    )
                )

                if valid_token.team_spend >= valid_token.team_max_budget:
                    raise litellm.BudgetExceededError(
                        current_cost=valid_token.team_spend,
                        max_budget=valid_token.team_max_budget,
                    )

            # Check 8: Additional Common Checks across jwt + key auth
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

            # Check 9: Check if key is a service account key
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
            _ = common_checks(
                request_body=request_data,
                team_object=_team_obj,
                user_object=user_obj,
                end_user_object=_end_user_object,
                general_settings=general_settings,
                global_proxy_spend=global_proxy_spend,
                route=route,
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
            return _return_user_api_key_auth_obj(
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

        # Log this exception to OTEL
        if open_telemetry_logger is not None:
            await open_telemetry_logger.async_post_call_failure_hook(  # type: ignore
                original_exception=e,
                request_data={},
                user_api_key_dict=UserAPIKeyAuth(parent_otel_span=parent_otel_span),
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


def _return_user_api_key_auth_obj(
    user_obj: Optional[LiteLLM_UserTable],
    api_key: str,
    parent_otel_span: Optional[Span],
    valid_token_dict: dict,
    route: str,
    start_time: datetime,
    user_role: Optional[LitellmUserRoles] = None,
) -> UserAPIKeyAuth:
    end_time = datetime.now()
    user_api_key_service_logger_obj.service_success_hook(
        service=ServiceTypes.AUTH,
        call_type=route,
        start_time=start_time,
        end_time=end_time,
        duration=end_time.timestamp() - start_time.timestamp(),
        parent_otel_span=parent_otel_span,
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
):
    # use this as the virtual key passed to litellm proxy
    custom_litellm_key_header_name = custom_litellm_key_header_name.lower()
    verbose_proxy_logger.debug(
        "searching for custom_litellm_key_header_name= %s, in headers=%s",
        custom_litellm_key_header_name,
        request.headers,
    )
    custom_api_key = request.headers.get(custom_litellm_key_header_name)
    if custom_api_key:
        api_key = _get_bearer_token(api_key=custom_api_key)
        verbose_proxy_logger.debug(
            "Found custom API key using header: {}, setting api_key={}".format(
                custom_litellm_key_header_name, api_key
            )
        )
    else:
        raise ValueError(
            f"No LiteLLM Virtual Key pass. Please set header={custom_litellm_key_header_name}: Bearer <api_key>"
        )
    return api_key
