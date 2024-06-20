"""
This file handles authentication for the LiteLLM Proxy.

it checks if the user passed a valid API Key to the LiteLLM Proxy

Returns a UserAPIKeyAuth object if the API key is valid

"""

import asyncio
import json
import secrets
import traceback
from datetime import datetime, timedelta, timezone
from typing import Optional
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
from litellm.proxy._types import *
from litellm.proxy.auth.auth_checks import (
    allowed_routes_check,
    can_key_call_model,
    common_checks,
    get_actual_routes,
    get_end_user_object,
    get_org_object,
    get_team_object,
    get_user_object,
    log_to_opentelemetry,
)
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy.utils import _to_ns

api_key_header = APIKeyHeader(
    name="Authorization", auto_error=False, description="Bearer token"
)
azure_api_key_header = APIKeyHeader(
    name="API-Key",
    auto_error=False,
    description="Some older versions of the openai Python package will send an API-Key header with just the API key ",
)


def _get_bearer_token(
    api_key: str,
):
    if api_key.startswith("Bearer "):  # ensure Bearer token passed in
        api_key = api_key.replace("Bearer ", "")  # extract the token
    else:
        api_key = ""
    return api_key


async def user_api_key_auth(
    request: Request,
    api_key: str = fastapi.Security(api_key_header),
    azure_api_key_header: str = fastapi.Security(azure_api_key_header),
) -> UserAPIKeyAuth:

    from litellm.proxy.proxy_server import (
        allowed_routes_check,
        common_checks,
        custom_db_client,
        general_settings,
        get_actual_routes,
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

    try:
        if isinstance(api_key, str):
            passed_in_key = api_key
            api_key = _get_bearer_token(api_key=api_key)

        elif isinstance(azure_api_key_header, str):
            api_key = azure_api_key_header

        parent_otel_span: Optional[Span] = None
        if open_telemetry_logger is not None:
            parent_otel_span = open_telemetry_logger.tracer.start_span(
                name="Received Proxy Server Request",
                start_time=_to_ns(datetime.now()),
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
        route: str = request.url.path

        if route in LiteLLMRoutes.public_routes.value:
            # check if public endpoint
            return UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY)

        if general_settings.get("enable_jwt_auth", False) is True:
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
                # [OPTIONAL] track spend against an internal employee - `LiteLLM_UserTable`
                user_object = None
                user_id = jwt_handler.get_user_id(
                    token=jwt_valid_token, default_value=None
                )
                if user_id is not None:
                    # get the user object
                    user_object = await get_user_object(
                        user_id=user_id,
                        prisma_client=prisma_client,
                        user_api_key_cache=user_api_key_cache,
                        user_id_upsert=jwt_handler.is_upsert_user_id(),
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
                # get the request body
                request_data = await _read_request_body(request=request)

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
            if general_settings.get("allow_user_auth", False) == True:
                return UserAPIKeyAuth()
            else:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="'allow_user_auth' not set or set to False",
                )

        ## Check END-USER OBJECT
        request_data = await _read_request_body(request=request)
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
        valid_token: Optional[UserAPIKeyAuth] = user_api_key_cache.get_cache(
            key=hash_token(api_key)
        )
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

        try:
            is_master_key_valid = secrets.compare_digest(api_key, master_key)  # type: ignore
        except Exception as e:
            is_master_key_valid = False

        ## VALIDATE MASTER KEY ##
        try:
            assert isinstance(master_key, str)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "Master key must be a valid string. Current type={}".format(
                        type(master_key)
                    )
                },
            )

        if is_master_key_valid:
            _user_api_key_obj = UserAPIKeyAuth(
                api_key=master_key,
                user_role=LitellmUserRoles.PROXY_ADMIN,
                user_id=litellm_proxy_admin_name,
                parent_otel_span=parent_otel_span,
                **end_user_params,
            )
            await user_api_key_cache.async_set_cache(
                key=hash_token(master_key), value=_user_api_key_obj
            )

            return _user_api_key_obj

        ## IF it's not a master key
        ## Route should not be in master_key_only_routes
        if route in LiteLLMRoutes.master_key_only_routes.value:
            raise Exception(
                f"Tried to access route={route}, which is only for MASTER KEY"
            )

        ## Check DB
        if isinstance(
            api_key, str
        ):  # if generated token, make sure it starts with sk-.
            assert api_key.startswith("sk-")  # prevent token hashes from being used

        if (
            prisma_client is None and custom_db_client is None
        ):  # if both master key + user key submitted, and user key != master key, and no db connected, raise an error
            raise Exception("No connected db.")

        ## check for cache hit (In-Memory Cache)
        original_api_key = api_key  # (Patch: For DynamoDB Backwards Compatibility)
        _user_role = None
        if api_key.startswith("sk-"):
            api_key = hash_token(token=api_key)
        valid_token: Optional[UserAPIKeyAuth] = user_api_key_cache.get_cache(  # type: ignore
            key=api_key
        )
        if valid_token is None:
            ## check db
            verbose_proxy_logger.debug("api key: %s", api_key)
            if prisma_client is not None:
                _valid_token: Optional[BaseModel] = await prisma_client.get_data(
                    token=api_key,
                    table_name="combined_view",
                    parent_otel_span=parent_otel_span,
                    proxy_logging_obj=proxy_logging_obj,
                )
                if _valid_token is not None:
                    valid_token = UserAPIKeyAuth(
                        **_valid_token.model_dump(exclude_none=True)
                    )
            verbose_proxy_logger.debug("Token from db: %s", valid_token)
        elif valid_token is not None and isinstance(valid_token, UserAPIKeyAuth):
            verbose_proxy_logger.debug("API Key Cache Hit!")

            # update end-user params on valid token
            # These can change per request - it's important to update them here
            valid_token.end_user_id = end_user_params.get("end_user_id")
            valid_token.end_user_tpm_limit = end_user_params.get("end_user_tpm_limit")
            valid_token.end_user_rpm_limit = end_user_params.get("end_user_rpm_limit")
            valid_token.allowed_model_region = end_user_params.get(
                "allowed_model_region"
            )

        user_id_information: Optional[List] = None
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

            # Check 1. If token can call model
            _model_alias_map = {}
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
                llm_model_list = model_list
                verbose_proxy_logger.debug(
                    f"\n new llm router model list {llm_model_list}"
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

            # Check 2. If user_id for this token is in budget
            if valid_token.user_id is not None:
                user_id_list = [valid_token.user_id]
                for id in user_id_list:
                    value = user_api_key_cache.get_cache(key=id)
                    if value is not None:
                        if user_id_information is None:
                            user_id_information = []
                        user_id_information.append(value)
                if user_id_information is None or (
                    isinstance(user_id_information, list)
                    and len(user_id_information) < 1
                ):
                    if prisma_client is not None:
                        user_id_information = await prisma_client.get_data(
                            user_id_list=[
                                valid_token.user_id,
                            ],
                            table_name="user",
                            query_type="find_all",
                        )
                        if user_id_information is not None:
                            for _id in user_id_information:
                                await user_api_key_cache.async_set_cache(
                                    key=_id["user_id"],
                                    value=_id,
                                )

                verbose_proxy_logger.debug(
                    f"user_id_information: {user_id_information}"
                )

                if user_id_information is not None:
                    if isinstance(user_id_information, list):
                        ## Check if user in budget
                        for _user in user_id_information:
                            if _user is None:
                                continue
                            assert isinstance(_user, dict)
                            # check if user is admin #

                            # Token exists, not expired now check if its in budget for the user
                            user_max_budget = _user.get("max_budget", None)
                            user_current_spend = _user.get("spend", None)

                            verbose_proxy_logger.debug(
                                f"user_id: {_user.get('user_id', None)}; user_max_budget: {user_max_budget}; user_current_spend: {user_current_spend}"
                            )

                            if (
                                user_max_budget is not None
                                and user_current_spend is not None
                            ):
                                call_info = CallInfo(
                                    token=valid_token.token,
                                    spend=user_current_spend,
                                    max_budget=user_max_budget,
                                    user_id=_user.get("user_id", None),
                                    user_email=_user.get("user_email", None),
                                    key_alias=valid_token.key_alias,
                                )
                                asyncio.create_task(
                                    proxy_logging_obj.budget_alerts(
                                        type="user_budget",
                                        user_info=call_info,
                                    )
                                )

                                _user_id = _user.get("user_id", None)
                                if user_current_spend > user_max_budget:
                                    raise litellm.BudgetExceededError(
                                        current_cost=user_current_spend,
                                        max_budget=user_max_budget,
                                    )
                    else:
                        # Token exists, not expired now check if its in budget for the user
                        user_max_budget = getattr(
                            user_id_information, "max_budget", None
                        )
                        user_current_spend = getattr(user_id_information, "spend", None)

                        if (
                            user_max_budget is not None
                            and user_current_spend is not None
                        ):
                            call_info = CallInfo(
                                token=valid_token.token,
                                spend=user_current_spend,
                                max_budget=user_max_budget,
                                user_id=getattr(user_id_information, "user_id", None),
                                user_email=getattr(
                                    user_id_information, "user_email", None
                                ),
                                key_alias=valid_token.key_alias,
                            )
                            asyncio.create_task(
                                proxy_logging_obj.budget_alerts(
                                    type="user_budget",
                                    user_info=call_info,
                                )
                            )

                            if user_current_spend > user_max_budget:
                                raise litellm.BudgetExceededError(
                                    current_cost=user_current_spend,
                                    max_budget=user_max_budget,
                                )

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
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Authentication Error - Expired Key. Key Expiry time {expiry_time} and current time {current_time}",
                    )

            # Check 4. Token Spend is under budget
            if valid_token.spend is not None and valid_token.max_budget is not None:

                ####################################
                # collect information for alerting #
                ####################################

                user_email: Optional[str] = None
                # Check if the token has any user id information
                if user_id_information is not None and len(user_id_information) > 0:
                    specific_user_id_information = user_id_information[0]
                    _user_email = specific_user_id_information.get("user_email", None)
                    if _user_email is not None:
                        user_email = str(_user_email)

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
                        model_spend[0]["model"] == current_model
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
            _team_obj = LiteLLM_TeamTable(
                team_id=valid_token.team_id,
                max_budget=valid_token.team_max_budget,
                spend=valid_token.team_spend,
                tpm_limit=valid_token.team_tpm_limit,
                rpm_limit=valid_token.team_rpm_limit,
                blocked=valid_token.team_blocked,
                models=valid_token.team_models,
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
                user_object=None,
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
            await user_api_key_cache.async_set_cache(
                key=api_key,
                value=valid_token,
            )
            valid_token_dict = valid_token.model_dump(exclude_none=True)
            valid_token_dict.pop("token", None)

            if _end_user_object is not None:
                valid_token_dict.update(end_user_params)

            _user_role = _get_user_role(user_id_information=user_id_information)

            if not _is_user_proxy_admin(user_id_information):  # if non-admin
                if route in LiteLLMRoutes.openai_routes.value:
                    pass
                elif request["route"].name in LiteLLMRoutes.openai_route_names.value:
                    pass
                elif (
                    route in LiteLLMRoutes.info_routes.value
                ):  # check if user allowed to call an info route
                    if route == "/key/info":
                        # check if user can access this route
                        query_params = request.query_params
                        key = query_params.get("key")
                        if key is not None and hash_token(token=key) != api_key:
                            raise HTTPException(
                                status_code=status.HTTP_403_FORBIDDEN,
                                detail="user not allowed to access this key's info",
                            )
                    elif route == "/user/info":
                        # check if user can access this route
                        query_params = request.query_params
                        user_id = query_params.get("user_id")
                        verbose_proxy_logger.debug(
                            f"user_id: {user_id} & valid_token.user_id: {valid_token.user_id}"
                        )
                        if user_id != valid_token.user_id:
                            raise HTTPException(
                                status_code=status.HTTP_403_FORBIDDEN,
                                detail="key not allowed to access this user's info. user_id={}, key's user_id={}".format(
                                    user_id, valid_token.user_id
                                ),
                            )
                    elif route == "/model/info":
                        # /model/info just shows models user has access to
                        pass
                    elif route == "/team/info":
                        # check if key can access this team's info
                        query_params = request.query_params
                        team_id = query_params.get("team_id")
                        if team_id != valid_token.team_id:
                            raise HTTPException(
                                status_code=status.HTTP_403_FORBIDDEN,
                                detail="key not allowed to access this team's info",
                            )
                elif (
                    _has_user_setup_sso()
                    and route in LiteLLMRoutes.sso_only_routes.value
                ):
                    pass
                elif (
                    route in LiteLLMRoutes.global_spend_tracking_routes.value
                    and getattr(valid_token, "permissions", None) is not None
                    and "get_spend_routes" in getattr(valid_token, "permissions", [])
                ):

                    pass
                elif _user_role == LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value:
                    if route in LiteLLMRoutes.openai_routes.value:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail=f"user not allowed to access this OpenAI routes, role= {_user_role}",
                        )
                    if route in LiteLLMRoutes.management_routes.value:
                        # the Admin Viewer is only allowed to call /user/update for their own user_id and can only update
                        if route == "/user/update":

                            # Check the Request params are valid for PROXY_ADMIN_VIEW_ONLY
                            if request_data is not None and isinstance(
                                request_data, dict
                            ):
                                _params_updated = request_data.keys()
                                for param in _params_updated:
                                    if param not in ["user_email", "password"]:
                                        raise HTTPException(
                                            status_code=status.HTTP_403_FORBIDDEN,
                                            detail=f"user not allowed to access this route, role= {_user_role}. Trying to access: {route} and updating invalid param: {param}. only user_email and password can be updated",
                                        )
                        else:
                            raise HTTPException(
                                status_code=status.HTTP_403_FORBIDDEN,
                                detail=f"user not allowed to access this route, role= {_user_role}. Trying to access: {route}",
                            )
                elif (
                    _user_role == LitellmUserRoles.INTERNAL_USER.value
                    and route in LiteLLMRoutes.internal_user_routes.value
                ):
                    pass
                else:
                    user_role = "unknown"
                    user_id = "unknown"
                    if (
                        user_id_information is not None
                        and isinstance(user_id_information, list)
                        and len(user_id_information) > 0
                    ):
                        _user = user_id_information[0]
                        user_role = _user.get("user_role", "unknown")
                        user_id = _user.get("user_id", "unknown")
                    raise Exception(
                        f"Only proxy admin can be used to generate, delete, update info for new keys/users/teams. Route={route}. Your role={user_role}. Your user_id={user_id}"
                    )

        # check if token is from litellm-ui, litellm ui makes keys to allow users to login with sso. These keys can only be used for LiteLLM UI functions
        # sso/login, ui/login, /key functions and /user functions
        # this will never be allowed to call /chat/completions
        token_team = getattr(valid_token, "team_id", None)

        if token_team is not None and token_team == "litellm-dashboard":
            # this token is only used for managing the ui
            allowed_routes = [
                "/sso",
                "/sso/get/logout_url",
                "/login",
                "/key/generate",
                "/key/update",
                "/key/info",
                "/config",
                "/spend",
                "/user",
                "/model/info",
                "/v2/model/info",
                "/v2/key/info",
                "/models",
                "/v1/models",
                "/global/spend",
                "/global/spend/logs",
                "/global/spend/keys",
                "/global/spend/models",
                "/global/predict/spend/logs",
                "/global/activity",
                "/health/services",
            ] + LiteLLMRoutes.info_routes.value
            # check if the current route startswith any of the allowed routes
            if (
                route is not None
                and isinstance(route, str)
                and any(
                    route.startswith(allowed_route) for allowed_route in allowed_routes
                )
            ):
                # Do something if the current route starts with any of the allowed routes
                pass
            else:
                if user_id_information is not None and _is_user_proxy_admin(
                    user_id_information
                ):
                    return UserAPIKeyAuth(
                        api_key=api_key,
                        user_role=LitellmUserRoles.PROXY_ADMIN,
                        parent_otel_span=parent_otel_span,
                        **valid_token_dict,
                    )
                elif (
                    _has_user_setup_sso()
                    and route in LiteLLMRoutes.sso_only_routes.value
                ):
                    return UserAPIKeyAuth(
                        api_key=api_key,
                        user_role=_user_role,
                        parent_otel_span=parent_otel_span,
                        **valid_token_dict,
                    )
                else:
                    raise Exception(
                        f"This key is made for LiteLLM UI, Tried to access route: {route}. Not allowed"
                    )
        if valid_token is None:
            # No token was found when looking up in the DB
            raise Exception("Invalid proxy server token passed")
        if valid_token_dict is not None:
            if user_id_information is not None and _is_user_proxy_admin(
                user_id_information
            ):
                return UserAPIKeyAuth(
                    api_key=api_key,
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                    parent_otel_span=parent_otel_span,
                    **valid_token_dict,
                )
            elif _has_user_setup_sso() and route in LiteLLMRoutes.sso_only_routes.value:
                return UserAPIKeyAuth(
                    api_key=api_key,
                    user_role=LitellmUserRoles.INTERNAL_USER,
                    parent_otel_span=parent_otel_span,
                    **valid_token_dict,
                )
            else:
                return UserAPIKeyAuth(
                    api_key=api_key,
                    user_role=LitellmUserRoles.INTERNAL_USER,
                    parent_otel_span=parent_otel_span,
                    **valid_token_dict,
                )
        else:
            raise Exception()
    except Exception as e:
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.user_api_key_auth(): Exception occured - {}\n{}".format(
                str(e), traceback.format_exc()
            )
        )

        # Log this exception to OTEL
        if open_telemetry_logger is not None:
            await open_telemetry_logger.async_post_call_failure_hook(
                original_exception=e,
                user_api_key_dict=UserAPIKeyAuth(parent_otel_span=parent_otel_span),
            )

        if isinstance(e, litellm.BudgetExceededError):
            raise ProxyException(
                message=e.message, type="auth_error", param=None, code=400
            )
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type="auth_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_401_UNAUTHORIZED),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type="auth_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_401_UNAUTHORIZED,
        )


def _has_user_setup_sso():
    """
    Check if the user has set up single sign-on (SSO) by verifying the presence of Microsoft client ID, Google client ID, and UI username environment variables.
    Returns a boolean indicating whether SSO has been set up.
    """
    microsoft_client_id = os.getenv("MICROSOFT_CLIENT_ID", None)
    google_client_id = os.getenv("GOOGLE_CLIENT_ID", None)
    ui_username = os.getenv("UI_USERNAME", None)

    sso_setup = (
        (microsoft_client_id is not None)
        or (google_client_id is not None)
        or (ui_username is not None)
    )

    return sso_setup


def _is_user_proxy_admin(user_id_information: Optional[list]):
    if user_id_information is None:
        return False

    if len(user_id_information) == 0 or user_id_information[0] is None:
        return False

    _user = user_id_information[0]
    if (
        _user.get("user_role", None) is not None
        and _user.get("user_role") == LitellmUserRoles.PROXY_ADMIN.value
    ):
        return True

    # if user_id_information contains litellm-proxy-budget
    # get first user_id that is not litellm-proxy-budget
    for user in user_id_information:
        if user.get("user_id") != "litellm-proxy-budget":
            _user = user
            break

    if (
        _user.get("user_role", None) is not None
        and _user.get("user_role") == LitellmUserRoles.PROXY_ADMIN.value
    ):
        return True

    return False


def _get_user_role(user_id_information: Optional[list]):
    if user_id_information is None:
        return None

    if len(user_id_information) == 0 or user_id_information[0] is None:
        return None

    _user = user_id_information[0]
    return _user.get("user_role")
