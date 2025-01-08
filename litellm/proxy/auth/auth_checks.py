# What is this?
## Common auth checks between jwt + key based auth
"""
Got Valid Token from Cache, DB
Run checks for: 

1. If user can call model
2. If user is in budget 
3. If end_user ('user' passed to /chat/completions, /embeddings endpoint) is in budget 
"""

import inspect
import time
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from pydantic import BaseModel

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.caching.dual_cache import LimitedSizeOrderedDict
from litellm.proxy._types import (
    DB_CONNECTION_ERROR_TYPES,
    CommonProxyErrors,
    LiteLLM_EndUserTable,
    LiteLLM_JWTAuth,
    LiteLLM_OrganizationTable,
    LiteLLM_TeamTable,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
    LiteLLMRoutes,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.utils import PrismaClient, ProxyLogging, log_db_metrics
from litellm.router import Router
from litellm.types.services import ServiceTypes

from .auth_checks_organization import organization_role_based_access_check

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = _Span
else:
    Span = Any


last_db_access_time = LimitedSizeOrderedDict(max_size=100)
db_cache_expiry = 5  # refresh every 5s

all_routes = LiteLLMRoutes.openai_routes.value + LiteLLMRoutes.management_routes.value


def _allowed_import_check() -> bool:
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    # Get the calling frame
    caller_frame = inspect.stack()[2]
    caller_function = caller_frame.function
    caller_function_callable = caller_frame.frame.f_globals.get(caller_function)

    allowed_function = "user_api_key_auth"
    allowed_signature = inspect.signature(user_api_key_auth)
    if caller_function_callable is None or not callable(caller_function_callable):
        raise Exception(f"Caller function {caller_function} is not callable")
    caller_signature = inspect.signature(caller_function_callable)

    if caller_signature != allowed_signature:
        raise TypeError(
            f"The function '{caller_function}' does not match the required signature of 'user_api_key_auth'. {CommonProxyErrors.not_premium_user.value}"
        )
    # Check if the caller module is allowed
    if caller_function != allowed_function:
        raise ImportError(
            f"This function can only be imported by '{allowed_function}'. {CommonProxyErrors.not_premium_user.value}"
        )

    return True


def common_checks(  # noqa: PLR0915
    request_body: dict,
    team_object: Optional[LiteLLM_TeamTable],
    user_object: Optional[LiteLLM_UserTable],
    end_user_object: Optional[LiteLLM_EndUserTable],
    global_proxy_spend: Optional[float],
    general_settings: dict,
    route: str,
    llm_router: Optional[Router],
) -> bool:
    """
    Common checks across jwt + key-based auth.

    1. If team is blocked
    2. If team can call model
    3. If team is in budget
    4. If user passed in (JWT or key.user_id) - is in budget
    5. If end_user (either via JWT or 'user' passed to /chat/completions, /embeddings endpoint) is in budget
    6. [OPTIONAL] If 'enforce_end_user' enabled - did developer pass in 'user' param for openai endpoints
    7. [OPTIONAL] If 'litellm.max_budget' is set (>0), is proxy under budget
    8. [OPTIONAL] If guardrails modified - is request allowed to change this
    9. Check if request body is safe
    10. [OPTIONAL] Organization checks - is user_object.organization_id is set, run these checks
    """
    _allowed_import_check()
    _model = request_body.get("model", None)
    if team_object is not None and team_object.blocked is True:
        raise Exception(
            f"Team={team_object.team_id} is blocked. Update via `/team/unblock` if your admin."
        )
    # 2. If team can call model
    if (
        _model is not None
        and team_object is not None
        and team_object.models is not None
        and len(team_object.models) > 0
        and _model not in team_object.models
    ):
        # this means the team has access to all models on the proxy
        if (
            "all-proxy-models" in team_object.models
            or "*" in team_object.models
            or "openai/*" in team_object.models
        ):
            # this means the team has access to all models on the proxy
            pass
        # check if the team model is an access_group
        elif (
            model_in_access_group(
                model=_model, team_models=team_object.models, llm_router=llm_router
            )
            is True
        ):
            pass
        elif _model and "*" in _model:
            pass
        else:
            raise Exception(
                f"Team={team_object.team_id} not allowed to call model={_model}. Allowed team models = {team_object.models}"
            )
    # 3. If team is in budget
    if (
        team_object is not None
        and team_object.max_budget is not None
        and team_object.spend is not None
        and team_object.spend > team_object.max_budget
    ):
        raise litellm.BudgetExceededError(
            current_cost=team_object.spend,
            max_budget=team_object.max_budget,
            message=f"Team={team_object.team_id} over budget. Spend={team_object.spend}, Budget={team_object.max_budget}",
        )
    # 4. If user is in budget
    ## 4.1 check personal budget, if personal key
    if (
        (team_object is None or team_object.team_id is None)
        and user_object is not None
        and user_object.max_budget is not None
    ):
        user_budget = user_object.max_budget
        if user_budget < user_object.spend:
            raise litellm.BudgetExceededError(
                current_cost=user_object.spend,
                max_budget=user_budget,
                message=f"ExceededBudget: User={user_object.user_id} over budget. Spend={user_object.spend}, Budget={user_budget}",
            )
    ## 4.2 check team member budget, if team key
    # 5. If end_user ('user' passed to /chat/completions, /embeddings endpoint) is in budget
    if end_user_object is not None and end_user_object.litellm_budget_table is not None:
        end_user_budget = end_user_object.litellm_budget_table.max_budget
        if end_user_budget is not None and end_user_object.spend > end_user_budget:
            raise litellm.BudgetExceededError(
                current_cost=end_user_object.spend,
                max_budget=end_user_budget,
                message=f"ExceededBudget: End User={end_user_object.user_id} over budget. Spend={end_user_object.spend}, Budget={end_user_budget}",
            )
    # 6. [OPTIONAL] If 'enforce_user_param' enabled - did developer pass in 'user' param for openai endpoints
    if (
        general_settings.get("enforce_user_param", None) is not None
        and general_settings["enforce_user_param"] is True
    ):
        if RouteChecks.is_llm_api_route(route=route) and "user" not in request_body:
            raise Exception(
                f"'user' param not passed in. 'enforce_user_param'={general_settings['enforce_user_param']}"
            )
    # 7. [OPTIONAL] If 'litellm.max_budget' is set (>0), is proxy under budget
    if (
        litellm.max_budget > 0
        and global_proxy_spend is not None
        # only run global budget checks for OpenAI routes
        # Reason - the Admin UI should continue working if the proxy crosses it's global budget
        and RouteChecks.is_llm_api_route(route=route)
        and route != "/v1/models"
        and route != "/models"
    ):
        if global_proxy_spend > litellm.max_budget:
            raise litellm.BudgetExceededError(
                current_cost=global_proxy_spend, max_budget=litellm.max_budget
            )

    _request_metadata: dict = request_body.get("metadata", {}) or {}
    if _request_metadata.get("guardrails"):
        # check if team allowed to modify guardrails
        from litellm.proxy.guardrails.guardrail_helpers import can_modify_guardrails

        can_modify: bool = can_modify_guardrails(team_object)
        if can_modify is False:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Your team does not have permission to modify guardrails."
                },
            )

    # 10 [OPTIONAL] Organization RBAC checks
    organization_role_based_access_check(
        user_object=user_object, route=route, request_body=request_body
    )

    return True


def _allowed_routes_check(user_route: str, allowed_routes: list) -> bool:
    """
    Return if a user is allowed to access route. Helper function for `allowed_routes_check`.

    Parameters:
    - user_route: str - the route the user is trying to call
    - allowed_routes: List[str|LiteLLMRoutes] - the list of allowed routes for the user.
    """
    for allowed_route in allowed_routes:
        if (
            allowed_route in LiteLLMRoutes.__members__
            and user_route in LiteLLMRoutes[allowed_route].value
        ):
            return True
        elif allowed_route == user_route:
            return True
    return False


def allowed_routes_check(
    user_role: Literal[
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.TEAM,
        LitellmUserRoles.INTERNAL_USER,
    ],
    user_route: str,
    litellm_proxy_roles: LiteLLM_JWTAuth,
) -> bool:
    """
    Check if user -> not admin - allowed to access these routes
    """

    if user_role == LitellmUserRoles.PROXY_ADMIN:
        is_allowed = _allowed_routes_check(
            user_route=user_route,
            allowed_routes=litellm_proxy_roles.admin_allowed_routes,
        )
        return is_allowed

    elif user_role == LitellmUserRoles.TEAM:
        if litellm_proxy_roles.team_allowed_routes is None:
            """
            By default allow a team to call openai + info routes
            """
            is_allowed = _allowed_routes_check(
                user_route=user_route, allowed_routes=["openai_routes", "info_routes"]
            )
            return is_allowed
        elif litellm_proxy_roles.team_allowed_routes is not None:
            is_allowed = _allowed_routes_check(
                user_route=user_route,
                allowed_routes=litellm_proxy_roles.team_allowed_routes,
            )
            return is_allowed
    return False


def allowed_route_check_inside_route(
    user_api_key_dict: UserAPIKeyAuth,
    requested_user_id: Optional[str],
) -> bool:
    ret_val = True
    if (
        user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN
        and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY
    ):
        ret_val = False
    if requested_user_id is not None and user_api_key_dict.user_id is not None:
        if user_api_key_dict.user_id == requested_user_id:
            ret_val = True
    return ret_val


def get_actual_routes(allowed_routes: list) -> list:
    actual_routes: list = []
    for route_name in allowed_routes:
        try:
            route_value = LiteLLMRoutes[route_name].value
            actual_routes = actual_routes + route_value
        except KeyError:
            actual_routes.append(route_name)
    return actual_routes


@log_db_metrics
async def get_end_user_object(
    end_user_id: Optional[str],
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
    parent_otel_span: Optional[Span] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
) -> Optional[LiteLLM_EndUserTable]:
    """
    Returns end user object, if in db.

    Do a isolated check for end user in table vs. doing a combined key + team + user + end-user check, as key might come in frequently for different end-users. Larger call will slowdown query time. This way we get to cache the constant (key/team/user info) and only update based on the changing value (end-user).
    """
    if prisma_client is None:
        raise Exception("No db connected")

    if end_user_id is None:
        return None
    _key = "end_user_id:{}".format(end_user_id)

    def check_in_budget(end_user_obj: LiteLLM_EndUserTable):
        if end_user_obj.litellm_budget_table is None:
            return
        end_user_budget = end_user_obj.litellm_budget_table.max_budget
        if end_user_budget is not None and end_user_obj.spend > end_user_budget:
            raise litellm.BudgetExceededError(
                current_cost=end_user_obj.spend, max_budget=end_user_budget
            )

    # check if in cache
    cached_user_obj = await user_api_key_cache.async_get_cache(key=_key)
    if cached_user_obj is not None:
        if isinstance(cached_user_obj, dict):
            return_obj = LiteLLM_EndUserTable(**cached_user_obj)
            check_in_budget(end_user_obj=return_obj)
            return return_obj
        elif isinstance(cached_user_obj, LiteLLM_EndUserTable):
            return_obj = cached_user_obj
            check_in_budget(end_user_obj=return_obj)
            return return_obj
    # else, check db
    try:
        response = await prisma_client.db.litellm_endusertable.find_unique(
            where={"user_id": end_user_id},
            include={"litellm_budget_table": True},
        )

        if response is None:
            raise Exception

        # save the end-user object to cache
        await user_api_key_cache.async_set_cache(
            key="end_user_id:{}".format(end_user_id), value=response
        )

        _response = LiteLLM_EndUserTable(**response.dict())

        check_in_budget(end_user_obj=_response)

        return _response
    except Exception as e:  # if end-user not in db
        if isinstance(e, litellm.BudgetExceededError):
            raise e
        return None


def model_in_access_group(
    model: str, team_models: Optional[List[str]], llm_router: Optional[Router]
) -> bool:
    from collections import defaultdict

    if team_models is None:
        return True
    if model in team_models:
        return True

    access_groups: dict[str, list[str]] = defaultdict(list)
    if llm_router:
        access_groups = llm_router.get_model_access_groups(model_name=model)

    if len(access_groups) > 0:  # check if token contains any model access groups
        for idx, m in enumerate(
            team_models
        ):  # loop token models, if any of them are an access group add the access group
            if m in access_groups:
                return True

    # Filter out models that are access_groups
    filtered_models = [m for m in team_models if m not in access_groups]

    if model in filtered_models:
        return True

    return False


def _should_check_db(
    key: str, last_db_access_time: LimitedSizeOrderedDict, db_cache_expiry: int
) -> bool:
    """
    Prevent calling db repeatedly for items that don't exist in the db.
    """
    current_time = time.time()
    # if key doesn't exist in last_db_access_time -> check db
    if key not in last_db_access_time:
        return True
    elif (
        last_db_access_time[key][0] is not None
    ):  # check db for non-null values (for refresh operations)
        return True
    elif last_db_access_time[key][0] is None:
        if current_time - last_db_access_time[key] >= db_cache_expiry:
            return True
    return False


def _update_last_db_access_time(
    key: str, value: Optional[Any], last_db_access_time: LimitedSizeOrderedDict
):
    last_db_access_time[key] = (value, time.time())


@log_db_metrics
async def get_user_object(
    user_id: str,
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
    user_id_upsert: bool,
    parent_otel_span: Optional[Span] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
) -> Optional[LiteLLM_UserTable]:
    """
    - Check if user id in proxy User Table
    - if valid, return LiteLLM_UserTable object with defined limits
    - if not, then raise an error
    """

    if user_id is None:
        return None

    # check if in cache
    cached_user_obj = await user_api_key_cache.async_get_cache(key=user_id)
    if cached_user_obj is not None:
        if isinstance(cached_user_obj, dict):
            return LiteLLM_UserTable(**cached_user_obj)
        elif isinstance(cached_user_obj, LiteLLM_UserTable):
            return cached_user_obj
    # else, check db
    if prisma_client is None:
        raise Exception("No db connected")
    try:
        db_access_time_key = "user_id:{}".format(user_id)
        should_check_db = _should_check_db(
            key=db_access_time_key,
            last_db_access_time=last_db_access_time,
            db_cache_expiry=db_cache_expiry,
        )

        if should_check_db:
            response = await prisma_client.db.litellm_usertable.find_unique(
                where={"user_id": user_id}, include={"organization_memberships": True}
            )
        else:
            response = None

        if response is None:
            if user_id_upsert:
                response = await prisma_client.db.litellm_usertable.create(
                    data={"user_id": user_id},
                    include={"organization_memberships": True},
                )
            else:
                raise Exception

        if (
            response.organization_memberships is not None
            and len(response.organization_memberships) > 0
        ):
            # dump each organization membership to type LiteLLM_OrganizationMembershipTable
            _dumped_memberships = [
                membership.model_dump()
                for membership in response.organization_memberships
                if membership is not None
            ]
            response.organization_memberships = _dumped_memberships

        _response = LiteLLM_UserTable(**dict(response))
        response_dict = _response.model_dump()

        # save the user object to cache
        await user_api_key_cache.async_set_cache(key=user_id, value=response_dict)

        # save to db access time
        _update_last_db_access_time(
            key=db_access_time_key,
            value=response_dict,
            last_db_access_time=last_db_access_time,
        )

        return _response
    except Exception as e:  # if user not in db
        raise ValueError(
            f"User doesn't exist in db. 'user_id'={user_id}. Create user via `/user/new` call. Got error - {e}"
        )


async def _cache_management_object(
    key: str,
    value: BaseModel,
    user_api_key_cache: DualCache,
    proxy_logging_obj: Optional[ProxyLogging],
):
    await user_api_key_cache.async_set_cache(key=key, value=value)


async def _cache_team_object(
    team_id: str,
    team_table: LiteLLM_TeamTableCachedObj,
    user_api_key_cache: DualCache,
    proxy_logging_obj: Optional[ProxyLogging],
):
    key = "team_id:{}".format(team_id)

    ## CACHE REFRESH TIME!
    team_table.last_refreshed_at = time.time()

    await _cache_management_object(
        key=key,
        value=team_table,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )


async def _cache_key_object(
    hashed_token: str,
    user_api_key_obj: UserAPIKeyAuth,
    user_api_key_cache: DualCache,
    proxy_logging_obj: Optional[ProxyLogging],
):
    key = hashed_token

    ## CACHE REFRESH TIME
    user_api_key_obj.last_refreshed_at = time.time()

    await _cache_management_object(
        key=key,
        value=user_api_key_obj,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )


async def _delete_cache_key_object(
    hashed_token: str,
    user_api_key_cache: DualCache,
    proxy_logging_obj: Optional[ProxyLogging],
):
    key = hashed_token

    user_api_key_cache.delete_cache(key=key)

    ## UPDATE REDIS CACHE ##
    if proxy_logging_obj is not None:
        await proxy_logging_obj.internal_usage_cache.dual_cache.async_delete_cache(
            key=key
        )


@log_db_metrics
async def _get_team_db_check(team_id: str, prisma_client: PrismaClient):
    return await prisma_client.db.litellm_teamtable.find_unique(
        where={"team_id": team_id}
    )


async def _get_team_object_from_db(team_id: str, prisma_client: PrismaClient):
    return await prisma_client.db.litellm_teamtable.find_unique(
        where={"team_id": team_id}
    )


async def _get_team_object_from_user_api_key_cache(
    team_id: str,
    prisma_client: PrismaClient,
    user_api_key_cache: DualCache,
    last_db_access_time: LimitedSizeOrderedDict,
    db_cache_expiry: int,
    proxy_logging_obj: Optional[ProxyLogging],
    key: str,
) -> LiteLLM_TeamTableCachedObj:
    db_access_time_key = key
    should_check_db = _should_check_db(
        key=db_access_time_key,
        last_db_access_time=last_db_access_time,
        db_cache_expiry=db_cache_expiry,
    )
    if should_check_db:
        response = await _get_team_db_check(
            team_id=team_id, prisma_client=prisma_client
        )
    else:
        response = None

    if response is None:
        raise Exception

    _response = LiteLLM_TeamTableCachedObj(**response.dict())
    # save the team object to cache
    await _cache_team_object(
        team_id=team_id,
        team_table=_response,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )

    # save to db access time
    # save to db access time
    _update_last_db_access_time(
        key=db_access_time_key,
        value=_response,
        last_db_access_time=last_db_access_time,
    )

    return _response


async def _get_team_object_from_cache(
    key: str,
    proxy_logging_obj: Optional[ProxyLogging],
    user_api_key_cache: DualCache,
    parent_otel_span: Optional[Span],
) -> Optional[LiteLLM_TeamTableCachedObj]:
    cached_team_obj: Optional[LiteLLM_TeamTableCachedObj] = None

    ## CHECK REDIS CACHE ##
    if (
        proxy_logging_obj is not None
        and proxy_logging_obj.internal_usage_cache.dual_cache
    ):

        cached_team_obj = (
            await proxy_logging_obj.internal_usage_cache.dual_cache.async_get_cache(
                key=key, parent_otel_span=parent_otel_span
            )
        )

    if cached_team_obj is None:
        cached_team_obj = await user_api_key_cache.async_get_cache(key=key)

    if cached_team_obj is not None:
        if isinstance(cached_team_obj, dict):
            return LiteLLM_TeamTableCachedObj(**cached_team_obj)
        elif isinstance(cached_team_obj, LiteLLM_TeamTableCachedObj):
            return cached_team_obj

    return None


async def get_team_object(
    team_id: str,
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
    parent_otel_span: Optional[Span] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
    check_cache_only: Optional[bool] = None,
    check_db_only: Optional[bool] = None,
) -> LiteLLM_TeamTableCachedObj:
    """
    - Check if team id in proxy Team Table
    - if valid, return LiteLLM_TeamTable object with defined limits
    - if not, then raise an error
    """
    if prisma_client is None:
        raise Exception(
            "No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys"
        )

    # check if in cache
    key = "team_id:{}".format(team_id)

    if not check_db_only:
        cached_team_obj = await _get_team_object_from_cache(
            key=key,
            proxy_logging_obj=proxy_logging_obj,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=parent_otel_span,
        )

        if cached_team_obj is not None:
            return cached_team_obj

        if check_cache_only:
            raise Exception(
                f"Team doesn't exist in cache + check_cache_only=True. Team={team_id}."
            )

    # else, check db
    try:
        return await _get_team_object_from_user_api_key_cache(
            team_id=team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
            last_db_access_time=last_db_access_time,
            db_cache_expiry=db_cache_expiry,
            key=key,
        )
    except Exception:
        raise Exception(
            f"Team doesn't exist in db. Team={team_id}. Create team via `/team/new` call."
        )


@log_db_metrics
async def get_key_object(
    hashed_token: str,
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
    parent_otel_span: Optional[Span] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
    check_cache_only: Optional[bool] = None,
) -> UserAPIKeyAuth:
    """
    - Check if team id in proxy Team Table
    - if valid, return LiteLLM_TeamTable object with defined limits
    - if not, then raise an error
    """
    if prisma_client is None:
        raise Exception(
            "No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys"
        )

    # check if in cache
    key = hashed_token

    cached_key_obj: Optional[UserAPIKeyAuth] = await user_api_key_cache.async_get_cache(
        key=key
    )

    if cached_key_obj is not None:
        if isinstance(cached_key_obj, dict):
            return UserAPIKeyAuth(**cached_key_obj)
        elif isinstance(cached_key_obj, UserAPIKeyAuth):
            return cached_key_obj

    if check_cache_only:
        raise Exception(
            f"Key doesn't exist in cache + check_cache_only=True. key={key}."
        )

    # else, check db
    try:
        _valid_token: Optional[BaseModel] = await prisma_client.get_data(
            token=hashed_token,
            table_name="combined_view",
            parent_otel_span=parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
        )

        if _valid_token is None:
            raise Exception

        _response = UserAPIKeyAuth(**_valid_token.model_dump(exclude_none=True))

        # save the key object to cache
        await _cache_key_object(
            hashed_token=hashed_token,
            user_api_key_obj=_response,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

        return _response
    except DB_CONNECTION_ERROR_TYPES as e:
        return await _handle_failed_db_connection_for_get_key_object(e=e)
    except Exception:
        traceback.print_exc()
        raise Exception(
            f"Key doesn't exist in db. key={hashed_token}. Create key via `/key/generate` call."
        )


async def _handle_failed_db_connection_for_get_key_object(
    e: Exception,
) -> UserAPIKeyAuth:
    """
    Handles httpx.ConnectError when reading a Virtual Key from LiteLLM DB

    Use this if you don't want failed DB queries to block LLM API reqiests

    Returns:
        - UserAPIKeyAuth: If general_settings.allow_requests_on_db_unavailable is True

    Raises:
        - Orignal Exception in all other cases
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        litellm_proxy_admin_name,
        proxy_logging_obj,
    )

    # If this flag is on, requests failing to connect to the DB will be allowed
    if general_settings.get("allow_requests_on_db_unavailable", False) is True:
        # log this as a DB failure on prometheus
        proxy_logging_obj.service_logging_obj.service_failure_hook(
            service=ServiceTypes.DB,
            call_type="get_key_object",
            error=e,
            duration=0.0,
        )

        return UserAPIKeyAuth(
            key_name="failed-to-connect-to-db",
            token="failed-to-connect-to-db",
            user_id=litellm_proxy_admin_name,
        )
    else:
        # raise the original exception, the wrapper on `get_key_object` handles logging db failure to prometheus
        raise e


@log_db_metrics
async def get_org_object(
    org_id: str,
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
    parent_otel_span: Optional[Span] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
):
    """
    - Check if org id in proxy Org Table
    - if valid, return LiteLLM_OrganizationTable object
    - if not, then raise an error
    """
    if prisma_client is None:
        raise Exception(
            "No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys"
        )

    # check if in cache
    cached_org_obj = user_api_key_cache.async_get_cache(key="org_id:{}".format(org_id))
    if cached_org_obj is not None:
        if isinstance(cached_org_obj, dict):
            return cached_org_obj
        elif isinstance(cached_org_obj, LiteLLM_OrganizationTable):
            return cached_org_obj
    # else, check db
    try:
        response = await prisma_client.db.litellm_organizationtable.find_unique(
            where={"organization_id": org_id}
        )

        if response is None:
            raise Exception

        return response
    except Exception:
        raise Exception(
            f"Organization doesn't exist in db. Organization={org_id}. Create organization via `/organization/new` call."
        )


async def can_key_call_model(
    model: str,
    llm_model_list: Optional[list],
    valid_token: UserAPIKeyAuth,
    llm_router: Optional[litellm.Router],
) -> Literal[True]:
    """
    Checks if token can call a given model

    Returns:
        - True: if token allowed to call model

    Raises:
        - Exception: If token not allowed to call model
    """
    if model in litellm.model_alias_map:
        model = litellm.model_alias_map[model]

    ## check if model in allowed model names
    verbose_proxy_logger.debug(
        f"LLM Model List pre access group check: {llm_model_list}"
    )
    from collections import defaultdict

    access_groups: Dict[str, List[str]] = defaultdict(list)
    if llm_router:
        access_groups = llm_router.get_model_access_groups(model_name=model)
    if (
        len(access_groups) > 0 and llm_router is not None
    ):  # check if token contains any model access groups
        for idx, m in enumerate(
            valid_token.models
        ):  # loop token models, if any of them are an access group add the access group
            if m in access_groups:
                return True

    # Filter out models that are access_groups
    filtered_models = [m for m in valid_token.models if m not in access_groups]

    verbose_proxy_logger.debug(f"model: {model}; allowed_models: {filtered_models}")

    all_model_access: bool = False

    if (
        len(filtered_models) == 0 and len(valid_token.models) == 0
    ) or "*" in filtered_models:
        all_model_access = True

    if model is not None and model not in filtered_models and all_model_access is False:
        raise ValueError(
            f"API Key not allowed to access model. This token can only access models={valid_token.models}. Tried to access {model}"
        )
    valid_token.models = filtered_models
    verbose_proxy_logger.debug(
        f"filtered allowed_models: {filtered_models}; valid_token.models: {valid_token.models}"
    )
    return True
