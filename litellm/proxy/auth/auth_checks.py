# What is this?
## Common auth checks between jwt + key based auth
"""
Got Valid Token from Cache, DB
Run checks for: 

1. If user can call model
2. If user is in budget 
3. If end_user ('user' passed to /chat/completions, /embeddings endpoint) is in budget 
"""
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Literal, Optional

from pydantic import BaseModel

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching import DualCache
from litellm.proxy._types import (
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
from litellm.proxy.auth.route_checks import is_llm_api_route
from litellm.proxy.utils import PrismaClient, ProxyLogging, log_to_opentelemetry
from litellm.types.services import ServiceLoggerPayload, ServiceTypes

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = _Span
else:
    Span = Any

all_routes = LiteLLMRoutes.openai_routes.value + LiteLLMRoutes.management_routes.value


def common_checks(
    request_body: dict,
    team_object: Optional[LiteLLM_TeamTable],
    user_object: Optional[LiteLLM_UserTable],
    end_user_object: Optional[LiteLLM_EndUserTable],
    global_proxy_spend: Optional[float],
    general_settings: dict,
    route: str,
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
    """
    _model = request_body.get("model", None)
    if team_object is not None and team_object.blocked is True:
        raise Exception(
            f"Team={team_object.team_id} is blocked. Update via `/team/unblock` if your admin."
        )
    # 2. If team can call model
    if (
        _model is not None
        and team_object is not None
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
        elif model_in_access_group(_model, team_object.models) is True:
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
        if is_llm_api_route(route=route) and "user" not in request_body:
            raise Exception(
                f"'user' param not passed in. 'enforce_user_param'={general_settings['enforce_user_param']}"
            )
    if general_settings.get("enforced_params", None) is not None:
        # Enterprise ONLY Feature
        # we already validate if user is premium_user when reading the config
        # Add an extra premium_usercheck here too, just incase
        from litellm.proxy.proxy_server import CommonProxyErrors, premium_user

        if premium_user is not True:
            raise ValueError(
                "Trying to use `enforced_params`"
                + CommonProxyErrors.not_premium_user.value
            )

        if is_llm_api_route(route=route):
            # loop through each enforced param
            # example enforced_params ['user', 'metadata', 'metadata.generation_name']
            for enforced_param in general_settings["enforced_params"]:
                _enforced_params = enforced_param.split(".")
                if len(_enforced_params) == 1:
                    if _enforced_params[0] not in request_body:
                        raise ValueError(
                            f"BadRequest please pass param={_enforced_params[0]} in request body. This is a required param"
                        )
                elif len(_enforced_params) == 2:
                    # this is a scenario where user requires request['metadata']['generation_name'] to exist
                    if _enforced_params[0] not in request_body:
                        raise ValueError(
                            f"BadRequest please pass param={_enforced_params[0]} in request body. This is a required param"
                        )
                    if _enforced_params[1] not in request_body[_enforced_params[0]]:
                        raise ValueError(
                            f"BadRequest please pass param=[{_enforced_params[0]}][{_enforced_params[1]}] in request body. This is a required param"
                        )

        pass
    # 7. [OPTIONAL] If 'litellm.max_budget' is set (>0), is proxy under budget
    if (
        litellm.max_budget > 0
        and global_proxy_spend is not None
        # only run global budget checks for OpenAI routes
        # Reason - the Admin UI should continue working if the proxy crosses it's global budget
        and is_llm_api_route(route=route)
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


def get_actual_routes(allowed_routes: list) -> list:
    actual_routes: list = []
    for route_name in allowed_routes:
        try:
            route_value = LiteLLMRoutes[route_name].value
            actual_routes = actual_routes + route_value
        except KeyError:
            actual_routes.append(route_name)
    return actual_routes


@log_to_opentelemetry
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


def model_in_access_group(model: str, team_models: Optional[List[str]]) -> bool:
    from collections import defaultdict

    from litellm.proxy.proxy_server import llm_router

    if team_models is None:
        return True
    if model in team_models:
        return True

    access_groups = defaultdict(list)
    if llm_router:
        access_groups = llm_router.get_model_access_groups()

    models_in_current_access_groups = []
    if len(access_groups) > 0:  # check if token contains any model access groups
        for idx, m in enumerate(
            team_models
        ):  # loop token models, if any of them are an access group add the access group
            if m in access_groups:
                # if it is an access group we need to remove it from valid_token.models
                models_in_group = access_groups[m]
                models_in_current_access_groups.extend(models_in_group)

    # Filter out models that are access_groups
    filtered_models = [m for m in team_models if m not in access_groups]
    filtered_models += models_in_current_access_groups

    if model in filtered_models:
        return True
    return False


@log_to_opentelemetry
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

        response = await prisma_client.db.litellm_usertable.find_unique(
            where={"user_id": user_id}
        )

        if response is None:
            if user_id_upsert:
                response = await prisma_client.db.litellm_usertable.create(
                    data={"user_id": user_id}
                )
            else:
                raise Exception

        _response = LiteLLM_UserTable(**dict(response))
        response_dict = _response.model_dump()

        # save the user object to cache
        await user_api_key_cache.async_set_cache(key=user_id, value=response_dict)

        return _response
    except Exception:  # if user not in db
        raise ValueError(
            f"User doesn't exist in db. 'user_id'={user_id}. Create user via `/user/new` call."
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


@log_to_opentelemetry
async def get_team_object(
    team_id: str,
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
    parent_otel_span: Optional[Span] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
    check_cache_only: Optional[bool] = None,
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
    cached_team_obj: Optional[LiteLLM_TeamTableCachedObj] = None

    ## CHECK REDIS CACHE ##
    if (
        proxy_logging_obj is not None
        and proxy_logging_obj.internal_usage_cache.dual_cache
    ):
        cached_team_obj = (
            await proxy_logging_obj.internal_usage_cache.dual_cache.async_get_cache(
                key=key
            )
        )

    if cached_team_obj is None:
        cached_team_obj = await user_api_key_cache.async_get_cache(key=key)

    if cached_team_obj is not None:
        if isinstance(cached_team_obj, dict):
            return LiteLLM_TeamTableCachedObj(**cached_team_obj)
        elif isinstance(cached_team_obj, LiteLLM_TeamTableCachedObj):
            return cached_team_obj

    if check_cache_only:
        raise Exception(
            f"Team doesn't exist in cache + check_cache_only=True. Team={team_id}."
        )

    # else, check db
    try:
        response = await prisma_client.db.litellm_teamtable.find_unique(
            where={"team_id": team_id}
        )

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

        return _response
    except Exception:
        raise Exception(
            f"Team doesn't exist in db. Team={team_id}. Create team via `/team/new` call."
        )


@log_to_opentelemetry
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
    cached_team_obj: Optional[UserAPIKeyAuth] = None

    if cached_team_obj is None:
        cached_team_obj = await user_api_key_cache.async_get_cache(key=key)

    if cached_team_obj is not None:
        if isinstance(cached_team_obj, dict):
            return UserAPIKeyAuth(**cached_team_obj)
        elif isinstance(cached_team_obj, UserAPIKeyAuth):
            return cached_team_obj

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
    except Exception:
        raise Exception(
            f"Key doesn't exist in db. key={hashed_token}. Create key via `/key/generate` call."
        )


@log_to_opentelemetry
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
    model: str, llm_model_list: Optional[list], valid_token: UserAPIKeyAuth
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

    from litellm.proxy.proxy_server import llm_router

    access_groups = defaultdict(list)
    if llm_router:
        access_groups = llm_router.get_model_access_groups()

    models_in_current_access_groups = []
    if len(access_groups) > 0:  # check if token contains any model access groups
        for idx, m in enumerate(
            valid_token.models
        ):  # loop token models, if any of them are an access group add the access group
            if m in access_groups:
                # if it is an access group we need to remove it from valid_token.models
                models_in_group = access_groups[m]
                models_in_current_access_groups.extend(models_in_group)

    # Filter out models that are access_groups
    filtered_models = [m for m in valid_token.models if m not in access_groups]

    filtered_models += models_in_current_access_groups
    verbose_proxy_logger.debug(f"model: {model}; allowed_models: {filtered_models}")

    all_model_access: bool = False

    if (
        len(filtered_models) == 0
        or "*" in filtered_models
        or "openai/*" in filtered_models
    ):
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
