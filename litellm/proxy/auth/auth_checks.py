# What is this?
## Common auth checks between jwt + key based auth
"""
Got Valid Token from Cache, DB
Run checks for: 

1. If user can call model
2. If user is in budget 
3. If end_user ('user' passed to /chat/completions, /embeddings endpoint) is in budget 
"""
from litellm.proxy._types import (
    LiteLLM_UserTable,
    LiteLLM_EndUserTable,
    LiteLLM_JWTAuth,
    LiteLLM_TeamTable,
    LiteLLMRoutes,
)
from typing import Optional, Literal, Union
from litellm.proxy.utils import PrismaClient
from litellm.caching import DualCache
import litellm

all_routes = LiteLLMRoutes.openai_routes.value + LiteLLMRoutes.management_routes.value


def common_checks(
    request_body: dict,
    team_object: LiteLLM_TeamTable,
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
    4. If end_user ('user' passed to /chat/completions, /embeddings endpoint) is in budget
    5. [OPTIONAL] If 'enforce_end_user' enabled - did developer pass in 'user' param for openai endpoints
    6. [OPTIONAL] If 'litellm.max_budget' is set (>0), is proxy under budget
    """
    _model = request_body.get("model", None)
    if team_object.blocked == True:
        raise Exception(
            f"Team={team_object.team_id} is blocked. Update via `/team/unblock` if your admin."
        )
    # 2. If user can call model
    if (
        _model is not None
        and len(team_object.models) > 0
        and _model not in team_object.models
    ):
        # this means the team has access to all models on the proxy
        if "all-proxy-models" in team_object.models:
            # this means the team has access to all models on the proxy
            pass
        else:
            raise Exception(
                f"Team={team_object.team_id} not allowed to call model={_model}. Allowed team models = {team_object.models}"
            )
    # 3. If team is in budget
    if (
        team_object.max_budget is not None
        and team_object.spend is not None
        and team_object.spend > team_object.max_budget
    ):
        raise Exception(
            f"Team={team_object.team_id} over budget. Spend={team_object.spend}, Budget={team_object.max_budget}"
        )
    # 4. If end_user ('user' passed to /chat/completions, /embeddings endpoint) is in budget
    if end_user_object is not None and end_user_object.litellm_budget_table is not None:
        end_user_budget = end_user_object.litellm_budget_table.max_budget
        if end_user_budget is not None and end_user_object.spend > end_user_budget:
            raise Exception(
                f"ExceededBudget: End User={end_user_object.user_id} over budget. Spend={end_user_object.spend}, Budget={end_user_budget}"
            )
    # 5. [OPTIONAL] If 'enforce_user_param' enabled - did developer pass in 'user' param for openai endpoints
    if (
        general_settings.get("enforce_user_param", None) is not None
        and general_settings["enforce_user_param"] == True
    ):
        if route in LiteLLMRoutes.openai_routes.value and "user" not in request_body:
            raise Exception(
                f"'user' param not passed in. 'enforce_user_param'={general_settings['enforce_user_param']}"
            )
    # 6. [OPTIONAL] If 'litellm.max_budget' is set (>0), is proxy under budget
    if litellm.max_budget > 0 and global_proxy_spend is not None:
        if global_proxy_spend > litellm.max_budget:
            raise Exception(
                f"ExceededBudget: LiteLLM Proxy has exceeded its budget. Current spend: {global_proxy_spend}; Max Budget: {litellm.max_budget}"
            )
    return True


def _allowed_routes_check(user_route: str, allowed_routes: list) -> bool:
    for allowed_route in allowed_routes:
        if (
            allowed_route == LiteLLMRoutes.openai_routes.name
            and user_route in LiteLLMRoutes.openai_routes.value
        ):
            return True
        elif (
            allowed_route == LiteLLMRoutes.info_routes.name
            and user_route in LiteLLMRoutes.info_routes.value
        ):
            return True
        elif (
            allowed_route == LiteLLMRoutes.management_routes.name
            and user_route in LiteLLMRoutes.management_routes.value
        ):
            return True
        elif allowed_route == user_route:
            return True
    return False


def allowed_routes_check(
    user_role: Literal["proxy_admin", "team"],
    user_route: str,
    litellm_proxy_roles: LiteLLM_JWTAuth,
) -> bool:
    """
    Check if user -> not admin - allowed to access these routes
    """

    if user_role == "proxy_admin":
        if litellm_proxy_roles.admin_allowed_routes is None:
            is_allowed = _allowed_routes_check(
                user_route=user_route, allowed_routes=["management_routes"]
            )
            return is_allowed
        elif litellm_proxy_roles.admin_allowed_routes is not None:
            is_allowed = _allowed_routes_check(
                user_route=user_route,
                allowed_routes=litellm_proxy_roles.admin_allowed_routes,
            )
            return is_allowed

    elif user_role == "team":
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


async def get_end_user_object(
    end_user_id: Optional[str],
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
) -> Optional[LiteLLM_EndUserTable]:
    """
    Returns end user object, if in db.

    Do a isolated check for end user in table vs. doing a combined key + team + user + end-user check, as key might come in frequently for different end-users. Larger call will slowdown query time. This way we get to cache the constant (key/team/user info) and only update based on the changing value (end-user).
    """
    if prisma_client is None:
        raise Exception("No db connected")

    if end_user_id is None:
        return None

    # check if in cache
    cached_user_obj = user_api_key_cache.async_get_cache(key=end_user_id)
    if cached_user_obj is not None:
        if isinstance(cached_user_obj, dict):
            return LiteLLM_EndUserTable(**cached_user_obj)
        elif isinstance(cached_user_obj, LiteLLM_EndUserTable):
            return cached_user_obj
    # else, check db
    try:
        response = await prisma_client.db.litellm_endusertable.find_unique(
            where={"user_id": end_user_id}
        )

        if response is None:
            raise Exception

        return LiteLLM_EndUserTable(**response.dict())
    except Exception as e:  # if end-user not in db
        return None


async def get_user_object(self, user_id: str) -> LiteLLM_UserTable:
    """
    - Check if user id in proxy User Table
    - if valid, return LiteLLM_UserTable object with defined limits
    - if not, then raise an error
    """
    if self.prisma_client is None:
        raise Exception(
            "No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys"
        )

    # check if in cache
    cached_user_obj = self.user_api_key_cache.async_get_cache(key=user_id)
    if cached_user_obj is not None:
        if isinstance(cached_user_obj, dict):
            return LiteLLM_UserTable(**cached_user_obj)
        elif isinstance(cached_user_obj, LiteLLM_UserTable):
            return cached_user_obj
    # else, check db
    try:
        response = await self.prisma_client.db.litellm_usertable.find_unique(
            where={"user_id": user_id}
        )

        if response is None:
            raise Exception

        return LiteLLM_UserTable(**response.dict())
    except Exception as e:
        raise Exception(
            f"User doesn't exist in db. User={user_id}. Create user via `/user/new` call."
        )


async def get_team_object(
    team_id: str,
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
) -> LiteLLM_TeamTable:
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
    cached_team_obj = user_api_key_cache.async_get_cache(key=team_id)
    if cached_team_obj is not None:
        if isinstance(cached_team_obj, dict):
            return LiteLLM_TeamTable(**cached_team_obj)
        elif isinstance(cached_team_obj, LiteLLM_TeamTable):
            return cached_team_obj
    # else, check db
    try:
        response = await prisma_client.db.litellm_teamtable.find_unique(
            where={"team_id": team_id}
        )

        if response is None:
            raise Exception

        return LiteLLM_TeamTable(**response.dict())
    except Exception as e:
        raise Exception(
            f"Team doesn't exist in db. Team={team_id}. Create team via `/team/new` call."
        )
