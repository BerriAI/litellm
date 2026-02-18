# What is this?
## Common auth checks between jwt + key based auth
"""
Got Valid Token from Cache, DB
Run checks for:

1. If user can call model
2. If user is in budget
3. If end_user ('user' passed to /chat/completions, /embeddings endpoint) is in budget
"""
import asyncio
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union, cast

from fastapi import HTTPException, Request, status
from pydantic import BaseModel

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.caching.dual_cache import LimitedSizeOrderedDict
from litellm.constants import (
    CLI_JWT_EXPIRATION_HOURS,
    CLI_JWT_TOKEN_NAME,
    DEFAULT_ACCESS_GROUP_CACHE_TTL,
    DEFAULT_IN_MEMORY_TTL,
    DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
    DEFAULT_MAX_RECURSE_DEPTH,
    EMAIL_BUDGET_ALERT_MAX_SPEND_ALERT_PERCENTAGE,
)
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.proxy._types import (
    RBAC_ROLES,
    CallInfo,
    LiteLLM_AccessGroupTable,
    LiteLLM_BudgetTable,
    LiteLLM_EndUserTable,
    Litellm_EntityType,
    LiteLLM_JWTAuth,
    LiteLLM_ObjectPermissionTable,
    LiteLLM_OrganizationMembershipTable,
    LiteLLM_OrganizationTable,
    LiteLLM_TagTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
    LiteLLMRoutes,
    LitellmUserRoles,
    NewTeamRequest,
    ProxyErrorTypes,
    ProxyException,
    RoleBasedPermissions,
    SpecialModelNames,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.route_llm_request import route_request
from litellm.proxy.utils import PrismaClient, ProxyLogging, log_db_metrics
from litellm.router import Router
from litellm.utils import get_utc_datetime

from .auth_checks_organization import organization_role_based_access_check
from .auth_utils import get_model_from_request

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = Union[_Span, Any]
else:
    Span = Any


last_db_access_time = LimitedSizeOrderedDict(max_size=100)
db_cache_expiry = DEFAULT_IN_MEMORY_TTL  # refresh every 5s

all_routes = LiteLLMRoutes.openai_routes.value + LiteLLMRoutes.management_routes.value

def _log_budget_lookup_failure(entity: str, error: Exception) -> None:
    """
    Log a warning when budget lookup fails; cache will not be populated.

    Skips logging for expected "user not found" cases (bare Exception from
    get_user_object when user_id_upsert=False). Adds a schema migration hint
    when the error appears schema-related.
    """
    # Skip logging for expected "user not found" - not caching is correct
    if str(error) == "" and type(error).__name__ == "Exception":
        return
    err_str = str(error).lower()
    hint = ""
    if any(
        x in err_str
        for x in ("column", "schema", "does not exist", "prisma", "migrate")
    ):
        hint = " Run `prisma db push` or `prisma migrate deploy` to fix schema mismatches."
    verbose_proxy_logger.error(
        f"Budget lookup failed for {entity}; cache will not be populated. "
        f"Each request will hit the database. Error: {error}.{hint}"
    )

def _is_model_cost_zero(
    model: Optional[Union[str, List[str]]], llm_router: Optional[Router]
) -> bool:
    """
    Check if a model has zero cost (no configured pricing).
    
    Uses the router's get_model_group_info method to get pricing information.
    
    Args:
        model: The model name or list of model names
        llm_router: The LiteLLM router instance
    
    Returns:
        bool: True if all costs for the model are zero, False otherwise
    """
    if model is None or llm_router is None:
        return False
    
    # Handle list of models
    model_list = [model] if isinstance(model, str) else model
    
    for model_name in model_list:
        try:
            # Use router's get_model_group_info method directly for better reliability
            model_group_info = llm_router.get_model_group_info(model_group=model_name)
            
            if model_group_info is None:
                # Model not found or no pricing info available
                # Conservative approach: assume it has cost
                verbose_proxy_logger.debug(
                    f"No model group info found for {model_name}, assuming it has cost"
                )
                return False
            
            # Check costs for this model
            # Only allow bypass if BOTH costs are explicitly set to 0 (not None)
            input_cost = model_group_info.input_cost_per_token
            output_cost = model_group_info.output_cost_per_token
            
            # If costs are not explicitly configured (None), assume it has cost
            if input_cost is None or output_cost is None:
                verbose_proxy_logger.debug(
                    f"Model {model_name} has undefined cost (input: {input_cost}, output: {output_cost}), assuming it has cost"
                )
                return False
            
            # If either cost is non-zero, return False
            if input_cost > 0 or output_cost > 0:
                verbose_proxy_logger.debug(
                    f"Model {model_name} has non-zero cost (input: {input_cost}, output: {output_cost})"
                )
                return False
            
            # This model has zero cost explicitly configured
            verbose_proxy_logger.debug(
                f"Model {model_name} has zero cost explicitly configured (input: {input_cost}, output: {output_cost})"
            )
            
        except Exception as e:
            # If we can't determine the cost, assume it has cost (conservative approach)
            verbose_proxy_logger.debug(
                f"Error checking cost for model {model_name}: {str(e)}, assuming it has cost"
            )
            return False
    
    # All models checked have zero cost
    return True


async def common_checks(
    request_body: dict,
    team_object: Optional[LiteLLM_TeamTable],
    user_object: Optional[LiteLLM_UserTable],
    end_user_object: Optional[LiteLLM_EndUserTable],
    global_proxy_spend: Optional[float],
    general_settings: dict,
    route: str,
    llm_router: Optional[Router],
    proxy_logging_obj: ProxyLogging,
    valid_token: Optional[UserAPIKeyAuth],
    request: Request,
    skip_budget_checks: bool = False,
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
    11. [OPTIONAL] Vector store checks - is the object allowed to access the vector store
    """
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    _model: Optional[Union[str, List[str]]] = get_model_from_request(
        request_body, route
    )

    # 1. If team is blocked
    if team_object is not None and team_object.blocked is True:
        raise Exception(
            f"Team={team_object.team_id} is blocked. Update via `/team/unblock` if your admin."
        )

    # 2. If team can call model
    if _model and team_object:
        if not await can_team_access_model(
            model=_model,
            team_object=team_object,
            llm_router=llm_router,
            team_model_aliases=valid_token.team_model_aliases if valid_token else None,
        ):
            raise ProxyException(
                message=f"Team not allowed to access model. Team={team_object.team_id}, Model={_model}. Allowed team models = {team_object.models}",
                type=ProxyErrorTypes.team_model_access_denied,
                param="model",
                code=status.HTTP_401_UNAUTHORIZED,
            )

    ## 2.1 If user can call model (if personal key)
    if _model and team_object is None and user_object is not None:
        await can_user_call_model(
            model=_model,
            llm_router=llm_router,
            user_object=user_object,
        )

    # If this is a free model, skip all budget checks
    if not skip_budget_checks:
        # 3. If team is in budget
        await _team_max_budget_check(
            team_object=team_object,
            proxy_logging_obj=proxy_logging_obj,
            valid_token=valid_token,
        )

        # 3.0.5. If team is over soft budget (alert only, doesn't block)
        await _team_soft_budget_check(
            team_object=team_object,
            proxy_logging_obj=proxy_logging_obj,
            valid_token=valid_token,
        )

        # 3.1. If organization is in budget
        await _organization_max_budget_check(
            valid_token=valid_token,
            team_object=team_object,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

        await _tag_max_budget_check(
            request_body=request_body,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
            valid_token=valid_token,
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
        await _check_team_member_budget(
            team_object=team_object,
            user_object=user_object,
            valid_token=valid_token,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

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
        # Get HTTP method from request
        http_method = request.method if hasattr(request, "method") else None

        # Check if it's a POST request and if it's an OpenAI route but not MCP
        is_post_method = http_method and http_method.upper() == "POST"
        is_openai_route = RouteChecks.is_llm_api_route(route=route)
        is_mcp_route = (
            route in LiteLLMRoutes.mcp_routes.value
            or RouteChecks.check_route_access(
                route=route, allowed_routes=LiteLLMRoutes.mcp_routes.value
            )
        )

        # Enforce user param only for POST requests on OpenAI routes (excluding MCP routes)
        if (
            is_post_method
            and is_openai_route
            and not is_mcp_route
            and "user" not in request_body
        ):
            raise Exception(
                f"'user' param not passed in. 'enforce_user_param'={general_settings['enforce_user_param']}"
            )

    # 6.1 [OPTIONAL] If 'reject_clientside_metadata_tags' enabled - reject request if it has client-side 'metadata.tags'
    if (
        general_settings.get("reject_clientside_metadata_tags", None) is not None
        and general_settings["reject_clientside_metadata_tags"] is True
    ):
        if (
            RouteChecks.is_llm_api_route(route=route)
            and "metadata" in request_body
            and isinstance(request_body["metadata"], dict)
            and "tags" in request_body["metadata"]
        ):
            raise ProxyException(
                message=f"Client-side 'metadata.tags' not allowed in request. 'reject_clientside_metadata_tags'={general_settings['reject_clientside_metadata_tags']}. Tags can only be set via API key metadata.",
                type=ProxyErrorTypes.bad_request_error,
                param="metadata.tags",
                code=status.HTTP_400_BAD_REQUEST,
            )
    # 7. [OPTIONAL] If 'litellm.max_budget' is set (>0), is proxy under budget
    if (
        litellm.max_budget > 0
        and not skip_budget_checks
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

    token_team = getattr(valid_token, "team_id", None)
    token_type: Literal["ui", "api"] = (
        "ui" if token_team is not None and token_team == "litellm-dashboard" else "api"
    )
    _is_route_allowed = _is_allowed_route(
        route=route,
        token_type=token_type,
        user_obj=user_object,
        request=request,
        request_data=request_body,
        valid_token=valid_token,
    )

    # 11. [OPTIONAL] Vector store checks - is the object allowed to access the vector store
    await vector_store_access_check(
        request_body=request_body,
        team_object=team_object,
        valid_token=valid_token,
    )

    return True


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


def _is_api_route_allowed(
    route: str,
    request: Request,
    request_data: dict,
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
            valid_token=valid_token,
        )
    return True


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


def _is_allowed_route(
    route: str,
    token_type: Literal["ui", "api"],
    request: Request,
    request_data: dict,
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
            valid_token=valid_token,
            user_obj=user_obj,
        )


def _allowed_routes_check(user_route: str, allowed_routes: list) -> bool:
    """
    Return if a user is allowed to access route. Helper function for `allowed_routes_check`.

    Parameters:
    - user_route: str - the route the user is trying to call
    - allowed_routes: List[str|LiteLLMRoutes] - the list of allowed routes for the user.
    """
    from starlette.routing import compile_path

    for allowed_route in allowed_routes:
        if allowed_route in LiteLLMRoutes.__members__:
            for template in LiteLLMRoutes[allowed_route].value:
                regex, _, _ = compile_path(template)
                if regex.match(user_route):
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
            if isinstance(route_value, set):
                actual_routes.extend(list(route_value))
            else:
                actual_routes.extend(route_value)

        except KeyError:
            actual_routes.append(route_name)
    return actual_routes


async def get_default_end_user_budget(
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
    parent_otel_span: Optional[Span] = None,
) -> Optional[LiteLLM_BudgetTable]:
    """
    Fetches the default end user budget from the database if litellm.max_end_user_budget_id is configured.

    This budget is applied to end users who don't have an explicit budget_id set.
    Results are cached for performance.

    Args:
        prisma_client: Database client instance
        user_api_key_cache: Cache for storing/retrieving budget data
        parent_otel_span: Optional OpenTelemetry span for tracing

    Returns:
        LiteLLM_BudgetTable if configured and found, None otherwise
    """
    if prisma_client is None or litellm.max_end_user_budget_id is None:
        return None

    cache_key = f"default_end_user_budget:{litellm.max_end_user_budget_id}"

    # Check cache first
    cached_budget = await user_api_key_cache.async_get_cache(key=cache_key)
    if cached_budget is not None:
        return LiteLLM_BudgetTable(**cached_budget)

    # Fetch from database
    try:
        budget_record = await prisma_client.db.litellm_budgettable.find_unique(
            where={"budget_id": litellm.max_end_user_budget_id}
        )

        if budget_record is None:
            verbose_proxy_logger.warning(
                f"Default end user budget not found in database: {litellm.max_end_user_budget_id}"
            )
            return None

        # Cache the budget for 60 seconds
        await user_api_key_cache.async_set_cache(
            key=cache_key,
            value=budget_record.dict(),
            ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
        )

        return LiteLLM_BudgetTable(**budget_record.dict())

    except Exception as e:
        verbose_proxy_logger.error(f"Error fetching default end user budget: {str(e)}")
        return None


async def _apply_default_budget_to_end_user(
    end_user_obj: LiteLLM_EndUserTable,
    prisma_client: PrismaClient,
    user_api_key_cache: DualCache,
    parent_otel_span: Optional[Span] = None,
) -> LiteLLM_EndUserTable:
    """
    Helper function to apply default budget to end user if they don't have a budget assigned.

    Args:
        end_user_obj: The end user object to potentially apply default budget to
        prisma_client: Database client instance
        user_api_key_cache: Cache for storing/retrieving data
        parent_otel_span: Optional OpenTelemetry span for tracing

    Returns:
        Updated end user object with default budget applied if applicable
    """
    # If end user already has a budget assigned, no need to apply default
    if end_user_obj.litellm_budget_table is not None:
        return end_user_obj

    # If no default budget configured, return as-is
    if litellm.max_end_user_budget_id is None:
        return end_user_obj

    # Fetch and apply default budget
    default_budget = await get_default_end_user_budget(
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=parent_otel_span,
    )

    if default_budget is not None:
        # Apply default budget to end user object
        end_user_obj.litellm_budget_table = default_budget
        verbose_proxy_logger.debug(
            f"Applied default budget {litellm.max_end_user_budget_id} to end user {end_user_obj.user_id}"
        )

    return end_user_obj


def _check_end_user_budget(
    end_user_obj: LiteLLM_EndUserTable,
    route: str,
) -> None:
    """
    Check if end user is within their budget limit.

    Args:
        end_user_obj: The end user object to check
        route: The request route

    Raises:
        litellm.BudgetExceededError: If end user has exceeded their budget
    """
    if route in LiteLLMRoutes.info_routes.value:
        return

    if end_user_obj.litellm_budget_table is None:
        return

    end_user_budget = end_user_obj.litellm_budget_table.max_budget
    if end_user_budget is not None and end_user_obj.spend > end_user_budget:
        raise litellm.BudgetExceededError(
            current_cost=end_user_obj.spend,
            max_budget=end_user_budget,
            message=f"ExceededBudget: End User={end_user_obj.user_id} over budget. Spend={end_user_obj.spend}, Budget={end_user_budget}",
        )


@log_db_metrics
async def get_end_user_object(
    end_user_id: Optional[str],
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
    route: str,
    parent_otel_span: Optional[Span] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
) -> Optional[LiteLLM_EndUserTable]:
    """
    Returns end user object from database or cache.

    If end user exists but has no budget_id, applies the default budget
    (if configured via litellm.max_end_user_budget_id).

    Args:
        end_user_id: The ID of the end user
        prisma_client: Database client instance
        user_api_key_cache: Cache for storing/retrieving data
        route: The request route
        parent_otel_span: Optional OpenTelemetry span for tracing
        proxy_logging_obj: Optional proxy logging object

    Returns:
        LiteLLM_EndUserTable if found, None otherwise
    """
    if prisma_client is None:
        raise Exception("No db connected")

    if end_user_id is None:
        return None

    _key = "end_user_id:{}".format(end_user_id)

    # Check cache first
    cached_user_obj = await user_api_key_cache.async_get_cache(key=_key)
    if cached_user_obj is not None:
        return_obj = LiteLLM_EndUserTable(**cached_user_obj)

        # Apply default budget if needed
        return_obj = await _apply_default_budget_to_end_user(
            end_user_obj=return_obj,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=parent_otel_span,
        )

        # Check budget limits
        _check_end_user_budget(end_user_obj=return_obj, route=route)

        return return_obj

    # Fetch from database
    try:
        response = await prisma_client.db.litellm_endusertable.find_unique(
            where={"user_id": end_user_id},
            include={"litellm_budget_table": True},
        )

        if response is None:
            raise Exception

        # Convert to LiteLLM_EndUserTable object
        _response = LiteLLM_EndUserTable(**response.dict())

        # Apply default budget if needed
        _response = await _apply_default_budget_to_end_user(
            end_user_obj=_response,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=parent_otel_span,
        )

        # Save to cache (always store as dict for consistency)
        await user_api_key_cache.async_set_cache(
            key="end_user_id:{}".format(end_user_id), value=_response.dict()
        )

        # Check budget limits
        _check_end_user_budget(end_user_obj=_response, route=route)

        return _response

    except Exception as e:
        if isinstance(e, litellm.BudgetExceededError):
            raise e
        return None


@log_db_metrics
async def get_tag_objects_batch(
    tag_names: List[str],
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
    parent_otel_span: Optional[Span] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
) -> Dict[str, LiteLLM_TagTable]:
    """
    Batch fetch multiple tag objects from cache and db.

    Optimizes for latency by:
    1. Fetching all cached tags in parallel
    2. Batch fetching uncached tags in one DB query

    Args:
        tag_names: List of tag names to fetch
        prisma_client: Prisma database client
        user_api_key_cache: Cache for storing tag objects
        parent_otel_span: Optional OpenTelemetry span for tracing
        proxy_logging_obj: Optional proxy logging object

    Returns:
        Dictionary mapping tag_name to LiteLLM_TagTable object
    """
    if prisma_client is None:
        return {}

    if not tag_names:
        return {}

    tag_objects = {}
    uncached_tags = []

    # Try to get all tags from cache first
    for tag_name in tag_names:
        cache_key = f"tag:{tag_name}"
        cached_tag = await user_api_key_cache.async_get_cache(key=cache_key)
        if cached_tag is not None:
            if isinstance(cached_tag, dict):
                tag_objects[tag_name] = LiteLLM_TagTable(**cached_tag)
            else:
                tag_objects[tag_name] = cached_tag
        else:
            uncached_tags.append(tag_name)

    # Batch fetch uncached tags from DB in one query
    if uncached_tags:
        try:
            db_tags = await prisma_client.db.litellm_tagtable.find_many(
                where={"tag_name": {"in": uncached_tags}},
                include={"litellm_budget_table": True},
            )

            # Cache and add to tag_objects
            for db_tag in db_tags:
                tag_name = db_tag.tag_name
                cache_key = f"tag:{tag_name}"
                # Cache with default TTL (same as end_user objects)
                await user_api_key_cache.async_set_cache(
                    key=cache_key, value=db_tag.dict()
                )
                tag_objects[tag_name] = LiteLLM_TagTable(**db_tag.dict())
        except Exception as e:
            verbose_proxy_logger.debug(f"Error batch fetching tags from database: {e}")

    return tag_objects


@log_db_metrics
async def get_tag_object(
    tag_name: Optional[str],
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
    parent_otel_span: Optional[Span] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
) -> Optional[LiteLLM_TagTable]:
    """
    Returns tag object from cache or db.

    Uses default cache TTL (same as end_user objects) to avoid drift.

    Args:
        tag_name: Name of the tag to fetch
        prisma_client: Prisma database client
        user_api_key_cache: Cache for storing tag objects
        parent_otel_span: Optional OpenTelemetry span for tracing
        proxy_logging_obj: Optional proxy logging object

    Returns:
        LiteLLM_TagTable object if found, None otherwise
    """
    if prisma_client is None or tag_name is None:
        return None

    # Use batch helper for consistency
    tag_objects = await get_tag_objects_batch(
        tag_names=[tag_name],
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
    )

    return tag_objects.get(tag_name)


@log_db_metrics
async def get_team_membership(
    user_id: str,
    team_id: str,
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
    parent_otel_span: Optional[Span] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
) -> Optional["LiteLLM_TeamMembership"]:
    """
    Returns team membership object if user is member of team.

    Do a isolated check for team membership vs. doing a combined key + team + user + team-membership check, as key might come in frequently for different users/teams. Larger call will slowdown query time. This way we get to cache the constant (key/team/user info) and only update based on the changing value (team membership).
    """
    from litellm.proxy._types import LiteLLM_TeamMembership

    if prisma_client is None:
        raise Exception("No db connected")

    if user_id is None or team_id is None:
        return None

    _key = "team_membership:{}:{}".format(user_id, team_id)

    # check if in cache
    cached_membership_obj = await user_api_key_cache.async_get_cache(key=_key)
    if cached_membership_obj is not None:
        return LiteLLM_TeamMembership(**cached_membership_obj)

    # else, check db
    try:
        response = await prisma_client.db.litellm_teammembership.find_unique(
            where={"user_id_team_id": {"user_id": user_id, "team_id": team_id}},
            include={"litellm_budget_table": True},
        )

        if response is None:
            return None

        # save the team membership object to cache (store as dict)
        await user_api_key_cache.async_set_cache(key=_key, value=response.dict())

        _response = LiteLLM_TeamMembership(**response.dict())

        return _response
    except Exception:
        verbose_proxy_logger.exception(
            "Error getting team membership for user_id: %s, team_id: %s",
            user_id,
            team_id,
        )
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


def _get_role_based_permissions(
    rbac_role: RBAC_ROLES,
    general_settings: dict,
    key: Literal["models", "routes"],
) -> Optional[List[str]]:
    """
    Get the role based permissions from the general settings.
    """
    role_based_permissions = cast(
        Optional[List[RoleBasedPermissions]],
        general_settings.get("role_permissions", []),
    )
    if role_based_permissions is None:
        return None

    for role_based_permission in role_based_permissions:
        if role_based_permission.role == rbac_role:
            return getattr(role_based_permission, key)

    return None


def get_role_based_models(
    rbac_role: RBAC_ROLES,
    general_settings: dict,
) -> Optional[List[str]]:
    """
    Get the models allowed for a user role.

    Used by JWT Auth.
    """

    return _get_role_based_permissions(
        rbac_role=rbac_role,
        general_settings=general_settings,
        key="models",
    )


def get_role_based_routes(
    rbac_role: RBAC_ROLES,
    general_settings: dict,
) -> Optional[List[str]]:
    """
    Get the routes allowed for a user role.
    """

    return _get_role_based_permissions(
        rbac_role=rbac_role,
        general_settings=general_settings,
        key="routes",
    )


async def _get_fuzzy_user_object(
    prisma_client: PrismaClient,
    sso_user_id: Optional[str] = None,
    user_email: Optional[str] = None,
) -> Optional[LiteLLM_UserTable]:
    """
    Checks if sso user is in db.

    Called when user id match is not found in db.

    - Check if sso_user_id is user_id in db
    - Check if sso_user_id is sso_user_id in db
    - Check if user_email is user_email in db
    - If not, create new user with user_email and sso_user_id and user_id = sso_user_id
    """

    response = None
    if sso_user_id is not None:
        response = await prisma_client.db.litellm_usertable.find_unique(
            where={"sso_user_id": sso_user_id},
            include={"organization_memberships": True},
        )

    if response is None and user_email is not None:
        # Use case-insensitive query to handle emails with different casing
        # This matches the pattern used in _check_duplicate_user_email
        response = await prisma_client.db.litellm_usertable.find_first(
            where={"user_email": {"equals": user_email, "mode": "insensitive"}},
            include={"organization_memberships": True},
        )

        if response is not None and sso_user_id is not None:  # update sso_user_id
            asyncio.create_task(  # background task to update user with sso id
                prisma_client.db.litellm_usertable.update(
                    where={"user_id": response.user_id},
                    data={"sso_user_id": sso_user_id},
                )
            )

    return response


@log_db_metrics
async def get_user_object(
    user_id: Optional[str],
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
    user_id_upsert: bool,
    parent_otel_span: Optional[Span] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
    sso_user_id: Optional[str] = None,
    user_email: Optional[str] = None,
    check_db_only: Optional[bool] = None,
) -> Optional[LiteLLM_UserTable]:
    """
    - Check if user id in proxy User Table
    - if valid, return LiteLLM_UserTable object with defined limits
    - if not, then raise an error
    """

    if user_id is None:
        return None

    # check if in cache
    if not check_db_only:
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

            if response is None:
                response = await _get_fuzzy_user_object(
                    prisma_client=prisma_client,
                    sso_user_id=sso_user_id,
                    user_email=user_email,
                )

        else:
            response = None

        if response is None:
            if user_id_upsert:
                new_user_params: Dict[str, Any] = {
                    "user_id": user_id,
                }
                if litellm.default_internal_user_params is not None:
                    new_user_params.update(litellm.default_internal_user_params)

                response = await prisma_client.db.litellm_usertable.create(
                    data=new_user_params,
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
                LiteLLM_OrganizationMembershipTable(**membership.model_dump())
                for membership in response.organization_memberships
                if membership is not None
            ]
            response.organization_memberships = _dumped_memberships

        _response = LiteLLM_UserTable(**dict(response))
        response_dict = _response.model_dump()

        # save the user object to cache
        await user_api_key_cache.async_set_cache(
            key=user_id,
            value=response_dict,
            ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
        )

        # save to db access time
        _update_last_db_access_time(
            key=db_access_time_key,
            value=response_dict,
            last_db_access_time=last_db_access_time,
        )

        return _response
    except Exception as e:  # if user not in db
        _log_budget_lookup_failure("user", e)
        raise ValueError(
            f"User doesn't exist in db. 'user_id'={user_id}. Create user via `/user/new` call. Got error - {e}"
        )


async def _cache_management_object(
    key: str,
    value: BaseModel,
    user_api_key_cache: DualCache,
    proxy_logging_obj: Optional[ProxyLogging],
):
    await user_api_key_cache.async_set_cache(
        key=key,
        value=value,
        ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
    )


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
async def _get_team_db_check(
    team_id: str, prisma_client: PrismaClient, team_id_upsert: Optional[bool] = None
):
    response = await prisma_client.db.litellm_teamtable.find_unique(
        where={"team_id": team_id}
    )

    if response is None and team_id_upsert:
        from litellm.proxy.management_endpoints.team_endpoints import new_team

        new_team_data = NewTeamRequest(team_id=team_id)

        mock_request = Request(scope={"type": "http"})
        system_admin_user = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

        created_team_dict = await new_team(
            data=new_team_data,
            http_request=mock_request,
            user_api_key_dict=system_admin_user,
        )
        response = LiteLLM_TeamTable(**created_team_dict)
    return response


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
    team_id_upsert: Optional[bool] = None,
) -> LiteLLM_TeamTableCachedObj:
    db_access_time_key = key
    should_check_db = _should_check_db(
        key=db_access_time_key,
        last_db_access_time=last_db_access_time,
        db_cache_expiry=db_cache_expiry,
    )
    if should_check_db:
        response = await _get_team_db_check(
            team_id=team_id, prisma_client=prisma_client, team_id_upsert=team_id_upsert
        )
    else:
        response = None

    if response is None:
        raise Exception

    _response = LiteLLM_TeamTableCachedObj(**response.dict())
    
    # Load object_permission if object_permission_id exists but object_permission is not loaded
    if _response.object_permission_id and not _response.object_permission:
        try:
            _response.object_permission = await get_object_permission(
                object_permission_id=_response.object_permission_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=None,
                proxy_logging_obj=proxy_logging_obj,
            )
        except Exception as e:
            verbose_proxy_logger.debug(
                f"Failed to load object_permission for team {team_id} with object_permission_id={_response.object_permission_id}: {e}"
            )
    
    # save the team object to cache
    await _cache_team_object(
        team_id=team_id,
        team_table=_response,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )

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
    team_id_upsert: Optional[bool] = None,
) -> LiteLLM_TeamTableCachedObj:
    """
    - Check if team id in proxy Team Table
    - if valid, return LiteLLM_TeamTable object with defined limits
    - if not, then raise an error

    Raises:
        - HTTPException: If team doesn't exist in db or cache (status_code=404)
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
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Team doesn't exist in cache + check_cache_only=True. Team={team_id}."
                },
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
            team_id_upsert=team_id_upsert,
        )
    except Exception:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Team doesn't exist in db. Team={team_id}. Create team via `/team/new` call."
            },
        )


async def _cache_access_object(
    access_group_id: str,
    access_group_table: LiteLLM_AccessGroupTable,
    user_api_key_cache: DualCache,
    proxy_logging_obj: Optional[ProxyLogging] = None,
):
    key = "access_group_id:{}".format(access_group_id)
    await user_api_key_cache.async_set_cache(
        key=key,
        value=access_group_table,
        ttl=DEFAULT_ACCESS_GROUP_CACHE_TTL,
    )


async def _delete_cache_access_object(
    access_group_id: str,
    user_api_key_cache: DualCache,
    proxy_logging_obj: Optional[ProxyLogging] = None,
):
    key = "access_group_id:{}".format(access_group_id)

    user_api_key_cache.delete_cache(key=key)

    ## UPDATE REDIS CACHE ##
    if proxy_logging_obj is not None:
        await proxy_logging_obj.internal_usage_cache.dual_cache.async_delete_cache(
            key=key
        )


@log_db_metrics
async def get_access_object(
    access_group_id: str,
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
    proxy_logging_obj: Optional[ProxyLogging] = None,
) -> LiteLLM_AccessGroupTable:
    """
    - Check if access_group_id in proxy AccessGroupTable
    - Always checks cache first, then DB only when not found in cache
    - if valid, return LiteLLM_AccessGroupTable object
    - if not, then raise an error

    Unlike get_team_object, this has no check_cache_only or check_db_only flags;
    it always follows cache-first-then-db semantics.

    Raises:
        - HTTPException: If access group doesn't exist in db or cache (status_code=404)
    """
    if prisma_client is None:
        raise Exception(
            "No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys"
        )

    key = "access_group_id:{}".format(access_group_id)

    # Always check cache first
    cached_access_obj = await user_api_key_cache.async_get_cache(key=key)
    if cached_access_obj is not None:
        if isinstance(cached_access_obj, dict):
            return LiteLLM_AccessGroupTable(**cached_access_obj)
        elif isinstance(cached_access_obj, LiteLLM_AccessGroupTable):
            return cached_access_obj

    # Not in cache - fetch from DB
    try:
        response = await prisma_client.db.litellm_accessgrouptable.find_unique(
            where={"access_group_id": access_group_id}
        )

        if response is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Access group doesn't exist in db. Access group={access_group_id}."
                },
            )

        _response = LiteLLM_AccessGroupTable(**response.dict())

        # Save to cache
        await _cache_access_object(
            access_group_id=access_group_id,
            access_group_table=_response,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

        return _response
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(
            "Error getting access group for access_group_id: %s",
            access_group_id,
        )
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Access group doesn't exist in db. Access group={access_group_id}. Error: {e}"
            },
        )


@log_db_metrics
async def get_team_object_by_alias(
    team_alias: str,
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
    parent_otel_span: Optional["Span"] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
) -> LiteLLM_TeamTableCachedObj:
    """
    Look up a team by its team_alias (name) in the database.

    Args:
        team_alias: The team name/alias to look up
        prisma_client: Database client
        user_api_key_cache: Cache for storing results
        parent_otel_span: Optional OpenTelemetry span
        proxy_logging_obj: Optional proxy logging object

    Returns:
        LiteLLM_TeamTableCachedObj: The team object if found

    Raises:
        HTTPException: If team doesn't exist or multiple teams have the same alias
    """
    if prisma_client is None:
        raise Exception(
            "No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys"
        )

    # Check cache first (keyed by alias)
    cache_key = "team_alias:{}".format(team_alias)

    cached_team_obj = await _get_team_object_from_cache(
        key=cache_key,
        proxy_logging_obj=proxy_logging_obj,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=parent_otel_span,
    )

    if cached_team_obj is not None:
        return cached_team_obj

    # Query database by team_alias
    try:
        teams = await prisma_client.db.litellm_teamtable.find_many(
            where={"team_alias": team_alias}
        )

        if not teams:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Team with alias '{team_alias}' doesn't exist in db. Create team via `/team/new` call."
                },
            )

        if len(teams) > 1:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Multiple teams found with alias '{team_alias}'. Please use team_id_jwt_field instead or ensure team aliases are unique."
                },
            )

        team = teams[0]
        team_obj = LiteLLM_TeamTableCachedObj(**team.model_dump())

        # Load object_permission if object_permission_id exists but object_permission is not loaded
        if team_obj.object_permission_id and not team_obj.object_permission:
            try:
                team_obj.object_permission = await get_object_permission(
                    object_permission_id=team_obj.object_permission_id,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    parent_otel_span=parent_otel_span,
                    proxy_logging_obj=proxy_logging_obj,
                )
            except Exception as e:
                verbose_proxy_logger.debug(
                    f"Failed to load object_permission for team {team_obj.team_id} with object_permission_id={team_obj.object_permission_id}: {e}"
                )

        # Cache the result by both alias and team_id
        await user_api_key_cache.async_set_cache(
            key=cache_key,
            value=team_obj,
            ttl=DEFAULT_IN_MEMORY_TTL,
        )
        # Also cache by team_id for consistency
        team_id_cache_key = "team_id:{}".format(team_obj.team_id)
        await user_api_key_cache.async_set_cache(
            key=team_id_cache_key,
            value=team_obj,
            ttl=DEFAULT_IN_MEMORY_TTL,
        )

        return team_obj

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error looking up team by alias: %s", team_alias)
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Error looking up team by alias '{team_alias}': {str(e)}"
            },
        )


@log_db_metrics
async def get_org_object_by_alias(
    org_alias: str,
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
    parent_otel_span: Optional["Span"] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
) -> Optional[LiteLLM_OrganizationTable]:
    """
    Look up an organization by its organization_alias in the database.

    Args:
        org_alias: The organization name/alias to look up
        prisma_client: Database client
        user_api_key_cache: Cache for storing results
        parent_otel_span: Optional OpenTelemetry span
        proxy_logging_obj: Optional proxy logging object

    Returns:
        LiteLLM_OrganizationTable if found, None otherwise

    Raises:
        HTTPException: If organization not found or multiple orgs have the same alias
    """
    if prisma_client is None:
        raise Exception(
            "No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys"
        )

    # Check cache first (keyed by alias)
    cache_key = "org_alias:{}".format(org_alias)
    cached_org_obj = await user_api_key_cache.async_get_cache(key=cache_key)
    if cached_org_obj is not None:
        if isinstance(cached_org_obj, dict):
            return LiteLLM_OrganizationTable(**cached_org_obj)
        elif isinstance(cached_org_obj, LiteLLM_OrganizationTable):
            return cached_org_obj

    # Query database by organization_alias
    try:
        orgs = await prisma_client.db.litellm_organizationtable.find_many(
            where={"organization_alias": org_alias}
        )

        if not orgs:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Organization with alias '{org_alias}' doesn't exist in db. Create organization via `/organization/new` call."
                },
            )

        if len(orgs) > 1:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Multiple organizations found with alias '{org_alias}'. Please use org_id_jwt_field instead or ensure organization aliases are unique."
                },
            )

        org = orgs[0]
        org_obj = LiteLLM_OrganizationTable(**org.model_dump())

        # Cache the result
        await user_api_key_cache.async_set_cache(
            key=cache_key,
            value=org_obj.model_dump(),
            ttl=DEFAULT_IN_MEMORY_TTL,
        )
        # Also cache by org_id for consistency
        await user_api_key_cache.async_set_cache(
            key="org_id:{}".format(org_obj.organization_id),
            value=org_obj.model_dump(),
            ttl=DEFAULT_IN_MEMORY_TTL,
        )

        return org_obj

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(
            "Error looking up organization by alias: %s", org_alias
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Error looking up organization by alias '{org_alias}': {str(e)}"
            },
        )


class ExperimentalUIJWTToken:
    @staticmethod
    def get_experimental_ui_login_jwt_auth_token(user_info: LiteLLM_UserTable) -> str:
        from datetime import timedelta

        from litellm.proxy.common_utils.encrypt_decrypt_utils import (
            encrypt_value_helper,
        )

        if user_info.user_role is None:
            raise Exception("User role is required for experimental UI login")

        # Calculate expiration time (10 minutes from now)
        expiration_time = get_utc_datetime() + timedelta(minutes=10)

        # Format the expiration time as ISO 8601 string
        expires = expiration_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+00:00"

        valid_token = UserAPIKeyAuth(
            token="ui-token",
            key_name="ui-token",
            key_alias="ui-token",
            max_budget=litellm.max_ui_session_budget,
            rpm_limit=100,  # allow user to have a conversation on test key pane of UI
            expires=expires,
            user_id=user_info.user_id,
            team_id="litellm-dashboard",
            models=user_info.models,
            max_parallel_requests=None,
            user_role=LitellmUserRoles(user_info.user_role),
        )

        return encrypt_value_helper(valid_token.model_dump_json(exclude_none=True))

    @staticmethod
    def get_cli_jwt_auth_token(
        user_info: LiteLLM_UserTable, team_id: Optional[str] = None
    ) -> str:
        """
        Generate a JWT token for CLI authentication with configurable expiration.

        The expiration time can be controlled via the LITELLM_CLI_JWT_EXPIRATION_HOURS
        environment variable (defaults to 24 hours).

        Args:
            user_info: User information from the database
            team_id: Team ID for the user (optional, uses user's team if available)

        Returns:
            Encrypted JWT token string
        """
        from datetime import timedelta

        from litellm.proxy.common_utils.encrypt_decrypt_utils import (
            encrypt_value_helper,
        )

        if user_info.user_role is None:
            raise Exception("User role is required for CLI JWT login")

        # Calculate expiration time (configurable via LITELLM_CLI_JWT_EXPIRATION_HOURS env var)
        expiration_time = get_utc_datetime() + timedelta(hours=CLI_JWT_EXPIRATION_HOURS)

        # Format the expiration time as ISO 8601 string
        expires = expiration_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+00:00"

        # Use provided team_id, or fall back to user's teams if available
        _team_id = team_id
        if _team_id is None and hasattr(user_info, "teams") and user_info.teams:
            # Use first team if user has teams
            _team_id = user_info.teams[0] if len(user_info.teams) > 0 else None

        valid_token = UserAPIKeyAuth(
            token=CLI_JWT_TOKEN_NAME,
            key_name=CLI_JWT_TOKEN_NAME,
            key_alias=CLI_JWT_TOKEN_NAME,
            max_budget=litellm.max_ui_session_budget,
            expires=expires,
            user_id=user_info.user_id,
            team_id=_team_id,
            models=user_info.models,
            max_parallel_requests=None,
            user_role=LitellmUserRoles(user_info.user_role),
        )

        return encrypt_value_helper(valid_token.model_dump_json(exclude_none=True))

    @staticmethod
    def get_key_object_from_ui_hash_key(
        hashed_token: str,
    ) -> Optional[UserAPIKeyAuth]:
        import json

        from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
        from litellm.proxy.common_utils.encrypt_decrypt_utils import (
            decrypt_value_helper,
        )

        decrypted_token = decrypt_value_helper(
            hashed_token, key="ui_hash_key", exception_type="debug"
        )
        if decrypted_token is None:
            return None
        try:
            return UserAPIKeyAuth(**json.loads(decrypted_token))
        except Exception as e:
            raise Exception(
                f"Invalid hash key. Hash key={hashed_token}. Decrypted token={decrypted_token}. Error: {e}"
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
    _valid_token: Optional[BaseModel] = await prisma_client.get_data(
        token=hashed_token,
        table_name="combined_view",
        parent_otel_span=parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
    )

    if _valid_token is None:
        raise ProxyException(
            message="Authentication Error, Invalid proxy server token passed. key={}, not found in db. Create key via `/key/generate` call.".format(
                hashed_token
            ),
            type=ProxyErrorTypes.token_not_found_in_db,
            param="key",
            code=status.HTTP_401_UNAUTHORIZED,
        )

    _response = UserAPIKeyAuth(**_valid_token.model_dump(exclude_none=True))

    # Load object_permission if object_permission_id exists but object_permission is not loaded
    if _response.object_permission_id and not _response.object_permission:
        try:
            _response.object_permission = await get_object_permission(
                object_permission_id=_response.object_permission_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )
        except Exception as e:
            verbose_proxy_logger.debug(
                f"Failed to load object_permission for key with object_permission_id={_response.object_permission_id}: {e}"
            )

    # save the key object to cache
    await _cache_key_object(
        hashed_token=hashed_token,
        user_api_key_obj=_response,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )

    return _response


@log_db_metrics
async def get_object_permission(
    object_permission_id: str,
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
    parent_otel_span: Optional[Span] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
) -> Optional[LiteLLM_ObjectPermissionTable]:
    """
    - Check if object permission id in proxy ObjectPermissionTable
    - if valid, return LiteLLM_ObjectPermissionTable object
    - if not, then raise an error
    """
    if prisma_client is None:
        raise Exception(
            "No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys"
        )

    # check if in cache
    key = "object_permission_id:{}".format(object_permission_id)
    cached_obj_permission = await user_api_key_cache.async_get_cache(key=key)
    if cached_obj_permission is not None:
        if isinstance(cached_obj_permission, dict):
            return LiteLLM_ObjectPermissionTable(**cached_obj_permission)
        elif isinstance(cached_obj_permission, LiteLLM_ObjectPermissionTable):
            return cached_obj_permission

    # else, check db
    try:
        response = await prisma_client.db.litellm_objectpermissiontable.find_unique(
            where={"object_permission_id": object_permission_id}
        )

        if response is None:
            return None

        # save the object permission to cache
        await user_api_key_cache.async_set_cache(
            key=key,
            value=response.model_dump(),
            ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
        )

        return LiteLLM_ObjectPermissionTable(**response.dict())
    except Exception:
        return None


@log_db_metrics
async def get_org_object(
    org_id: str,
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
    parent_otel_span: Optional[Span] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
    include_budget_table: bool = False,
) -> Optional[LiteLLM_OrganizationTable]:
    """
    - Check if org id in proxy Org Table
    - if valid, return LiteLLM_OrganizationTable object
    - if not, then raise an error

    Args:
        org_id: Organization ID to look up
        prisma_client: Database client
        user_api_key_cache: Cache for storing results
        parent_otel_span: Optional OpenTelemetry span
        proxy_logging_obj: Optional proxy logging object
        include_budget_table: If True, includes litellm_budget_table in the query
    """
    if prisma_client is None:
        raise Exception(
            "No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys"
        )
    if not isinstance(org_id, str):
        return None

    # Use different cache key if budget table is included
    cache_key = "org_id:{}".format(org_id)
    if include_budget_table:
        cache_key = "org_id:{}:with_budget".format(org_id)

    # check if in cache
    cached_org_obj = user_api_key_cache.async_get_cache(key=cache_key)
    if cached_org_obj is not None:
        if isinstance(cached_org_obj, dict):
            return LiteLLM_OrganizationTable(**cached_org_obj)
        elif isinstance(cached_org_obj, LiteLLM_OrganizationTable):
            return cached_org_obj
    # else, check db
    try:
        query_kwargs: Dict[str, Any] = {"where": {"organization_id": org_id}}
        if include_budget_table:
            query_kwargs["include"] = {"litellm_budget_table": True}

        response = await prisma_client.db.litellm_organizationtable.find_unique(
            **query_kwargs
        )

        if response is None:
            raise Exception

        # Cache the result
        await user_api_key_cache.async_set_cache(
            key=cache_key,
            value=response.model_dump()
            if hasattr(response, "model_dump")
            else response,
            ttl=DEFAULT_IN_MEMORY_TTL,
        )

        return response
    except Exception:
        raise Exception(
            f"Organization doesn't exist in db. Organization={org_id}. Create organization via `/organization/new` call."
        )


async def _get_resources_from_access_groups(
    access_group_ids: List[str],
    resource_field: Literal[
        "access_model_names", "access_mcp_server_ids", "access_agent_ids"
    ],
    prisma_client: Optional[PrismaClient] = None,
    user_api_key_cache: Optional[DualCache] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
) -> List[str]:
    """
    Fetch access groups by their IDs (from cache or DB) and collect
    the specified resource field across all of them.

    Args:
        access_group_ids: List of access group IDs to fetch
        resource_field: Which resource list to extract from each access group
            - "access_model_names": model names (for model access checks)
            - "access_mcp_server_ids": MCP server IDs (for MCP access checks)
            - "access_agent_ids": agent IDs (for agent access checks)
        prisma_client: Optional PrismaClient (lazy-imported from proxy_server if None)
        user_api_key_cache: Optional DualCache (lazy-imported from proxy_server if None)
        proxy_logging_obj: Optional ProxyLogging (lazy-imported from proxy_server if None)

    Returns:
        Deduplicated list of resource identifiers from all resolved access groups.
    """
    if not access_group_ids:
        return []

    # Lazy import to avoid circular imports
    if prisma_client is None or user_api_key_cache is None:
        from litellm.proxy.proxy_server import (
            prisma_client as _prisma_client,
            proxy_logging_obj as _proxy_logging_obj,
            user_api_key_cache as _user_api_key_cache,
        )

        prisma_client = prisma_client or _prisma_client
        user_api_key_cache = user_api_key_cache or _user_api_key_cache
        proxy_logging_obj = proxy_logging_obj or _proxy_logging_obj

    if user_api_key_cache is None:
        return []

    resources: List[str] = []
    for ag_id in access_group_ids:
        try:
            ag = await get_access_object(
                access_group_id=ag_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )
            resources.extend(getattr(ag, resource_field, []))
        except Exception:
            verbose_proxy_logger.debug(
                "Could not fetch access group %s for resource field %s",
                ag_id,
                resource_field,
            )
    return list(set(resources))


async def _get_models_from_access_groups(
    access_group_ids: List[str],
    prisma_client: Optional[PrismaClient] = None,
    user_api_key_cache: Optional[DualCache] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
) -> List[str]:
    """
    Collect model names from unified access groups.
    Models are matched by model name for backwards compatibility.
    """
    return await _get_resources_from_access_groups(
        access_group_ids=access_group_ids,
        resource_field="access_model_names",
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )


async def _get_mcp_server_ids_from_access_groups(
    access_group_ids: List[str],
    prisma_client: Optional[PrismaClient] = None,
    user_api_key_cache: Optional[DualCache] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
) -> List[str]:
    """
    Collect MCP server IDs from unified access groups.
    MCPs are matched by server ID.
    """
    return await _get_resources_from_access_groups(
        access_group_ids=access_group_ids,
        resource_field="access_mcp_server_ids",
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )


async def _get_agent_ids_from_access_groups(
    access_group_ids: List[str],
    prisma_client: Optional[PrismaClient] = None,
    user_api_key_cache: Optional[DualCache] = None,
    proxy_logging_obj: Optional[ProxyLogging] = None,
) -> List[str]:
    """
    Collect agent IDs from unified access groups.
    Agents are matched by agent ID.
    """
    return await _get_resources_from_access_groups(
        access_group_ids=access_group_ids,
        resource_field="access_agent_ids",
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )


def _check_model_access_helper(
    model: str,
    llm_router: Optional[Router],
    models: List[str],
    team_model_aliases: Optional[Dict[str, str]] = None,
    team_id: Optional[str] = None,
) -> bool:
    ## check if model in allowed model names
    from collections import defaultdict

    access_groups: Dict[str, List[str]] = defaultdict(list)

    if llm_router:
        access_groups = llm_router.get_model_access_groups(
            model_name=model, team_id=team_id
        )

    if (
        len(access_groups) > 0 and llm_router is not None
    ):  # check if token contains any model access groups
        for idx, m in enumerate(
            models
        ):  # loop token models, if any of them are an access group add the access group
            if m in access_groups:
                return True

    # Filter out models that are access_groups
    filtered_models = [m for m in models if m not in access_groups]

    if _model_in_team_aliases(model=model, team_model_aliases=team_model_aliases):
        return True

    if _model_matches_any_wildcard_pattern_in_list(
        model=model, allowed_model_list=filtered_models
    ):
        return True

    all_model_access: bool = False

    if (len(filtered_models) == 0 and len(models) == 0) or "*" in filtered_models:
        all_model_access = True

    if SpecialModelNames.all_proxy_models.value in filtered_models:
        all_model_access = True

    if model is not None and model not in filtered_models and all_model_access is False:
        return False
    return True


def _can_object_call_model(
    model: Union[str, List[str]],
    llm_router: Optional[Router],
    models: List[str],
    team_model_aliases: Optional[Dict[str, str]] = None,
    team_id: Optional[str] = None,
    object_type: Literal["user", "team", "key", "org"] = "user",
    fallback_depth: int = 0,
) -> Literal[True]:
    """
    Checks if token can call a given model

    Args:
        - model: str
        - llm_router: Optional[Router]
        - models: List[str]
        - team_model_aliases: Optional[Dict[str, str]]
        - object_type: Literal["user", "team", "key", "org"]. We use the object type to raise the correct exception type

    Returns:
        - True: if token allowed to call model

    Raises:
        - Exception: If token not allowed to call model
    """
    if fallback_depth >= DEFAULT_MAX_RECURSE_DEPTH:
        raise Exception(
            "Unable to parse model, max fallback depth exceeded - received model: {}".format(
                model
            )
        )
    if isinstance(model, list):
        for m in model:
            _can_object_call_model(
                model=m,
                llm_router=llm_router,
                models=models,
                team_model_aliases=team_model_aliases,
                team_id=team_id,
                object_type=object_type,
                fallback_depth=fallback_depth + 1,
            )
        return True

    potential_models = [model]
    if model in litellm.model_alias_map:
        potential_models.append(litellm.model_alias_map[model])
    elif llm_router and model in llm_router.model_group_alias:
        _model = llm_router._get_model_from_alias(model)
        if _model:
            potential_models.append(_model)

    ## check model access for alias + underlying model - allow if either is in allowed models
    for m in potential_models:
        if _check_model_access_helper(
            model=m,
            llm_router=llm_router,
            models=models,
            team_model_aliases=team_model_aliases,
            team_id=team_id,
        ):
            return True

    raise ProxyException(
        message=f"{object_type} not allowed to access model. This {object_type} can only access models={models}. Tried to access {model}",
        type=ProxyErrorTypes.get_model_access_error_type_for_object(
            object_type=object_type
        ),
        param="model",
        code=status.HTTP_401_UNAUTHORIZED,
    )


def _model_in_team_aliases(
    model: str, team_model_aliases: Optional[Dict[str, str]] = None
) -> bool:
    """
    Returns True if `model` being accessed is an alias of a team model

    - `model=gpt-4o`
    - `team_model_aliases={"gpt-4o": "gpt-4o-team-1"}`
        - returns True

    - `model=gp-4o`
    - `team_model_aliases={"o-3": "o3-preview"}`
        - returns False
    """
    if team_model_aliases:
        if model in team_model_aliases:
            return True
    return False


async def can_key_call_model(
    model: Union[str, List[str]],
    llm_model_list: Optional[list],
    valid_token: UserAPIKeyAuth,
    llm_router: Optional[litellm.Router],
) -> Literal[True]:
    """
    Checks if token can call a given model

    1. First checks native key-level model permissions (current implementation)
    2. If not allowed natively, falls back to access_group_ids on the key

    Returns:
        - True: if token allowed to call model

    Raises:
        - Exception: If token not allowed to call model
    """
    try:
        return _can_object_call_model(
            model=model,
            llm_router=llm_router,
            models=valid_token.models,
            team_model_aliases=valid_token.team_model_aliases,
            team_id=valid_token.team_id,
            object_type="key",
        )
    except ProxyException:
        # Fallback: check key's access_group_ids
        key_access_group_ids = valid_token.access_group_ids or []
        if key_access_group_ids:
            models_from_groups = await _get_models_from_access_groups(
                access_group_ids=key_access_group_ids,
            )
            if models_from_groups:
                return _can_object_call_model(
                    model=model,
                    llm_router=llm_router,
                    models=models_from_groups,
                    team_model_aliases=valid_token.team_model_aliases,
                    team_id=valid_token.team_id,
                    object_type="key",
                )
        raise


def can_org_access_model(
    model: str,
    org_object: Optional[LiteLLM_OrganizationTable],
    llm_router: Optional[Router],
    team_model_aliases: Optional[Dict[str, str]] = None,
) -> Literal[True]:
    """
    Returns True if the team can access a specific model.

    """
    return _can_object_call_model(
        model=model,
        llm_router=llm_router,
        models=org_object.models if org_object else [],
        team_model_aliases=team_model_aliases,
        object_type="org",
    )


async def can_team_access_model(
    model: Union[str, List[str]],
    team_object: Optional[LiteLLM_TeamTable],
    llm_router: Optional[Router],
    team_model_aliases: Optional[Dict[str, str]] = None,
) -> Literal[True]:
    """
    Returns True if the team can access a specific model.

    1. First checks native team-level model permissions (current implementation)
    2. If not allowed natively, falls back to access_group_ids on the team
    """
    try:
        return _can_object_call_model(
            model=model,
            llm_router=llm_router,
            models=team_object.models if team_object else [],
            team_model_aliases=team_model_aliases,
            team_id=team_object.team_id if team_object else None,
            object_type="team",
        )
    except ProxyException:
        # Fallback: check team's access_group_ids
        team_access_group_ids = (
            (team_object.access_group_ids or []) if team_object else []
        )
        if team_access_group_ids:
            models_from_groups = await _get_models_from_access_groups(
                access_group_ids=team_access_group_ids,
            )
            if models_from_groups:
                return _can_object_call_model(
                    model=model,
                    llm_router=llm_router,
                    models=models_from_groups,
                    team_model_aliases=team_model_aliases,
                    team_id=team_object.team_id if team_object else None,
                    object_type="team",
                )
        raise


async def can_user_call_model(
    model: Union[str, List[str]],
    llm_router: Optional[Router],
    user_object: Optional[LiteLLM_UserTable],
) -> Literal[True]:
    if user_object is None:
        return True

    if SpecialModelNames.no_default_models.value in user_object.models:
        raise ProxyException(
            message=f"User not allowed to access model. No default model access, only team models allowed. Tried to access {model}",
            type=ProxyErrorTypes.key_model_access_denied,
            param="model",
            code=status.HTTP_401_UNAUTHORIZED,
        )

    return _can_object_call_model(
        model=model,
        llm_router=llm_router,
        models=user_object.models,
        object_type="user",
    )


async def is_valid_fallback_model(
    model: str,
    llm_router: Optional[Router],
    user_model: Optional[str],
) -> Literal[True]:
    """
    Try to route the fallback model request.

    Validate if it can't be routed.

    Help catch invalid fallback models.
    """
    await route_request(
        data={
            "model": model,
            "messages": [{"role": "user", "content": "Who was Alexander?"}],
        },
        llm_router=llm_router,
        user_model=user_model,
        route_type="acompletion",  # route type shouldn't affect the fallback model check
    )

    return True


async def _virtual_key_max_budget_check(
    valid_token: UserAPIKeyAuth,
    proxy_logging_obj: ProxyLogging,
    user_obj: Optional[LiteLLM_UserTable] = None,
):
    """
    Raises:
        BudgetExceededError if the token is over it's max budget.
        Triggers a budget alert if the token is over it's max budget.

    """
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
            soft_budget=valid_token.soft_budget,
            user_id=valid_token.user_id,
            team_id=valid_token.team_id,
            organization_id=valid_token.org_id,
            user_email=user_email,
            key_alias=valid_token.key_alias,
            event_group=Litellm_EntityType.KEY,
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


async def _virtual_key_soft_budget_check(
    valid_token: UserAPIKeyAuth,
    proxy_logging_obj: ProxyLogging,
    user_obj: Optional[LiteLLM_UserTable] = None,
):
    """
    Triggers a budget alert if the token is over it's soft budget.

    """

    if valid_token.soft_budget and valid_token.spend >= valid_token.soft_budget:
        verbose_proxy_logger.debug(
            "Crossed Soft Budget for token %s, spend %s, soft_budget %s",
            valid_token.token,
            valid_token.spend,
            valid_token.soft_budget,
        )
        call_info = CallInfo(
            token=valid_token.token,
            spend=valid_token.spend,
            max_budget=valid_token.max_budget,
            soft_budget=valid_token.soft_budget,
            user_id=valid_token.user_id,
            team_id=valid_token.team_id,
            team_alias=valid_token.team_alias,
            organization_id=valid_token.org_id,
            user_email=user_obj.user_email if user_obj else None,
            key_alias=valid_token.key_alias,
            event_group=Litellm_EntityType.KEY,
        )

        asyncio.create_task(
            proxy_logging_obj.budget_alerts(
                type="soft_budget",
                user_info=call_info,
            )
        )


async def _virtual_key_max_budget_alert_check(
    valid_token: UserAPIKeyAuth,
    proxy_logging_obj: ProxyLogging,
    user_obj: Optional[LiteLLM_UserTable] = None,
):
    """
    Triggers a budget alert if the token has reached EMAIL_BUDGET_ALERT_MAX_SPEND_ALERT_PERCENTAGE
    (default 80%) of its max budget.
    This is a warning alert before the token actually exceeds the max budget.

    """

    if (
        valid_token.max_budget is not None
        and valid_token.spend is not None
        and valid_token.spend > 0
    ):
        alert_threshold = (
            valid_token.max_budget * EMAIL_BUDGET_ALERT_MAX_SPEND_ALERT_PERCENTAGE
        )

        # Only alert if we've crossed the threshold but haven't exceeded max_budget yet
        if (
            valid_token.spend >= alert_threshold
            and valid_token.spend < valid_token.max_budget
        ):
            verbose_proxy_logger.debug(
                "Reached Max Budget Alert Threshold for token %s, spend %s, max_budget %s, alert_threshold %s",
                valid_token.token,
                valid_token.spend,
                valid_token.max_budget,
                alert_threshold,
            )
            call_info = CallInfo(
                token=valid_token.token,
                spend=valid_token.spend,
                max_budget=valid_token.max_budget,
                soft_budget=valid_token.soft_budget,
                user_id=valid_token.user_id,
                team_id=valid_token.team_id,
                team_alias=valid_token.team_alias,
                organization_id=valid_token.org_id,
                user_email=user_obj.user_email if user_obj else None,
                key_alias=valid_token.key_alias,
                event_group=Litellm_EntityType.KEY,
            )

            asyncio.create_task(
                proxy_logging_obj.budget_alerts(
                    type="max_budget_alert",
                    user_info=call_info,
                )
            )


async def _check_team_member_budget(
    team_object: Optional[LiteLLM_TeamTable],
    user_object: Optional[LiteLLM_UserTable],
    valid_token: Optional[UserAPIKeyAuth],
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
    proxy_logging_obj: ProxyLogging,
):
    """Check if team member is over their max budget within the team."""
    if (
        team_object is not None
        and team_object.team_id is not None
        and user_object is not None
        and valid_token is not None
        and valid_token.user_id is not None
    ):
        team_membership = await get_team_membership(
            user_id=valid_token.user_id,
            team_id=team_object.team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

        if (
            team_membership is not None
            and team_membership.litellm_budget_table is not None
            and team_membership.litellm_budget_table.max_budget is not None
        ):
            team_member_budget = team_membership.litellm_budget_table.max_budget
            team_member_spend = team_membership.spend or 0.0

            if team_member_spend >= team_member_budget:
                raise litellm.BudgetExceededError(
                    current_cost=team_member_spend,
                    max_budget=team_member_budget,
                    message=f"Budget has been exceeded! User={valid_token.user_id} in Team={team_object.team_id} Current cost: {team_member_spend}, Max budget: {team_member_budget}",
                )


async def _team_max_budget_check(
    team_object: Optional[LiteLLM_TeamTable],
    valid_token: Optional[UserAPIKeyAuth],
    proxy_logging_obj: ProxyLogging,
):
    """
    Check if the team is over it's max budget.

    Raises:
        BudgetExceededError if the team is over it's max budget.
        Triggers a budget alert if the team is over it's max budget.
    """
    if (
        team_object is not None
        and team_object.max_budget is not None
        and team_object.spend is not None
        and team_object.spend > team_object.max_budget
    ):
        if valid_token:
            call_info = CallInfo(
                token=valid_token.token,
                spend=team_object.spend,
                max_budget=team_object.max_budget,
                user_id=valid_token.user_id,
                team_id=valid_token.team_id,
                team_alias=valid_token.team_alias,
                organization_id=valid_token.org_id,
                event_group=Litellm_EntityType.TEAM,
            )
            asyncio.create_task(
                proxy_logging_obj.budget_alerts(
                    type="team_budget",
                    user_info=call_info,
                )
            )

        raise litellm.BudgetExceededError(
            current_cost=team_object.spend,
            max_budget=team_object.max_budget,
            message=f"Budget has been exceeded! Team={team_object.team_id} Current cost: {team_object.spend}, Max budget: {team_object.max_budget}",
        )


async def _team_soft_budget_check(
    team_object: Optional[LiteLLM_TeamTable],
    valid_token: Optional[UserAPIKeyAuth],
    proxy_logging_obj: ProxyLogging,
):
    """
    Triggers a budget alert if the team is over it's soft budget.
    """
    if (
        team_object is not None
        and team_object.soft_budget is not None
        and team_object.spend is not None
        and team_object.spend >= team_object.soft_budget
    ):
        verbose_proxy_logger.debug(
            "Crossed Soft Budget for team %s, spend %s, soft_budget %s",
            team_object.team_id,
            team_object.spend,
            team_object.soft_budget,
        )
        if valid_token:
            # Extract alert emails from team metadata
            alert_emails: Optional[List[str]] = None
            if team_object.metadata is not None and isinstance(team_object.metadata, dict):
                soft_budget_alert_emails = team_object.metadata.get("soft_budget_alerting_emails")
                if soft_budget_alert_emails is not None:
                    if isinstance(soft_budget_alert_emails, list):
                        alert_emails = [email for email in soft_budget_alert_emails if isinstance(email, str) and email.strip()]
                    elif isinstance(soft_budget_alert_emails, str):
                        # Handle comma-separated string
                        alert_emails = [email.strip() for email in soft_budget_alert_emails.split(",") if email.strip()]
                    # Filter out empty strings
                    if alert_emails:
                        alert_emails = [email for email in alert_emails if email]
                    else:
                        alert_emails = None

            # Only send team soft budget alerts if alert_emails are configured
            # Team soft budget alerts are sent via metadata.soft_budget_alerting_emails, not global alerting
            if alert_emails is None or len(alert_emails) == 0:
                verbose_proxy_logger.debug(
                    "Skipping team soft budget alert for team %s: no alert_emails configured in metadata.soft_budget_alerting_emails",
                    team_object.team_id,
                )
                return

            call_info = CallInfo(
                token=valid_token.token,
                spend=team_object.spend,
                max_budget=team_object.max_budget,
                soft_budget=team_object.soft_budget,
                user_id=valid_token.user_id,
                team_id=valid_token.team_id,
                team_alias=valid_token.team_alias,
                organization_id=valid_token.org_id,
                user_email=None,  # Team-level alert, no specific user email
                key_alias=valid_token.key_alias,
                event_group=Litellm_EntityType.TEAM,
                alert_emails=alert_emails,
            )

            asyncio.create_task(
                proxy_logging_obj.budget_alerts(
                    type="soft_budget",
                    user_info=call_info,
                )
            )


async def _organization_max_budget_check(
    valid_token: Optional[UserAPIKeyAuth],
    team_object: Optional[LiteLLM_TeamTable],
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
    proxy_logging_obj: ProxyLogging,
):
    """
    Check if the organization is over its max budget.

    This function checks the organization budget using:
    1. First, tries to use valid_token.org_id (if key has organization_id set)
    2. Falls back to team_object.organization_id (if key doesn't have org_id but team does)

    This ensures organization budget checks work even when keys don't have organization_id
    set directly, as long as their team belongs to an organization.

    Raises:
        BudgetExceededError if the organization is over its max budget.
        Triggers a budget alert if the organization is over its max budget.
    """
    if valid_token is None or prisma_client is None:
        return

    # Determine organization_id: first try from token, then fallback to team
    org_id: Optional[str] = None
    if valid_token.org_id is not None:
        org_id = valid_token.org_id
    elif team_object is not None and team_object.organization_id is not None:
        org_id = team_object.organization_id

    # If no organization_id found, skip the check
    if org_id is None:
        return

    # Get organization object with budget table - use get_org_object so it can be mocked in tests
    try:
        org_table = await get_org_object(
            org_id=org_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
            include_budget_table=True,
        )
    except Exception:
        # If organization lookup fails, skip the check
        return

    if org_table is None:
        return

    # Get max_budget from organization's budget table
    org_max_budget: Optional[float] = None
    if org_table.litellm_budget_table is not None:
        org_max_budget = org_table.litellm_budget_table.max_budget

    # Only check if organization has a valid max_budget set
    if org_max_budget is None or org_max_budget <= 0:
        return

    # Check if organization spend exceeds max budget
    if org_table.spend >= org_max_budget:
        # Trigger budget alert
        call_info = CallInfo(
            token=valid_token.token,
            spend=org_table.spend,
            max_budget=org_max_budget,
            user_id=valid_token.user_id,
            team_id=valid_token.team_id,
            team_alias=valid_token.team_alias,
            organization_id=org_id,
            event_group=Litellm_EntityType.ORGANIZATION,
        )
        asyncio.create_task(
            proxy_logging_obj.budget_alerts(
                type="organization_budget",
                user_info=call_info,
            )
        )

        raise litellm.BudgetExceededError(
            current_cost=org_table.spend,
            max_budget=org_max_budget,
            message=f"Budget has been exceeded! Organization={org_id} Current cost: {org_table.spend}, Max budget: {org_max_budget}",
        )


async def _tag_max_budget_check(
    request_body: dict,
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
    proxy_logging_obj: ProxyLogging,
    valid_token: Optional[UserAPIKeyAuth],
):
    """
    Check if any tags in the request are over their max budget.

    Raises:
        BudgetExceededError if any tag is over its max budget.
        Triggers a budget alert if any tag is over its max budget.
    """
    from litellm.proxy.common_utils.http_parsing_utils import get_tags_from_request_body

    if prisma_client is None:
        return

    # Get tags from request metadata
    tags = get_tags_from_request_body(request_body=request_body)
    if not tags:
        return

    # Batch fetch all tags in one go
    tag_objects = await get_tag_objects_batch(
        tag_names=tags,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )

    # Check budget for each tag
    for tag_name in tags:
        tag_object = tag_objects.get(tag_name)
        if tag_object is None:
            continue

        # Check if tag has budget limits
        if (
            tag_object.litellm_budget_table is not None
            and tag_object.litellm_budget_table.max_budget is not None
            and tag_object.spend is not None
            and tag_object.spend > tag_object.litellm_budget_table.max_budget
        ):
            raise litellm.BudgetExceededError(
                current_cost=tag_object.spend,
                max_budget=tag_object.litellm_budget_table.max_budget,
                message=f"Budget has been exceeded! Tag={tag_name} Current cost: {tag_object.spend}, Max budget: {tag_object.litellm_budget_table.max_budget}",
            )


def is_model_allowed_by_pattern(model: str, allowed_model_pattern: str) -> bool:
    """
    Check if a model matches an allowed pattern.
    Handles exact matches and wildcard patterns.

    Args:
        model (str): The model to check (e.g., "bedrock/anthropic.claude-3-5-sonnet-20240620")
        allowed_model_pattern (str): The allowed pattern (e.g., "bedrock/*", "*", "openai/*")

    Returns:
        bool: True if model matches the pattern, False otherwise
    """
    if "*" in allowed_model_pattern:
        pattern = f"^{allowed_model_pattern.replace('*', '.*')}$"
        return bool(re.match(pattern, model))

    return False


def _model_matches_any_wildcard_pattern_in_list(
    model: str, allowed_model_list: list
) -> bool:
    """
    Returns True if a model matches any wildcard pattern in a list.

    eg.
    - model=`bedrock/us.amazon.nova-micro-v1:0`, allowed_models=`bedrock/*` returns True
    - model=`bedrock/us.amazon.nova-micro-v1:0`, allowed_models=`bedrock/us.*` returns True
    - model=`bedrockzzzz/us.amazon.nova-micro-v1:0`, allowed_models=`bedrock/*` returns False
    """

    if any(
        _is_wildcard_pattern(allowed_model_pattern)
        and is_model_allowed_by_pattern(
            model=model, allowed_model_pattern=allowed_model_pattern
        )
        for allowed_model_pattern in allowed_model_list
    ):
        return True

    if any(
        _is_wildcard_pattern(allowed_model_pattern)
        and _model_custom_llm_provider_matches_wildcard_pattern(
            model=model, allowed_model_pattern=allowed_model_pattern
        )
        for allowed_model_pattern in allowed_model_list
    ):
        return True

    return False


def _model_custom_llm_provider_matches_wildcard_pattern(
    model: str, allowed_model_pattern: str
) -> bool:
    """
    Returns True for this scenario:
    - `model=gpt-4o`
    - `allowed_model_pattern=openai/*`

    or
    - `model=claude-3-5-sonnet-20240620`
    - `allowed_model_pattern=anthropic/*`
    """
    try:
        model, custom_llm_provider, _, _ = get_llm_provider(model=model)
    except Exception:
        return False

    return is_model_allowed_by_pattern(
        model=f"{custom_llm_provider}/{model}",
        allowed_model_pattern=allowed_model_pattern,
    )


def _is_wildcard_pattern(allowed_model_pattern: str) -> bool:
    """
    Returns True if the pattern is a wildcard pattern.

    Checks if `*` is in the pattern.
    """
    return "*" in allowed_model_pattern


async def vector_store_access_check(
    request_body: dict,
    team_object: Optional[LiteLLM_TeamTable],
    valid_token: Optional[UserAPIKeyAuth],
):
    """
    Checks if the object (key, team, org) has access to the vector store.

    Raises ProxyException if the object (key, team, org) cannot access the specific vector store.
    """
    from litellm.proxy.proxy_server import prisma_client

    #########################################################
    # Get the vector store the user is trying to access
    #########################################################
    if prisma_client is None:
        verbose_proxy_logger.debug(
            "Prisma client not found, skipping vector store access check"
        )
        return True

    if litellm.vector_store_registry is None:
        verbose_proxy_logger.debug(
            "Vector store registry not found, skipping vector store access check"
        )
        return True

    vector_store_ids_to_run = litellm.vector_store_registry.get_vector_store_ids_to_run(
        non_default_params=request_body, tools=request_body.get("tools", None)
    )
    if vector_store_ids_to_run is None:
        verbose_proxy_logger.debug(
            "Vector store to run not found, skipping vector store access check"
        )
        return True

    #########################################################
    # Check if the object (key, team, org) has access to the vector store
    #########################################################
    # Check if the key can access the vector store
    if valid_token is not None and valid_token.object_permission_id is not None:
        key_object_permission = (
            await prisma_client.db.litellm_objectpermissiontable.find_unique(
                where={"object_permission_id": valid_token.object_permission_id},
            )
        )
        if key_object_permission is not None:
            _can_object_call_vector_stores(
                object_type="key",
                vector_store_ids_to_run=vector_store_ids_to_run,
                object_permissions=key_object_permission,
            )

    # Check if the team can access the vector store
    if team_object is not None and team_object.object_permission_id is not None:
        team_object_permission = (
            await prisma_client.db.litellm_objectpermissiontable.find_unique(
                where={"object_permission_id": team_object.object_permission_id},
            )
        )
        if team_object_permission is not None:
            _can_object_call_vector_stores(
                object_type="team",
                vector_store_ids_to_run=vector_store_ids_to_run,
                object_permissions=team_object_permission,
            )
    return True


def _can_object_call_vector_stores(
    object_type: Literal["key", "team", "org"],
    vector_store_ids_to_run: List[str],
    object_permissions: Optional[LiteLLM_ObjectPermissionTable],
):
    """
    Raises ProxyException if the object (key, team, org) cannot access the specific vector store.
    """
    if object_permissions is None:
        return True

    if object_permissions.vector_stores is None:
        return True

    # If length is 0, then the object has access to all vector stores.
    if len(object_permissions.vector_stores) == 0:
        return True

    for vector_store_id in vector_store_ids_to_run:
        if vector_store_id not in object_permissions.vector_stores:
            raise ProxyException(
                message=f"User not allowed to access vector store. Tried to access {vector_store_id}. Only allowed to access {object_permissions.vector_stores}",
                type=ProxyErrorTypes.get_vector_store_access_error_type_for_object(
                    object_type
                ),
                param="vector_store",
                code=status.HTTP_401_UNAUTHORIZED,
            )

    return True
