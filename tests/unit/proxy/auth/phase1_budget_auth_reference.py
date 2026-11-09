"""Policy frozen from 8bd7f5089a7500d9c915a5670c1b314ee051f9c9, with independently supplied raw IO."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Final, Optional, Protocol

from opentelemetry.trace import Span
from pydantic import ConfigDict, TypeAdapter

import litellm
from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.decisions_pricing import decisions_pricing_fields, decisions_request_price
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.proxy._types import (
    MODEL_LIST_ROUTES,
    CallInfo,
    LiteLLM_BudgetTable,
    LiteLLM_EndUserTable,
    Litellm_EntityType,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import (
    _dictable_table,  # pyright: ignore[reportPrivateUsage]  # data-only Prisma row projection
)
from litellm.proxy.common_utils.user_api_key_cache import (
    AUTH_OBJECTS_TARGET,
    UserApiKeyCache,
    get_management_object_ttl,
)
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.repositories.budget_repository import BudgetRepository
from litellm.router import Router
from litellm.router_api import RegistryReader
from litellm.types.router import DeploymentTypedDict

from .phase1_budget_route_reference import RouteChecks
from .phase1_budget_throttle_reference import budget_throttle_percentage, should_throttle_budget_exceeded


class _RawCacheRead(Protocol):
    async def async_get_cache(self, *, key: str, local_only: bool = ...) -> object: ...


def _raw_cache(cache: _RawCacheRead) -> _RawCacheRead:
    return cache


def _get_router_zero_cost_cache(llm_router: RegistryReader) -> dict[str, bool] | None:  # mutable-ok: Phase 1 public collection and identity contract

    cache: Final = llm_router.registry().zero_cost()
    return cache if isinstance(cache, dict) else None


def is_model_cost_zero(model: str | list[str] | None, llm_router: RegistryReader | None) -> bool:  # mutable-ok: Phase 1 public collection and identity contract

    if model is None or llm_router is None:
        return False

    model_list: Final = [model] if isinstance(model, str) else model

    zero_cost_cache: Final = _get_router_zero_cost_cache(llm_router)

    for model_name in model_list:
        if zero_cost_cache is not None:
            cached = zero_cost_cache.get(model_name)
            if cached is not None:
                if cached is False:
                    return False
                continue
        try:
            model_group_info = llm_router.registry().group(model_name, include_hidden=True)

            if model_group_info is None:
                verbose_proxy_logger.debug("No model group info found for %s, assuming it has cost", model_name)
                if zero_cost_cache is not None:
                    zero_cost_cache[model_name] = False
                return False

            input_cost = model_group_info.input_cost_per_token
            output_cost = model_group_info.output_cost_per_token

            if input_cost is None or output_cost is None:
                verbose_proxy_logger.debug(
                    "Model %s has undefined cost (input: %s, output: %s), assuming it has cost",
                    model_name,
                    input_cost,
                    output_cost,
                )
                if zero_cost_cache is not None:
                    zero_cost_cache[model_name] = False
                return False

            if input_cost > 0 or output_cost > 0:
                verbose_proxy_logger.debug(
                    "Model %s has non-zero cost (input: %s, output: %s)", model_name, input_cost, output_cost
                )
                if zero_cost_cache is not None:
                    zero_cost_cache[model_name] = False
                return False

            safe_name = str(model_name).replace("\n", "").replace("\r", "")
            served_deployments = _deployments_served_for(model_name, llm_router)
            if not _is_cost_explicitly_configured(served_deployments):
                verbose_proxy_logger.debug(
                    "Model %s has zero cost but no explicit cost "
                    "configuration in model_cost entry — treating as unknown "
                    "cost (enforce budget)",
                    safe_name,
                )
                if zero_cost_cache is not None:
                    zero_cost_cache[model_name] = False
                return False

            if _has_explicit_positive_cost(served_deployments):
                verbose_proxy_logger.debug(
                    "Model %s routes to a deployment with an explicit positive per-token price (enforce budget)",
                    safe_name,
                )
                if zero_cost_cache is not None:
                    zero_cost_cache[model_name] = False
                return False

            if _has_ptu_flat_cost(served_deployments):
                verbose_proxy_logger.debug(
                    "Model %s prices reserved PTU capacity as a flat cost, so its zero per-token "
                    "rate is not a free model (enforce budget)",
                    safe_name,
                )
                if zero_cost_cache is not None:
                    zero_cost_cache[model_name] = False
                return False

            verbose_proxy_logger.debug(
                "Model %s has zero cost explicitly configured (input: %s, output: %s)",
                safe_name,
                input_cost,
                output_cost,
            )
            if zero_cost_cache is not None:
                zero_cost_cache[model_name] = True

        except Exception as e:
            verbose_proxy_logger.debug("Error checking cost for model %s: %s, assuming it has cost", model_name, e)
            return False

    return True


_NO_MODEL_INFO: Final[Mapping[str, object]] = MappingProxyType({})


def _deployments_served_for(model: str, llm_router: RegistryReader) -> Sequence[DeploymentTypedDict]:

    registry: Final = llm_router.registry()
    routed_group: Final = registry.resolve(model)
    if registry.resolve(routed_group) == routed_group:
        return registry.deployments(routed_group) or ()
    named: Final = registry.deployments(model, source="alias")
    if named:
        return named
    return tuple(
        DeploymentTypedDict(**deployment)  # pyright: ignore[reportUnknownArgumentType]  # frozen Phase 1 accepts unvalidated legacy values
        for deployment in registry.patterns().get_deployments_by_pattern(model=routed_group)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # frozen Phase 1 accepts unvalidated legacy values
    )


def _has_ptu_flat_cost(deployments: Sequence[DeploymentTypedDict]) -> bool:

    for deployment in deployments:
        model_info = deployment.get("model_info") or _NO_MODEL_INFO  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # frozen Phase 1 accepts unvalidated legacy values
        if model_info.get("ptu_count") is not None and model_info.get("cost_per_ptu_per_hour") is not None:  # pyright: ignore[reportUnknownMemberType]  # frozen Phase 1 accepts unvalidated legacy values
            return True
    return False


def _is_cost_explicitly_configured(deployments: Sequence[DeploymentTypedDict]) -> bool:

    for deployment in deployments:
        model_id = (deployment.get("model_info") or _EMPTY_COST_ENTRY).get("id")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # frozen Phase 1 accepts unvalidated legacy values
        if model_id is None:
            continue
        raw_entry = litellm.model_cost.get(model_id, _EMPTY_COST_ENTRY)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # frozen Phase 1 accepts unvalidated legacy values
        if "input_cost_per_token" in raw_entry or "output_cost_per_token" in raw_entry:
            return True
    return False


def _has_explicit_positive_cost(deployments: Sequence[DeploymentTypedDict]) -> bool:

    for deployment in deployments:
        model_id = (deployment.get("model_info") or _EMPTY_COST_ENTRY).get("id")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # frozen Phase 1 accepts unvalidated legacy values
        if model_id is None:
            continue
        raw_entry = as_str_mapping(litellm.model_cost.get(model_id)) or _EMPTY_COST_ENTRY  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]  # frozen Phase 1 accepts unvalidated legacy values
        if (
            _is_positive_cost(raw_entry.get("input_cost_per_token"))
            or _is_positive_cost(raw_entry.get("output_cost_per_token"))
            or _is_positive_cost(_deployment_query_price(deployment["litellm_params"], raw_entry))
        ):
            return True
    return False


def _deployment_query_price(litellm_params: Mapping[str, object], cost_entry: Mapping[str, object]) -> float | None:
    provider: Final = litellm_params.get("custom_llm_provider") or str(litellm_params.get("model")).partition("/")[0]
    return decisions_request_price({**cost_entry, "litellm_provider": provider}, decisions_pricing_fields(cost_entry))


_EMPTY_COST_ENTRY: Final[Mapping[str, object]] = MappingProxyType({})


def _is_positive_cost(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


MODEL_DISCOVERY_ROUTES: Final = frozenset(
    {
        *MODEL_LIST_ROUTES,
        "/model/info",
        "/v1/model/info",
        "/v2/model/info",
        "/model_group/info",
        "/utils/model_info",
    }
)


BUDGET_ENFORCED_SIDE_EFFECT_ROUTES: Final = frozenset(
    {
        "/health",
        "/health/services",
        "/health/test_connection",
        "/auto_router/test_routing",
    }
)


MCP_DISCOVERY_ROUTES: Final = frozenset({"/mcp-rest/tools/list", "/v1/mcp/tools"})


MCP_ZERO_SPEND_JSONRPC_METHODS: Final = frozenset({"initialize", "notifications/initialized", "ping", "tools/list"})


MCP_TOOL_CALL_ROUTES: Final = frozenset({"/mcp/tools/call", "/mcp-rest/tools/call"})


def is_mcp_discovery_request(route: str, request_body: Mapping[str, object]) -> bool:
    if route in MCP_DISCOVERY_ROUTES:
        return True
    if route in MCP_TOOL_CALL_ROUTES or not (route == "/mcp" or route.startswith("/mcp/")):
        return False
    method: Final = request_body.get("method")
    return isinstance(method, str) and method in MCP_ZERO_SPEND_JSONRPC_METHODS


def route_skips_budget_checks(route: str) -> bool:
    if route in BUDGET_ENFORCED_SIDE_EFFECT_ROUTES:
        return False
    if route in MODEL_DISCOVERY_ROUTES:
        return True
    if RouteChecks.is_llm_api_route(route=route):
        return False
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (  # noqa: PLC0415  # pass-through endpoints import the auth stack
        InitPassThroughEndpointHelpers,
    )

    return not InitPassThroughEndpointHelpers.is_registered_pass_through_route(route=route)


def request_skips_budget_checks(
    route: str,
    model: str | list[str] | None,  # mutable-ok: Phase 1 public collection and identity contract
    llm_router: Router | None,
    *,
    request_body: Mapping[str, object] | None = None,
) -> bool:
    if route_skips_budget_checks(route=route):
        return True
    if request_body is not None and is_mcp_discovery_request(route=route, request_body=request_body):
        return True
    return is_model_cost_zero(model=model, llm_router=llm_router)


KEY_END_USER_BUDGET_ID_METADATA_FIELD: Final = "end_user_budget_id"


def get_key_end_user_budget_id(key_metadata: Mapping[str, object] | None) -> str | None:

    if key_metadata is None:
        return None
    budget_id: Final = key_metadata.get(KEY_END_USER_BUDGET_ID_METADATA_FIELD)
    return budget_id if isinstance(budget_id, str) and budget_id != "" else None


@with_service_target(AUTH_OBJECTS_TARGET)
async def get_default_end_user_budget(
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None = None,
    budget_id: str | None = None,
) -> LiteLLM_BudgetTable | None:

    default_budget_id: Final = budget_id if budget_id is not None else litellm.max_end_user_budget_id
    if prisma_client is None or default_budget_id is None:
        return None

    cache_key: Final = f"default_end_user_budget:{default_budget_id}"

    cached_budget: Final = await user_api_key_cache.async_get_cache(
        key=cache_key,
        model_type=LiteLLM_BudgetTable,
    )
    if cached_budget is not None:
        return cached_budget

    try:
        budget_record: Final = await _dictable_table(BudgetRepository(prisma_client), "budget").find_unique(
            where={"budget_id": default_budget_id}
        )

        if budget_record is None:
            verbose_proxy_logger.warning(
                "Default end user budget not found in database: %s",
                default_budget_id.replace("\r", "").replace("\n", ""),
            )
            return None

        _budget_obj: Final = LiteLLM_BudgetTable.model_validate(budget_record.dict())

        await user_api_key_cache.async_set_cache(
            key=cache_key,
            value=_budget_obj,
            model_type=LiteLLM_BudgetTable,
            ttl=get_management_object_ttl(user_api_key_cache),
        )

        return _budget_obj

    except Exception as e:
        verbose_proxy_logger.error("Error fetching default end user budget: %s", e)
        return None


async def get_team_member_default_budget(
    budget_id: str,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    raise_on_lookup_error: bool = False,
) -> LiteLLM_BudgetTable | None:
    raise AssertionError("Frozen reference raw IO must be supplied by the test: get_team_member_default_budget")


def resolve_effective_team_member_budget(
    member_budget: LiteLLM_BudgetTable | None,
    team_default_budget: LiteLLM_BudgetTable | None,
    now: datetime,
) -> float | None:
    if member_budget is not None and member_budget.max_budget is not None:
        return _effective_member_max(member_budget, now)
    if team_default_budget is None or team_default_budget.max_budget is None or team_default_budget.max_budget <= 0:
        return None
    return team_default_budget.max_budget + (
        _active_member_increase(member_budget, now) if member_budget is not None else 0.0
    )


async def resolve_default_end_user_budget(
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    key_end_user_budget_id: str | None,
    parent_otel_span: Span | None = None,
) -> LiteLLM_BudgetTable | None:

    if key_end_user_budget_id is not None:
        key_budget: Final = await get_default_end_user_budget(
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=parent_otel_span,
            budget_id=key_end_user_budget_id,
        )
        if key_budget is not None:
            return key_budget

    if litellm.max_end_user_budget_id is None:
        return None

    return await get_default_end_user_budget(
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=parent_otel_span,
    )


async def _apply_default_budget_to_end_user(
    end_user_obj: LiteLLM_EndUserTable,
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None = None,
    key_end_user_budget_id: str | None = None,
) -> LiteLLM_EndUserTable:

    if end_user_obj.budget_id is not None and end_user_obj.litellm_budget_table is not None:
        return end_user_obj

    if key_end_user_budget_id is None and litellm.max_end_user_budget_id is None:
        return end_user_obj

    default_budget: Final = await resolve_default_end_user_budget(
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        key_end_user_budget_id=key_end_user_budget_id,
        parent_otel_span=parent_otel_span,
    )

    if default_budget is None:
        return end_user_obj

    verbose_proxy_logger.debug(
        "Applied default budget %s to end user %s", default_budget.budget_id, end_user_obj.user_id
    )
    return end_user_obj.model_copy(update=MappingProxyType({"litellm_budget_table": default_budget}))


async def check_end_user_budget(
    end_user_obj: LiteLLM_EndUserTable,
    route: str,
) -> None:

    if RouteChecks.is_info_route(route):
        return

    if end_user_obj.litellm_budget_table is None:
        return

    end_user_budget: Final = end_user_obj.litellm_budget_table.max_budget
    if end_user_budget is None:
        return

    from litellm.proxy.proxy_server import get_current_spend

    end_user_spend: Final = await get_current_spend(
        counter_key=f"spend:end_user:{end_user_obj.user_id}",
        fallback_spend=end_user_obj.spend or 0.0,
        max_budget=end_user_budget,
        fallback_authoritative=True,
    )
    if end_user_spend > end_user_budget:
        raise litellm.BudgetExceededError(
            current_cost=end_user_spend,
            max_budget=end_user_budget,
            message=f"ExceededBudget: End User={end_user_obj.user_id} over budget. Spend={end_user_spend}, Budget={end_user_budget}",
            entity_type=Litellm_EntityType.END_USER.value,
            entity_id=end_user_obj.user_id,
        )


async def _load_end_user_restricted_registry(
    prisma_client: PrismaClient, user_api_key_cache: UserApiKeyCache
) -> frozenset[str] | None:
    raise AssertionError("Frozen reference raw IO must be supplied by the test: _load_end_user_restricted_registry")


async def _end_user_is_known_unrestricted(
    end_user_id: str,
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    token_end_user_max_budget: float | None,
    key_end_user_budget_id: str | None = None,
) -> bool:

    if (
        litellm.max_end_user_budget_id is not None
        or key_end_user_budget_id is not None
        or litellm.validate_end_user_id_in_db
        or token_end_user_max_budget is not None
    ):
        return False

    registry: Final = await _load_end_user_restricted_registry(
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
    )
    return registry is not None and end_user_id not in registry


async def get_end_user_object(
    end_user_id: str | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    route: str | None = "",
    parent_otel_span: Span | None = None,
    proxy_logging_obj: ProxyLogging | None = None,
    token_end_user_max_budget: float | None = None,
    key_end_user_budget_id: str | None = None,
) -> LiteLLM_EndUserTable | None:
    raise AssertionError("Frozen reference raw IO must be supplied by the test: get_end_user_object")


_END_USER_VALIDATION_NEGATIVE_TTL: Final = 60


_END_USER_VALIDATION_POSITIVE_TTL: Final = 300


@with_service_target(AUTH_OBJECTS_TARGET)
async def resolve_and_validate_end_user_id(
    raw_end_user_id: str | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None = None,
    proxy_logging_obj: ProxyLogging | None = None,
    route: str = "",
    key_end_user_budget_id: str | None = None,
) -> str | None:

    if raw_end_user_id is None:
        return None
    if not litellm.validate_end_user_id_in_db:
        return raw_end_user_id
    if prisma_client is None:
        return raw_end_user_id

    has_default_budget: Final = bool(litellm.max_end_user_budget_id) or key_end_user_budget_id is not None
    cache_key: Final = f"end_user_validation:{raw_end_user_id}"
    cached: Final = await _raw_cache(user_api_key_cache).async_get_cache(key=cache_key)
    if cached == "valid":
        return raw_end_user_id
    if cached == "invalid":
        return raw_end_user_id if has_default_budget else None

    is_valid: Final = await _end_user_id_exists_in_db(
        end_user_id=raw_end_user_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
        route=route,
    )

    await user_api_key_cache.async_set_cache(
        key=cache_key,
        value="valid" if is_valid else "invalid",
        ttl=(_END_USER_VALIDATION_POSITIVE_TTL if is_valid else _END_USER_VALIDATION_NEGATIVE_TTL),
    )

    return raw_end_user_id if is_valid or has_default_budget else None


async def _end_user_id_exists_in_db(
    end_user_id: str,
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None = None,
    proxy_logging_obj: ProxyLogging | None = None,
    route: str = "",
) -> bool:

    try:
        end_user_obj: Final = await get_end_user_object(
            end_user_id=end_user_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
            route=route,
        )
        if end_user_obj is not None:
            return True
    except Exception as e:
        verbose_proxy_logger.debug("end_user validation: get_end_user_object lookup failed: %s", e)

    try:
        user_obj: Final = await get_user_object(
            user_id=end_user_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            user_id_upsert=False,
            parent_otel_span=parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
            check_db_only=False,
            user_email=end_user_id if "@" in end_user_id else None,
        )
        if user_obj is not None:
            return True
    except Exception as e:
        verbose_proxy_logger.debug("end_user validation: get_user_object lookup failed: %s", e)

    return False


async def get_team_membership(
    user_id: str,
    team_id: str,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None = None,
    proxy_logging_obj: ProxyLogging | None = None,
    *,
    check_db_only: bool = False,
) -> Optional["LiteLLM_TeamMembership"]:
    raise AssertionError("Frozen reference raw IO must be supplied by the test: get_team_membership")


async def get_user_object(
    user_id: str | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    user_id_upsert: bool,
    parent_otel_span: Span | None = None,
    proxy_logging_obj: ProxyLogging | None = None,
    sso_user_id: str | None = None,
    user_email: str | None = None,
    check_db_only: bool | None = None,
) -> LiteLLM_UserTable | None:
    raise AssertionError("Frozen reference raw IO must be supplied by the test: get_user_object")


_MASKED_KEY_NAME_RE: Final = re.compile(r"sk-\.\.\.(?:[^\s\x00-\x1f\x7f-\x9f]{4})?")


def _apply_budget_exceeded_throttle(valid_token: UserAPIKeyAuth) -> bool:

    pct: Final = budget_throttle_percentage()
    if pct is None or not should_throttle_budget_exceeded(valid_token):
        return False
    valid_token.budget_throttle_pct = pct  # rebind-ok: preserve Phase 1 mutation order and identity
    return True


def send_token_budget_alert(
    valid_token: UserAPIKeyAuth,
    spend: float,
    proxy_logging_obj: ProxyLogging,
    user_obj: LiteLLM_UserTable | None,
) -> None:
    call_info: Final = CallInfo(
        token=valid_token.token,
        spend=spend,
        max_budget=valid_token.max_budget,
        soft_budget=valid_token.soft_budget,
        user_id=valid_token.user_id,
        team_id=valid_token.team_id,
        organization_id=valid_token.org_id,
        user_email=user_obj.user_email if user_obj is not None else None,
        key_alias=valid_token.key_alias,
        event_group=Litellm_EntityType.KEY,
    )
    asyncio.create_task(
        proxy_logging_obj.budget_alerts(
            type="token_budget",
            user_info=call_info,
        )
    )


def key_budget_exceeded_error(
    valid_token: UserAPIKeyAuth, spend: float, max_budget: float
) -> litellm.BudgetExceededError:

    key_label: Final = valid_token.key_alias or "key"
    key_descriptor: Final = (
        f"{key_label} ({valid_token.key_name})"
        if valid_token.key_name and _MASKED_KEY_NAME_RE.fullmatch(valid_token.key_name)
        else key_label
    )
    return litellm.BudgetExceededError(
        current_cost=spend,
        max_budget=max_budget,
        message=f"Budget has been exceeded! Key={key_descriptor} Current cost: {spend}, Max budget: {max_budget}",
        entity_type=Litellm_EntityType.KEY.value,
        entity_id=valid_token.token,
    )


async def virtual_key_max_budget_check(
    valid_token: UserAPIKeyAuth,
    proxy_logging_obj: ProxyLogging,
    user_obj: LiteLLM_UserTable | None = None,
) -> None:

    if valid_token.max_budget is not None:
        from litellm.proxy.proxy_server import get_current_spend

        spend: Final = await get_current_spend(
            counter_key=f"spend:key:{valid_token.token}",
            fallback_spend=valid_token.spend or 0.0,
            max_budget=valid_token.max_budget,
        )
        send_token_budget_alert(valid_token, spend, proxy_logging_obj, user_obj)

        if math.isfinite(valid_token.max_budget) and spend >= valid_token.max_budget:
            if _apply_budget_exceeded_throttle(valid_token):
                return
            raise key_budget_exceeded_error(valid_token, spend, valid_token.max_budget)


async def _virtual_key_multi_budget_check(
    valid_token: UserAPIKeyAuth,
):

    if not valid_token.budget_limits:  # pyright: ignore[reportUnknownMemberType]  # frozen Phase 1 accepts unvalidated legacy values
        return

    from litellm.proxy.proxy_server import get_current_spend

    for window in valid_token.budget_limits:  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # frozen Phase 1 accepts unvalidated legacy values
        w: dict = window if isinstance(window, dict) else window.model_dump()  # pyright: ignore[reportMissingTypeArgument, reportUnknownVariableType, reportUnnecessaryIsInstance]  # frozen Phase 1 accepts unvalidated legacy values  # mutable-ok: Phase 1 public collection and identity contract
        counter_key = f"spend:key:{valid_token.token}:window:{w['budget_duration']}"
        window_spend = await get_current_spend(
            counter_key=counter_key,
            fallback_spend=0.0,
            max_budget=w["max_budget"],  # pyright: ignore[reportUnknownArgumentType]  # frozen Phase 1 accepts unvalidated legacy values
            window_entity_type="Key",
            window_entity_id=valid_token.token,
            window_duration=str(w["budget_duration"]),  # pyright: ignore[reportUnknownArgumentType]  # frozen Phase 1 accepts unvalidated legacy values
            window_start=get_budget_window_start(w),  # pyright: ignore[reportUnknownArgumentType]  # frozen Phase 1 accepts unvalidated legacy values
        )
        if math.isfinite(w["max_budget"]) and window_spend >= w["max_budget"]:  # pyright: ignore[reportUnknownArgumentType]  # frozen Phase 1 accepts unvalidated legacy values
            raise litellm.BudgetExceededError(
                current_cost=window_spend,
                max_budget=w["max_budget"],  # pyright: ignore[reportUnknownArgumentType]  # frozen Phase 1 accepts unvalidated legacy values
                message=(
                    f"ExceededBudget: Key over {w['budget_duration']} budget. "
                    f"Spend=${window_spend:.4f}, Limit=${w['max_budget']:.2f}"
                ),
                entity_type=Litellm_EntityType.KEY.value,
                entity_id=valid_token.token,
            )


async def team_max_budget_check(
    team_object: LiteLLM_TeamTable | None,
    valid_token: UserAPIKeyAuth | None,
    proxy_logging_obj: ProxyLogging,
) -> None:

    if team_object is not None and team_object.max_budget is not None:
        from litellm.proxy.proxy_server import get_current_spend

        spend: Final = await get_current_spend(
            counter_key=f"spend:team:{team_object.team_id}",
            fallback_spend=team_object.spend or 0.0,
            max_budget=team_object.max_budget,
        )

        if math.isfinite(team_object.max_budget) and spend > team_object.max_budget:
            if valid_token:
                call_info: Final = CallInfo(
                    token=valid_token.token,
                    spend=spend,
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
                current_cost=spend,
                max_budget=team_object.max_budget,
                message=f"Budget has been exceeded! Team={team_object.team_id} Current cost: {spend}, Max budget: {team_object.max_budget}",
                entity_type=Litellm_EntityType.TEAM.value,
                entity_id=team_object.team_id,
            )


async def _team_multi_budget_check(
    team_object: LiteLLM_TeamTable | None,
):

    if team_object is None or not team_object.budget_limits:
        return

    from litellm.proxy.proxy_server import get_current_spend

    for window in team_object.budget_limits:
        w: dict = window if isinstance(window, dict) else window.model_dump()  # pyright: ignore[reportMissingTypeArgument]  # frozen Phase 1 accepts unvalidated legacy values  # mutable-ok: Phase 1 public collection and identity contract
        counter_key = f"spend:team:{team_object.team_id}:window:{w['budget_duration']}"
        window_spend = await get_current_spend(
            counter_key=counter_key,
            fallback_spend=0.0,
            max_budget=w["max_budget"],  # pyright: ignore[reportUnknownArgumentType]  # frozen Phase 1 accepts unvalidated legacy values
            window_entity_type="Team",
            window_entity_id=team_object.team_id,
            window_duration=str(w["budget_duration"]),  # pyright: ignore[reportUnknownArgumentType]  # frozen Phase 1 accepts unvalidated legacy values
            window_start=get_budget_window_start(w),  # pyright: ignore[reportUnknownArgumentType]  # frozen Phase 1 accepts unvalidated legacy values
        )
        if math.isfinite(w["max_budget"]) and window_spend >= w["max_budget"]:  # pyright: ignore[reportUnknownArgumentType]  # frozen Phase 1 accepts unvalidated legacy values
            raise litellm.BudgetExceededError(
                current_cost=window_spend,
                max_budget=w["max_budget"],  # pyright: ignore[reportUnknownArgumentType]  # frozen Phase 1 accepts unvalidated legacy values
                message=(
                    f"ExceededBudget: Team={team_object.team_id} over {w['budget_duration']} budget. "
                    f"Spend=${window_spend:.4f}, Limit=${w['max_budget']:.2f}"
                ),
                entity_type=Litellm_EntityType.TEAM.value,
                entity_id=team_object.team_id,
            )


class ModelMaxBudgetHolder(Protocol):
    @property
    def model_max_budget(self) -> Mapping[str, object] | None: ...


def model_max_budget_of(holder: ModelMaxBudgetHolder) -> Mapping[str, object] | None:
    return holder.model_max_budget


def _active_member_increase(self: LiteLLM_BudgetTable, now: datetime) -> float:
    if self.temp_budget_increase is None or self.temp_budget_expiry is None:
        return 0.0
    expiry: Final = (
        self.temp_budget_expiry.replace(tzinfo=timezone.utc)
        if self.temp_budget_expiry.tzinfo is None
        else self.temp_budget_expiry
    )
    return 0.0 if expiry <= now else self.temp_budget_increase


def _effective_member_max(self: LiteLLM_BudgetTable, now: datetime) -> float | None:
    if self.max_budget is None:
        return None
    return self.max_budget + _active_member_increase(self, now)


_WINDOW_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))


def _coerce_window(window: object) -> Mapping[str, object]:
    if isinstance(window, Mapping):
        return window  # pyright: ignore[reportUnknownVariableType]  # frozen Phase 1 accepts unvalidated legacy values
    if isinstance(window, str):
        try:
            return _WINDOW_OBJECT.validate_python(json.loads(window))
        except Exception:
            return {}
    model_dump: Final = getattr(window, "model_dump", None)
    if not callable(model_dump):
        return {}
    dumped: Final[object] = model_dump()
    return dumped if isinstance(dumped, Mapping) else {}  # pyright: ignore[reportUnknownVariableType]  # frozen Phase 1 accepts unvalidated legacy values


def get_budget_window_start(window: object) -> datetime | None:
    window_dict: Final = _coerce_window(window)
    budget_duration: Final = window_dict.get("budget_duration")
    if budget_duration is None:
        return None
    try:
        duration_seconds: Final = duration_in_seconds(str(budget_duration))
    except Exception:
        return None

    reset_at = _coerce_datetime(window_dict.get("reset_at"))  # rebind-ok: preserve Phase 1 mutation order and identity
    if reset_at is None:
        return datetime.now(timezone.utc) - timedelta(seconds=duration_seconds)
    if reset_at.tzinfo is None:
        reset_at = reset_at.replace(tzinfo=timezone.utc)  # rebind-ok: preserve Phase 1 mutation order and identity
    return reset_at - timedelta(seconds=duration_seconds)


def _coerce_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


__all__ = (
    "_raw_cache",
    "_get_router_zero_cost_cache",
    "_deployments_served_for",
    "_has_ptu_flat_cost",
    "_is_cost_explicitly_configured",
    "_has_explicit_positive_cost",
    "_deployment_query_price",
    "_is_positive_cost",
    "_apply_default_budget_to_end_user",
    "_load_end_user_restricted_registry",
    "_end_user_is_known_unrestricted",
    "_end_user_id_exists_in_db",
    "_apply_budget_exceeded_throttle",
    "_virtual_key_multi_budget_check",
    "_team_multi_budget_check",
    "_active_member_increase",
    "_effective_member_max",
    "_coerce_window",
    "_coerce_datetime",
)
