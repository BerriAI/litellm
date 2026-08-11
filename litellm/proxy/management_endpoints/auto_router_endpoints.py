"""
AUTO ROUTER MANAGEMENT ENDPOINTS

POST /auto_router/test_routing - Route one prompt through an unsaved complexity-router config
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Final

from pydantic import BaseModel, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import BudgetExceededError
from litellm.proxy._types import (
    CommonProxyErrors,
    LiteLLM_TeamTable,
    LitellmUserRoles,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import (
    _virtual_key_max_budget_check,
    can_key_call_resolved_model,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.db.autorouter_session_rollup import AUTOROUTER_BENCHMARKS_SQL
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.repositories.team_repository import TeamRepository
from litellm.router_strategy.complexity_router import ComplexityRouter
from litellm.types.management_endpoints.auto_router_endpoints import (
    AutoRouterBenchmarkGroup,
    AutoRouterBenchmarksResponse,
    AutoRouterBenchmarkTotals,
    AutoRouterCacheBucket,
    AutoRouterCacheStats,
    AutoRouterRoutingTestRequest,
    AutoRouterRoutingTestResponse,
    RequestComplexityRouterConfig,
)

if TYPE_CHECKING:
    from fastapi import APIRouter, Depends, HTTPException, Query, status

    from litellm.router import Router
else:
    try:
        from fastapi import APIRouter, Depends, HTTPException, Query, status
    except ImportError:
        # fastapi is only required for proxy, not for SDK usage
        pass

router: Final = APIRouter()


async def _authorize_routing_test(user_api_key_dict: UserAPIKeyAuth, team_id: str | None) -> None:
    """Allow exactly the callers who could create this router.

    Routing a prompt can spend money (an `llm` classifier config calls its classifier, a
    semantic config embeds the prompt), so this is gated like a write rather than a read:
    a proxy admin, or a team admin naming their own team, matching /model/new.
    """
    from litellm.proxy.management_endpoints.model_management_endpoints import (
        ModelManagementAuthChecks,
    )
    from litellm.proxy.proxy_server import premium_user, prisma_client

    if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN:
        return

    if team_id is None:
        raise HTTPException(
            status_code=403,
            detail={  # mutable-ok: HTTPException detail must be a plain mapping to keep this route's {"error": ...} response shape
                "error": f"User does not have permission to test an auto router. Your role={user_api_key_dict.user_role}. Test as a PROXY_ADMIN, or as a team admin by specifying a team_id."
            },
        )

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={  # mutable-ok: HTTPException detail must be a plain mapping
                "error": CommonProxyErrors.db_not_connected_error.value
            },
        )

    team_row: Final = await TeamRepository(prisma_client).table.find_unique(
        where={"team_id": team_id},  # mutable-ok: Prisma query filters are dict-shaped
    )
    if team_row is None:
        raise HTTPException(
            status_code=400,
            detail={  # mutable-ok: HTTPException detail must be a plain mapping
                "error": f"Team id={team_id} does not exist in db"
            },
        )

    ModelManagementAuthChecks.can_user_make_team_model_call(
        team_id=team_id,
        user_api_key_dict=user_api_key_dict,
        team_obj=LiteLLM_TeamTable.model_validate(team_row.model_dump()),
        premium_user=premium_user,
    )


def _models_this_test_can_call(config: RequestComplexityRouterConfig) -> tuple[str, ...]:
    """The models the routing test itself would send a request to, and so spend on.

    Excludes every tier's models: the prompt is never sent to the model it routed to.
    """
    return tuple(
        model
        for model in (
            config.classifier_llm_config.model
            if config.classifier_type == "llm" and config.classifier_llm_config is not None
            else None,
            config.embedding_model if config.semantic_keyword_matching else None,
        )
        if model is not None
    )


async def _authorize_models_this_test_can_call(
    config: RequestComplexityRouterConfig,
    user_api_key_dict: UserAPIKeyAuth,
    llm_router: "Router",
) -> None:
    """Hold a classifier or embedding call to the caller's model access and key budget.

    Those calls go through the router rather than through /v1/chat/completions, so the model
    checks a real request gets in user_api_key_auth would otherwise be skipped, letting a
    caller spend on a model their key cannot call, and this route is not an LLM API route, so
    the key's own budget is not checked either. Test Connection gets both for free by routing
    its calls through the proxy. Team and member budgets are already enforced on every route.
    """
    models: Final = _models_this_test_can_call(config)
    if not models:
        return

    from litellm.proxy.proxy_server import proxy_logging_obj

    for model in models:
        await can_key_call_resolved_model(
            model=model,
            llm_model_list=llm_router.model_list,
            valid_token=user_api_key_dict,
            llm_router=llm_router,
        )

    try:
        await _virtual_key_max_budget_check(
            valid_token=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
        )
    except BudgetExceededError as e:
        raise ProxyException(
            message=e.message,
            type=ProxyErrorTypes.budget_exceeded,
            param=None,
            code=status.HTTP_400_BAD_REQUEST,
        ) from e


@router.post(
    "/auto_router/test_routing",
    tags=["model management"],  # mutable-ok: fastapi's decorator signature types tags as a list
    dependencies=[Depends(user_api_key_auth)],  # mutable-ok: fastapi's decorator signature types dependencies as a list
    response_model=AutoRouterRoutingTestResponse,
    status_code=status.HTTP_200_OK,
)
async def preview_auto_router_routing(
    data: AutoRouterRoutingTestRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> AutoRouterRoutingTestResponse:
    """
    Route a single prompt through a complexity-router config and report where it landed.

    Answers "which model would this prompt get?" for a config that only exists in a form,
    so an auto router can be checked before it is created. The prompt is classified by the
    same pre-routing hook a live request runs, then dropped: nothing is sent to the model it
    routed to, and no auto router is created. A heuristic config therefore spends nothing, while
    an `llm` classifier or semantic keyword matching bills its classifier/embedding call to the
    calling key, like Test Connection does.

    **Example Request:**
    ```json
    {
        "prompt": "think step by step about how to shard this table",
        "complexity_router_config": {
            "tiers": {"SIMPLE": ["gpt-4o-mini"], "REASONING": ["o3"]},
            "classifier_type": "heuristic"
        }
    }
    ```
    """
    from litellm.proxy.proxy_server import llm_router

    await _authorize_routing_test(user_api_key_dict=user_api_key_dict, team_id=data.team_id)

    if llm_router is None:
        raise HTTPException(
            status_code=500,
            detail={  # mutable-ok: HTTPException detail must be a plain mapping
                "error": CommonProxyErrors.no_llm_router.value
            },
        )

    await _authorize_models_this_test_can_call(
        config=data.complexity_router_config,
        user_api_key_dict=user_api_key_dict,
        llm_router=llm_router,
    )

    complexity_router: Final = ComplexityRouter(
        model_name=data.router_name,
        litellm_router_instance=llm_router,
        complexity_router_config=data.complexity_router_config.model_dump(exclude_none=True),
        default_model=data.default_model,
        derive_savings_baseline=False,
    )

    request_kwargs: Final = LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
        data={"metadata": {}},  # mutable-ok: the request-metadata helper takes and returns request kwargs as a dict
        user_api_key_dict=user_api_key_dict,
        _metadata_variable_name="metadata",
    )

    try:
        hook_response: Final = await complexity_router.async_pre_routing_hook(
            model=data.router_name,
            request_kwargs=request_kwargs,
            messages=[  # mutable-ok: the routing hook's signature takes a list of message dicts
                {"role": "user", "content": data.prompt},  # mutable-ok: a message is dict-shaped
            ],
        )
    except Exception as e:  # noqa: BLE001 -- surfaces any classifier/plugin failure to the caller as a 400 instead of a 500, since the config under test is caller input
        verbose_proxy_logger.exception("Auto router routing test failed. Due to error - %s", e)
        raise HTTPException(
            status_code=400,
            detail={  # mutable-ok: HTTPException detail must be a plain mapping
                "error": f"Could not route this prompt: {e}"
            },
        ) from e

    if hook_response is None or hook_response.routing_decision is None:
        raise HTTPException(
            status_code=400,
            detail={  # mutable-ok: HTTPException detail must be a plain mapping
                "error": "The router made no decision for this prompt. Check that at least one tier has a model."
            },
        )

    return AutoRouterRoutingTestResponse(
        routed_model=hook_response.model,
        routed_model_configured=hook_response.model in frozenset(llm_router.get_model_names()),
        routing_decision=hook_response.routing_decision,
    )


class _SessionAggRow(BaseModel):
    router_name: str
    router_type: str
    tier_turns: Mapping[str, int]
    sessions: int
    turns: int
    unordered_turns: int
    covered_turns: int
    cache_hits: int
    same_model_turns: int
    same_model_hits: int
    first_visit_turns: int
    first_visit_hits: int
    return_turns: int
    return_hits: int
    return_expired_misses: int
    return_within_ttl_misses: int
    ttl_5m_turns: int
    ttl_1h_turns: int
    total_tokens: int
    spend: float
    saved_spend: float
    session_seconds: float


_SESSION_AGG_ROWS: Final = TypeAdapter(list[_SessionAggRow])


def _parse_benchmark_day(value: str) -> datetime:
    try:
        parsed: Final = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {value}. Expected: 'YYYY-MM-DD'")
    return parsed.replace(tzinfo=None)


def _pct(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round(100.0 * numerator / denominator, 1)


def _cache_bucket(turns: int, hits: int) -> AutoRouterCacheBucket:
    return AutoRouterCacheBucket(turns=turns, hits=hits, hit_rate_pct=_pct(hits, turns))


def _benchmark_totals(row: _SessionAggRow) -> AutoRouterBenchmarkTotals:
    return_misses: Final = row.return_turns - row.return_hits
    baseline_spend: Final = row.spend + row.saved_spend
    sessions: Final = row.sessions
    return AutoRouterBenchmarkTotals(
        sessions=sessions,
        turns=row.turns,
        avg_turns_per_session=row.turns / sessions if sessions else 0.0,
        avg_session_seconds=row.session_seconds / sessions if sessions else 0.0,
        avg_tokens_per_session=row.total_tokens / sessions if sessions else 0.0,
        spend=row.spend,
        saved_spend=row.saved_spend,
        baseline_spend=baseline_spend,
        saved_pct=_pct(row.saved_spend, baseline_spend),
        saved_per_session=row.saved_spend / sessions if sessions else 0.0,
        cache=AutoRouterCacheStats(
            coverage_pct=_pct(row.covered_turns, row.turns),
            hit_rate_pct=_pct(row.cache_hits, row.covered_turns),
            same_model=_cache_bucket(row.same_model_turns, row.same_model_hits),
            first_visit=_cache_bucket(row.first_visit_turns, row.first_visit_hits),
            return_to_tier=_cache_bucket(row.return_turns, row.return_hits),
            unordered_turns=row.unordered_turns,
            return_misses_expired=row.return_expired_misses,
            return_misses_within_ttl=row.return_within_ttl_misses,
            return_misses_unknown=max(return_misses - row.return_expired_misses - row.return_within_ttl_misses, 0),
            ttl_5m_turns=row.ttl_5m_turns,
            ttl_1h_turns=row.ttl_1h_turns,
        ),
    )


def _summed_agg_row(rows: Sequence[_SessionAggRow]) -> _SessionAggRow:
    return _SessionAggRow(
        router_name="",
        router_type="",
        tier_turns=MappingProxyType({}),
        sessions=sum(row.sessions for row in rows),
        turns=sum(row.turns for row in rows),
        unordered_turns=sum(row.unordered_turns for row in rows),
        covered_turns=sum(row.covered_turns for row in rows),
        cache_hits=sum(row.cache_hits for row in rows),
        same_model_turns=sum(row.same_model_turns for row in rows),
        same_model_hits=sum(row.same_model_hits for row in rows),
        first_visit_turns=sum(row.first_visit_turns for row in rows),
        first_visit_hits=sum(row.first_visit_hits for row in rows),
        return_turns=sum(row.return_turns for row in rows),
        return_hits=sum(row.return_hits for row in rows),
        return_expired_misses=sum(row.return_expired_misses for row in rows),
        return_within_ttl_misses=sum(row.return_within_ttl_misses for row in rows),
        ttl_5m_turns=sum(row.ttl_5m_turns for row in rows),
        ttl_1h_turns=sum(row.ttl_1h_turns for row in rows),
        total_tokens=sum(row.total_tokens for row in rows),
        spend=sum(row.spend for row in rows),
        saved_spend=sum(row.saved_spend for row in rows),
        session_seconds=sum(row.session_seconds for row in rows),
    )


@router.get(
    "/auto_router/benchmarks",
    tags=("auto router",),
    dependencies=(Depends(user_api_key_auth),),
    response_model=AutoRouterBenchmarksResponse,
)
async def get_auto_router_benchmarks(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    start_date: Annotated[
        str | None, Query(description="YYYY-MM-DD UTC, inclusive (defaults to 30 days before end_date)")
    ] = None,
    end_date: Annotated[str | None, Query(description="YYYY-MM-DD UTC, inclusive (defaults to today)")] = None,
) -> AutoRouterBenchmarksResponse:
    """
    Benchmarks for the auto-router dashboard: session shape, savings against the configured
    baseline, and prompt-caching behaviour bucketed by what the router did.

    Reads the LiteLLM_AutoRouterSession rollup, folded once per request at spend-write time,
    so this endpoint never scans LiteLLM_SpendLogs. A session is in the window when it
    overlaps it: its last turn is on or after start_date and its first turn is on or before
    end_date. Overall hit rate is over telemetry-bearing turns; each bucket's hit rate is
    over that bucket's turns.
    """
    from litellm.proxy.proxy_server import prisma_client

    if user_api_key_dict.user_role not in (
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    ):
        raise HTTPException(
            status_code=403,
            detail="Only proxy admin roles can view auto-router benchmarks across the deployment",
        )
    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)

    end_day: Final = (
        _parse_benchmark_day(end_date)
        if end_date
        else datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    )
    start_day: Final = _parse_benchmark_day(start_date) if start_date else end_day - timedelta(days=30)
    if end_day < start_day:
        raise HTTPException(status_code=400, detail="end_date must not be earlier than start_date")

    raw_rows: Final = await prisma_client.db.query_raw(
        AUTOROUTER_BENCHMARKS_SQL,
        start_day.isoformat(),
        (end_day + timedelta(days=1)).isoformat(),
    )
    rows: Final = _SESSION_AGG_ROWS.validate_python(raw_rows or ())
    groups: Final = tuple(
        AutoRouterBenchmarkGroup(
            router_name=row.router_name,
            router_type=row.router_type,
            tier_turns=row.tier_turns,
            **_benchmark_totals(row).model_dump(),
        )
        for row in rows
    )
    return AutoRouterBenchmarksResponse(
        start_date=start_day.strftime("%Y-%m-%d"),
        end_date=end_day.strftime("%Y-%m-%d"),
        routers_in_scope=len(rows),
        totals=_benchmark_totals(_summed_agg_row(rows)),
        groups=groups,
    )
