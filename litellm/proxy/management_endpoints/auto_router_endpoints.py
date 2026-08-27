"""
AUTO ROUTER MANAGEMENT ENDPOINTS

POST /auto_router/test_routing - Route one prompt through an unsaved complexity-router config
POST /auto_router/validate_complexity_router_config - Dry-run the complexity-router write gate without saving
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from itertools import groupby
from operator import attrgetter
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Final, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, TypeAdapter, field_validator

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import BudgetExceededError
from litellm.litellm_core_utils.llm_judge import router_resolves_model
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
from litellm.repositories.base_repository import SupportsModelDump
from litellm.repositories.team_repository import TeamRepository
from litellm.router_strategy.complexity_router import ComplexityRouter
from litellm.router_utils.auto_router_model_naming import classify_strategy_router_model
from litellm.types.management_endpoints.auto_router_endpoints import (
    SHADOW_EVAL_TURN_VALVE,
    AutoRouterBenchmarkGroup,
    AutoRouterBenchmarksResponse,
    AutoRouterBenchmarkTotals,
    AutoRouterCacheBucket,
    AutoRouterCacheStats,
    AutoRouterRoutingTestRequest,
    AutoRouterRoutingTestResponse,
    ComplexityRouterConfigValidationRequest,
    ComplexityRouterConfigValidationResponse,
    RequestComplexityRouterConfig,
    ShadowEvalDirection,
    ShadowEvalJobKeyResponse,
    ShadowEvalJobResponse,
    ShadowEvalResult,
    ShadowEvalSlice,
    StartShadowEvalRequest,
)

if TYPE_CHECKING:
    from fastapi import APIRouter, Depends, HTTPException, Query, status

    from litellm.proxy.utils import PrismaClient
    from litellm.router import Router
else:
    try:
        from fastapi import APIRouter, Depends, HTTPException, Query, status
    except ImportError:
        # fastapi is only required for proxy, not for SDK usage
        pass

router: Final = APIRouter()


class _TeamTable(Protocol):
    async def find_unique(self, *, where: Mapping[str, object]) -> SupportsModelDump | None: ...


class _VerificationTokenRow(Protocol):
    @property
    def token(self) -> str: ...

    @property
    def key_alias(self) -> str | None: ...

    @property
    def key_name(self) -> str | None: ...


class _VerificationTokenTable(Protocol):
    async def find_unique(self, *, where: Mapping[str, object]) -> _VerificationTokenRow | None: ...

    async def find_many(self, *, where: Mapping[str, object]) -> Sequence[_VerificationTokenRow]: ...


class _ShadowEvalJobRow(Protocol):
    @property
    def id(self) -> str: ...


class _ShadowEvalJobTable(Protocol):
    async def find_many(self, *, where: Mapping[str, object]) -> Sequence[_ShadowEvalJobRow]: ...

    async def create_many(self, data: Sequence[Mapping[str, object]]) -> int: ...


class _ShadowEvalAttemptRow(Protocol):
    @property
    def error(self) -> str | None: ...


class _ShadowEvalAttemptTable(Protocol):
    async def find_first(
        self, *, where: Mapping[str, object], order: Mapping[str, str]
    ) -> _ShadowEvalAttemptRow | None: ...


def _team_table(prisma_client: "PrismaClient") -> _TeamTable:
    return TeamRepository(prisma_client).table


def _verification_tokens(prisma_client: "PrismaClient") -> _VerificationTokenTable:
    return prisma_client.db.litellm_verificationtoken


def _shadow_eval_jobs(prisma_client: "PrismaClient") -> _ShadowEvalJobTable:
    return prisma_client.db.litellm_shadowevaljob


def _shadow_eval_attempts(prisma_client: "PrismaClient") -> _ShadowEvalAttemptTable:
    return prisma_client.db.litellm_shadowevalattempt


async def _query_raw(prisma_client: "PrismaClient", query: str, *args: object) -> Sequence[Mapping[str, object]]:
    return await prisma_client.db.query_raw(query, *args)


async def _authorize_router_dry_run(user_api_key_dict: UserAPIKeyAuth, team_id: str | None) -> None:
    """Allow exactly the callers who could create this router.

    Both dry runs are gated like the write they rehearse rather than as reads: a proxy
    admin, or a team admin naming their own team, matching /model/new. Routing a test
    prompt can also spend money (an `llm` classifier config calls its classifier, a
    semantic config embeds the prompt), so a read-level gate would be too loose anyway.
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
                "error": f"User does not have permission to dry-run an auto router. Your role={user_api_key_dict.user_role}. Call as a PROXY_ADMIN, or as a team admin by specifying a team_id."
            },
        )

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={  # mutable-ok: HTTPException detail must be a plain mapping
                "error": CommonProxyErrors.db_not_connected_error.value
            },
        )

    team_row: Final = await _team_table(prisma_client).find_unique(
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
            if config.uses_llm_classifier and config.classifier_llm_config is not None
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
    "/auto_router/validate_complexity_router_config",
    tags=["model management"],  # mutable-ok: fastapi's decorator signature types tags as a list
    dependencies=[Depends(user_api_key_auth)],  # mutable-ok: fastapi's decorator signature types dependencies as a list
    response_model=ComplexityRouterConfigValidationResponse,
    status_code=status.HTTP_200_OK,
)
async def validate_complexity_router_config(
    data: ComplexityRouterConfigValidationRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> ComplexityRouterConfigValidationResponse:
    """
    Validate a complexity-router config without saving it.

    Runs the same check every write path runs (the router's own pydantic model), so a form can
    show the backend's exact verdict while the operator is still editing rather than after a
    rejected save. Gated exactly like the save it rehearses: a proxy admin, or a team admin
    naming their own team. Nothing is created, routed, or billed.
    """
    await _authorize_router_dry_run(user_api_key_dict=user_api_key_dict, team_id=data.team_id)

    from litellm.router_utils.auto_router_model_naming import (
        validate_complexity_router_config_write,
    )

    error: Final = validate_complexity_router_config_write(data.complexity_router_config)
    return ComplexityRouterConfigValidationResponse(valid=error is None, error=error)


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
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
        user_model,
    )
    from litellm.proxy.utils import get_available_models_for_user

    await _authorize_router_dry_run(user_api_key_dict=user_api_key_dict, team_id=data.team_id)

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

    available_models: Final = await get_available_models_for_user(
        user_api_key_dict=user_api_key_dict,
        llm_router=llm_router,
        general_settings=general_settings,
        user_model=user_model,
        prisma_client=prisma_client,
        proxy_logging_obj=proxy_logging_obj,
        team_id=data.team_id,
        user_api_key_cache=user_api_key_cache,
    )
    return AutoRouterRoutingTestResponse(
        routed_model=hook_response.model,
        routed_model_configured=hook_response.model in frozenset(available_models),
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


def _benchmark_group(row: _SessionAggRow) -> AutoRouterBenchmarkGroup:
    totals: Final = _benchmark_totals(row)
    return AutoRouterBenchmarkGroup(
        router_name=row.router_name,
        router_type=row.router_type,
        tier_turns=row.tier_turns,
        sessions=totals.sessions,
        turns=totals.turns,
        avg_turns_per_session=totals.avg_turns_per_session,
        avg_session_seconds=totals.avg_session_seconds,
        avg_tokens_per_session=totals.avg_tokens_per_session,
        spend=totals.spend,
        saved_spend=totals.saved_spend,
        baseline_spend=totals.baseline_spend,
        saved_pct=totals.saved_pct,
        saved_per_session=totals.saved_per_session,
        cache=totals.cache,
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


def _strategy_router_key(deployment: object) -> tuple[str, str] | None:
    """``(model_name, kind)`` for a deployment whose routing the session rollup records.

    Kinds come from ``classify_strategy_router_model``, the same rule the Router registers a
    deployment by, so this arm cannot disagree with the arm that stamped ``router_type`` onto
    the session rows. Semantic auto-routers return None: they record no routing decision, so
    they can never own a session row, and ``AutoRouterBenchmarkGroup.router_type`` has no
    value for them. A permanent zero would read as "no traffic" rather than "not instrumented".
    """
    if not isinstance(deployment, Mapping):
        return None
    litellm_params: Final = deployment.get("litellm_params")
    router_name: Final = deployment.get("model_name")
    if not (isinstance(litellm_params, Mapping) and isinstance(router_name, str) and router_name):
        return None
    model: Final = litellm_params.get("model")
    if not isinstance(model, str):
        return None
    kind: Final = classify_strategy_router_model(model)
    return None if kind is None or kind == "semantic" else (router_name, kind)


def _idle_router_groups(
    llm_router: "Router | None", covered: frozenset[tuple[str, str]]
) -> tuple[AutoRouterBenchmarkGroup, ...]:
    """Zeroed groups for configured strategy routers the window's traffic did not cover.

    The dashboard's router picker has to list a router the moment it is created rather than
    once it has spent something, so the registry drives the list and the rollup only supplies
    the measures. ``_summed_agg_row`` over no sessions is already the zero element of the
    fold, so a group with every measure at zero costs one relabel rather than a literal that
    would go stale the next time the response grows a field.
    """
    if llm_router is None:
        return ()
    zero: Final = _summed_agg_row(())
    idle: Final = frozenset(
        key
        for key in (_strategy_router_key(deployment) for deployment in llm_router.model_list or ())
        if key is not None and key not in covered
    )
    return tuple(
        _benchmark_group(zero.model_copy(update=MappingProxyType({"router_name": name, "router_type": kind})))
        for name, kind in sorted(idle)
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

    The rollup supplies the measures, never the list. Which routers appear comes from the
    model registry, so one shows up as soon as it is configured and reads zero until it
    serves traffic, and `routers_in_scope` counts those too rather than only the routers the
    window recorded.
    """
    from litellm.proxy.proxy_server import llm_router, prisma_client

    _require_admin_viewer(user_api_key_dict, "view auto-router benchmarks across the deployment")
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

    raw_rows: Final = await _query_raw(
        prisma_client,
        AUTOROUTER_BENCHMARKS_SQL,
        start_day.isoformat(),
        (end_day + timedelta(days=1)).isoformat(),
    )
    rows: Final = _SESSION_AGG_ROWS.validate_python(raw_rows or ())
    groups: Final = (
        *(_benchmark_group(row) for row in rows),
        *_idle_router_groups(llm_router, frozenset((row.router_name, row.router_type) for row in rows)),
    )
    return AutoRouterBenchmarksResponse(
        start_date=start_day.strftime("%Y-%m-%d"),
        end_date=end_day.strftime("%Y-%m-%d"),
        routers_in_scope=len(groups),
        totals=_benchmark_totals(_summed_agg_row(rows)),
        groups=groups,
    )


# ---------------------------------------------------------------------------
# Shadow eval: pre-adoption evaluation of an auto-router against live traffic.
# The job row is immutable config plus stopped_at; status, counts, spend, and errors
# are derived from the append-only attempt rows, so reads here are aggregations
# bounded by each job's max_turns through the attempt table's job_id index.
# ---------------------------------------------------------------------------


def _require_admin_viewer(user_api_key_dict: UserAPIKeyAuth, action: str) -> None:
    if user_api_key_dict.user_role not in (
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    ):
        raise HTTPException(status_code=403, detail=f"Only proxy admin roles can {action}")


def _require_admin_writer(user_api_key_dict: UserAPIKeyAuth, action: str) -> None:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail=f"Only a proxy admin can {action}")


def _is_configured_pre_routing_strategy(llm_router: "Router", router_name: str) -> bool:
    return any(
        router_name in registry
        for registry in (
            llm_router.auto_routers,
            llm_router.complexity_routers,
            llm_router.adaptive_routers,
            llm_router.quality_routers,
        )
    )


def _validate_plain_model(llm_router: "Router | None", model: str, field_name: str) -> None:
    """Reject a model the dispatch path cannot resolve, at start rather than as a silently
    growing error count once the job is already sampling and billing. Both the judge and a
    reverse job's baseline must be plain models: an auto-router in either slot would
    re-route per turn, so the comparison would have no fixed arm to attribute results to."""
    if llm_router is not None and _is_configured_pre_routing_strategy(llm_router, model):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} '{model}' is an auto-router; it must be a plain model",
        )
    if router_resolves_model(llm_router, model):
        return
    import litellm

    try:
        litellm.get_llm_provider(model=model)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{field_name} '{model}' is neither a model configured on this proxy nor a "
                "provider-qualified public model name (e.g. 'anthropic/claude-sonnet-5')"
            ),
        ) from e


def _is_unique_violation(error: Exception) -> bool:
    """Whether a Prisma create failed on a unique index. One active job per key and
    direction lives in a partial unique index (raw SQL in the migration; schema.prisma
    cannot express partial indexes), so the read-then-create check above it is advisory:
    two concurrent starts pass the read, and the loser must surface as the same 409
    rather than a 500."""
    try:
        from prisma.errors import UniqueViolationError
    except ImportError:
        return "unique constraint" in str(error).lower() or "P2002" in str(error)
    return isinstance(error, UniqueViolationError)


class _AttemptAggRow(BaseModel):
    grp: str
    turn_count: int
    real_wins: int
    shadow_wins: int
    ties: int
    avg_confidence: float | None


_ATTEMPT_AGG_ROWS: Final = TypeAdapter(list[_AttemptAggRow])

_ATTEMPT_AGG_SELECT: Final = """
    COUNT(*)::int AS turn_count,
    COUNT(*) FILTER (WHERE outcome = 'real')::int AS real_wins,
    COUNT(*) FILTER (WHERE outcome = 'shadow')::int AS shadow_wins,
    COUNT(*) FILTER (WHERE outcome = 'tie')::int AS ties,
    AVG(confidence)::float AS avg_confidence
FROM "LiteLLM_ShadowEvalAttempt"
WHERE job_id = ANY($1::text[]) AND outcome != 'error'
GROUP BY 1
"""

_ATTEMPT_AGG_BY_TIER_SQL: Final = "SELECT COALESCE(tier, 'UNCLASSIFIED') AS grp," + _ATTEMPT_AGG_SELECT
_ATTEMPT_AGG_BY_MODEL_SQL: Final = "SELECT COALESCE(real_model, 'unknown') AS grp," + _ATTEMPT_AGG_SELECT
_ATTEMPT_AGG_BY_LEG_SQL: Final = "SELECT job_id AS grp," + _ATTEMPT_AGG_SELECT

# These guards derive spend from attempt rows, the cross-pod authority; the sampler also
# reads the live counter, so admission can stop before a row-based guard would fire (safe
# direction, and mid-deploy rows from old pods price as judge-only until the deploy ends).
_SWEEP_FINISHED_JOBS_SQL: Final = """
UPDATE "LiteLLM_ShadowEvalJob" j SET stopped_at = (NOW() AT TIME ZONE 'utc')
WHERE j.api_key_id = ANY($1::text[]) AND j.stopped_at IS NULL
  AND (
    j.ends_at <= (NOW() AT TIME ZONE 'utc')
    OR (SELECT COUNT(*) FROM "LiteLLM_ShadowEvalAttempt" a WHERE a.job_id = j.id) >= j.max_turns
    OR (
      j.max_budget IS NOT NULL
      AND (SELECT COALESCE(SUM(a.judge_cost + a.shadow_cost), 0) FROM "LiteLLM_ShadowEvalAttempt" a WHERE a.job_id = j.id) >= j.max_budget
    )
  )
"""

_ATTEMPT_TOTALS_SQL: Final = """
SELECT
    COUNT(*) FILTER (WHERE outcome != 'error')::int AS judged_count,
    COUNT(*) FILTER (WHERE outcome = 'error')::int AS error_count,
    COALESCE(SUM(judge_cost), 0)::float AS judge_spend
FROM "LiteLLM_ShadowEvalAttempt"
WHERE job_id = ANY($1::text[])
"""

_ATTEMPT_COUNTS_SQL: Final = """
SELECT a.job_id, COUNT(*)::int AS attempt_count, COALESCE(SUM(a.judge_cost + a.shadow_cost), 0)::float AS spend
FROM "LiteLLM_ShadowEvalAttempt" a
JOIN "LiteLLM_ShadowEvalJob" j ON j.id = a.job_id
WHERE a.job_id = ANY($1::text[]) AND (j.stopped_at IS NULL OR a.created_at <= j.stopped_at)
GROUP BY a.job_id
"""

_STOP_JOB_SQL: Final = """
UPDATE "LiteLLM_ShadowEvalJob"
SET stopped_by = $2, stopped_at = COALESCE(stopped_at, $3::timestamp)
WHERE group_id = $1 AND stopped_by IS NULL
  AND ends_at > (NOW() AT TIME ZONE 'utc')
  AND EXISTS (
    SELECT 1 FROM "LiteLLM_ShadowEvalJob" k
    WHERE k.group_id = $1 AND k.stopped_at IS NULL
      AND (SELECT COUNT(*) FROM "LiteLLM_ShadowEvalAttempt" a WHERE a.job_id = k.id) < k.max_turns
      AND (
        k.max_budget IS NULL
        OR (SELECT COALESCE(SUM(a.judge_cost + a.shadow_cost), 0) FROM "LiteLLM_ShadowEvalAttempt" a WHERE a.job_id = k.id) < k.max_budget
      )
  )
"""


class _AttemptCountRow(BaseModel):
    job_id: str
    attempt_count: int
    spend: float


_ATTEMPT_COUNT_ROWS: Final = TypeAdapter(list[_AttemptCountRow])


_LIST_LEGS_SQL: Final = """
SELECT * FROM "LiteLLM_ShadowEvalJob"
WHERE group_id IN (
    SELECT group_id FROM "LiteLLM_ShadowEvalJob"
    GROUP BY group_id ORDER BY MAX(created_at) DESC LIMIT $1::int
)
"""

_LIST_LEGS_BY_KEY_SQL: Final = """
SELECT * FROM "LiteLLM_ShadowEvalJob"
WHERE group_id IN (
    SELECT group_id FROM "LiteLLM_ShadowEvalJob" WHERE api_key_id = $2
    GROUP BY group_id ORDER BY MAX(created_at) DESC LIMIT $1::int
)
"""


class _AttemptTotalsRow(BaseModel):
    judged_count: int
    error_count: int
    judge_spend: float


_ATTEMPT_TOTALS_ROWS: Final = TypeAdapter(list[_AttemptTotalsRow])


def _pct_of(numerator: int, denominator: int) -> float:
    return _pct(numerator, denominator)


def _slices(rows: Sequence[_AttemptAggRow]) -> tuple[ShadowEvalSlice, ...]:
    return tuple(
        ShadowEvalSlice(
            group=row.grp,
            turn_count=row.turn_count,
            real_win_rate_pct=_pct_of(row.real_wins, row.turn_count),
            shadow_win_rate_pct=_pct_of(row.shadow_wins, row.turn_count),
            tie_rate_pct=_pct_of(row.ties, row.turn_count),
            avg_judge_confidence=round(row.avg_confidence or 0.0, 3),
        )
        for row in sorted(rows, key=lambda r: r.turn_count, reverse=True)
    )


class _LegRow(BaseModel):
    """One LiteLLM_ShadowEvalJob row, validated off the untyped prisma record. A row is
    one key's leg of a job; the legs of a job share group_id and identical config, written
    together by one create_many. The API's job id is the group id, so leg ids never leave
    the server (attempts reference them internally)."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    group_id: str
    api_key_id: str
    router_name: str
    direction: ShadowEvalDirection
    baseline_model: str | None = None
    judge_model: str
    shadow_percentage: float
    max_turns: int
    max_budget: float | None = None
    created_at: datetime
    ends_at: datetime
    stopped_at: datetime | None = None
    stopped_by: str | None = None

    @field_validator("created_at", "ends_at", "stopped_at")
    @classmethod
    def _as_aware_utc(cls, value: datetime | None) -> datetime | None:
        """The columns store naive UTC wall time (prisma's convention); prisma reads hand
        back aware datetimes while raw SQL reads hand back naive ones, so this boundary
        makes every read aware UTC before anything compares or serializes them."""
        if value is None or value.tzinfo is not None:
            return value
        return value.replace(tzinfo=timezone.utc)


_LEG_ROWS: Final = TypeAdapter(list[_LegRow])


async def _leg_attempt_counts(prisma_client: "PrismaClient", legs: Sequence[_LegRow]) -> Mapping[str, _AttemptCountRow]:
    """Each leg's attempt count and recorded spend by leg id, judged and errored alike, in
    one grouped read. They are the same figures the sampler budgets against max_turns and
    max_budget, so the derived status flips to completed exactly when sampling actually
    ends. A stamped leg's figures freeze at its stopped_at: in-flight attempts that land
    after the stamp are excluded, so they can never reclassify a leg that was stopped
    under budget as budget-spent."""
    if not legs:
        return MappingProxyType({})
    rows: Final = _ATTEMPT_COUNT_ROWS.validate_python(
        await _query_raw(prisma_client, _ATTEMPT_COUNTS_SQL, [leg.id for leg in legs])  # mutable-ok: query param
        or ()
    )
    return MappingProxyType({row.job_id: row for row in rows})


def _group_response(
    group_id: str, legs: Sequence[_LegRow], attempt_counts: Mapping[str, _AttemptCountRow]
) -> ShadowEvalJobResponse:
    """The one constructor of a job response: the caller names the group and passes that
    group's legs. Config is read off the first leg because every leg carries the same copy,
    written by one create_many. No caller may serialize a raw row (that would leak a leg id
    as the job id)."""
    first: Final = legs[0]
    return ShadowEvalJobResponse(
        job_id=group_id,
        keys=tuple(
            ShadowEvalJobKeyResponse(
                api_key_id=leg.api_key_id,
                max_turns=leg.max_turns,
                max_budget=leg.max_budget,
                stopped_at=leg.stopped_at,
                attempt_count=stats.attempt_count if (stats := attempt_counts.get(leg.id)) else 0,
                spend=round(stats.spend, 6) if stats else 0.0,
            )
            for leg in sorted(legs, key=lambda leg: leg.api_key_id)
        ),
        router_name=first.router_name,
        direction=first.direction,
        baseline_model=first.baseline_model,
        judge_model=first.judge_model,
        shadow_percentage=first.shadow_percentage,
        created_at=first.created_at,
        ends_at=first.ends_at,
        stopped_by=next((leg.stopped_by for leg in legs if leg.stopped_by is not None), None),
    )


_NO_KEY_LABELS: Final[tuple[str | None, str | None]] = (None, None)


async def _with_key_labels(
    prisma_client: "PrismaClient", responses: Sequence[ShadowEvalJobResponse]
) -> tuple[ShadowEvalJobResponse, ...]:
    """Resolve every scoped key's hash to its alias and masked name in one batched read,
    so the UI can say whose traffic a job shadows. Deleted keys resolve to None."""
    if not responses:
        return ()
    tokens: Final = sorted(frozenset(key.api_key_id for response in responses for key in response.keys))
    key_rows: Final = await _verification_tokens(prisma_client).find_many(
        where={"token": {"in": tokens}}  # mutable-ok: Prisma filter
    )
    labels: Final[Mapping[str, tuple[str | None, str | None]]] = {
        row.token: (row.key_alias, row.key_name) for row in key_rows or ()
    }
    return tuple(
        response.model_copy(
            update={  # mutable-ok: pydantic update payload
                "keys": tuple(
                    key.model_copy(
                        update={  # mutable-ok: pydantic update payload
                            "key_alias": labels.get(key.api_key_id, _NO_KEY_LABELS)[0],
                            "key_name": labels.get(key.api_key_id, _NO_KEY_LABELS)[1],
                        }
                    )
                    for key in response.keys
                )
            }
        )
        for response in responses
    )


async def _shadow_eval_results(prisma_client: "PrismaClient", legs: Sequence[_LegRow]) -> ShadowEvalResult | None:
    """All three stratifications of one job's verdicts. Tier answers "where does the router
    do well"; the model stratification groups by whichever model served the real arm, so it
    answers "which of the models these keys use today would the router beat" forward, and
    "for the turns the router sent to X, did X beat the baseline" in reverse; key answers
    "which key's traffic does the router suit". Reads are bounded by the job's own attempts
    (<= the sum of its keys' max_turns) via the job_id index."""
    leg_ids: Final = [leg.id for leg in legs]  # mutable-ok: query param
    by_tier: Final = _ATTEMPT_AGG_ROWS.validate_python(
        await _query_raw(prisma_client, _ATTEMPT_AGG_BY_TIER_SQL, leg_ids) or ()
    )
    if not by_tier:
        return None
    by_model: Final = _ATTEMPT_AGG_ROWS.validate_python(
        await _query_raw(prisma_client, _ATTEMPT_AGG_BY_MODEL_SQL, leg_ids) or ()
    )
    key_by_leg: Final = MappingProxyType({leg.id: leg.api_key_id for leg in legs})
    by_leg: Final = _ATTEMPT_AGG_ROWS.validate_python(
        await _query_raw(prisma_client, _ATTEMPT_AGG_BY_LEG_SQL, leg_ids) or ()
    )
    by_key: Final = tuple(
        row.model_copy(update={"grp": key_by_leg[row.grp]})  # mutable-ok: pydantic update payload
        for row in by_leg
    )
    total_turns: Final = sum(r.turn_count for r in by_tier)
    return ShadowEvalResult(
        by_tier=_slices(by_tier),
        by_current_model=_slices(by_model),
        by_key=_slices(by_key),
        overall_shadow_win_rate_pct=_pct_of(sum(r.shadow_wins for r in by_tier), total_turns),
        overall_tie_rate_pct=_pct_of(sum(r.ties for r in by_tier), total_turns),
    )


@router.post(
    "/auto_router/shadow_eval/start",
    tags=("auto router",),
    dependencies=(Depends(user_api_key_auth),),
    response_model=ShadowEvalJobResponse,
    status_code=status.HTTP_201_CREATED,
)
async def start_shadow_eval(
    data: StartShadowEvalRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> ShadowEvalJobResponse:
    """
    Start a shadow eval: duplicate a sampled slice of one or more keys' live traffic against
    a second arm, judge the two responses blind, and stratify win rates by tier, by the model
    that served the real arm, and by key.

    A forward job answers whether the keys should adopt router_name: it samples the requests
    the router did not serve and duplicates them through it. A reverse job answers whether a
    key already on the router still gains from it: it samples the requests the router did
    serve and duplicates them against baseline_model. A key can hold one active job per
    direction, so both questions can run at once.

    Shadow responses are never served to users. Each key samples until its recorded eval
    spend, the shadow and judge calls' own cost, reaches max_budget dollars, the job's
    window ends, or the job is stopped, so one key running out of budget does not end
    sampling for the others; sampling changes propagate to pods within about 10 seconds.
    Shadow and judge calls bill to the shadowed key but are excluded from request counts
    and auto-router adoption metrics.
    """
    from litellm.proxy.proxy_server import llm_router, prisma_client

    _require_admin_writer(user_api_key_dict, "start a shadow eval")
    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    if llm_router is None or not _is_configured_pre_routing_strategy(llm_router, data.router_name):
        raise HTTPException(status_code=400, detail=f"'{data.router_name}' is not a configured auto-router")
    _validate_plain_model(llm_router, data.judge_model, "judge_model")
    if data.baseline_model is not None:
        _validate_plain_model(llm_router, data.baseline_model, "baseline_model")
    token_rows: Final = await _verification_tokens(prisma_client).find_many(
        where={"token": {"in": list(data.api_key_ids)}}  # mutable-ok: Prisma filter
    )
    unknown: Final = tuple(sorted(frozenset(data.api_key_ids) - frozenset(row.token for row in token_rows or ())))
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=(
                f"api_key_ids not on this proxy: {', '.join(unknown)}; pass each key's token hash, "
                "the value the key list and key info endpoints report"
            ),
        )

    # A job whose window passed or whose budget ran out stopped sampling on its own,
    # but its legs still hold their slots in the per-key, per-direction partial unique index
    # until stamped; free them so a new eval can start. Sweeping both directions is deliberate.
    requested: Final = list(data.api_key_ids)  # mutable-ok: query param
    await prisma_client.db.execute_raw(_SWEEP_FINISHED_JOBS_SQL, requested)
    claimed: Final = await _shadow_eval_jobs(prisma_client).find_many(
        where={  # mutable-ok: Prisma filter
            "api_key_id": {"in": requested},  # mutable-ok: Prisma filter
            "direction": data.direction,
            "stopped_at": None,
        },
    )
    if claimed:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Already in an active {data.direction} shadow eval job: "
                + ", ".join(sorted(f"{row.api_key_id} (job {row.group_id})" for row in claimed))
                + ". Stop it first."
            ),
        )
    now: Final = datetime.now(timezone.utc)
    group_id: Final = str(uuid4())
    ends_at: Final = now + timedelta(days=data.duration_days)
    shared_config: Final = {  # mutable-ok: Prisma payload
        "group_id": group_id,
        "router_name": data.router_name,
        "direction": data.direction,
        "baseline_model": data.baseline_model,
        "judge_model": data.judge_model,
        "shadow_percentage": data.shadow_percentage,
        "max_turns": SHADOW_EVAL_TURN_VALVE,
        "max_budget": data.max_budget,
        "created_by": user_api_key_dict.user_id,
        "created_at": now,
        "ends_at": ends_at,
    }
    try:
        await _shadow_eval_jobs(prisma_client).create_many(
            data=[{**shared_config, "api_key_id": key} for key in data.api_key_ids]  # mutable-ok: Prisma payload
        )
    except Exception as e:
        if not _is_unique_violation(e):
            raise
        raise HTTPException(
            status_code=409,
            detail=(
                f"A requested key was claimed by another {data.direction} shadow eval job concurrently. Stop it first."
            ),
        ) from e
    labels: Final = MappingProxyType({row.token: row for row in token_rows})
    return ShadowEvalJobResponse(
        job_id=group_id,
        keys=tuple(
            ShadowEvalJobKeyResponse(
                api_key_id=api_key_id,
                max_turns=SHADOW_EVAL_TURN_VALVE,
                max_budget=data.max_budget,
                key_alias=labels[api_key_id].key_alias,
                key_name=labels[api_key_id].key_name,
            )
            for api_key_id in sorted(data.api_key_ids)
        ),
        router_name=data.router_name,
        direction=data.direction,
        baseline_model=data.baseline_model,
        judge_model=data.judge_model,
        shadow_percentage=data.shadow_percentage,
        created_at=now,
        ends_at=ends_at,
    )


@router.get(
    "/auto_router/shadow_eval",
    tags=("auto router",),
    dependencies=(Depends(user_api_key_auth),),
    response_model=list[ShadowEvalJobResponse],
)
async def list_shadow_eval_jobs(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    api_key_id: Annotated[
        str | None, Query(description="Filter to jobs that shadow this key, alone or alongside others")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=200, description="Newest jobs to return")] = 50,
) -> tuple[ShadowEvalJobResponse, ...]:
    """List shadow eval jobs, newest first, each key with its attempt count so status is
    accurate. Judged counts, spend, and results ride the detail endpoint only."""
    from litellm.proxy.proxy_server import prisma_client

    _require_admin_viewer(user_api_key_dict, "view shadow evals")
    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    legs: Final = _LEG_ROWS.validate_python(
        (
            await _query_raw(prisma_client, _LIST_LEGS_BY_KEY_SQL, limit, api_key_id)
            if api_key_id
            else await _query_raw(prisma_client, _LIST_LEGS_SQL, limit)
        )
        or ()
    )
    by_group: Final[Mapping[str, tuple[_LegRow, ...]]] = MappingProxyType(
        {
            group_id: tuple(group)
            for group_id, group in groupby(sorted(legs, key=attrgetter("group_id")), key=attrgetter("group_id"))
        }
    )
    newest_first: Final = sorted(
        by_group, key=lambda group_id: max(leg.created_at for leg in by_group[group_id]), reverse=True
    )
    counts: Final = await _leg_attempt_counts(prisma_client, legs)
    return await _with_key_labels(
        prisma_client, tuple(_group_response(group_id, by_group[group_id], counts) for group_id in newest_first)
    )


@router.get(
    "/auto_router/shadow_eval/{job_id}",
    tags=("auto router",),
    dependencies=(Depends(user_api_key_auth),),
    response_model=ShadowEvalJobResponse,
)
async def get_shadow_eval_job(
    job_id: str,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> ShadowEvalJobResponse:
    """One job with derived counts, judge spend, latest error, and stratified results."""
    from litellm.proxy.proxy_server import prisma_client

    _require_admin_viewer(user_api_key_dict, "view shadow evals")
    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    legs: Final = _LEG_ROWS.validate_python(
        await _shadow_eval_jobs(prisma_client).find_many(
            where={"group_id": job_id}  # mutable-ok: Prisma filter
        )
        or ()
    )
    if not legs:
        raise HTTPException(status_code=404, detail=f"No shadow eval job {job_id}")
    leg_ids: Final = [leg.id for leg in legs]  # mutable-ok: query param
    totals: Final = _ATTEMPT_TOTALS_ROWS.validate_python(
        await _query_raw(prisma_client, _ATTEMPT_TOTALS_SQL, leg_ids) or ()
    )
    latest_error: Final = await _shadow_eval_attempts(prisma_client).find_first(
        where={"job_id": {"in": leg_ids}, "outcome": "error"},  # mutable-ok: Prisma filter
        order={"created_at": "desc"},  # mutable-ok: Prisma order
    )
    labeled: Final = await _with_key_labels(
        prisma_client, (_group_response(job_id, legs, await _leg_attempt_counts(prisma_client, legs)),)
    )
    return labeled[0].model_copy(
        update={  # mutable-ok: pydantic update payload
            "judged_count": totals[0].judged_count if totals else 0,
            "error_count": totals[0].error_count if totals else 0,
            "judge_spend": round(totals[0].judge_spend, 6) if totals else 0.0,
            "last_error": latest_error.error if latest_error else None,
            "results": await _shadow_eval_results(prisma_client, legs),
        }
    )


@router.post(
    "/auto_router/shadow_eval/{job_id}/stop",
    tags=("auto router",),
    dependencies=(Depends(user_api_key_auth),),
    response_model=ShadowEvalJobResponse,
)
async def stop_shadow_eval_job(
    job_id: str,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> ShadowEvalJobResponse:
    """Stop an active shadow eval job, every key it scopes at once. Attempts are kept;
    sampling halts within ~10s. Keys that already stopped on their own budget keep the
    stopped_at they earned. The statement is the whole state machine: it claims the job
    only while a leg still samples inside the window with no stop recorded, so a racing
    operator, a same-instant budget spend, and a repeat stop all read the same 400 with
    the status the job actually holds."""
    from litellm.proxy.proxy_server import prisma_client

    _require_admin_writer(user_api_key_dict, "stop a shadow eval")
    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    stamp: Final = datetime.now(timezone.utc)
    operator: Final = user_api_key_dict.user_id or "operator"
    claimed: Final = await prisma_client.db.execute_raw(
        _STOP_JOB_SQL, job_id, operator, stamp.replace(tzinfo=None).isoformat()
    )
    legs: Final = _LEG_ROWS.validate_python(
        await _shadow_eval_jobs(prisma_client).find_many(
            where={"group_id": job_id}  # mutable-ok: Prisma filter
        )
        or ()
    )
    if not legs:
        raise HTTPException(status_code=404, detail=f"No shadow eval job {job_id}")
    counts: Final = await _leg_attempt_counts(prisma_client, legs)
    current: Final = _group_response(job_id, legs, counts)
    if claimed == 0:
        raise HTTPException(status_code=400, detail=f"Job {job_id} is already {current.status}")
    labeled: Final = await _with_key_labels(prisma_client, (current,))
    return labeled[0]
