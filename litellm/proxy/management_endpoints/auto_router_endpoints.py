"""
AUTO ROUTER MANAGEMENT ENDPOINTS

POST /auto_router/test_routing - Route one request through an unsaved complexity-router config
POST /auto_router/validate_complexity_router_config - Dry-run the complexity-router write gate without saving
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from itertools import chain, groupby
from operator import attrgetter
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Final, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, TypeAdapter, field_validator

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.exceptions import BudgetExceededError
from litellm.litellm_core_utils.llm_judge import judge_target
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
from litellm.proxy.litellm_pre_call_utils import (
    LiteLLMProxyRequestSetup,
    refresh_proxy_server_request_body_snapshot,
)
from litellm.repositories.base_repository import SupportsModelDump
from litellm.repositories.team_repository import TeamRepository
from litellm.router_strategy.capability_router import CapabilityRouter
from litellm.router_strategy.complexity_router import ComplexityRouter
from litellm.router_utils.auto_router_model_naming import (
    StrategyRouterDependencyRole,
    classify_strategy_router_model,
    strategy_router_dependencies,
)
from litellm.types.management_endpoints.auto_router_endpoints import (
    SHADOW_EVAL_TURN_VALVE,
    AutoRouterBenchmarkGroup,
    AutoRouterBenchmarksResponse,
    AutoRouterBenchmarkTotals,
    AutoRouterCacheBucket,
    AutoRouterCacheStats,
    AutoRouterRoutingTestRequest,
    AutoRouterRoutingTestResponse,
    CapabilityRouterConfigValidationRequest,
    CapabilityRouterConfigValidationResponse,
    ComplexityRouterConfigValidationRequest,
    ComplexityRouterConfigValidationResponse,
    RequestComplexityRouterConfig,
    ShadowEvalDirection,
    ShadowEvalJobResponse,
    ShadowEvalJobTargetResponse,
    ShadowEvalResult,
    ShadowEvalSlice,
    ShadowEvalTargetType,
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

    @property
    def team_id(self) -> str | None: ...


class _VerificationTokenTable(Protocol):
    async def find_unique(self, *, where: Mapping[str, object]) -> _VerificationTokenRow | None: ...

    async def find_many(self, *, where: Mapping[str, object]) -> Sequence[_VerificationTokenRow]: ...


class _TeamRow(Protocol):
    @property
    def team_id(self) -> str: ...

    @property
    def team_alias(self) -> str | None: ...


class _TeamRowsTable(Protocol):
    async def find_many(self, *, where: Mapping[str, object]) -> Sequence[_TeamRow]: ...


class _UserRow(Protocol):
    @property
    def user_id(self) -> str: ...

    @property
    def user_email(self) -> str | None: ...


class _UserRowsTable(Protocol):
    async def find_many(self, *, where: Mapping[str, object]) -> Sequence[_UserRow]: ...


class _ShadowEvalJobRow(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def group_id(self) -> str: ...

    @property
    def target_type(self) -> str: ...

    @property
    def target_id(self) -> str: ...


class _ShadowEvalJobTable(Protocol):
    async def find_many(self, *, where: Mapping[str, object]) -> Sequence[_ShadowEvalJobRow]: ...

    async def create_many(self, data: Sequence[Mapping[str, object]]) -> int: ...


class _ShadowEvalAttemptRow(Protocol):
    @property
    def error(self) -> str | None: ...


class _ShadowEvalFunnelTable(Protocol):
    async def create_many(self, data: Sequence[Mapping[str, object]], skip_duplicates: bool) -> int: ...


class _ShadowEvalAttemptTable(Protocol):
    async def find_first(
        self, *, where: Mapping[str, object], order: Mapping[str, str]
    ) -> _ShadowEvalAttemptRow | None: ...


def _team_table(prisma_client: "PrismaClient") -> _TeamTable:
    return TeamRepository(prisma_client).table


def _verification_tokens(prisma_client: "PrismaClient") -> _VerificationTokenTable:
    return prisma_client.db.litellm_verificationtoken


def _team_rows(prisma_client: "PrismaClient") -> _TeamRowsTable:
    return prisma_client.db.litellm_teamtable


def _user_rows(prisma_client: "PrismaClient") -> _UserRowsTable:
    return prisma_client.db.litellm_usertable


def _shadow_eval_jobs(prisma_client: "PrismaClient") -> _ShadowEvalJobTable:
    return prisma_client.db.litellm_shadowevaljob


def _shadow_eval_funnel(prisma_client: "PrismaClient") -> _ShadowEvalFunnelTable:
    return prisma_client.db.litellm_shadowevalfunnel  # pyright: ignore[reportAttributeAccessIssue]  # generated client


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
    await _authorize_model_names_this_test_can_call(
        models=_models_this_test_can_call(config),
        user_api_key_dict=user_api_key_dict,
        llm_router=llm_router,
    )


async def _authorize_model_names_this_test_can_call(
    models: Sequence[str],
    user_api_key_dict: UserAPIKeyAuth,
    llm_router: "Router",
) -> None:
    """Apply model-access and key-budget checks to internal dry-run calls."""
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
    "/auto_router/validate_capability_router_config",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=CapabilityRouterConfigValidationResponse,
    status_code=status.HTTP_200_OK,
)
async def validate_capability_router_config(
    data: CapabilityRouterConfigValidationRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> CapabilityRouterConfigValidationResponse:
    """Validate a capability-router config without saving it."""
    await _authorize_router_dry_run(user_api_key_dict=user_api_key_dict, team_id=data.team_id)
    from litellm.router_utils.auto_router_model_naming import (
        validate_capability_router_config_write,
    )

    error: Final = validate_capability_router_config_write(data.capability_router_config)
    return CapabilityRouterConfigValidationResponse(valid=error is None, error=error)


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
    Route a single request through a complexity-router config and report where it landed.

    Answers "which model would this request get?" for a config that only exists in a form,
    so an auto router can be checked before it is created. The request is classified by the
    same pre-routing hook a live request runs, over the same messages, system prompt and tool
    definitions, then dropped: nothing is sent to the model it routed to, and no auto router is
    created. A heuristic config therefore spends nothing, while an `llm` classifier or semantic
    keyword matching bills its classifier/embedding call to the calling key, like Test Connection
    does.

    Send `messages` to classify a real turn, with `system` and `tools` beside it when the surface
    carries them top level, as Anthropic /v1/messages does. `prompt` is the single-ask shorthand and
    routes as one user turn with nothing around it.

    **Example Request:**
    ```json
    {
        "messages": [
            {"role": "system", "content": "You are a database migration assistant"},
            {"role": "user", "content": "the index is not unique"},
            {"role": "assistant", "content": "Then two workers can both insert. Add a unique index"},
            {"role": "user", "content": "ok do it"}
        ],
        "tools": [{"type": "function", "function": {"name": "Bash", "description": "Run a command"}}],
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

    if data.capability_router_config is not None:
        await _authorize_model_names_this_test_can_call(
            models=(data.capability_router_config.classifier.model,),
            user_api_key_dict=user_api_key_dict,
            llm_router=llm_router,
        )
        strategy = CapabilityRouter(
            model_name=data.router_name,
            litellm_router_instance=llm_router,
            capability_router_config=data.capability_router_config.model_dump(exclude_none=True),
        )
    else:
        assert data.complexity_router_config is not None
        await _authorize_models_this_test_can_call(
            config=data.complexity_router_config,
            user_api_key_dict=user_api_key_dict,
            llm_router=llm_router,
        )
        strategy = ComplexityRouter(
            model_name=data.router_name,
            litellm_router_instance=llm_router,
            complexity_router_config=data.complexity_router_config.model_dump(exclude_none=True),
            default_model=data.default_model,
            derive_savings_baseline=False,
        )

    request_kwargs: Final = LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
        data={  # mutable-ok: the request-metadata helper takes and returns request kwargs as a dict
            **data.wire_body(),
            "metadata": {},  # mutable-ok: the request-metadata helper writes the auth fields into this dict
            "proxy_server_request": {"body": None},  # mutable-ok: the snapshot owner fills body in place
        },
        user_api_key_dict=user_api_key_dict,
        _metadata_variable_name="metadata",
    )
    refresh_proxy_server_request_body_snapshot(request_kwargs)

    try:
        hook_response: Final = await strategy.async_pre_routing_hook(
            model=data.router_name,
            request_kwargs=request_kwargs,
            messages=request_kwargs["messages"],
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


def _sdk_model_is_missing_anthropic_credentials(model: str) -> bool:
    _, provider, _, _ = litellm.get_llm_provider(model=model)
    if provider != "anthropic" or litellm.anthropic_key or litellm.api_key:
        return False
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo
    from litellm.secret_managers.main import secret_manager_would_be_consulted

    if AnthropicModelInfo.get_api_key() or AnthropicModelInfo.get_auth_token():
        return False
    return not any(
        secret_manager_would_be_consulted(secret_name) for secret_name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
    )


def _validate_plain_model(
    llm_router: "Router | None", model: str, field_name: str, team_ids: Sequence[str | None]
) -> None:
    """Reject a model the dispatch path cannot resolve, at start rather than as a silently
    growing error count once the job is already sampling and billing. Both the judge and a
    reverse job's baseline must be plain models: an auto-router in either slot would
    re-route per turn, so the comparison would have no fixed arm to attribute results to.

    Resolvability is asked once per team the job samples for, because that is the identity
    the call carries: a name only one team can reach fails every turn for the other keys,
    which is the growing error count this check exists to prevent."""
    if llm_router is not None and _is_configured_pre_routing_strategy(llm_router, model):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} '{model}' is an auto-router; it must be a plain model",
        )
    targets: Final = tuple((team, judge_target(llm_router, model, team)) for team in team_ids)
    unreachable: Final = tuple(team for team, target in targets if target.via == "nothing")
    if unreachable:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{field_name} '{model}' is neither a model configured on this proxy nor a "
                "provider-qualified public model name (e.g. 'anthropic/claude-sonnet-5')" + _for_teams(unreachable)
            ),
        )
    sdk_teams: Final = tuple(team for team, target in targets if target.via == "sdk")
    if not sdk_teams:
        return
    if not _sdk_model_is_missing_anthropic_credentials(model):
        return
    raise HTTPException(
        status_code=400,
        detail=(
            f"{field_name} '{model}' uses the LiteLLM SDK but required credentials are not configured: "
            "ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN" + _for_teams(sdk_teams)
        ),
    )


def _for_teams(team_ids: Sequence[str | None]) -> str:
    """Name the teams a fault applies to, when it does not apply to every key alike."""
    named: Final = tuple(sorted(team for team in team_ids if team is not None))
    return f" for team {', '.join(named)}" if named else ""


_JUDGED_ROLES: Final[frozenset[StrategyRouterDependencyRole]] = frozenset({"tier", "default"})


def _router_arm_models(llm_router: "Router | None", router_name: str) -> tuple[tuple[str, str], ...]:
    """``(role, model_name)`` for every model the router under evaluation can answer with.

    Drawn from ``strategy_router_dependencies``, the single answer to "what does this router
    call", so this cannot disagree with the health check's reading of the same deployment.
    Only the roles that SERVE are arms: the classifier and embedding models pick the tier,
    they never produce a response anyone judges, so a judge sharing them carries no
    self-preference.

    A semantic auto-router keeps its routes in an opaque config blob or a file, so only its
    default model is enumerable and the guard below is incomplete for it. That direction is
    deliberate: it can miss a collision, never invent one.

    Which tiers a router declares is a property of its config and not of who is calling, so
    this lookup is unscoped; what each tier NAME resolves to is the team-dependent half, and
    it belongs to the caller that compares them.
    """
    deployments: Final = llm_router.get_model_list(model_name=router_name) if llm_router is not None else None
    return tuple(
        dict.fromkeys(
            (dependency.role, dependency.model_name)
            for deployment in deployments or ()
            for dependency in strategy_router_dependencies(deployment["litellm_params"])
            if dependency.role in _JUDGED_ROLES
        )
    )


def _judge_collisions_for_team(
    llm_router: "Router | None", data: StartShadowEvalRequest, team_id: str | None
) -> tuple[tuple[str, str], ...]:
    """``(role, model_name)`` for each arm the judge would also be, as one team's keys see it.

    Both sides resolve under the SAME team, since two names are the same model only for a
    caller who can reach both; resolving the judge for one team against an arm for another
    invents a collision no request could produce.
    """
    judge: Final = judge_target(llm_router, data.judge_model, team_id).models
    return tuple(
        (role, model)
        for role, model in (
            *(arm for name in data.router_names for arm in _router_arm_models(llm_router, name)),
            *((("baseline", data.baseline_model),) if data.baseline_model is not None else ()),
        )
        if judge & judge_target(llm_router, model, team_id).models
    )


def _validate_judge_is_not_a_candidate(
    llm_router: "Router | None", data: StartShadowEvalRequest, team_ids: Sequence[str | None]
) -> None:
    """Reject a judge that is one of the two arms it grades.

    A judge scores its own output higher than a rival's, so a run whose judge also serves an
    arm reports a win rate for that arm that measures the judge rather than the models, and
    the whole job's spend buys a result that has to be discarded. Both arms are in scope: the
    router answers with a tier or default model in either direction, and a reverse job's
    ``baseline_model`` is the fixed arm the router is compared against.

    Names are compared by what would ANSWER them, not by spelling: the shipped default judge
    ``anthropic/claude-sonnet-5`` collides with a tier deployment an admin named
    ``sonnet-tier``, and an alias collides with its target, neither of which a string
    comparison sees.

    A collision for ONE team is a collision for the job, because the verdicts every key
    produces land in the same win rates.
    """
    collisions: Final = tuple(
        dict.fromkeys(
            collision for team_id in team_ids for collision in _judge_collisions_for_team(llm_router, data, team_id)
        )
    )
    if not collisions:
        return
    raise HTTPException(
        status_code=400,
        detail=(
            f"judge_model '{data.judge_model}' is also an arm this job would judge: "
            + ", ".join(f"{role} model '{model}'" for role, model in collisions)
            + ". A judge scores its own answers higher than a rival's, so the win rates would "
            "measure the judge; pick a judge that serves neither arm"
        ),
    )


def _is_unique_violation(error: Exception) -> bool:
    """Whether a Prisma create failed on a unique index. One active job per target and
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
    real_spend: float
    shadow_spend: float
    cache_hit_turns: int


_ATTEMPT_AGG_ROWS: Final = TypeAdapter(list[_AttemptAggRow])

_ATTEMPT_AGG_COLUMNS: Final = """
    COUNT(*)::int AS turn_count,
    COUNT(*) FILTER (WHERE outcome = 'real')::int AS real_wins,
    COUNT(*) FILTER (WHERE outcome = 'shadow')::int AS shadow_wins,
    COUNT(*) FILTER (WHERE outcome = 'tie')::int AS ties,
    AVG(confidence)::float AS avg_confidence,
    COALESCE(SUM(real_cost + real_classifier_cost) FILTER (WHERE real_cost IS NOT NULL AND NOT real_cache_hit), 0)::float AS real_spend,
    COALESCE(SUM(shadow_cost + shadow_classifier_cost) FILTER (WHERE real_cost IS NOT NULL AND NOT real_cache_hit), 0)::float AS shadow_spend,
    COUNT(*) FILTER (WHERE real_cache_hit)::int AS cache_hit_turns
"""

_ATTEMPT_AGG_SELECT: Final = (
    _ATTEMPT_AGG_COLUMNS
    + """
FROM "LiteLLM_ShadowEvalAttempt"
WHERE job_id = ANY($1::text[]) AND outcome != 'error'
GROUP BY 1
"""
)

_ATTEMPT_AGG_BY_TIER_SQL: Final = "SELECT COALESCE(tier, 'UNCLASSIFIED') AS grp," + _ATTEMPT_AGG_SELECT
_ATTEMPT_AGG_BY_MODEL_SQL: Final = "SELECT COALESCE(real_model, 'unknown') AS grp," + _ATTEMPT_AGG_SELECT
_ATTEMPT_AGG_BY_LEG_SQL: Final = "SELECT job_id AS grp," + _ATTEMPT_AGG_SELECT

# Attempt rows from before arm stamping carry no router_name; they belong to the job's
# own router, which the join reads off the leg.
_ATTEMPT_AGG_BY_ROUTER_SQL: Final = (
    "SELECT COALESCE(a.router_name, j.router_name) AS grp,"
    + _ATTEMPT_AGG_COLUMNS
    + """
FROM "LiteLLM_ShadowEvalAttempt" a
JOIN "LiteLLM_ShadowEvalJob" j ON j.id = a.job_id
WHERE a.job_id = ANY($1::text[]) AND a.outcome != 'error'
GROUP BY 1
"""
)

# These guards derive spend from attempt rows, the cross-pod authority; the sampler also
# reads the live counter, so admission can stop before a row-based guard would fire (safe
# direction, and mid-deploy rows from old pods price as judge-only until the deploy ends).
_SWEEP_FINISHED_JOBS_SQL: Final = """
UPDATE "LiteLLM_ShadowEvalJob" j SET stopped_at = (NOW() AT TIME ZONE 'utc')
WHERE j.target_type = $2 AND j.target_id = ANY($1::text[]) AND j.stopped_at IS NULL
  AND (
    j.ends_at <= (NOW() AT TIME ZONE 'utc')
    OR (SELECT COUNT(*) FROM "LiteLLM_ShadowEvalAttempt" a WHERE a.job_id = j.id) >= j.max_turns
    OR (
      j.max_budget IS NOT NULL
      AND (SELECT COALESCE(SUM(a.judge_cost + a.shadow_cost + a.shadow_classifier_cost), 0) FROM "LiteLLM_ShadowEvalAttempt" a WHERE a.job_id = j.id) >= j.max_budget
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
SELECT a.job_id, COUNT(*)::int AS attempt_count, COALESCE(SUM(a.judge_cost + a.shadow_cost + a.shadow_classifier_cost), 0)::float AS spend
FROM "LiteLLM_ShadowEvalAttempt" a
JOIN "LiteLLM_ShadowEvalJob" j ON j.id = a.job_id
WHERE a.job_id = ANY($1::text[]) AND (j.stopped_at IS NULL OR a.created_at <= j.stopped_at)
GROUP BY a.job_id
"""

_FUNNEL_TOTALS_SQL: Final = """
SELECT COUNT(*)::int AS legs_with_rows,
    COALESCE(SUM(not_sampled), 0)::int AS not_sampled,
    COALESCE(SUM(unjudgeable), 0)::int AS unjudgeable,
    COALESCE(SUM(shed), 0)::int AS shed,
    COALESCE(SUM(withheld), 0)::int AS withheld
FROM "LiteLLM_ShadowEvalFunnel"
WHERE job_id = ANY($1::text[])
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
        OR (SELECT COALESCE(SUM(a.judge_cost + a.shadow_cost + a.shadow_classifier_cost), 0) FROM "LiteLLM_ShadowEvalAttempt" a WHERE a.job_id = k.id) < k.max_budget
      )
  )
"""


class _FunnelTotalsRow(BaseModel):
    legs_with_rows: int
    not_sampled: int
    unjudgeable: int
    shed: int
    withheld: int


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

_LIST_LEGS_BY_TARGET_SQL: Final = """
SELECT * FROM "LiteLLM_ShadowEvalJob"
WHERE group_id IN (
    SELECT group_id FROM "LiteLLM_ShadowEvalJob" WHERE target_type = $2 AND target_id = $3
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
            real_spend=row.real_spend,
            shadow_spend=row.shadow_spend,
            cache_hit_turns=row.cache_hit_turns,
        )
        for row in sorted(rows, key=lambda r: r.turn_count, reverse=True)
    )


class _LegRow(BaseModel):
    """One LiteLLM_ShadowEvalJob row, validated off the untyped prisma record. A row is
    one target's leg of a job; the legs of a job share group_id and identical config,
    written together by one create_many. The API's job id is the group id, so leg ids
    never leave the server (attempts reference them internally)."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    group_id: str
    target_type: ShadowEvalTargetType
    target_id: str
    router_name: str
    router_names: tuple[str, ...] = ()
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

    @property
    def arm_router_names(self) -> tuple[str, ...]:
        """The job's full router set; rows from before router_names existed hold it in
        router_name alone. The one place that reading lives on the endpoint side."""
        return self.router_names or (self.router_name,)

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
        targets=tuple(
            ShadowEvalJobTargetResponse(
                target_type=leg.target_type,
                target_id=leg.target_id,
                max_turns=leg.max_turns,
                max_budget=leg.max_budget,
                stopped_at=leg.stopped_at,
                attempt_count=stats.attempt_count if (stats := attempt_counts.get(leg.id)) else 0,
                spend=round(stats.spend, 6) if stats else 0.0,
            )
            for leg in sorted(legs, key=lambda leg: (leg.target_type, leg.target_id))
        ),
        router_names=first.arm_router_names,
        direction=first.direction,
        baseline_model=first.baseline_model,
        judge_model=first.judge_model,
        shadow_percentage=first.shadow_percentage,
        created_at=first.created_at,
        ends_at=first.ends_at,
        stopped_by=next((leg.stopped_by for leg in legs if leg.stopped_by is not None), None),
    )


_NO_TARGET_LABELS: Final[tuple[str | None, str | None]] = (None, None)


def _target_labels(
    key_rows: Sequence[_VerificationTokenRow],
    team_rows: Sequence[_TeamRow],
    user_rows: Sequence[_UserRow],
) -> Mapping[tuple[str, str], tuple[str | None, str | None]]:
    """Display labels by (target_type, target_id): a key's (alias, masked name), a
    team's (alias, None), a user's (email, None)."""
    return MappingProxyType(
        {  # mutable-ok: MappingProxyType needs a dict to wrap
            key: value
            for key, value in chain(
                ((("key", row.token), (row.key_alias, row.key_name)) for row in key_rows),
                ((("team", row.team_id), (row.team_alias, None)) for row in team_rows),
                ((("user", row.user_id), (row.user_email, None)) for row in user_rows),
            )
        }
    )


def _target_ids_of(responses: Sequence[ShadowEvalJobResponse], target_type: ShadowEvalTargetType) -> tuple[str, ...]:
    return tuple(
        sorted(
            frozenset(
                target.target_id
                for response in responses
                for target in response.targets
                if target.target_type == target_type
            )
        )
    )


async def _with_target_labels(
    prisma_client: "PrismaClient", responses: Sequence[ShadowEvalJobResponse]
) -> tuple[ShadowEvalJobResponse, ...]:
    """Resolve every scoped target's id to a display label in one batched read per kind,
    so the UI can say whose traffic a job shadows: a key's alias and masked name, a
    team's alias, a user's email. Deleted targets resolve to None."""
    if not responses:
        return ()
    tokens: Final = _target_ids_of(responses, "key")
    team_ids: Final = _target_ids_of(responses, "team")
    user_ids: Final = _target_ids_of(responses, "user")
    key_rows: Final = (
        await _verification_tokens(prisma_client).find_many(
            where={"token": {"in": list(tokens)}}  # mutable-ok: Prisma filter
        )
        if tokens
        else ()
    )
    team_rows: Final = (
        await _team_rows(prisma_client).find_many(
            where={"team_id": {"in": list(team_ids)}}  # mutable-ok: Prisma filter
        )
        if team_ids
        else ()
    )
    user_rows: Final = (
        await _user_rows(prisma_client).find_many(
            where={"user_id": {"in": list(user_ids)}}  # mutable-ok: Prisma filter
        )
        if user_ids
        else ()
    )
    labels: Final = _target_labels(key_rows or (), team_rows or (), user_rows or ())
    return tuple(
        response.model_copy(
            update={  # mutable-ok: pydantic update payload
                "targets": tuple(
                    target.model_copy(
                        update={  # mutable-ok: pydantic update payload
                            "target_alias": labels.get((target.target_type, target.target_id), _NO_TARGET_LABELS)[0],
                            "key_name": labels.get((target.target_type, target.target_id), _NO_TARGET_LABELS)[1],
                        }
                    )
                    for target in response.targets
                )
            }
        )
        for response in responses
    )


async def _shadow_eval_results(
    prisma_client: "PrismaClient", legs: Sequence[_LegRow]
) -> tuple[ShadowEvalResult | None, Mapping[tuple[str, str], ShadowEvalSlice]]:
    """One job's stratified verdicts, plus each target's own slice keyed by the
    (target_type, target_id) pair so a key, team, and user sharing an id can never
    collapse into one entry. Tier answers "where does the router do well"; the model
    stratification groups by whichever model served the real arm, so it answers "which
    of the models these targets use today would the router beat" forward, and "for the
    turns the router sent to X, did X beat the baseline" in reverse; the per-target
    slices answer "which target's traffic does the router suit". Reads are bounded by
    the job's own attempts (<= the sum of its targets' max_turns) via the job_id index."""
    leg_ids: Final = [leg.id for leg in legs]  # mutable-ok: query param
    by_tier: Final = _ATTEMPT_AGG_ROWS.validate_python(
        await _query_raw(prisma_client, _ATTEMPT_AGG_BY_TIER_SQL, leg_ids) or ()
    )
    if not by_tier:
        return None, MappingProxyType({})
    by_model: Final = _ATTEMPT_AGG_ROWS.validate_python(
        await _query_raw(prisma_client, _ATTEMPT_AGG_BY_MODEL_SQL, leg_ids) or ()
    )
    target_by_leg: Final = MappingProxyType({leg.id: (leg.target_type, leg.target_id) for leg in legs})
    by_leg: Final = _ATTEMPT_AGG_ROWS.validate_python(
        await _query_raw(prisma_client, _ATTEMPT_AGG_BY_LEG_SQL, leg_ids) or ()
    )
    verdicts_by_target: Final[Mapping[tuple[str, str], ShadowEvalSlice]] = MappingProxyType(
        {  # mutable-ok: MappingProxyType needs a dict to wrap
            target_by_leg[slice.group]: slice.model_copy(
                update={"group": target_by_leg[slice.group][1]}  # mutable-ok: pydantic update payload
            )
            for slice in _slices(by_leg)
        }
    )
    by_router: Final = _ATTEMPT_AGG_ROWS.validate_python(
        await _query_raw(prisma_client, _ATTEMPT_AGG_BY_ROUTER_SQL, leg_ids) or ()
    )
    total_turns: Final = sum(r.turn_count for r in by_tier)
    funnel_rows: Final = await _query_raw(prisma_client, _FUNNEL_TOTALS_SQL, leg_ids)
    counted: Final = _FunnelTotalsRow.model_validate(funnel_rows[0]) if funnel_rows else None
    # Coverage only when EVERY leg has a funnel row: a partial seed (one leg's insert
    # failed) must read as unknown, not as job-level counts missing a leg's traffic.
    funnel: Final = counted if counted is not None and counted.legs_with_rows == len(leg_ids) else None
    result: Final = ShadowEvalResult(
        by_tier=_slices(by_tier),
        by_current_model=_slices(by_model),
        by_router=_slices(by_router),
        overall_shadow_win_rate_pct=_pct_of(sum(r.shadow_wins for r in by_tier), total_turns),
        overall_tie_rate_pct=_pct_of(sum(r.ties for r in by_tier), total_turns),
        sampled_real_spend=sum(r.real_spend for r in by_tier),
        sampled_shadow_spend=sum(r.shadow_spend for r in by_tier),
        not_sampled_count=funnel.not_sampled if funnel is not None else None,
        unjudgeable_count=funnel.unjudgeable if funnel is not None else None,
        shed_count=funnel.shed if funnel is not None else None,
        withheld_count=funnel.withheld if funnel is not None else None,
    )
    return result, verdicts_by_target


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
    Start a shadow eval: duplicate a sampled slice of one or more targets' live traffic
    against a second arm, judge the two responses blind, and stratify win rates by tier,
    by the model that served the real arm, and by target.

    A target is a virtual key, a team, or a user. Team and user targets match on the
    identity every request resolves to at auth time, so they cover JWT-authenticated
    traffic, which presents no virtual key; a user target samples that user's traffic
    across all their teams, whether it arrives on a JWT or a key they own.

    A forward job answers whether the targets should adopt router_name: it samples the
    requests the router did not serve and duplicates them through it. A reverse job
    answers whether a target already on the router still gains from it: it samples the
    requests the router did serve and duplicates them against baseline_model. A target
    can hold one active job per direction, so both questions can run at once, and a
    request matching several jobs' targets (say its key and its team) is sampled by
    each, separately budgeted.

    Shadow responses are never served to users. Each target samples until its recorded
    eval spend, the shadow and judge calls' own cost, reaches max_budget dollars, the
    job's window ends, or the job is stopped, so one target running out of budget does
    not end sampling for the others; sampling changes propagate to pods within about 10
    seconds. Shadow and judge calls bill to the sampled request's own identity but are
    excluded from request counts and auto-router adoption metrics.
    """
    from litellm.proxy.proxy_server import llm_router, prisma_client

    _require_admin_writer(user_api_key_dict, "start a shadow eval")
    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    unconfigured: Final = tuple(
        name
        for name in data.router_names
        if llm_router is None or not _is_configured_pre_routing_strategy(llm_router, name)
    )
    if unconfigured:
        raise HTTPException(
            status_code=400, detail=f"Not a configured auto-router: {', '.join(repr(n) for n in unconfigured)}"
        )
    token_rows: Final = (
        await _verification_tokens(prisma_client).find_many(
            where={"token": {"in": list(data.api_key_ids)}}  # mutable-ok: Prisma filter
        )
        if data.api_key_ids
        else ()
    )
    team_rows: Final = (
        await _team_rows(prisma_client).find_many(
            where={"team_id": {"in": list(data.team_ids)}}  # mutable-ok: Prisma filter
        )
        if data.team_ids
        else ()
    )
    user_rows: Final = (
        await _user_rows(prisma_client).find_many(
            where={"user_id": {"in": list(data.user_ids)}}  # mutable-ok: Prisma filter
        )
        if data.user_ids
        else ()
    )
    unknown_keys: Final = sorted(frozenset(data.api_key_ids) - frozenset(row.token for row in token_rows or ()))
    unknown_teams: Final = sorted(frozenset(data.team_ids) - frozenset(row.team_id for row in team_rows or ()))
    unknown_users: Final = sorted(frozenset(data.user_ids) - frozenset(row.user_id for row in user_rows or ()))
    unknown_parts: Final = tuple(
        part
        for part in (
            (
                f"api_key_ids not on this proxy: {', '.join(unknown_keys)}; pass each key's token hash, "
                "the value the key list and key info endpoints report"
            )
            if unknown_keys
            else None,
            f"team_ids not on this proxy: {', '.join(unknown_teams)}" if unknown_teams else None,
            f"user_ids not on this proxy: {', '.join(unknown_users)}" if unknown_users else None,
        )
        if part is not None
    )
    if unknown_parts:
        raise HTTPException(status_code=400, detail=". ".join(unknown_parts))

    # Every model check below runs once per team the job samples for, since that is the
    # identity the shadow and judge calls carry and therefore what the router selects on.
    # A user target's traffic can span teams, so it validates unscoped (None); each
    # sampled attempt still resolves the judge under its own request's team at eval time.
    team_ids: Final = tuple(
        dict.fromkeys(
            (
                *(row.team_id for row in token_rows or ()),
                *data.team_ids,
                *((None,) if data.user_ids else ()),
            )
        )
    )
    _validate_plain_model(llm_router, data.judge_model, "judge_model", team_ids)
    if data.baseline_model is not None:
        _validate_plain_model(llm_router, data.baseline_model, "baseline_model", team_ids)
    _validate_judge_is_not_a_candidate(llm_router, data, team_ids)

    requested_targets: Final[tuple[tuple[ShadowEvalTargetType, str], ...]] = (
        *(("key", key) for key in data.api_key_ids),
        *(("team", team) for team in data.team_ids),
        *(("user", user) for user in data.user_ids),
    )
    requested_by_type: Final[tuple[tuple[ShadowEvalTargetType, tuple[str, ...]], ...]] = tuple(
        (target_type, ids)
        for target_type, ids in (("key", data.api_key_ids), ("team", data.team_ids), ("user", data.user_ids))
        if ids
    )
    # A job whose window passed or whose budget ran out stopped sampling on its own,
    # but its legs still hold their slots in the per-target, per-direction partial unique
    # index until stamped; free them so a new eval can start. Sweeping both directions is
    # deliberate. Sweep and claim filter on exact (target_type, id) pairs so a team id
    # that happens to equal a key hash never matches the other kind's slot.
    for target_type, ids in requested_by_type:
        await prisma_client.db.execute_raw(_SWEEP_FINISHED_JOBS_SQL, list(ids), target_type)  # mutable-ok: query param
    claimed: Final = await _shadow_eval_jobs(prisma_client).find_many(
        where={  # mutable-ok: Prisma filter
            "OR": [  # mutable-ok: Prisma filter
                {"target_type": target_type, "target_id": {"in": list(ids)}}  # mutable-ok: Prisma filter
                for target_type, ids in requested_by_type
            ],
            "direction": data.direction,
            "stopped_at": None,
        },
    )
    if claimed:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Already in an active {data.direction} shadow eval job: "
                + ", ".join(sorted(f"{row.target_type} {row.target_id} (job {row.group_id})" for row in claimed))
                + ". Stop it first."
            ),
        )
    now: Final = datetime.now(timezone.utc)
    group_id: Final = str(uuid4())
    ends_at: Final = now + timedelta(days=data.duration_days)
    shared_config: Final = {  # mutable-ok: Prisma payload
        "group_id": group_id,
        # a pre-router_names pod samples router_name alone, so it must be a real arm
        "router_name": data.router_names[0],
        "router_names": list(data.router_names),  # mutable-ok: Prisma payload
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
        # Leg ids are minted here rather than by the DB default so the funnel seed below
        # writes from the same values with no read-back, which a lagging read replica
        # (DATABASE_URL_READ_REPLICA) could otherwise return empty.
        leg_ids: Final = tuple(str(uuid4()) for _ in requested_targets)
        await _shadow_eval_jobs(prisma_client).create_many(
            data=[  # mutable-ok: Prisma payload
                {  # mutable-ok: Prisma payload
                    **shared_config,
                    "id": leg_id,
                    "target_type": target_type,
                    "target_id": target_id,
                }  # mutable-ok: Prisma payload
                for leg_id, (target_type, target_id) in zip(leg_ids, requested_targets)
            ]
        )
    except Exception as e:
        if not _is_unique_violation(e):
            raise
        raise HTTPException(
            status_code=409,
            detail=(
                f"A requested target was claimed by another {data.direction} shadow eval job concurrently. "
                "Stop it first."
            ),
        ) from e
    # Seed a zero funnel row per leg NOW: a fully covered job never skips a request, so
    # waiting for the first skip would leave it indistinguishable from a pre-funnel job
    # (null coverage). A failed seed degrades this job to exactly that, nothing worse.
    try:
        await _shadow_eval_funnel(prisma_client).create_many(
            data=[{"job_id": leg_id} for leg_id in leg_ids],  # mutable-ok: Prisma payload
            skip_duplicates=True,
        )
    except Exception as seed_err:  # noqa: BLE001  # coverage is advisory; the job must still start
        verbose_proxy_logger.error("shadow_eval: funnel seed failed for job %s: %s", group_id, seed_err)
    labels: Final = _target_labels(token_rows or (), team_rows or (), user_rows or ())
    return ShadowEvalJobResponse(
        job_id=group_id,
        targets=tuple(
            ShadowEvalJobTargetResponse(
                target_type=target_type,
                target_id=target_id,
                max_turns=SHADOW_EVAL_TURN_VALVE,
                max_budget=data.max_budget,
                target_alias=labels.get((target_type, target_id), _NO_TARGET_LABELS)[0],
                key_name=labels.get((target_type, target_id), _NO_TARGET_LABELS)[1],
            )
            for target_type, target_id in sorted(requested_targets)
        ),
        router_names=data.router_names,
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
    target_type: Annotated[
        ShadowEvalTargetType | None, Query(description="Kind of target to filter on; requires target_id")
    ] = None,
    target_id: Annotated[
        str | None, Query(description="Filter to jobs that shadow this target, alone or alongside others")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=200, description="Newest jobs to return")] = 50,
) -> tuple[ShadowEvalJobResponse, ...]:
    """List shadow eval jobs, newest first, each target with its attempt count so status
    is accurate. Judged counts, spend, and results ride the detail endpoint only."""
    from litellm.proxy.proxy_server import prisma_client

    _require_admin_viewer(user_api_key_dict, "view shadow evals")
    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    filter_type: Final = target_type if isinstance(target_type, str) else None
    filter_id: Final = target_id if isinstance(target_id, str) else None
    if (filter_type is None) != (filter_id is None):
        raise HTTPException(status_code=400, detail="target_type and target_id filter together; pass both or neither")
    legs: Final = _LEG_ROWS.validate_python(
        (
            await _query_raw(prisma_client, _LIST_LEGS_BY_TARGET_SQL, limit, filter_type, filter_id)
            if filter_type and filter_id
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
    return await _with_target_labels(
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
    labeled: Final = await _with_target_labels(
        prisma_client, (_group_response(job_id, legs, await _leg_attempt_counts(prisma_client, legs)),)
    )
    results, verdicts_by_target = await _shadow_eval_results(prisma_client, legs)
    return labeled[0].model_copy(
        update={  # mutable-ok: pydantic update payload
            "judged_count": totals[0].judged_count if totals else 0,
            "error_count": totals[0].error_count if totals else 0,
            "judge_spend": round(totals[0].judge_spend, 6) if totals else 0.0,
            "last_error": latest_error.error if latest_error else None,
            "results": results,
            "targets": tuple(
                target.model_copy(
                    update={  # mutable-ok: pydantic update payload
                        "verdicts": verdicts_by_target.get((target.target_type, target.target_id))
                    }
                )
                for target in labeled[0].targets
            ),
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
    """Stop an active shadow eval job, every target it scopes at once. Attempts are kept;
    sampling halts within ~10s. Targets that already stopped on their own budget keep the
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
    labeled: Final = await _with_target_labels(prisma_client, (current,))
    return labeled[0]
