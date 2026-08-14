"""
Unit tests for auto router management endpoints
"""

import os
import sys

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

sys.path.insert(0, os.path.abspath("../../../.."))  # Adds the parent directory to the system path

from litellm.proxy._types import (
    LitellmUserRoles,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.management_endpoints.auto_router_endpoints import (
    preview_auto_router_routing,
)
from litellm.router import Router
from litellm.types.utils import Choices, Message, ModelResponse
from litellm.types.management_endpoints.auto_router_endpoints import (
    AutoRouterRoutingTestRequest,
)

ADMIN = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-test", user_id="admin")

TIERS = {
    "SIMPLE": ["cheap-model"],
    "MEDIUM": ["mid-model"],
    "COMPLEX": ["strong-model"],
    "REASONING": ["reasoning-model"],
}


def _router() -> Router:
    return Router(
        model_list=[
            {"model_name": name, "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "fake-key"}}
            for name in ("cheap-model", "mid-model", "strong-model", "reasoning-model")
        ]
    )


def _request(prompt: str, **config_overrides: object) -> AutoRouterRoutingTestRequest:
    return AutoRouterRoutingTestRequest.model_validate(
        {
            "prompt": prompt,
            "complexity_router_config": {"tiers": TIERS, "classifier_type": "heuristic", **config_overrides},
        }
    )


async def _route(prompt: str, monkeypatch: pytest.MonkeyPatch, **config_overrides: object):
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "llm_router", _router())
    return await preview_auto_router_routing(
        data=_request(prompt, **config_overrides),
        user_api_key_dict=ADMIN,
    )


@pytest.mark.asyncio
async def test_simple_prompt_routes_to_the_simple_tier(monkeypatch: pytest.MonkeyPatch):
    response = await _route("what is 2+2", monkeypatch)

    assert response.routed_model == "cheap-model"
    assert response.routed_model_configured is True
    assert response.routing_decision["tier"] == "SIMPLE"
    assert response.routing_decision["cause"] == "heuristic_scorer"
    assert response.routing_decision["routed_model"] == "cheap-model"
    assert "score" in response.routing_decision


@pytest.mark.asyncio
async def test_reasoning_markers_route_to_the_reasoning_tier(monkeypatch: pytest.MonkeyPatch):
    response = await _route(
        "think step by step and explain your reasoning about sharding this table",
        monkeypatch,
    )

    assert response.routed_model == "reasoning-model"
    assert response.routing_decision["tier"] == "REASONING"


@pytest.mark.asyncio
async def test_keyword_rule_beats_the_heuristic_scorer(monkeypatch: pytest.MonkeyPatch):
    response = await _route(
        "what is 2+2",
        monkeypatch,
        keyword_tier_rules=[{"keywords": ["2+2"], "tier": "COMPLEX"}],
    )

    assert response.routed_model == "strong-model"
    assert response.routing_decision["cause"] == "literal_keyword_match"
    assert response.routing_decision["matched_keyword"] == "2+2"


@pytest.mark.asyncio
async def test_escalation_keyword_bumps_the_classified_tier(monkeypatch: pytest.MonkeyPatch):
    response = await _route("what is 2+2, ultrathink", monkeypatch, escalation_keywords=["ultrathink"])

    assert response.routed_model == "mid-model"
    assert response.routing_decision["escalated"] is True
    assert response.routing_decision["escalation_keyword"] == "ultrathink"


@pytest.mark.asyncio
async def test_tier_model_missing_from_the_proxy_is_reported(monkeypatch: pytest.MonkeyPatch):
    response = await _route("what is 2+2", monkeypatch, tiers={**TIERS, "SIMPLE": ["never-configured"]})

    assert response.routed_model == "never-configured"
    assert response.routed_model_configured is False


@pytest.mark.asyncio
async def test_llm_classifier_call_is_billed_to_the_calling_key(monkeypatch: pytest.MonkeyPatch):
    import litellm.proxy.proxy_server as proxy_server

    router = _router()
    calls: list[dict] = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return ModelResponse(
            choices=[Choices(message=Message(content='{"tier": "COMPLEX"}'))],
            model="classifier-model",
        )

    monkeypatch.setattr(router, "acompletion", fake_acompletion)
    monkeypatch.setattr(proxy_server, "llm_router", router)

    response = await preview_auto_router_routing(
        data=_request(
            "what is 2+2",
            classifier_type="llm",
            classifier_llm_config={"model": "classifier-model"},
        ),
        user_api_key_dict=ADMIN,
    )

    assert response.routed_model == "strong-model"
    assert len(calls) == 1
    assert calls[0]["metadata"]["user_api_key"] == ADMIN.api_key
    assert calls[0]["metadata"]["user_api_key_user_id"] == ADMIN.user_id


@pytest.mark.parametrize(
    "config_overrides",
    [
        {"classifier_type": "llm", "classifier_llm_config": {"model": "classifier-model"}},
        {
            "semantic_keyword_matching": True,
            "embedding_model": "classifier-model",
            "keyword_tier_rules": [{"keywords": ["2+2"], "tier": "COMPLEX"}],
        },
    ],
)
@pytest.mark.asyncio
async def test_a_key_that_cannot_call_the_classifier_model_is_rejected_before_it_is_called(
    monkeypatch: pytest.MonkeyPatch, config_overrides: dict
):
    import litellm.proxy.proxy_server as proxy_server

    router = _router()
    calls: list[dict] = []

    async def fail_if_called(**kwargs):
        calls.append(kwargs)
        raise AssertionError("the classifier must not be called by a key that cannot call it")

    monkeypatch.setattr(router, "acompletion", fail_if_called)
    monkeypatch.setattr(router, "aembedding", fail_if_called)
    monkeypatch.setattr(proxy_server, "llm_router", router)

    with pytest.raises(ProxyException) as exc_info:
        await preview_auto_router_routing(
            data=_request("what is 2+2", **config_overrides),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-restricted",
                user_id="admin",
                models=["cheap-model"],
            ),
        )

    assert exc_info.value.type == ProxyErrorTypes.key_model_access_denied
    assert calls == []


@pytest.mark.asyncio
async def test_a_key_over_its_budget_cannot_run_a_classifier_config(monkeypatch: pytest.MonkeyPatch):
    import litellm.proxy.proxy_server as proxy_server

    router = _router()
    calls: list[dict] = []

    async def fail_if_called(**kwargs):
        calls.append(kwargs)
        raise AssertionError("an exhausted key must not reach the classifier")

    monkeypatch.setattr(router, "acompletion", fail_if_called)
    monkeypatch.setattr(proxy_server, "llm_router", router)

    with pytest.raises(ProxyException) as exc_info:
        await preview_auto_router_routing(
            data=_request(
                "what is 2+2",
                classifier_type="llm",
                classifier_llm_config={"model": "classifier-model"},
            ),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-broke",
                user_id="admin",
                max_budget=1.0,
                spend=2.0,
            ),
        )

    assert exc_info.value.type == ProxyErrorTypes.budget_exceeded
    assert calls == []


@pytest.mark.asyncio
async def test_a_heuristic_config_does_not_need_a_budget(monkeypatch: pytest.MonkeyPatch):
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "llm_router", _router())

    response = await preview_auto_router_routing(
        data=_request("what is 2+2"),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-broke",
            user_id="admin",
            max_budget=1.0,
            spend=2.0,
            models=["cheap-model"],
        ),
    )

    assert response.routed_model == "cheap-model"


@pytest.mark.asyncio
async def test_no_llm_router_on_the_proxy_is_a_500(monkeypatch: pytest.MonkeyPatch):
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "llm_router", None)

    with pytest.raises(HTTPException) as exc_info:
        await preview_auto_router_routing(data=_request("what is 2+2"), user_api_key_dict=ADMIN)

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_non_admin_without_a_team_is_rejected(monkeypatch: pytest.MonkeyPatch):
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "llm_router", _router())

    with pytest.raises(HTTPException) as exc_info:
        await preview_auto_router_routing(
            data=_request("what is 2+2"),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.INTERNAL_USER, api_key="sk-user", user_id="user"
            ),
        )

    assert exc_info.value.status_code == 403


def test_blank_prompt_is_rejected():
    with pytest.raises(ValidationError):
        _request("   ")


def test_semantic_matching_without_an_embedding_model_is_rejected():
    with pytest.raises(ValidationError):
        _request("what is 2+2", semantic_keyword_matching=True)


class TestAutoRouterBenchmarks:
    from litellm.proxy.management_endpoints.auto_router_endpoints import _SessionAggRow

    ROW = _SessionAggRow(
        router_name="live-auto",
        router_type="complexity",
        tier_turns={},
        sessions=4,
        turns=40,
        unordered_turns=1,
        covered_turns=38,
        cache_hits=28,
        same_model_turns=20,
        same_model_hits=19,
        first_visit_turns=8,
        first_visit_hits=2,
        return_turns=11,
        return_hits=6,
        return_expired_misses=2,
        return_within_ttl_misses=1,
        ttl_5m_turns=30,
        ttl_1h_turns=5,
        total_tokens=4000,
        spend=10.0,
        saved_spend=30.0,
        session_seconds=400.0,
    )

    def test_overall_hit_rate_counts_hits_independently_of_bucketing(self):
        from litellm.proxy.management_endpoints.auto_router_endpoints import _benchmark_totals

        totals = _benchmark_totals(self.ROW)
        bucket_hits = (
            totals.cache.same_model.hits + totals.cache.first_visit.hits + totals.cache.return_to_tier.hits
        )
        assert bucket_hits == 27
        assert totals.cache.hit_rate_pct == pytest.approx(100.0 * 28 / 38, abs=0.1)

    def test_fold_math_matches_hand_computed_truth(self):
        from litellm.proxy.management_endpoints.auto_router_endpoints import _benchmark_totals

        totals = _benchmark_totals(self.ROW)
        assert totals.sessions == 4
        assert totals.turns == 40
        assert totals.avg_turns_per_session == 10.0
        assert totals.avg_session_seconds == 100.0
        assert totals.avg_tokens_per_session == 1000.0
        assert totals.baseline_spend == 40.0
        assert totals.saved_pct == 75.0
        assert totals.saved_per_session == 7.5
        assert totals.cache.coverage_pct == 95.0
        assert totals.cache.hit_rate_pct == pytest.approx(73.7)
        assert totals.cache.same_model.hit_rate_pct == 95.0
        assert totals.cache.first_visit.hit_rate_pct == 25.0
        assert totals.cache.return_to_tier.hit_rate_pct == pytest.approx(54.5)
        assert totals.cache.return_misses_unknown == 2
        assert totals.cache.unordered_turns == 1

    def test_a_losing_router_reports_negative_savings(self):
        from litellm.proxy.management_endpoints.auto_router_endpoints import _benchmark_totals

        losing = self.ROW.model_copy(update={"saved_spend": -5.0})
        totals = _benchmark_totals(losing)
        assert totals.baseline_spend == 5.0
        assert totals.saved_pct == -100.0

    def test_an_empty_window_folds_to_zeros(self):
        from litellm.proxy.management_endpoints.auto_router_endpoints import (
            _benchmark_totals,
            _summed_agg_row,
        )

        totals = _benchmark_totals(_summed_agg_row([]))
        assert totals.sessions == 0
        assert totals.turns == 0
        assert totals.saved_pct == 0.0
        assert totals.cache.hit_rate_pct == 0.0

    def test_totals_sum_counters_across_groups_before_deriving_ratios(self):
        from litellm.proxy.management_endpoints.auto_router_endpoints import (
            _benchmark_totals,
            _summed_agg_row,
        )

        other = self.ROW.model_copy(update={"router_name": "auto-2", "sessions": 1, "turns": 10, "spend": 0.0})
        summed = _summed_agg_row([self.ROW, other])
        totals = _benchmark_totals(summed)
        assert summed.sessions == 5
        assert summed.turns == 50
        assert totals.avg_turns_per_session == 10.0
        assert totals.spend == 10.0

    def test_tier_names_stay_scoped_to_the_router_type_that_recorded_them(self):
        quality = self.ROW.model_copy(
            update={"router_name": "quality-auto", "router_type": "quality", "tier_turns": {"2": 7}}
        )
        complexity = self.ROW.model_copy(update={"tier_turns": {"medium": 7}})
        assert complexity.tier_turns == {"medium": 7}
        assert quality.tier_turns == {"2": 7}

    def test_summed_totals_carry_no_tier_map_because_names_are_router_scoped(self):
        from litellm.proxy.management_endpoints.auto_router_endpoints import _summed_agg_row

        quality = self.ROW.model_copy(update={"router_type": "quality", "tier_turns": {"2": 7}})
        complexity = self.ROW.model_copy(update={"tier_turns": {"medium": 7}})
        assert _summed_agg_row([complexity, quality]).tier_turns == {}

    @pytest.mark.asyncio
    async def test_non_admin_roles_cannot_read_benchmarks(self):
        from litellm.proxy.management_endpoints.auto_router_endpoints import get_auto_router_benchmarks

        with pytest.raises(HTTPException) as err:
            await get_auto_router_benchmarks(
                user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, api_key="sk-x"),
                start_date="2026-08-01",
                end_date="2026-08-02",
            )
        assert err.value.status_code == 403

    @pytest.mark.asyncio
    async def test_a_reversed_window_is_rejected(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.auto_router_endpoints import get_auto_router_benchmarks

        monkeypatch.setattr(proxy_server, "prisma_client", object())
        with pytest.raises(HTTPException) as err:
            await get_auto_router_benchmarks(
                user_api_key_dict=ADMIN,
                start_date="2026-08-05",
                end_date="2026-08-01",
            )
        assert err.value.status_code == 400

    @pytest.mark.asyncio
    async def test_endpoint_returns_groups_and_totals_from_the_rollup(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.auto_router_endpoints import get_auto_router_benchmarks

        captured: dict = {}

        class _DB:
            async def query_raw(self, sql: str, *params: object):
                captured["sql"] = sql
                captured["params"] = params
                return [TestAutoRouterBenchmarks.ROW.model_dump()]

        monkeypatch.setattr(proxy_server, "prisma_client", type("P", (), {"db": _DB()})())

        response = await get_auto_router_benchmarks(
            user_api_key_dict=ADMIN,
            start_date="2026-07-01",
            end_date="2026-08-01",
        )
        assert captured["params"] == ("2026-07-01T00:00:00", "2026-08-02T00:00:00")
        assert response.routers_in_scope == 1
        assert response.groups[0].router_name == "live-auto"
        assert response.groups[0].saved_pct == response.totals.saved_pct == 75.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "wire_value, expected", [({"simple": 24, "complex": 16}, {"simple": 24, "complex": 16}), ({}, {})]
    )
    async def test_the_tier_map_reaches_the_response_as_the_jsonb_column_returns_it(
        self, wire_value: dict, expected: dict, monkeypatch: pytest.MonkeyPatch
    ):
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.auto_router_endpoints import get_auto_router_benchmarks

        class _DB:
            async def query_raw(self, sql: str, *params: object):
                return [{**TestAutoRouterBenchmarks.ROW.model_dump(), "tier_turns": wire_value}]

        monkeypatch.setattr(proxy_server, "prisma_client", type("P", (), {"db": _DB()})())

        response = await get_auto_router_benchmarks(
            user_api_key_dict=ADMIN,
            start_date="2026-07-01",
            end_date="2026-08-01",
        )
        assert response.groups[0].tier_turns == expected


# ---------------------------------------------------------------------------
# Shadow eval endpoints
# ---------------------------------------------------------------------------

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException

from litellm.proxy.management_endpoints.auto_router_endpoints import (
    get_shadow_eval_job,
    list_shadow_eval_jobs,
    start_shadow_eval,
    stop_shadow_eval_job,
)
from litellm.types.management_endpoints.auto_router_endpoints import ShadowEvalJobResponse, StartShadowEvalRequest

VIEWER = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, api_key="sk-view", user_id="viewer")
NON_ADMIN = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, api_key="sk-user", user_id="user")


def _shadow_router() -> MagicMock:
    router = MagicMock()
    router.auto_routers = {}
    router.complexity_routers = {"my-router": [MagicMock()]}
    router.adaptive_routers = {}
    router.quality_routers = {}
    router.model_group_alias = {}
    router.get_model_list = MagicMock(return_value=None)
    return router


def _job_record(**overrides: object) -> MagicMock:
    """Spec'd like a real prisma row: only the table's columns exist as attributes, so
    from_attributes validation falls back to model defaults for everything else."""
    defaults = {
        "id": "job-1",
        "api_key_id": "key-hash",
        "router_name": "my-router",
        "judge_model": "anthropic/claude-sonnet-5",
        "shadow_percentage": 10.0,
        "max_turns": 200,
        "created_at": datetime(2026, 8, 11, tzinfo=timezone.utc),
        "ends_at": datetime.now(timezone.utc) + timedelta(days=7),
        "stopped_at": None,
    }
    fields = {**defaults, **overrides}
    record = MagicMock(spec=list(fields))
    for key, value in fields.items():
        setattr(record, key, value)
    return record


def _shadow_prisma(active_job=None, agg_rows=None) -> MagicMock:
    prisma = MagicMock()
    prisma.db.litellm_verificationtoken.find_unique = AsyncMock(return_value=MagicMock())
    prisma.db.execute_raw = AsyncMock(return_value=0)
    prisma.db.litellm_shadowevaljob.find_first = AsyncMock(return_value=active_job)
    prisma.db.litellm_shadowevaljob.find_unique = AsyncMock(return_value=None)
    prisma.db.litellm_shadowevaljob.find_many = AsyncMock(return_value=[])
    prisma.db.litellm_shadowevaljob.create = AsyncMock(return_value=_job_record())
    prisma.db.litellm_shadowevaljob.update = AsyncMock(
        return_value=_job_record(stopped_at=datetime.now(timezone.utc))
    )
    prisma.db.litellm_shadowevalattempt.find_first = AsyncMock(return_value=None)

    async def query_raw(sql: str, *params: object):
        if "FILTER (WHERE outcome != 'error')::int AS judged_count" in sql:
            return [{"judged_count": 10, "error_count": 2, "judge_spend": 0.031}]
        return agg_rows if agg_rows is not None else []

    prisma.db.query_raw = AsyncMock(side_effect=query_raw)
    return prisma


def _start_request(**overrides: object) -> StartShadowEvalRequest:
    payload = {
        "api_key_id": "key-hash",
        "router_name": "my-router",
        "shadow_percentage": 10.0,
        "judge_model": "anthropic/claude-sonnet-5",
        "duration_days": 7,
        "max_turns": 200,
    }
    payload.update(overrides)
    return StartShadowEvalRequest.model_validate(payload)


@pytest.mark.asyncio
async def test_start_shadow_eval_creates_job_and_frees_expired_or_exhausted_ones(monkeypatch: pytest.MonkeyPatch):
    """Expiry and turn-budget exhaustion both end sampling on their own; either must
    release the one-active-per-key index so a new eval can start."""
    import litellm.proxy.proxy_server as proxy_server

    prisma = _shadow_prisma()
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(proxy_server, "llm_router", _shadow_router())

    response = await start_shadow_eval(_start_request(), ADMIN)

    assert response.status == "running"
    assert response.max_turns == 200
    assert response.judged_count is None
    sweep_sql, sweep_key = prisma.db.execute_raw.call_args.args
    assert "stopped_at IS NULL" in sweep_sql
    assert "ends_at <= NOW()" in sweep_sql
    assert ">= j.max_turns" in sweep_sql
    assert sweep_key == "key-hash"
    create_data = prisma.db.litellm_shadowevaljob.create.call_args.kwargs["data"]
    assert create_data["api_key_id"] == "key-hash"
    assert create_data["created_by"] == "admin"
    assert "status" not in create_data


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "caller,request_overrides,active,expected_status",
    [
        (NON_ADMIN, {}, None, 403),
        (VIEWER, {}, None, 403),
        (ADMIN, {"router_name": "not-a-router"}, None, 400),
        (ADMIN, {"judge_model": "not/a real model!"}, None, 400),
        (ADMIN, {"judge_model": "my-router"}, None, 400),
        (ADMIN, {}, "active", 409),
    ],
    ids=["non-admin", "view-only", "unknown-router", "unresolvable-judge", "router-as-judge", "already-active"],
)
async def test_start_shadow_eval_rejections(
    monkeypatch: pytest.MonkeyPatch, caller, request_overrides, active, expected_status
):
    import litellm.proxy.proxy_server as proxy_server

    prisma = _shadow_prisma(active_job=_job_record() if active else None)
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(proxy_server, "llm_router", _shadow_router())

    with pytest.raises(HTTPException) as exc:
        await start_shadow_eval(_start_request(**request_overrides), caller)
    assert exc.value.status_code == expected_status


@pytest.mark.asyncio
async def test_start_shadow_eval_rejects_a_key_this_proxy_does_not_know(monkeypatch: pytest.MonkeyPatch):
    """A typo'd api_key_id would otherwise create a job no traffic can ever match."""
    import litellm.proxy.proxy_server as proxy_server

    prisma = _shadow_prisma()
    prisma.db.litellm_verificationtoken.find_unique = AsyncMock(return_value=None)
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(proxy_server, "llm_router", _shadow_router())

    with pytest.raises(HTTPException) as exc:
        await start_shadow_eval(_start_request(), ADMIN)
    assert exc.value.status_code == 400
    assert "not a key on this proxy" in exc.value.detail


@pytest.mark.asyncio
async def test_start_shadow_eval_concurrent_unique_violation_is_a_409(monkeypatch: pytest.MonkeyPatch):
    import litellm.proxy.proxy_server as proxy_server
    from prisma.errors import UniqueViolationError

    prisma = _shadow_prisma()
    prisma.db.litellm_shadowevaljob.create = AsyncMock(
        side_effect=UniqueViolationError(MagicMock(message="unique constraint"))
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(proxy_server, "llm_router", _shadow_router())

    with pytest.raises(HTTPException) as exc:
        await start_shadow_eval(_start_request(), ADMIN)
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_get_shadow_eval_job_derives_counts_spend_and_stratified_results(monkeypatch: pytest.MonkeyPatch):
    import litellm.proxy.proxy_server as proxy_server

    tier_rows = [
        {"grp": "SIMPLE", "turn_count": 8, "real_wins": 2, "shadow_wins": 4, "ties": 2, "avg_confidence": 0.8},
        {"grp": "REASONING", "turn_count": 2, "real_wins": 2, "shadow_wins": 0, "ties": 0, "avg_confidence": 0.9},
    ]
    prisma = _shadow_prisma(agg_rows=tier_rows)
    prisma.db.litellm_shadowevaljob.find_unique = AsyncMock(return_value=_job_record())
    prisma.db.litellm_shadowevalattempt.find_first = AsyncMock(
        return_value=MagicMock(error="judge call failed: boom")
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)

    response = await get_shadow_eval_job("job-1", VIEWER)

    assert response.job_id == "job-1"
    assert response.status == "running"
    assert response.judged_count == 10
    assert response.error_count == 2
    assert response.judge_spend == 0.031
    assert response.last_error == "judge call failed: boom"
    assert [s.group for s in response.results.by_tier] == ["SIMPLE", "REASONING"]
    assert response.results.by_tier[0].shadow_win_rate_pct == 50.0
    assert response.results.overall_shadow_win_rate_pct == 40.0
    assert response.results.overall_tie_rate_pct == 20.0


@pytest.mark.asyncio
async def test_get_shadow_eval_job_404s_and_gates_on_role(monkeypatch: pytest.MonkeyPatch):
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", _shadow_prisma())

    with pytest.raises(HTTPException) as missing:
        await get_shadow_eval_job("nope", VIEWER)
    assert missing.value.status_code == 404

    with pytest.raises(HTTPException) as forbidden:
        await get_shadow_eval_job("job-1", NON_ADMIN)
    assert forbidden.value.status_code == 403


@pytest.mark.asyncio
async def test_list_shadow_eval_jobs_returns_derived_status_without_aggregates(monkeypatch: pytest.MonkeyPatch):
    import litellm.proxy.proxy_server as proxy_server

    prisma = _shadow_prisma()
    prisma.db.litellm_shadowevaljob.find_many = AsyncMock(
        return_value=[
            _job_record(),
            _job_record(id="job-2", ends_at=datetime.now(timezone.utc) - timedelta(days=1)),
            _job_record(id="job-3", stopped_at=datetime.now(timezone.utc)),
        ]
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)

    jobs = await list_shadow_eval_jobs(VIEWER, api_key_id=None, limit=50)

    assert [job.status for job in jobs] == ["running", "completed", "stopped"]
    swept = ShadowEvalJobResponse.model_validate(
        _job_record(
            id="job-4",
            ends_at=datetime.now(timezone.utc) - timedelta(days=1),
            stopped_at=datetime.now(timezone.utc),
        ),
        from_attributes=True,
    )
    assert swept.status == "completed"
    assert all(job.judged_count is None and job.results is None for job in jobs)
    assert prisma.db.query_raw.await_count == 0


@pytest.mark.asyncio
async def test_stop_shadow_eval_sets_stopped_at_and_rejects_non_running(monkeypatch: pytest.MonkeyPatch):
    import litellm.proxy.proxy_server as proxy_server

    prisma = _shadow_prisma()
    prisma.db.litellm_shadowevaljob.find_unique = AsyncMock(return_value=_job_record())
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)

    stopped = await stop_shadow_eval_job("job-1", ADMIN)
    assert stopped.status == "stopped"
    update = prisma.db.litellm_shadowevaljob.update.call_args.kwargs
    assert set(update["data"]) == {"stopped_at"}

    prisma.db.litellm_shadowevaljob.find_unique = AsyncMock(
        return_value=_job_record(ends_at=datetime.now(timezone.utc) - timedelta(days=1))
    )
    with pytest.raises(HTTPException) as exc:
        await stop_shadow_eval_job("job-1", ADMIN)
    assert exc.value.status_code == 400

    with pytest.raises(HTTPException) as forbidden:
        await stop_shadow_eval_job("job-1", VIEWER)
    assert forbidden.value.status_code == 403
