"""
Unit tests for auto router management endpoints
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

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
from litellm.types.management_endpoints.auto_router_endpoints import (
    AutoRouterRoutingTestRequest,
    StartShadowEvalRequest,
)
from litellm.types.utils import Choices, Message, ModelResponse

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
    from litellm.proxy import proxy_server

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
    from litellm.proxy import proxy_server

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
    from litellm.proxy import proxy_server

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
    from litellm.proxy import proxy_server

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
    from litellm.proxy import proxy_server

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
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "llm_router", None)

    with pytest.raises(HTTPException) as exc_info:
        await preview_auto_router_routing(data=_request("what is 2+2"), user_api_key_dict=ADMIN)

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_non_admin_without_a_team_is_rejected(monkeypatch: pytest.MonkeyPatch):
    from litellm.proxy import proxy_server

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
        bucket_hits = totals.cache.same_model.hits + totals.cache.first_visit.hits + totals.cache.return_to_tier.hits
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


class TestShadowEvalJobResponseNamesWhatIsBeingEvaluated:
    """A shadow eval only samples one key's traffic. An admin running several jobs cannot
    tell which key a win rate belongs to unless the job response says so."""

    @staticmethod
    def _record(**overrides: object):
        from datetime import datetime, timezone

        fields = {
            "id": "job-1",
            "status": "running",
            "router_name": "claude-auto",
            "api_key_id": "hashed-key-abc",
            "team_id": "team-7",
            "shadow_percentage": 10.0,
            "request_count": 100,
            "completed_count": 9,
            "failed_count": 1,
            "cost_estimate": 3.0,
            "cost_actual": 0.42,
            "created_at": datetime(2026, 8, 1, tzinfo=timezone.utc),
            "completed_at": None,
            **overrides,
        }
        return type("Row", (), fields)()

    def test_response_reports_the_shadowed_key_and_its_team(self):
        from litellm.proxy.management_endpoints.auto_router_endpoints import _job_to_response

        response = _job_to_response(self._record(), None)

        assert response.api_key_id == "hashed-key-abc"
        assert response.team_id == "team-7"

    def test_a_keyless_team_still_produces_a_response(self):
        from litellm.proxy.management_endpoints.auto_router_endpoints import _job_to_response

        response = _job_to_response(self._record(team_id=None), None)

        assert response.api_key_id == "hashed-key-abc"
        assert response.team_id is None


class TestJudgeCostEstimate:
    """The upfront estimate must price the judge's real output budget, not a stale hardcoded one."""

    JUDGE_MODEL = "gpt-4o-mini"

    def test_estimate_prices_the_configured_judge_output_budget(self):
        import litellm as litellm_module
        from litellm.integrations.shadow_eval_logger import JUDGE_MAX_OUTPUT_TOKENS
        from litellm.proxy.management_endpoints.auto_router_endpoints import (
            _JUDGE_PROMPT_TOKENS_ESTIMATE,
            _estimate_judge_cost_per_call,
        )

        prompt_cost, completion_cost = litellm_module.cost_per_token(
            model=self.JUDGE_MODEL,
            prompt_tokens=_JUDGE_PROMPT_TOKENS_ESTIMATE,
            completion_tokens=JUDGE_MAX_OUTPUT_TOKENS,
        )
        assert completion_cost > 0
        assert _estimate_judge_cost_per_call(None, self.JUDGE_MODEL) == prompt_cost + completion_cost

    def test_estimate_is_not_still_pinned_to_the_old_200_token_budget(self):
        import litellm as litellm_module
        from litellm.proxy.management_endpoints.auto_router_endpoints import (
            _JUDGE_PROMPT_TOKENS_ESTIMATE,
            _estimate_judge_cost_per_call,
        )

        stale = sum(
            litellm_module.cost_per_token(
                model=self.JUDGE_MODEL, prompt_tokens=_JUDGE_PROMPT_TOKENS_ESTIMATE, completion_tokens=200
            )
        )
        assert _estimate_judge_cost_per_call(None, self.JUDGE_MODEL) > stale


class TestJudgeModelValidation:
    """A judge the dispatch path cannot resolve fails every sampled turn silently, so
    start must reject it up front instead of accepting a job that only ever bills
    shadow calls."""

    @staticmethod
    def _router() -> MagicMock:
        router = MagicMock()
        router.auto_routers = {}
        router.complexity_routers = {"claude-auto": [MagicMock()]}
        router.adaptive_routers = {}
        router.quality_routers = {}
        router.model_group_alias = {}
        router.get_model_list = MagicMock(return_value=None)
        return router

    def test_unresolvable_judge_model_is_a_400(self):
        from litellm.proxy.management_endpoints.auto_router_endpoints import _validate_judge_model

        with pytest.raises(HTTPException) as exc:
            _validate_judge_model(self._router(), "opus")

        assert exc.value.status_code == 400
        assert "opus" in str(exc.value.detail)

    def test_configured_deployment_name_is_accepted(self):
        from litellm.proxy.management_endpoints.auto_router_endpoints import _validate_judge_model

        router = self._router()
        router.get_model_list = MagicMock(return_value=[{"litellm_params": {"model": "openai/gpt-4o"}}])
        _validate_judge_model(router, "opus")

    def test_provider_qualified_public_name_is_accepted_without_a_router(self):
        from litellm.proxy.management_endpoints.auto_router_endpoints import _validate_judge_model

        _validate_judge_model(None, "openai/gpt-4o")

    def test_an_auto_router_is_rejected_as_judge(self):
        from litellm.proxy.management_endpoints.auto_router_endpoints import _validate_judge_model

        with pytest.raises(HTTPException) as exc:
            _validate_judge_model(self._router(), "claude-auto")

        assert exc.value.status_code == 400
        assert "auto-router" in str(exc.value.detail)

    def test_estimate_prices_a_deployment_by_its_underlying_model(self):
        """The deployment name an admin typed is not a pricing key; the estimate must
        price the provider model behind it, not fall back to the flat $0.01."""
        from litellm.proxy.management_endpoints.auto_router_endpoints import _judge_pricing_model

        router = self._router()
        router.get_model_list = MagicMock(return_value=[{"litellm_params": {"model": "anthropic/claude-sonnet-5"}}])

        assert _judge_pricing_model(router, "opus") == "anthropic/claude-sonnet-5"
        assert _judge_pricing_model(None, "openai/gpt-4o") == "openai/gpt-4o"


class TestShadowEvalJobsAreTimeBound:
    """A shadow eval samples ongoing traffic, so without an end date a forgotten job
    would keep billing judge calls indefinitely. Every job gets an ends_at, and the
    upfront estimate must price the requested window, not always one week."""

    @staticmethod
    def _start_request(**overrides: object) -> StartShadowEvalRequest:
        return StartShadowEvalRequest.model_validate(
            {
                "api_key_id": "hashed-key",
                "router_name": "claude-auto",
                "shadow_percentage": 10.0,
                **overrides,
            }
        )

    @staticmethod
    def _proxy_mocks(monkeypatch: pytest.MonkeyPatch, recent_requests: int) -> MagicMock:
        from litellm.proxy import proxy_server

        router = MagicMock()
        router.auto_routers = {}
        router.complexity_routers = {"claude-auto": [MagicMock()]}
        router.adaptive_routers = {}
        router.quality_routers = {}
        prisma = MagicMock()
        prisma.db.query_raw = AsyncMock(return_value=[{"request_count": recent_requests}])
        prisma.db.litellm_shadowevaljob.find_first = AsyncMock(return_value=None)
        created = MagicMock()
        created.id = "job-1"
        prisma.db.litellm_shadowevaljob.create = AsyncMock(return_value=created)
        monkeypatch.setattr(proxy_server, "llm_router", router)
        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        return prisma

    @pytest.mark.asyncio
    async def test_losing_a_concurrent_start_race_is_a_409_not_a_500(self, monkeypatch: pytest.MonkeyPatch):
        """Two admins can pass the advisory read simultaneously; the partial unique
        index rejects the second create, which must read as the same conflict."""
        from prisma.errors import UniqueViolationError

        from litellm.proxy.management_endpoints.auto_router_endpoints import start_shadow_eval

        prisma = self._proxy_mocks(monkeypatch, recent_requests=700)
        prisma.db.litellm_shadowevaljob.create = AsyncMock(
            side_effect=UniqueViolationError(data={"user_facing_error": {"meta": {"target": "api_key_id"}}})
        )

        with pytest.raises(HTTPException) as exc:
            await start_shadow_eval(self._start_request(), ADMIN)

        assert exc.value.status_code == 409

    @pytest.mark.asyncio
    async def test_estimate_reads_the_daily_rollup_not_the_raw_spend_log_table(self, monkeypatch: pytest.MonkeyPatch):
        """LiteLLM_SpendLogs has no api_key index; a per-key count there scans every
        request in the window. The estimate must come from LiteLLM_DailyUserSpend."""
        from litellm.proxy.management_endpoints.auto_router_endpoints import start_shadow_eval

        prisma = self._proxy_mocks(monkeypatch, recent_requests=700)

        response = await start_shadow_eval(self._start_request(), ADMIN)

        assert response.estimated_request_count == 70
        prisma.db.litellm_spendlogs.count.assert_not_called()
        sql = prisma.db.query_raw.call_args.args[0]
        assert 'FROM "LiteLLM_DailyUserSpend"' in sql
        assert "LiteLLM_SpendLogs" not in sql

    def test_duration_defaults_to_a_week_and_rejects_zero_and_over_a_month(self):
        assert self._start_request().duration_days == 7
        with pytest.raises(ValidationError):
            self._start_request(duration_days=0)
        with pytest.raises(ValidationError):
            self._start_request(duration_days=31)

    @pytest.mark.asyncio
    async def test_job_is_created_with_ends_at_duration_days_from_now(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.proxy.management_endpoints.auto_router_endpoints import start_shadow_eval

        prisma = self._proxy_mocks(monkeypatch, recent_requests=700)
        before = datetime.now(timezone.utc)

        await start_shadow_eval(self._start_request(duration_days=3), ADMIN)

        after = datetime.now(timezone.utc)
        ends_at = prisma.db.litellm_shadowevaljob.create.call_args.kwargs["data"]["ends_at"]
        assert before + timedelta(days=3) <= ends_at <= after + timedelta(days=3)

    @pytest.mark.asyncio
    async def test_estimate_scales_with_duration_not_always_a_week(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.proxy.management_endpoints.auto_router_endpoints import start_shadow_eval

        self._proxy_mocks(monkeypatch, recent_requests=7000)
        one_day = await start_shadow_eval(self._start_request(duration_days=1), ADMIN)

        self._proxy_mocks(monkeypatch, recent_requests=7000)
        two_weeks = await start_shadow_eval(self._start_request(duration_days=14), ADMIN)

        assert one_day.estimated_request_count == 100
        assert two_weeks.estimated_request_count == 1400
        assert two_weeks.estimated_cost == pytest.approx(one_day.estimated_cost * 14, rel=0.02)

    def test_response_surfaces_ends_at(self):
        from litellm.proxy.management_endpoints.auto_router_endpoints import _job_to_response

        fields = {
            "id": "job-1",
            "status": "running",
            "router_name": "claude-auto",
            "api_key_id": "hashed-key",
            "team_id": None,
            "shadow_percentage": 10.0,
            "request_count": 0,
            "completed_count": 0,
            "failed_count": 0,
            "cost_estimate": 1.0,
            "cost_actual": 0.0,
            "created_at": datetime(2026, 8, 1, tzinfo=timezone.utc),
            "ends_at": datetime(2026, 8, 8, tzinfo=timezone.utc),
            "completed_at": None,
        }
        response = _job_to_response(type("Row", (), fields)(), None)
        assert response.ends_at == "2026-08-08T00:00:00+00:00"


class TestShadowEvalResultsStratifyByTierAndByCurrentModel:
    """A key's real traffic can mix models. Per-tier rates blend the incumbents, so
    the results also slice by real_model — which of today's models did the router beat."""

    @staticmethod
    def _row(tier: str | None, real_model: str, real_wins: int, shadow_wins: int, ties: int, conf: float):
        from litellm.proxy.management_endpoints.auto_router_endpoints import _VerdictAggRow

        return _VerdictAggRow(
            tier_classification=tier,
            real_model=real_model,
            turn_count=real_wins + shadow_wins + ties,
            real_wins=real_wins,
            shadow_wins=shadow_wins,
            ties=ties,
            avg_confidence=conf,
        )

    def test_same_rollup_produces_both_stratifications(self):
        from litellm.proxy.management_endpoints.auto_router_endpoints import _shadow_eval_results

        rows = [
            self._row("SIMPLE", "gpt-4o", 1, 8, 1, 0.9),
            self._row("SIMPLE", "my-finetune", 6, 2, 2, 0.7),
            self._row("REASONING", "gpt-4o", 3, 5, 2, 0.8),
        ]
        result = _shadow_eval_results(rows)

        assert result is not None
        simple = next(g for g in result.groups if g.tier == "SIMPLE")
        assert simple.turn_count == 20
        assert simple.shadow_win_rate_pct == 50.0

        gpt4o = next(m for m in result.by_current_model if m.current_model == "gpt-4o")
        finetune = next(m for m in result.by_current_model if m.current_model == "my-finetune")
        assert gpt4o.turn_count == 20
        assert gpt4o.shadow_win_rate_pct == 65.0
        assert finetune.turn_count == 10
        assert finetune.shadow_win_rate_pct == 20.0

    def test_tier_confidence_is_turn_weighted_not_a_plain_average(self):
        from litellm.proxy.management_endpoints.auto_router_endpoints import _shadow_eval_results

        rows = [
            self._row("SIMPLE", "gpt-4o", 0, 9, 0, 1.0),
            self._row("SIMPLE", "my-finetune", 1, 0, 0, 0.0),
        ]
        result = _shadow_eval_results(rows)

        assert result is not None
        assert result.groups[0].avg_judge_confidence == 0.9

    def test_single_incumbent_still_reports_one_model_slice(self):
        from litellm.proxy.management_endpoints.auto_router_endpoints import _shadow_eval_results

        result = _shadow_eval_results([self._row("SIMPLE", "gpt-4o", 2, 6, 2, 0.8)])

        assert result is not None
        assert len(result.by_current_model) == 1
        assert result.by_current_model[0].current_model == "gpt-4o"

    def test_overall_rates_are_unchanged_by_the_extra_grouping_column(self):
        from litellm.proxy.management_endpoints.auto_router_endpoints import _shadow_eval_results

        rows = [
            self._row("SIMPLE", "gpt-4o", 1, 8, 1, 0.9),
            self._row("SIMPLE", "my-finetune", 6, 2, 2, 0.7),
        ]
        result = _shadow_eval_results(rows)

        assert result is not None
        assert result.overall_shadow_win_rate_pct == 50.0
        assert result.overall_tie_rate_pct == 15.0


class TestShadowEvalJobLifecycleEndpoints:
    """List shows every job newest-first, get returns one job with its results, and
    stop flips only active jobs while keeping the verdicts already collected."""

    @staticmethod
    def _job_record(job_id: str = "job-1", status: str = "running") -> MagicMock:
        record = MagicMock()
        record.id = job_id
        record.status = status
        record.router_name = "claude-auto"
        record.api_key_id = "hashed-key"
        record.team_id = None
        record.shadow_percentage = 10.0
        record.request_count = 100
        record.completed_count = 9
        record.failed_count = 1
        record.last_error = None
        record.cost_estimate = 3.0
        record.cost_actual = 0.42
        record.created_at = datetime(2026, 8, 1, tzinfo=timezone.utc)
        record.ends_at = None
        record.completed_at = None
        return record

    def _prisma(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        from litellm.proxy import proxy_server

        prisma = MagicMock()
        monkeypatch.setattr(proxy_server, "prisma_client", prisma)
        return prisma

    @pytest.mark.asyncio
    async def test_list_returns_jobs_without_results(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.proxy.management_endpoints.auto_router_endpoints import list_shadow_eval_jobs

        prisma = self._prisma(monkeypatch)
        prisma.db.litellm_shadowevaljob.find_many = AsyncMock(
            return_value=[self._job_record("job-2"), self._job_record("job-1", status="completed")]
        )

        jobs = await list_shadow_eval_jobs(ADMIN)

        assert [j.job_id for j in jobs] == ["job-2", "job-1"]
        assert all(j.results is None for j in jobs)
        assert prisma.db.litellm_shadowevaljob.find_many.call_args.kwargs["order"] == {"created_at": "desc"}

    @pytest.mark.asyncio
    async def test_get_returns_the_job_with_aggregated_results(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.proxy.management_endpoints.auto_router_endpoints import get_shadow_eval_job

        prisma = self._prisma(monkeypatch)
        prisma.db.litellm_shadowevaljob.find_unique = AsyncMock(return_value=self._job_record())
        prisma.db.query_raw = AsyncMock(
            return_value=[
                {
                    "tier_classification": "SIMPLE",
                    "real_model": "gpt-4o",
                    "turn_count": 10,
                    "real_wins": 2,
                    "shadow_wins": 6,
                    "ties": 2,
                    "avg_confidence": 0.8,
                }
            ]
        )

        response = await get_shadow_eval_job("job-1", ADMIN)

        assert response.job_id == "job-1"
        assert response.results is not None
        assert response.results.groups[0].tier == "SIMPLE"
        assert response.results.groups[0].shadow_win_rate_pct == 60.0

    @pytest.mark.asyncio
    async def test_get_unknown_job_is_a_404(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.proxy.management_endpoints.auto_router_endpoints import get_shadow_eval_job

        prisma = self._prisma(monkeypatch)
        prisma.db.litellm_shadowevaljob.find_unique = AsyncMock(return_value=None)

        with pytest.raises(HTTPException) as exc:
            await get_shadow_eval_job("nope", ADMIN)

        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_stop_completes_an_active_job_and_returns_its_verdicts(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.proxy.management_endpoints.auto_router_endpoints import stop_shadow_eval_job

        prisma = self._prisma(monkeypatch)
        prisma.db.litellm_shadowevaljob.find_unique = AsyncMock(return_value=self._job_record())
        stopped = self._job_record(status="completed")
        prisma.db.litellm_shadowevaljob.update = AsyncMock(return_value=stopped)
        prisma.db.query_raw = AsyncMock(return_value=[])

        response = await stop_shadow_eval_job("job-1", ADMIN)

        assert response.status == "completed"
        data = prisma.db.litellm_shadowevaljob.update.call_args.kwargs["data"]
        assert data["status"] == "completed"
        assert data["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_stopping_a_finished_job_is_a_400_not_a_silent_rewrite(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.proxy.management_endpoints.auto_router_endpoints import stop_shadow_eval_job

        prisma = self._prisma(monkeypatch)
        prisma.db.litellm_shadowevaljob.find_unique = AsyncMock(return_value=self._job_record(status="completed"))

        with pytest.raises(HTTPException) as exc:
            await stop_shadow_eval_job("job-1", ADMIN)

        assert exc.value.status_code == 400
        prisma.db.litellm_shadowevaljob.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_admin_cannot_stop_and_viewer_can_list(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.proxy.management_endpoints.auto_router_endpoints import (
            list_shadow_eval_jobs,
            stop_shadow_eval_job,
        )

        prisma = self._prisma(monkeypatch)
        prisma.db.litellm_shadowevaljob.find_many = AsyncMock(return_value=[])
        viewer = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, api_key="sk-view", user_id="viewer")

        assert await list_shadow_eval_jobs(viewer) == []
        with pytest.raises(HTTPException) as exc:
            await stop_shadow_eval_job("job-1", viewer)
        assert exc.value.status_code == 403
