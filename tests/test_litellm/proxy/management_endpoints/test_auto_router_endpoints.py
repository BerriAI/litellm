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
from litellm.types.management_endpoints.auto_router_endpoints import (
    AutoRouterRoutingTestRequest,
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


class TestAutoRouterQualitySignals:
    """The endpoint's cohort split and its refusal to report a baseline it cannot stand behind.

    Escalation/abandonment arithmetic itself is covered in
    tests/test_litellm/proxy/db/test_autorouter_quality_signals.py; these tests cover the
    wiring around it -- which rows become which cohort, and when the baseline is withheld.
    """

    CHEAP = "openai/gpt-4o-mini"
    PRICEY = "openai/gpt-4o"

    @staticmethod
    def _row(
        session_id: str,
        model: str,
        started_at: float,
        *,
        router_name: str | None = "live-auto",
        client_disconnected: bool = False,
        session_turn_count: int = 2,
        api_key: str = "key-1",
    ) -> dict:
        return {
            "session_id": session_id,
            "model": model,
            "started_at": started_at,
            "router_name": router_name,
            "client_disconnected": client_disconnected,
            "session_turn_count": session_turn_count,
            "api_key": api_key,
        }

    @staticmethod
    def _pricing_router() -> Router:
        return Router(
            model_list=[
                {"model_name": name, "litellm_params": {"model": name, "api_key": "fake-key"}}
                for name in (TestAutoRouterQualitySignals.CHEAP, TestAutoRouterQualitySignals.PRICEY)
            ]
        )

    async def _call(self, rows: list[dict], monkeypatch: pytest.MonkeyPatch, **kwargs: object):
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.auto_router_endpoints import (
            get_auto_router_quality_signals,
        )

        class _DB:
            async def query_raw(self, sql: str, *params: object):
                return rows

        monkeypatch.setattr(proxy_server, "prisma_client", type("P", (), {"db": _DB()})())
        monkeypatch.setattr(proxy_server, "llm_router", self._pricing_router())
        return await get_auto_router_quality_signals(
            user_api_key_dict=ADMIN,
            start_date="2026-08-01",
            end_date="2026-08-02",
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_non_admin_roles_cannot_read_quality_signals(self):
        from litellm.proxy.management_endpoints.auto_router_endpoints import (
            get_auto_router_quality_signals,
        )

        with pytest.raises(HTTPException) as err:
            await get_auto_router_quality_signals(
                user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, api_key="sk-x"),
                start_date="2026-08-01",
                end_date="2026-08-02",
            )
        assert err.value.status_code == 403

    @pytest.mark.asyncio
    async def test_a_reversed_window_is_rejected(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.auto_router_endpoints import (
            get_auto_router_quality_signals,
        )

        monkeypatch.setattr(proxy_server, "prisma_client", object())
        with pytest.raises(HTTPException) as err:
            await get_auto_router_quality_signals(
                user_api_key_dict=ADMIN, start_date="2026-08-05", end_date="2026-08-01"
            )
        assert err.value.status_code == 400

    @pytest.mark.asyncio
    async def test_routed_escalation_is_measured_from_routed_rows(self, monkeypatch: pytest.MonkeyPatch):
        rows = [
            self._row("s1", self.CHEAP, 1.0),
            self._row("s1", self.PRICEY, 2.0),
        ]
        response = await self._call(rows, monkeypatch)
        assert response.totals.routed.sessions == 1
        assert response.totals.routed.escalation_rate_pct == 100.0

    @pytest.mark.asyncio
    async def test_rows_without_a_routing_decision_are_the_baseline_not_the_routed_cohort(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # Same key, same models; only the presence of a router name separates the cohorts.
        # The routed session escalates and the direct one does not, so a cohort mix-up
        # cannot produce these two numbers.
        rows = [
            self._row("routed-1", self.CHEAP, 1.0),
            self._row("routed-1", self.PRICEY, 2.0),
        ]
        rows += [self._row(f"direct-{i}", self.CHEAP, 1.0, router_name=None) for i in range(30)]
        rows += [self._row(f"direct-{i}", self.CHEAP, 2.0, router_name=None) for i in range(30)]
        response = await self._call(rows, monkeypatch)
        assert response.totals.routed.escalation_rate_pct == 100.0
        assert response.totals.baseline is not None
        assert response.totals.baseline.sessions == 30
        assert response.totals.baseline.escalation_rate_pct == 0.0

    @pytest.mark.asyncio
    async def test_baseline_withheld_when_sessions_are_mostly_one_request_fallback_ids(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # session_turn_count == 1 marks a session id the spend writer minted rather than one
        # the caller supplied. A cohort made of those is not sessions, and reporting a rate
        # over it would describe the fallback rather than the traffic.
        rows = [self._row("routed-1", self.CHEAP, 1.0), self._row("routed-1", self.PRICEY, 2.0)]
        rows += [self._row(f"direct-{i}", self.CHEAP, 1.0, router_name=None, session_turn_count=1) for i in range(50)]
        response = await self._call(rows, monkeypatch)
        assert response.totals.baseline is None
        assert response.totals.baseline_unavailable_reason == "no_session_ids"

    @pytest.mark.asyncio
    async def test_baseline_withheld_when_there_is_too_little_comparable_traffic(self, monkeypatch: pytest.MonkeyPatch):
        rows = [self._row("routed-1", self.CHEAP, 1.0), self._row("routed-1", self.PRICEY, 2.0)]
        rows += [
            self._row("direct-1", self.CHEAP, 1.0, router_name=None),
            self._row("direct-1", self.CHEAP, 2.0, router_name=None),
        ]
        response = await self._call(rows, monkeypatch)
        assert response.totals.baseline is None
        assert response.totals.baseline_unavailable_reason == "insufficient_sessions"

    @pytest.mark.asyncio
    async def test_baseline_only_draws_on_keys_that_used_the_router(self, monkeypatch: pytest.MonkeyPatch):
        # A key that never touched the router says nothing about the router, and its traffic
        # must not be folded into the comparison.
        rows = [self._row("routed-1", self.CHEAP, 1.0), self._row("routed-1", self.PRICEY, 2.0)]
        rows += [self._row(f"other-{i}", self.CHEAP, 1.0, router_name=None, api_key="unrelated-key") for i in range(50)]
        response = await self._call(rows, monkeypatch)
        assert response.totals.baseline is None

    @pytest.mark.asyncio
    async def test_two_keys_reusing_the_same_session_id_are_not_merged(self, monkeypatch: pytest.MonkeyPatch):
        # key-1 and key-2 both happen to send session_id="shared". key-1's own two turns
        # under it escalate; key-2's own two turns under it do not (key-2's access to the
        # pricey model is established by a separate "other" session so it is still eligible
        # to escalate). Merging the two keys' rows by session_id alone would splice all four
        # turns into one session and read key-1's escalation and key-2's non-escalation as a
        # single, order-dependent sequence instead of two independent sessions.
        rows = [
            self._row("shared", self.CHEAP, 1.0, api_key="key-1"),
            self._row("shared", self.PRICEY, 2.0, api_key="key-1"),
            self._row("shared", self.CHEAP, 1.0, api_key="key-2"),
            self._row("shared", self.CHEAP, 2.0, api_key="key-2"),
            self._row("other", self.PRICEY, 1.0, api_key="key-2"),
        ]
        response = await self._call(rows, monkeypatch)
        assert response.totals.routed.sessions == 2
        assert response.totals.routed.escalation_rate_pct == 50.0

    @pytest.mark.asyncio
    async def test_reachability_is_not_pooled_across_keys_in_the_same_cohort(self, monkeypatch: pytest.MonkeyPatch):
        # key-1 only ever calls the cheap model; key-2 reaches both. Pooling reachability
        # across the cohort would credit key-1's session with key-2's ceiling and count it
        # as eligible to escalate when key-1 never had a pricier model to reach for.
        rows = [
            self._row("k1-s1", self.CHEAP, 1.0, api_key="key-1"),
            self._row("k1-s1", self.CHEAP, 2.0, api_key="key-1"),
            self._row("k2-s1", self.CHEAP, 1.0, api_key="key-2"),
            self._row("k2-s1", self.PRICEY, 2.0, api_key="key-2"),
        ]
        response = await self._call(rows, monkeypatch)
        assert response.totals.routed.sessions == 1
        assert response.totals.routed.escalation_rate_pct == 100.0

    @pytest.mark.asyncio
    async def test_groups_are_reported_per_router(self, monkeypatch: pytest.MonkeyPatch):
        rows = [
            self._row("a1", self.CHEAP, 1.0, router_name="router-a"),
            self._row("a1", self.PRICEY, 2.0, router_name="router-a"),
            self._row("b1", self.CHEAP, 1.0, router_name="router-b"),
            self._row("b1", self.CHEAP, 2.0, router_name="router-b"),
        ]
        response = await self._call(rows, monkeypatch)
        by_name = {group.router_name: group for group in response.groups}
        assert set(by_name) == {"router-a", "router-b"}
        assert by_name["router-a"].routed.escalation_rate_pct == 100.0
        assert by_name["router-b"].routed.escalation_rate_pct == 0.0

    @pytest.mark.asyncio
    async def test_abandonment_reads_the_disconnect_flag_not_the_status(self, monkeypatch: pytest.MonkeyPatch):
        # s2 gives the key a pricier model in the window, so s1 is eligible; without it s1
        # sits at its own ceiling and is correctly excluded, measuring nothing.
        rows = [
            self._row("s1", self.CHEAP, 1.0, client_disconnected=True),
            self._row("s1", self.CHEAP, 2.0, client_disconnected=False),
            self._row("s2", self.PRICEY, 1.0),
            self._row("s2", self.PRICEY, 2.0),
        ]
        response = await self._call(rows, monkeypatch)
        assert response.totals.routed.sessions == 1
        assert response.totals.routed.abandonment_rate_pct == 50.0
