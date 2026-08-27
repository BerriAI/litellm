"""
Unit tests for auto router management endpoints
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import pytest
from fastapi import HTTPException
from pydantic import ValidationError


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
    AutoRouterBenchmarksResponse,
    AutoRouterRoutingTestRequest,
)

ADMIN = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-test", user_id="admin")


def _deployment(model_name: str, model: str, *, db_model: bool) -> dict[str, object]:
    """One entry as `Router.model_list` holds it, for either origin."""
    return {
        "model_name": model_name,
        "litellm_params": {"model": model},
        "model_info": {"id": f"{model_name}-{int(db_model)}", "db_model": db_model},
    }


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


def test_classifier_plugin_is_not_settable_over_http():
    """classifier_plugin holds a live runtime object, closed off like `plugins`; a plugin-mode
    config is therefore unrepresentable in a request body."""
    with pytest.raises(ValidationError):
        _request("what is 2+2", classifier_type="custom", classifier_plugin="my_module.instance")


class TestAutoRouterBenchmarks:
    from litellm.proxy.management_endpoints.auto_router_endpoints import _SessionAggRow

    @pytest.fixture(autouse=True)
    def _pin_the_router_global(self, monkeypatch: pytest.MonkeyPatch):
        """Every test here reads proxy_server.llm_router, so no test may inherit a sibling's."""
        from litellm.proxy import proxy_server

        monkeypatch.setattr(proxy_server, "llm_router", None)

    @staticmethod
    async def _benchmarks(
        monkeypatch: pytest.MonkeyPatch,
        rows: Sequence[Mapping[str, object]],
        model_list: Sequence[object],
    ) -> AutoRouterBenchmarksResponse:
        from litellm.proxy import proxy_server
        from litellm.proxy.management_endpoints.auto_router_endpoints import get_auto_router_benchmarks

        class _DB:
            async def query_raw(self, sql: str, *params: object):
                return rows

        monkeypatch.setattr(proxy_server, "prisma_client", type("P", (), {"db": _DB()})())
        monkeypatch.setattr(proxy_server, "llm_router", type("R", (), {"model_list": model_list})())
        return await get_auto_router_benchmarks(
            user_api_key_dict=ADMIN,
            start_date="2026-07-01",
            end_date="2026-08-01",
        )

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

    @pytest.mark.asyncio
    async def test_the_picker_lists_configured_routers_before_they_have_traffic(self, monkeypatch: pytest.MonkeyPatch):
        """A router must be selectable the moment it exists, from either origin.

        `live-auto` is the only router the rollup knows about, so before this it was the only
        thing the dropdown could offer. Both a config.yaml router and a DB-created one now
        arrive zeroed, and neither moves the totals or duplicates the router that has traffic.
        """
        response = await self._benchmarks(
            monkeypatch,
            rows=[self.ROW.model_dump()],
            model_list=[
                _deployment("live-auto", "auto_router/complexity_router", db_model=False),
                _deployment("idle-from-config", "auto_router/complexity_router", db_model=False),
                _deployment("idle-from-db", "auto_router/complexity_router", db_model=True),
            ],
        )

        by_name = {group.router_name: group for group in response.groups}
        assert sorted(by_name) == ["idle-from-config", "idle-from-db", "live-auto"]
        assert len(response.groups) == 3
        assert response.routers_in_scope == 3
        assert by_name["live-auto"].spend == 10.0
        assert response.totals.spend == 10.0
        assert response.totals.sessions == 4
        for name in ("idle-from-config", "idle-from-db"):
            idle = by_name[name]
            assert idle.router_type == "complexity"
            assert (idle.sessions, idle.turns, idle.spend, idle.saved_spend, idle.baseline_spend) == (
                0,
                0,
                0.0,
                0.0,
                0.0,
            )
            assert (idle.saved_pct, idle.saved_per_session, idle.avg_turns_per_session) == (0.0, 0.0, 0.0)
            assert (idle.cache.hit_rate_pct, idle.cache.coverage_pct) == (0.0, 0.0)
            assert idle.cache.same_model.turns == idle.cache.return_to_tier.hits == 0
            assert idle.tier_turns == {}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model, listed_as",
        [
            ("auto_router/complexity_router", "complexity"),
            ("auto_router/adaptive_router", "adaptive"),
            ("auto_router/quality_router", "quality"),
            ("auto_router/my-semantic-router", None),
            ("openai/gpt-5", None),
        ],
    )
    async def test_only_kinds_whose_routing_the_rollup_records_are_listed(
        self, model: str, listed_as: str | None, monkeypatch: pytest.MonkeyPatch
    ):
        """A semantic auto-router records no routing decision, so it can never own a session
        row; listing it would show $0 forever even while it serves traffic."""
        response = await self._benchmarks(
            monkeypatch, rows=[], model_list=[_deployment("candidate", model, db_model=True)]
        )

        assert [group.router_type for group in response.groups] == ([listed_as] if listed_as else [])

    @pytest.mark.asyncio
    async def test_a_malformed_deployment_is_skipped_rather_than_failing_the_dashboard(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        response = await self._benchmarks(
            monkeypatch,
            rows=[self.ROW.model_dump()],
            model_list=[
                "not-a-mapping",
                {},
                {"model_name": "no-params"},
                {"model_name": "", "litellm_params": {"model": "auto_router/complexity_router"}},
                {"model_name": 7, "litellm_params": {"model": "auto_router/complexity_router"}},
                {"model_name": "no-model", "litellm_params": {}},
                {"model_name": "unreadable-model", "litellm_params": {"model": None}},
            ],
        )

        assert [group.router_name for group in response.groups] == ["live-auto"]

    @pytest.mark.asyncio
    async def test_two_deployments_of_one_router_are_listed_once(self, monkeypatch: pytest.MonkeyPatch):
        """Tagged variants share a model_name, and the picker selects by name and type."""
        response = await self._benchmarks(
            monkeypatch,
            rows=[],
            model_list=[
                _deployment("tagged", "auto_router/complexity_router", db_model=True),
                _deployment("tagged", "auto_router/complexity_router", db_model=True),
            ],
        )

        assert [group.router_name for group in response.groups] == ["tagged"]

    def test_the_listed_kinds_match_the_router_types_traffic_can_record(self):
        """The one reason semantic is excluded, pinned against both declarations: a kind the
        rollup can record must be listable, and a kind it cannot must not be."""
        from typing import get_args, get_type_hints

        from litellm.router_utils.auto_router_model_naming import StrategyRouterKind
        from litellm.types.utils import StandardLoggingRoutingDecision

        recorded = set(get_args(get_type_hints(StandardLoggingRoutingDecision)["router_type"]))
        assert set(get_args(StrategyRouterKind)) - {"semantic"} == recorded


# ---------------------------------------------------------------------------
# Shadow eval endpoints
# ---------------------------------------------------------------------------

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock


from litellm.proxy.management_endpoints.auto_router_endpoints import (
    get_shadow_eval_job,
    list_shadow_eval_jobs,
    start_shadow_eval,
    stop_shadow_eval_job,
)
from litellm.types.management_endpoints.auto_router_endpoints import SHADOW_EVAL_TURN_VALVE, StartShadowEvalRequest

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


def _leg_record(**overrides: object) -> MagicMock:
    """Spec'd like a real prisma row: only the table's columns exist as attributes. One
    row is one key's leg of a job; legs sharing group_id are one job."""
    defaults = {
        "id": "leg-1",
        "group_id": "job-1",
        "api_key_id": "key-hash",
        "router_name": "my-router",
        "direction": "forward",
        "baseline_model": None,
        "judge_model": "anthropic/claude-sonnet-5",
        "shadow_percentage": 10.0,
        "max_turns": 200,
        "max_budget": None,
        "created_at": datetime(2026, 8, 11, tzinfo=timezone.utc),
        "ends_at": datetime.now(timezone.utc) + timedelta(days=7),
        "stopped_at": None,
        "stopped_by": None,
    }
    fields = {**defaults, **overrides}
    record = MagicMock(spec=list(fields))
    for key, value in fields.items():
        setattr(record, key, value)
    return record


def _key_record(
    token: str = "key-hash", key_alias: str | None = "prod-alpha", key_name: str | None = "sk-...lpha"
) -> MagicMock:
    record = MagicMock(spec=["token", "key_alias", "key_name"])
    record.token = token
    record.key_alias = key_alias
    record.key_name = key_name
    return record


def _shadow_prisma(legs=(), agg_rows=None, by_leg_rows=None, known_keys=("key-hash", "key-hash-2")) -> MagicMock:
    """The job-table fake honours the filters it is handed, so a read that forgets
    stopped_at sees rows the partial index would have released, one that forgets
    direction sees the opposite-direction legs a key may hold at the same time, and a
    group read that matched on a leg id would come back empty."""
    prisma = MagicMock()
    prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[_key_record(token) for token in known_keys])

    async def execute_raw(sql: str, *params: object):
        if "SET stopped_by" in sql:
            group = [row for row in stored if row.group_id == params[0]]
            counts = {row["job_id"]: row["attempt_count"] for row in prisma.attempt_rows}
            spends = {row["job_id"]: row["spend"] for row in prisma.attempt_rows}
            sampling = any(
                row.stopped_at is None
                and counts.get(row.id, 0) < row.max_turns
                and (row.max_budget is None or spends.get(row.id, 0.0) < row.max_budget)
                for row in group
            )
            window_open = bool(group) and group[0].ends_at > datetime.now(timezone.utc)
            claimable = [row for row in group if row.stopped_by is None]
            if not (claimable and sampling and window_open):
                return 0
            for row in claimable:
                row.stopped_by = params[1]
                if row.stopped_at is None:
                    row.stopped_at = datetime.fromisoformat(str(params[2])).replace(tzinfo=timezone.utc)
            return len(claimable)
        return 0

    prisma.db.execute_raw = AsyncMock(side_effect=execute_raw)
    stored = legs if isinstance(legs, list) else list(legs)

    async def find_many_legs(where=None, **_: object):
        current = list(stored)
        w = dict(where or {})
        if "api_key_id" in w:
            wanted = w["api_key_id"]["in"] if isinstance(w["api_key_id"], dict) else [w["api_key_id"]]
            current = [row for row in current if row.api_key_id in wanted]
        if "direction" in w:
            current = [row for row in current if row.direction == w["direction"]]
        if "stopped_at" in w:
            current = [row for row in current if row.stopped_at is w["stopped_at"]]
        if "group_id" in w:
            wanted = w["group_id"]["in"] if isinstance(w["group_id"], dict) else [w["group_id"]]
            current = [row for row in current if row.group_id in wanted]
        return current

    def newest_groups(rows, limit):
        latest: dict = {}
        for row in rows:
            if row.group_id not in latest or row.created_at > latest[row.group_id]:
                latest[row.group_id] = row.created_at
        ordered = sorted(latest, key=lambda group_id: latest[group_id], reverse=True)
        return ordered[: int(limit)]

    def leg_dict(row):
        fields = (
            "id",
            "group_id",
            "api_key_id",
            "router_name",
            "direction",
            "baseline_model",
            "judge_model",
            "shadow_percentage",
            "max_turns",
            "max_budget",
            "created_at",
            "ends_at",
            "stopped_at",
            "stopped_by",
        )
        return {field: getattr(row, field) for field in fields}

    prisma.db.litellm_shadowevaljob.find_many = AsyncMock(side_effect=find_many_legs)
    prisma.db.litellm_shadowevaljob.create_many = AsyncMock(return_value=1)
    prisma.db.litellm_shadowevaljob.update_many = AsyncMock(return_value=1)
    prisma.db.litellm_shadowevalattempt.find_first = AsyncMock(return_value=None)
    prisma.attempt_rows = []

    async def query_raw(sql: str, *params: object):
        if "AS attempt_count" in sql:
            return prisma.attempt_rows
        if "GROUP BY group_id" in sql:
            scoped = [row for row in stored if "api_key_id = $2" not in sql or row.api_key_id == params[1]]
            keep = set(newest_groups(scoped, params[0]))
            return [leg_dict(row) for row in stored if row.group_id in keep]
        if "FILTER (WHERE outcome != 'error')::int AS judged_count" in sql:
            return [{"judged_count": 10, "error_count": 2, "judge_spend": 0.031}]
        if "SELECT job_id AS grp" in sql:
            return by_leg_rows if by_leg_rows is not None else []
        return agg_rows if agg_rows is not None else []

    prisma.db.query_raw = AsyncMock(side_effect=query_raw)
    return prisma


def _start_request(**overrides: object) -> StartShadowEvalRequest:
    payload = {
        "api_key_ids": ("key-hash",),
        "router_name": "my-router",
        "shadow_percentage": 10.0,
        "judge_model": "anthropic/claude-sonnet-5",
        "duration_days": 7,
        "max_budget": 5.0,
    }
    payload.update(overrides)
    return StartShadowEvalRequest.model_validate(payload)


@pytest.mark.asyncio
async def test_start_shadow_eval_writes_one_leg_per_key_in_one_statement(monkeypatch: pytest.MonkeyPatch):
    """N keys become N sibling rows sharing group_id and identical config, written by a
    single create_many so a unique-index loser rolls back the whole claim, and expiry or
    budget exhaustion frees every requested key's slot first."""
    import litellm.proxy.proxy_server as proxy_server

    prisma = _shadow_prisma()
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(proxy_server, "llm_router", _shadow_router())

    response = await start_shadow_eval(_start_request(api_key_ids=("key-hash", "key-hash-2")), ADMIN)

    sweep_sql, sweep_keys = prisma.db.execute_raw.call_args.args
    assert "stopped_at IS NULL" in sweep_sql
    assert "j.ends_at <= (NOW() AT TIME ZONE 'utc')" in sweep_sql
    assert "SET stopped_at = (NOW() AT TIME ZONE 'utc')" in sweep_sql
    assert ">= j.max_turns" in sweep_sql
    assert "j.max_budget IS NOT NULL" in sweep_sql
    assert ">= j.max_budget" in sweep_sql
    assert "SUM(a.judge_cost + a.shadow_cost)" in sweep_sql
    assert "j.api_key_id = ANY($1::text[])" in sweep_sql
    assert sweep_keys == ["key-hash", "key-hash-2"]
    prisma.db.litellm_shadowevaljob.create_many.assert_awaited_once()
    rows = prisma.db.litellm_shadowevaljob.create_many.call_args.kwargs["data"]
    assert [row["api_key_id"] for row in rows] == ["key-hash", "key-hash-2"]
    assert len({frozenset((k, v) for k, v in row.items() if k != "api_key_id") for row in rows}) == 1
    assert len({row["group_id"] for row in rows}) == 1
    assert all(row["max_turns"] == SHADOW_EVAL_TURN_VALVE and row["created_by"] == "admin" for row in rows)
    assert all(row["max_budget"] == 5.0 for row in rows)
    assert all("status" not in row and "id" not in row for row in rows)
    assert response.job_id == rows[0]["group_id"]
    assert response.status == "running"
    assert response.judged_count is None
    assert [(key.api_key_id, key.max_budget, key.key_alias) for key in response.keys] == [
        ("key-hash", 5.0, "prod-alpha"),
        ("key-hash-2", 5.0, "prod-alpha"),
    ]
    assert all(key.max_turns == SHADOW_EVAL_TURN_VALVE for key in response.keys)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "caller,request_overrides,claimed,expected_status",
    [
        (NON_ADMIN, {}, (), 403),
        (VIEWER, {}, (), 403),
        (ADMIN, {"router_name": "not-a-router"}, (), 400),
        (ADMIN, {"judge_model": "not/a real model!"}, (), 400),
        (ADMIN, {"judge_model": "my-router"}, (), 400),
        (ADMIN, {}, ("key-hash",), 409),
        (ADMIN, {"api_key_ids": ("key-hash", "key-hash-2")}, ("key-hash-2",), 409),
        (ADMIN, {"direction": "reverse", "baseline_model": "my-router"}, (), 400),
        (ADMIN, {"direction": "reverse", "baseline_model": "not/a real model!"}, (), 400),
        (ADMIN, {"direction": "reverse", "baseline_model": "openai/gpt-4o", "router_name": "not-a-router"}, (), 400),
    ],
    ids=[
        "non-admin",
        "view-only",
        "unknown-router",
        "unresolvable-judge",
        "router-as-judge",
        "already-active",
        "one-of-several-keys-already-active",
        "router-as-baseline",
        "unresolvable-baseline",
        "reverse-still-needs-an-auto-router",
    ],
)
async def test_start_shadow_eval_rejections(
    monkeypatch: pytest.MonkeyPatch, caller, request_overrides, claimed, expected_status
):
    import litellm.proxy.proxy_server as proxy_server

    prisma = _shadow_prisma(legs=[_leg_record(id=f"leg-{key}", group_id="job-7", api_key_id=key) for key in claimed])
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(proxy_server, "llm_router", _shadow_router())

    with pytest.raises(HTTPException) as exc:
        await start_shadow_eval(_start_request(**request_overrides), caller)
    assert exc.value.status_code == expected_status
    prisma.db.litellm_shadowevaljob.create_many.assert_not_called()


@pytest.mark.asyncio
async def test_start_shadow_eval_names_the_busy_key_and_its_job(monkeypatch: pytest.MonkeyPatch):
    """A key busy elsewhere blocks the whole start rather than being silently dropped from
    it, and the 409 names which key and which job so the caller can stop or drop it."""
    import litellm.proxy.proxy_server as proxy_server

    prisma = _shadow_prisma(legs=[_leg_record(id="leg-b", group_id="job-7", api_key_id="key-hash-2")])
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(proxy_server, "llm_router", _shadow_router())

    with pytest.raises(HTTPException) as exc:
        await start_shadow_eval(_start_request(api_key_ids=("key-hash", "key-hash-2")), ADMIN)
    assert exc.value.status_code == 409
    assert "key-hash-2 (job job-7)" in exc.value.detail


@pytest.mark.asyncio
async def test_start_shadow_eval_reuses_a_key_whose_previous_job_already_stopped(monkeypatch: pytest.MonkeyPatch):
    """The claim is held by unstopped legs only, matching the partial unique index. A read
    that forgets that would strand every key that has ever finished a job."""
    import litellm.proxy.proxy_server as proxy_server

    prisma = _shadow_prisma(legs=[_leg_record(group_id="job-7", stopped_at=datetime.now(timezone.utc))])
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(proxy_server, "llm_router", _shadow_router())

    job = await start_shadow_eval(_start_request(), ADMIN)

    assert job.status == "running"
    prisma.db.litellm_shadowevaljob.create_many.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_shadow_eval_reverse_records_its_arms_and_holds_its_own_slot(monkeypatch: pytest.MonkeyPatch):
    """The two directions ask opposite questions of the same key, so a forward job holding
    the slot must not block a reverse one. The second reverse start still 409s."""
    import litellm.proxy.proxy_server as proxy_server

    legs = [_leg_record(group_id="job-fwd")]
    prisma = _shadow_prisma(legs=legs)
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(proxy_server, "llm_router", _shadow_router())

    reverse = _start_request(direction="reverse", baseline_model="openai/gpt-4o")
    response = await start_shadow_eval(reverse, ADMIN)

    assert (response.direction, response.baseline_model) == ("reverse", "openai/gpt-4o")
    rows = prisma.db.litellm_shadowevaljob.create_many.call_args.kwargs["data"]
    assert rows[0]["direction"] == "reverse"
    assert rows[0]["baseline_model"] == "openai/gpt-4o"

    legs.append(_leg_record(id="leg-2", group_id="job-rev", direction="reverse"))
    with pytest.raises(HTTPException) as exc:
        await start_shadow_eval(reverse, ADMIN)
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_start_shadow_eval_forward_leaves_the_baseline_column_empty(monkeypatch: pytest.MonkeyPatch):
    import litellm.proxy.proxy_server as proxy_server

    prisma = _shadow_prisma()
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(proxy_server, "llm_router", _shadow_router())

    await start_shadow_eval(_start_request(), ADMIN)

    rows = prisma.db.litellm_shadowevaljob.create_many.call_args.kwargs["data"]
    assert rows[0]["direction"] == "forward"
    assert rows[0]["baseline_model"] is None


@pytest.mark.asyncio
async def test_start_shadow_eval_rejects_keys_this_proxy_does_not_know(monkeypatch: pytest.MonkeyPatch):
    """A typo'd api_key_id would otherwise create a leg no traffic can ever match. Every
    unknown key is named at once, so a caller passing several fixes them in one round."""
    import litellm.proxy.proxy_server as proxy_server

    prisma = _shadow_prisma(known_keys=("key-hash",))
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(proxy_server, "llm_router", _shadow_router())

    with pytest.raises(HTTPException) as exc:
        await start_shadow_eval(_start_request(api_key_ids=("key-hash", "typo-a", "typo-b")), ADMIN)
    assert exc.value.status_code == 400
    assert "typo-a, typo-b" in exc.value.detail
    assert "key-hash," not in exc.value.detail
    prisma.db.litellm_shadowevaljob.create_many.assert_not_called()


def test_start_shadow_eval_request_dedupes_and_bounds_the_key_set():
    """A key named twice would collide with itself on the one-active-per-key index, a job
    scoping no key samples nothing, and the key-count cap bounds every downstream read."""
    assert _start_request(api_key_ids=("a", "b", "a")).api_key_ids == ("a", "b")
    assert len(_start_request(api_key_ids=tuple(f"k{i}" for i in range(100))).api_key_ids) == 100
    with pytest.raises(ValidationError):
        _start_request(api_key_ids=())
    with pytest.raises(ValidationError):
        _start_request(api_key_ids=tuple(f"k{i}" for i in range(101)))


@pytest.mark.asyncio
async def test_start_shadow_eval_concurrent_unique_violation_is_a_409(monkeypatch: pytest.MonkeyPatch):
    import litellm.proxy.proxy_server as proxy_server
    from prisma.errors import UniqueViolationError

    prisma = _shadow_prisma()
    prisma.db.litellm_shadowevaljob.create_many = AsyncMock(
        side_effect=UniqueViolationError(MagicMock(message="unique constraint"))
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(proxy_server, "llm_router", _shadow_router())

    with pytest.raises(HTTPException) as exc:
        await start_shadow_eval(_start_request(), ADMIN)
    assert exc.value.status_code == 409


@pytest.mark.parametrize(
    "overrides",
    [
        {"direction": "reverse"},
        {"baseline_model": "openai/gpt-4o"},
        {"direction": "sideways", "baseline_model": "openai/gpt-4o"},
    ],
    ids=["reverse-without-baseline", "forward-with-baseline", "unknown-direction"],
)
def test_start_request_pins_baseline_model_to_reverse(overrides):
    """A forward job has no second arm to name and a reverse job cannot run without one,
    so neither shape reaches the endpoint to be half-validated there."""
    with pytest.raises(ValidationError):
        _start_request(**overrides)


@pytest.mark.asyncio
async def test_get_shadow_eval_job_pools_counts_and_slices_results_per_key(monkeypatch: pytest.MonkeyPatch):
    """One read answers for every leg: totals and stratifications aggregate over the
    group's leg ids, and the by-key slice maps each leg id back to its key hash."""
    import litellm.proxy.proxy_server as proxy_server

    tier_rows = [
        {"grp": "SIMPLE", "turn_count": 8, "real_wins": 2, "shadow_wins": 4, "ties": 2, "avg_confidence": 0.8},
        {"grp": "REASONING", "turn_count": 2, "real_wins": 2, "shadow_wins": 0, "ties": 0, "avg_confidence": 0.9},
    ]
    leg_rows = [
        {"grp": "leg-1", "turn_count": 6, "real_wins": 1, "shadow_wins": 4, "ties": 1, "avg_confidence": 0.7},
        {"grp": "leg-2", "turn_count": 4, "real_wins": 3, "shadow_wins": 0, "ties": 1, "avg_confidence": 0.6},
    ]
    prisma = _shadow_prisma(
        legs=[_leg_record(), _leg_record(id="leg-2", api_key_id="key-hash-2", max_turns=50)],
        agg_rows=tier_rows,
        by_leg_rows=leg_rows,
    )
    prisma.db.litellm_shadowevalattempt.find_first = AsyncMock(return_value=MagicMock(error="judge call failed: boom"))
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
    assert [(s.group, s.turn_count) for s in response.results.by_key] == [("key-hash", 6), ("key-hash-2", 4)]
    assert response.results.by_key[0].shadow_win_rate_pct == 66.7
    assert [(key.api_key_id, key.max_turns) for key in response.keys] == [("key-hash", 200), ("key-hash-2", 50)]
    totals_args = [call.args for call in prisma.db.query_raw.await_args_list if "judged_count" in call.args[0]]
    assert totals_args == [(totals_args[0][0], ["leg-1", "leg-2"])]
    error_where = prisma.db.litellm_shadowevalattempt.find_first.call_args.kwargs["where"]
    assert error_where == {"job_id": {"in": ["leg-1", "leg-2"]}, "outcome": "error"}


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
async def test_list_shadow_eval_jobs_collapses_legs_into_jobs_newest_first(monkeypatch: pytest.MonkeyPatch):
    """A job over two keys is one list entry with both keys, not two entries, and a job
    whose keys all stopped reads stopped while a half-stopped one still runs."""
    import litellm.proxy.proxy_server as proxy_server

    stamp = datetime.now(timezone.utc)
    prisma = _shadow_prisma(
        legs=[
            _leg_record(created_at=datetime(2026, 8, 13, tzinfo=timezone.utc)),
            _leg_record(
                id="leg-2",
                api_key_id="key-hash-2",
                stopped_at=stamp,
                created_at=datetime(2026, 8, 13, tzinfo=timezone.utc),
            ),
            _leg_record(
                id="leg-3",
                group_id="job-2",
                stopped_at=stamp,
                created_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
            ),
            _leg_record(
                id="leg-4",
                group_id="job-3",
                ends_at=datetime.now(timezone.utc) - timedelta(days=1),
                created_at=datetime(2026, 8, 11, tzinfo=timezone.utc),
            ),
        ]
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)

    jobs = await list_shadow_eval_jobs(VIEWER, api_key_id=None, limit=50)

    assert [(job.job_id, job.status) for job in jobs] == [
        ("job-1", "running"),
        ("job-2", "stopped"),
        ("job-3", "completed"),
    ]
    assert [key.api_key_id for key in jobs[0].keys] == ["key-hash", "key-hash-2"]
    assert all(job.judged_count is None and job.results is None for job in jobs)
    legs_sql, legs_limit = prisma.db.query_raw.await_args_list[0].args
    assert "GROUP BY group_id ORDER BY MAX(created_at) DESC LIMIT $1::int" in legs_sql
    assert legs_limit == 50
    counts_sql, _ = prisma.db.query_raw.await_args_list[1].args
    assert "AS attempt_count" in counts_sql
    assert "j.stopped_at IS NULL OR a.created_at <= j.stopped_at" in counts_sql
    assert prisma.db.query_raw.await_count == 2
    prisma.db.litellm_shadowevaljob.find_many.assert_not_called()


@pytest.mark.asyncio
async def test_list_shadow_eval_jobs_filters_to_jobs_containing_the_key(monkeypatch: pytest.MonkeyPatch):
    """The filter matches a key anywhere in a job's key set and still returns the whole
    job, sibling keys included."""
    import litellm.proxy.proxy_server as proxy_server

    prisma = _shadow_prisma(
        legs=[
            _leg_record(),
            _leg_record(id="leg-2", api_key_id="key-hash-2"),
            _leg_record(id="leg-3", group_id="job-2", api_key_id="key-hash-2"),
            _leg_record(id="leg-4", group_id="job-3"),
        ]
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)

    jobs = await list_shadow_eval_jobs(VIEWER, api_key_id="key-hash-2", limit=50)

    assert [job.job_id for job in jobs] == ["job-1", "job-2"]
    assert [key.api_key_id for key in jobs[0].keys] == ["key-hash", "key-hash-2"]


@pytest.mark.parametrize(
    ("stopped_flags", "days_left", "expected"),
    [
        ((False, False), 7, "running"),
        ((True, False), 7, "running"),
        ((True, True), 7, "stopped"),
        ((True, True), -1, "completed"),
        ((False, False), -1, "completed"),
    ],
)
@pytest.mark.asyncio
async def test_job_status_runs_until_every_key_stops_and_completed_outranks_stopped(
    monkeypatch: pytest.MonkeyPatch, stopped_flags: tuple[bool, ...], days_left: int, expected: str
):
    import litellm.proxy.proxy_server as proxy_server

    stamp = datetime.now(timezone.utc)
    prisma = _shadow_prisma(
        legs=[
            _leg_record(
                id=f"leg-{index}",
                api_key_id=f"key-{index}",
                stopped_at=stamp if stopped else None,
                ends_at=datetime.now(timezone.utc) + timedelta(days=days_left),
            )
            for index, stopped in enumerate(stopped_flags)
        ]
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)

    jobs = await list_shadow_eval_jobs(VIEWER, api_key_id=None, limit=50)

    assert [job.status for job in jobs] == [expected]


@pytest.mark.asyncio
async def test_list_reads_completed_once_every_key_spends_its_budget(monkeypatch: pytest.MonkeyPatch):
    """A job whose keys all exhausted their turn budgets stopped sampling on its own, so
    it must read completed on the very next list, before any sweep stamps its legs; one
    key under budget keeps the whole job running. An operator starting an unrelated eval
    must never look like it terminated a finished one."""
    import litellm.proxy.proxy_server as proxy_server

    prisma = _shadow_prisma(
        legs=[
            _leg_record(max_turns=5),
            _leg_record(id="leg-2", api_key_id="key-hash-2", max_turns=5),
            _leg_record(id="leg-3", group_id="job-2", api_key_id="key-hash", max_turns=5),
            _leg_record(id="leg-4", group_id="job-2", api_key_id="key-hash-2", max_turns=5),
        ]
    )
    prisma.attempt_rows = [
        {"job_id": "leg-1", "attempt_count": 5, "spend": 0.0},
        {"job_id": "leg-2", "attempt_count": 6, "spend": 0.0},
        {"job_id": "leg-3", "attempt_count": 5, "spend": 0.0},
        {"job_id": "leg-4", "attempt_count": 3, "spend": 0.0},
    ]
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)

    jobs = await list_shadow_eval_jobs(VIEWER, api_key_id=None, limit=50)

    by_id = {job.job_id: job for job in jobs}
    assert by_id["job-1"].status == "completed"
    assert all(key.stopped_at is None for key in by_id["job-1"].keys)
    assert by_id["job-2"].status == "running"
    assert {key.api_key_id: key.attempt_count for key in by_id["job-2"].keys} == {"key-hash": 5, "key-hash-2": 3}


@pytest.mark.asyncio
async def test_recorded_operator_stop_outranks_budget_arithmetic(monkeypatch: pytest.MonkeyPatch):
    """A detached attempt can land around the stop and push the raw count past the
    budget; the recorded stopped_by must keep the job reading stopped regardless."""
    import litellm.proxy.proxy_server as proxy_server

    stamp = datetime.now(timezone.utc)
    prisma = _shadow_prisma(legs=[_leg_record(max_turns=5, stopped_at=stamp, stopped_by="admin")])
    prisma.attempt_rows = [{"job_id": "leg-1", "attempt_count": 6, "spend": 0.0}]
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)

    jobs = await list_shadow_eval_jobs(VIEWER, api_key_id=None, limit=50)
    assert jobs[0].status == "stopped"
    assert jobs[0].stopped_by == "admin"

    detail = await get_shadow_eval_job("job-1", VIEWER)
    assert detail.status == "stopped"


@pytest.mark.asyncio
async def test_backfilled_legacy_stop_never_reads_as_completion(monkeypatch: pytest.MonkeyPatch):
    """Jobs stopped before stopped_by existed are backfilled with 'unknown' by the
    migration, so even one whose stray attempts crossed the budget stays stopped."""
    import litellm.proxy.proxy_server as proxy_server

    prisma = _shadow_prisma(
        legs=[_leg_record(max_turns=5, stopped_at=datetime.now(timezone.utc), stopped_by="unknown")]
    )
    prisma.attempt_rows = [{"job_id": "leg-1", "attempt_count": 6, "spend": 0.0}]
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)

    jobs = await list_shadow_eval_jobs(VIEWER, api_key_id=None, limit=50)
    assert jobs[0].status == "stopped"


def test_stopped_by_migration_backfills_every_job_that_displayed_stopped():
    """The migration must close the pre-column population: without the backfill, a
    legacy stop whose stray attempts crossed the budget would read completed."""
    import litellm_proxy_extras

    sql = (
        Path(litellm_proxy_extras.__file__).parent
        / "migrations"
        / "20260818224500_add_shadow_eval_stopped_by"
        / "migration.sql"
    ).read_text()
    assert 'ADD COLUMN     "stopped_by" TEXT' in sql
    assert "SET stopped_by = 'unknown'" in sql
    assert "WHERE stopped_at IS NOT NULL AND ends_at > (NOW() AT TIME ZONE 'utc')" in sql


def test_a_start_request_still_sending_max_turns_is_rejected_not_silently_defaulted():
    """Pydantic ignores unknown fields, so without the explicit rejection a caller still
    sending the retired turn budget would silently run on the default dollar budget."""
    with pytest.raises(ValidationError, match="max_budget"):
        _start_request(max_turns=200)


def test_max_budget_migration_is_additive_and_leaves_legacy_rows_null():
    """max_budget stays NULL on pre-migration rows so they keep the turn budget they were
    configured with, and shadow_cost defaults to 0 so old rows price as judge-only."""
    import litellm_proxy_extras

    sql = (
        Path(litellm_proxy_extras.__file__).parent
        / "migrations"
        / "20260819000000_shadow_eval_max_budget"
        / "migration.sql"
    ).read_text()
    assert 'ALTER TABLE "LiteLLM_ShadowEvalJob" ADD COLUMN     "max_budget" DOUBLE PRECISION' in sql
    assert (
        'ALTER TABLE "LiteLLM_ShadowEvalAttempt" ADD COLUMN     "shadow_cost" DOUBLE PRECISION NOT NULL DEFAULT 0'
        in sql
    )
    assert "UPDATE" not in sql
    assert "DROP" not in sql


@pytest.mark.asyncio
async def test_stop_rejects_a_job_that_already_spent_its_budget(monkeypatch: pytest.MonkeyPatch):
    import litellm.proxy.proxy_server as proxy_server

    prisma = _shadow_prisma(legs=[_leg_record(max_turns=3)])
    prisma.attempt_rows = [{"job_id": "leg-1", "attempt_count": 3, "spend": 0.0}]
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)

    with pytest.raises(HTTPException) as exhausted:
        await stop_shadow_eval_job("job-1", ADMIN)
    assert exhausted.value.status_code == 400
    assert "completed" in exhausted.value.detail
    prisma.db.litellm_shadowevaljob.update_many.assert_not_called()


@pytest.mark.asyncio
async def test_list_reads_completed_once_every_key_spends_its_dollar_budget(monkeypatch: pytest.MonkeyPatch):
    """A spend-budgeted job completes on dollars, not turns: every key's recorded shadow
    plus judge spend reaching max_budget reads completed long before the turn valve, while
    one key with budget left keeps the whole job running."""
    import litellm.proxy.proxy_server as proxy_server

    prisma = _shadow_prisma(
        legs=[
            _leg_record(max_turns=SHADOW_EVAL_TURN_VALVE, max_budget=1.0),
            _leg_record(id="leg-2", api_key_id="key-hash-2", max_turns=SHADOW_EVAL_TURN_VALVE, max_budget=1.0),
            _leg_record(
                id="leg-3", group_id="job-2", api_key_id="key-hash", max_turns=SHADOW_EVAL_TURN_VALVE, max_budget=1.0
            ),
        ]
    )
    prisma.attempt_rows = [
        {"job_id": "leg-1", "attempt_count": 40, "spend": 1.0},
        {"job_id": "leg-2", "attempt_count": 55, "spend": 1.25},
        {"job_id": "leg-3", "attempt_count": 40, "spend": 0.99},
    ]
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)

    jobs = await list_shadow_eval_jobs(VIEWER, api_key_id=None, limit=50)

    by_id = {job.job_id: job for job in jobs}
    assert by_id["job-1"].status == "completed"
    assert by_id["job-2"].status == "running"
    assert {key.api_key_id: key.spend for key in by_id["job-1"].keys} == {"key-hash": 1.0, "key-hash-2": 1.25}
    assert all(key.max_budget == 1.0 for key in by_id["job-1"].keys)


@pytest.mark.asyncio
async def test_stop_rejects_a_job_whose_dollar_budget_is_spent(monkeypatch: pytest.MonkeyPatch):
    import litellm.proxy.proxy_server as proxy_server

    prisma = _shadow_prisma(legs=[_leg_record(max_turns=SHADOW_EVAL_TURN_VALVE, max_budget=0.5)])
    prisma.attempt_rows = [{"job_id": "leg-1", "attempt_count": 7, "spend": 0.5}]
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)

    with pytest.raises(HTTPException) as exhausted:
        await stop_shadow_eval_job("job-1", ADMIN)
    assert exhausted.value.status_code == 400
    assert "completed" in exhausted.value.detail
    prisma.db.litellm_shadowevaljob.update_many.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_jobs_without_a_dollar_budget_stay_turn_gated(monkeypatch: pytest.MonkeyPatch):
    """A job from before spend budgets existed carries max_budget NULL: recorded spend
    can never complete it, only its own max_turns can, so migration changes nothing about
    what it was configured to do."""
    import litellm.proxy.proxy_server as proxy_server

    prisma = _shadow_prisma(legs=[_leg_record(max_turns=200, max_budget=None)])
    prisma.attempt_rows = [{"job_id": "leg-1", "attempt_count": 40, "spend": 250.0}]
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)

    jobs = await list_shadow_eval_jobs(VIEWER, api_key_id=None, limit=50)

    assert jobs[0].status == "running"
    assert jobs[0].keys[0].max_budget is None
    assert jobs[0].keys[0].spend == 250.0


@pytest.mark.asyncio
async def test_shadow_eval_responses_name_every_shadowed_key(monkeypatch: pytest.MonkeyPatch):
    import litellm.proxy.proxy_server as proxy_server

    prisma = _shadow_prisma(
        legs=[_leg_record(), _leg_record(id="leg-2", api_key_id="deleted-key-hash")],
        known_keys=("key-hash", "key-hash-2"),
    )
    monkeypatch.setattr(proxy_server, "llm_router", _shadow_router())
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)

    jobs = await list_shadow_eval_jobs(VIEWER, api_key_id=None, limit=50)
    assert [(key.key_alias, key.key_name) for key in jobs[0].keys] == [
        (None, None),
        ("prod-alpha", "sk-...lpha"),
    ]
    batched_where = prisma.db.litellm_verificationtoken.find_many.call_args.kwargs["where"]
    assert batched_where == {"token": {"in": ["deleted-key-hash", "key-hash"]}}

    detail = await get_shadow_eval_job("job-1", VIEWER)
    assert [key.key_alias for key in detail.keys] == [None, "prod-alpha"]


@pytest.mark.asyncio
async def test_stop_shadow_eval_stops_every_unstopped_leg_and_rejects_non_running(
    monkeypatch: pytest.MonkeyPatch,
):
    """One stop ends sampling for the whole job, while a leg that already stopped on its
    own budget keeps the stopped_at it earned."""
    import litellm.proxy.proxy_server as proxy_server

    earned = datetime.now(timezone.utc) - timedelta(hours=1)
    prisma = _shadow_prisma(legs=[_leg_record(), _leg_record(id="leg-2", api_key_id="key-hash-2", stopped_at=earned)])
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)

    stopped = await stop_shadow_eval_job("job-1", ADMIN)

    assert stopped.status == "stopped"
    assert stopped.stopped_by == "admin"
    stop_sql, stop_group, stop_operator, stop_stamp = prisma.db.execute_raw.call_args.args
    assert "SET stopped_by = $2, stopped_at = COALESCE(stopped_at, $3::timestamp)" in stop_sql
    assert "WHERE group_id = $1 AND stopped_by IS NULL" in stop_sql
    assert "ends_at > (NOW() AT TIME ZONE 'utc')" in stop_sql
    assert ") < k.max_turns" in stop_sql
    assert "k.max_budget IS NULL" in stop_sql
    assert ") < k.max_budget" in stop_sql
    assert "SUM(a.judge_cost + a.shadow_cost)" in stop_sql
    assert (stop_group, stop_operator) == ("job-1", "admin")
    assert datetime.fromisoformat(stop_stamp).tzinfo is None
    assert prisma.db.execute_raw.await_count == 1
    prisma.db.litellm_shadowevaljob.update_many.assert_not_called()
    by_key = {key.api_key_id: key.stopped_at for key in stopped.keys}
    assert by_key["key-hash-2"] == earned
    assert by_key["key-hash"] is not None and by_key["key-hash"] != earned

    done_leg = _leg_record(ends_at=datetime.now(timezone.utc) - timedelta(days=1))
    prisma_done = _shadow_prisma(legs=[done_leg])
    monkeypatch.setattr(proxy_server, "prisma_client", prisma_done)
    with pytest.raises(HTTPException) as exc:
        await stop_shadow_eval_job("job-1", ADMIN)
    assert exc.value.status_code == 400
    assert "already completed" in exc.value.detail
    assert done_leg.stopped_by is None

    with pytest.raises(HTTPException) as forbidden:
        await stop_shadow_eval_job("job-1", VIEWER)
    assert forbidden.value.status_code == 403


@pytest.mark.asyncio
async def test_validate_config_returns_the_write_gates_verdict_without_saving():
    """The dry-run endpoint must agree with the write gate exactly, so a form showing its
    verdict inline can never pass a config the save would then reject."""
    from litellm.proxy.management_endpoints.auto_router_endpoints import (
        validate_complexity_router_config,
    )
    from litellm.types.management_endpoints.auto_router_endpoints import (
        ComplexityRouterConfigValidationRequest,
    )

    valid = await validate_complexity_router_config(
        ComplexityRouterConfigValidationRequest(
            complexity_router_config={
                "tiers": {"CASUAL": "m1", "AUDIT": "m2"},
                "tier_definitions": [
                    {"name": "CASUAL", "description": "casual chat"},
                    {"name": "AUDIT", "description": "security audits"},
                ],
                "fallback_tier": "AUDIT",
                "classifier_type": "llm",
                "classifier_llm_config": {"model": "clf"},
            }
        ),
        ADMIN,
    )
    assert valid.valid is True
    assert valid.error is None

    rejected = await validate_complexity_router_config(
        ComplexityRouterConfigValidationRequest(
            complexity_router_config={
                "tiers": {"CASUAL": "m1", "AUDIT": "m2"},
                "tier_definitions": [
                    {"name": "CASUAL", "description": "casual chat"},
                    {"name": "AUDIT", "description": "security\naudits"},
                ],
                "fallback_tier": "AUDIT",
                "classifier_type": "llm",
                "classifier_llm_config": {"model": "clf"},
            }
        ),
        ADMIN,
    )
    assert rejected.valid is False
    assert rejected.error is not None and "newline" in rejected.error


@pytest.mark.asyncio
async def test_routing_test_never_confirms_models_the_caller_cannot_use(monkeypatch: pytest.MonkeyPatch):
    """routed_model_configured must not be an existence oracle for the whole proxy: a team
    admin probing a guessed global model name reads False unless the named team could
    actually use that model, and True once the team grants it."""
    from litellm.proxy import proxy_server

    def _team_prisma(team_id: str, models: list[str]) -> MagicMock:
        row_data = {
            "team_id": team_id,
            "members_with_roles": [{"role": "admin", "user_id": "team-admin"}],
            "models": models,
        }
        team_row = MagicMock()
        team_row.model_dump.return_value = row_data
        team_row.dict.return_value = row_data
        prisma = MagicMock()
        prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_row)
        return prisma

    def _request(team_id: str) -> AutoRouterRoutingTestRequest:
        return AutoRouterRoutingTestRequest.model_validate(
            {
                "prompt": "what is 2+2",
                "complexity_router_config": {"tiers": TIERS, "classifier_type": "heuristic"},
                "team_id": team_id,
            }
        )

    monkeypatch.setattr(proxy_server, "premium_user", True)
    monkeypatch.setattr(proxy_server, "llm_router", _router())

    team_admin: Final = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER, api_key="sk-team", user_id="team-admin"
    )

    monkeypatch.setattr(proxy_server, "prisma_client", _team_prisma("team-probe", models=["mid-model"]))
    probing = await preview_auto_router_routing(data=_request("team-probe"), user_api_key_dict=team_admin)
    assert probing.routed_model == "cheap-model"
    assert probing.routed_model_configured is False

    monkeypatch.setattr(proxy_server, "prisma_client", _team_prisma("team-grant", models=["cheap-model"]))
    granted = await preview_auto_router_routing(data=_request("team-grant"), user_api_key_dict=team_admin)
    assert granted.routed_model == "cheap-model"
    assert granted.routed_model_configured is True


@pytest.mark.asyncio
async def test_validate_config_gates_like_the_write_it_rehearses(monkeypatch: pytest.MonkeyPatch):
    """A caller who could not save the router must not get the dry run either: matching
    /model/new, a team admin passes only when naming their own team, and a caller who is
    neither proxy admin nor team admin is rejected before validation runs."""
    from litellm.proxy import proxy_server
    from litellm.proxy.management_endpoints.auto_router_endpoints import (
        validate_complexity_router_config,
    )
    from litellm.types.management_endpoints.auto_router_endpoints import (
        ComplexityRouterConfigValidationRequest,
    )

    config: Final = {"tiers": {"SIMPLE": "m1"}, "classifier_type": "heuristic"}

    with pytest.raises(HTTPException) as forbidden:
        await validate_complexity_router_config(
            ComplexityRouterConfigValidationRequest(complexity_router_config=config), VIEWER
        )
    assert forbidden.value.status_code == 403

    team_row: Final = MagicMock()
    team_row.model_dump.return_value = {
        "team_id": "team-1",
        "members_with_roles": [{"role": "admin", "user_id": "team-admin"}],
    }
    prisma: Final = MagicMock()
    prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_row)
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(proxy_server, "premium_user", True)

    team_admin: Final = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER, api_key="sk-team", user_id="team-admin"
    )
    verdict = await validate_complexity_router_config(
        ComplexityRouterConfigValidationRequest(complexity_router_config=config, team_id="team-1"),
        team_admin,
    )
    assert verdict.valid is True

    with pytest.raises(HTTPException) as not_their_team:
        await validate_complexity_router_config(
            ComplexityRouterConfigValidationRequest(complexity_router_config=config, team_id="team-1"),
            UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, api_key="sk-other", user_id="someone-else"),
        )
    assert not_their_team.value.status_code == 403


def test_every_shadow_eval_sql_constant_speaks_naive_utc():
    """The tables store naive UTC wall time (prisma's convention), so SQL-side time must be
    NOW() AT TIME ZONE 'utc' and python-side params must cast ::timestamp; a bare NOW() or a
    timestamptz cast writes session-local wall time into the naive column and skews every
    comparison against prisma-written stamps."""
    import litellm.proxy.management_endpoints.auto_router_endpoints as module

    sql_constants = {name: value for name, value in vars(module).items() if name.endswith("_SQL")}
    assert sql_constants
    for name, sql in sql_constants.items():
        assert "::timestamptz" not in sql, name
        for occurrence in sql.split("NOW()")[1:]:
            assert occurrence.startswith(" AT TIME ZONE 'utc'"), name


@pytest.mark.asyncio
async def test_a_stop_racing_the_last_budgeted_attempt_reports_completed_not_stopped(
    monkeypatch: pytest.MonkeyPatch,
):
    """The statement claims the job only while a leg still samples, so a stop landing in
    the same instant the budget spends records nothing and the job keeps reading
    completed; stamping it would misreport a self-ended job as operator-stopped forever."""
    import litellm.proxy.proxy_server as proxy_server

    prisma = _shadow_prisma(legs=[_leg_record(max_turns=2)])
    prisma.attempt_rows = [{"job_id": "leg-1", "attempt_count": 2, "spend": 0.0}]
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)

    with pytest.raises(HTTPException) as exc:
        await stop_shadow_eval_job("job-1", ADMIN)
    assert exc.value.status_code == 400
    assert "already completed" in exc.value.detail
    assert prisma.db.litellm_shadowevaljob.find_many.await_args.kwargs["where"] == {"group_id": "job-1"}


@pytest.mark.asyncio
async def test_two_racing_stops_produce_exactly_one_winner(monkeypatch: pytest.MonkeyPatch):
    """The statement's stopped_by IS NULL predicate lets only one racer claim rows; the
    loser reads the stamped state and gets the same answer a late caller gets."""
    import litellm.proxy.proxy_server as proxy_server

    prisma = _shadow_prisma(legs=[_leg_record()])
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)

    first = await stop_shadow_eval_job("job-1", ADMIN)
    assert first.status == "stopped"

    with pytest.raises(HTTPException) as exc:
        await stop_shadow_eval_job("job-1", ADMIN)
    assert exc.value.status_code == 400
    assert "already stopped" in exc.value.detail
