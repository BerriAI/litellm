"""
Tests for the QualityRouter.

Covers:
- Tier index construction from `model_info.litellm_routing_preferences`.
- Quality-tier resolution (exact, round-up, default fallback).
- Keyword override (match, tiebreaking by quality + price).
- Pre-routing hook end-to-end.
- Decision metadata stash + Router.set_response_headers lift.
"""

import os
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.router_strategy.quality_router.config import (
    DEFAULT_COMPLEXITY_TO_QUALITY,
)
from litellm.router_strategy.quality_router.quality_router import QualityRouter


def _make_model_list(spec: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build a router model_list from a compact spec.

    spec entry shape: {
        "model_name": str,
        "quality_tier": Optional[int],
        "keywords": Optional[List[str]],
        "order": Optional[int],
        "input_cost_per_token": Optional[float],
    }
    If quality_tier is None, the deployment is created without
    `litellm_routing_preferences`.
    """
    out: List[Dict[str, Any]] = []
    for entry in spec:
        model_info: Dict[str, Any] = {"id": f"id-{entry['model_name']}"}
        if entry.get("quality_tier") is not None:
            prefs: Dict[str, Any] = {"quality_tier": entry["quality_tier"]}
            if "keywords" in entry:
                prefs["keywords"] = entry["keywords"]
            if "order" in entry:
                prefs["order"] = entry["order"]
            model_info["litellm_routing_preferences"] = prefs
        if "input_cost_per_token" in entry:
            model_info["input_cost_per_token"] = entry["input_cost_per_token"]
        out.append(
            {
                "model_name": entry["model_name"],
                "litellm_params": {"model": f"openai/{entry['model_name']}"},
                "model_info": model_info,
            }
        )
    return out


@pytest.fixture
def four_tier_model_list() -> List[Dict[str, Any]]:
    """A standard haiku(1)/sonnet(2)/opus(3)/opus-next(4) model list."""
    return _make_model_list(
        [
            {"model_name": "haiku", "quality_tier": 1},
            {"model_name": "sonnet", "quality_tier": 2},
            {"model_name": "opus", "quality_tier": 3},
            {"model_name": "opus-next", "quality_tier": 4},
        ]
    )


@pytest.fixture
def mock_router(four_tier_model_list):
    """A MagicMock router preloaded with the four-tier model list."""
    router = MagicMock()
    router.model_list = four_tier_model_list
    return router


@pytest.fixture
def quality_router(mock_router) -> QualityRouter:
    """Default QualityRouter wired to all four tiers."""
    config = {
        "available_models": ["haiku", "sonnet", "opus", "opus-next"],
        "complexity_to_quality": DEFAULT_COMPLEXITY_TO_QUALITY,
    }
    return QualityRouter(
        model_name="quality-router-test",
        litellm_router_instance=mock_router,
        default_model="haiku",
        quality_router_config=config,
    )


# ─── Tier index ─────────────────────────────────────────────────────────────


class TestTierIndex:
    def test_builds_correct_tier_to_models_map(self, quality_router):
        assert quality_router._tier_to_models == {
            1: ["haiku"],
            2: ["sonnet"],
            3: ["opus"],
            4: ["opus-next"],
        }

    def test_ignores_models_not_in_available_models(self, four_tier_model_list):
        # Add a model the config doesn't list — it should be ignored.
        extra = _make_model_list([{"model_name": "ghost", "quality_tier": 5}])
        router = MagicMock()
        router.model_list = four_tier_model_list + extra

        qr = QualityRouter(
            model_name="qr",
            litellm_router_instance=router,
            default_model="haiku",
            quality_router_config={
                "available_models": ["haiku", "sonnet", "opus", "opus-next"]
            },
        )

        for models in qr._tier_to_models.values():
            assert "ghost" not in models

    def test_raises_when_routing_preferences_missing(self):
        # `sonnet` is in available_models but has no preferences.
        ml = _make_model_list(
            [
                {"model_name": "haiku", "quality_tier": 1},
                {"model_name": "sonnet", "quality_tier": None},
            ]
        )
        router = MagicMock()
        router.model_list = ml

        # Construction succeeds (tier index is lazy); the error surfaces on
        # first use so the router entry doesn't have to appear after all of
        # its referenced models in config.yaml.
        qr = QualityRouter(
            model_name="qr",
            litellm_router_instance=router,
            default_model="haiku",
            quality_router_config={"available_models": ["haiku", "sonnet"]},
        )
        with pytest.raises(ValueError, match="sonnet"):
            _ = qr._tier_to_models


# ─── Resolve model for quality tier ─────────────────────────────────────────


class TestResolveModelForQualityTier:
    def test_exact_match(self, quality_router):
        assert quality_router._resolve_model_for_quality_tier(2) == "sonnet"
        assert quality_router._resolve_model_for_quality_tier(4) == "opus-next"

    def test_rounds_up_when_tier_missing(self, mock_router):
        # Available tiers: 1, 3, 4. Asking for 2 should round up to 3.
        spec = [
            {"model_name": "haiku", "quality_tier": 1},
            {"model_name": "opus", "quality_tier": 3},
            {"model_name": "opus-next", "quality_tier": 4},
        ]
        router = MagicMock()
        router.model_list = _make_model_list(spec)

        qr = QualityRouter(
            model_name="qr",
            litellm_router_instance=router,
            default_model="haiku",
            quality_router_config={"available_models": ["haiku", "opus", "opus-next"]},
        )

        assert qr._resolve_model_for_quality_tier(2) == "opus"

    def test_rounds_down_when_no_higher_tier_exists(self):
        # Only tier 1 available. Asking for tier 4 rounds up (nothing), then
        # rounds DOWN to the closest lower tier — tier 1.
        spec = [{"model_name": "haiku", "quality_tier": 1}]
        router = MagicMock()
        router.model_list = _make_model_list(spec)

        qr = QualityRouter(
            model_name="qr",
            litellm_router_instance=router,
            default_model="emergency-default",
            quality_router_config={"available_models": ["haiku"]},
        )

        assert qr._resolve_model_for_quality_tier(4) == "haiku"

    def test_rounds_down_prefers_closest_lower_tier(self):
        # Available: 1, 2. Asking for 4 rounds down to tier 2 (not tier 1).
        spec = [
            {"model_name": "haiku", "quality_tier": 1},
            {"model_name": "sonnet", "quality_tier": 2},
        ]
        router = MagicMock()
        router.model_list = _make_model_list(spec)

        qr = QualityRouter(
            model_name="qr",
            litellm_router_instance=router,
            default_model="emergency-default",
            quality_router_config={"available_models": ["haiku", "sonnet"]},
        )

        assert qr._resolve_model_for_quality_tier(4) == "sonnet"

    def test_prefers_round_up_over_round_down(self):
        # Available: 1, 3. Asking for 2 rounds UP to 3, not DOWN to 1.
        spec = [
            {"model_name": "haiku", "quality_tier": 1},
            {"model_name": "opus", "quality_tier": 3},
        ]
        router = MagicMock()
        router.model_list = _make_model_list(spec)

        qr = QualityRouter(
            model_name="qr",
            litellm_router_instance=router,
            default_model="emergency-default",
            quality_router_config={"available_models": ["haiku", "opus"]},
        )

        assert qr._resolve_model_for_quality_tier(2) == "opus"


# ─── RoutingPreferences validation ─────────────────────────────────────────


class TestRoutingPreferencesValidation:
    def test_invalid_quality_tier_type_raises_clear_error(self):
        # quality_tier must be an int — pass a non-coercible string.
        ml = [
            {
                "model_name": "haiku",
                "litellm_params": {"model": "openai/gpt-4o-mini"},
                "model_info": {
                    "id": "id-haiku",
                    "litellm_routing_preferences": {"quality_tier": "not-an-int"},
                },
            }
        ]
        router = MagicMock()
        router.model_list = ml

        qr = QualityRouter(
            model_name="qr",
            litellm_router_instance=router,
            default_model="haiku",
            quality_router_config={"available_models": ["haiku"]},
        )
        with pytest.raises(ValueError, match="invalid litellm_routing_preferences"):
            _ = qr._tier_to_models


# ─── Config-ordering independence (lazy index build) ───────────────────────


class TestConfigOrderingIndependence:
    def test_router_can_be_instantiated_before_its_targets_exist(self):
        # Build a router instance whose referenced model_list is EMPTY at
        # construction time (simulating a config where the router entry
        # appears before its target deployments). The tier index must not be
        # built eagerly — it's deferred until first use.
        router = MagicMock()
        router.model_list = []  # <- targets haven't been added yet

        qr = QualityRouter(
            model_name="qr",
            litellm_router_instance=router,
            default_model="haiku",
            quality_router_config={"available_models": ["haiku", "sonnet", "opus"]},
        )

        # Now the targets come online. This mirrors the incremental add by
        # `Router._create_deployment`.
        router.model_list = _make_model_list(
            [
                {"model_name": "haiku", "quality_tier": 1},
                {"model_name": "sonnet", "quality_tier": 2},
                {"model_name": "opus", "quality_tier": 3},
            ]
        )

        # First access triggers the index build and sees the full list.
        assert qr._tier_to_models == {
            1: ["haiku"],
            2: ["sonnet"],
            3: ["opus"],
        }


# ─── Router.set_model_list resets quality_routers (hot reload) ─────────────


class TestSetModelListResetsQualityRouters:
    def test_set_model_list_clears_quality_routers_registry(self):
        from litellm.router import Router

        router = Router(
            model_list=[
                {
                    "model_name": "haiku",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "sk-test",
                    },
                    "model_info": {"litellm_routing_preferences": {"quality_tier": 1}},
                },
                {
                    "model_name": "my-qr",
                    "litellm_params": {
                        "model": "auto_router/quality_router",
                        "quality_router_default_model": "haiku",
                        "quality_router_config": {"available_models": ["haiku"]},
                    },
                },
            ]
        )

        assert "my-qr" in router.quality_routers

        # Hot-reload with a new model_list that doesn't define the router.
        router.set_model_list(
            [
                {
                    "model_name": "haiku",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "sk-test",
                    },
                }
            ]
        )

        # Stale router from before must be cleared.
        assert "my-qr" not in router.quality_routers


# ─── Pre-routing hook ───────────────────────────────────────────────────────


class TestPreRoutingHook:
    @pytest.mark.asyncio
    async def test_simple_message_routes_to_tier_1(self, quality_router):
        messages = [{"role": "user", "content": "hi"}]
        resp = await quality_router.async_pre_routing_hook(
            model="quality-router-test",
            request_kwargs={},
            messages=messages,
        )
        assert resp is not None
        assert resp.model == "haiku"

    @pytest.mark.asyncio
    async def test_reasoning_message_routes_to_tier_4(self, quality_router):
        # Two reasoning markers triggers ComplexityTier.REASONING → quality 4.
        messages = [
            {
                "role": "user",
                "content": (
                    "Think step by step and reason through this problem. "
                    "Analyze this carefully and break down each component."
                ),
            }
        ]
        resp = await quality_router.async_pre_routing_hook(
            model="quality-router-test",
            request_kwargs={},
            messages=messages,
        )
        assert resp is not None
        assert resp.model == "opus-next"

    @pytest.mark.asyncio
    async def test_empty_messages_returns_none(self, quality_router):
        resp = await quality_router.async_pre_routing_hook(
            model="quality-router-test",
            request_kwargs={},
            messages=[],
        )
        assert resp is None

    @pytest.mark.asyncio
    async def test_only_system_message_routes_to_default(self, quality_router):
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        resp = await quality_router.async_pre_routing_hook(
            model="quality-router-test",
            request_kwargs={},
            messages=messages,
        )
        assert resp is not None
        assert resp.model == "haiku"  # the configured default_model


# ─── Keyword override ──────────────────────────────────────────────────────


@pytest.fixture
def keyword_router():
    """
    Router where multiple deployments declare overlapping keywords so we can
    exercise the (quality DESC, price ASC) tiebreak.

      - cheap-coder       tier 2, keywords [code, python], cost 0.000001
      - smart-coder       tier 3, keywords [code, python], cost 0.000010
      - law-bot           tier 2, keywords [legal, contract], cost 0.000005
      - default-haiku     tier 1, no keywords, cost 0.0000005
    """
    spec = [
        {
            "model_name": "default-haiku",
            "quality_tier": 1,
            "keywords": [],
            "input_cost_per_token": 0.0000005,
        },
        {
            "model_name": "cheap-coder",
            "quality_tier": 2,
            "keywords": ["code", "python"],
            "input_cost_per_token": 0.000001,
        },
        {
            "model_name": "smart-coder",
            "quality_tier": 3,
            "keywords": ["code", "python"],
            "input_cost_per_token": 0.000010,
        },
        {
            "model_name": "law-bot",
            "quality_tier": 2,
            "keywords": ["legal", "contract"],
            "input_cost_per_token": 0.000005,
        },
    ]
    router = MagicMock()
    router.model_list = _make_model_list(spec)
    return QualityRouter(
        model_name="qr",
        litellm_router_instance=router,
        default_model="default-haiku",
        quality_router_config={
            "available_models": [
                "default-haiku",
                "cheap-coder",
                "smart-coder",
                "law-bot",
            ],
        },
    )


class TestKeywordOverride:
    def test_no_keyword_in_message_returns_none(self, keyword_router):
        assert keyword_router._keyword_override("hello there") is None

    def test_single_match_returns_that_model(self, keyword_router):
        # Only law-bot declares "legal".
        assert keyword_router._keyword_override("review this legal doc") == (
            "law-bot",
            "legal",
        )

    def test_case_insensitive_match(self, keyword_router):
        assert keyword_router._keyword_override("LEGAL question") == (
            "law-bot",
            "legal",
        )

    def test_overlap_picks_highest_quality_tier(self, keyword_router):
        # Both cheap-coder (tier 2) and smart-coder (tier 3) declare "code".
        # Quality wins over price → smart-coder.
        assert keyword_router._keyword_override("write some code for me") == (
            "smart-coder",
            "code",
        )

    def test_same_tier_picks_cheapest(self):
        # Two models at the same tier, both matching "data" — cheapest wins.
        spec = [
            {
                "model_name": "expensive",
                "quality_tier": 2,
                "keywords": ["data"],
                "input_cost_per_token": 0.000050,
            },
            {
                "model_name": "cheap",
                "quality_tier": 2,
                "keywords": ["data"],
                "input_cost_per_token": 0.000005,
            },
        ]
        router = MagicMock()
        router.model_list = _make_model_list(spec)
        qr = QualityRouter(
            model_name="qr",
            litellm_router_instance=router,
            default_model="cheap",
            quality_router_config={"available_models": ["expensive", "cheap"]},
        )
        match = qr._keyword_override("show me the data")
        assert match == ("cheap", "data")

    def test_unpriced_loses_to_priced_at_same_tier(self):
        # Same quality tier, one has cost, one doesn't → priced wins.
        spec = [
            {
                "model_name": "no-price",
                "quality_tier": 2,
                "keywords": ["data"],
                # input_cost_per_token deliberately omitted
            },
            {
                "model_name": "with-price",
                "quality_tier": 2,
                "keywords": ["data"],
                "input_cost_per_token": 0.000005,
            },
        ]
        router = MagicMock()
        router.model_list = _make_model_list(spec)
        qr = QualityRouter(
            model_name="qr",
            litellm_router_instance=router,
            default_model="no-price",
            quality_router_config={"available_models": ["no-price", "with-price"]},
        )
        match = qr._keyword_override("show me the data")
        assert match == ("with-price", "data")

    @pytest.mark.asyncio
    async def test_hook_short_circuits_complexity_on_keyword_match(
        self, keyword_router
    ):
        # A reasoning-style prompt would normally route to a high-quality model
        # via the complexity flow — but the keyword "code" should short-circuit
        # to smart-coder (highest tier among "code" models).
        messages = [
            {
                "role": "user",
                "content": (
                    "Think step by step and reason through this code problem. "
                    "Analyze this carefully and break down each component."
                ),
            }
        ]
        request_kwargs: Dict[str, Any] = {}
        resp = await keyword_router.async_pre_routing_hook(
            model="qr",
            request_kwargs=request_kwargs,
            messages=messages,
        )
        assert resp is not None
        assert resp.model == "smart-coder"

        decision = request_kwargs["metadata"]["quality_router_decision"]
        assert decision["routed_via"] == "keyword"
        assert decision["matched_keyword"] == "code"
        assert decision["complexity_tier"] is None  # short-circuited

    def test_quality_wins_over_explicit_order(self):
        # Quality always beats order. A tier-3 model with no `order` wins over
        # a tier-2 model with `order=1`.
        spec = [
            {
                "model_name": "ordered-tier2",
                "quality_tier": 2,
                "keywords": ["code"],
                "order": 1,
                "input_cost_per_token": 0.000010,
            },
            {
                "model_name": "implicit-tier3",
                "quality_tier": 3,
                "keywords": ["code"],
                "input_cost_per_token": 0.000005,
            },
        ]
        router = MagicMock()
        router.model_list = _make_model_list(spec)
        qr = QualityRouter(
            model_name="qr",
            litellm_router_instance=router,
            default_model="ordered-tier2",
            quality_router_config={
                "available_models": ["ordered-tier2", "implicit-tier3"]
            },
        )
        match = qr._keyword_override("write some code")
        assert match == ("implicit-tier3", "code")

    def test_order_breaks_tie_within_same_quality_tier(self):
        # Two tier-3 models, both match "code". Lower `order` wins.
        spec = [
            {
                "model_name": "preferred",
                "quality_tier": 3,
                "keywords": ["code"],
                "order": 1,
                "input_cost_per_token": 0.000050,  # more expensive
            },
            {
                "model_name": "default-tier3",
                "quality_tier": 3,
                "keywords": ["code"],
                "input_cost_per_token": 0.000005,  # cheaper
            },
        ]
        router = MagicMock()
        router.model_list = _make_model_list(spec)
        qr = QualityRouter(
            model_name="qr",
            litellm_router_instance=router,
            default_model="default-tier3",
            quality_router_config={"available_models": ["preferred", "default-tier3"]},
        )
        match = qr._keyword_override("write some code")
        assert match == ("preferred", "code")

    def test_explicit_order_overrides_price(self):
        # Same tier, but the more expensive one has a lower `order` and wins.
        spec = [
            {
                "model_name": "expensive-but-preferred",
                "quality_tier": 2,
                "keywords": ["data"],
                "order": 1,
                "input_cost_per_token": 0.000050,
            },
            {
                "model_name": "cheap-default",
                "quality_tier": 2,
                "keywords": ["data"],
                "input_cost_per_token": 0.000005,
            },
        ]
        router = MagicMock()
        router.model_list = _make_model_list(spec)
        qr = QualityRouter(
            model_name="qr",
            litellm_router_instance=router,
            default_model="cheap-default",
            quality_router_config={
                "available_models": ["expensive-but-preferred", "cheap-default"]
            },
        )
        match = qr._keyword_override("show me the data")
        assert match == ("expensive-but-preferred", "data")

    def test_lower_order_wins_between_two_explicitly_ordered(self):
        spec = [
            {
                "model_name": "second",
                "quality_tier": 2,
                "keywords": ["data"],
                "order": 5,
            },
            {
                "model_name": "first",
                "quality_tier": 2,
                "keywords": ["data"],
                "order": 1,
            },
        ]
        router = MagicMock()
        router.model_list = _make_model_list(spec)
        qr = QualityRouter(
            model_name="qr",
            litellm_router_instance=router,
            default_model="first",
            quality_router_config={"available_models": ["first", "second"]},
        )
        match = qr._keyword_override("show me the data")
        assert match == ("first", "data")

    def test_same_order_falls_through_to_quality_then_price(self):
        # All three models share order=1 → tiebreak falls through to
        # (quality DESC, cost ASC).
        spec = [
            {
                "model_name": "low-tier",
                "quality_tier": 1,
                "keywords": ["data"],
                "order": 1,
                "input_cost_per_token": 0.000001,
            },
            {
                "model_name": "high-tier-cheap",
                "quality_tier": 3,
                "keywords": ["data"],
                "order": 1,
                "input_cost_per_token": 0.000005,
            },
            {
                "model_name": "high-tier-expensive",
                "quality_tier": 3,
                "keywords": ["data"],
                "order": 1,
                "input_cost_per_token": 0.000050,
            },
        ]
        router = MagicMock()
        router.model_list = _make_model_list(spec)
        qr = QualityRouter(
            model_name="qr",
            litellm_router_instance=router,
            default_model="low-tier",
            quality_router_config={
                "available_models": [
                    "low-tier",
                    "high-tier-cheap",
                    "high-tier-expensive",
                ]
            },
        )
        match = qr._keyword_override("show me the data")
        assert match == ("high-tier-cheap", "data")

    def test_order_is_used_in_tier_resolution_too(self):
        # Two models at the same tier. Explicit `order=1` on the second one
        # should make _resolve_model_for_quality_tier(2) pick it.
        spec = [
            {
                "model_name": "default-pick",
                "quality_tier": 2,
            },
            {
                "model_name": "preferred-pick",
                "quality_tier": 2,
                "order": 1,
            },
        ]
        router = MagicMock()
        router.model_list = _make_model_list(spec)
        qr = QualityRouter(
            model_name="qr",
            litellm_router_instance=router,
            default_model="default-pick",
            quality_router_config={
                "available_models": ["default-pick", "preferred-pick"]
            },
        )
        assert qr._resolve_model_for_quality_tier(2) == "preferred-pick"

    @pytest.mark.asyncio
    async def test_hook_falls_back_to_complexity_when_no_keyword(self, keyword_router):
        # No declared keyword in the message → complexity-based routing.
        # "hi" is SIMPLE → quality 1 → default-haiku (the only tier-1 model).
        messages = [{"role": "user", "content": "hi"}]
        request_kwargs: Dict[str, Any] = {}
        resp = await keyword_router.async_pre_routing_hook(
            model="qr",
            request_kwargs=request_kwargs,
            messages=messages,
        )
        assert resp is not None
        assert resp.model == "default-haiku"

        decision = request_kwargs["metadata"]["quality_router_decision"]
        assert decision["routed_via"] == "quality_tier"
        assert decision["matched_keyword"] is None
        assert decision["complexity_tier"] == "SIMPLE"


# ─── Routing-decision metadata (powers x-litellm-quality-router-* headers) ──


class TestDecisionMetadata:
    @pytest.mark.asyncio
    async def test_hook_stashes_decision_in_request_kwargs_metadata(
        self, quality_router
    ):
        # Reasoning prompt → REASONING → quality tier 4 → opus-next.
        messages = [
            {
                "role": "user",
                "content": (
                    "Think step by step and reason through this problem. "
                    "Analyze this carefully and break down each component."
                ),
            }
        ]
        request_kwargs: Dict[str, Any] = {}

        resp = await quality_router.async_pre_routing_hook(
            model="quality-router-test",
            request_kwargs=request_kwargs,
            messages=messages,
        )
        assert resp is not None and resp.model == "opus-next"

        decision = request_kwargs["metadata"]["quality_router_decision"]
        assert decision["routed_model"] == "opus-next"
        assert decision["quality_tier"] == 4
        assert decision["complexity_tier"] == "REASONING"
        assert decision["router_model_name"] == "quality-router-test"
        assert decision["routed_via"] == "quality_tier"
        assert decision["matched_keyword"] is None

    @pytest.mark.asyncio
    async def test_decision_metadata_preserves_existing_metadata(self, quality_router):
        request_kwargs: Dict[str, Any] = {
            "metadata": {"trace_id": "abc-123", "user_id": "u-1"}
        }

        await quality_router.async_pre_routing_hook(
            model="quality-router-test",
            request_kwargs=request_kwargs,
            messages=[{"role": "user", "content": "hi"}],
        )

        # Existing metadata keys are intact and the decision is added alongside.
        assert request_kwargs["metadata"]["trace_id"] == "abc-123"
        assert request_kwargs["metadata"]["user_id"] == "u-1"
        assert "quality_router_decision" in request_kwargs["metadata"]


# ─── Router.set_response_headers lifts decision into x-litellm-quality-* ────


class TestSetResponseHeadersLiftsDecision:
    """
    Verify the Router.set_response_headers helper turns a stashed quality-router
    decision into x-litellm-quality-router-* headers on the response.
    """

    @pytest.mark.asyncio
    async def test_lifts_decision_into_additional_headers(self):
        from pydantic import BaseModel

        from litellm.router import Router

        class FakeResponse(BaseModel):
            model_config = {"arbitrary_types_allowed": True}
            _hidden_params: Dict[str, Any] = {}

        # Build a real Router with a tiny model_list — enough to satisfy
        # set_response_headers without needing the rest of the router stack.
        router = Router(
            model_list=[
                {
                    "model_name": "haiku",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "sk-test",
                    },
                }
            ]
        )

        response = FakeResponse()
        response._hidden_params = {}

        request_kwargs = {
            "metadata": {
                "quality_router_decision": {
                    "router_model_name": "qr",
                    "routed_model": "smart-coder",
                    "routed_via": "keyword",
                    "matched_keyword": "code",
                    "quality_tier": 3,
                    "complexity_tier": None,
                }
            }
        }

        await router.set_response_headers(
            response=response,
            model_group="qr",
            request_kwargs=request_kwargs,
        )

        headers = response._hidden_params["additional_headers"]
        assert headers["x-litellm-quality-router-model"] == "smart-coder"
        assert headers["x-litellm-quality-router-tier"] == "3"
        assert headers["x-litellm-quality-router-via"] == "keyword"
        assert headers["x-litellm-quality-router-keyword"] == "code"
        # Keyword route short-circuits classification → no complexity header.
        assert "x-litellm-quality-router-complexity" not in headers
        # Existing x-litellm-model-group behavior is unchanged.
        assert headers["x-litellm-model-group"] == "qr"

    @pytest.mark.asyncio
    async def test_quality_tier_route_emits_complexity_not_keyword(self):
        from pydantic import BaseModel

        from litellm.router import Router

        class FakeResponse(BaseModel):
            model_config = {"arbitrary_types_allowed": True}
            _hidden_params: Dict[str, Any] = {}

        router = Router(
            model_list=[
                {
                    "model_name": "haiku",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "sk-test",
                    },
                }
            ]
        )

        response = FakeResponse()
        response._hidden_params = {}

        request_kwargs = {
            "metadata": {
                "quality_router_decision": {
                    "router_model_name": "qr",
                    "routed_model": "haiku",
                    "routed_via": "quality_tier",
                    "matched_keyword": None,
                    "quality_tier": 1,
                    "complexity_tier": "SIMPLE",
                }
            }
        }

        await router.set_response_headers(
            response=response,
            model_group="qr",
            request_kwargs=request_kwargs,
        )

        headers = response._hidden_params["additional_headers"]
        assert headers["x-litellm-quality-router-via"] == "quality_tier"
        assert headers["x-litellm-quality-router-complexity"] == "SIMPLE"
        # Quality-tier route → no keyword header.
        assert "x-litellm-quality-router-keyword" not in headers

    @pytest.mark.asyncio
    async def test_no_decision_leaves_quality_router_headers_unset(self):
        from pydantic import BaseModel

        from litellm.router import Router

        class FakeResponse(BaseModel):
            model_config = {"arbitrary_types_allowed": True}
            _hidden_params: Dict[str, Any] = {}

        router = Router(
            model_list=[
                {
                    "model_name": "haiku",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "sk-test",
                    },
                }
            ]
        )

        response = FakeResponse()
        response._hidden_params = {}

        await router.set_response_headers(
            response=response,
            model_group="haiku",
            request_kwargs={},  # no quality_router_decision
        )

        headers = response._hidden_params["additional_headers"]
        assert "x-litellm-quality-router-model" not in headers
        assert "x-litellm-quality-router-tier" not in headers


class TestRouterQualityDeploymentMethods:
    """Tests for Router._is_quality_router_deployment and Router.init_quality_router_deployment."""

    def test_is_quality_router_deployment_true(self):
        """_is_quality_router_deployment returns True for quality router models."""
        from litellm.router import Router
        from litellm.types.router import LiteLLM_Params

        router = Router(
            model_list=[
                {
                    "model_name": "gpt-4o-mini",
                    "litellm_params": {"model": "openai/gpt-4o-mini"},
                }
            ]
        )
        params = LiteLLM_Params(model="auto_router/quality_router/my-router")
        assert router._is_quality_router_deployment(params) is True

    def test_is_quality_router_deployment_false(self):
        """_is_quality_router_deployment returns False for regular models."""
        from litellm.router import Router
        from litellm.types.router import LiteLLM_Params

        router = Router(
            model_list=[
                {
                    "model_name": "gpt-4o-mini",
                    "litellm_params": {"model": "openai/gpt-4o-mini"},
                }
            ]
        )
        params = LiteLLM_Params(model="openai/gpt-4o-mini")
        assert router._is_quality_router_deployment(params) is False

    def test_init_quality_router_deployment(self):
        """init_quality_router_deployment registers a QualityRouter."""
        from litellm.router import Router
        from litellm.types.router import Deployment, LiteLLM_Params

        router = Router(
            model_list=[
                {
                    "model_name": "gpt-4o-mini",
                    "litellm_params": {"model": "openai/gpt-4o-mini"},
                }
            ]
        )
        deployment = Deployment(
            model_name="auto_router/quality_router/test-router",
            litellm_params=LiteLLM_Params(
                model="auto_router/quality_router/test-router",
                quality_router_default_model="gpt-4o-mini",
            ),
            model_info={"id": "test-id"},
        )
        router.init_quality_router_deployment(deployment)
        assert "auto_router/quality_router/test-router" in router.quality_routers
