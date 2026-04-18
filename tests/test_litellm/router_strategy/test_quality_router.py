"""
Tests for the QualityRouter.

Covers:
- Tier index construction from `model_info.litellm_routing_preferences`.
- Quality-tier resolution (exact, round-up, default fallback).
- Pre-routing hook end-to-end (classification → quality tier → model).
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
        "capabilities": Optional[List[str]],   # default: omitted
    }
    If quality_tier is None, the deployment is created without
    `litellm_routing_preferences`.
    """
    out: List[Dict[str, Any]] = []
    for entry in spec:
        model_info: Dict[str, Any] = {"id": f"id-{entry['model_name']}"}
        if entry.get("quality_tier") is not None:
            prefs: Dict[str, Any] = {"quality_tier": entry["quality_tier"]}
            if "capabilities" in entry:
                prefs["capabilities"] = entry["capabilities"]
            model_info["litellm_routing_preferences"] = prefs
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

        with pytest.raises(ValueError, match="sonnet"):
            QualityRouter(
                model_name="qr",
                litellm_router_instance=router,
                default_model="haiku",
                quality_router_config={"available_models": ["haiku", "sonnet"]},
            )


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

    def test_falls_back_to_default_when_nothing_higher_exists(self):
        # Only tier 1 available. Asking for tier 4 should fall back to default.
        spec = [{"model_name": "haiku", "quality_tier": 1}]
        router = MagicMock()
        router.model_list = _make_model_list(spec)

        qr = QualityRouter(
            model_name="qr",
            litellm_router_instance=router,
            default_model="emergency-default",
            quality_router_config={"available_models": ["haiku"]},
        )

        assert qr._resolve_model_for_quality_tier(4) == "emergency-default"


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


# ─── Capabilities ───────────────────────────────────────────────────────────


@pytest.fixture
def capability_router():
    """
    Router with mixed capabilities at each tier:
      tier 1: haiku-text (no caps), haiku-vision (vision)
      tier 2: sonnet-text (no caps), sonnet-vision (vision, function_calling)
      tier 3: opus-vision (vision, function_calling, json_mode)
    """
    spec = [
        {"model_name": "haiku-text", "quality_tier": 1, "capabilities": []},
        {"model_name": "haiku-vision", "quality_tier": 1, "capabilities": ["vision"]},
        {"model_name": "sonnet-text", "quality_tier": 2, "capabilities": []},
        {
            "model_name": "sonnet-vision",
            "quality_tier": 2,
            "capabilities": ["vision", "function_calling"],
        },
        {
            "model_name": "opus-vision",
            "quality_tier": 3,
            "capabilities": ["vision", "function_calling", "json_mode"],
        },
    ]
    router = MagicMock()
    router.model_list = _make_model_list(spec)
    return QualityRouter(
        model_name="qr",
        litellm_router_instance=router,
        default_model="haiku-text",
        quality_router_config={
            "available_models": [
                "haiku-text",
                "haiku-vision",
                "sonnet-text",
                "sonnet-vision",
                "opus-vision",
            ],
        },
    )


class TestCapabilities:
    def test_index_records_capabilities(self, capability_router):
        assert capability_router._model_capabilities["haiku-text"] == frozenset()
        assert capability_router._model_capabilities["haiku-vision"] == frozenset(
            {"vision"}
        )
        assert capability_router._model_capabilities["opus-vision"] == frozenset(
            {"vision", "function_calling", "json_mode"}
        )

    def test_no_required_capabilities_picks_first_in_tier(self, capability_router):
        # tier 2, no required caps → first registered model at tier 2.
        assert capability_router._resolve_model_for_quality_tier(2) == "sonnet-text"

    def test_required_capabilities_filter_within_tier(self, capability_router):
        # tier 2 with vision → must pick sonnet-vision over sonnet-text.
        assert (
            capability_router._resolve_model_for_quality_tier(
                2, required_capabilities={"vision"}
            )
            == "sonnet-vision"
        )

    def test_round_up_when_no_capable_model_at_tier(self, capability_router):
        # tier 1 with function_calling: nothing at tier 1 has it → round up to
        # tier 2 (sonnet-vision).
        assert (
            capability_router._resolve_model_for_quality_tier(
                1, required_capabilities={"function_calling"}
            )
            == "sonnet-vision"
        )

    def test_raises_when_no_model_satisfies_capabilities(self, capability_router):
        # No model anywhere has "audio".
        with pytest.raises(ValueError, match="audio"):
            capability_router._resolve_model_for_quality_tier(
                1, required_capabilities={"audio"}
            )

    def test_default_model_used_only_if_it_satisfies_caps(self):
        # Build a router whose default model has NO capabilities, then ask for
        # a capability that nothing satisfies. Must raise rather than silently
        # routing to the default.
        spec = [{"model_name": "only-tier-1", "quality_tier": 1, "capabilities": []}]
        router = MagicMock()
        router.model_list = _make_model_list(spec)
        qr = QualityRouter(
            model_name="qr",
            litellm_router_instance=router,
            default_model="only-tier-1",
            quality_router_config={"available_models": ["only-tier-1"]},
        )
        with pytest.raises(ValueError, match="vision"):
            qr._resolve_model_for_quality_tier(1, required_capabilities={"vision"})

    @pytest.mark.asyncio
    async def test_hook_reads_litellm_capabilities_from_request_kwargs(
        self, capability_router
    ):
        # Simple "hi" → tier 1 by complexity; with vision required, must pick
        # haiku-vision (the tier-1 model that has vision).
        messages = [{"role": "user", "content": "hi"}]
        resp = await capability_router.async_pre_routing_hook(
            model="qr",
            request_kwargs={"litellm_capabilities": ["vision"]},
            messages=messages,
        )
        assert resp is not None
        assert resp.model == "haiku-vision"

    @pytest.mark.asyncio
    async def test_hook_with_no_capabilities_kwarg_behaves_as_before(
        self, capability_router
    ):
        messages = [{"role": "user", "content": "hi"}]
        resp = await capability_router.async_pre_routing_hook(
            model="qr",
            request_kwargs={},
            messages=messages,
        )
        assert resp is not None
        # tier 1, no caps required → first registered model at tier 1.
        assert resp.model == "haiku-text"


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
        assert decision["required_capabilities"] == []

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

    @pytest.mark.asyncio
    async def test_decision_metadata_includes_required_capabilities(
        self, capability_router
    ):
        request_kwargs: Dict[str, Any] = {
            "litellm_capabilities": ["vision"],
        }

        await capability_router.async_pre_routing_hook(
            model="qr",
            request_kwargs=request_kwargs,
            messages=[{"role": "user", "content": "hi"}],
        )

        decision = request_kwargs["metadata"]["quality_router_decision"]
        assert decision["routed_model"] == "haiku-vision"
        assert decision["required_capabilities"] == ["vision"]


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
                    "routed_model": "haiku-vision",
                    "quality_tier": 1,
                    "complexity_tier": "SIMPLE",
                    "required_capabilities": ["vision"],
                }
            }
        }

        await router.set_response_headers(
            response=response,
            model_group="qr",
            request_kwargs=request_kwargs,
        )

        headers = response._hidden_params["additional_headers"]
        assert headers["x-litellm-quality-router-model"] == "haiku-vision"
        assert headers["x-litellm-quality-router-tier"] == "1"
        assert headers["x-litellm-quality-router-complexity"] == "SIMPLE"
        # Existing x-litellm-model-group behavior is unchanged.
        assert headers["x-litellm-model-group"] == "qr"

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
        assert "x-litellm-quality-router-complexity" not in headers
