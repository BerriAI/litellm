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

    spec entry shape: {"model_name": str, "quality_tier": Optional[int]}
    If quality_tier is None, the deployment is created without
    `litellm_routing_preferences`.
    """
    out: List[Dict[str, Any]] = []
    for entry in spec:
        model_info: Dict[str, Any] = {"id": f"id-{entry['model_name']}"}
        if entry.get("quality_tier") is not None:
            model_info["litellm_routing_preferences"] = {
                "quality_tier": entry["quality_tier"]
            }
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
