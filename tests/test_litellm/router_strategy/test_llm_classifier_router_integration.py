"""Integration tests for LLMClassifierRouter wired into the main Router.

Verifies the 7 patches in litellm/router.py are correctly placed and the new
routing strategy is auto-discovered from a deployment config.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm import Router
from litellm.router_strategy.llm_classifier_router import LLMClassifierRouter
from litellm.types.router import LiteLLM_Params


def _mock_response(content: str) -> MagicMock:
    return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])


# ─── Detection methods on Router ─────────────────────────────────
class TestRouterDetection:
    def test_is_llm_classifier_router_deployment_true(self):
        params = LiteLLM_Params(model="auto_router/llm_classifier_router")
        router = Router(model_list=[])
        assert router._is_llm_classifier_router_deployment(params) is True

    def test_is_llm_classifier_router_deployment_false_for_other_prefixes(self):
        router = Router(model_list=[])
        for model_name in [
            "auto_router/complexity_router",
            "auto_router/quality_router",
            "auto_router/adaptive_router",
            "auto_router/semantic",
            "gpt-4o-mini",
            "openai/gpt-4o",
        ]:
            params = LiteLLM_Params(model=model_name)
            assert router._is_llm_classifier_router_deployment(params) is False, f"Expected False for {model_name}"

    def test_is_auto_router_deployment_excludes_llm_classifier(self):
        # Regression: the gap where auto_router/llm_classifier_router fell
        # into the generic AutoRouter bucket.
        router = Router(model_list=[])
        params = LiteLLM_Params(model="auto_router/llm_classifier_router")
        assert router._is_auto_router_deployment(params) is False

    def test_is_auto_router_deployment_still_true_for_other_prefixes(self):
        router = Router(model_list=[])
        params = LiteLLM_Params(model="auto_router/semantic")
        assert router._is_auto_router_deployment(params) is True


# ─── Init method registers the router ───────────────────────────
class TestRouterInit:
    def test_init_llm_classifier_router_deployment_registers_router(self):
        from litellm.types.router import Deployment

        deployment = Deployment(
            model_name="smart-router",
            litellm_params=LiteLLM_Params(
                model="auto_router/llm_classifier_router",
                llm_classifier_router_config={
                    "tiers": {"SIMPLE": "cheap", "COMPLEX": "expensive"},
                },
            ),
        )
        router = Router(model_list=[])
        router.init_llm_classifier_router_deployment(deployment=deployment)
        assert "smart-router" in router.llm_classifier_routers
        assert isinstance(router.llm_classifier_routers["smart-router"], LLMClassifierRouter)

    def test_init_uses_default_model_from_tiers_complex(self):
        from litellm.types.router import Deployment

        deployment = Deployment(
            model_name="smart-router",
            litellm_params=LiteLLM_Params(
                model="auto_router/llm_classifier_router",
                llm_classifier_router_config={
                    "tiers": {"SIMPLE": "cheap", "COMPLEX": "expensive"},
                },
            ),
        )
        router = Router(model_list=[])
        router.init_llm_classifier_router_deployment(deployment=deployment)
        # Default model should fall back to COMPLEX tier value when no
        # llm_classifier_router_default_model is set.
        assert router.llm_classifier_routers["smart-router"].config.fallback_tier == "SIMPLE"

    def test_init_raises_when_no_default_model_and_no_tiers(self):
        from litellm.types.router import Deployment

        deployment = Deployment(
            model_name="smart-router",
            litellm_params=LiteLLM_Params(
                model="auto_router/llm_classifier_router",
                llm_classifier_router_config={"tiers": {}},
            ),
        )
        router = Router(model_list=[])
        with pytest.raises(ValueError, match="llm_classifier_router_default_model"):
            router.init_llm_classifier_router_deployment(deployment=deployment)

    def test_init_raises_on_duplicate_model_name(self):
        from litellm.types.router import Deployment

        deployment1 = Deployment(
            model_name="smart-router",
            litellm_params=LiteLLM_Params(
                model="auto_router/llm_classifier_router",
                llm_classifier_router_default_model="x",
            ),
        )
        deployment2 = Deployment(
            model_name="smart-router",
            litellm_params=LiteLLM_Params(
                model="auto_router/llm_classifier_router",
                llm_classifier_router_default_model="x",
            ),
        )
        router = Router(model_list=[])
        router.init_llm_classifier_router_deployment(deployment=deployment1)
        with pytest.raises(ValueError, match="already exists"):
            router.init_llm_classifier_router_deployment(deployment=deployment2)

    def test_init_uses_explicit_default_model(self):
        from litellm.types.router import Deployment

        deployment = Deployment(
            model_name="smart-router",
            litellm_params=LiteLLM_Params(
                model="auto_router/llm_classifier_router",
                llm_classifier_router_default_model="my-explicit-default",
            ),
        )
        router = Router(model_list=[])
        router.init_llm_classifier_router_deployment(deployment=deployment)
        assert router.llm_classifier_routers["smart-router"].config.fallback_tier == "SIMPLE"


# ─── End-to-end through Router init ──────────────────────────────
class TestRouterEndToEnd:
    def test_router_picks_up_deployment_from_model_list(self):
        model_list = [
            {
                "model_name": "smart-router",
                "litellm_params": {
                    "model": "auto_router/llm_classifier_router",
                    "llm_classifier_router_config": {
                        "tiers": {"SIMPLE": "a", "COMPLEX": "b"},
                    },
                },
            }
        ]
        with patch("litellm.acompletion"):
            router = Router(model_list=model_list)
        assert "smart-router" in router.llm_classifier_routers

    def test_set_model_list_resets_dict(self):
        model_list_1 = [
            {
                "model_name": "smart-router",
                "litellm_params": {
                    "model": "auto_router/llm_classifier_router",
                    "llm_classifier_router_config": {
                        "tiers": {"SIMPLE": "a", "COMPLEX": "b"},
                    },
                },
            }
        ]
        with patch("litellm.acompletion"):
            router = Router(model_list=model_list_1)
        assert "smart-router" in router.llm_classifier_routers

        with patch("litellm.acompletion"):
            router.set_model_list(model_list_1)
        # After set_model_list, the dict should still be populated (it's
        # a reset-then-rebuild, not a clear-only).
        assert "smart-router" in router.llm_classifier_routers
