"""
Registry tests for Together AI models in the bundled model cost map.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../../.."))

import pytest

from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

INKLING_KEYS = [
    "together_ai/thinkingmachines/Inkling",
    "together_ai/thinkingmachines/inkling",
]


class TestInklingModelRegistry:
    """Tests that Thinking Machines Lab's Inkling is registered under together_ai."""

    @pytest.fixture(autouse=True)
    def model_cost_map(self):
        """Load directly from the bundled backup so tests don't depend on remote fetch."""
        return GetModelCostMap.load_local_model_cost_map()

    @pytest.mark.parametrize("model", INKLING_KEYS)
    def test_inkling_in_model_cost_map(self, model, model_cost_map):
        assert model in model_cost_map, f"{model} not found in model_cost"

    @pytest.mark.parametrize("model", INKLING_KEYS)
    def test_inkling_pricing(self, model, model_cost_map):
        """Inkling pricing should match Together AI's published serverless rates."""
        model_info = model_cost_map[model]
        assert model_info["input_cost_per_token"] == pytest.approx(1e-06)
        assert model_info["output_cost_per_token"] == pytest.approx(4.05e-06)
        assert model_info["cache_read_input_token_cost"] == pytest.approx(1.7e-07)

    @pytest.mark.parametrize("model", INKLING_KEYS)
    def test_inkling_context_window(self, model, model_cost_map):
        """Inkling should expose a 524K token context window."""
        assert model_cost_map[model]["max_input_tokens"] == 524288

    @pytest.mark.parametrize("model", INKLING_KEYS)
    def test_inkling_capabilities(self, model, model_cost_map):
        """Inkling is a multimodal reasoning model with tools and json schema support."""
        model_info = model_cost_map[model]
        assert model_info.get("supports_function_calling") is True
        assert model_info.get("supports_tool_choice") is True
        assert model_info.get("supports_reasoning") is True
        assert model_info.get("supports_vision") is True
        assert model_info.get("supports_audio_input") is True
        assert model_info.get("supports_response_schema") is True

    @pytest.mark.parametrize("model", INKLING_KEYS)
    def test_inkling_provider(self, model, model_cost_map):
        assert model_cost_map[model]["litellm_provider"] == "together_ai"
