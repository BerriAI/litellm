"""
Validate Claude Fable 5 and Claude Fable 5.1 model configuration entries.

Fable 5 is a new tier above Opus ($10/$50 per MTok) with the same adaptive-only
API surface as Opus 4.7/4.8. The cost-map entries below are what make the model
resolvable across Anthropic, Bedrock, Vertex AI, and Azure AI (Microsoft
Foundry), and the ``supports_adaptive_thinking`` flag is what makes LiteLLM send
``thinking.type='adaptive'`` instead of the legacy ``enabled``/``budget_tokens``
shape, which Fable 5 rejects with a 400.
"""

import json
import os

import pytest

from litellm.constants import BEDROCK_CONVERSE_MODELS
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

REPO_ROOT = os.path.join(os.path.dirname(__file__), "../..")


def _load_root_cost_map() -> dict:
    json_path = os.path.join(REPO_ROOT, "model_prices_and_context_window.json")
    with open(json_path) as f:
        return json.load(f)


def test_fable_5_registered_for_bedrock_converse():
    assert "anthropic.claude-fable-5" in BEDROCK_CONVERSE_MODELS


@pytest.mark.parametrize(
    "model",
    [
        "claude-fable-5",
        "anthropic/claude-fable-5",
        "anthropic.claude-fable-5",
        "bedrock/us.anthropic.claude-fable-5",
        "bedrock/invoke/eu.anthropic.claude-fable-5",
        "bedrock/global.anthropic.claude-fable-5",
        "vertex_ai/claude-fable-5",
        "azure_ai/claude-fable-5",
    ],
)
def test_adaptive_thinking_detected_for_fable_5(local_model_cost_map, model):
    """Provider-routed ids must resolve to a flagged entry so ``reasoning_effort``
    maps to ``thinking.type='adaptive'`` + ``output_config.effort``."""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    assert AnthropicModelInfo._is_adaptive_thinking_model(model, "anthropic") is True


FABLE_5_1_VARIANTS = (
    "claude-fable-5-1",
    "anthropic.claude-fable-5-1",
    "global.anthropic.claude-fable-5-1",
    "us.anthropic.claude-fable-5-1",
    "eu.anthropic.claude-fable-5-1",
    "vertex_ai/claude-fable-5-1",
    "vertex_ai/claude-fable-5-1@default",
    "azure_ai/claude-fable-5-1",
)


def test_fable_5_1_present_in_bundled_backup():
    backup = GetModelCostMap.load_local_model_cost_map()
    root = _load_root_cost_map()
    for model_name in FABLE_5_1_VARIANTS:
        assert model_name in backup, f"Missing from backup cost map: {model_name}"
        assert backup[model_name] == root[model_name], model_name


def test_fable_5_1_registered_for_bedrock_converse():
    assert "anthropic.claude-fable-5-1" in BEDROCK_CONVERSE_MODELS


@pytest.mark.parametrize(
    "model",
    [
        "claude-fable-5-1",
        "anthropic/claude-fable-5-1",
        "anthropic.claude-fable-5-1",
        "bedrock/us.anthropic.claude-fable-5-1",
        "bedrock/invoke/eu.anthropic.claude-fable-5-1",
        "bedrock/global.anthropic.claude-fable-5-1",
        "vertex_ai/claude-fable-5-1",
        "azure_ai/claude-fable-5-1",
    ],
)
def test_adaptive_thinking_detected_for_fable_5_1(local_model_cost_map, model):
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    assert AnthropicModelInfo._is_adaptive_thinking_model(model, "anthropic") is True
