"""
Validate Claude Opus 4.8 model configuration entries.

Regression coverage for the wildcard-routing failure where a bare model name
(``claude-opus-4-8``) could not match an ``anthropic/*`` deployment because
LiteLLM could not infer its provider — the model was simply missing from the
model cost map, so ``get_llm_provider`` raised and the router returned
"no healthy deployments for this model". The fix is the cost-map entries added
for Anthropic, Bedrock, Vertex AI, and Azure AI; those entries are what populate
``litellm.anthropic_models`` at import time, which is what the bare-name lookup
in ``get_llm_provider`` consumes.
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


def test_opus_4_8_fast_mode_multiplier():
    """Opus 4.8 dropped fast-mode pricing to 2x base ($10/$50 per MTok);
    Opus 4.7 was 6x ($30/$150)."""
    model_data = _load_root_cost_map()
    entry = model_data["claude-opus-4-8"]["provider_specific_entry"]
    assert entry["us"] == 1.1
    assert entry["fast"] == 2.0


def test_opus_4_8_registered_for_bedrock_converse():
    assert "anthropic.claude-opus-4-8" in BEDROCK_CONVERSE_MODELS


@pytest.mark.parametrize(
    "cost_map",
    [_load_root_cost_map(), GetModelCostMap.load_local_model_cost_map()],
    ids=["root", "bundled_backup"],
)
def test_opus_4_8_all_variants_carry_adaptive_thinking_flag(cost_map):
    """Every Opus 4.8 entry must advertise ``supports_adaptive_thinking``.

    Adaptive-thinking detection is cost-map driven, so a single variant missing
    the flag silently sends the legacy ``thinking.type='enabled'`` shape and the
    provider 400s (issue #29188, which the Bedrock/Vertex/Azure variants hit
    because only the bare ``claude-opus-4-8`` entry carried the flag). This guards
    against a future variant being added without it."""
    variants = [k for k in cost_map if "claude-opus-4-8" in k]
    assert variants, "no claude-opus-4-8 entries found in cost map"
    missing = [
        k for k in variants if cost_map[k].get("supports_adaptive_thinking") is not True
    ]
    assert not missing, f"missing supports_adaptive_thinking: {missing}"
