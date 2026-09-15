"""
Validate Claude Sonnet 5 model configuration entries.

Sonnet 5 ships with the gen-5 adaptive-thinking profile (adaptive thinking
always on, no extended thinking, ``effort`` defaults to ``high``), so it must
mirror the sampling-param and prefill restrictions that Fable 5 / Opus 4.8 carry
rather than the older Sonnet 4.6 behavior. The cost-map entries are also what
populate ``litellm.anthropic_models`` at import, which is what lets a bare
``claude-sonnet-5`` name resolve to the ``anthropic`` provider (and match an
``anthropic/*`` wildcard deployment).
"""

import json
import os

import pytest

from litellm.constants import BEDROCK_CONVERSE_MODELS
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

REPO_ROOT = os.path.join(os.path.dirname(__file__), "../..")

ALL_SONNET_5_VARIANTS = (
    "claude-sonnet-5",
    "anthropic.claude-sonnet-5",
    "global.anthropic.claude-sonnet-5",
    "us.anthropic.claude-sonnet-5",
    "eu.anthropic.claude-sonnet-5",
    "au.anthropic.claude-sonnet-5",
    "jp.anthropic.claude-sonnet-5",
    "vertex_ai/claude-sonnet-5",
    "vertex_ai/claude-sonnet-5@default",
    "azure_ai/claude-sonnet-5",
)


def _load_root_cost_map() -> dict:
    json_path = os.path.join(REPO_ROOT, "model_prices_and_context_window.json")
    with open(json_path) as f:
        return json.load(f)


def test_sonnet_5_present_in_bundled_backup():
    """The bundled backup is the runtime fallback (and what tests load with
    ``LITELLM_LOCAL_MODEL_COST_MAP=True``); it must carry the same entries as the
    root cost map, otherwise the model resolves on one path but not the other."""
    backup = GetModelCostMap.load_local_model_cost_map()
    for model_name in ALL_SONNET_5_VARIANTS:
        assert model_name in backup, f"Missing from backup cost map: {model_name}"


def test_sonnet_5_registered_for_bedrock_converse():
    assert "anthropic.claude-sonnet-5" in BEDROCK_CONVERSE_MODELS


@pytest.mark.parametrize(
    "cost_map",
    [_load_root_cost_map(), GetModelCostMap.load_local_model_cost_map()],
    ids=["root", "bundled_backup"],
)
def test_sonnet_5_all_variants_carry_adaptive_thinking_flag(cost_map):
    """Every Sonnet 5 entry must advertise ``supports_adaptive_thinking``.

    Adaptive-thinking detection is cost-map driven, so a single variant missing
    the flag silently sends the legacy ``thinking.type='enabled'`` shape and the
    provider 400s. This guards against a future variant being added without it."""
    variants = [k for k in cost_map if "claude-sonnet-5" in k]
    assert variants, "no claude-sonnet-5 entries found in cost map"
    missing = [k for k in variants if cost_map[k].get("supports_adaptive_thinking") is not True]
    assert not missing, f"missing supports_adaptive_thinking: {missing}"
