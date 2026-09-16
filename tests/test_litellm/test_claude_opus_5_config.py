"""
Validate Claude Opus 5 model configuration entries.

Opus 5 carries Opus 4.8's pricing ($5 / $25 per MTok) and the gen-5 adaptive
thinking profile, but differs from 4.8 in two ways that are behavior-bearing in
LiteLLM: the cacheable-prefix minimum drops to 512 tokens, and Bedrock's Opus 5
validator accepts the full effort ladder, so the entries must not carry the
``bedrock_output_config_effort_ceiling`` that silently clamps ``max`` to
``xhigh`` on 4.8. The cost-map entries are also what populate
``litellm.anthropic_models`` at import, which is what lets a bare
``claude-opus-5`` name resolve to the ``anthropic`` provider (and match an
``anthropic/*`` wildcard deployment).
"""

import json
import os

import pytest

from litellm.constants import BEDROCK_CONVERSE_MODELS
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

REPO_ROOT = os.path.join(os.path.dirname(__file__), "../..")

ALL_OPUS_5_VARIANTS = (
    "claude-opus-5",
    "anthropic.claude-opus-5",
    "global.anthropic.claude-opus-5",
    "us.anthropic.claude-opus-5",
    "eu.anthropic.claude-opus-5",
    "au.anthropic.claude-opus-5",
    "jp.anthropic.claude-opus-5",
    "vertex_ai/claude-opus-5",
    "vertex_ai/claude-opus-5@default",
    "azure_ai/claude-opus-5",
)

BEDROCK_OPUS_5_VARIANTS = (
    "anthropic.claude-opus-5",
    "global.anthropic.claude-opus-5",
    "us.anthropic.claude-opus-5",
    "eu.anthropic.claude-opus-5",
    "au.anthropic.claude-opus-5",
    "jp.anthropic.claude-opus-5",
)


def _load_root_cost_map() -> dict:
    json_path = os.path.join(REPO_ROOT, "model_prices_and_context_window.json")
    with open(json_path) as f:
        return json.load(f)


@pytest.mark.parametrize("model_name", BEDROCK_OPUS_5_VARIANTS)
def test_opus_5_bedrock_rejects_strict_tools(model_name, local_model_cost_map):
    """Bedrock Converse routes Opus through a validator that rejects
    ``toolSpec.strict`` (``tools.0.custom.strict: Extra inputs are not
    permitted``), same as Opus 4.7/4.8; verified against Bedrock on 2026-07-24.
    Without the flag LiteLLM forwards ``strict`` and every tool call 400s."""
    from litellm.llms.bedrock.common_utils import bedrock_converse_supports_strict_tools

    assert bedrock_converse_supports_strict_tools(model_name) is False


def test_opus_5_present_in_bundled_backup():
    """The bundled backup is the runtime fallback (and what tests load with
    ``LITELLM_LOCAL_MODEL_COST_MAP=True``); it must carry the same entries as the
    root cost map, otherwise the model resolves on one path but not the other."""
    backup = GetModelCostMap.load_local_model_cost_map()
    for model_name in ALL_OPUS_5_VARIANTS:
        assert model_name in backup, f"Missing from backup cost map: {model_name}"


def test_opus_5_registered_for_bedrock_converse():
    assert "anthropic.claude-opus-5" in BEDROCK_CONVERSE_MODELS


@pytest.mark.parametrize(
    "cost_map",
    [_load_root_cost_map(), GetModelCostMap.load_local_model_cost_map()],
    ids=["root", "bundled_backup"],
)
def test_opus_5_all_variants_carry_adaptive_thinking_flag(cost_map):
    """Every Opus 5 entry must advertise ``supports_adaptive_thinking``.

    Adaptive-thinking detection is cost-map driven, so a single variant missing
    the flag silently sends the legacy ``thinking.type='enabled'`` shape, which
    Opus 5 rejects with a 400."""
    variants = [k for k in cost_map if "claude-opus-5" in k]
    assert variants, "no claude-opus-5 entries found in cost map"
    missing = [k for k in variants if cost_map[k].get("supports_adaptive_thinking") is not True]
    assert not missing, f"missing supports_adaptive_thinking: {missing}"
