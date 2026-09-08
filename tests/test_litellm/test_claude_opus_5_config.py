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
def test_opus_5_bedrock_entries_declare_no_effort_ceiling(model_name):
    """Bedrock accepts every effort level for Opus 5, so no clamp belongs here.

    Opus 4.7/4.8 carry ``bedrock_output_config_effort_ceiling: "xhigh"``, which
    is what ``normalize_bedrock_opus_output_config_effort`` reads to rewrite a
    caller's effort down. Verified against Bedrock on 2026-07-24 that
    ``output_config.effort="max"`` returns 200 for the Opus 5 profiles, so the
    ceiling is deliberately absent; adding one back would silently downgrade
    requests.

    This asserts the cost-map entry rather than calling the normalizer because
    ``_BEDROCK_OUTPUT_CONFIG_EFFORT_ORDER`` currently ranks ``max`` (3) below
    ``xhigh`` (4), so an ``xhigh`` ceiling never clamps ``max`` and a behavioral
    assertion would pass either way. Keeping the entry clean means Opus 5 stays
    correct once that ordering is fixed."""
    info = _load_root_cost_map()[model_name]
    assert "bedrock_output_config_effort_ceiling" not in info


@pytest.mark.parametrize("model_name", BEDROCK_OPUS_5_VARIANTS)
def test_opus_5_bedrock_rejects_strict_tools(model_name, local_model_cost_map):
    """Bedrock Converse routes Opus through a validator that rejects
    ``toolSpec.strict`` (``tools.0.custom.strict: Extra inputs are not
    permitted``), same as Opus 4.7/4.8; verified against Bedrock on 2026-07-24.
    Without the flag LiteLLM forwards ``strict`` and every tool call 400s."""
    from litellm.llms.bedrock.common_utils import bedrock_converse_supports_strict_tools

    assert bedrock_converse_supports_strict_tools(model_name) is False


def test_opus_5_prompt_cache_minimum_is_512(local_model_cost_map):
    """Opus 5 halves the cacheable-prefix minimum (Opus 4.8 is 1024).

    The router's prompt-caching deployment check reads this value, so a stale
    1024 would route prompts of 512-1023 tokens away from a warm Opus 5
    deployment even though they cache fine."""
    from litellm.utils import get_prompt_cache_min_tokens

    assert get_prompt_cache_min_tokens(model="claude-opus-5") == 512
    assert get_prompt_cache_min_tokens(model="us.anthropic.claude-opus-5") == 512


def test_opus_5_supports_fast_mode(local_model_cost_map):
    """Fast mode is Opus 5 on the first-party API at $10 / $50 per MTok, i.e. 2x
    base. ``supports_speed`` gates whether ``speed="fast"`` is forwarded at all,
    and ``provider_specific_entry.fast`` is what prices the response."""
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig
    from litellm.llms.anthropic.cost_calculation import (
        cost_per_token as anthropic_cost_per_token,
    )
    from litellm.types.utils import Usage

    assert (
        AnthropicConfig._model_supports_speed_param("claude-opus-5", "anthropic") is True
    )

    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    usage.speed = "fast"
    prompt_cost, completion_cost = anthropic_cost_per_token(
        model="claude-opus-5", usage=usage
    )
    assert prompt_cost == pytest.approx(1000 * 5e-06 * 2.0)
    assert completion_cost == pytest.approx(500 * 2.5e-05 * 2.0)


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
    missing = [
        k for k in variants if cost_map[k].get("supports_adaptive_thinking") is not True
    ]
    assert not missing, f"missing supports_adaptive_thinking: {missing}"


@pytest.mark.parametrize(
    "cost_map",
    [_load_root_cost_map(), GetModelCostMap.load_local_model_cost_map()],
    ids=["root", "bundled_backup"],
)
def test_opus_5_all_variants_carry_512_token_cache_minimum(cost_map):
    variants = [k for k in cost_map if "claude-opus-5" in k]
    assert variants, "no claude-opus-5 entries found in cost map"
    wrong = {
        k: cost_map[k].get("prompt_cache_min_tokens")
        for k in variants
        if cost_map[k].get("prompt_cache_min_tokens") != 512
    }
    assert not wrong, f"prompt_cache_min_tokens must be 512: {wrong}"
