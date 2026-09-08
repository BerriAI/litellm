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







def test_fable_5_geo_multiplier_without_fast_mode():
    """First-party ``inference_geo='us'`` carries the 1.1x premium, but unlike
    the Opus line there is no fast-mode variant for Fable 5; a ``fast`` key
    here would silently misprice ``speed='fast'`` requests."""
    model_data = _load_root_cost_map()
    entry = model_data["claude-fable-5"]["provider_specific_entry"]
    assert entry == {"us": 1.1}


def test_fable_5_present_in_bundled_backup():
    """The bundled backup is the runtime fallback (and what tests load with
    ``LITELLM_LOCAL_MODEL_COST_MAP=True``) — it must carry the same entries as
    the root cost map, otherwise the model resolves on one path but not the
    other."""
    backup = GetModelCostMap.load_local_model_cost_map()
    root = _load_root_cost_map()
    for model_name in (
        "claude-fable-5",
        "anthropic.claude-fable-5",
        "global.anthropic.claude-fable-5",
        "us.anthropic.claude-fable-5",
        "eu.anthropic.claude-fable-5",
        "vertex_ai/claude-fable-5",
        "vertex_ai/claude-fable-5@default",
        "azure_ai/claude-fable-5",
    ):
        assert model_name in backup, f"Missing from backup cost map: {model_name}"
        assert backup[model_name] == root[model_name], model_name


def test_fable_5_registered_for_bedrock_converse():
    assert "anthropic.claude-fable-5" in BEDROCK_CONVERSE_MODELS




@pytest.mark.parametrize(
    "cost_map",
    [_load_root_cost_map(), GetModelCostMap.load_local_model_cost_map()],
    ids=["root", "bundled_backup"],
)
def test_fable_5_all_variants_carry_adaptive_thinking_flag(cost_map):
    """Every Fable 5 entry must advertise ``supports_adaptive_thinking``.

    Adaptive-thinking detection is cost-map driven, so a single variant missing
    the flag silently sends the legacy ``thinking.type='enabled'`` shape and the
    provider 400s (issue #29188 for the Opus 4.8 equivalent). Fable 5 is even
    stricter than Opus 4.8: an explicit ``thinking.type='disabled'`` also 400s,
    so adaptive is the only valid thinking shape LiteLLM can emit for it."""
    variants = [k for k in cost_map if "claude-fable-5" in k]
    assert variants, "no claude-fable-5 entries found in cost map"
    missing = [
        k for k in variants if cost_map[k].get("supports_adaptive_thinking") is not True
    ]
    assert not missing, f"missing supports_adaptive_thinking: {missing}"


@pytest.mark.parametrize(
    "cost_map",
    [_load_root_cost_map(), GetModelCostMap.load_local_model_cost_map()],
    ids=["root", "bundled_backup"],
)
def test_fable_5_all_variants_carry_thinking_always_on_flag(cost_map):
    """Every Fable 5 entry must advertise ``thinking_always_on``.

    The flag drives the Anthropic transformations to omit an explicit
    ``thinking.type='disabled'``, which Fable 5 rejects with a 400; a variant
    missing the flag forwards the param verbatim and the provider 400s."""
    variants = [k for k in cost_map if "claude-fable-5" in k]
    assert variants, "no claude-fable-5 entries found in cost map"
    missing = [k for k in variants if cost_map[k].get("thinking_always_on") is not True]
    assert not missing, f"missing thinking_always_on: {missing}"


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




@pytest.mark.parametrize(
    "cost_map",
    [_load_root_cost_map(), GetModelCostMap.load_local_model_cost_map()],
    ids=["root", "bundled_backup"],
)
def test_fable_5_1_cache_reads_cost_a_quarter_of_fable_5(cost_map):
    """Fable 5.1 prices cache hits at 0.025x base input instead of the usual
    0.1x, so copying Fable 5's cache-read price overcharges every cache hit 4x."""
    for model_name in FABLE_5_1_VARIANTS:
        info = cost_map[model_name]
        geo_premium = model_name.startswith(("us.", "eu."))
        expected = 2.75e-07 if geo_premium else 2.5e-07
        assert info["cache_read_input_token_cost"] == expected, model_name
        assert info["cache_read_input_token_cost"] == pytest.approx(
            info["input_cost_per_token"] * 0.025
        ), model_name






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


@pytest.mark.parametrize(
    "cost_map",
    [_load_root_cost_map(), GetModelCostMap.load_local_model_cost_map()],
    ids=["root", "bundled_backup"],
)
def test_sampling_params_flag_on_all_models_that_removed_them(cost_map):
    """Fable 5 and Opus 4.7/4.8 reject ``top_p``/``top_k``/``temperature != 1``;
    the drop/raise gating is cost-map driven, so every variant must carry an
    explicit ``supports_sampling_params: false``. The perplexity route is
    exempt: it is OpenAI-compatible and maps sampling params upstream."""
    variants = [
        k
        for k in cost_map
        if any(v in k for v in ("claude-fable-5", "claude-opus-4-7", "claude-opus-4-8"))
        and not k.startswith("perplexity/")
    ]
    assert variants, "no matching entries found in cost map"
    missing = [
        k for k in variants if cost_map[k].get("supports_sampling_params") is not False
    ]
    assert not missing, f"missing supports_sampling_params=false: {missing}"
