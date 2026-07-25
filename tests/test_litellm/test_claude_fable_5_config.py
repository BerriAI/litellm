"""
Validate Claude Fable 5 model configuration entries.

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

import litellm
from litellm.constants import BEDROCK_CONVERSE_MODELS
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

REPO_ROOT = os.path.join(os.path.dirname(__file__), "../..")


def _load_root_cost_map() -> dict:
    json_path = os.path.join(REPO_ROOT, "model_prices_and_context_window.json")
    with open(json_path) as f:
        return json.load(f)


@pytest.fixture
def local_model_cost_map(monkeypatch):
    """Force the bundled backup cost map so assertions don't depend on the
    network-fetched ``main`` copy (which lags this branch until merge)."""
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()


def test_fable_5_model_pricing_and_capabilities():
    model_data = _load_root_cost_map()

    expected_models = [
        ("claude-fable-5", "anthropic"),
        ("anthropic.claude-fable-5", "bedrock_converse"),
        ("vertex_ai/claude-fable-5", "vertex_ai-anthropic_models"),
        # Unlike Opus 4.8 (200k on Foundry), Fable 5 has the full 1M context
        # window on Microsoft Foundry.
        ("azure_ai/claude-fable-5", "azure_ai"),
    ]

    for model_name, provider in expected_models:
        assert model_name in model_data, f"Missing model entry: {model_name}"
        info = model_data[model_name]

        assert info["litellm_provider"] == provider
        assert info["mode"] == "chat"
        assert info["max_input_tokens"] == 1000000
        assert info["max_output_tokens"] == 128000
        assert info["max_tokens"] == 128000

        # $10 / $50 per MTok (2x Opus 4.8), with the standard 1.25x 5m
        # cache-write, 2x 1h cache-write, and 0.1x cache-read multipliers.
        assert info["input_cost_per_token"] == 1e-05
        assert info["output_cost_per_token"] == 5e-05
        assert info["cache_creation_input_token_cost"] == 1.25e-05
        assert info["cache_creation_input_token_cost_above_1hr"] == 2e-05
        assert info["cache_read_input_token_cost"] == 1e-06

        # Flat-rate across the full 1M context window.
        assert "input_cost_per_token_above_200k_tokens" not in info
        assert "output_cost_per_token_above_200k_tokens" not in info

        assert info["supports_assistant_prefill"] is False
        assert info["supports_function_calling"] is True
        assert info["supports_prompt_caching"] is True
        assert info["supports_reasoning"] is True
        assert info["supports_tool_choice"] is True
        assert info["supports_vision"] is True
        assert info["supports_xhigh_reasoning_effort"] is True
        assert info["supports_max_reasoning_effort"] is True


def test_fable_5_bedrock_regional_model_pricing():
    model_data = _load_root_cost_map()

    # Fable 5 launched with us/eu geo inference profiles plus a global profile
    # (no au/apac/jp). Global uses base pricing; geo profiles carry the
    # standard 10% regional premium.
    expected_models = {
        "global.anthropic.claude-fable-5": {
            "input_cost_per_token": 1e-05,
            "output_cost_per_token": 5e-05,
            "cache_creation_input_token_cost": 1.25e-05,
            "cache_read_input_token_cost": 1e-06,
        },
        "us.anthropic.claude-fable-5": {
            "input_cost_per_token": 1.1e-05,
            "output_cost_per_token": 5.5e-05,
            "cache_creation_input_token_cost": 1.375e-05,
            "cache_read_input_token_cost": 1.1e-06,
        },
        "eu.anthropic.claude-fable-5": {
            "input_cost_per_token": 1.1e-05,
            "output_cost_per_token": 5.5e-05,
            "cache_creation_input_token_cost": 1.375e-05,
            "cache_read_input_token_cost": 1.1e-06,
        },
    }

    for model_name, expected in expected_models.items():
        assert model_name in model_data, f"Missing model entry: {model_name}"
        info = model_data[model_name]
        assert info["litellm_provider"] == "bedrock_converse"
        assert info["max_input_tokens"] == 1000000
        assert info["max_output_tokens"] == 128000
        assert info["bedrock_output_config_effort_ceiling"] == "xhigh"
        for key, value in expected.items():
            assert info[key] == value


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


def test_fable_5_provider_resolves_via_model_info(local_model_cost_map):
    info = litellm.get_model_info(model="claude-fable-5")
    assert info["litellm_provider"] == "anthropic"
    assert info["max_input_tokens"] == 1000000
    assert info["max_output_tokens"] == 128000


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


@pytest.mark.parametrize(
    "cost_map",
    [_load_root_cost_map(), GetModelCostMap.load_local_model_cost_map()],
    ids=["root", "bundled_backup"],
)
def test_sampling_params_flag_on_all_models_that_removed_them(cost_map):
    """Fable 5 and Opus 4.7/4.8 reject ``top_p``/``top_k``/``temperature != 1``;
    the drop/raise gating is cost-map driven, so every variant must carry an
    explicit ``supports_sampling_params: false``. The perplexity and openrouter
    routes are exempt: they are OpenAI-compatible and map sampling params
    upstream rather than going through the Anthropic gating."""
    variants = [
        k
        for k in cost_map
        if any(v in k for v in ("claude-fable-5", "claude-opus-4-7", "claude-opus-4-8"))
        and not k.startswith(("perplexity/", "openrouter/"))
    ]
    assert variants, "no matching entries found in cost map"
    missing = [
        k for k in variants if cost_map[k].get("supports_sampling_params") is not False
    ]
    assert not missing, f"missing supports_sampling_params=false: {missing}"
