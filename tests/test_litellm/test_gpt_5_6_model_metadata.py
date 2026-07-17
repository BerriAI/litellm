import json
from pathlib import Path

import pytest

from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

GPT_5_6_MODELS = ("gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")

STANDARD_PRICING = {
    "gpt-5.6": (5e-06, 3e-05, 5e-07, 6.25e-06),
    "gpt-5.6-sol": (5e-06, 3e-05, 5e-07, 6.25e-06),
    "gpt-5.6-terra": (2.5e-06, 1.5e-05, 2.5e-07, 3.125e-06),
    "gpt-5.6-luna": (1e-06, 6e-06, 1e-07, 1.25e-06),
}


@pytest.mark.parametrize("model", GPT_5_6_MODELS)
def test_openai_gpt_5_6_model_info(model):
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    with open(json_path) as f:
        model_cost = json.load(f)

    info = model_cost.get(model)
    assert info is not None, f"{model} not found in model_prices_and_context_window.json"

    assert info["litellm_provider"] == "openai"
    assert info["mode"] == "chat"

    input_cost, output_cost, cache_read_cost, cache_write_cost = STANDARD_PRICING[model]
    assert info["input_cost_per_token"] == input_cost
    assert info["output_cost_per_token"] == output_cost
    assert info["cache_read_input_token_cost"] == cache_read_cost
    assert info["cache_creation_input_token_cost"] == cache_write_cost
    assert info["cache_creation_input_token_cost"] == pytest.approx(input_cost * 1.25)

    assert info["input_cost_per_token_above_272k_tokens"] == pytest.approx(input_cost * 2)
    assert info["output_cost_per_token_above_272k_tokens"] == pytest.approx(output_cost * 1.5)
    assert info["cache_read_input_token_cost_above_272k_tokens"] == pytest.approx(cache_read_cost * 2)

    assert info["max_input_tokens"] == 1050000
    assert info["max_output_tokens"] == 128000
    assert info["max_tokens"] == 128000

    assert info["supports_function_calling"] is True
    assert info["supports_prompt_caching"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_vision"] is True
    assert info["supports_web_search"] is True
    assert info["supports_none_reasoning_effort"] is True
    assert info["supports_xhigh_reasoning_effort"] is True
    assert info["supports_minimal_reasoning_effort"] is False

    assert info["supported_endpoints"] == ["/v1/chat/completions", "/v1/batch", "/v1/responses"]
    assert info["supported_modalities"] == ["text", "image"]
    assert info["supported_output_modalities"] == ["text"]

    routed_model, provider, _, _ = get_llm_provider(model=f"openai/{model}")
    assert routed_model == model
    assert provider == "openai"


AZURE_GLOBAL_MODELS = (
    "azure/gpt-5.6",
    "azure/gpt-5.6-sol",
    "azure/gpt-5.6-terra",
    "azure/gpt-5.6-luna",
)

AZURE_REGIONAL_MODELS = tuple(
    f"azure/{region}/{tier}"
    for region in ("us", "eu")
    for tier in ("gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")
)


def _tier_key(azure_model):
    return azure_model.split("/")[-1]


def _load_main():
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    with open(json_path) as f:
        return json.load(f)


@pytest.mark.parametrize("model", AZURE_GLOBAL_MODELS)
def test_azure_gpt_5_6_global_model_info(model):
    model_cost = _load_main()
    info = model_cost.get(model)
    assert info is not None, f"{model} not found in model_prices_and_context_window.json"

    assert info["litellm_provider"] == "azure"
    assert info["mode"] == "chat"

    input_cost, output_cost, cache_read_cost, _ = STANDARD_PRICING[_tier_key(model)]
    assert info["input_cost_per_token"] == input_cost
    assert info["output_cost_per_token"] == output_cost
    assert info["cache_read_input_token_cost"] == cache_read_cost

    assert info["input_cost_per_token_above_272k_tokens"] == pytest.approx(input_cost * 2)
    assert info["output_cost_per_token_above_272k_tokens"] == pytest.approx(output_cost * 1.5)
    assert info["input_cost_per_token_priority"] == pytest.approx(input_cost * 2)
    assert info["output_cost_per_token_priority"] == pytest.approx(output_cost * 2)
    assert info["input_cost_per_token_above_272k_tokens_priority"] == pytest.approx(input_cost * 4)
    assert info["output_cost_per_token_above_272k_tokens_priority"] == pytest.approx(output_cost * 3)

    assert info["max_input_tokens"] == 1050000
    assert info["max_output_tokens"] == 128000
    assert info["supports_reasoning"] is True

    routed_model, provider, _, _ = get_llm_provider(model=model)
    assert provider == "azure"


@pytest.mark.parametrize("model", AZURE_REGIONAL_MODELS)
def test_azure_gpt_5_6_regional_model_info(model):
    model_cost = _load_main()
    info = model_cost.get(model)
    assert info is not None, f"{model} not found in model_prices_and_context_window.json"

    assert info["litellm_provider"] == "azure"
    assert info["mode"] == "chat"

    input_cost, output_cost, cache_read_cost, _ = STANDARD_PRICING[_tier_key(model)]

    assert info["input_cost_per_token"] == pytest.approx(input_cost * 1.1)
    assert info["output_cost_per_token"] == pytest.approx(output_cost * 1.1)
    assert info["cache_read_input_token_cost"] == pytest.approx(cache_read_cost * 1.1)
    assert info["input_cost_per_token_above_272k_tokens"] == pytest.approx(input_cost * 2.2)
    assert info["output_cost_per_token_above_272k_tokens"] == pytest.approx(output_cost * 1.65)
    assert info["input_cost_per_token_priority"] == pytest.approx(input_cost * 2.75)
    assert info["output_cost_per_token_priority"] == pytest.approx(output_cost * 2.75)

    assert info["max_input_tokens"] == 1050000
    assert info["max_output_tokens"] == 128000
    assert info["supports_reasoning"] is True

    _, provider, _, _ = get_llm_provider(model=model)
    assert provider == "azure"


def test_gpt_5_6_backup_matches_main():
    """Ensure the bundled model cost map stays in sync with the canonical file."""
    repo_root = Path(__file__).parents[2]
    main_path = repo_root / "model_prices_and_context_window.json"
    backup_path = repo_root / "litellm" / "model_prices_and_context_window_backup.json"

    with open(main_path) as f:
        main_cost = json.load(f)
    with open(backup_path) as f:
        backup_cost = json.load(f)

    for model in GPT_5_6_MODELS + AZURE_GLOBAL_MODELS + AZURE_REGIONAL_MODELS:
        assert backup_cost.get(model) == main_cost.get(model), (
            f"{model} differs between main and backup model cost maps"
        )
