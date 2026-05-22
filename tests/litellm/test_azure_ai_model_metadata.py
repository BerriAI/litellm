"""
Tests for new azure_ai model entries in model_prices_and_context_window.json.
Covers models added as part of LIT-3157 (add missing models to azure_ai prefix).
"""

import json
from pathlib import Path

import pytest

from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


JSON_PATH = Path(__file__).parents[2] / "model_prices_and_context_window.json"


@pytest.fixture(scope="module")
def model_cost():
    with open(JSON_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# AI21 Jamba models
# ---------------------------------------------------------------------------


def test_azure_ai_ai21_jamba_1_5_large(model_cost):
    model = "azure_ai/AI21-Jamba-1.5-Large"
    info = model_cost.get(model)
    assert (
        info is not None
    ), f"{model} not found in model_prices_and_context_window.json"
    assert info["litellm_provider"] == "azure_ai"
    assert info["mode"] == "chat"
    assert info["input_cost_per_token"] > 0
    assert info["output_cost_per_token"] > 0
    assert info["max_input_tokens"] == 256000
    assert info["max_output_tokens"] == 4096
    assert info["supports_function_calling"] is True
    assert info["supports_tool_choice"] is True

    routed_model, provider, _, _ = get_llm_provider(model=model)
    assert routed_model == "AI21-Jamba-1.5-Large"
    assert provider == "azure_ai"


def test_azure_ai_ai21_jamba_1_5_mini(model_cost):
    model = "azure_ai/AI21-Jamba-1.5-Mini"
    info = model_cost.get(model)
    assert (
        info is not None
    ), f"{model} not found in model_prices_and_context_window.json"
    assert info["litellm_provider"] == "azure_ai"
    assert info["mode"] == "chat"
    assert info["input_cost_per_token"] > 0
    assert info["output_cost_per_token"] > 0
    assert info["max_input_tokens"] == 256000
    assert info["max_output_tokens"] == 4096
    assert info["supports_function_calling"] is True
    assert info["supports_tool_choice"] is True

    routed_model, provider, _, _ = get_llm_provider(model=model)
    assert routed_model == "AI21-Jamba-1.5-Mini"
    assert provider == "azure_ai"


def test_azure_ai_ai21_jamba_instruct(model_cost):
    model = "azure_ai/AI21-Jamba-Instruct"
    info = model_cost.get(model)
    assert (
        info is not None
    ), f"{model} not found in model_prices_and_context_window.json"
    assert info["litellm_provider"] == "azure_ai"
    assert info["mode"] == "chat"
    assert info["input_cost_per_token"] > 0
    assert info["output_cost_per_token"] > 0
    assert info["max_input_tokens"] == 70000
    assert info["max_output_tokens"] == 4096


# ---------------------------------------------------------------------------
# Codestral-2501
# ---------------------------------------------------------------------------


def test_azure_ai_codestral_2501(model_cost):
    model = "azure_ai/Codestral-2501"
    info = model_cost.get(model)
    assert (
        info is not None
    ), f"{model} not found in model_prices_and_context_window.json"
    assert info["litellm_provider"] == "azure_ai"
    assert info["mode"] == "chat"
    assert info["input_cost_per_token"] > 0
    assert info["output_cost_per_token"] > 0
    assert info["max_input_tokens"] == 256000
    assert info["max_output_tokens"] == 65536
    assert info["supports_function_calling"] is True

    routed_model, provider, _, _ = get_llm_provider(model=model)
    assert routed_model == "Codestral-2501"
    assert provider == "azure_ai"


# ---------------------------------------------------------------------------
# Cohere Command R family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name",
    [
        "Cohere-command-r",
        "Cohere-command-r-08-2024",
        "Cohere-command-r-plus",
        "Cohere-command-r-plus-08-2024",
    ],
)
def test_azure_ai_cohere_command_r_variants(model_cost, model_name):
    model = f"azure_ai/{model_name}"
    info = model_cost.get(model)
    assert (
        info is not None
    ), f"{model} not found in model_prices_and_context_window.json"
    assert info["litellm_provider"] == "azure_ai"
    assert info["mode"] == "chat"
    assert info["input_cost_per_token"] > 0
    assert info["output_cost_per_token"] > 0
    assert info["max_input_tokens"] == 128000
    assert info["max_output_tokens"] == 4096
    assert info["supports_function_calling"] is True
    assert info["supports_tool_choice"] is True

    routed_model, provider, _, _ = get_llm_provider(model=model)
    assert routed_model == model_name
    assert provider == "azure_ai"


def test_azure_ai_cohere_command_a(model_cost):
    model = "azure_ai/cohere-command-a"
    info = model_cost.get(model)
    assert (
        info is not None
    ), f"{model} not found in model_prices_and_context_window.json"
    assert info["litellm_provider"] == "azure_ai"
    assert info["mode"] == "chat"
    assert info["input_cost_per_token"] > 0
    assert info["output_cost_per_token"] > 0
    assert info["max_input_tokens"] == 256000
    assert info["max_output_tokens"] == 8192
    assert info["supports_function_calling"] is True


# ---------------------------------------------------------------------------
# Meta Llama 3 8B
# ---------------------------------------------------------------------------


def test_azure_ai_meta_llama_3_8b_instruct(model_cost):
    model = "azure_ai/Meta-Llama-3-8B-Instruct"
    info = model_cost.get(model)
    assert (
        info is not None
    ), f"{model} not found in model_prices_and_context_window.json"
    assert info["litellm_provider"] == "azure_ai"
    assert info["mode"] == "chat"
    assert info["input_cost_per_token"] > 0
    assert info["output_cost_per_token"] > 0
    assert info["max_input_tokens"] == 8192
    assert info["max_output_tokens"] == 2048
    assert info["supports_tool_choice"] is True

    routed_model, provider, _, _ = get_llm_provider(model=model)
    assert routed_model == "Meta-Llama-3-8B-Instruct"
    assert provider == "azure_ai"


# ---------------------------------------------------------------------------
# Mistral-Large-2411
# ---------------------------------------------------------------------------


def test_azure_ai_mistral_large_2411(model_cost):
    model = "azure_ai/Mistral-Large-2411"
    info = model_cost.get(model)
    assert (
        info is not None
    ), f"{model} not found in model_prices_and_context_window.json"
    assert info["litellm_provider"] == "azure_ai"
    assert info["mode"] == "chat"
    assert info["input_cost_per_token"] > 0
    assert info["output_cost_per_token"] > 0
    assert info["max_input_tokens"] == 128000
    assert info["max_output_tokens"] == 8191
    assert info["supports_function_calling"] is True
    assert info["supports_tool_choice"] is True

    routed_model, provider, _, _ = get_llm_provider(model=model)
    assert routed_model == "Mistral-Large-2411"
    assert provider == "azure_ai"
