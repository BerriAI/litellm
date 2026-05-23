import json
from pathlib import Path

import pytest

from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

COHERE_CHAT_MODELS = {
    "command-a-plus-05-2026": {
        "max_input_tokens": 128000,
        "max_output_tokens": 64000,
        "supports_function_calling": True,
        "supports_reasoning": True,
        "supports_tool_choice": True,
        "supports_vision": True,
    },
    "command-a-reasoning-08-2025": {
        "max_input_tokens": 256000,
        "max_output_tokens": 32000,
        "supports_function_calling": True,
        "supports_reasoning": True,
        "supports_tool_choice": True,
    },
    "command-a-translate-08-2025": {
        "max_input_tokens": 8000,
        "max_output_tokens": 8000,
    },
    "command-a-vision-07-2025": {
        "max_input_tokens": 128000,
        "max_output_tokens": 8000,
        "supports_function_calling": True,
        "supports_tool_choice": True,
        "supports_vision": True,
    },
    "c4ai-aya-expanse-32b": {
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
    },
    "c4ai-aya-vision-32b": {
        "max_input_tokens": 16384,
        "max_output_tokens": 4096,
        "supports_vision": True,
    },
    "tiny-aya-global": {
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
    },
    "tiny-aya-earth": {
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
    },
    "tiny-aya-fire": {
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
    },
    "tiny-aya-water": {
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
    },
}

COHERE_EMBED_MODELS = {
    "embed-v4.0": {
        "max_input_tokens": 128000,
        "supports_embedding_image_input": True,
    },
}

COHERE_RERANK_MODELS = {
    "rerank-v4.0-fast": {
        "max_input_tokens": 32768,
        "input_cost_per_query": 0.002,
    },
    "rerank-v4.0-pro": {
        "max_input_tokens": 32768,
        "input_cost_per_query": 0.0025,
    },
}


@pytest.fixture(scope="module")
def model_cost():
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    with open(json_path) as f:
        return json.load(f)


@pytest.mark.parametrize("model,expected", COHERE_CHAT_MODELS.items())
def test_cohere_chat_model_info(model, expected, model_cost):
    info = model_cost.get(model)
    assert (
        info is not None
    ), f"{model} not found in model_prices_and_context_window.json"
    assert info["litellm_provider"] == "cohere_chat"
    assert info["mode"] == "chat"
    assert info["input_cost_per_token"] > 0
    assert info["output_cost_per_token"] > 0

    for key, value in expected.items():
        assert info[key] == value, f"{model}: expected {key}={value}, got {info.get(key)}"

    routed_model, provider, _, _ = get_llm_provider(model=f"cohere_chat/v2/{model}")
    assert routed_model == f"v2/{model}"
    assert provider == "cohere_chat"


@pytest.mark.parametrize("model,expected", COHERE_EMBED_MODELS.items())
def test_cohere_embed_model_info(model, expected, model_cost):
    info = model_cost.get(model)
    assert info is not None
    assert info["litellm_provider"] == "cohere"
    assert info["mode"] == "embedding"

    for key, value in expected.items():
        assert info[key] == value


@pytest.mark.parametrize("model,expected", COHERE_RERANK_MODELS.items())
def test_cohere_rerank_model_info(model, expected, model_cost):
    info = model_cost.get(model)
    assert info is not None
    assert info["litellm_provider"] == "cohere"
    assert info["mode"] == "rerank"

    for key, value in expected.items():
        assert info[key] == value
