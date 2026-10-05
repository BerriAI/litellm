import json
from pathlib import Path

from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


def test_sambanova_minimax_m27_model_info():
    model = "sambanova/MiniMax-M2.7"
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    with open(json_path) as f:
        model_cost = json.load(f)

    info = model_cost.get(model)
    assert info is not None, f"{model} not found in model_prices_and_context_window.json"
    assert info["litellm_provider"] == "sambanova"
    assert info["mode"] == "chat"
    assert info["input_cost_per_token"] > 0
    assert info["output_cost_per_token"] > 0
    assert info["supports_function_calling"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_tool_choice"] is True

    routed_model, provider, _, _ = get_llm_provider(model=model)
    assert routed_model == "MiniMax-M2.7"
    assert provider == "sambanova"


def _sambanova_models() -> dict:
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    with open(json_path) as f:
        model_cost = json.load(f)
    return {k: v for k, v in model_cost.items() if k.startswith("sambanova/")}


def test_sambanova_minimax_m3_model_info():
    info = _sambanova_models().get("sambanova/MiniMax-M3")
    assert info is not None, "sambanova/MiniMax-M3 missing from model_prices_and_context_window.json"
    assert info["litellm_provider"] == "sambanova"
    assert info["mode"] == "chat"
    assert info["max_input_tokens"] == 1048576
    assert info["input_cost_per_token"] == 6e-07
    assert info["output_cost_per_token"] == 2.4e-06
    assert info["supports_function_calling"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_response_schema"] is True
    assert not info.get("supports_vision")

    routed_model, provider, _, _ = get_llm_provider(model="sambanova/MiniMax-M3")
    assert routed_model == "MiniMax-M3"
    assert provider == "sambanova"


def test_sambanova_max_output_tokens_match_live_catalog():
    models = _sambanova_models()
    assert models["sambanova/DeepSeek-V3.1"]["max_output_tokens"] == 7168
    assert models["sambanova/DeepSeek-V3.2"]["max_output_tokens"] == 7168
    assert models["sambanova/Meta-Llama-3.3-70B-Instruct"]["max_output_tokens"] == 3072


def test_sambanova_retired_models_removed():
    models = _sambanova_models()
    assert "sambanova/DeepSeek-R1" not in models
    assert "sambanova/Llama-4-Maverick-17B-128E-Instruct" not in models
