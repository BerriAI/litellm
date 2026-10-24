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


def test_sambanova_chat_entries_are_consistent():
    chat_models = {k: v for k, v in _sambanova_models().items() if v["mode"] == "chat"}
    assert chat_models
    for name, info in chat_models.items():
        assert info["litellm_provider"] == "sambanova", name
        assert info["input_cost_per_token"] >= 0, name
        assert info["output_cost_per_token"] >= 0, name
        assert info["max_tokens"] == info["max_output_tokens"], name
        assert info["max_output_tokens"] <= info["max_input_tokens"], name
        if info.get("supports_tool_choice"):
            assert info.get("supports_function_calling"), name


def test_sambanova_minimax_m3_is_routed_to_sambanova():
    assert "sambanova/MiniMax-M3" in _sambanova_models()

    routed_model, provider, _, _ = get_llm_provider(model="sambanova/MiniMax-M3")
    assert routed_model == "MiniMax-M3"
    assert provider == "sambanova"
