import json
from pathlib import Path

from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


def test_sambanova_minimax_m27_model_info():
    model = "sambanova/MiniMax-M2.7"
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    with open(json_path) as f:
        model_cost = json.load(f)

    info = model_cost.get(model)
    assert (
        info is not None
    ), f"{model} not found in model_prices_and_context_window.json"
    assert info["litellm_provider"] == "sambanova"
    assert info["mode"] == "chat"
    assert info["input_cost_per_token"] > 0
    assert info["output_cost_per_token"] > 0
    assert info["max_input_tokens"] == 204800
    assert info["max_output_tokens"] == 131072
    assert info["supports_function_calling"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_tool_choice"] is True

    routed_model, provider, _, _ = get_llm_provider(model=model)
    assert routed_model == "MiniMax-M2.7"
    assert provider == "sambanova"
