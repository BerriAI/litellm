import json
from pathlib import Path

from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


def test_friendli_glm_5_3_model_info():
    model = "friendliai/zai-org/GLM-5.3"
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    with open(json_path) as f:
        model_cost = json.load(f)

    info = model_cost.get(model)
    assert (
        info is not None
    ), f"{model} not found in model_prices_and_context_window.json"
    assert info["litellm_provider"] == "friendliai"
    assert info["mode"] == "chat"
    assert info["input_cost_per_token"] == 1.4e-06
    assert info["output_cost_per_token"] == 4.4e-06
    assert info["cache_read_input_token_cost"] == 2.6e-07
    assert info["max_input_tokens"] == 1048576
    assert info["max_output_tokens"] == 1048576
    assert info["supports_function_calling"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_low_reasoning_effort"] is True
    assert info["supports_max_reasoning_effort"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_prompt_caching"] is True
    assert info["supports_vision"] is False
    assert info["supports_image_input"] is False

    routed_model, provider, _, _ = get_llm_provider(model=model)
    assert routed_model == "zai-org/GLM-5.3"
    assert provider == "friendliai"
