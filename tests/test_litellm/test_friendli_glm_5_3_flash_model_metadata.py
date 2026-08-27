import json
from pathlib import Path

from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


def test_friendli_glm_5_3_flash_model_info():
    model = "friendliai/zai-org/GLM-5.3-Flash"
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    with open(json_path) as f:
        model_cost = json.load(f)

    info = model_cost.get(model)
    assert (
        info is not None
    ), f"{model} not found in model_prices_and_context_window.json"
    assert info["litellm_provider"] == "friendliai"
    assert info["mode"] == "chat"
    # $0.15 / $0.50 per MTok, $0.03 cached input (per Friendli /v1/models)
    assert info["input_cost_per_token"] == 1.5e-07
    assert info["output_cost_per_token"] == 5e-07
    assert info["cache_read_input_token_cost"] == 3e-08
    assert info["max_input_tokens"] == 1048576
    assert info["max_output_tokens"] == 1048576
    assert info["supports_function_calling"] is True
    assert info["supports_reasoning"] is True
    # chat_template.jinja accepts reasoning_effort in {"low", "high"} and defaults to "max"
    assert info["supports_low_reasoning_effort"] is True
    assert info["supports_max_reasoning_effort"] is True
    assert info["supports_tool_choice"] is True

    routed_model, provider, _, _ = get_llm_provider(model=model)
    assert routed_model == "zai-org/GLM-5.3-Flash"
    assert provider == "friendliai"
