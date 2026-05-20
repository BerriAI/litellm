import json
from pathlib import Path

import pytest

from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


@pytest.mark.parametrize(
    "model,input_cost,output_cost,cache_hit_cost",
    [
        ("deepseek/deepseek-v4-flash", 1.4e-07, 2.8e-07, 2.8e-09),
        ("deepseek/deepseek-v4-pro", 1.74e-06, 3.48e-06, 1.45e-08),
    ],
)
def test_deepseek_v4_model_info(model, input_cost, output_cost, cache_hit_cost):
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    with open(json_path) as f:
        model_cost = json.load(f)

    info = model_cost.get(model)
    assert (
        info is not None
    ), f"{model} not found in model_prices_and_context_window.json"

    assert info["litellm_provider"] == "deepseek"
    assert info["mode"] == "chat"

    assert info["input_cost_per_token"] == input_cost
    assert info["output_cost_per_token"] == output_cost
    assert info["input_cost_per_token_cache_hit"] == cache_hit_cost
    assert info["cache_read_input_token_cost"] == cache_hit_cost

    assert info["max_input_tokens"] == 1048576
    assert info["max_output_tokens"] == 393216
    assert info["max_tokens"] == 393216

    assert info["supports_function_calling"] is True
    assert info["supports_prompt_caching"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_system_messages"] is True

    routed_model, provider, _, _ = get_llm_provider(model=model)
    assert routed_model == model.split("/", 1)[1]
    assert provider == "deepseek"


def test_deepseek_v4_backup_matches_main():
    """Ensure the bundled model cost map stays in sync with the canonical file."""
    repo_root = Path(__file__).parents[2]
    main_path = repo_root / "model_prices_and_context_window.json"
    backup_path = repo_root / "litellm" / "model_prices_and_context_window_backup.json"

    with open(main_path) as f:
        main_cost = json.load(f)
    with open(backup_path) as f:
        backup_cost = json.load(f)

    for model in ("deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-pro"):
        assert backup_cost.get(model) == main_cost.get(
            model
        ), f"{model} differs between main and backup model cost maps"
