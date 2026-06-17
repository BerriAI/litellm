import json
from pathlib import Path

import pytest

from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


@pytest.mark.parametrize("model", ["mistral/mistral-medium-3-5"])
def test_mistral_medium_3_5_model_info(model):
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    with open(json_path) as f:
        model_cost = json.load(f)

    info = model_cost.get(model)
    assert (
        info is not None
    ), f"{model} not found in model_prices_and_context_window.json"

    assert info["litellm_provider"] == "mistral"
    assert info["mode"] == "chat"

    assert info["input_cost_per_token"] == 1.5e-06
    assert info["output_cost_per_token"] == 7.5e-06

    assert info["max_input_tokens"] == 262144
    assert info["max_output_tokens"] == 262144
    assert info["max_tokens"] == 262144

    assert info["supports_function_calling"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_vision"] is True
    assert info["supports_assistant_prefill"] is True

    routed_model, provider, _, _ = get_llm_provider(model=model)
    assert routed_model == model.split("/", 1)[1]
    assert provider == "mistral"


def test_mistral_medium_3_5_backup_matches_main():
    """Ensure the bundled model cost map stays in sync with the canonical file."""
    repo_root = Path(__file__).parents[2]
    main_path = repo_root / "model_prices_and_context_window.json"
    backup_path = repo_root / "litellm" / "model_prices_and_context_window_backup.json"

    with open(main_path) as f:
        main_cost = json.load(f)
    with open(backup_path) as f:
        backup_cost = json.load(f)

    for model in ("mistral/mistral-medium-3-5",):
        assert backup_cost.get(model) == main_cost.get(
            model
        ), f"{model} differs between main and backup model cost maps"
