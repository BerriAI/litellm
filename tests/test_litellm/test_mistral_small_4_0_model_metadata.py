import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

SMALL_4_0_MODELS = (
    "mistral/mistral-small-latest",
    "mistral/mistral-small-2603",
)


def _load(path):
    with open(path) as f:
        return json.load(f)


@pytest.mark.parametrize("model", SMALL_4_0_MODELS)
def test_small_4_0_specs(model):
    info = _load(MAIN_PATH).get(model)
    assert info is not None, f"{model} missing from model_prices_and_context_window.json"

    assert info["litellm_provider"] == "mistral"
    assert info["mode"] == "chat"

    assert info["input_cost_per_token"] == 1.5e-07
    assert info["output_cost_per_token"] == 6e-07

    assert info["max_input_tokens"] == 262144
    assert info["max_output_tokens"] == 262144
    assert info["max_tokens"] == 262144

    assert info["supports_reasoning"] is True
    assert info["supports_vision"] is True
    assert info["supports_function_calling"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_assistant_prefill"] is True


@pytest.mark.parametrize("model", SMALL_4_0_MODELS)
def test_backup_matches_main(model):
    main_cost = _load(MAIN_PATH)
    backup_cost = _load(BACKUP_PATH)

    assert backup_cost.get(model) == main_cost.get(model), f"{model} differs between main and backup model cost maps"
