import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

DAYBREAK_MODELS = (
    "gpt-5.6-cyber",
    "daybreak-red-latest",
    "daybreak-blue-latest",
)
BLUE_ALIAS = "daybreak-blue-latest"
BLUE_SNAPSHOT = "gpt-5.6-sol"


def _load(path):
    with open(path) as f:
        return json.load(f)


@pytest.mark.parametrize("model", DAYBREAK_MODELS)
def test_daybreak_capability_contract(model):
    info = _load(MAIN_PATH).get(model)
    assert info is not None, f"{model} missing from model_prices_and_context_window.json"

    assert info["litellm_provider"] == "openai"
    assert info["mode"] == "chat"
    assert info["supported_endpoints"] == ["/v1/chat/completions", "/v1/responses"]

    assert info["supports_computer_use"] is True
    assert info["supports_parallel_function_calling"] is True
    assert info["supports_function_calling"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_vision"] is True


def test_blue_alias_matches_its_snapshot_computer_use():
    cost_map = _load(MAIN_PATH)

    assert cost_map[BLUE_ALIAS]["supports_computer_use"] is True
    assert cost_map[BLUE_SNAPSHOT]["supports_computer_use"] is True


@pytest.mark.parametrize("model", (*DAYBREAK_MODELS, BLUE_SNAPSHOT))
def test_backup_matches_main(model):
    main_cost = _load(MAIN_PATH)
    backup_cost = _load(BACKUP_PATH)

    assert backup_cost.get(model) == main_cost.get(model), f"{model} differs between main and backup model cost maps"
