import json
from pathlib import Path

import pytest

import litellm

VOYAGE_4_MODELS = {
    "voyage/voyage-4-large": 1.2e-07,
    "voyage/voyage-4": 6e-08,
    "voyage/voyage-4-lite": 2e-08,
}


@pytest.mark.parametrize("model,input_cost", VOYAGE_4_MODELS.items())
def test_voyage_4_model_info(model, input_cost):
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    with open(json_path) as f:
        model_cost = json.load(f)

    info = model_cost.get(model)
    assert info is not None, f"{model} not found in model_prices_and_context_window.json"

    assert info["litellm_provider"] == "voyage"
    assert info["mode"] == "embedding"
    assert info["input_cost_per_token"] == input_cost
    assert info["output_cost_per_token"] == 0.0
    assert info["max_input_tokens"] == 32000
    assert info["max_tokens"] == 32000


@pytest.mark.parametrize("model,input_cost", VOYAGE_4_MODELS.items())
def test_voyage_4_get_model_info_surfaces_mode(model, input_cost):
    litellm.model_cost = litellm.get_model_cost_map(url="")
    info = litellm.get_model_info(model=model)
    assert info["mode"] == "embedding"
    assert info["input_cost_per_token"] == input_cost


@pytest.mark.parametrize("model", VOYAGE_4_MODELS.keys())
def test_voyage_4_backup_matches_main(model):
    repo_root = Path(__file__).parents[2]
    main_path = repo_root / "model_prices_and_context_window.json"
    backup_path = repo_root / "litellm" / "model_prices_and_context_window_backup.json"

    with open(main_path) as f:
        main_cost = json.load(f)
    with open(backup_path) as f:
        backup_cost = json.load(f)

    assert backup_cost.get(model) == main_cost.get(model), (
        f"{model} differs between main and backup model cost maps"
    )
