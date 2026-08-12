import json
from pathlib import Path

import pytest

import litellm

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

LIGHTNING_MODELS = (
    ("openrouter/nvidia/nemotron-3.5-lightning", "openrouter"),
    ("deepinfra/nvidia/NVIDIA-Nemotron-3.5-Lightning", "deepinfra"),
)


def _load(path):
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def local_model_cost_map(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


@pytest.mark.parametrize("model,provider", LIGHTNING_MODELS)
def test_lightning_specs(model, provider):
    info = _load(MAIN_PATH).get(model)
    assert info is not None, f"{model} missing from model_prices_and_context_window.json"

    assert info["litellm_provider"] == provider
    assert info["mode"] == "chat"

    assert info["input_cost_per_token"] == 5e-08
    assert info["output_cost_per_token"] == 2e-07

    assert info["max_input_tokens"] == 262144
    assert "max_output_tokens" not in info, "neither provider publishes an output cap"

    assert info["supports_reasoning"] is True
    assert info["supports_function_calling"] is True
    assert info["supports_tool_choice"] is True


@pytest.mark.parametrize("model,provider", LIGHTNING_MODELS)
def test_lightning_resolves_through_get_model_info(model, provider, local_model_cost_map):
    info = litellm.get_model_info(model=model)

    assert info["litellm_provider"] == provider
    assert info["input_cost_per_token"] == 5e-08
    assert info["output_cost_per_token"] == 2e-07
    assert info["max_input_tokens"] == 262144
    assert info["supports_reasoning"] is True


@pytest.mark.parametrize("model,provider", LIGHTNING_MODELS)
def test_backup_matches_main(model, provider):
    main_cost = _load(MAIN_PATH)
    backup_cost = _load(BACKUP_PATH)

    assert backup_cost.get(model) == main_cost.get(model), f"{model} differs between main and backup model cost maps"
