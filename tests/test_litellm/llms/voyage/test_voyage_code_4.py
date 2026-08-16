import json
import os

import pytest

import litellm

ROOT_MAP = os.path.join(
    os.path.dirname(os.path.dirname(litellm.__file__)),
    "model_prices_and_context_window.json",
)
BACKUP_MAP = os.path.join(
    os.path.dirname(litellm.__file__),
    "model_prices_and_context_window_backup.json",
)


@pytest.fixture(autouse=True)
def _use_local_model_cost_map(monkeypatch):
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_voyage_code_4_maps_are_in_sync():
    """voyage/voyage-code-4 must exist and match across both cost maps."""
    root_entry = _load(ROOT_MAP)["voyage/voyage-code-4"]
    backup_entry = _load(BACKUP_MAP)["voyage/voyage-code-4"]

    assert root_entry == backup_entry
    assert root_entry["litellm_provider"] == "voyage"
    assert root_entry["mode"] == "embedding"
    assert root_entry["max_input_tokens"] == 32000
    assert root_entry["max_tokens"] == 32000
    assert root_entry["input_cost_per_token"] == 1.8e-07


def test_voyage_code_4_get_model_info():
    """litellm.get_model_info should resolve voyage/voyage-code-4 from the local map."""
    model_info = litellm.get_model_info(model="voyage/voyage-code-4")
    assert model_info["litellm_provider"] == "voyage"
    assert model_info["mode"] == "embedding"
    assert model_info["input_cost_per_token"] == 1.8e-07
