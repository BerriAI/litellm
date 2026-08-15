import json
import os

import litellm

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
BACKUP_MAP = os.path.join(
    REPO_ROOT, "litellm", "model_prices_and_context_window_backup.json"
)


def test_voyage_code_4_in_backup_map():
    """voyage/voyage-code-4 must be present in the local model cost map."""
    with open(BACKUP_MAP) as f:
        model_cost = json.load(f)

    assert "voyage/voyage-code-4" in model_cost
    entry = model_cost["voyage/voyage-code-4"]
    assert entry["litellm_provider"] == "voyage"
    assert entry["mode"] == "embedding"
    assert entry["max_input_tokens"] == 32000
    assert entry["max_tokens"] == 32000
    assert entry["input_cost_per_token"] == 1.8e-07


def test_voyage_code_4_get_model_info(monkeypatch):
    """litellm.get_model_info should resolve voyage/voyage-code-4 from the local map."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    model_info = litellm.get_model_info(model="voyage/voyage-code-4")
    assert model_info["litellm_provider"] == "voyage"
    assert model_info["mode"] == "embedding"
    assert model_info["input_cost_per_token"] == 1.8e-07
