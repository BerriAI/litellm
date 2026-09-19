import json
from pathlib import Path

import pytest

import litellm

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

GLM_5_2_MODELS = ("mistral/zai-glm-5-2", "mistral/glm-5-2")

INPUT_COST = 1.4e-06
CACHED_INPUT_COST = 1.4e-07
OUTPUT_COST = 4.4e-06


def _load(path):
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def local_model_cost_map(monkeypatch):
    """Force get_model_info to resolve against the in-repo cost map instead of the
    remote one fetched at import time, which still carries the pre-merge pricing."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


@pytest.mark.parametrize("model", GLM_5_2_MODELS)
def test_backup_matches_main(model):
    """Ensure the bundled (backup) cost map stays in sync with the canonical file."""
    main_cost = _load(MAIN_PATH)
    backup_cost = _load(BACKUP_PATH)

    assert backup_cost.get(model) == main_cost.get(model), f"{model} differs between main and backup model cost maps"
