import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

MEDIUM_3_5_MODELS = (
    "mistral/mistral-medium-3-5",
    "mistral/mistral-medium-2604",
    "mistral/mistral-medium-latest",
)

SYNCED_MODELS = MEDIUM_3_5_MODELS + (
    "mistral/mistral-medium-2508",
    "mistral/mistral-medium-3-1-2508",
)


def _load(path):
    with open(path) as f:
        return json.load(f)









@pytest.mark.parametrize("model", SYNCED_MODELS)
def test_backup_matches_main(model):
    """Ensure the bundled (backup) cost map stays in sync with the canonical file."""
    main_cost = _load(MAIN_PATH)
    backup_cost = _load(BACKUP_PATH)

    assert backup_cost.get(model) == main_cost.get(model), f"{model} differs between main and backup model cost maps"
