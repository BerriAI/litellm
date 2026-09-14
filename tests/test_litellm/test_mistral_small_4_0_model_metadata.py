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
def test_backup_matches_main(model):
    main_cost = _load(MAIN_PATH)
    backup_cost = _load(BACKUP_PATH)

    assert backup_cost.get(model) == main_cost.get(model), f"{model} differs between main and backup model cost maps"
