import json
from pathlib import Path

import pytest

from litellm.constants import bedrock_embedding_models

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

BASE_MODEL = "twelvelabs.marengo-embed-3-0-v1:0"
PROFILE_MODELS = ("us.twelvelabs.marengo-embed-3-0-v1:0", "eu.twelvelabs.marengo-embed-3-0-v1:0")
ALL_MODELS = (BASE_MODEL, *PROFILE_MODELS)
MARENGO_2_7_MODELS = (
    "twelvelabs.marengo-embed-2-7-v1:0",
    "us.twelvelabs.marengo-embed-2-7-v1:0",
    "eu.twelvelabs.marengo-embed-2-7-v1:0",
)
PER_REQUEST_MODELS = (*ALL_MODELS, *MARENGO_2_7_MODELS)

TEXT_REQUEST_COST = 7e-05
IMAGE_REQUEST_COST = 0.0001
VIDEO_COST_PER_SECOND = 0.0007
AUDIO_COST_PER_SECOND = 0.00014


def _load(path):
    with open(path) as f:
        return json.load(f)


def test_marengo_embed_3_is_a_known_bedrock_embedding_model():
    assert BASE_MODEL in bedrock_embedding_models


@pytest.mark.parametrize("model", PER_REQUEST_MODELS)
def test_backup_matches_main(model):
    main_cost = _load(MAIN_PATH)
    backup_cost = _load(BACKUP_PATH)

    assert model in main_cost, f"{model} missing from model_prices_and_context_window.json"
    assert model in backup_cost, f"{model} missing from model_prices_and_context_window_backup.json"
    assert backup_cost[model] == main_cost[model], f"{model} differs between main and backup model cost maps"
