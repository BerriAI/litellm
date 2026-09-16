import json
from pathlib import Path

import pytest

import litellm
from litellm.constants import bedrock_embedding_models
from litellm.types.utils import PromptTokensDetailsWrapper, Usage

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


@pytest.mark.parametrize("model", ALL_MODELS)
def test_marengo_embed_3_is_visible_to_callers(model, local_model_cost_map):
    info = litellm.get_model_info(model=model, custom_llm_provider="bedrock")
    assert info["mode"] == "embedding"
    assert info["output_vector_size"] == 512
    assert info["max_input_tokens"] == 500


@pytest.mark.parametrize("model", PER_REQUEST_MODELS)
@pytest.mark.parametrize(
    "details,expected_cost",
    [
        (PromptTokensDetailsWrapper(query_count=1), TEXT_REQUEST_COST),
        (PromptTokensDetailsWrapper(image_count=1), IMAGE_REQUEST_COST),
        (PromptTokensDetailsWrapper(query_count=1, image_count=1), TEXT_REQUEST_COST + IMAGE_REQUEST_COST),
        (PromptTokensDetailsWrapper(query_count=1, image_count=2), TEXT_REQUEST_COST + 2 * IMAGE_REQUEST_COST),
        (PromptTokensDetailsWrapper(video_length_seconds=10), 10 * VIDEO_COST_PER_SECOND),
        (PromptTokensDetailsWrapper(audio_length_seconds=10), 10 * AUDIO_COST_PER_SECOND),
    ],
)
def test_marengo_requests_are_billed_per_request(model, details, expected_cost, local_model_cost_map):
    usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0, prompt_tokens_details=details)
    prompt_cost, completion_cost = litellm.cost_per_token(
        model=model, usage_object=usage, custom_llm_provider="bedrock"
    )
    assert prompt_cost == pytest.approx(expected_cost)
    assert completion_cost == 0.0


@pytest.mark.parametrize("model", PER_REQUEST_MODELS)
def test_marengo_token_counts_bill_nothing(model, local_model_cost_map):
    usage = Usage(prompt_tokens=128, completion_tokens=0, total_tokens=128)
    prompt_cost, completion_cost = litellm.cost_per_token(
        model=model, usage_object=usage, custom_llm_provider="bedrock"
    )
    assert prompt_cost == 0.0
    assert completion_cost == 0.0


def test_marengo_embed_3_is_a_known_bedrock_embedding_model():
    assert BASE_MODEL in bedrock_embedding_models


@pytest.mark.parametrize("model", PER_REQUEST_MODELS)
def test_backup_matches_main(model):
    main_cost = _load(MAIN_PATH)
    backup_cost = _load(BACKUP_PATH)

    assert model in main_cost, f"{model} missing from model_prices_and_context_window.json"
    assert model in backup_cost, f"{model} missing from model_prices_and_context_window_backup.json"
    assert backup_cost[model] == main_cost[model], f"{model} differs between main and backup model cost maps"
