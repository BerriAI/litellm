import json
from pathlib import Path

import pytest

import litellm
from litellm.constants import bedrock_embedding_models
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.types.utils import Usage

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

BASE_MODEL = "twelvelabs.marengo-embed-3-0-v1:0"
PROFILE_MODELS = ("us.twelvelabs.marengo-embed-3-0-v1:0", "eu.twelvelabs.marengo-embed-3-0-v1:0")
ALL_MODELS = (BASE_MODEL, *PROFILE_MODELS)

TEXT_REQUEST_COST = 7e-05
IMAGE_REQUEST_COST = 0.0001
VIDEO_COST_PER_SECOND = 0.0007
AUDIO_COST_PER_SECOND = 0.00014


def _load(path):
    with open(path) as f:
        return json.load(f)


@pytest.mark.parametrize("model", ALL_MODELS)
def test_marengo_embed_3_specs(model):
    info = _load(MAIN_PATH).get(model)
    assert info is not None, f"{model} missing from model_prices_and_context_window.json"

    assert info["litellm_provider"] == "bedrock"
    assert info["mode"] == "embedding"
    assert info["input_cost_per_token"] == TEXT_REQUEST_COST
    assert info["output_cost_per_token"] == 0.0
    assert info["max_input_tokens"] == 500
    assert info["max_tokens"] == 500
    assert info["output_vector_size"] == 512
    assert info["supports_embedding_image_input"] is True
    assert info["supports_image_input"] is True
    assert "deprecation_date" not in info

    routed_model, provider, _, _ = get_llm_provider(model=f"bedrock/{model}")
    assert routed_model == model
    assert provider == "bedrock"


@pytest.mark.parametrize("model", PROFILE_MODELS)
def test_marengo_embed_3_inference_profiles_price_image_video_and_audio(model):
    info = _load(MAIN_PATH)[model]
    assert info["input_cost_per_image"] == IMAGE_REQUEST_COST
    assert info["input_cost_per_video_per_second"] == VIDEO_COST_PER_SECOND
    assert info["input_cost_per_audio_per_second"] == AUDIO_COST_PER_SECOND


@pytest.mark.parametrize("model", ALL_MODELS)
def test_marengo_embed_3_is_visible_to_callers(model, local_model_cost_map):
    info = litellm.get_model_info(model=model, custom_llm_provider="bedrock")
    assert info["mode"] == "embedding"
    assert info["output_vector_size"] == 512
    assert info["max_input_tokens"] == 500


@pytest.mark.parametrize("model", ALL_MODELS)
def test_marengo_embed_3_text_request_is_billed(model, local_model_cost_map):
    usage = Usage(prompt_tokens=128, completion_tokens=0, total_tokens=128)
    prompt_cost, completion_cost = litellm.cost_per_token(
        model=model, usage_object=usage, custom_llm_provider="bedrock"
    )
    assert prompt_cost == pytest.approx(128 * TEXT_REQUEST_COST)
    assert completion_cost == 0.0


def test_marengo_embed_3_is_a_known_bedrock_embedding_model():
    assert BASE_MODEL in bedrock_embedding_models


@pytest.mark.parametrize("model", ALL_MODELS)
def test_backup_matches_main(model):
    main_cost = _load(MAIN_PATH)
    backup_cost = _load(BACKUP_PATH)

    assert model in main_cost, f"{model} missing from model_prices_and_context_window.json"
    assert model in backup_cost, f"{model} missing from model_prices_and_context_window_backup.json"
    assert backup_cost[model] == main_cost[model], f"{model} differs between main and backup model cost maps"
