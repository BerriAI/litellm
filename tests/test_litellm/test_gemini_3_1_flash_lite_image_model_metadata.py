import json
from pathlib import Path

import pytest

import litellm
from litellm import completion_cost
from litellm.cost_calculator import cost_per_token
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.llms.gemini.image_generation.cost_calculator import (
    cost_calculator as gemini_image_generation_cost_calculator,
)
from litellm.llms.vertex_ai.image_generation.cost_calculator import (
    cost_calculator as vertex_image_generation_cost_calculator,
)
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    ImageObject,
    ImageResponse,
    ImageUsage,
    ImageUsageInputTokensDetails,
    ModelResponse,
    PromptTokensDetailsWrapper,
    Usage,
)

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

UNPREFIXED = "gemini-3.1-flash-lite-image"
GEMINI = "gemini/gemini-3.1-flash-lite-image"
VERTEX = "vertex_ai/gemini-3.1-flash-lite-image"
ALL_KEYS = (UNPREFIXED, GEMINI, VERTEX)

INPUT_COST = 2.5e-07
INPUT_COST_BATCHES = 1.25e-07
OUTPUT_TEXT_COST = 1.5e-06
OUTPUT_TEXT_COST_BATCHES = 7.5e-07
OUTPUT_IMAGE_TOKEN_COST = 3e-05
OUTPUT_COST_PER_1K_IMAGE = 0.0336
INPUT_COST_PER_IMAGE = 0.00028
CACHE_READ_COST = 2.5e-08
MAX_INPUT_TOKENS = 65536
MAX_OUTPUT_TOKENS = 4096
TOKENS_PER_1K_IMAGE = 1120


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def local_model_cost_map(monkeypatch):
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()


@pytest.mark.parametrize("model", ALL_KEYS)
def test_gemini_3_1_flash_lite_image_is_registered(model: str):
    info = _load(MAIN_PATH).get(model)
    assert info is not None, f"{model} not found in model_prices_and_context_window.json"

    assert info["mode"] == "image_generation"
    assert info["input_cost_per_token"] == INPUT_COST
    assert info["input_cost_per_token_batches"] == INPUT_COST_BATCHES
    assert info["output_cost_per_token"] == OUTPUT_TEXT_COST
    assert info["output_cost_per_token_batches"] == OUTPUT_TEXT_COST_BATCHES
    assert info["output_cost_per_image"] == OUTPUT_COST_PER_1K_IMAGE
    assert info["output_cost_per_image_token"] == OUTPUT_IMAGE_TOKEN_COST
    assert info["max_input_tokens"] == MAX_INPUT_TOKENS
    assert info["max_output_tokens"] == MAX_OUTPUT_TOKENS
    assert info["max_tokens"] == MAX_OUTPUT_TOKENS
    assert info["supports_reasoning"] is False
    assert info["supports_response_schema"] is False
    assert info["supports_vision"] is True
    for field in ("supports_web_search", "search_context_cost_per_query", "web_search_billing_unit"):
        assert field not in info


def test_gemini_3_1_flash_lite_image_provider_specific_fields():
    cost_map = _load(MAIN_PATH)

    unprefixed = cost_map[UNPREFIXED]
    assert unprefixed["litellm_provider"] == "vertex_ai-language-models"
    assert unprefixed["cache_read_input_token_cost"] == CACHE_READ_COST
    assert unprefixed["input_cost_per_image"] == INPUT_COST_PER_IMAGE
    assert unprefixed["supports_function_calling"] is False
    assert unprefixed["supports_prompt_caching"] is True
    assert unprefixed["supports_pdf_input"] is True
    assert unprefixed["supports_video_input"] is True
    assert unprefixed["supported_modalities"] == ["text", "image", "video"]

    gemini = cost_map[GEMINI]
    assert gemini["litellm_provider"] == "gemini"
    assert gemini["supports_function_calling"] is True
    assert gemini["supports_prompt_caching"] is False
    assert "cache_read_input_token_cost" not in gemini
    assert gemini["supported_modalities"] == ["text", "image"]
    assert gemini["supported_output_modalities"] == ["text", "image"]
    assert gemini["rpm"] == 1000
    assert gemini["tpm"] == 4000000
    assert gemini["input_cost_per_image"] == INPUT_COST_PER_IMAGE

    vertex = cost_map[VERTEX]
    assert vertex["litellm_provider"] == "vertex_ai-language-models"
    assert vertex["cache_read_input_token_cost"] == CACHE_READ_COST
    assert vertex["input_cost_per_image"] == INPUT_COST_PER_IMAGE
    assert vertex["supports_function_calling"] is False
    assert vertex["supports_prompt_caching"] is True


def test_one_k_image_price_matches_official_token_math():
    assert TOKENS_PER_1K_IMAGE * OUTPUT_IMAGE_TOKEN_COST == OUTPUT_COST_PER_1K_IMAGE
    assert TOKENS_PER_1K_IMAGE * INPUT_COST == INPUT_COST_PER_IMAGE


@pytest.mark.parametrize("model", ALL_KEYS)
def test_backup_matches_main(model: str):
    main_cost = _load(MAIN_PATH)
    backup_cost = _load(BACKUP_PATH)
    assert backup_cost.get(model) == main_cost.get(model), f"{model} differs between main and backup model cost maps"


def test_gemini_prefix_routes_to_gemini():
    routed_model, provider, _, _ = get_llm_provider(model=GEMINI)
    assert routed_model == UNPREFIXED
    assert provider == "gemini"


def test_vertex_prefix_routes_to_vertex():
    routed_model, provider, _, _ = get_llm_provider(model=VERTEX)
    assert routed_model == UNPREFIXED
    assert provider == "vertex_ai"


def test_text_token_cost(local_model_cost_map):
    prompt_cost, text_completion_cost = cost_per_token(model=GEMINI, prompt_tokens=1000, completion_tokens=500)
    assert prompt_cost == pytest.approx(1000 * INPUT_COST)
    assert text_completion_cost == pytest.approx(500 * OUTPUT_TEXT_COST)


def test_completion_cost_bills_one_k_image(local_model_cost_map):
    response = ModelResponse()
    response.model = UNPREFIXED
    response.usage = Usage(
        prompt_tokens=7,
        completion_tokens=TOKENS_PER_1K_IMAGE,
        total_tokens=7 + TOKENS_PER_1K_IMAGE,
        completion_tokens_details=CompletionTokensDetailsWrapper(image_tokens=TOKENS_PER_1K_IMAGE, text_tokens=0),
    )
    billed = completion_cost(
        completion_response=response,
        model=UNPREFIXED,
        custom_llm_provider="vertex_ai",
    )
    expected = TOKENS_PER_1K_IMAGE * OUTPUT_IMAGE_TOKEN_COST + 7 * INPUT_COST
    assert billed == pytest.approx(expected)


def test_image_tokens_are_not_billed_as_text(local_model_cost_map):
    usage = Usage(
        completion_tokens=1345,
        prompt_tokens=10,
        total_tokens=1355,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=225,
            rejected_prediction_tokens=None,
            text_tokens=0,
            image_tokens=TOKENS_PER_1K_IMAGE,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None, cached_tokens=None, text_tokens=10, image_tokens=None
        ),
    )

    _, image_completion_cost = generic_cost_per_token(
        model=UNPREFIXED,
        usage=usage,
        custom_llm_provider="vertex_ai",
    )

    expected_completion_cost = TOKENS_PER_1K_IMAGE * OUTPUT_IMAGE_TOKEN_COST + 225 * OUTPUT_TEXT_COST
    bugged_text_only_cost = 1345 * OUTPUT_TEXT_COST
    assert image_completion_cost > bugged_text_only_cost * 2
    assert image_completion_cost == pytest.approx(expected_completion_cost)


def test_gemini_image_generation_uses_token_pricing(local_model_cost_map):
    image_response = ImageResponse(
        data=[ImageObject(b64_json="img1")],
        usage=ImageUsage(
            input_tokens=50 + TOKENS_PER_1K_IMAGE,
            input_tokens_details=ImageUsageInputTokensDetails(
                text_tokens=50,
                image_tokens=TOKENS_PER_1K_IMAGE,
            ),
            output_tokens=TOKENS_PER_1K_IMAGE,
            total_tokens=50 + TOKENS_PER_1K_IMAGE + TOKENS_PER_1K_IMAGE,
        ),
    )

    cost = gemini_image_generation_cost_calculator(model=GEMINI, image_response=image_response)
    expected = (50 + TOKENS_PER_1K_IMAGE) * INPUT_COST + TOKENS_PER_1K_IMAGE * OUTPUT_IMAGE_TOKEN_COST
    assert cost == pytest.approx(expected)
    assert cost != OUTPUT_COST_PER_1K_IMAGE


def test_vertex_image_generation_uses_token_pricing(local_model_cost_map):
    image_response = ImageResponse(
        data=[ImageObject(b64_json="img1")],
        usage=ImageUsage(
            input_tokens=50 + TOKENS_PER_1K_IMAGE,
            input_tokens_details=ImageUsageInputTokensDetails(
                text_tokens=50,
                image_tokens=TOKENS_PER_1K_IMAGE,
            ),
            output_tokens=TOKENS_PER_1K_IMAGE,
            total_tokens=50 + TOKENS_PER_1K_IMAGE + TOKENS_PER_1K_IMAGE,
        ),
    )

    cost = vertex_image_generation_cost_calculator(model=UNPREFIXED, image_response=image_response)
    expected = (50 + TOKENS_PER_1K_IMAGE) * INPUT_COST + TOKENS_PER_1K_IMAGE * OUTPUT_IMAGE_TOKEN_COST
    assert cost == pytest.approx(expected)


def test_vertex_image_generation_falls_back_to_flat_image_price(local_model_cost_map):
    image_response = ImageResponse(data=[ImageObject(b64_json="img1"), ImageObject(b64_json="img2")])
    cost = vertex_image_generation_cost_calculator(model=UNPREFIXED, image_response=image_response)
    assert cost == pytest.approx(2 * OUTPUT_COST_PER_1K_IMAGE)
