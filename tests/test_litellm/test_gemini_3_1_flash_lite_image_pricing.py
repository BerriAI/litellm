import json
from pathlib import Path

import pytest

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.types.utils import (
    ImageObject,
    ImageResponse,
    ImageUsage,
    ImageUsageInputTokensDetails,
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

SHARED_FIELDS = {
    "mode": "image_generation",
    "input_cost_per_token": INPUT_COST,
    "input_cost_per_token_batches": INPUT_COST_BATCHES,
    "input_cost_per_image": INPUT_COST_PER_IMAGE,
    "output_cost_per_token": OUTPUT_TEXT_COST,
    "output_cost_per_token_batches": OUTPUT_TEXT_COST_BATCHES,
    "output_cost_per_image": OUTPUT_COST_PER_1K_IMAGE,
    "output_cost_per_image_token": OUTPUT_IMAGE_TOKEN_COST,
    "max_input_tokens": MAX_INPUT_TOKENS,
    "max_output_tokens": MAX_OUTPUT_TOKENS,
    "max_tokens": MAX_OUTPUT_TOKENS,
    "supported_endpoints": ["/v1/chat/completions", "/v1/completions", "/v1/batch"],
    "supported_output_modalities": ["text", "image"],
    "supports_reasoning": False,
    "supports_response_schema": False,
    "supports_system_messages": True,
    "supports_vision": True,
}

VERTEX_ROUTE_FIELDS = {
    "litellm_provider": "vertex_ai-language-models",
    "cache_read_input_token_cost": CACHE_READ_COST,
    "supported_modalities": ["text", "image", "video"],
    "supports_function_calling": False,
    "supports_pdf_input": True,
    "supports_prompt_caching": True,
    "supports_video_input": True,
}

PER_ROUTE_FIELDS = {
    UNPREFIXED: VERTEX_ROUTE_FIELDS,
    VERTEX: VERTEX_ROUTE_FIELDS,
    GEMINI: {
        "litellm_provider": "gemini",
        "supported_modalities": ["text", "image"],
        "supports_function_calling": True,
        "supports_prompt_caching": False,
        "rpm": 1000,
        "tpm": 4000000,
    },
}

GROUNDING_FIELDS = (
    "supports_web_search",
    "search_context_cost_per_query",
    "web_search_billing_unit",
)


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
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
@pytest.mark.parametrize("path", (MAIN_PATH, BACKUP_PATH), ids=("main", "backup"))
def test_per_route_capabilities_match_model_cards(model: str, path: Path):
    info = _load(path)[model]
    for field, value in PER_ROUTE_FIELDS[model].items():
        assert info[field] == value, f"{model} {field} in {path.name}: {info.get(field)} != {value}"


@pytest.mark.parametrize("model", ALL_KEYS)
def test_backup_matches_main(model: str):
    assert _load(BACKUP_PATH).get(model) == _load(MAIN_PATH).get(model)


def test_gemini_prefix_routes_to_gemini():
    routed_model, provider, _, _ = get_llm_provider(model=GEMINI)
    assert routed_model == UNPREFIXED
    assert provider == "gemini"


def test_vertex_prefix_routes_to_vertex():
    routed_model, provider, _, _ = get_llm_provider(model=VERTEX)
    assert routed_model == UNPREFIXED
    assert provider == "vertex_ai"


def _one_k_image_response() -> ImageResponse:
    return ImageResponse(
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
