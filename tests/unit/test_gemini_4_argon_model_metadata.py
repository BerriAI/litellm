from pathlib import Path
from typing import Final
from unittest.mock import MagicMock

import pytest
from pydantic import TypeAdapter

import litellm
from litellm import get_model_info
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig
from litellm.types.llms.vertex_ai import UsageMetadata
from litellm.types.utils import ModelResponse

REPO_ROOT: Final = Path(__file__).parents[2]
MODELS: Final = (
    ("gemini/gemini-4-argon", "gemini-4-argon", "gemini"),
    ("vertex_ai/gemini-4-argon", "gemini-4-argon", "vertex_ai"),
)
COST_MAP_ADAPTER: Final = TypeAdapter(dict[str, dict[str, object]])


def _cost_map_entry(path: Path, model: str) -> dict[str, object]:
    return COST_MAP_ADAPTER.validate_json(path.read_bytes())[model]


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize(("model", "expected_model", "expected_provider"), MODELS)
def test_gemini_4_argon_routes_to_its_provider_on_the_thinking_level_path(
    model: str, expected_model: str, expected_provider: str
) -> None:
    routed_model, provider, _, _ = get_llm_provider(model=model)
    assert (routed_model, provider) == (expected_model, expected_provider)

    info = get_model_info(model=routed_model, custom_llm_provider=provider)
    assert info["litellm_provider"] == expected_provider
    assert info["max_output_tokens"] == info["max_input_tokens"]
    assert VertexGeminiConfig.is_gemini_3_or_newer(routed_model) is True
    assert VertexGeminiConfig.is_gemini_4_or_newer(routed_model) is True


@pytest.mark.parametrize("model", [row[0] for row in MODELS])
def test_gemini_4_argon_entry_source_and_backup_match(model: str) -> None:
    main_entry = _cost_map_entry(REPO_ROOT / "model_prices_and_context_window.json", model)
    backup_entry = _cost_map_entry(REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json", model)

    assert backup_entry == main_entry


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gemini-4-argon", True),
        ("gemini/gemini-4-argon", True),
        ("vertex_ai/gemini-4-argon", True),
        ("vertex_ai/gemini-4-argon-cyber-permissive", True),
        ("gemini-3.1-pro-preview", False),
        ("gemini-3.8-flash", False),
        ("gemini-2.5-flash", False),
        ("gemma-4-31b-it", False),
    ],
)
def test_is_gemini_4_or_newer_classification(model: str, expected: bool) -> None:
    assert VertexGeminiConfig.is_gemini_4_or_newer(model) is expected


@pytest.mark.parametrize(
    ("config_cls", "model", "provider"),
    [
        (VertexGeminiConfig, "gemini-4-argon", "vertex_ai"),
        (VertexGeminiConfig, "gemini-4-argon-cyber-permissive", "vertex_ai"),
        (GoogleAIStudioGeminiConfig, "gemini-4-argon", "gemini"),
    ],
)
def test_gemini_4_argon_rejects_candidate_count_and_penalty_parameters(
    config_cls: type[VertexGeminiConfig],
    model: str,
    provider: str,
) -> None:
    config = config_cls()
    supported = config.get_supported_openai_params(model)
    assert "n" not in supported
    assert "frequency_penalty" not in supported
    assert "presence_penalty" not in supported

    mapped = config.map_openai_params(
        non_default_params={
            "n": 2,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
            "max_tokens": 256,
        },
        optional_params={},
        model=model,
        drop_params=False,
    )
    assert "candidate_count" not in mapped
    assert "frequency_penalty" not in mapped
    assert "presence_penalty" not in mapped
    assert mapped["max_output_tokens"] == 256

    for unsupported_kwarg in ({"n": 2}, {"frequency_penalty": 0.5}, {"presence_penalty": 0.3}):
        with pytest.raises(UnsupportedParamsError):
            litellm.utils.get_optional_params(
                model=model,
                custom_llm_provider=provider,
                drop_params=False,
                **unsupported_kwarg,
            )


@pytest.mark.parametrize("config_cls", [VertexGeminiConfig, GoogleAIStudioGeminiConfig])
def test_pre_gemini_4_keeps_candidate_count_support(config_cls: type[VertexGeminiConfig]) -> None:
    config = config_cls()
    for model in ("gemini-3.1-pro-preview", "gemini-2.5-flash"):
        supported = config.get_supported_openai_params(model)
        assert "n" in supported
        mapped = config.map_openai_params(
            non_default_params={"n": 3},
            optional_params={},
            model=model,
            drop_params=False,
        )
        assert mapped["candidate_count"] == 3


def test_google_ai_studio_gemini_3_excludes_penalty_parameters() -> None:
    config = GoogleAIStudioGeminiConfig()
    supported = config.get_supported_openai_params("gemini-3.1-pro-preview")
    assert "frequency_penalty" not in supported
    assert "presence_penalty" not in supported


@pytest.mark.usefixtures("local_model_cost_map")
def test_gemini_4_argon_grounding_bills_tool_use_prompt_tokens_and_search_queries() -> None:
    """
    Gemini 4 Argon bills every token the model processes, including internal tool system
    instructions and function declarations when using grounding, in addition to the per-query
    Google Search grounding fee.
    """
    completion_response = {
        "modelVersion": "gemini-4-argon",
        "candidates": [
            {
                "content": {"parts": [{"text": "Grounded answer"}], "role": "model"},
                "finishReason": "STOP",
                "groundingMetadata": {"webSearchQueries": ["gemini 4 argon"]},
            }
        ],
        "usageMetadata": UsageMetadata(
            promptTokenCount=50,
            candidatesTokenCount=100,
            toolUsePromptTokenCount=300,
            totalTokenCount=450,
            promptTokensDetails=[{"modality": "TEXT", "tokenCount": 50}],
            toolUsePromptTokensDetails=[{"modality": "TEXT", "tokenCount": 300}],
        ),
    }

    raw_response = MagicMock()
    raw_response.text = "{}"
    raw_response.headers = {}

    response = VertexGeminiConfig()._transform_google_generate_content_to_openai_model_response(
        completion_response=completion_response,
        model_response=ModelResponse(),
        model="gemini-4-argon",
        logging_obj=MagicMock(optional_params={}),
        raw_response=raw_response,
    )

    usage = response.usage
    assert usage.prompt_tokens == 50 + 300
    assert usage.prompt_tokens_details.text_tokens == 50 + 300
    assert usage.prompt_tokens_details.tool_use_tokens == 300
    assert usage.prompt_tokens_details.web_search_requests == 1
