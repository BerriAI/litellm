from typing import Final

import pytest

from litellm import get_model_info
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig

MODELS: Final = (
    ("gemini/gemini-4-argon", "gemini-4-argon", "gemini"),
    ("vertex_ai/gemini-4-argon", "gemini-4-argon", "vertex_ai"),
)


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
