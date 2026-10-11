from pathlib import Path
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm import get_model_info
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig

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
    assert info["mode"] == "chat"
    assert info["max_input_tokens"] == 1_048_576
    assert info["max_output_tokens"] == 65_536
    assert info["supports_reasoning"] is True
    assert info["supports_function_calling"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_native_streaming"] is True
    assert VertexGeminiConfig.is_gemini_3_or_newer(routed_model) is True


@pytest.mark.parametrize("model", [row[0] for row in MODELS])
def test_gemini_4_argon_entry_source_and_backup_match(model: str) -> None:
    main_entry = _cost_map_entry(REPO_ROOT / "model_prices_and_context_window.json", model)
    backup_entry = _cost_map_entry(REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json", model)

    assert backup_entry == main_entry


def test_gemini_4_argon_vertex_row_matches_the_ai_studio_row_modulo_provider_and_source() -> None:
    cost_map = COST_MAP_ADAPTER.validate_json((REPO_ROOT / "model_prices_and_context_window.json").read_bytes())
    ai_studio: Final = cost_map["gemini/gemini-4-argon"]
    vertex: Final = cost_map["vertex_ai/gemini-4-argon"]

    assert vertex["litellm_provider"] == "vertex_ai"
    assert vertex["source"] != ai_studio["source"]
    assert {**vertex, "litellm_provider": "gemini", "source": ai_studio["source"]} == ai_studio
