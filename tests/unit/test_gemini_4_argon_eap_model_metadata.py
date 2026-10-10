from pathlib import Path
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm import get_model_info
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig

REPO_ROOT: Final = Path(__file__).parents[2]
MODEL: Final = "gemini/gemini-4-argon-eap"
COST_MAP_ADAPTER: Final = TypeAdapter(dict[str, dict[str, object]])


def _cost_map_entry(path: Path) -> dict[str, object]:
    return COST_MAP_ADAPTER.validate_json(path.read_bytes())[MODEL]


@pytest.mark.usefixtures("local_model_cost_map")
def test_gemini_4_argon_eap_routes_to_gemini_on_the_thinking_level_path() -> None:
    routed_model, provider, _, _ = get_llm_provider(model=MODEL)
    assert (routed_model, provider) == ("gemini-4-argon-eap", "gemini")

    info = get_model_info(model=routed_model, custom_llm_provider=provider)
    assert info["litellm_provider"] == "gemini"
    assert info["mode"] == "chat"
    assert info["max_input_tokens"] == 1_048_576
    assert info["max_output_tokens"] == 65_536
    assert info["supports_reasoning"] is True
    assert info["supports_function_calling"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_native_streaming"] is True
    assert VertexGeminiConfig.is_gemini_3_or_newer(routed_model) is True


def test_gemini_4_argon_eap_entry_source_and_backup_match() -> None:
    main_entry = _cost_map_entry(REPO_ROOT / "model_prices_and_context_window.json")
    backup_entry = _cost_map_entry(REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json")

    assert backup_entry == main_entry
