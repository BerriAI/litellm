from pathlib import Path
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm import cost_per_token, get_model_info
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

REPO_ROOT: Final = Path(__file__).parents[2]
MODEL: Final = "azure_ai/grok-4.6"
SOURCE: Final = (
    "https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/"
    "grok-4-6-comes-to-microsoft-foundry-models-built-for-long-horizon-reasoning-and-/4547578"
)
COST_MAP_ADAPTER: Final = TypeAdapter(dict[str, dict[str, object]])


def _cost_map_entry(path: Path) -> dict[str, object]:
    return COST_MAP_ADAPTER.validate_json(path.read_bytes())[MODEL]


@pytest.mark.usefixtures("local_model_cost_map")
def test_azure_ai_grok_4_6_is_priced_and_routed() -> None:
    routed_model, provider, _, _ = get_llm_provider(model=MODEL)
    assert (routed_model, provider) == ("grok-4.6", "azure_ai")

    info = get_model_info(model=routed_model, custom_llm_provider=provider)
    assert info["litellm_provider"] == "azure_ai"
    assert info["mode"] == "chat"
    assert info["input_cost_per_token"] == 2e-06
    assert info["output_cost_per_token"] == 6e-06
    assert info["cache_read_input_token_cost"] == 5e-07
    assert info["max_input_tokens"] == 200000
    assert info["max_output_tokens"] == 128000
    assert info["max_tokens"] == 128000
    assert info["supports_function_calling"] is True
    assert info["supports_prompt_caching"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_vision"] is True
    assert info["supports_web_search"] is True

    prompt_cost, completion_cost = cost_per_token(model=MODEL, prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert prompt_cost == pytest.approx(2.0)
    assert completion_cost == pytest.approx(6.0)


def test_azure_ai_grok_4_6_entry_source_and_backup_match() -> None:
    main_entry = _cost_map_entry(REPO_ROOT / "model_prices_and_context_window.json")
    backup_entry = _cost_map_entry(REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json")

    assert main_entry["source"] == SOURCE
    assert backup_entry == main_entry
