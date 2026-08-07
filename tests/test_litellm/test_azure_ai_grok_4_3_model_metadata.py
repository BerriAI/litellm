import json
from pathlib import Path

import pytest

import litellm
from litellm import get_model_info
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


AZURE_AI_GROK_4_3_MODEL = "azure_ai/grok-4.3"
AZURE_AI_GROK_4_3_SOURCE = "https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-grok-4-3-on-microsoft-foundry-latest-generation-agentic-capabilities/4517096"


def _load_model_cost(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@pytest.fixture(autouse=True)
def reload_model_costs():
    original_model_cost = litellm.model_cost
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    litellm.model_cost = _load_model_cost(json_path)
    get_model_info.cache_clear()
    yield
    litellm.model_cost = original_model_cost
    get_model_info.cache_clear()


def test_azure_ai_grok_4_3_model_info():
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    model_cost = _load_model_cost(json_path)

    info = model_cost.get(AZURE_AI_GROK_4_3_MODEL)
    assert (
        info is not None
    ), f"{AZURE_AI_GROK_4_3_MODEL} not found in model_prices_and_context_window.json"

    assert info["litellm_provider"] == "azure_ai"
    assert info["mode"] == "chat"

    assert info["input_cost_per_token"] == 1.25e-06
    assert info["output_cost_per_token"] == 2.5e-06
    assert info["cache_read_input_token_cost"] == 2e-07

    assert info["max_input_tokens"] == 200000
    assert info["max_output_tokens"] == 200000
    assert info["max_tokens"] == 200000
    assert info["source"] == AZURE_AI_GROK_4_3_SOURCE

    assert info["supports_function_calling"] is True
    assert info["supports_prompt_caching"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_vision"] is True
    assert info["supports_web_search"] is True

    routed_model, provider, _, _ = get_llm_provider(model=AZURE_AI_GROK_4_3_MODEL)
    assert routed_model == "grok-4.3"
    assert provider == "azure_ai"

    resolved_info = get_model_info(model="grok-4.3", custom_llm_provider="azure_ai")
    assert resolved_info["litellm_provider"] == "azure_ai"
    assert resolved_info["input_cost_per_token"] == info["input_cost_per_token"]
    assert resolved_info["output_cost_per_token"] == info["output_cost_per_token"]
    assert (
        resolved_info["cache_read_input_token_cost"]
        == info["cache_read_input_token_cost"]
    )


def test_azure_ai_grok_4_3_backup_matches_main():
    repo_root = Path(__file__).parents[2]
    main_path = repo_root / "model_prices_and_context_window.json"
    backup_path = repo_root / "litellm" / "model_prices_and_context_window_backup.json"

    main_cost = _load_model_cost(main_path)
    backup_cost = _load_model_cost(backup_path)

    assert backup_cost.get(AZURE_AI_GROK_4_3_MODEL) == main_cost.get(
        AZURE_AI_GROK_4_3_MODEL
    )
