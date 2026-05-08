import json
from pathlib import Path

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


def test_openai_chat_latest_model_info():
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    with open(json_path) as f:
        model_cost = json.load(f)
    litellm.add_known_models(model_cost)

    info = model_cost.get("chat-latest")
    assert (
        info is not None
    ), "chat-latest not found in model_prices_and_context_window.json"

    assert info["litellm_provider"] == "openai"
    assert info["mode"] == "chat"

    assert info["input_cost_per_token"] == 5e-06
    assert info["output_cost_per_token"] == 3e-05
    assert info["cache_read_input_token_cost"] == 5e-07

    assert info["max_input_tokens"] == 272000
    assert info["max_output_tokens"] == 128000
    assert info["max_tokens"] == 128000

    assert info["supported_endpoints"] == [
        "/v1/chat/completions",
        "/v1/responses",
    ]
    assert info["supports_function_calling"] is True
    assert info["supports_parallel_function_calling"] is True
    assert info["supports_prompt_caching"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_system_messages"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_vision"] is True
    assert info["supports_web_search"] is True

    routed_model, provider, _, _ = get_llm_provider(model="chat-latest")
    assert routed_model == "chat-latest"
    assert provider == "openai"


def test_openai_chat_latest_backup_matches_main():
    repo_root = Path(__file__).parents[2]
    main_path = repo_root / "model_prices_and_context_window.json"
    backup_path = repo_root / "litellm" / "model_prices_and_context_window_backup.json"

    with open(main_path) as f:
        main_cost = json.load(f)
    with open(backup_path) as f:
        backup_cost = json.load(f)

    assert "chat-latest" in main_cost
    assert "chat-latest" in backup_cost
    assert backup_cost.get("chat-latest") == main_cost.get("chat-latest")
