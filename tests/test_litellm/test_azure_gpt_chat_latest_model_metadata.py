import json
from pathlib import Path

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


MODEL = "azure/gpt-chat-latest"
SOURCE = "https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-openais-newest-chat-model-in-microsoft-foundry/4516848"


def _load_model_cost(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _assert_gpt_chat_latest_metadata(info: dict) -> None:
    assert info["litellm_provider"] == "azure"
    assert info["mode"] == "chat"

    assert info["input_cost_per_token"] == 5e-06
    assert info["cache_read_input_token_cost"] == 5e-07
    assert info["output_cost_per_token"] == 3e-05

    assert info["max_input_tokens"] == 128000
    assert info["max_output_tokens"] == 16384
    assert info["max_tokens"] == 16384

    assert info["supports_function_calling"] is True
    assert info["supports_native_streaming"] is True
    assert info["supports_pdf_input"] is True
    assert info["supports_prompt_caching"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_system_messages"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_vision"] is True


def test_azure_gpt_chat_latest_model_info() -> None:
    repo_root = Path(__file__).parents[2]
    model_cost = _load_model_cost(repo_root / "model_prices_and_context_window.json")

    info = model_cost.get(MODEL)
    assert info is not None
    assert info["source"] == SOURCE
    assert info["supported_endpoints"] == [
        "/v1/chat/completions",
        "/v1/batch",
        "/v1/responses",
    ]
    assert info["supported_modalities"] == ["text", "image"]
    assert info["supported_output_modalities"] == ["text"]
    assert info["supports_parallel_function_calling"] is True
    _assert_gpt_chat_latest_metadata(info)

    routed_model, provider, _, _ = get_llm_provider(model=MODEL)
    assert routed_model == "gpt-chat-latest"
    assert provider == "azure"


def test_azure_gpt_chat_latest_get_model_info(monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    info = litellm.get_model_info(model="gpt-chat-latest", custom_llm_provider="azure")
    _assert_gpt_chat_latest_metadata(info)
    assert info["key"] == MODEL


def test_azure_gpt_chat_latest_backup_matches_main() -> None:
    repo_root = Path(__file__).parents[2]
    main_cost = _load_model_cost(repo_root / "model_prices_and_context_window.json")
    backup_cost = _load_model_cost(
        repo_root / "litellm" / "model_prices_and_context_window_backup.json"
    )

    assert backup_cost.get(MODEL) == main_cost.get(MODEL)
