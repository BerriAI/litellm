import json
from pathlib import Path

import pytest

from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


@pytest.mark.parametrize(
    "model,expected_provider",
    [
        # gpt-5.5 is a known OpenAI model name, so azure_ai/gpt-5.5 is routed
        # through the Azure OpenAI path (custom_llm_provider="azure").
        ("azure_ai/gpt-5.5", "azure"),
        # 2026-04-24 is Azure's own GPT-5.5 snapshot (OpenAI's is 2026-04-23),
        # so it is not an OpenAI model name and stays on the azure_ai path.
        ("azure_ai/gpt-5.5-2026-04-24", "azure_ai"),
    ],
)
def test_azure_ai_gpt_5_5_model_info(model, expected_provider):
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    with open(json_path) as f:
        model_cost = json.load(f)

    info = model_cost.get(model)
    assert (
        info is not None
    ), f"{model} not found in model_prices_and_context_window.json"

    assert info["litellm_provider"] == "azure_ai"
    assert info["mode"] == "chat"

    assert info["input_cost_per_token"] == 5e-06
    assert info["output_cost_per_token"] == 3e-05
    assert info["cache_read_input_token_cost"] == 5e-07

    assert info["input_cost_per_token_above_272k_tokens"] == 1e-05
    assert info["output_cost_per_token_above_272k_tokens"] == 4.5e-05
    assert info["cache_read_input_token_cost_above_272k_tokens"] == 1e-06

    assert info["input_cost_per_token_priority"] == 1e-05
    assert info["output_cost_per_token_priority"] == 6e-05

    assert info["max_input_tokens"] == 1050000
    assert info["max_output_tokens"] == 128000
    assert info["max_tokens"] == 128000

    assert info["supports_function_calling"] is True
    assert info["supports_prompt_caching"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_vision"] is True
    assert info["supports_web_search"] is True
    # gpt-5.5 dropped minimal reasoning effort support (true on gpt-5.4)
    assert info["supports_minimal_reasoning_effort"] is False

    routed_model, provider, _, _ = get_llm_provider(model=model)
    assert routed_model == model.split("/", 1)[1]
    assert provider == expected_provider


def test_azure_ai_gpt_5_5_backup_matches_main():
    """Ensure the bundled model cost map stays in sync with the canonical file."""
    repo_root = Path(__file__).parents[2]
    main_path = repo_root / "model_prices_and_context_window.json"
    backup_path = repo_root / "litellm" / "model_prices_and_context_window_backup.json"

    with open(main_path) as f:
        main_cost = json.load(f)
    with open(backup_path) as f:
        backup_cost = json.load(f)

    for model in ("azure_ai/gpt-5.5", "azure_ai/gpt-5.5-2026-04-24"):
        assert backup_cost.get(model) == main_cost.get(
            model
        ), f"{model} differs between main and backup model cost maps"
