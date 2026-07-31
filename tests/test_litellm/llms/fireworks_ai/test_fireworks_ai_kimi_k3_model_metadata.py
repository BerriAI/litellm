import json
from pathlib import Path

import pytest

import litellm
from litellm.llms.fireworks_ai.cost_calculator import cost_per_token
from litellm.types.utils import PromptTokensDetailsWrapper, Usage


@pytest.mark.parametrize(
    "model",
    [
        "fireworks_ai/accounts/fireworks/models/kimi-k3",
        "fireworks_ai/kimi-k3",
    ],
)
def test_kimi_k3_model_info(model):
    json_path = Path(__file__).parents[4] / "model_prices_and_context_window.json"
    with open(json_path) as f:
        model_cost = json.load(f)

    info = model_cost.get(model)
    assert info is not None, f"{model} not found in model_prices_and_context_window.json"

    assert info["litellm_provider"] == "fireworks_ai"
    assert info["mode"] == "chat"

    assert info["input_cost_per_token"] == 3e-06
    assert info["output_cost_per_token"] == 1.5e-05
    assert info["cache_read_input_token_cost"] == 3e-07

    assert info["max_input_tokens"] == 1040000
    assert info["max_output_tokens"] == 1040000
    assert info["max_tokens"] == 1040000

    assert info["supports_function_calling"] is True
    assert info["supports_vision"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_response_schema"] is True


def test_kimi_k3_backup_matches_main():
    repo_root = Path(__file__).parents[4]
    main_path = repo_root / "model_prices_and_context_window.json"
    backup_path = repo_root / "litellm" / "model_prices_and_context_window_backup.json"

    with open(main_path) as f:
        main_cost = json.load(f)
    with open(backup_path) as f:
        backup_cost = json.load(f)

    for model in (
        "fireworks_ai/accounts/fireworks/models/kimi-k3",
        "fireworks_ai/kimi-k3",
    ):
        assert backup_cost.get(model) == main_cost.get(model), (
            f"{model} differs between main and backup model cost maps"
        )


def _usage(prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> Usage:
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=cached_tokens),
    )


def test_kimi_k3_cached_prompt_tokens_billed_at_cache_read_rate():
    model = "accounts/fireworks/models/kimi-k3"
    repo_root = Path(__file__).parents[4]
    backup_path = repo_root / "litellm" / "model_prices_and_context_window_backup.json"
    with open(backup_path) as f:
        backup_cost = json.load(f)

    original = litellm.model_cost
    try:
        litellm.model_cost = backup_cost
        info = litellm.get_model_info(model=model, custom_llm_provider="fireworks_ai")
        input_cost = info["input_cost_per_token"]
        cache_read_cost = info["cache_read_input_token_cost"]
        output_cost = info["output_cost_per_token"]

        prompt_tokens = 10000
        cached_tokens = 9000
        completion_tokens = 50

        prompt_cost, completion_cost = cost_per_token(
            model=model, usage=_usage(prompt_tokens, cached_tokens, completion_tokens)
        )

        expected_prompt_cost = (prompt_tokens - cached_tokens) * input_cost + cached_tokens * cache_read_cost
        assert prompt_cost == pytest.approx(expected_prompt_cost)
        assert completion_cost == pytest.approx(completion_tokens * output_cost)
        assert prompt_cost < prompt_tokens * input_cost
    finally:
        litellm.model_cost = original
