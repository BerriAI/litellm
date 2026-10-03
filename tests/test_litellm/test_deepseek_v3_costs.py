"""
Tests for deepseek-v3 model prices and context window configuration.

Closes https://github.com/BerriAI/litellm/issues/43660
"""

import json
from pathlib import Path

import pytest

import litellm
from litellm import completion_cost
from litellm.types.utils import Choices, Message, ModelResponse, Usage
from litellm.utils import get_model_info

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

EXPECTED_CONFIG = {
    "litellm_provider": "deepseek",
    "mode": "chat",
    "input_cost_per_token": 2.7e-07,
    "output_cost_per_token": 1.1e-06,
    "cache_read_input_token_cost": 7e-08,
    "max_input_tokens": 65536,
    "max_output_tokens": 8192,
    "supports_function_calling": True,
    "supports_tool_choice": True,
    "supports_assistant_prefill": True,
    "supports_prompt_caching": True,
}


@pytest.fixture(autouse=True)
def local_model_cost_map(monkeypatch):
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()


def test_deepseek_v3_in_cost_map():
    with open(MAIN_PATH, encoding="utf-8") as f:
        model_cost = json.load(f)

    for key in ("deepseek-v3", "deepseek/deepseek-v3"):
        info = model_cost.get(key)
        assert info is not None, f"{key} missing from model_prices_and_context_window.json"
        for field, expected_val in EXPECTED_CONFIG.items():
            assert info.get(field) == expected_val, f"Mismatch for {field} in {key}"


def test_deepseek_v3_in_backup_cost_map():
    with open(BACKUP_PATH, encoding="utf-8") as f:
        model_cost = json.load(f)

    for key in ("deepseek-v3", "deepseek/deepseek-v3"):
        info = model_cost.get(key)
        assert info is not None, f"{key} missing from backup JSON"
        for field, expected_val in EXPECTED_CONFIG.items():
            assert info.get(field) == expected_val, f"Mismatch for {field} in {key}"


def test_deepseek_v3_get_model_info():
    info = get_model_info(model="deepseek-v3")
    assert info is not None
    assert info["litellm_provider"] == "deepseek"
    assert info["input_cost_per_token"] == pytest.approx(2.7e-07)
    assert info["output_cost_per_token"] == pytest.approx(1.1e-06)
    assert info["cache_read_input_token_cost"] == pytest.approx(7e-08)
    assert info["max_input_tokens"] == 65536
    assert info["max_output_tokens"] == 8192


def test_deepseek_v3_completion_cost():
    response = ModelResponse(
        model="deepseek-v3",
        choices=[Choices(index=0, message=Message(role="assistant", content="test"))],
        usage=Usage(
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            total_tokens=2_000_000,
        ),
    )

    cost_bare = completion_cost(
        completion_response=response,
        model="deepseek-v3",
        custom_llm_provider="deepseek",
    )
    assert cost_bare == pytest.approx(1.37, abs=1e-9)

    response_prefixed = ModelResponse(
        model="deepseek/deepseek-v3",
        choices=[Choices(index=0, message=Message(role="assistant", content="test"))],
        usage=Usage(
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            total_tokens=2_000_000,
        ),
    )
    cost_prefixed = completion_cost(
        completion_response=response_prefixed,
        model="deepseek/deepseek-v3",
    )
    assert cost_prefixed == pytest.approx(1.37, abs=1e-9)
