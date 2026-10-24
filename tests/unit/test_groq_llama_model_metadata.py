"""Tests for Groq Llama 3 model metadata in model_prices_and_context_window.json."""

import json
from pathlib import Path

import pytest

import litellm

REPO_ROOT = Path(__file__).parents[2]
MAIN_COST_MAP = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_COST_MAP = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

GROQ_LLAMA_MODELS = [
    "groq/llama-3.1-8b-instant",
    "groq/llama-3.3-70b-specdec",
    "groq/llama-3.3-70b-versatile",
]


@pytest.fixture(autouse=True)
def local_model_cost_map(monkeypatch):
    """Ensure tests run against local repository cost map."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


@pytest.mark.parametrize("model", GROQ_LLAMA_MODELS)
def test_groq_llama_metadata_in_main_and_backup(model: str):
    """Verify each model exists in both main and backup JSON files and matches."""
    main_data = json.loads(MAIN_COST_MAP.read_text())
    backup_data = json.loads(BACKUP_COST_MAP.read_text())

    assert model in main_data, f"{model} missing from {MAIN_COST_MAP.name}"
    assert model in backup_data, f"{model} missing from {BACKUP_COST_MAP.name}"
    assert main_data[model] == backup_data[model]

    info = litellm.get_model_info(model)
    assert info["litellm_provider"] == "groq"
    assert info["mode"] == "chat"
    assert info["supports_function_calling"] is True
    assert info["supports_parallel_function_calling"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_tool_choice"] is True


def test_groq_llama_3_1_8b_instant():
    """Verify specific rates and context window for Llama 3.1 8B Instant on Groq."""
    info = litellm.get_model_info("groq/llama-3.1-8b-instant")
    assert info["max_input_tokens"] == 131072
    assert info["max_output_tokens"] == 8192
    assert info["max_tokens"] == 8192
    assert info["input_cost_per_token"] == 5e-08
    assert info["output_cost_per_token"] == 8e-08

    # Calculate cost for 1,000,000 prompt + 1,000,000 completion tokens
    prompt_cost, completion_cost = litellm.cost_per_token(
        model="groq/llama-3.1-8b-instant",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    assert abs(prompt_cost - 0.05) < 1e-6
    assert abs(completion_cost - 0.08) < 1e-6


def test_groq_llama_3_3_70b_versatile():
    """Verify specific rates and context window for Llama 3.3 70B Versatile on Groq."""
    info = litellm.get_model_info("groq/llama-3.3-70b-versatile")
    assert info["max_input_tokens"] == 131072
    assert info["max_output_tokens"] == 32768
    assert info["max_tokens"] == 32768
    assert info["input_cost_per_token"] == 5.9e-07
    assert info["output_cost_per_token"] == 7.9e-07

    # Calculate cost for 1,000,000 prompt + 1,000,000 completion tokens
    prompt_cost, completion_cost = litellm.cost_per_token(
        model="groq/llama-3.3-70b-versatile",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    assert abs(prompt_cost - 0.59) < 1e-6
    assert abs(completion_cost - 0.79) < 1e-6


def test_groq_llama_3_3_70b_specdec():
    """Verify specific rates and context window for Llama 3.3 70B Speculative Decoding on Groq."""
    info = litellm.get_model_info("groq/llama-3.3-70b-specdec")
    assert info["max_input_tokens"] == 8192
    assert info["max_output_tokens"] == 8192
    assert info["max_tokens"] == 8192
    assert info["input_cost_per_token"] == 5.9e-07
    assert info["output_cost_per_token"] == 9.9e-07

    prompt_cost, completion_cost = litellm.cost_per_token(
        model="groq/llama-3.3-70b-specdec",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    assert abs(prompt_cost - 0.59) < 1e-6
    assert abs(completion_cost - 0.99) < 1e-6
