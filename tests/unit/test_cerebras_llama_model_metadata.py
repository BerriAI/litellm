# What does this test?
"""
Verifies that Cerebras Llama 3.1 models (standard hyphenated aliases):
- cerebras/llama-3.1-8b
- cerebras/llama-3.1-70b

have correct rate card metadata, context window limits, and cost calculation parity
with the un-hyphenated variants in model_prices_and_context_window.json.
Fixes: https://github.com/BerriAI/litellm/issues/48157
"""

import json
from pathlib import Path

import pytest

import litellm


@pytest.fixture(scope="module")
def model_cost_maps():
    root = Path(__file__).resolve().parents[2]
    main_path = root / "model_prices_and_context_window.json"
    backup_path = root / "litellm" / "model_prices_and_context_window_backup.json"

    with open(main_path, "r", encoding="utf-8") as f:
        main_map = json.load(f)
    with open(backup_path, "r", encoding="utf-8") as f:
        backup_map = json.load(f)

    return main_map, backup_map


CEREBRAS_MODELS_EXPECTED = {
    "cerebras/llama-3.1-8b": {
        "input_cost_per_token": 1e-07,
        "output_cost_per_token": 1e-07,
        "max_input_tokens": 128000,
        "max_output_tokens": 128000,
        "mode": "chat",
        "litellm_provider": "cerebras",
    },
    "cerebras/llama-3.1-70b": {
        "input_cost_per_token": 6e-07,
        "output_cost_per_token": 6e-07,
        "max_input_tokens": 128000,
        "max_output_tokens": 128000,
        "mode": "chat",
        "litellm_provider": "cerebras",
    },
}

PAIR_PARITY = [
    ("cerebras/llama-3.1-8b", "cerebras/llama3.1-8b"),
    ("cerebras/llama-3.1-70b", "cerebras/llama3.1-70b"),
]


@pytest.mark.parametrize("model_id,expected_data", CEREBRAS_MODELS_EXPECTED.items())
def test_cerebras_llama_model_metadata(model_cost_maps, model_id, expected_data):
    """Test that hyphenated Cerebras Llama 3.1 models exist and have expected metadata in both catalogs."""
    main_map, backup_map = model_cost_maps

    assert model_id in main_map, f"Missing {model_id} in main catalog"
    assert model_id in backup_map, f"Missing {model_id} in backup catalog"

    main_entry = main_map[model_id]
    backup_entry = backup_map[model_id]
    assert main_entry == backup_entry, f"Parity mismatch between main and backup for {model_id}"

    for key, expected_val in expected_data.items():
        assert main_entry.get(key) == expected_val, (
            f"{model_id}[{key}] got {main_entry.get(key)}, expected {expected_val}"
        )


@pytest.mark.parametrize("hyphenated,unhyphenated", PAIR_PARITY)
def test_cerebras_llama_parity_with_unhyphenated(model_cost_maps, hyphenated, unhyphenated):
    """Test that hyphenated and un-hyphenated variants have identical pricing and context limits."""
    main_map, _ = model_cost_maps

    assert hyphenated in main_map
    assert unhyphenated in main_map

    entry_hyphen = main_map[hyphenated]
    entry_unhyphen = main_map[unhyphenated]

    assert entry_hyphen["input_cost_per_token"] == entry_unhyphen["input_cost_per_token"]
    assert entry_hyphen["output_cost_per_token"] == entry_unhyphen["output_cost_per_token"]
    assert entry_hyphen["max_input_tokens"] == entry_unhyphen["max_input_tokens"]
    assert entry_hyphen["max_output_tokens"] == entry_unhyphen["max_output_tokens"]


def test_cerebras_llama_cost_per_token():
    """Test that litellm.cost_per_token accurately computes costs for hyphenated Cerebras models."""
    prompt_tokens = 1_000_000
    completion_tokens = 1_000_000

    # llama-3.1-8b: $0.10 input + $0.10 output per 1M tokens
    cost_prompt, cost_comp = litellm.cost_per_token(
        model="cerebras/llama-3.1-8b",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    assert pytest.approx(cost_prompt, rel=1e-5) == 0.10
    assert pytest.approx(cost_comp, rel=1e-5) == 0.10

    # llama-3.1-70b: $0.60 input + $0.60 output per 1M tokens
    cost_prompt_70b, cost_comp_70b = litellm.cost_per_token(
        model="cerebras/llama-3.1-70b",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    assert pytest.approx(cost_prompt_70b, rel=1e-5) == 0.60
    assert pytest.approx(cost_comp_70b, rel=1e-5) == 0.60
