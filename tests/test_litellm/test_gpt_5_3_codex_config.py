"""
Validate GPT-5.3 Codex model configuration entries.
"""

import json
import os


def _load_model_data() -> dict:
    json_path = os.path.join(
        os.path.dirname(__file__), "../../model_prices_and_context_window.json"
    )
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def test_chatgpt_gpt_5_3_codex_entry_exists():
    model_data = _load_model_data()
    assert "chatgpt/gpt-5.3-codex" in model_data

    info = model_data["chatgpt/gpt-5.3-codex"]
    assert info["litellm_provider"] == "chatgpt"
    assert info["mode"] == "responses"
    assert info["max_input_tokens"] == 128000
    assert info["max_output_tokens"] == 128000
    assert info["max_tokens"] == 128000


def test_openai_gpt_5_3_codex_has_pricing():
    model_data = _load_model_data()
    assert "gpt-5.3-codex" in model_data

    info = model_data["gpt-5.3-codex"]
    assert info["litellm_provider"] == "openai"
    assert info["mode"] == "responses"
    assert info["input_cost_per_token"] == 1.75e-06
    assert info["output_cost_per_token"] == 1.4e-05
    assert info["cache_read_input_token_cost"] == 1.75e-07


def test_openai_gpt_5_3_codex_pricing_matches_gpt_5_2_codex():
    model_data = _load_model_data()

    baseline = model_data["gpt-5.2-codex"]
    target = model_data["gpt-5.3-codex"]

    keys_to_match = [
        "input_cost_per_token",
        "input_cost_per_token_priority",
        "output_cost_per_token",
        "output_cost_per_token_priority",
        "cache_read_input_token_cost",
        "cache_read_input_token_cost_priority",
    ]

    for key in keys_to_match:
        assert target[key] == baseline[key], f"Mismatch for {key}"
