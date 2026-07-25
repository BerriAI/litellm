"""
Test Groq qwen3.6-27b model metadata in the cost map.
"""

import json
from importlib.resources import files

import pytest


@pytest.fixture(scope="module")
def use_local_model_cost_map():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    import litellm
    from litellm.utils import _invalidate_model_cost_lowercase_map

    original_model_cost = litellm.model_cost
    litellm.model_cost = json.loads(
        files("litellm").joinpath("model_prices_and_context_window_backup.json").read_text(encoding="utf-8")
    )
    litellm.get_model_info.cache_clear()
    _invalidate_model_cost_lowercase_map()
    try:
        yield litellm
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()
        _invalidate_model_cost_lowercase_map()
        monkeypatch.undo()


def test_groq_qwen3_6_27b_model_info(use_local_model_cost_map):
    model_info = use_local_model_cost_map.get_model_info(model="groq/qwen/qwen3.6-27b")

    assert model_info["litellm_provider"] == "groq"
    assert model_info["mode"] == "chat"
    assert model_info["max_input_tokens"] == 131072
    assert model_info["max_output_tokens"] == 16384
    assert model_info["max_tokens"] == 16384
    assert model_info["input_cost_per_token"] == pytest.approx(6e-07)
    assert model_info["output_cost_per_token"] == pytest.approx(3e-06)
    assert model_info["cache_read_input_token_cost"] == pytest.approx(3e-07)
    assert model_info["supports_function_calling"] is True
    assert model_info["supports_reasoning"] is True
    assert model_info["supports_response_schema"] is True
    assert model_info["supports_tool_choice"] is True
    assert model_info["supports_vision"] is True


def test_groq_qwen3_6_27b_raw_model_cost_entry(use_local_model_cost_map):
    model_info = use_local_model_cost_map.model_cost["groq/qwen/qwen3.6-27b"]

    assert model_info["litellm_provider"] == "groq"
    assert model_info["mode"] == "chat"
    assert model_info["input_cost_per_token"] == pytest.approx(6e-07)
    assert model_info["output_cost_per_token"] == pytest.approx(3e-06)
    assert model_info["cache_read_input_token_cost"] == pytest.approx(3e-07)
    assert model_info["supports_vision"] is True
