"""Metadata regression test for dashscope/qwen3.8-max (Qwen3.8 Max, released 2026-08-03)."""

import json
from importlib.resources import files
import pytest

MODEL = "dashscope/qwen3.8-max"


@pytest.fixture(scope="module")
def local_cost_map():
    mp = pytest.MonkeyPatch()
    mp.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import litellm
    from litellm.utils import _invalidate_model_cost_lowercase_map

    orig = litellm.model_cost
    litellm.model_cost = json.loads(
        files("litellm").joinpath("model_prices_and_context_window_backup.json").read_text(encoding="utf-8")
    )
    litellm.get_model_info.cache_clear()
    _invalidate_model_cost_lowercase_map()
    try:
        yield litellm
    finally:
        litellm.model_cost = orig
        litellm.get_model_info.cache_clear()
        _invalidate_model_cost_lowercase_map()
        mp.undo()


def test_qwen38_max_raw_entry(local_cost_map):
    e = local_cost_map.model_cost[MODEL]
    assert e["litellm_provider"] == "dashscope"
    assert e["max_input_tokens"] == 1000000
    assert e["max_output_tokens"] == 131072
    assert e["input_cost_per_token"] == 2e-06
    assert e["output_cost_per_token"] == 6e-06
    assert e["supports_tool_choice"] is True
    assert e["supports_function_calling"] is True


def test_qwen38_max_get_model_info(local_cost_map):
    info = local_cost_map.get_model_info(model=MODEL)
    assert info["max_input_tokens"] == 1000000
    assert info["max_output_tokens"] == 131072
    assert info["supports_tool_choice"] is True
