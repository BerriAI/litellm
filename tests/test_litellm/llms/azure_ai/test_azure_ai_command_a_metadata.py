"""
Test Azure AI Cohere Command A model metadata.
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
        files("litellm")
        .joinpath("model_prices_and_context_window_backup.json")
        .read_text(encoding="utf-8")
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


def test_azure_ai_command_a_model_info(use_local_model_cost_map):
    model_info = use_local_model_cost_map.get_model_info(model="azure_ai/cohere-command-a-03-2025")

    assert model_info["litellm_provider"] == "azure_ai"
    assert model_info["mode"] == "chat"
    assert model_info["max_input_tokens"] == 256000
    assert model_info["max_output_tokens"] == 8000
    assert model_info["max_tokens"] == 8000
    assert model_info["input_cost_per_token"] == pytest.approx(2.5e-06)
    assert model_info["output_cost_per_token"] == pytest.approx(1e-05)
    assert model_info["supports_function_calling"] is True
    assert model_info["supports_tool_choice"] is True


def test_azure_ai_command_a_cost_per_token(use_local_model_cost_map):
    from litellm.llms.azure_ai.cost_calculator import cost_per_token
    from litellm.types.utils import Usage

    usage = Usage(
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        total_tokens=2_000_000,
    )

    prompt_cost, completion_cost = cost_per_token(model="cohere-command-a-03-2025", usage=usage)

    assert prompt_cost == pytest.approx(2.5)
    assert completion_cost == pytest.approx(10.0)
