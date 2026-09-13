"""
Test Azure AI Kimi K2.6 model metadata.
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


def test_azure_ai_kimi_k26_cost_per_token(use_local_model_cost_map):
    from litellm.llms.azure_ai.cost_calculator import cost_per_token
    from litellm.types.utils import Usage

    usage = Usage(
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        total_tokens=2_000_000,
    )

    prompt_cost, completion_cost = cost_per_token(model="kimi-k2.6", usage=usage)

    assert prompt_cost == pytest.approx(0.95)
    assert completion_cost == pytest.approx(4.0)
