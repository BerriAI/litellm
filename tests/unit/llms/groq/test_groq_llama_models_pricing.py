"""Unit tests and pricing verification for Groq Llama and Mixtral models.

This module validates that LiteLLM correctly maps Groq's official flagship models
in `model_prices_and_context_window.json` and its backup, preventing ModelNotMappedError
and ensuring exact token cost calculation.

Supported Models & Specifications (Reference: https://console.groq.com/docs/models):
- groq/llama-3.3-70b-versatile: 131,072 max input tokens, 32,768 max output tokens ($0.59 / $0.79 per 1M tokens)
- groq/llama-3.1-70b-versatile: 131,072 max input tokens, 32,768 max output tokens ($0.59 / $0.79 per 1M tokens)
- groq/llama-3.1-8b-instant:    131,072 max input tokens, 8,192 max output tokens  ($0.05 / $0.08 per 1M tokens)
- groq/llama-3.2-1b-preview:    131,072 max input tokens, 8,192 max output tokens  ($0.04 / $0.04 per 1M tokens)
- groq/llama-3.2-3b-preview:    131,072 max input tokens, 8,192 max output tokens  ($0.06 / $0.06 per 1M tokens)
- groq/llama-3.2-11b-vision-preview: 8,192 max tokens ($0.18 / $0.18 per 1M tokens, supports vision)
- groq/llama-3.2-90b-vision-preview: 8,192 max tokens ($0.79 / $0.79 per 1M tokens, supports vision)
- groq/mixtral-8x7b-32768:      32,768 max tokens     ($0.24 / $0.24 per 1M tokens)
"""

import pytest

import litellm


@pytest.fixture(autouse=True)
def setup_local_model_cost(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

    monkeypatch.setattr(litellm, "model_cost", GetModelCostMap.load_local_model_cost_map())


@pytest.mark.parametrize(
    "model_name,expected_input_cost,expected_output_cost,expected_max_tokens,expected_dep_date",
    [
        ("groq/llama-3.3-70b-versatile", 5.9e-07, 7.9e-07, 32768, "2026-08-16"),
        ("groq/llama-3.1-70b-versatile", 5.9e-07, 7.9e-07, 32768, "2025-01-24"),
        ("groq/llama-3.1-8b-instant", 5.0e-08, 8.0e-08, 8192, "2026-08-16"),
        ("groq/llama-3.2-1b-preview", 4.0e-08, 4.0e-08, 8192, "2025-04-14"),
        ("groq/llama-3.2-3b-preview", 6.0e-08, 6.0e-08, 8192, "2025-04-14"),
        ("groq/llama-3.2-11b-vision-preview", 1.8e-07, 1.8e-07, 8192, "2025-04-14"),
        ("groq/llama-3.2-90b-vision-preview", 7.9e-07, 7.9e-07, 8192, "2025-04-14"),
        ("groq/mixtral-8x7b-32768", 2.4e-07, 2.4e-07, 32768, "2025-03-20"),
    ],
)
def test_groq_llama_models_info_and_pricing(
    model_name: str,
    expected_input_cost: float,
    expected_output_cost: float,
    expected_max_tokens: int,
    expected_dep_date: str,
):
    model_info = litellm.get_model_info(model_name)
    assert model_info is not None
    assert model_info.get("litellm_provider") == "groq"
    assert model_info.get("mode") == "chat"
    assert model_info.get("max_tokens") == expected_max_tokens

    # Verify deprecation schedule metadata
    model_cost_entry = litellm.model_cost.get(model_name)
    assert model_cost_entry is not None
    assert model_cost_entry.get("deprecation_date") == expected_dep_date
    assert model_cost_entry.get("source") == "https://console.groq.com/docs/deprecations"

    # Verify token cost calculation
    prompt_tokens = 1000
    completion_tokens = 500
    expected_cost = (prompt_tokens * expected_input_cost) + (completion_tokens * expected_output_cost)

    cost = litellm.completion_cost(
        completion_response={
            "model": model_name,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }
    )
    assert cost == pytest.approx(expected_cost)
