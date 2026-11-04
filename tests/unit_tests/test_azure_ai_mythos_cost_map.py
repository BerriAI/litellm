"""
Regression test for https://github.com/BerriAI/litellm/issues/49865

`azure_ai/claude-mythos-5-1` had no entry in the model cost map. The bare
`claude-mythos-5-1` entry belongs to the `anthropic` provider and is rejected by
the azure_ai provider lookup, so `cost_per_token()` raised `ModelNotMappedError`
and the proxy recorded zero spend for a model that actually bills.

This test asserts the dedicated azure_ai entry resolves and returns non-zero
input/output costs, mirroring the issue's reproduction (no credentials needed).
"""
import pytest

from litellm import cost_per_token, get_model_info


def test_azure_ai_claude_mythos_5_1_get_model_info():
    info = get_model_info("claude-mythos-5-1", custom_llm_provider="azure_ai")
    assert info["litellm_provider"] == "azure_ai"
    assert info["input_cost_per_token"] > 0
    assert info["output_cost_per_token"] > 0


def test_azure_ai_claude_mythos_5_1_cost_per_token():
    spend = cost_per_token(
        model="claude-mythos-5-1",
        custom_llm_provider="azure_ai",
        prompt_tokens=1271,
        completion_tokens=389,
    )
    try:
        prompt_cost, completion_cost = spend
    except (TypeError, ValueError):
        prompt_cost = spend["prompt_cost"]
        completion_cost = spend["completion_cost"]
    assert prompt_cost > 0
    assert completion_cost > 0
