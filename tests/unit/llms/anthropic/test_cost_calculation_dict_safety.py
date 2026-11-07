"""
Tests that ``get_cost_for_anthropic_web_search`` tolerates ``server_tool_use``
being either a ``dict`` or a ``ServerToolUse`` pydantic instance.

See https://github.com/BerriAI/litellm/issues/26153.
"""

import pytest

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import get_web_search_requests
from litellm.llms.anthropic.cost_calculation import cost_per_token, get_cost_for_anthropic_web_search
from litellm.types.utils import ModelInfo, ServerToolUse


class _UsageWithServerToolUse:
    def __init__(self, server_tool_use):
        self.server_tool_use = server_tool_use


def _make_model_info(cost_per_query: float = 0.01) -> ModelInfo:
    info: ModelInfo = {  # type: ignore[typeddict-item]
        "search_context_cost_per_query": {
            "search_context_size_low": cost_per_query,
            "search_context_size_medium": cost_per_query,
            "search_context_size_high": cost_per_query,
        }
    }
    return info


def test_get_web_search_requests_handles_none():
    assert get_web_search_requests(None) is None


def test_get_web_search_requests_handles_dict():
    assert get_web_search_requests({"web_search_requests": 4}) == 4


def test_get_web_search_requests_handles_dict_missing_key():
    assert get_web_search_requests({}) is None


def test_get_web_search_requests_handles_pydantic():
    assert get_web_search_requests(ServerToolUse(web_search_requests=2)) == 2


def test_get_cost_for_anthropic_web_search_with_dict_server_tool_use():
    """
    Regression: ``server_tool_use`` was a dict from ``stream_chunk_builder`` and
    direct attribute access on it raised ``AttributeError``.
    """
    usage = _UsageWithServerToolUse({"web_search_requests": 3})
    info = _make_model_info(cost_per_query=0.01)

    cost = get_cost_for_anthropic_web_search(
        model_info=info,
        usage=usage,  # type: ignore[arg-type]
    )

    assert cost == pytest.approx(0.03)


def test_get_cost_for_anthropic_web_search_with_pydantic_server_tool_use():
    usage = _UsageWithServerToolUse(ServerToolUse(web_search_requests=3))
    info = _make_model_info(cost_per_query=0.01)

    cost = get_cost_for_anthropic_web_search(
        model_info=info,
        usage=usage,  # type: ignore[arg-type]
    )

    assert cost == pytest.approx(0.03)


def test_get_cost_for_anthropic_web_search_with_none_server_tool_use():
    usage = _UsageWithServerToolUse(None)
    info = _make_model_info(cost_per_query=0.01)

    cost = get_cost_for_anthropic_web_search(
        model_info=info,
        usage=usage,  # type: ignore[arg-type]
    )

    assert cost == 0.0


def test_get_cost_for_anthropic_web_search_with_no_usage():
    info = _make_model_info(cost_per_query=0.01)
    cost = get_cost_for_anthropic_web_search(model_info=info, usage=None)
    assert cost == 0.0


def test_anthropic_cost_per_token_applies_deployment_prices_only_to_the_fallback_attempt_on_that_model():
    """Fallback chain usage from https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback (fetched
    2026-11-03), with a bio refusal, which bills the declined attempt too"""
    chain_usage = litellm.AnthropicConfig().calculate_usage(
        usage_object={
            "input_tokens": 412,
            "output_tokens": 264,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "iterations": [
                {"type": "message", "model": "claude-fable-5", "input_tokens": 535, "output_tokens": 0},
                {"type": "fallback_message", "model": "claude-opus-4-8", "input_tokens": 412, "output_tokens": 264},
            ],
        },
        reasoning_content=None,
        completion_response={
            "content": [{"type": "fallback", "trigger": {"type": "refusal", "category": "bio"}}],
            "stop_reason": "end_turn",
        },
    )
    deployment_prices: ModelInfo = {
        **litellm.get_model_info(model="claude-opus-4-8", custom_llm_provider="anthropic"),
        "input_cost_per_token": 1e-3,
        "output_cost_per_token": 2e-3,
    }
    fable_prompt_cost, fable_completion_cost = litellm.cost_per_token(
        model="claude-fable-5", prompt_tokens=535, completion_tokens=0
    )

    prompt_cost, completion_cost = cost_per_token(
        model="claude-opus-4-8", usage=chain_usage, model_info=deployment_prices
    )

    assert prompt_cost == pytest.approx(fable_prompt_cost + 412 * 1e-3)
    assert completion_cost == pytest.approx(fable_completion_cost + 264 * 2e-3)
