import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.fireworks_ai.cost_calculator import cost_per_token
from litellm.types.utils import PromptTokensDetailsWrapper, Usage

MODEL = "accounts/fireworks/models/glm-5p2"
INPUT_COST = 1.4e-06
# Read the cached rate from the price map so this test tracks the shipped value
# (glm-5p2 is $0.14/1M) instead of hardcoding a number that breaks when it changes.
CACHE_READ_COST = litellm.get_model_info(model=MODEL, custom_llm_provider="fireworks_ai")["cache_read_input_token_cost"]
OUTPUT_COST = 4.4e-06


def _usage(prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> Usage:
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=cached_tokens),
    )


def test_cached_prompt_tokens_billed_at_cache_read_rate():
    prompt_tokens = 7036
    cached_tokens = 7020
    completion_tokens = 8

    prompt_cost, completion_cost = cost_per_token(
        model=MODEL, usage=_usage(prompt_tokens, cached_tokens, completion_tokens)
    )

    expected_prompt_cost = (prompt_tokens - cached_tokens) * INPUT_COST + cached_tokens * CACHE_READ_COST
    assert prompt_cost == pytest.approx(expected_prompt_cost)
    assert completion_cost == pytest.approx(completion_tokens * OUTPUT_COST)

    full_rate_cost = prompt_tokens * INPUT_COST
    assert prompt_cost < full_rate_cost


def test_warm_call_cheaper_than_cold_call():
    prompt_tokens = 7036
    completion_tokens = 8

    cold_prompt_cost, _ = cost_per_token(
        model=MODEL, usage=_usage(prompt_tokens, 16, completion_tokens)
    )
    warm_prompt_cost, _ = cost_per_token(
        model=MODEL, usage=_usage(prompt_tokens, 7020, completion_tokens)
    )

    assert warm_prompt_cost < cold_prompt_cost


def test_no_cached_tokens_matches_full_input_rate():
    prompt_tokens = 100
    completion_tokens = 10

    prompt_cost, completion_cost = cost_per_token(
        model=MODEL, usage=_usage(prompt_tokens, 0, completion_tokens)
    )

    assert prompt_cost == pytest.approx(prompt_tokens * INPUT_COST)
    assert completion_cost == pytest.approx(completion_tokens * OUTPUT_COST)
