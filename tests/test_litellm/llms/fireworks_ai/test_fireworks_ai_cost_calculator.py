"""Tests for Fireworks AI cost calculator cache token accounting.

Regression coverage for https://github.com/BerriAI/litellm/issues/24774 — the
fireworks_ai cost calculator now subtracts the full-price portion of cache read
and cache creation tokens, then adds the discounted/premium rate.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.fireworks_ai.cost_calculator import cost_per_token
from litellm.types.utils import Usage


@pytest.fixture(autouse=True)
def _local_model_cost_map(monkeypatch):
    """Load model prices from the bundled JSON so tests are deterministic and offline."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    yield


def test_cache_read_tokens_reduce_prompt_cost():
    """800 cache-read tokens should make the call cheaper than the full-price path.

    kimi-k2p5 pricing (model_cost map):
      input_cost_per_token        = 6e-7
      cache_read_input_token_cost = 1e-7   (cache hit discount)
    """
    usage_with_cache = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        cache_read_input_tokens=800,
        cache_creation_input_tokens=0,
    )
    usage_no_cache = Usage(prompt_tokens=1000, completion_tokens=500)

    prompt_cost_cached, completion_cost_cached = cost_per_token(
        model="fireworks_ai/kimi-k2p5", usage=usage_with_cache
    )
    prompt_cost_no_cache, completion_cost_no_cache = cost_per_token(
        model="fireworks_ai/kimi-k2p5", usage=usage_no_cache
    )

    # Completion cost is unaffected by cache state.
    assert completion_cost_cached == completion_cost_no_cache
    # Cache hits are cheaper than full-price input tokens.
    assert (
        prompt_cost_cached < prompt_cost_no_cache
    ), "800 cache-read tokens at 1e-7 should be cheaper than 800 full-price tokens at 6e-7"


def test_cache_read_token_pricing_is_deterministic():
    """Exercises the cache_read_input_tokens adjustment branch with exact arithmetic.

    With prompt_tokens=1000 and cache_read_input_tokens=100:
      base                  = 1000 * 6e-7 = 6e-4
      cache adjustment      = 100 * (1e-7 - 6e-7) = -5e-5
      expected prompt_cost  = 5.5e-4
    """
    INPUT_COST = 6e-7
    CACHE_READ_COST = 1e-7
    PROMPT_TOKENS = 1000
    CACHE_READ_TOKENS = 100
    COMPLETION_TOKENS = 50

    usage = Usage(
        prompt_tokens=PROMPT_TOKENS,
        completion_tokens=COMPLETION_TOKENS,
        cache_read_input_tokens=CACHE_READ_TOKENS,
    )

    prompt_cost, completion_cost_val = cost_per_token(
        model="fireworks_ai/kimi-k2p5", usage=usage
    )

    expected_prompt_cost = PROMPT_TOKENS * INPUT_COST + CACHE_READ_TOKENS * (
        CACHE_READ_COST - INPUT_COST
    )
    expected_completion_cost = COMPLETION_TOKENS * 3e-6

    assert (
        abs(prompt_cost - expected_prompt_cost) < 1e-12
    ), f"Cache-read prompt cost {prompt_cost} != expected {expected_prompt_cost}"
    assert abs(completion_cost_val - expected_completion_cost) < 1e-12
    # Sanity: cheaper than paying full rate for all 1000 tokens.
    assert prompt_cost < PROMPT_TOKENS * INPUT_COST


def test_cache_creation_token_pricing_is_deterministic():
    """Exercises the cache_creation_input_tokens adjustment branch.

    No live fireworks model currently has cache_creation_input_token_cost set,
    so this test injects a synthetic value into the local model_cost copy.
    The fixture rebuilds model_cost per test, so the injection is scoped.

    With prompt_tokens=1000 and cache_creation_input_tokens=100:
      base                  = 1000 * 6e-7 = 6e-4
      creation adjustment   = 100 * (8e-7 - 6e-7) = +2e-5
      expected prompt_cost  = 6.2e-4
    """
    INPUT_COST = 6e-7
    CACHE_CREATION_COST = 8e-7
    PROMPT_TOKENS = 1000
    CACHE_CREATION_TOKENS = 100
    COMPLETION_TOKENS = 50

    litellm.model_cost["fireworks_ai/kimi-k2p5"][
        "cache_creation_input_token_cost"
    ] = CACHE_CREATION_COST

    usage = Usage(
        prompt_tokens=PROMPT_TOKENS,
        completion_tokens=COMPLETION_TOKENS,
        cache_creation_input_tokens=CACHE_CREATION_TOKENS,
    )

    prompt_cost, completion_cost_val = cost_per_token(
        model="fireworks_ai/kimi-k2p5", usage=usage
    )

    expected_prompt_cost = PROMPT_TOKENS * INPUT_COST + CACHE_CREATION_TOKENS * (
        CACHE_CREATION_COST - INPUT_COST
    )
    expected_completion_cost = COMPLETION_TOKENS * 3e-6

    assert abs(prompt_cost - expected_prompt_cost) < 1e-12
    assert abs(completion_cost_val - expected_completion_cost) < 1e-12
    # Sanity: more expensive than full input rate.
    assert prompt_cost > PROMPT_TOKENS * INPUT_COST


def test_no_cache_tokens_uses_plain_input_rate():
    """When neither cache_read nor cache_creation tokens are present,
    prompt_cost is exactly prompt_tokens * input_cost_per_token.

    Catches a regression where the cache branches incorrectly fire on zero tokens.
    """
    INPUT_COST = 6e-7
    PROMPT_TOKENS = 500
    COMPLETION_TOKENS = 10

    usage = Usage(prompt_tokens=PROMPT_TOKENS, completion_tokens=COMPLETION_TOKENS)

    prompt_cost, completion_cost_val = cost_per_token(
        model="fireworks_ai/kimi-k2p5", usage=usage
    )

    assert abs(prompt_cost - PROMPT_TOKENS * INPUT_COST) < 1e-12
    assert abs(completion_cost_val - COMPLETION_TOKENS * 3e-6) < 1e-12
