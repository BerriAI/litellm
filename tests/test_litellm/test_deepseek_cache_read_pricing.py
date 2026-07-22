"""
Regression test for DeepSeek and OpenRouter/DeepSeek models that historically
only declared ``input_cost_per_token_cache_hit`` in the pricing JSON.

The cost calculator (``litellm.litellm_core_utils.llm_cost_calc.utils``) only
reads ``cache_read_input_token_cost`` for prompt-cache-hit pricing.  When a
model defines ``input_cost_per_token_cache_hit`` without the canonical key,
cache-hit tokens are billed at $0 because the cache-read cost resolves to
``None`` -> ``0.0``.

This test ensures the two keys are kept in sync for the affected models and
that a representative model actually bills cache-hit tokens at the cache rate.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import PromptTokensDetailsWrapper, Usage

# Models that previously only carried ``input_cost_per_token_cache_hit``.
# Each must also carry ``cache_read_input_token_cost`` with an identical value.
DEEPSEEK_CACHE_HIT_MODELS = [
    "deepseek/deepseek-coder",
    "deepseek/deepseek-r1",
    "deepseek/deepseek-v3.2",
    "openrouter/deepseek/deepseek-chat-v3.1",
    "openrouter/deepseek/deepseek-v3.2",
    "openrouter/deepseek/deepseek-v3.2-exp",
    "openrouter/deepseek/deepseek-r1",
    "openrouter/deepseek/deepseek-r1-0528",
]


@pytest.fixture(autouse=True)
def _use_local_model_cost_map():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")


@pytest.mark.parametrize("model", DEEPSEEK_CACHE_HIT_MODELS)
def test_cache_read_input_token_cost_present(model):
    """``cache_read_input_token_cost`` must be present and equal to the legacy
    ``input_cost_per_token_cache_hit`` field."""
    info = litellm.model_cost[model]
    legacy = info.get("input_cost_per_token_cache_hit")
    canonical = info.get("cache_read_input_token_cost")
    assert legacy is not None, f"{model} is missing input_cost_per_token_cache_hit"
    assert canonical is not None, (
        f"{model} is missing cache_read_input_token_cost; cache-hit tokens would "
        f"be billed at $0 because the cost calculator only reads the canonical key."
    )
    assert canonical == legacy, (
        f"{model} cache_read_input_token_cost ({canonical}) must match "
        f"input_cost_per_token_cache_hit ({legacy})"
    )


def test_deepseek_r1_cache_hit_billed_at_cache_rate():
    """End-to-end check that a cache hit on deepseek/deepseek-r1 is billed at
    the cache-read rate instead of the regular input rate."""
    model = "deepseek/deepseek-r1"
    info = litellm.model_cost[model]

    prompt_tokens = 1000
    cached_tokens = 800
    text_tokens = prompt_tokens - cached_tokens
    completion_tokens = 100

    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=cached_tokens,
            text_tokens=text_tokens,
        ),
    )

    input_cost, output_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="deepseek",
    )

    expected_input_cost = (
        info["input_cost_per_token"] * text_tokens
        + info["cache_read_input_token_cost"] * cached_tokens
    )
    expected_output_cost = info["output_cost_per_token"] * completion_tokens

    assert (
        abs(input_cost - expected_input_cost) < 1e-12
    ), f"input cost mismatch: got {input_cost}, expected {expected_input_cost}"
    assert (
        abs(output_cost - expected_output_cost) < 1e-12
    ), f"output cost mismatch: got {output_cost}, expected {expected_output_cost}"

    # Sanity check: regression scenario (cache_read_input_token_cost missing)
    # would have billed cached tokens at the full input rate.
    naive_full_input_cost = info["input_cost_per_token"] * prompt_tokens
    assert input_cost < naive_full_input_cost, (
        "cache-hit tokens were not discounted; cache_read_input_token_cost is "
        "likely missing from this model entry."
    )
