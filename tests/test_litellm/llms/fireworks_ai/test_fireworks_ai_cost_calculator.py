import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map
from litellm.llms.fireworks_ai.cost_calculator import cost_per_token
from litellm.types.utils import PromptTokensDetailsWrapper, Usage


@pytest.fixture(autouse=True)
def force_local_model_cost(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = get_model_cost_map(url=litellm.model_cost_map_url)


def test_cost_per_token_applies_cache_read_discount():
    """
    Regression test for https://github.com/BerriAI/litellm/issues/25950.

    Fireworks reports cached prompt tokens in
    ``usage.prompt_tokens_details.cached_tokens`` and the price map defines
    ``cache_read_input_token_cost`` for cache-capable models. The pre-fix
    calculator multiplied every prompt token by ``input_cost_per_token`` and
    silently dropped the cache discount, so users were overbilled on any
    cache-hit turn.
    """
    model = "accounts/fireworks/models/kimi-k2p5"
    model_info = litellm.model_cost[f"fireworks_ai/{model}"]
    input_rate = model_info["input_cost_per_token"]
    cache_read_rate = model_info["cache_read_input_token_cost"]
    assert cache_read_rate < input_rate

    prompt_tokens = 1000
    cached_tokens = 800
    completion_tokens = 50

    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=cached_tokens,
        ),
    )

    prompt_cost, completion_cost = cost_per_token(model=model, usage=usage)

    fresh_tokens = prompt_tokens - cached_tokens
    expected_prompt = fresh_tokens * input_rate + cached_tokens * cache_read_rate
    expected_completion = completion_tokens * model_info["output_cost_per_token"]

    assert prompt_cost == pytest.approx(expected_prompt, rel=1e-9)
    assert completion_cost == pytest.approx(expected_completion, rel=1e-9)


def test_cost_per_token_no_cache_matches_flat_rate():
    """
    Cache-free requests must keep charging every prompt token at the standard
    input rate; this guards against a regression where the cache-aware path
    accidentally re-classifies plain prompt tokens.
    """
    model = "accounts/fireworks/models/kimi-k2p5"
    model_info = litellm.model_cost[f"fireworks_ai/{model}"]

    prompt_tokens = 400
    completion_tokens = 60
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )

    prompt_cost, completion_cost = cost_per_token(model=model, usage=usage)

    assert prompt_cost == pytest.approx(prompt_tokens * model_info["input_cost_per_token"], rel=1e-9)
    assert completion_cost == pytest.approx(completion_tokens * model_info["output_cost_per_token"], rel=1e-9)


def test_cost_per_token_unmapped_model_falls_back_to_size_tier():
    """
    Models that are not in the local price map must still be priced via the
    parameter-size tier fallback (``get_base_model_for_pricing``) rather than
    raising.
    """
    model = "some-brand-new-70b-model-that-is-not-mapped"
    usage = Usage(prompt_tokens=100, completion_tokens=25, total_tokens=125)

    prompt_cost, completion_cost = cost_per_token(model=model, usage=usage)

    tier_info = litellm.model_cost["fireworks-ai-above-16b"]
    assert prompt_cost == pytest.approx(100 * tier_info["input_cost_per_token"], rel=1e-9)
    assert completion_cost == pytest.approx(25 * tier_info["output_cost_per_token"], rel=1e-9)
