import pytest

import litellm
from litellm.llms.fireworks_ai.cost_calculator import cost_per_token
from litellm.types.utils import PromptTokensDetailsWrapper, Usage


@pytest.fixture
def local_model_cost_map(monkeypatch):
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()


def test_cost_per_token_discounts_cached_prompt_tokens(local_model_cost_map):
    """
    Regression: Fireworks reports cached prompt tokens via prompt_tokens_details.cached_tokens.
    Those tokens must be billed at cache_read_input_token_cost, not the full input rate.
    """
    model = "accounts/fireworks/models/deepseek-v4-flash"
    input_cost = 1.4e-07
    cache_read_cost = 2.8e-08
    output_cost = 2.8e-07

    usage = Usage(
        prompt_tokens=43,
        completion_tokens=1530,
        total_tokens=1573,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=42),
    )

    prompt_cost, completion_cost = cost_per_token(model=model, usage=usage)

    non_cached_tokens = 43 - 42
    assert prompt_cost == pytest.approx(non_cached_tokens * input_cost + 42 * cache_read_cost)
    assert completion_cost == pytest.approx(1530 * output_cost)

    full_price = 43 * input_cost
    assert prompt_cost < full_price


def test_cost_per_token_without_cache_matches_full_input_rate(local_model_cost_map):
    model = "accounts/fireworks/models/deepseek-v4-flash"
    input_cost = 1.4e-07
    output_cost = 2.8e-07

    usage = Usage(prompt_tokens=1000, completion_tokens=2000, total_tokens=3000)

    prompt_cost, completion_cost = cost_per_token(model=model, usage=usage)

    assert prompt_cost == pytest.approx(1000 * input_cost)
    assert completion_cost == pytest.approx(2000 * output_cost)


def test_cost_per_token_falls_back_to_size_based_bucket(local_model_cost_map):
    """A serverless model not present in the map resolves to its parameter-size pricing bucket."""
    bucket_cost = litellm.model_cost["fireworks-ai-4.1b-to-16b"]["input_cost_per_token"]

    usage = Usage(prompt_tokens=1000, completion_tokens=2000, total_tokens=3000)

    prompt_cost, completion_cost = cost_per_token(
        model="accounts/fireworks/models/some-unmapped-7b", usage=usage
    )

    assert prompt_cost == pytest.approx(1000 * bucket_cost)
    assert completion_cost == pytest.approx(2000 * bucket_cost)


def test_top_level_dispatcher_applies_cache_discount(local_model_cost_map):
    from litellm.cost_calculator import cost_per_token as dispatch_cost_per_token

    input_cost = 1.4e-07
    cache_read_cost = 2.8e-08

    usage = Usage(
        prompt_tokens=43,
        completion_tokens=1530,
        total_tokens=1573,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=42),
    )

    prompt_cost, _ = dispatch_cost_per_token(
        model="fireworks_ai/accounts/fireworks/models/deepseek-v4-flash",
        custom_llm_provider="fireworks_ai",
        usage_object=usage,
    )

    assert prompt_cost == pytest.approx((43 - 42) * input_cost + 42 * cache_read_cost)
