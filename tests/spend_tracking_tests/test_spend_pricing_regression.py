"""
Pricing-table regression tests.

These are pure unit tests (no live proxy) that pin litellm.cost_per_token and
litellm.completion_cost against the pricing map in litellm.model_cost. Their job
is to catch silent pricing regressions: a changed price, a broken lookup, or a
corrupted map.

Two layers of defense:
  - GOLDEN values: hardcoded dollar amounts for well-known stable models. If the
    map is corrupted or a price is edited, these fail even though the calculation
    code is unchanged. A test that only recomputes from the same map and compares
    to itself kills no mutants; these golden constants are the anti-drift anchor.
  - DERIVED + SANITY: expected cost derived from the map, plus ordering/positivity
    invariants (cache reads cheaper than fresh prompt tokens, completion pricier
    than prompt for these models, per-component additivity) that catch a map whose
    structure is wrong even if individual numbers look plausible.
"""

import pytest

import litellm

PROMPT_TOKENS = 1000
COMPLETION_TOKENS = 500

# Golden per-token prices for stable, widely-used models. Sourced from
# litellm/model_prices_and_context_window_backup.json. If a maintainer changes a
# price (or breaks the lookup), the golden assertions below fail loudly.
GOLDEN_PRICES = {
    "gpt-4o": {"input": 2.5e-06, "output": 1e-05},
    "gpt-4o-mini": {"input": 1.5e-07, "output": 6e-07},
    "gpt-3.5-turbo": {"input": 5e-07, "output": 1.5e-06},
    "gpt-5-mini": {"input": 2.5e-07, "output": 2e-06},
}


@pytest.mark.parametrize("model,prices", GOLDEN_PRICES.items())
def test_golden_prompt_completion_cost(model, prices):
    prompt_cost, completion_cost = litellm.cost_per_token(
        model=model,
        prompt_tokens=PROMPT_TOKENS,
        completion_tokens=COMPLETION_TOKENS,
    )

    expected_prompt = PROMPT_TOKENS * prices["input"]
    expected_completion = COMPLETION_TOKENS * prices["output"]

    assert prompt_cost == pytest.approx(
        expected_prompt, rel=1e-9
    ), f"{model} prompt cost regressed: got {prompt_cost}, expected {expected_prompt}"
    assert completion_cost == pytest.approx(
        expected_completion, rel=1e-9
    ), f"{model} completion cost regressed: got {completion_cost}, expected {expected_completion}"


@pytest.mark.parametrize("model,prices", GOLDEN_PRICES.items())
def test_map_matches_golden_prices(model, prices):
    entry = litellm.model_cost[model]
    assert entry["input_cost_per_token"] == pytest.approx(
        prices["input"], rel=1e-9
    ), f"{model} input price in litellm.model_cost drifted from golden value"
    assert entry["output_cost_per_token"] == pytest.approx(
        prices["output"], rel=1e-9
    ), f"{model} output price in litellm.model_cost drifted from golden value"


def test_cost_per_token_derives_from_map():
    """cost_per_token must equal tokens * map price for both components, scaling linearly."""
    model = "gpt-4o"
    entry = litellm.model_cost[model]

    prompt_cost, completion_cost = litellm.cost_per_token(
        model=model, prompt_tokens=PROMPT_TOKENS, completion_tokens=COMPLETION_TOKENS
    )
    assert prompt_cost == pytest.approx(
        PROMPT_TOKENS * entry["input_cost_per_token"], rel=1e-9
    )
    assert completion_cost == pytest.approx(
        COMPLETION_TOKENS * entry["output_cost_per_token"], rel=1e-9
    )

    double_prompt, _ = litellm.cost_per_token(
        model=model, prompt_tokens=2 * PROMPT_TOKENS, completion_tokens=0
    )
    assert double_prompt == pytest.approx(
        2 * prompt_cost, rel=1e-9
    ), "Prompt cost must scale linearly with token count"


def test_completion_cost_matches_cost_per_token():
    """completion_cost on a response must equal summed cost_per_token components."""
    from litellm.types.utils import ModelResponse, Usage, Choices, Message

    model = "gpt-4o"
    response = ModelResponse(
        model=model,
        choices=[Choices(index=0, message=Message(role="assistant", content="hi"))],
        usage=Usage(
            prompt_tokens=PROMPT_TOKENS,
            completion_tokens=COMPLETION_TOKENS,
            total_tokens=PROMPT_TOKENS + COMPLETION_TOKENS,
        ),
    )

    total = litellm.completion_cost(completion_response=response, model=model)
    prompt_cost, completion_cost = litellm.cost_per_token(
        model=model, prompt_tokens=PROMPT_TOKENS, completion_tokens=COMPLETION_TOKENS
    )
    assert total == pytest.approx(prompt_cost + completion_cost, rel=1e-9)


def test_cache_read_cheaper_than_fresh_prompt_tokens():
    """
    For a cache-priced model, cached prompt tokens must be billed at the cache-read
    rate, strictly cheaper than the same tokens at the full input rate.
    """
    model = "gpt-5-mini"
    entry = litellm.model_cost[model]
    cache_read_price = entry["cache_read_input_token_cost"]
    input_price = entry["input_cost_per_token"]

    assert cache_read_price is not None and cache_read_price > 0
    assert (
        cache_read_price < input_price
    ), "Cache-read price must be strictly less than the full input price"

    cached_tokens = 800
    fresh_tokens = PROMPT_TOKENS - cached_tokens
    prompt_cost, _ = litellm.cost_per_token(
        model=model,
        prompt_tokens=PROMPT_TOKENS,
        completion_tokens=0,
        cache_read_input_tokens=cached_tokens,
    )

    expected = fresh_tokens * input_price + cached_tokens * cache_read_price
    assert prompt_cost == pytest.approx(
        expected, rel=1e-9
    ), f"Cache-read pricing regressed for {model}: got {prompt_cost}, expected {expected}"

    full_price_cost, _ = litellm.cost_per_token(
        model=model, prompt_tokens=PROMPT_TOKENS, completion_tokens=0
    )
    assert (
        prompt_cost < full_price_cost
    ), "Billing with cache reads must be cheaper than billing all tokens at full price"


def test_cache_creation_priced_separately_for_anthropic():
    """
    Anthropic prompt-caching prices cache-creation (write) tokens distinctly from
    cache-read and fresh tokens. Verify each bucket bills at its own rate.
    """
    model = "claude-3-7-sonnet-20250219"
    entry = litellm.model_cost[model]
    input_price = entry["input_cost_per_token"]
    cache_read_price = entry["cache_read_input_token_cost"]
    cache_creation_price = entry["cache_creation_input_token_cost"]

    assert cache_creation_price is not None and cache_creation_price > 0
    assert cache_read_price is not None and cache_read_price > 0
    assert (
        cache_read_price < input_price < cache_creation_price
    ), "Expected ordering cache_read < input < cache_creation for Anthropic caching"

    cache_creation_tokens = 500
    cache_read_tokens = 200
    fresh_tokens = PROMPT_TOKENS - cache_creation_tokens - cache_read_tokens

    prompt_cost, _ = litellm.cost_per_token(
        model=model,
        prompt_tokens=PROMPT_TOKENS,
        completion_tokens=0,
        cache_creation_input_tokens=cache_creation_tokens,
        cache_read_input_tokens=cache_read_tokens,
    )

    expected = (
        fresh_tokens * input_price
        + cache_creation_tokens * cache_creation_price
        + cache_read_tokens * cache_read_price
    )
    assert prompt_cost == pytest.approx(
        expected, rel=1e-9
    ), f"Anthropic cache pricing regressed: got {prompt_cost}, expected {expected}"


def test_known_model_costs_are_positive_and_ordered():
    """
    Sanity guard against a corrupted map: prices for these models must be positive,
    and output must be priced higher than input (true for all four golden models).
    """
    for model, prices in GOLDEN_PRICES.items():
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model, prompt_tokens=1, completion_tokens=1
        )
        assert prompt_cost > 0, f"{model} prompt cost must be > 0"
        assert completion_cost > 0, f"{model} completion cost must be > 0"
        assert (
            completion_cost > prompt_cost
        ), f"{model} output should be priced above input per token"
