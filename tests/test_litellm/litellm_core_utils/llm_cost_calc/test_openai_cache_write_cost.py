"""Cache-write pricing for OpenAI models, on both the chat and Responses paths.

Regression guard for https://github.com/BerriAI/litellm/issues/33772: OpenAI
reports cache-write tokens under ``prompt_tokens_details.cache_write_tokens``
(chat) / ``input_tokens_details.cache_write_tokens`` (responses), where
Anthropic reports ``cache_creation_tokens``. The cost path read only the
Anthropic name, so cache-write tokens were billed at the plain input rate,
the Responses transform dropped the split before cost ran, and the tiered
``cache_creation_input_token_cost_{priority,flex,above_272k_tokens}`` keys were
discarded by ``get_model_info``.
"""

import pytest

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.responses.utils import ResponseAPILoggingUtils
from litellm.types.utils import Usage

MODEL = "gpt-5.6"


def _openai_chat_usage(prompt_tokens: int, cache_write_tokens: int, completion_tokens: int) -> Usage:
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details={
            "cached_tokens": 0,
            "cache_write_tokens": cache_write_tokens,
        },
    )


def test_openai_cache_write_tokens_billed_at_the_cache_creation_rate(local_model_cost_map):
    """A cache-write request costs the cache-creation rate on the written tokens,
    not the plain input rate."""
    rates = litellm.model_cost[MODEL]
    input_rate = rates["input_cost_per_token"]
    cache_write_rate = rates["cache_creation_input_token_cost"]
    output_rate = rates["output_cost_per_token"]
    assert cache_write_rate == pytest.approx(input_rate * 1.25)

    prompt_tokens = 12317
    cache_write_tokens = 12314
    fresh_tokens = prompt_tokens - cache_write_tokens
    completion_tokens = 5

    prompt_cost, completion_cost = generic_cost_per_token(
        model=MODEL,
        usage=_openai_chat_usage(prompt_tokens, cache_write_tokens, completion_tokens),
        custom_llm_provider="openai",
    )

    assert prompt_cost == pytest.approx(fresh_tokens * input_rate + cache_write_tokens * cache_write_rate)
    assert completion_cost == pytest.approx(completion_tokens * output_rate)
    assert prompt_cost > prompt_tokens * input_rate


def test_responses_api_cache_write_costs_the_same_as_chat(local_model_cost_map):
    """The Responses usage transform carries the cache-write split through, so the
    same request costs the same whichever route it took."""
    prompt_tokens = 12317
    cache_write_tokens = 12314
    completion_tokens = 5

    responses_usage = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(
        {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "input_tokens_details": {
                "cached_tokens": 0,
                "cache_write_tokens": cache_write_tokens,
            },
        }
    )
    assert responses_usage.prompt_tokens_details.cache_write_tokens == cache_write_tokens

    responses_prompt_cost, responses_completion_cost = generic_cost_per_token(
        model=MODEL,
        usage=responses_usage,
        custom_llm_provider="openai",
    )
    chat_prompt_cost, chat_completion_cost = generic_cost_per_token(
        model=MODEL,
        usage=_openai_chat_usage(prompt_tokens, cache_write_tokens, completion_tokens),
        custom_llm_provider="openai",
    )

    rates = litellm.model_cost[MODEL]
    fresh_tokens = prompt_tokens - cache_write_tokens
    assert responses_prompt_cost == pytest.approx(
        fresh_tokens * rates["input_cost_per_token"]
        + cache_write_tokens * rates["cache_creation_input_token_cost"]
    )
    assert responses_prompt_cost == pytest.approx(chat_prompt_cost)
    assert responses_completion_cost == pytest.approx(chat_completion_cost)


@pytest.mark.parametrize(
    "service_tier,prompt_tokens,rate_key",
    [
        (None, 100000, "cache_creation_input_token_cost"),
        ("priority", 100000, "cache_creation_input_token_cost_priority"),
        ("flex", 100000, "cache_creation_input_token_cost_flex"),
        (None, 300000, "cache_creation_input_token_cost_above_272k_tokens"),
    ],
)
def test_tiered_cache_creation_rates_are_registered_and_billed(
    local_model_cost_map, service_tier, prompt_tokens, rate_key
):
    """The tiered cache-creation keys survive ``get_model_info`` and are the rate the
    cost path actually charges for priority, flex, and >272k requests."""
    model_info = litellm.get_model_info(model=MODEL, custom_llm_provider="openai")
    tiered_rate = litellm.model_cost[MODEL][rate_key]
    assert model_info.get(rate_key) == tiered_rate

    cache_write_tokens = prompt_tokens - 1000
    fresh_tokens = 1000
    input_rate_key = {
        "cache_creation_input_token_cost": "input_cost_per_token",
        "cache_creation_input_token_cost_priority": "input_cost_per_token_priority",
        "cache_creation_input_token_cost_flex": "input_cost_per_token_flex",
        "cache_creation_input_token_cost_above_272k_tokens": "input_cost_per_token_above_272k_tokens",
    }[rate_key]
    input_rate = litellm.model_cost[MODEL][input_rate_key]

    prompt_cost, _ = generic_cost_per_token(
        model=MODEL,
        usage=_openai_chat_usage(prompt_tokens, cache_write_tokens, completion_tokens=100),
        custom_llm_provider="openai",
        service_tier=service_tier,
    )

    assert prompt_cost == pytest.approx(fresh_tokens * input_rate + cache_write_tokens * tiered_rate)
