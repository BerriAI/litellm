"""Itemized cache costs in the logged cost breakdown for the OpenAI Responses API.

Regression guard for https://github.com/BerriAI/litellm/issues/34309: the
breakdown block derived its cache token counts only from the Anthropic-style
top-level ``cache_read_input_tokens`` / ``cache_creation_input_tokens``. OpenAI's
Responses API reports them under ``input_tokens_details.{cached_tokens,
cache_write_tokens}`` instead, so ``cost_breakdown.cache_read_cost`` and
``cache_creation_cost`` serialized as null for every OpenAI request while the
grand total stayed correct.

``input_cost`` is the full prompt-side cost and the two cache fields are additive
break-outs that overlap it, matching the Anthropic path. A request with no cache
activity leaves both fields unset, which is the documented behavior.
"""

from datetime import datetime

import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.responses.utils import ResponseAPILoggingUtils
from litellm.types.utils import Choices, Message, ModelResponse

MODEL = "gpt-5.6"


def _logging_obj() -> Logging:
    return Logging(
        model=MODEL,
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="completion",
        start_time=datetime.now(),
        litellm_call_id="responses-cache-breakdown",
        function_id="f",
    )


def _responses_completion(cached_tokens: int, cache_write_tokens: int, fresh_tokens: int, output_tokens: int):
    input_tokens = cached_tokens + cache_write_tokens + fresh_tokens
    usage = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(
        {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_details": {
                "cached_tokens": cached_tokens,
                "cache_write_tokens": cache_write_tokens,
            },
        }
    )
    return ModelResponse(
        id="x",
        created=1,
        model=MODEL,
        object="chat.completion",
        choices=[Choices(index=0, message=Message(role="assistant", content="hi"), finish_reason="stop")],
        usage=usage,
    )


def test_responses_api_cache_costs_are_itemized_in_the_breakdown(local_model_cost_map):
    """Cache-read and cache-write dollars are broken out for an OpenAI Responses
    request, at the model's own cache rates."""
    rates = litellm.model_cost[MODEL]
    cached_tokens = 4012
    cache_write_tokens = 5000
    fresh_tokens = 1000
    output_tokens = 200

    logging_obj = _logging_obj()
    total = litellm.completion_cost(
        completion_response=_responses_completion(cached_tokens, cache_write_tokens, fresh_tokens, output_tokens),
        model=MODEL,
        custom_llm_provider="openai",
        litellm_logging_obj=logging_obj,
    )

    breakdown = logging_obj.cost_breakdown
    assert breakdown is not None

    expected_cache_read = cached_tokens * rates["cache_read_input_token_cost"]
    expected_cache_creation = cache_write_tokens * rates["cache_creation_input_token_cost"]
    expected_input = fresh_tokens * rates["input_cost_per_token"] + expected_cache_read + expected_cache_creation
    expected_output = output_tokens * rates["output_cost_per_token"]

    assert breakdown["cache_read_cost"] == pytest.approx(expected_cache_read)
    assert breakdown["cache_creation_cost"] == pytest.approx(expected_cache_creation)
    assert breakdown["input_cost"] == pytest.approx(expected_input)
    assert breakdown["output_cost"] == pytest.approx(expected_output)
    assert breakdown["total_cost"] == pytest.approx(expected_input + expected_output)
    assert total == pytest.approx(breakdown["total_cost"])


def test_cache_fields_stay_unset_when_there_was_no_cache_activity(local_model_cost_map):
    """A request that neither read nor wrote the cache leaves both break-out fields
    off the breakdown, so they serialize as null rather than a misleading $0."""
    logging_obj = _logging_obj()
    litellm.completion_cost(
        completion_response=_responses_completion(
            cached_tokens=0, cache_write_tokens=0, fresh_tokens=1000, output_tokens=200
        ),
        model=MODEL,
        custom_llm_provider="openai",
        litellm_logging_obj=logging_obj,
    )

    breakdown = logging_obj.cost_breakdown
    assert breakdown is not None
    assert "cache_read_cost" not in breakdown
    assert "cache_creation_cost" not in breakdown
    assert breakdown["input_cost"] == pytest.approx(1000 * litellm.model_cost[MODEL]["input_cost_per_token"])
