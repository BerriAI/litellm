"""Token and spend reconciliation for /v1/responses batches.

Regression guard for https://github.com/BerriAI/litellm/issues/35363: a batch
output line built by the Responses API reports ``input_tokens`` /
``output_tokens`` where a chat line reports ``prompt_tokens`` /
``completion_tokens``. The usage object was constructed straight from the raw
dict, which accepts the unrecognized names without raising and yields zeros, so
a completed Responses batch reconciled to 0 tokens and $0.00 spend with no
error, and per-key budgets were never charged for it.

Line shape decides the parse, not the batch's declared endpoint, so an output
file mixing Responses-shaped and chat-shaped lines sums across both.
"""

from typing import Literal, get_args, get_type_hints

import pytest

import litellm
import litellm.batches.batch_utils as bu
from litellm.types.llms.openai import CreateBatchRequest

MODEL = "gpt-5.6"


def _responses_line(input_tokens: int, output_tokens: int) -> dict:
    return {
        "response": {
            "status_code": 200,
            "body": {
                "model": MODEL,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            },
        }
    }


def _chat_line(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "response": {
            "status_code": 200,
            "body": {
                "model": MODEL,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        }
    }


def test_responses_shaped_usage_maps_onto_prompt_and_completion_tokens():
    """The Responses names land on the chat-shaped counters instead of being
    dropped for unrecognized keys."""
    usage = bu._get_batch_job_usage_from_response_body(
        {"usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}}
    )
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (100, 50, 150)


async def test_responses_batch_reconciles_to_real_tokens_and_spend(local_model_cost_map):
    """A completed Responses batch records the provider's token counts and a
    non-zero spend at the model's batch rates."""
    model_info = litellm.get_model_info(model=MODEL, custom_llm_provider="openai")
    input_tokens = 33
    output_tokens = 57

    cost, usage, models = await bu.calculate_batch_cost_and_usage(
        file_content_dictionary=[_responses_line(input_tokens, output_tokens)],
        custom_llm_provider="openai",
        model_name=MODEL,
        model_info=model_info,
    )

    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (
        input_tokens,
        output_tokens,
        input_tokens + output_tokens,
    )
    assert models == [MODEL]
    assert cost == pytest.approx(
        input_tokens * model_info["input_cost_per_token_batches"]
        + output_tokens * model_info["output_cost_per_token_batches"]
    )
    assert cost > 0.0


async def test_mixed_shape_batch_output_sums_across_both_line_shapes(local_model_cost_map):
    """An output file carrying both line shapes sums both. A fix keyed off the
    batch's declared endpoint rather than each line's shape would miss this."""
    model_info = litellm.get_model_info(model=MODEL, custom_llm_provider="openai")

    cost, usage, _ = await bu.calculate_batch_cost_and_usage(
        file_content_dictionary=[_responses_line(100, 50), _chat_line(33, 57)],
        custom_llm_provider="openai",
        model_name=MODEL,
        model_info=model_info,
    )

    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (133, 107, 240)
    assert cost == pytest.approx(
        133 * model_info["input_cost_per_token_batches"] + 107 * model_info["output_cost_per_token_batches"]
    )


def test_create_batch_endpoint_accepts_v1_responses():
    """A type-checked caller can pass endpoint="/v1/responses", which the runtime
    already forwarded correctly."""
    endpoint_annotation = get_type_hints(CreateBatchRequest)["endpoint"]
    assert "/v1/responses" in get_args(endpoint_annotation)

    for create_fn in (litellm.create_batch, litellm.acreate_batch):
        assert "/v1/responses" in get_args(get_type_hints(create_fn)["endpoint"])
