"""
Test that batch cost calculation uses custom deployment-level pricing
when model_info is provided.

Reproduces the bug where `input_cost_per_token_batches` /
`output_cost_per_token_batches` set on a proxy deployment's model_info
are ignored by the batch cost pipeline because they are never threaded
through to `batch_cost_calculator`.
"""

import pytest

from litellm.batches.batch_utils import (
    _batch_cost_calculator,
    _get_batch_job_cost_from_file_content,
    calculate_batch_cost_and_usage,
)
from litellm.cost_calculator import batch_cost_calculator
from litellm.types.utils import Usage


# --- helpers ---

def _make_batch_output_line(prompt_tokens: int = 10, completion_tokens: int = 5):
    """Return a single successful batch output line (OpenAI JSONL format)."""
    return {
        "id": "batch_req_1",
        "custom_id": "req-1",
        "response": {
            "status_code": 200,
            "body": {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "fake-batch-model",
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
            },
        },
        "error": None,
    }


CUSTOM_MODEL_INFO = {
    "input_cost_per_token_batches": 0.00125,
    "output_cost_per_token_batches": 0.005,
}


# --- tests ---


def test_batch_cost_calculator_uses_custom_model_info():
    """batch_cost_calculator should use model_info override when provided."""
    usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    prompt_cost, completion_cost = batch_cost_calculator(
        usage=usage,
        model="fake-batch-model",
        custom_llm_provider="openai",
        model_info=CUSTOM_MODEL_INFO,
    )

    expected_prompt = 10 * 0.00125
    expected_completion = 5 * 0.005
    assert prompt_cost == pytest.approx(expected_prompt), (
        f"Expected prompt cost {expected_prompt}, got {prompt_cost}"
    )
    assert completion_cost == pytest.approx(expected_completion), (
        f"Expected completion cost {expected_completion}, got {completion_cost}"
    )


def test_get_batch_job_cost_from_file_content_uses_custom_model_info():
    """_get_batch_job_cost_from_file_content should thread model_info to completion_cost."""
    file_content = [_make_batch_output_line(prompt_tokens=10, completion_tokens=5)]

    cost = _get_batch_job_cost_from_file_content(
        file_content_dictionary=file_content,
        custom_llm_provider="openai",
        model_info=CUSTOM_MODEL_INFO,
    )

    expected = (10 * 0.00125) + (5 * 0.005)
    assert cost == pytest.approx(expected), (
        f"Expected total cost {expected}, got {cost}"
    )


def test_batch_cost_calculator_func_uses_custom_model_info():
    """_batch_cost_calculator should thread model_info."""
    file_content = [_make_batch_output_line(prompt_tokens=10, completion_tokens=5)]

    cost = _batch_cost_calculator(
        file_content_dictionary=file_content,
        custom_llm_provider="openai",
        model_info=CUSTOM_MODEL_INFO,
    )

    expected = (10 * 0.00125) + (5 * 0.005)
    assert cost == pytest.approx(expected), (
        f"Expected total cost {expected}, got {cost}"
    )


@pytest.mark.asyncio
async def test_calculate_batch_cost_and_usage_uses_custom_model_info():
    """calculate_batch_cost_and_usage should thread model_info."""
    file_content = [_make_batch_output_line(prompt_tokens=10, completion_tokens=5)]

    batch_cost, batch_usage, batch_models = await calculate_batch_cost_and_usage(
        file_content_dictionary=file_content,
        custom_llm_provider="openai",
        model_info=CUSTOM_MODEL_INFO,
    )

    expected = (10 * 0.00125) + (5 * 0.005)
    assert batch_cost == pytest.approx(expected), (
        f"Expected total cost {expected}, got {batch_cost}"
    )
    assert batch_usage.prompt_tokens == 10
    assert batch_usage.completion_tokens == 5
