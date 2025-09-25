import asyncio
import json
import os
import sys
import traceback
from unittest.mock import AsyncMock, MagicMock, patch
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path
import logging
import time

import pytest
from typing import Optional
import litellm
from litellm import create_batch, create_file
from litellm._logging import verbose_logger
from litellm.types.llms.openai import Batch
from litellm.types.utils import Usage
from litellm.batches.batch_utils import (
    _batch_cost_calculator,
    _get_file_content_as_dictionary,
    _get_batch_job_cost_from_file_content,
    _get_batch_job_total_usage_from_file_content,
    _get_batch_job_usage_from_response_body,
    _get_response_from_batch_job_output_file,
    _batch_response_was_successful,
    calculate_batch_cost_and_usage,
    _handle_completed_batch,
)


@pytest.fixture
def sample_file_content():
    return b"""
{"id": "batch_req_6769ca596b38819093d7ae9f522de924", "custom_id": "request-1", "response": {"status_code": 200, "request_id": "07bc45ab4e7e26ac23a0c949973327e7", "body": {"id": "chatcmpl-AhjSMl7oZ79yIPHLRYgmgXSixTJr7", "object": "chat.completion", "created": 1734986202, "model": "gpt-4o-mini-2024-07-18", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello! How can I assist you today?", "refusal": null}, "logprobs": null, "finish_reason": "stop"}], "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0}, "completion_tokens_details": {"reasoning_tokens": 0, "audio_tokens": 0, "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0}}, "system_fingerprint": "fp_0aa8d3e20b"}}, "error": null}
{"id": "batch_req_6769ca597e588190920666612634e2b4", "custom_id": "request-2", "response": {"status_code": 200, "request_id": "82e04f4c001fe2c127cbad199f5fd31b", "body": {"id": "chatcmpl-AhjSNgVB4Oa4Hq0NruTRsBaEbRWUP", "object": "chat.completion", "created": 1734986203, "model": "gpt-4o-mini-2024-07-18", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello! What can I do for you today?", "refusal": null}, "logprobs": null, "finish_reason": "length"}], "usage": {"prompt_tokens": 22, "completion_tokens": 10, "total_tokens": 32, "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0}, "completion_tokens_details": {"reasoning_tokens": 0, "audio_tokens": 0, "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0}}, "system_fingerprint": "fp_0aa8d3e20b"}}, "error": null}
"""


@pytest.fixture
def sample_file_content_dict():
    return [
        {
            "id": "batch_req_6769ca596b38819093d7ae9f522de924",
            "custom_id": "request-1",
            "response": {
                "status_code": 200,
                "request_id": "07bc45ab4e7e26ac23a0c949973327e7",
                "body": {
                    "id": "chatcmpl-AhjSMl7oZ79yIPHLRYgmgXSixTJr7",
                    "object": "chat.completion",
                    "created": 1734986202,
                    "model": "gpt-4o-mini-2024-07-18",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Hello! How can I assist you today?",
                                "refusal": None,
                            },
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 10,
                        "total_tokens": 30,
                        "prompt_tokens_details": {
                            "cached_tokens": 0,
                            "audio_tokens": 0,
                        },
                        "completion_tokens_details": {
                            "reasoning_tokens": 0,
                            "audio_tokens": 0,
                            "accepted_prediction_tokens": 0,
                            "rejected_prediction_tokens": 0,
                        },
                    },
                    "system_fingerprint": "fp_0aa8d3e20b",
                },
            },
            "error": None,
        },
        {
            "id": "batch_req_6769ca597e588190920666612634e2b4",
            "custom_id": "request-2",
            "response": {
                "status_code": 200,
                "request_id": "82e04f4c001fe2c127cbad199f5fd31b",
                "body": {
                    "id": "chatcmpl-AhjSNgVB4Oa4Hq0NruTRsBaEbRWUP",
                    "object": "chat.completion",
                    "created": 1734986203,
                    "model": "gpt-4o-mini-2024-07-18",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Hello! What can I do for you today?",
                                "refusal": None,
                            },
                            "logprobs": None,
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 22,
                        "completion_tokens": 10,
                        "total_tokens": 32,
                        "prompt_tokens_details": {
                            "cached_tokens": 0,
                            "audio_tokens": 0,
                        },
                        "completion_tokens_details": {
                            "reasoning_tokens": 0,
                            "audio_tokens": 0,
                            "accepted_prediction_tokens": 0,
                            "rejected_prediction_tokens": 0,
                        },
                    },
                    "system_fingerprint": "fp_0aa8d3e20b",
                },
            },
            "error": None,
        },
    ]


def test_get_file_content_as_dictionary(sample_file_content):
    result = _get_file_content_as_dictionary(sample_file_content)
    assert len(result) == 2
    assert result[0]["id"] == "batch_req_6769ca596b38819093d7ae9f522de924"
    assert result[0]["custom_id"] == "request-1"
    assert result[0]["response"]["status_code"] == 200
    assert result[0]["response"]["body"]["usage"]["total_tokens"] == 30


def test_get_batch_job_total_usage_from_file_content(sample_file_content_dict):
    usage = _get_batch_job_total_usage_from_file_content(
        sample_file_content_dict, custom_llm_provider="openai"
    )
    assert usage.total_tokens == 62  # 30 + 32
    assert usage.prompt_tokens == 42  # 20 + 22
    assert usage.completion_tokens == 20  # 10 + 10


@pytest.mark.asyncio
async def test_batch_cost_calculator(sample_file_content_dict):
    """
    mock litellm.completion_cost to return 0.5

    we know sample_file_content_dict has 2 successful responses

    so we expect the cost to be 0.5 * 2 = 1.0
    """
    with patch("litellm.completion_cost", return_value=0.5):
        cost = _batch_cost_calculator(
            file_content_dictionary=sample_file_content_dict,
            custom_llm_provider="openai",
        )
        assert cost == 1.0  # 0.5 * 2 successful responses


def test_get_response_from_batch_job_output_file(sample_file_content_dict):
    result = _get_response_from_batch_job_output_file(sample_file_content_dict[0])
    assert result["id"] == "chatcmpl-AhjSMl7oZ79yIPHLRYgmgXSixTJr7"
    assert result["object"] == "chat.completion"
    assert result["usage"]["total_tokens"] == 30


# Additional fixtures for batch testing
@pytest.fixture
def batch_with_usage_and_model():
    """Create a mock batch object with usage and model attributes."""
    batch = MagicMock(spec=Batch)
    batch.usage = Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150
    )
    batch.model = "gpt-4o-mini-2024-07-18"
    return batch


@pytest.fixture
def batch_without_usage_and_model():
    """Create a mock batch object without usage and model attributes."""
    batch = MagicMock(spec=Batch)
    batch.usage = None
    batch.model = None
    batch.output_file_id = "file-abc123"
    return batch


# Tests for calculate_batch_cost_and_usage method
@pytest.mark.asyncio
async def test_calculate_batch_cost_and_usage_with_batch_usage_and_model(
    batch_with_usage_and_model, sample_file_content_dict
):
    """Test calculate_batch_cost_and_usage when batch has usage and model."""
    with patch("litellm.cost_calculator.batch_cost_calculator", return_value=(0.001, 0.002)):
        batch_cost, batch_usage, batch_models = await calculate_batch_cost_and_usage(
            batch=batch_with_usage_and_model,
            file_content_dictionary=sample_file_content_dict,
            custom_llm_provider="openai",
        )
        
        # Should use batch.usage and batch.model, not file content
        assert batch_cost == 0.003  # 0.001 + 0.002
        assert batch_usage.prompt_tokens == 100
        assert batch_usage.completion_tokens == 50
        assert batch_usage.total_tokens == 150
        assert batch_models == ["gpt-4o-mini-2024-07-18"]  # Single model from batch.model


@pytest.mark.asyncio
async def test_calculate_batch_cost_and_usage_without_batch_usage_and_model(
    batch_without_usage_and_model, sample_file_content_dict
):
    """Test calculate_batch_cost_and_usage when batch doesn't have usage and model."""
    with patch("litellm.completion_cost", return_value=0.5):
        batch_cost, batch_usage, batch_models = await calculate_batch_cost_and_usage(
            batch=batch_without_usage_and_model,
            file_content_dictionary=sample_file_content_dict,
            custom_llm_provider="openai",
        )
        
        # Should calculate from file content
        assert batch_cost == 1.0  # 0.5 * 2 successful responses
        assert batch_usage.prompt_tokens == 42  # 20 + 22
        assert batch_usage.completion_tokens == 20  # 10 + 10
        assert batch_usage.total_tokens == 62  # 30 + 32
        assert batch_models == ["gpt-4o-mini-2024-07-18", "gpt-4o-mini-2024-07-18"]  # Models from file content


# Tests for _handle_completed_batch method
@pytest.mark.asyncio
async def test_handle_completed_batch_with_batch_usage_and_model(batch_with_usage_and_model):
    """Test _handle_completed_batch when batch has usage and model."""
    with patch("litellm.cost_calculator.batch_cost_calculator", return_value=(0.001, 0.002)):
        batch_cost, batch_usage, batch_models = await _handle_completed_batch(
            batch=batch_with_usage_and_model,
            custom_llm_provider="openai",
        )
        
        # Should use batch.usage and batch.model, not load file content
        assert batch_cost == 0.003  # 0.001 + 0.002
        assert batch_usage.prompt_tokens == 100
        assert batch_usage.completion_tokens == 50
        assert batch_usage.total_tokens == 150
        assert batch_models == ["gpt-4o-mini-2024-07-18"]  # Single model from batch.model


@pytest.mark.asyncio
async def test_handle_completed_batch_without_batch_usage_and_model(
    batch_without_usage_and_model, sample_file_content_dict
):
    """Test _handle_completed_batch when batch doesn't have usage and model."""
    with patch(
        "litellm.batches.batch_utils._get_batch_output_file_content_as_dictionary",
        return_value=sample_file_content_dict
    ):
        with patch("litellm.completion_cost", return_value=0.5):
            batch_cost, batch_usage, batch_models = await _handle_completed_batch(
                batch=batch_without_usage_and_model,
                custom_llm_provider="openai",
            )
            
            # Should load file content and calculate from it
            assert batch_cost == 1.0  # 0.5 * 2 successful responses
            assert batch_usage.prompt_tokens == 42  # 20 + 22
            assert batch_usage.completion_tokens == 20  # 10 + 10
            assert batch_usage.total_tokens == 62  # 30 + 32
            assert batch_models == ["gpt-4o-mini-2024-07-18", "gpt-4o-mini-2024-07-18"]  # Models from file content
