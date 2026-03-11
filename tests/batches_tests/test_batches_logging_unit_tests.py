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
from litellm.batches.batch_utils import (
    _batch_cost_calculator,
    _get_file_content_as_dictionary,
    _get_batch_job_cost_from_file_content,
    _get_batch_job_total_usage_from_file_content,
    _get_batch_job_usage_from_response_body,
    _get_response_from_batch_job_output_file,
    _batch_response_was_successful,
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


@pytest.mark.asyncio
async def test_batch_retrieve_cost_tracking_with_completed_batch_no_explicit_cost():
    """
    Test that cost is calculated for completed batches when no explicit cost data is provided.
    
    Regression test for: When batch status is "completed" and explicit batch_cost/batch_usage/batch_models
    are not provided, the system should compute batch data by calling _handle_completed_batch.
    """
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.types.utils import CallTypes
    from litellm.types.utils import LiteLLMBatch
    from unittest.mock import AsyncMock, patch
    
    # Mock batch result with completed status
    mock_batch = LiteLLMBatch(
        id="batch-test-123",
        object="batch",
        endpoint="/v1/chat/completions",
        errors=None,
        input_file_id="file-input-123",
        completion_window="24h",
        status="completed",
        output_file_id="file-output-123",
        error_file_id=None,
        created_at=1234567890,
        in_progress_at=1234567900,
        expires_at=1234654290,
        finalizing_at=1234568000,
        completed_at=1234568100,
        failed_at=None,
        expired_at=None,
        cancelling_at=None,
        cancelled_at=None,
        request_counts={
            "total": 10,
            "completed": 10,
            "failed": 0,
        },
        metadata=None,
    )
    mock_batch._hidden_params = {}
    
    # Create logging object
    logging_obj = Logging(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "test"}],
        stream=False,
        call_type=CallTypes.aretrieve_batch.value,
        litellm_call_id="test-call-123",
        function_id="test-function",
        start_time=time.time(),
        dynamic_success_callbacks=[],
    )
    logging_obj.custom_llm_provider = "openai"
    
    # Mock _handle_completed_batch to return cost data
    expected_cost = 0.05
    expected_usage = litellm.Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )
    expected_models = ["gpt-4o-mini"]
    
    with patch(
        "litellm.litellm_core_utils.litellm_logging._handle_completed_batch",
        new=AsyncMock(return_value=(expected_cost, expected_usage, expected_models))
    ) as mock_handle_batch:
        # Call async_success_handler
        await logging_obj.async_success_handler(
            result=mock_batch,
            start_time=time.time(),
            end_time=time.time() + 1,
        )
        
        # Verify _handle_completed_batch was called
        mock_handle_batch.assert_called_once()
        
        # Verify cost and usage were set on the batch result
        assert mock_batch._hidden_params["response_cost"] == expected_cost
        assert mock_batch._hidden_params["batch_models"] == expected_models
        assert mock_batch.usage == expected_usage


@pytest.mark.asyncio
async def test_batch_retrieve_cost_tracking_with_explicit_cost_data():
    """
    Test that explicit cost data is used when provided, skipping computation.
    
    Regression test for: When batch_cost, batch_usage, and batch_models are explicitly
    provided in kwargs, they should be used directly without calling _handle_completed_batch.
    """
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.types.utils import CallTypes
    from litellm.types.utils import LiteLLMBatch
    from unittest.mock import AsyncMock, patch
    
    # Mock batch result with completed status
    mock_batch = LiteLLMBatch(
        id="batch-test-456",
        object="batch",
        endpoint="/v1/chat/completions",
        errors=None,
        input_file_id="file-input-456",
        completion_window="24h",
        status="completed",
        output_file_id="file-output-456",
        error_file_id=None,
        created_at=1234567890,
        in_progress_at=1234567900,
        expires_at=1234654290,
        finalizing_at=1234568000,
        completed_at=1234568100,
        failed_at=None,
        expired_at=None,
        cancelling_at=None,
        cancelled_at=None,
        request_counts={
            "total": 5,
            "completed": 5,
            "failed": 0,
        },
        metadata=None,
    )
    mock_batch._hidden_params = {}
    
    # Create logging object
    logging_obj = Logging(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "test"}],
        stream=False,
        call_type=CallTypes.aretrieve_batch.value,
        litellm_call_id="test-call-456",
        function_id="test-function",
        start_time=time.time(),
        dynamic_success_callbacks=[],
    )
    logging_obj.custom_llm_provider = "openai"
    
    # Explicit cost data to pass in kwargs
    explicit_cost = 0.10
    explicit_usage = litellm.Usage(
        prompt_tokens=200,
        completion_tokens=100,
        total_tokens=300,
    )
    explicit_models = ["gpt-4o-mini", "gpt-3.5-turbo"]
    
    with patch(
        "litellm.litellm_core_utils.litellm_logging._handle_completed_batch",
        new=AsyncMock()
    ) as mock_handle_batch:
        # Call async_success_handler with explicit cost data
        await logging_obj.async_success_handler(
            result=mock_batch,
            start_time=time.time(),
            end_time=time.time() + 1,
            batch_cost=explicit_cost,
            batch_usage=explicit_usage,
            batch_models=explicit_models,
        )
        
        # Verify _handle_completed_batch was NOT called (since explicit data provided)
        mock_handle_batch.assert_not_called()
        
        # Verify explicit cost data was used
        assert mock_batch._hidden_params["response_cost"] == explicit_cost
        assert mock_batch._hidden_params["batch_models"] == explicit_models
        assert mock_batch.usage == explicit_usage


@pytest.mark.asyncio
async def test_batch_retrieve_cost_tracking_with_unified_file_id_incomplete_batch():
    """
    Test that cost computation is skipped for unified file IDs with non-completed batches.
    
    Regression test for: For unified file IDs (base64 encoded), cost should only be computed 
    when batch status is "completed" and explicit data is not provided.
    """
    import base64
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.types.utils import CallTypes, SpecialEnums
    from litellm.types.utils import LiteLLMBatch
    from unittest.mock import AsyncMock, patch
    
    # Create a proper unified file ID by encoding the correct prefix
    unified_id_str = f"{SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value}:test_file_789;unified_id:batch-789"
    encoded_unified_id = base64.urlsafe_b64encode(unified_id_str.encode()).decode().rstrip("=")
    
    # Mock batch result with in_progress status and unified file ID
    mock_batch = LiteLLMBatch(
        id=encoded_unified_id,  # Properly encoded unified ID
        object="batch",
        endpoint="/v1/chat/completions",
        errors=None,
        input_file_id="file-input-789",
        completion_window="24h",
        status="in_progress",  # Not completed
        output_file_id=None,
        error_file_id=None,
        created_at=1234567890,
        in_progress_at=1234567900,
        expires_at=1234654290,
        finalizing_at=None,
        completed_at=None,
        failed_at=None,
        expired_at=None,
        cancelling_at=None,
        cancelled_at=None,
        request_counts={
            "total": 10,
            "completed": 3,
            "failed": 0,
        },
        metadata=None,
    )
    mock_batch._hidden_params = {}
    
    # Create logging object
    logging_obj = Logging(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "test"}],
        stream=False,
        call_type=CallTypes.aretrieve_batch.value,
        litellm_call_id="test-call-789",
        function_id="test-function",
        start_time=time.time(),
        dynamic_success_callbacks=[],
    )
    logging_obj.custom_llm_provider = "openai"

    with patch(
        "litellm.litellm_core_utils.litellm_logging._handle_completed_batch",
        new=AsyncMock()
    ) as mock_handle_batch:
        # Call async_success_handler with in_progress batch (unified file ID)
        await logging_obj.async_success_handler(
            result=mock_batch,
            start_time=time.time(),
            end_time=time.time() + 1,
        )
        
        # Verify _handle_completed_batch was NOT called (batch not completed and is unified file ID)
        mock_handle_batch.assert_not_called()
        
        # Verify cost data was not set
        assert "response_cost" not in mock_batch._hidden_params
        assert "batch_models" not in mock_batch._hidden_params
        assert not hasattr(mock_batch, "usage") or mock_batch.usage is None


@pytest.mark.asyncio
async def test_batch_retrieve_cost_tracking_with_partial_explicit_data():
    """
    Test that cost is computed when only partial explicit data is provided.
    
    Regression test for: If batch_cost, batch_usage, or batch_models is missing
    (not all three provided), and batch is completed, system should compute the data.
    """
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.types.utils import CallTypes
    from litellm.types.utils import LiteLLMBatch
    from unittest.mock import AsyncMock, patch
    
    # Mock batch result with completed status
    mock_batch = LiteLLMBatch(
        id="batch-test-partial",
        object="batch",
        endpoint="/v1/chat/completions",
        errors=None,
        input_file_id="file-input-partial",
        completion_window="24h",
        status="completed",
        output_file_id="file-output-partial",
        error_file_id=None,
        created_at=1234567890,
        in_progress_at=1234567900,
        expires_at=1234654290,
        finalizing_at=1234568000,
        completed_at=1234568100,
        failed_at=None,
        expired_at=None,
        cancelling_at=None,
        cancelled_at=None,
        request_counts={
            "total": 8,
            "completed": 8,
            "failed": 0,
        },
        metadata=None,
    )
    mock_batch._hidden_params = {}
    
    # Create logging object
    logging_obj = Logging(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "test"}],
        stream=False,
        call_type=CallTypes.aretrieve_batch.value,
        litellm_call_id="test-call-partial",
        function_id="test-function",
        start_time=time.time(),
        dynamic_success_callbacks=[],
    )

    logging_obj.custom_llm_provider = "openai"
    
    # Only provide batch_cost, missing batch_usage and batch_models
    partial_cost = 0.08
    
    expected_cost = 0.06
    expected_usage = litellm.Usage(
        prompt_tokens=150,
        completion_tokens=75,
        total_tokens=225,
    )
    expected_models = ["gpt-4o-mini"]
    
    with patch(
        "litellm.litellm_core_utils.litellm_logging._handle_completed_batch",
        new=AsyncMock(return_value=(expected_cost, expected_usage, expected_models))
    ) as mock_handle_batch:
        # Call async_success_handler with partial explicit data
        await logging_obj.async_success_handler(
            result=mock_batch,
            start_time=time.time(),
            end_time=time.time() + 1,
            batch_cost=partial_cost,  # Only cost provided, not usage or models
        )
        
        # Verify _handle_completed_batch WAS called (since not all data provided)
        mock_handle_batch.assert_called_once()
        
        # Verify computed cost data was used (not partial explicit data)
        assert mock_batch._hidden_params["response_cost"] == expected_cost
        assert mock_batch._hidden_params["batch_models"] == expected_models
        assert mock_batch.usage == expected_usage
