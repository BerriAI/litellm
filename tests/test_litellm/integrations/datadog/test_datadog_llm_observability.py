import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

# Adds the grandparent directory to sys.path to allow importing project modules
sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.datadog.datadog_llm_obs import DataDogLLMObsLogger
from litellm.types.integrations.datadog_llm_obs import (
    DatadogLLMObsInitParams,
    LLMMetrics,
    LLMObsPayload,
)
from litellm.types.utils import (
    StandardLoggingHiddenParams,
    StandardLoggingMetadata,
    StandardLoggingModelInformation,
    StandardLoggingPayload,
)


def create_standard_logging_payload_with_cache() -> StandardLoggingPayload:
    """Create a real StandardLoggingPayload object for testing"""
    return StandardLoggingPayload(
        id="test-request-id-456",
        call_type="completion",
        response_cost=0.05,
        response_cost_failure_debug_info=None,
        status="success",
        total_tokens=30,
        prompt_tokens=10,
        completion_tokens=20,
        startTime=1234567890.0,
        endTime=1234567891.0,
        completionStartTime=1234567890.5,
        model_map_information=StandardLoggingModelInformation(
            model_map_key="gpt-4", model_map_value=None
        ),
        model="gpt-4",
        model_id="model-123",
        model_group="openai-gpt",
        api_base="https://api.openai.com",
        metadata=StandardLoggingMetadata(
            user_api_key_hash="test_hash",
            user_api_key_org_id=None,
            user_api_key_alias="test_alias",
            user_api_key_team_id="test_team",
            user_api_key_user_id="test_user",
            user_api_key_team_alias="test_team_alias",
            spend_logs_metadata=None,
            requester_ip_address="127.0.0.1",
            requester_metadata=None,
        ),
        cache_hit=True,
        cache_key="test-cache-key-789",
        saved_cache_cost=0.02,
        request_tags=[],
        end_user=None,
        requester_ip_address="127.0.0.1",
        messages=[{"role": "user", "content": "Hello, world!"}],
        response={"choices": [{"message": {"content": "Hi there!"}}]},
        error_str=None,
        model_parameters={"stream": True},
        hidden_params=StandardLoggingHiddenParams(
            model_id="model-123",
            cache_key="test-cache-key-789",
            api_base="https://api.openai.com",
            response_cost="0.05",
            additional_headers=None,
        ),
        trace_id="test-trace-id-123",
        custom_llm_provider="openai",
    )


class TestDataDogLLMObsLogger:
    """Test suite for DataDog LLM Observability Logger"""

    @pytest.fixture
    def mock_env_vars(self):
        """Mock environment variables for DataDog"""
        with patch.dict(os.environ, {
            "DD_API_KEY": "test_api_key",
            "DD_SITE": "us5.datadoghq.com"
        }):
            yield

    @pytest.fixture
    def mock_response_obj(self):
        """Create a mock response object"""
        mock_response = Mock()
        mock_response.__getitem__ = Mock(return_value={
            "choices": [{"message": Mock(json=Mock(return_value={"role": "assistant", "content": "Hello!"}))}]
        })
        return mock_response

    def test_cost_and_trace_id_integration(self, mock_env_vars, mock_response_obj):
        """Test that total_cost is passed and trace_id from standard payload is used"""
        with patch('litellm.integrations.datadog.datadog_llm_obs.get_async_httpx_client'), \
             patch('asyncio.create_task'):
            logger = DataDogLLMObsLogger()
            
            standard_payload = create_standard_logging_payload_with_cache()
            
            kwargs = {
                "standard_logging_object": standard_payload,
                "litellm_params": {"metadata": {"trace_id": "old-trace-id-should-be-ignored"}}
            }
            
            start_time = datetime.now()
            end_time = datetime.now()
            
            payload = logger.create_llm_obs_payload(kwargs, mock_response_obj, start_time, end_time)
            
            # Test 1: Verify total_cost is correctly extracted from response_cost
            assert payload["metrics"].get("total_cost") == 0.05
            
            # Test 2: Verify trace_id comes from standard_logging_payload, not metadata
            assert payload["trace_id"] == "test-trace-id-123"
            
            # Test 3: Verify saved_cache_cost is in metadata 
            metadata = payload["meta"]["metadata"]
            assert metadata["saved_cache_cost"] == 0.02
            assert metadata["cache_hit"] == True
            assert metadata["cache_key"] == "test-cache-key-789"

    def test_cache_metadata_fields(self, mock_env_vars, mock_response_obj):
        """Test that cache-related metadata fields are correctly tracked"""
        with patch('litellm.integrations.datadog.datadog_llm_obs.get_async_httpx_client'), \
             patch('asyncio.create_task'):
            logger = DataDogLLMObsLogger()
            
            standard_payload = create_standard_logging_payload_with_cache()
            
            kwargs = {
                "standard_logging_object": standard_payload,
                "litellm_params": {"metadata": {}}
            }
            
            start_time = datetime.now()
            end_time = datetime.now()
            
            payload = logger.create_llm_obs_payload(kwargs, mock_response_obj, start_time, end_time)
            
            # Test the _get_dd_llm_obs_payload_metadata method directly
            metadata = logger._get_dd_llm_obs_payload_metadata(standard_payload)
            
            # Verify all cache-related fields are present
            assert metadata["cache_hit"] == True
            assert metadata["cache_key"] == "test-cache-key-789"
            assert metadata["saved_cache_cost"] == 0.02
            assert metadata["id"] == "test-request-id-456"
            assert metadata["trace_id"] == "test-trace-id-123"
            assert metadata["model_name"] == "gpt-4"
            assert metadata["model_provider"] == "openai"

    def test_get_time_to_first_token_seconds(self, mock_env_vars):
        """Test the _get_time_to_first_token_seconds method for streaming calls"""
        with patch('litellm.integrations.datadog.datadog_llm_obs.get_async_httpx_client'), \
             patch('asyncio.create_task'):
            logger = DataDogLLMObsLogger()
            
            # Test streaming case (completion_start_time available)
            streaming_payload = create_standard_logging_payload_with_cache()
            # Modify times for testing: start=1000, completion_start=1002, end=1005
            streaming_payload["startTime"] = 1000.0
            streaming_payload["completionStartTime"] = 1002.0
            streaming_payload["endTime"] = 1005.0
            
            # Test streaming case: should use completion_start_time - start_time
            time_to_first_token = logger._get_time_to_first_token_seconds(streaming_payload)
            assert time_to_first_token == 2.0  # 1002.0 - 1000.0 = 2.0 seconds


    def test_datadog_span_kind_mapping(self, mock_env_vars):
        """Test that call_type values are correctly mapped to DataDog span kinds"""
        from litellm.types.utils import CallTypes
        
        with patch('litellm.integrations.datadog.datadog_llm_obs.get_async_httpx_client'), \
            patch('asyncio.create_task'):
            logger = DataDogLLMObsLogger()
        
        # Test embedding operations
        assert logger._get_datadog_span_kind(CallTypes.embedding.value) == "embedding"
        assert logger._get_datadog_span_kind(CallTypes.aembedding.value) == "embedding"
        
        # Test LLM completion operations
        assert logger._get_datadog_span_kind(CallTypes.completion.value) == "llm"
        assert logger._get_datadog_span_kind(CallTypes.acompletion.value) == "llm"
        assert logger._get_datadog_span_kind(CallTypes.text_completion.value) == "llm"
        assert logger._get_datadog_span_kind(CallTypes.generate_content.value) == "llm"
        assert logger._get_datadog_span_kind(CallTypes.anthropic_messages.value) == "llm"
        
        # Test tool operations
        assert logger._get_datadog_span_kind(CallTypes.call_mcp_tool.value) == "tool"
        
        # Test retrieval operations
        assert logger._get_datadog_span_kind(CallTypes.get_assistants.value) == "retrieval"
        assert logger._get_datadog_span_kind(CallTypes.file_retrieve.value) == "retrieval"
        assert logger._get_datadog_span_kind(CallTypes.retrieve_batch.value) == "retrieval"
        
        # Test task operations
        assert logger._get_datadog_span_kind(CallTypes.create_batch.value) == "task"
        assert logger._get_datadog_span_kind(CallTypes.image_generation.value) == "task"
        assert logger._get_datadog_span_kind(CallTypes.moderation.value) == "task"
        assert logger._get_datadog_span_kind(CallTypes.transcription.value) == "task"
        
        # Test default fallback
        assert logger._get_datadog_span_kind("unknown_call_type") == "llm"
        assert logger._get_datadog_span_kind(None) == "llm"



class TestDataDogLLMObsLogger(DataDogLLMObsLogger):
    """Test suite for DataDog LLM Observability Logger"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logged_standard_logging_payload: Optional[StandardLoggingPayload] = None
    
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.logged_standard_logging_payload = kwargs.get("standard_logging_object")


class TestS3Logger(CustomLogger):
    """Test suite for S3 Logger"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logged_standard_logging_payload: Optional[StandardLoggingPayload] = None
    
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.logged_standard_logging_payload = kwargs.get("standard_logging_object")


@pytest.mark.asyncio
async def test_dd_llms_obs_redaction(mock_env_vars):
    # init DD with turn_off_message_logging=True
    litellm._turn_on_debug()
    from litellm.types.utils import LiteLLMCommonStrings
    litellm.datadog_llm_observability_params = DatadogLLMObsInitParams(turn_off_message_logging=True)
    dd_llms_obs_logger = TestDataDogLLMObsLogger()
    test_s3_logger = TestS3Logger()
    litellm.callbacks = [
        dd_llms_obs_logger,
        test_s3_logger
    ]

    # call litellm
    await litellm.acompletion(
        model="gpt-4o",
        mock_response="Hi there!",
        messages=[{"role": "user", "content": "Hello, world!"}]
    )

    # sleep 1 second for logging to complete
    await asyncio.sleep(1)

    #################
    # test validation 
    # 1. both loggers logged a standard_logging_payload
    # 2. DD LLM Obs standard_logging_payload has messages and response redacted
    # 3. S3 standard_logging_payload does not have messages and response redacted

    assert dd_llms_obs_logger.logged_standard_logging_payload is not None
    assert test_s3_logger.logged_standard_logging_payload is not None

    print("logged DD LLM Obs payload", json.dumps(dd_llms_obs_logger.logged_standard_logging_payload, indent=4, default=str))
    print("\n\nlogged S3 payload", json.dumps(test_s3_logger.logged_standard_logging_payload, indent=4, default=str))

    assert dd_llms_obs_logger.logged_standard_logging_payload["messages"][0]["content"] == LiteLLMCommonStrings.redacted_by_litellm.value
    assert dd_llms_obs_logger.logged_standard_logging_payload["response"]["choices"][0]["message"]["content"] == LiteLLMCommonStrings.redacted_by_litellm.value

    assert test_s3_logger.logged_standard_logging_payload["messages"] == [{"role": "user", "content": "Hello, world!"}]
    assert test_s3_logger.logged_standard_logging_payload["response"]["choices"][0]["message"]["content"] == "Hi there!"
    
    
@pytest.fixture
def mock_env_vars():
    """Mock environment variables for DataDog"""
    with patch.dict(os.environ, {
        "DD_API_KEY": "test_api_key",
        "DD_SITE": "us5.datadoghq.com"
    }):
        yield

@pytest.mark.asyncio
async def test_create_llm_obs_payload(mock_env_vars):
    datadog_llm_obs_logger = DataDogLLMObsLogger()
    standard_logging_payload = create_standard_logging_payload_with_cache()
    payload = datadog_llm_obs_logger.create_llm_obs_payload(
        kwargs={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "standard_logging_object": standard_logging_payload,
        },
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(seconds=1),
    )

    print("dd created payload", payload)

    assert payload["name"] == "litellm_llm_call"
    assert payload["meta"]["kind"] == "llm"
    assert payload["meta"]["input"]["messages"] == [
        {"role": "user", "content": "Hello, world!"}
    ]
    assert payload["meta"]["output"]["messages"][0]["content"] == "Hi there!"
    assert payload["metrics"]["input_tokens"] == 10
    assert payload["metrics"]["output_tokens"] == 20
    assert payload["metrics"]["total_tokens"] == 30
