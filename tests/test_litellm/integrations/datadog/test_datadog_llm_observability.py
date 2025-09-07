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
    StandardLoggingGuardrailInformation,
    StandardLoggingHiddenParams,
    StandardLoggingMetadata,
    StandardLoggingModelInformation,
    StandardLoggingPayload,
    StandardLoggingPayloadErrorInformation,
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


def create_standard_logging_payload_with_failure() -> StandardLoggingPayload:
    """Create a StandardLoggingPayload object for failure testing"""
    return StandardLoggingPayload(
        id="test-request-id-failure-789",
        call_type="completion",
        response_cost=0.0,
        response_cost_failure_debug_info=None,
        status="failure",
        total_tokens=0,
        prompt_tokens=10,
        completion_tokens=0,
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
        cache_hit=False,
        cache_key=None,
        saved_cache_cost=0.0,
        request_tags=[],
        end_user=None,
        requester_ip_address="127.0.0.1",
        messages=[{"role": "user", "content": "Hello, world!"}],
        response=None,
        error_str="RateLimitError: You exceeded your current quota",
        error_information=StandardLoggingPayloadErrorInformation(
            error_code="rate_limit_exceeded",
            error_class="RateLimitError",
            llm_provider="openai",
            traceback="Traceback (most recent call last):\n  File test.py, line 1\n    RateLimitError: You exceeded your current quota",
            error_message="RateLimitError: You exceeded your current quota"
        ),
        model_parameters={"stream": False},
        hidden_params=StandardLoggingHiddenParams(
            model_id="model-123",
            cache_key=None,
            api_base="https://api.openai.com",
            response_cost="0.0",
            additional_headers=None,
        ),
        trace_id="test-trace-id-failure-456",
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
            
            payload = logger.create_llm_obs_payload(kwargs, start_time, end_time)
            
            # Test 1: Verify total_cost is correctly extracted from response_cost
            assert payload["metrics"].get("total_cost") == 0.05
            
            # Test 2: Verify trace_id comes from standard_logging_payload, not metadata
            assert payload["trace_id"] == "test-trace-id-123"
            
            # Test 3: Verify saved_cache_cost is in metadata 
            metadata = payload["meta"]["metadata"]
            assert metadata["saved_cache_cost"] == 0.02
            assert metadata["cache_hit"] == True
            assert metadata["cache_key"] == "test-cache-key-789"

    def test_apm_id_included(self, mock_env_vars, mock_response_obj):
        """Test that the current APM trace ID is attached to the payload"""
        with patch('litellm.integrations.datadog.datadog_llm_obs.get_async_httpx_client'), \
             patch('asyncio.create_task'):
            fake_tracer = MagicMock()
            fake_span = MagicMock()
            fake_span.trace_id = 987654321
            fake_tracer.current_span.return_value = fake_span

            with patch('litellm.integrations.datadog.datadog_llm_obs.tracer', fake_tracer):
                logger = DataDogLLMObsLogger()

                standard_payload = create_standard_logging_payload_with_cache()

                kwargs = {
                    "standard_logging_object": standard_payload,
                    "litellm_params": {"metadata": {}}
                }

                start_time = datetime.now()
                end_time = datetime.now()

                payload = logger.create_llm_obs_payload(kwargs, start_time, end_time)

                assert payload["apm_id"] == str(fake_span.trace_id)

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
            
            payload = logger.create_llm_obs_payload(kwargs, start_time, end_time)
            
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

    @pytest.mark.asyncio
    async def test_async_log_failure_event(self, mock_env_vars):
        """Test that async_log_failure_event correctly processes failure payloads according to DD LLM Obs API spec"""
        with patch('litellm.integrations.datadog.datadog_llm_obs.get_async_httpx_client'), \
             patch('asyncio.create_task'):
            logger = DataDogLLMObsLogger()
            
            # Ensure log_queue starts empty
            logger.log_queue = []
            
            standard_failure_payload = create_standard_logging_payload_with_failure()
            
            kwargs = {
                "standard_logging_object": standard_failure_payload,
                "model": "gpt-4",
                "litellm_params": {"metadata": {}}
            }
            
            start_time = datetime.now()
            end_time = datetime.now() + timedelta(seconds=2)
            
            # Mock async_send_batch to prevent actual network calls
            with patch.object(logger, 'async_send_batch') as mock_send_batch:
                # Call the method under test
                await logger.async_log_failure_event(kwargs, None, start_time, end_time)
                
                # Verify payload was added to queue
                assert len(logger.log_queue) == 1
                
                # Verify the payload has correct failure characteristics according to DD LLM Obs API spec
                payload = logger.log_queue[0]
                assert payload["trace_id"] == "test-trace-id-failure-456"
                assert payload["meta"]["metadata"]["id"] == "test-request-id-failure-789"
                assert payload["status"] == "error"
                
                # Verify error information follows DD LLM Obs API spec
                assert payload["meta"]["error"]["message"] == "RateLimitError: You exceeded your current quota"
                assert payload["meta"]["error"]["type"] == "RateLimitError"
                assert payload["meta"]["error"]["stack"] == "Traceback (most recent call last):\n  File test.py, line 1\n    RateLimitError: You exceeded your current quota"
                
                assert payload["metrics"]["total_cost"] == 0.0
                assert payload["metrics"]["total_tokens"] == 0
                assert payload["metrics"]["output_tokens"] == 0
                
                # Verify batch sending not triggered (queue size < batch_size)
                mock_send_batch.assert_not_called()



class TestDataDogLLMObsLoggerForRedaction(DataDogLLMObsLogger):
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
    dd_llms_obs_logger = TestDataDogLLMObsLoggerForRedaction()
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


def create_standard_logging_payload_with_latency_metrics() -> StandardLoggingPayload:
    """Create a StandardLoggingPayload object with latency metrics for testing"""
    guardrail_info = StandardLoggingGuardrailInformation(
        guardrail_name="test_guardrail",
        guardrail_status="success",
        start_time=1234567890.0,
        end_time=1234567890.5,
        duration=0.5,  # 500ms
        guardrail_request={"input": "test input message", "user_id": "test_user"},
        guardrail_response={"output": "filtered output", "flagged": False, "score": 0.1},
    )
    
    hidden_params = StandardLoggingHiddenParams(
        model_id="model-123",
        cache_key="test-cache-key",
        api_base="https://api.openai.com",
        response_cost="0.05",
        litellm_overhead_time_ms=150.0,  # 150ms
        additional_headers=None,
    )
    
    return StandardLoggingPayload(
        id="test-request-id-latency",
        call_type="completion",
        response_cost=0.05,
        response_cost_failure_debug_info=None,
        status="success",
        total_tokens=30,
        prompt_tokens=10,
        completion_tokens=20,
        startTime=1234567890.0,
        endTime=1234567892.0,
        completionStartTime=1234567890.8,  # 800ms after start
        response_time=2.0,
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
        cache_hit=False,
        cache_key=None,
        saved_cache_cost=0.0,
        request_tags=[],
        end_user=None,
        requester_ip_address="127.0.0.1",
        messages=[{"role": "user", "content": "Hello, world!"}],
        response={"choices": [{"message": {"content": "Hi there!"}}]},
        error_str=None,
        error_information=None,
        model_parameters={"stream": True},
        hidden_params=hidden_params,
        guardrail_information=guardrail_info,
        trace_id="test-trace-id-latency",
        custom_llm_provider="openai",
    )


def test_latency_metrics_in_metadata(mock_env_vars):
    """Test that time to first token, litellm overhead, and guardrail overhead are included in metadata"""
    with patch('litellm.integrations.datadog.datadog_llm_obs.get_async_httpx_client'), \
         patch('asyncio.create_task'):
        logger = DataDogLLMObsLogger()
        
        standard_payload = create_standard_logging_payload_with_latency_metrics()
        
        kwargs = {
            "standard_logging_object": standard_payload,
            "litellm_params": {"metadata": {}}
        }
        
        start_time = datetime.now()
        end_time = datetime.now()
        
        # Test the metadata generation directly
        metadata = logger._get_dd_llm_obs_payload_metadata(standard_payload)
        latency_metadata = metadata.get("latency_metrics", {})
        
        # Verify time to first token is included (800ms)
        assert "time_to_first_token_ms" in latency_metadata
        assert abs(latency_metadata["time_to_first_token_ms"] - 800.0) < 0.001  # 0.8 seconds * 1000 with tolerance for floating-point precision
        
        # Verify litellm overhead is included (150ms)
        assert "litellm_overhead_time_ms" in latency_metadata
        assert latency_metadata["litellm_overhead_time_ms"] == 150.0
        
        # Verify guardrail overhead is included (500ms) 
        assert "guardrail_overhead_time_ms" in latency_metadata
        assert latency_metadata["guardrail_overhead_time_ms"] == 500.0  # 0.5 seconds * 1000
        
        # Verify these metrics are also included in the full payload
        payload = logger.create_llm_obs_payload(kwargs, start_time, end_time)
        payload_metadata_latency = payload["meta"]["metadata"]["latency_metrics"]
        
        assert abs(payload_metadata_latency["time_to_first_token_ms"] - 800.0) < 0.001
        assert payload_metadata_latency["litellm_overhead_time_ms"] == 150.0
        assert payload_metadata_latency["guardrail_overhead_time_ms"] == 500.0


def test_latency_metrics_edge_cases(mock_env_vars):
    """Test latency metrics with edge cases (missing fields, zero values, etc.)"""
    with patch('litellm.integrations.datadog.datadog_llm_obs.get_async_httpx_client'), \
         patch('asyncio.create_task'):
        logger = DataDogLLMObsLogger()
        
        # Test case 1: No latency metrics present
        standard_payload = create_standard_logging_payload_with_cache()
        metadata = logger._get_dd_llm_obs_payload_metadata(standard_payload)
        
        # Should not have latency fields if data is missing/zero
        assert "time_to_first_token_ms" not in metadata  # Will be 0, so not included
        assert "litellm_overhead_time_ms" not in metadata  # Not present in hidden_params
        assert "guardrail_overhead_time_ms" not in metadata  # No guardrail_information
        
        # Test case 2: Zero time to first token should not be included
        standard_payload = create_standard_logging_payload_with_cache()
        standard_payload["startTime"] = 1000.0
        standard_payload["completionStartTime"] = 1000.0  # Same time = 0 difference
        metadata = logger._get_dd_llm_obs_payload_metadata(standard_payload)
        assert "time_to_first_token_ms" not in metadata
        
        # Test case 3: Missing guardrail duration should not crash
        standard_payload = create_standard_logging_payload_with_cache()
        standard_payload["guardrail_information"] = StandardLoggingGuardrailInformation(
            guardrail_name="test",
            guardrail_status="success",
            # duration is missing
        )
        metadata = logger._get_dd_llm_obs_payload_metadata(standard_payload)
        assert "guardrail_overhead_time_ms" not in metadata


def test_guardrail_information_in_metadata(mock_env_vars):
    """Test that guardrail_information is included in metadata with input/output fields"""
    with patch('litellm.integrations.datadog.datadog_llm_obs.get_async_httpx_client'), \
         patch('asyncio.create_task'):
        logger = DataDogLLMObsLogger()
        
        # Create a standard payload with guardrail information
        standard_payload = create_standard_logging_payload_with_latency_metrics()
        
        kwargs = {
            "standard_logging_object": standard_payload,
            "litellm_params": {"metadata": {}}
        }
        
        start_time = datetime.now()
        end_time = datetime.now()
        
        # Create the payload and verify guardrail_information is in metadata
        payload = logger.create_llm_obs_payload(kwargs, start_time, end_time)
        metadata = payload["meta"]["metadata"]
        
        # Verify guardrail_information is present in metadata
        assert "guardrail_information" in metadata
        assert metadata["guardrail_information"] is not None
        
        # Verify the guardrail information structure
        guardrail_info = metadata["guardrail_information"]
        assert guardrail_info["guardrail_name"] == "test_guardrail"
        assert guardrail_info["guardrail_status"] == "success"
        assert guardrail_info["duration"] == 0.5
        
        # Verify input/output fields are present
        assert "guardrail_request" in guardrail_info
        assert "guardrail_response" in guardrail_info
        
        # Validate the input/output content
        assert guardrail_info["guardrail_request"]["input"] == "test input message"
        assert guardrail_info["guardrail_request"]["user_id"] == "test_user"
        assert guardrail_info["guardrail_response"]["output"] == "filtered output"
        assert guardrail_info["guardrail_response"]["flagged"] == False
        assert guardrail_info["guardrail_response"]["score"] == 0.1
