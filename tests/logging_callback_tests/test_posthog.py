import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import pytest

from litellm.integrations.posthog import PostHogLogger
from litellm.types.utils import StandardLoggingPayload
from typing import cast

# Set env vars for tests
os.environ["POSTHOG_API_KEY"] = "test_key"
os.environ["POSTHOG_API_URL"] = "https://app.posthog.com"


def create_standard_logging_payload() -> StandardLoggingPayload:
    # Use cast to bypass strict TypedDict requirements for tests
    return cast(StandardLoggingPayload, {
        "id": "test_id",
        "trace_id": "test_trace_id", 
        "call_type": "completion",
        "stream": False,
        "response_cost": 0.1,
        "status": "success",
        "custom_llm_provider": "openai",
        "total_tokens": 30,
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "startTime": 1234567890.0,
        "endTime": 1234567891.0,
        "completionStartTime": 1234567890.5,
        "response_time": 1.0,
        "model": "gpt-3.5-turbo",
        "model_id": "model-123",
        "api_base": "https://api.openai.com",
        "cache_hit": False,
        "saved_cache_cost": 0.0,
        "request_tags": [],
        "end_user": None,
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "response": {"choices": [{"message": {"content": "Hi there!"}}]},
        "error_str": None,
        "model_parameters": {"stream": True},
    })


@pytest.mark.asyncio
async def test_create_posthog_event_payload():
    posthog_logger = PostHogLogger()
    standard_payload = create_standard_logging_payload()
    kwargs = {"standard_logging_object": standard_payload}

    event_payload = posthog_logger.create_posthog_event_payload(kwargs)

    assert event_payload["event"] == "$ai_generation"
    assert event_payload["properties"]["$ai_model"] == "gpt-3.5-turbo"
    assert event_payload["properties"]["$ai_input_tokens"] == 20
    assert event_payload["properties"]["$ai_output_tokens"] == 10


@pytest.mark.asyncio
async def test_posthog_failure_logging():
    posthog_logger = PostHogLogger()
    standard_payload = create_standard_logging_payload()
    standard_payload["status"] = "failure"
    standard_payload["error_str"] = "Test error"

    kwargs = {"standard_logging_object": standard_payload}

    event_payload = posthog_logger.create_posthog_event_payload(kwargs)

    assert event_payload["properties"]["$ai_is_error"] is True
    assert event_payload["properties"]["$ai_error"] == "Test error"


@pytest.mark.asyncio
async def test_posthog_embedding_event():
    posthog_logger = PostHogLogger()
    standard_payload = create_standard_logging_payload()
    standard_payload["call_type"] = "embedding"

    kwargs = {"standard_logging_object": standard_payload}

    event_payload = posthog_logger.create_posthog_event_payload(kwargs)

    assert event_payload["event"] == "$ai_embedding"
    assert "$ai_output_tokens" not in event_payload["properties"]


@pytest.mark.asyncio
async def test_trace_id_fallback_from_standard_logging_object():
    """Test that trace_id is properly extracted from standard_logging_object"""
    posthog_logger = PostHogLogger()
    standard_payload = create_standard_logging_payload()
    standard_payload["trace_id"] = "test-trace-123"
    
    kwargs = {"standard_logging_object": standard_payload}
    
    event_payload = posthog_logger.create_posthog_event_payload(kwargs)
    
    assert event_payload["properties"]["$ai_trace_id"] == "test-trace-123"
    assert event_payload["properties"]["$ai_span_id"] == "test_id"  # from standard_payload["id"]


@pytest.mark.asyncio 
async def test_trace_id_uuid_fallback():
    """Test that UUID is generated when no trace_id is available"""
    posthog_logger = PostHogLogger()
    standard_payload = create_standard_logging_payload()
    # Remove trace_id to test fallback
    del standard_payload["trace_id"]
    del standard_payload["id"]
    
    kwargs = {"standard_logging_object": standard_payload}
    
    event_payload = posthog_logger.create_posthog_event_payload(kwargs)
    
    # Should have generated UUIDs
    assert len(event_payload["properties"]["$ai_trace_id"]) == 36  # UUID length
    assert len(event_payload["properties"]["$ai_span_id"]) == 36   # UUID length
    assert "-" in event_payload["properties"]["$ai_trace_id"]       # UUID format


@pytest.mark.asyncio
async def test_distinct_id_fallback_chain():
    """Test the distinct_id fallback priority chain"""
    posthog_logger = PostHogLogger()
    
    # Test 1: user_id from metadata (highest priority)
    standard_payload = create_standard_logging_payload()
    kwargs = {
        "standard_logging_object": standard_payload,
        "litellm_params": {"metadata": {"user_id": "metadata-user-123"}}
    }
    
    distinct_id = posthog_logger._get_distinct_id(standard_payload, kwargs)
    assert distinct_id == "metadata-user-123"
    
    # Test 2: trace_id from standard_logging_object (second priority)  
    kwargs = {"standard_logging_object": standard_payload}  # no metadata
    distinct_id = posthog_logger._get_distinct_id(standard_payload, kwargs)
    assert distinct_id == "test_trace_id"
    
    # Test 3: end_user from standard_logging_object (third priority)
    standard_payload_no_trace = create_standard_logging_payload()
    del standard_payload_no_trace["trace_id"]
    standard_payload_no_trace["end_user"] = "end-user-456"
    
    distinct_id = posthog_logger._get_distinct_id(standard_payload_no_trace, {})
    assert distinct_id == "end-user-456"
    
    # Test 4: UUID fallback (lowest priority)
    standard_payload_empty = create_standard_logging_payload()
    del standard_payload_empty["trace_id"]
    del standard_payload_empty["end_user"]
    
    distinct_id = posthog_logger._get_distinct_id(standard_payload_empty, {})
    assert len(distinct_id) == 36  # UUID length
    assert "-" in distinct_id       # UUID format


@pytest.mark.asyncio
async def test_missing_standard_logging_object():
    """Test error handling when standard_logging_object is missing"""
    posthog_logger = PostHogLogger()
    
    kwargs = {}  # Missing standard_logging_object
    
    with pytest.raises(ValueError, match="standard_logging_object not found in kwargs"):
        posthog_logger.create_posthog_event_payload(kwargs)


@pytest.mark.asyncio
async def test_custom_metadata_support():
    """Test that custom metadata fields are added directly to properties"""
    posthog_logger = PostHogLogger()
    standard_payload = create_standard_logging_payload()
    
    kwargs = {
        "standard_logging_object": standard_payload,
        "litellm_params": {
            "metadata": {
                "user_id": "user-123",  # should be used for distinct_id, not custom property
                "project_name": "test_project",  # should appear as project_name
                "environment": "staging",  # should appear as environment  
                "custom_field": "custom_value"  # should appear as custom_field
            }
        }
    }
    
    event_payload = posthog_logger.create_posthog_event_payload(kwargs)
    
    # Check that custom fields are added directly
    assert event_payload["properties"]["project_name"] == "test_project"
    assert event_payload["properties"]["environment"] == "staging"
    assert event_payload["properties"]["custom_field"] == "custom_value"
    
    # Check that user_id is used for distinct_id, not as custom property
    assert event_payload["distinct_id"] == "user-123"
    assert "user_id" not in event_payload["properties"]


@pytest.mark.asyncio
async def test_custom_metadata_filters_internal_fields():
    """Test that LiteLLM internal fields are filtered out from custom metadata"""
    posthog_logger = PostHogLogger()
    standard_payload = create_standard_logging_payload()
    
    kwargs = {
        "standard_logging_object": standard_payload,
        "litellm_params": {
            "metadata": {
                "custom_field": "should_appear",
                "endpoint": "/chat/completions",  # internal field - should be filtered
                "user_api_key_hash": "hash123",  # internal field - should be filtered
                "headers": {"content-type": "application/json"},  # internal field - should be filtered
                "model_info": {"id": "123"},  # internal field - should be filtered
            }
        }
    }
    
    event_payload = posthog_logger.create_posthog_event_payload(kwargs)
    
    # Check that custom field appears
    assert event_payload["properties"]["custom_field"] == "should_appear"
    
    # Check that internal fields are filtered out
    assert "endpoint" not in event_payload["properties"]
    assert "user_api_key_hash" not in event_payload["properties"]
    assert "headers" not in event_payload["properties"]
    assert "model_info" not in event_payload["properties"]


@pytest.mark.asyncio
async def test_custom_metadata_with_no_metadata():
    """Test that logger handles cases with no metadata gracefully"""
    posthog_logger = PostHogLogger()
    standard_payload = create_standard_logging_payload()
    
    # Test with no litellm_params
    kwargs = {"standard_logging_object": standard_payload}
    event_payload = posthog_logger.create_posthog_event_payload(kwargs)
    
    # Should not error and should have standard properties
    assert event_payload["event"] == "$ai_generation"
    assert event_payload["properties"]["$ai_model"] == "gpt-3.5-turbo"
    
    # Test with empty metadata
    kwargs = {
        "standard_logging_object": standard_payload,
        "litellm_params": {"metadata": {}}
    }
    event_payload = posthog_logger.create_posthog_event_payload(kwargs)
    
    # Should not error and should have standard properties
    assert event_payload["event"] == "$ai_generation"
    assert event_payload["properties"]["$ai_model"] == "gpt-3.5-turbo"
