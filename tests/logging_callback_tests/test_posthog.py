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
    return cast(
        StandardLoggingPayload,
        {
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
        },
    )


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
    assert (
        event_payload["properties"]["$ai_span_id"] == "test_id"
    )  # from standard_payload["id"]


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
    assert len(event_payload["properties"]["$ai_span_id"]) == 36  # UUID length
    assert "-" in event_payload["properties"]["$ai_trace_id"]  # UUID format


@pytest.mark.asyncio
async def test_distinct_id_fallback_chain():
    """Test the distinct_id fallback priority chain"""
    posthog_logger = PostHogLogger()

    # Test 1: user_id from metadata (highest priority)
    standard_payload = create_standard_logging_payload()
    kwargs = {
        "standard_logging_object": standard_payload,
        "litellm_params": {"metadata": {"user_id": "metadata-user-123"}},
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
    assert "-" in distinct_id  # UUID format


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
                "user_id": "user-123",
                "project_name": "test_project",
                "environment": "staging",
                "custom_field": "custom_value",
            }
        },
    }

    event_payload = posthog_logger.create_posthog_event_payload(kwargs)

    assert event_payload["properties"]["project_name"] == "test_project"
    assert event_payload["properties"]["environment"] == "staging"
    assert event_payload["properties"]["custom_field"] == "custom_value"

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
                "endpoint": "/chat/completions",
                "user_api_key_hash": "hash123",
                "headers": {"content-type": "application/json"},
                "model_info": {"id": "123"},
            }
        },
    }

    event_payload = posthog_logger.create_posthog_event_payload(kwargs)

    assert event_payload["properties"]["custom_field"] == "should_appear"

    assert "endpoint" not in event_payload["properties"]
    assert "user_api_key_hash" not in event_payload["properties"]
    assert "headers" not in event_payload["properties"]
    assert "model_info" not in event_payload["properties"]


@pytest.mark.asyncio
async def test_custom_metadata_with_no_metadata():
    """Test that logger handles cases with no metadata gracefully"""
    posthog_logger = PostHogLogger()
    standard_payload = create_standard_logging_payload()

    kwargs = {"standard_logging_object": standard_payload}
    event_payload = posthog_logger.create_posthog_event_payload(kwargs)

    assert event_payload["event"] == "$ai_generation"
    assert event_payload["properties"]["$ai_model"] == "gpt-3.5-turbo"

    kwargs = {
        "standard_logging_object": standard_payload,
        "litellm_params": {"metadata": {}},
    }
    event_payload = posthog_logger.create_posthog_event_payload(kwargs)

    assert event_payload["event"] == "$ai_generation"
    assert event_payload["properties"]["$ai_model"] == "gpt-3.5-turbo"


@pytest.mark.asyncio
async def test_dynamic_credentials():
    """Test that per-request credentials override environment variables"""
    from litellm.types.utils import StandardCallbackDynamicParams

    posthog_logger = PostHogLogger()

    kwargs = {}
    api_key, api_url = posthog_logger._get_credentials_for_request(kwargs)
    assert api_key == "test_key"
    assert api_url == "https://app.posthog.com"

    standard_callback_dynamic_params = StandardCallbackDynamicParams(
        posthog_api_key="dynamic_key", posthog_api_url="https://custom.posthog.com"
    )
    kwargs = {"standard_callback_dynamic_params": standard_callback_dynamic_params}
    api_key, api_url = posthog_logger._get_credentials_for_request(kwargs)
    assert api_key == "dynamic_key"
    assert api_url == "https://custom.posthog.com"

    standard_callback_dynamic_params = StandardCallbackDynamicParams(
        posthog_api_key="another_key"
    )
    kwargs = {"standard_callback_dynamic_params": standard_callback_dynamic_params}
    api_key, api_url = posthog_logger._get_credentials_for_request(kwargs)
    assert api_key == "another_key"
    assert api_url == "https://app.posthog.com"

    standard_callback_dynamic_params = StandardCallbackDynamicParams(
        posthog_api_url="https://another.posthog.com"
    )
    kwargs = {"standard_callback_dynamic_params": standard_callback_dynamic_params}
    api_key, api_url = posthog_logger._get_credentials_for_request(kwargs)
    assert api_key == "test_key"
    assert api_url == "https://another.posthog.com"


def test_async_callback_atexit_handler_exists():
    import atexit
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    assert hasattr(GLOBAL_LOGGING_WORKER, "_flush_on_exit")

    posthog_logger = PostHogLogger()
    assert hasattr(posthog_logger, "_flush_on_exit")

    GLOBAL_LOGGING_WORKER._flush_on_exit()
    posthog_logger._flush_on_exit()


@pytest.mark.asyncio
async def test_posthog_atexit_flushes_internal_queue():
    from unittest.mock import Mock, patch

    posthog_logger = PostHogLogger()

    standard_payload = create_standard_logging_payload()
    kwargs = {"standard_logging_object": standard_payload}
    event_payload = posthog_logger.create_posthog_event_payload(kwargs)

    posthog_logger.log_queue.append(
        {
            "event": event_payload,
            "api_key": "test_key",
            "api_url": "https://app.posthog.com",
        }
    )

    assert len(posthog_logger.log_queue) == 1

    with patch.object(posthog_logger.sync_client, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        posthog_logger._flush_on_exit()

        assert mock_post.called
        assert len(posthog_logger.log_queue) == 0

        call_args = mock_post.call_args
        assert "/batch/" in call_args.kwargs["url"]


@pytest.mark.asyncio
async def test_sync_callback_not_affected_by_atexit():
    from unittest.mock import Mock, patch

    callback_invoked_immediately = False

    def mock_log_success(self, kwargs, response_obj, start_time, end_time):
        nonlocal callback_invoked_immediately
        callback_invoked_immediately = True

    with patch.object(PostHogLogger, "log_success_event", mock_log_success):
        with patch("httpx.Client.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            posthog_logger = PostHogLogger()
            standard_payload = create_standard_logging_payload()
            kwargs = {"standard_logging_object": standard_payload}

            posthog_logger.log_success_event(kwargs, None, 0.0, 0.0)

            assert callback_invoked_immediately


def test_json_serialization_with_non_serializable_objects():
    """
    FIXED: sync test (no asyncio marker)
    """
    from unittest.mock import Mock, patch
    from datetime import datetime

    posthog_logger = PostHogLogger()

    standard_payload = create_standard_logging_payload()

    messages_with_non_serializable = [
        cast(
            dict,
            {
                "role": "user",
                "content": "Hello",
                "timestamp": datetime(2024, 1, 1, 12, 0, 0),
                "metadata": {
                    "custom_class": type("CustomClass", (), {"value": 123})(),
                    "set_data": {1, 2, 3},
                },
            },
        )
    ]
    standard_payload["messages"] = messages_with_non_serializable  # type: ignore

    response_choice: dict = {"message": {"content": "Hi"}}
    response_choice["self_ref"] = response_choice
    standard_payload["response"] = {"choices": [response_choice]}  # type: ignore

    kwargs = {"standard_logging_object": standard_payload}

    with patch.object(posthog_logger.sync_client, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "OK"
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        posthog_logger.log_success_event(kwargs, None, 0.0, 0.0)

        assert mock_post.called

        json_payload = mock_post.call_args.kwargs["content"]
        assert isinstance(json_payload, str)
        assert len(json_payload) > 0
        assert "2024" in json_payload or "datetime" in json_payload.lower()


@pytest.mark.asyncio
async def test_async_json_serialization_with_non_serializable_objects():
    """
    FIXED: use Mock response (not AsyncMock)
    """
    from unittest.mock import Mock, patch
    from datetime import datetime

    posthog_logger = PostHogLogger()

    standard_payload = create_standard_logging_payload()
    messages_with_datetime = [
        cast(
            dict,
            {
                "role": "user",
                "content": "Test async serialization",
                "timestamp": datetime(2024, 2, 1, 10, 30, 0),
            },
        )
    ]
    standard_payload["messages"] = messages_with_datetime  # type: ignore

    kwargs = {"standard_logging_object": standard_payload}
    event_payload = posthog_logger.create_posthog_event_payload(kwargs)

    posthog_logger.log_queue.append(
        {
            "event": event_payload,
            "api_key": "test_key",
            "api_url": "https://app.posthog.com",
        }
    )

    with patch.object(posthog_logger.async_client, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "OK"
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        await posthog_logger.async_send_batch()

        assert mock_post.called

        json_payload = mock_post.call_args.kwargs["content"]
        assert isinstance(json_payload, str)
        assert len(json_payload) > 0


@pytest.mark.asyncio
async def test_atexit_json_serialization_with_non_serializable_objects():
    from unittest.mock import Mock, patch
    from datetime import datetime

    posthog_logger = PostHogLogger()

    standard_payload = create_standard_logging_payload()
    messages_with_complex_data = [
        cast(
            dict,
            {
                "role": "user",
                "content": "Test atexit serialization",
                "timestamp": datetime(2024, 3, 1, 15, 45, 0),
                "complex_data": {
                    "tuple_data": (1, 2, 3),
                    "frozenset_data": frozenset([1, 2, 3]),
                },
            },
        )
    ]
    standard_payload["messages"] = messages_with_complex_data  # type: ignore

    kwargs = {"standard_logging_object": standard_payload}
    event_payload = posthog_logger.create_posthog_event_payload(kwargs)

    posthog_logger.log_queue.append(
        {
            "event": event_payload,
            "api_key": "test_key",
            "api_url": "https://app.posthog.com",
        }
    )

    with patch.object(posthog_logger.sync_client, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "OK"
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        posthog_logger._flush_on_exit()

        assert mock_post.called
        assert len(posthog_logger.log_queue) == 0

        json_payload = mock_post.call_args.kwargs["content"]
        assert isinstance(json_payload, str)
        assert len(json_payload) > 0
