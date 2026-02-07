import os
import sys
from typing import cast

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.integrations.posthog import PostHogLogger
from litellm.types.utils import StandardLoggingPayload

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
    posthog_logger = PostHogLogger()
    standard_payload = create_standard_logging_payload()
    standard_payload["trace_id"] = "test-trace-123"

    kwargs = {"standard_logging_object": standard_payload}

    event_payload = posthog_logger.create_posthog_event_payload(kwargs)

    assert event_payload["properties"]["$ai_trace_id"] == "test-trace-123"
    assert event_payload["properties"]["$ai_span_id"] == "test_id"


@pytest.mark.asyncio
async def test_trace_id_uuid_fallback():
    posthog_logger = PostHogLogger()
    standard_payload = create_standard_logging_payload()
    del standard_payload["trace_id"]
    del standard_payload["id"]

    kwargs = {"standard_logging_object": standard_payload}

    event_payload = posthog_logger.create_posthog_event_payload(kwargs)

    assert len(event_payload["properties"]["$ai_trace_id"]) == 36
    assert len(event_payload["properties"]["$ai_span_id"]) == 36
    assert "-" in event_payload["properties"]["$ai_trace_id"]


@pytest.mark.asyncio
async def test_distinct_id_fallback_chain():
    posthog_logger = PostHogLogger()

    # 1. metadata.user_id
    standard_payload = create_standard_logging_payload()
    kwargs = {
        "standard_logging_object": standard_payload,
        "litellm_params": {"metadata": {"user_id": "metadata-user-123"}},
    }

    distinct_id = posthog_logger._get_distinct_id(standard_payload, kwargs)
    assert distinct_id == "metadata-user-123"

    # 2. trace_id
    kwargs = {"standard_logging_object": standard_payload}
    distinct_id = posthog_logger._get_distinct_id(standard_payload, kwargs)
    assert distinct_id == "test_trace_id"

    # 3. end_user
    standard_payload_no_trace = create_standard_logging_payload()
    del standard_payload_no_trace["trace_id"]
    standard_payload_no_trace["end_user"] = "end-user-456"

    distinct_id = posthog_logger._get_distinct_id(standard_payload_no_trace, {})
    assert distinct_id == "end-user-456"

    # 4. UUID fallback
    standard_payload_empty = create_standard_logging_payload()
    del standard_payload_empty["trace_id"]
    del standard_payload_empty["end_user"]

    distinct_id = posthog_logger._get_distinct_id(standard_payload_empty, {})
    assert len(distinct_id) == 36
    assert "-" in distinct_id


@pytest.mark.asyncio
async def test_missing_standard_logging_object():
    posthog_logger = PostHogLogger()

    with pytest.raises(ValueError, match="standard_logging_object not found in kwargs"):
        posthog_logger.create_posthog_event_payload({})


def test_json_serialization_with_non_serializable_objects():
    """
    Sync-only test. No asyncio marker.
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
