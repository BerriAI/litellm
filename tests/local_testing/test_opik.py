import io
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import logging

import pytest

import litellm
from litellm._logging import verbose_logger
from unittest.mock import AsyncMock, Mock

verbose_logger.setLevel(logging.DEBUG)

litellm.set_verbose = True
import time

@pytest.mark.asyncio
async def test_opik_logging_http_request():
    """
    - Test that HTTP requests are made to Opik
    - Traces and spans are batched correctly
    """
    try:
        from litellm.integrations.opik.opik import OpikLogger

        os.environ["OPIK_URL_OVERRIDE"] = "https://fake.comet.com/opik/api"
        os.environ["OPIK_API_KEY"] = "anything"
        os.environ["OPIK_WORKSPACE"] = "anything"

        # Initialize OpikLogger
        test_opik_logger = OpikLogger()

        litellm.callbacks = [test_opik_logger]
        test_opik_logger.batch_size = 12
        litellm.set_verbose = True

        # Create a mock for the async_client's post method
        mock_post = AsyncMock()
        mock_post.return_value.status_code = 202
        mock_post.return_value.text = "Accepted"
        test_opik_logger.async_httpx_client.post = mock_post

        # Make multiple calls to ensure we don't hit the batch size
        for _ in range(5):
            response = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test message"}],
                max_tokens=10,
                temperature=0.2,
                mock_response="This is a mock response",
            )
        await asyncio.sleep(1)

        # Check batching of events and that the queue contains 5 trace events and 5 span events
        assert mock_post.called == False, "HTTP request was made but events should have been batched"
        assert len(test_opik_logger.log_queue) == 10

        # Now make calls to exceed the batch size
        for _ in range(3):
            response = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test message"}],
                max_tokens=10,
                temperature=0.2,
                mock_response="This is a mock response",
            )
        
        # Wait a short time for any asynchronous operations to complete
        await asyncio.sleep(1)

        # Check that the queue was flushed after exceeding batch size
        assert len(test_opik_logger.log_queue) < test_opik_logger.batch_size

        # Check that the data has been sent when it goes above the flush interval
        await asyncio.sleep(test_opik_logger.flush_interval)
        assert len(test_opik_logger.log_queue) == 0

        # Clean up
        for cb in litellm.callbacks:
            if isinstance(cb, OpikLogger):
                await cb.async_httpx_client.client.aclose()

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

def test_sync_opik_logging_http_request():
    """
    - Test that HTTP requests are made to Opik
    - Traces and spans are batched correctly
    """
    try:
        from litellm.integrations.opik.opik import OpikLogger

        os.environ["OPIK_URL_OVERRIDE"] = "https://fake.comet.com/opik/api"
        os.environ["OPIK_API_KEY"] = "anything"
        os.environ["OPIK_WORKSPACE"] = "anything"

        # Initialize OpikLogger
        test_opik_logger = OpikLogger()

        litellm.callbacks = [test_opik_logger]
        litellm.set_verbose = True

        # Create a mock for the clients's post method
        mock_post = Mock()
        mock_post.return_value.status_code = 204
        mock_post.return_value.text = "Accepted"
        test_opik_logger.sync_httpx_client.post = mock_post

        # Make multiple calls to ensure we don't hit the batch size
        for _ in range(5):
            response = litellm.completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test message"}],
                max_tokens=10,
                temperature=0.2,
                mock_response="This is a mock response",
            )

        # Need to wait for a short amount of time as the log_success callback is called in a different thread. One or two seconds is often not enough.
        time.sleep(3)

        # Check that 5 spans and 5 traces were sent
        assert mock_post.call_count == 10, f"Expected 10 HTTP requests, but got {mock_post.call_count}"
        
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

@pytest.mark.asyncio
@pytest.mark.skip(reason="local-only test, to test if everything works fine.")
async def test_opik_logging():
    try:
        from litellm.integrations.opik.opik import OpikLogger
        
        # Initialize OpikLogger
        test_opik_logger = OpikLogger()
        litellm.callbacks = [test_opik_logger]
        litellm.set_verbose = True

        # Log a chat completion call
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "What LLM are you ?"}],
            max_tokens=10,
            temperature=0.2,
            metadata={"opik": {"custom_field": "custom_value"}}
        )
        print("Non-streaming response:", response)
        
        # Log a streaming completion call
        stream_response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Stream = True - What llm are you ?"}],
            max_tokens=10,
            temperature=0.2,
            stream=True,
            metadata={"opik": {"custom_field": "custom_value"}}
        )
        print("Streaming response:")
        async for chunk in stream_response:
            print(chunk.choices[0].delta.content, end='', flush=True)
        print()  # New line after streaming response

        await asyncio.sleep(2)

        assert len(test_opik_logger.log_queue) == 4
        
        await asyncio.sleep(test_opik_logger.flush_interval + 1)
        assert len(test_opik_logger.log_queue) == 0
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_opik_attach_to_existing_trace():
    """
    Test attaching spans to existing trace (regression fix for PR #14888)
    
    - When trace_id is provided via current_span_data, only create a span
    - Do NOT create a new trace (this was the bug)
    - Verify span has correct trace_id and parent_span_id
    """
    try:
        from litellm.integrations.opik.opik import OpikLogger

        os.environ["OPIK_URL_OVERRIDE"] = "https://fake.comet.com/opik/api"
        os.environ["OPIK_API_KEY"] = "anything"
        os.environ["OPIK_WORKSPACE"] = "anything"

        # Initialize OpikLogger
        test_opik_logger = OpikLogger()
        litellm.callbacks = [test_opik_logger]
        litellm.set_verbose = True

        # Create a mock for the sync client's post method
        mock_post = Mock()
        mock_post.return_value.status_code = 204
        mock_post.return_value.text = "Accepted"
        test_opik_logger.sync_httpx_client.post = mock_post

        # Simulate an existing trace and parent span
        existing_trace_id = "existing-trace-12345"
        existing_parent_span_id = "existing-span-67890"

        # Make a completion call with existing trace_id
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test message"}],
            max_tokens=10,
            temperature=0.2,
            mock_response="This is a mock response",
            metadata={
                "opik": {
                    "current_span_data": {
                        "trace_id": existing_trace_id,
                        "id": existing_parent_span_id
                    },
                    "tags": ["test-attach-span"]
                }
            }
        )

        # Need to wait for a short amount of time as the log_success callback is called in a different thread
        time.sleep(3)

        # Check the calls made to the mock
        calls_made = mock_post.call_args_list
        trace_calls = [call for call in calls_made if "/traces/batch" in str(call)]
        span_calls = [call for call in calls_made if "/spans/batch" in str(call)]

        # With the fix, when trace_id is provided, we should NOT create a new trace
        assert len(trace_calls) == 0, f"Expected 0 trace calls when attaching to existing trace, but got {len(trace_calls)}"
        assert len(span_calls) == 1, f"Expected exactly 1 span call, but got {len(span_calls)}"
        
        # Verify span has correct trace_id and parent_span_id
        span_payload = span_calls[0][1]['json']['spans'][0]
        assert span_payload['trace_id'] == existing_trace_id, f"Expected trace_id to be {existing_trace_id}, but got {span_payload['trace_id']}"
        assert span_payload['parent_span_id'] == existing_parent_span_id, f"Expected parent_span_id to be {existing_parent_span_id}, but got {span_payload['parent_span_id']}"
        assert "test-attach-span" in span_payload['tags'], f"Expected 'test-attach-span' tag in {span_payload['tags']}"
        
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_opik_create_new_trace():
    """
    Test normal trace creation when no trace_id is provided
    
    - When NO trace_id is provided, create both a new trace and a new span
    - Verify the span references the created trace
    - Verify tags are included in both trace and span
    """
    try:
        from litellm.integrations.opik.opik import OpikLogger

        os.environ["OPIK_URL_OVERRIDE"] = "https://fake.comet.com/opik/api"
        os.environ["OPIK_API_KEY"] = "anything"
        os.environ["OPIK_WORKSPACE"] = "anything"

        # Initialize OpikLogger
        test_opik_logger = OpikLogger()
        litellm.callbacks = [test_opik_logger]
        litellm.set_verbose = True

        # Create a mock for the sync client's post method
        mock_post = Mock()
        mock_post.return_value.status_code = 204
        mock_post.return_value.text = "Accepted"
        test_opik_logger.sync_httpx_client.post = mock_post

        # Make a completion call WITHOUT providing trace_id
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test message"}],
            max_tokens=10,
            temperature=0.2,
            mock_response="This is a mock response",
            metadata={
                "opik": {
                    "tags": ["test-new-trace"]
                }
            }
        )

        # Need to wait for a short amount of time as the log_success callback is called in a different thread
        time.sleep(3)

        # Check the calls made to the mock
        calls_made = mock_post.call_args_list
        trace_calls = [call for call in calls_made if "/traces/batch" in str(call)]
        span_calls = [call for call in calls_made if "/spans/batch" in str(call)]

        # Without trace_id provided, we should create both a new trace and a new span
        assert len(trace_calls) == 1, f"Expected exactly 1 trace call, but got {len(trace_calls)}"
        assert len(span_calls) == 1, f"Expected exactly 1 span call, but got {len(span_calls)}"
        
        # Verify the span references the created trace
        trace_payload = trace_calls[0][1]['json']['traces'][0]
        span_payload = span_calls[0][1]['json']['spans'][0]
        assert span_payload['trace_id'] == trace_payload['id'], "Span should reference the created trace"
        
        # Verify tags are included in both trace and span
        assert "test-new-trace" in trace_payload['tags'], f"Expected 'test-new-trace' tag in trace tags"
        assert "test-new-trace" in span_payload['tags'], f"Expected 'test-new-trace' tag in span tags"
        
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
