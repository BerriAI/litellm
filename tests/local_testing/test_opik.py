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
