import io
import os
import sys

from typing import Optional

sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import gzip
import json
import logging
import time
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import StandardLoggingPayload


class TestCustomLogger(CustomLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logged_standard_logging_payload: Optional[StandardLoggingPayload] = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        standard_logging_payload = kwargs.get("standard_logging_object", None)
        self.logged_standard_logging_payload = standard_logging_payload


@pytest.mark.asyncio
async def test_global_redaction_on():
    litellm.turn_off_message_logging = True
    test_custom_logger = TestCustomLogger()
    litellm.callbacks = [test_custom_logger]
    response = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hello",
    )

    await asyncio.sleep(1)
    standard_logging_payload = test_custom_logger.logged_standard_logging_payload
    assert standard_logging_payload is not None
    assert standard_logging_payload["response"] == {"text": "redacted-by-litellm"}
    assert standard_logging_payload["messages"][0]["content"] == "redacted-by-litellm"
    print(
        "logged standard logging payload",
        json.dumps(standard_logging_payload, indent=2),
    )


@pytest.mark.parametrize("turn_off_message_logging", [True, False])
@pytest.mark.asyncio
async def test_global_redaction_with_dynamic_params(turn_off_message_logging):
    litellm.turn_off_message_logging = True
    test_custom_logger = TestCustomLogger()
    litellm.callbacks = [test_custom_logger]
    response = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        turn_off_message_logging=turn_off_message_logging,
        mock_response="hello",
    )

    await asyncio.sleep(1)
    standard_logging_payload = test_custom_logger.logged_standard_logging_payload
    assert standard_logging_payload is not None
    print(
        "logged standard logging payload",
        json.dumps(standard_logging_payload, indent=2),
    )

    if turn_off_message_logging is True:
        assert standard_logging_payload["response"] == {"text": "redacted-by-litellm"}
        assert (
            standard_logging_payload["messages"][0]["content"] == "redacted-by-litellm"
        )
    else:
        assert (
            standard_logging_payload["response"]["choices"][0]["message"]["content"]
            == "hello"
        )
        assert standard_logging_payload["messages"][0]["content"] == "hi"


@pytest.mark.parametrize("turn_off_message_logging", [True, False])
@pytest.mark.asyncio
async def test_global_redaction_off_with_dynamic_params(turn_off_message_logging):
    litellm.turn_off_message_logging = False
    test_custom_logger = TestCustomLogger()
    litellm.callbacks = [test_custom_logger]
    response = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        turn_off_message_logging=turn_off_message_logging,
        mock_response="hello",
    )

    await asyncio.sleep(1)
    standard_logging_payload = test_custom_logger.logged_standard_logging_payload
    assert standard_logging_payload is not None
    print(
        "logged standard logging payload",
        json.dumps(standard_logging_payload, indent=2),
    )
    if turn_off_message_logging is True:
        assert standard_logging_payload["response"] == {"text": "redacted-by-litellm"}
        assert (
            standard_logging_payload["messages"][0]["content"] == "redacted-by-litellm"
        )
    else:
        assert (
            standard_logging_payload["response"]["choices"][0]["message"]["content"]
            == "hello"
        )
        assert standard_logging_payload["messages"][0]["content"] == "hi"


@pytest.mark.asyncio
async def test_redaction_responses_api():
    """Test redaction with ResponsesAPIResponse format"""
    litellm.turn_off_message_logging = True
    test_custom_logger = TestCustomLogger()
    litellm.callbacks = [test_custom_logger]
    
    # Mock a ResponsesAPIResponse-style response
    mock_response = {
        "output": [{"text": "This is a test response"}],
        "model": "gpt-3.5-turbo",
        "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10}
    }
    
    response = await litellm.aresponses(
        model="gpt-3.5-turbo",
        input="hi",
        mock_response=mock_response,
    )

    await asyncio.sleep(1)
    standard_logging_payload = test_custom_logger.logged_standard_logging_payload
    assert standard_logging_payload is not None
    
    # Verify redaction in ResponsesAPIResponse format
    assert standard_logging_payload["response"] == {"text": "redacted-by-litellm"}
    assert standard_logging_payload["messages"][0]["content"] == "redacted-by-litellm"
    print(
        "logged standard logging payload for ResponsesAPIResponse",
        json.dumps(standard_logging_payload, indent=2),
    )


@pytest.mark.asyncio
async def test_redaction_responses_api_stream():
    """Test redaction with ResponsesAPIResponse format"""
    litellm.turn_off_message_logging = True
    test_custom_logger = TestCustomLogger()
    litellm.callbacks = [test_custom_logger]
    
    # Mock a ResponsesAPIResponse-style response with streaming chunks
    mock_response = [
        {
            "output": [{"text": "This"}],
            "model": "gpt-3.5-turbo",
        },
        {
            "output": [{"text": " is"}],
            "model": "gpt-3.5-turbo",
        },
        {
            "output": [{"text": " a test response"}],
            "model": "gpt-3.5-turbo",
            "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10}
        }
    ]
    
    response = await litellm.aresponses(
        model="gpt-3.5-turbo",
        input="hi",
        mock_response=mock_response,
        stream=True,
    )

    # Consume the stream
    chunks = []
    async for chunk in response:
        chunks.append(chunk)
    
    await asyncio.sleep(1)
    standard_logging_payload = test_custom_logger.logged_standard_logging_payload
    assert standard_logging_payload is not None
    
    # Verify redaction in ResponsesAPIResponse format
    assert standard_logging_payload["response"] == {"text": "redacted-by-litellm"}
    assert standard_logging_payload["messages"][0]["content"] == "redacted-by-litellm"
    print(
        "logged standard logging payload for ResponsesAPIResponse stream",
        json.dumps(standard_logging_payload, indent=2),
    )


@pytest.mark.asyncio
async def test_redaction_with_coroutine_objects():
    """Test that redaction handles coroutine objects correctly without pickle errors"""
    from litellm.litellm_core_utils.redact_messages import perform_redaction
    
    # Test with a coroutine object (simulating streaming response)
    async def mock_async_generator():
        yield {"text": "test response"}
    
    coroutine = mock_async_generator()
    
    # This should not raise a pickle error
    result = perform_redaction({}, coroutine)
    assert result == {"text": "redacted-by-litellm"}
    
    # Test with an async function
    async def mock_async_function():
        return "test"
    
    async_func = mock_async_function()
    result = perform_redaction({}, async_func)
    assert result == {"text": "redacted-by-litellm"}
    
    # Test with an object that has __aiter__ method (async generator)
    class MockAsyncGenerator:
        def __aiter__(self):
            return self
        
        async def __anext__(self):
            raise StopAsyncIteration
    
    mock_gen = MockAsyncGenerator()
    result = perform_redaction({}, mock_gen)
    assert result == {"text": "redacted-by-litellm"}
    
    # Test with an object that has __anext__ method (async iterator)
    class MockAsyncIterator:
        def __anext__(self):
            raise StopAsyncIteration
    
    mock_iter = MockAsyncIterator()
    result = perform_redaction({}, mock_iter)
    assert result == {"text": "redacted-by-litellm"}


@pytest.mark.asyncio
async def test_redaction_with_streaming_response():
    """Test that redaction works correctly with streaming responses that return coroutines"""
    litellm.turn_off_message_logging = True
    test_custom_logger = TestCustomLogger()
    litellm.callbacks = [test_custom_logger]
    
    # This simulates the scenario where a streaming response returns a coroutine
    # that would normally cause the pickle error
    response = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        mock_response="hello",
    )
    
    # Consume the stream to trigger logging
    chunks = []
    async for chunk in response:
        chunks.append(chunk)
    
    await asyncio.sleep(1)
    standard_logging_payload = test_custom_logger.logged_standard_logging_payload
    assert standard_logging_payload is not None
    
    # Verify that redaction worked without pickle errors
    assert standard_logging_payload["response"] == {"text": "redacted-by-litellm"}
    assert standard_logging_payload["messages"][0]["content"] == "redacted-by-litellm"
    print(
        "logged standard logging payload for streaming with coroutine handling",
        json.dumps(standard_logging_payload, indent=2),
    )
