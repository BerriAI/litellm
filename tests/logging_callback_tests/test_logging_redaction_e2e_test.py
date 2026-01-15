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
    test_custom_logger = TestCustomLogger(turn_off_message_logging=True)
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
    # The response is now the full ResponsesAPIResponse object with transformed usage
    assert isinstance(standard_logging_payload["response"], dict)
    assert "usage" in standard_logging_payload["response"]
    # Check that usage has been transformed to chat completion format
    assert "prompt_tokens" in standard_logging_payload["response"]["usage"]
    assert "completion_tokens" in standard_logging_payload["response"]["usage"]
    
    assert standard_logging_payload["messages"][0]["content"] == "redacted-by-litellm"
    
    # Verify that output content is redacted
    assert "output" in standard_logging_payload["response"]
    output_items = standard_logging_payload["response"]["output"]
    for output_item in output_items:
        if "content" in output_item and isinstance(output_item["content"], list):
            for content_item in output_item["content"]:
                if "text" in content_item:
                    assert content_item["text"] == "redacted-by-litellm", f"Expected redacted text but got: {content_item['text']}"
    print(
        "logged standard logging payload for ResponsesAPIResponse",
        json.dumps(standard_logging_payload, indent=2),
    )


@pytest.mark.asyncio
async def test_redaction_responses_api_stream():
    """Test redaction with ResponsesAPIResponse format"""
    litellm.turn_off_message_logging = True
    test_custom_logger = TestCustomLogger(turn_off_message_logging=True)
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
    # The streaming response is in ModelResponse format (choices), not ResponsesAPIResponse format (output)
    assert isinstance(standard_logging_payload["response"], dict)
    assert standard_logging_payload["messages"][0]["content"] == "redacted-by-litellm"
    
    # Verify that response content is redacted (ModelResponse format)
    if "choices" in standard_logging_payload["response"]:
        # ModelResponse format
        assert standard_logging_payload["response"]["choices"][0]["message"]["content"] == "redacted-by-litellm"
    elif "output" in standard_logging_payload["response"]:
        # ResponsesAPIResponse format
        output_items = standard_logging_payload["response"]["output"]
        for output_item in output_items:
            if "content" in output_item and isinstance(output_item["content"], list):
                for content_item in output_item["content"]:
                    if "text" in content_item:
                        assert content_item["text"] == "redacted-by-litellm", f"Expected redacted text but got: {content_item['text']}"
    print(
        "logged standard logging payload for ResponsesAPIResponse stream",
        json.dumps(standard_logging_payload, indent=2),
    )


@pytest.mark.asyncio
async def test_redaction_responses_api_with_reasoning_summary():
    """Test that reasoning summary in ResponsesAPIResponse output is properly redacted"""
    from litellm.litellm_core_utils.redact_messages import perform_redaction
    
    # Create a simple mock object with output items that have reasoning summaries
    class MockResponsesAPIResponse:
        def __init__(self):
            self.output = [
                # Reasoning item with summary
                type('obj', (object,), {
                    'type': 'reasoning',
                    'id': 'rs_123',
                    'summary': [
                        type('obj', (object,), {
                            'text': 'This is a detailed reasoning summary that should be redacted',
                            'type': 'summary_text'
                        })()
                    ]
                })(),
                # Message item with content
                type('obj', (object,), {
                    'type': 'message',
                    'id': 'msg_123',
                    'content': [
                        type('obj', (object,), {
                            'text': 'This is the actual message content',
                            'type': 'output_text'
                        })()
                    ]
                })()
            ]
            self.reasoning = {"effort": "low", "summary": "auto"}
    
    # Mock as ResponsesAPIResponse so perform_redaction recognizes it
    mock_response = MockResponsesAPIResponse()
    mock_response.__class__.__name__ = 'ResponsesAPIResponse'
    
    # Patch isinstance to recognize our mock as ResponsesAPIResponse
    import litellm
    original_isinstance = isinstance
    def patched_isinstance(obj, cls):
        if cls == litellm.ResponsesAPIResponse and obj.__class__.__name__ == 'ResponsesAPIResponse':
            return True
        return original_isinstance(obj, cls)
    
    import builtins
    builtins.isinstance = patched_isinstance
    
    try:
        model_call_details = {
            "messages": [{"role": "user", "content": "test"}],
            "prompt": "test prompt",
            "input": "test input"
        }
        
        # Perform redaction
        redacted_result = perform_redaction(model_call_details, mock_response)
        
        # Verify reasoning summary text is redacted
        reasoning_item = redacted_result.output[0]
        assert reasoning_item.summary[0].text == "redacted-by-litellm", \
            "Reasoning summary text should be redacted"
        
        # Verify message content is also redacted
        message_item = redacted_result.output[1]
        assert message_item.content[0].text == "redacted-by-litellm", \
            "Message content text should be redacted"
        
        # Verify top-level reasoning field is removed
        assert redacted_result.reasoning is None, \
            "Top-level reasoning field should be None"
        
        # Verify input messages are redacted
        assert model_call_details["messages"][0]["content"] == "redacted-by-litellm", \
            "Input messages should be redacted"
        
        print("✓ Reasoning summary redaction test passed")
    finally:
        # Restore original isinstance
        builtins.isinstance = original_isinstance


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


@pytest.mark.asyncio
async def test_disable_redaction_header_responses_api():
    """
    Test that LiteLLM-Disable-Message-Redaction header works for Responses API.
    
    This test verifies the fix for the issue where the header wasn't respected
    because Responses API uses 'litellm_metadata' instead of 'metadata'.
    """
    litellm.turn_off_message_logging = True
    test_custom_logger = TestCustomLogger()
    litellm.callbacks = [test_custom_logger]
    
    # Mock a ResponsesAPIResponse-style response
    mock_response = {
        "output": [{"text": "This is a test response"}],
        "model": "gpt-3.5-turbo",
        "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10}
    }
    
    # Pass the header via litellm_metadata (as the proxy does for Responses API)
    response = await litellm.aresponses(
        model="gpt-3.5-turbo",
        input="hi",
        mock_response=mock_response,
        litellm_metadata={
            "headers": {
                "litellm-disable-message-redaction": "true"
            }
        }
    )

    await asyncio.sleep(1)
    standard_logging_payload = test_custom_logger.logged_standard_logging_payload
    assert standard_logging_payload is not None
    
    # Verify that messages are NOT redacted because the header was set
    print(
        "logged standard logging payload for ResponsesAPI with disable header",
        json.dumps(standard_logging_payload, indent=2, default=str),
    )
    
    # The content should NOT be redacted
    assert standard_logging_payload["response"] != {"text": "redacted-by-litellm"}
    assert standard_logging_payload["messages"][0]["content"] == "hi"


@pytest.mark.asyncio
async def test_redaction_with_metadata_completion_api():
    """
    Test redaction behavior with metadata field for Completion API.
    
    This test verifies that get_metadata_variable_name_from_kwargs properly
    selects the appropriate metadata field for header detection.
    """
    litellm.turn_off_message_logging = True
    test_custom_logger = TestCustomLogger()
    litellm.callbacks = [test_custom_logger]
    
    # When metadata is passed, the system uses get_metadata_variable_name_from_kwargs
    # to determine which field to check
    response = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hello",
        metadata={
            "headers": {
                "litellm-disable-message-redaction": "true"
            }
        }
    )

    await asyncio.sleep(1)
    standard_logging_payload = test_custom_logger.logged_standard_logging_payload
    assert standard_logging_payload is not None
    
    print(
        "logged standard logging payload for Completion API with metadata",
        json.dumps(standard_logging_payload, indent=2),
    )
    
    # Verify the helper function works correctly - with get_metadata_variable_name_from_kwargs,
    # the system checks the appropriate field for headers
    assert standard_logging_payload["response"] == {"text": "redacted-by-litellm"}
    assert standard_logging_payload["messages"][0]["content"] == "redacted-by-litellm"
