import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.llms.custom_httpx.http_handler import get_shared_realtime_ssl_context

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path


@pytest.mark.parametrize(
    "api_base", ["https://api.openai.com/v1", "https://api.openai.com"]
)
def test_openai_realtime_handler_url_construction(api_base):
    from litellm.llms.openai.realtime.handler import OpenAIRealtime

    handler = OpenAIRealtime()
    url = handler._construct_url(
        api_base=api_base, 
        query_params={
            "model": "gpt-4o-realtime-preview-2024-10-01",
        }
    )
    # Model parameter should be included in the URL
    assert url.startswith("wss://api.openai.com/v1/realtime?")
    assert "model=gpt-4o-realtime-preview-2024-10-01" in url


def test_openai_realtime_handler_url_with_extra_params():
    from litellm.llms.openai.realtime.handler import OpenAIRealtime
    from litellm.types.realtime import RealtimeQueryParams

    handler = OpenAIRealtime()
    api_base = "https://api.openai.com/v1"
    query_params: RealtimeQueryParams = {
        "model": "gpt-4o-realtime-preview-2024-10-01",
        "intent": "chat"
    }
    url = handler._construct_url(api_base=api_base, query_params=query_params)
    # Both 'model' and other params should be included in the query string
    assert url.startswith("wss://api.openai.com/v1/realtime?")
    assert "model=gpt-4o-realtime-preview-2024-10-01" in url
    assert "intent=chat" in url


def test_openai_realtime_handler_model_parameter_inclusion():
    """
    Test that the model parameter is properly included in the WebSocket URL
    to prevent 'missing_model' errors from OpenAI.
    
    This test specifically verifies the fix for the issue where model parameter
    was being excluded from the query string, causing OpenAI to return
    invalid_request_error.missing_model errors.
    """
    from litellm.llms.openai.realtime.handler import OpenAIRealtime
    from litellm.types.realtime import RealtimeQueryParams

    handler = OpenAIRealtime()
    api_base = "https://api.openai.com/"
    
    # Test with just model parameter
    query_params_model_only: RealtimeQueryParams = {
        "model": "gpt-4o-mini-realtime-preview"
    }
    url = handler._construct_url(api_base=api_base, query_params=query_params_model_only)
    
    # Verify the URL structure
    assert url.startswith("wss://api.openai.com/v1/realtime?")
    assert "model=gpt-4o-mini-realtime-preview" in url
    
    # Test with model + additional parameters
    query_params_with_extras: RealtimeQueryParams = {
        "model": "gpt-4o-mini-realtime-preview",
        "intent": "chat"
    }
    url_with_extras = handler._construct_url(api_base=api_base, query_params=query_params_with_extras)
    
    # Verify both parameters are included
    assert url_with_extras.startswith("wss://api.openai.com/v1/realtime?")
    assert "model=gpt-4o-mini-realtime-preview" in url_with_extras
    assert "intent=chat" in url_with_extras
    
    # Verify the URL is properly formatted for OpenAI
    # Should match the pattern: wss://api.openai.com/v1/realtime?model=MODEL_NAME
    expected_pattern = "wss://api.openai.com/v1/realtime?model="
    assert expected_pattern in url
    assert expected_pattern in url_with_extras


import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_async_realtime_success():
    from litellm.llms.openai.realtime.handler import OpenAIRealtime
    from litellm.types.realtime import RealtimeQueryParams

    handler = OpenAIRealtime()
    api_base = "https://api.openai.com/v1"
    api_key = "test-key"
    model = "gpt-4o-realtime-preview-2024-10-01"
    query_params: RealtimeQueryParams = {"model": model, "intent": "chat"}

    dummy_websocket = AsyncMock()
    dummy_logging_obj = MagicMock()
    mock_backend_ws = AsyncMock()

    class DummyAsyncContextManager:
        def __init__(self, value):
            self.value = value
        async def __aenter__(self):
            return self.value
        async def __aexit__(self, exc_type, exc, tb):
            return None

    shared_context = get_shared_realtime_ssl_context()
    with patch("websockets.connect", return_value=DummyAsyncContextManager(mock_backend_ws)) as mock_ws_connect, \
         patch("litellm.llms.openai.realtime.handler.RealTimeStreaming") as mock_realtime_streaming:
        mock_streaming_instance = MagicMock()
        mock_realtime_streaming.return_value = mock_streaming_instance
        mock_streaming_instance.bidirectional_forward = AsyncMock()

        await handler.async_realtime(
            model=model,
            websocket=dummy_websocket,
            logging_obj=dummy_logging_obj,
            api_base=api_base,
            api_key=api_key,
            query_params=query_params,
        )

        mock_realtime_streaming.assert_called_once()
        mock_streaming_instance.bidirectional_forward.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_realtime_url_contains_model():
    """
    Test that the async_realtime method properly constructs a URL with the model parameter
    when connecting to OpenAI, preventing 'missing_model' errors.
    """
    from litellm.llms.openai.realtime.handler import OpenAIRealtime
    from litellm.types.realtime import RealtimeQueryParams

    handler = OpenAIRealtime()
    api_base = "https://api.openai.com/"
    api_key = "test-key"
    model = "gpt-4o-mini-realtime-preview"
    query_params: RealtimeQueryParams = {"model": model}

    dummy_websocket = AsyncMock()
    dummy_logging_obj = MagicMock()
    mock_backend_ws = AsyncMock()

    class DummyAsyncContextManager:
        def __init__(self, value):
            self.value = value
        async def __aenter__(self):
            return self.value
        async def __aexit__(self, exc_type, exc, tb):
            return None

    shared_context = get_shared_realtime_ssl_context()
    with patch("websockets.connect", return_value=DummyAsyncContextManager(mock_backend_ws)) as mock_ws_connect, \
         patch("litellm.llms.openai.realtime.handler.RealTimeStreaming") as mock_realtime_streaming:
        
        mock_streaming_instance = MagicMock()
        mock_realtime_streaming.return_value = mock_streaming_instance
        mock_streaming_instance.bidirectional_forward = AsyncMock()

        await handler.async_realtime(
            model=model,
            websocket=dummy_websocket,
            logging_obj=dummy_logging_obj,
            api_base=api_base,
            api_key=api_key,
            query_params=query_params,
        )

        # Verify websockets.connect was called with the correct URL
        mock_ws_connect.assert_called_once()
        called_url = mock_ws_connect.call_args[0][0]
        
        # Verify the URL contains the model parameter
        assert called_url.startswith("wss://api.openai.com/v1/realtime?")
        assert f"model={model}" in called_url
        
        # Verify proper headers were set
        called_kwargs = mock_ws_connect.call_args[1]
        assert "extra_headers" in called_kwargs
        extra_headers = called_kwargs["extra_headers"]
        assert extra_headers["Authorization"] == f"Bearer {api_key}"
        assert extra_headers["OpenAI-Beta"] == "realtime=v1"
        assert called_kwargs["ssl"] is shared_context
        
        mock_realtime_streaming.assert_called_once()
        mock_streaming_instance.bidirectional_forward.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_realtime_uses_max_size_parameter():
    """
    Test that the async_realtime method uses the REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
    constant for the max_size parameter to handle large base64 audio payloads.
    
    This verifies the fix for: https://github.com/BerriAI/litellm/issues/15747
    """
    from litellm.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
    from litellm.llms.openai.realtime.handler import OpenAIRealtime
    from litellm.types.realtime import RealtimeQueryParams

    handler = OpenAIRealtime()
    api_base = "https://api.openai.com/"
    api_key = "test-key"
    model = "gpt-4o-realtime-preview"
    query_params: RealtimeQueryParams = {"model": model}

    dummy_websocket = AsyncMock()
    dummy_logging_obj = MagicMock()
    mock_backend_ws = AsyncMock()

    class DummyAsyncContextManager:
        def __init__(self, value):
            self.value = value
        async def __aenter__(self):
            return self.value
        async def __aexit__(self, exc_type, exc, tb):
            return None

    shared_context = get_shared_realtime_ssl_context()
    with patch("websockets.connect", return_value=DummyAsyncContextManager(mock_backend_ws)) as mock_ws_connect, \
         patch("litellm.llms.openai.realtime.handler.RealTimeStreaming") as mock_realtime_streaming:
        
        mock_streaming_instance = MagicMock()
        mock_realtime_streaming.return_value = mock_streaming_instance
        mock_streaming_instance.bidirectional_forward = AsyncMock()

        await handler.async_realtime(
            model=model,
            websocket=dummy_websocket,
            logging_obj=dummy_logging_obj,
            api_base=api_base,
            api_key=api_key,
            query_params=query_params,
        )

        # Verify websockets.connect was called with the max_size parameter
        mock_ws_connect.assert_called_once()
        called_kwargs = mock_ws_connect.call_args[1]
        
        # Verify max_size is set (default None for unlimited, matching OpenAI's SDK)
        assert "max_size" in called_kwargs
        assert called_kwargs["max_size"] is None
        assert called_kwargs["ssl"] is shared_context
        # Default should be None (unlimited) to match OpenAI's official agents SDK
        # https://github.com/openai/openai-agents-python/blob/cf1b933660e44fd37b4350c41febab8221801409/src/agents/realtime/openai_realtime.py#L235

        mock_realtime_streaming.assert_called_once()
        mock_streaming_instance.bidirectional_forward.assert_awaited_once()
