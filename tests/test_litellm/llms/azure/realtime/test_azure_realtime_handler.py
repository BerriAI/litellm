import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.llms.custom_httpx.http_handler import get_shared_realtime_ssl_context

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path


@pytest.mark.asyncio
async def test_async_realtime_uses_max_size_parameter():
    """
    Test that Azure's async_realtime method uses the REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
    constant for the max_size parameter to handle large base64 audio payloads.
    
    This verifies the fix for: https://github.com/BerriAI/litellm/issues/15747
    """
    from litellm.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
    from litellm.llms.azure.realtime.handler import AzureOpenAIRealtime

    handler = AzureOpenAIRealtime()
    api_base = "https://my-endpoint.openai.azure.com"
    api_key = "test-key"
    api_version = "2024-10-01-preview"
    model = "gpt-4o-realtime-preview"

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
         patch("litellm.llms.azure.realtime.handler.RealTimeStreaming") as mock_realtime_streaming:
        
        mock_streaming_instance = MagicMock()
        mock_realtime_streaming.return_value = mock_streaming_instance
        mock_streaming_instance.bidirectional_forward = AsyncMock()

        await handler.async_realtime(
            model=model,
            websocket=dummy_websocket,
            logging_obj=dummy_logging_obj,
            api_base=api_base,
            api_key=api_key,
            api_version=api_version,
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


@pytest.mark.asyncio
async def test_construct_url_default_beta_protocol():
    """
    Test that _construct_url uses /openai/realtime (beta) by default.
    This maintains backwards compatibility.
    """
    from litellm.llms.azure.realtime.handler import AzureOpenAIRealtime

    handler = AzureOpenAIRealtime()
    url = handler._construct_url(
        api_base="https://my-endpoint.openai.azure.com",
        model="gpt-4o-realtime-preview",
        api_version="2024-10-01-preview",
    )
    
    assert url.startswith("wss://my-endpoint.openai.azure.com/openai/realtime?")
    assert "/openai/realtime?" in url
    assert "/openai/v1/realtime" not in url
    assert "api-version=2024-10-01-preview" in url
    assert "deployment=gpt-4o-realtime-preview" in url


@pytest.mark.asyncio
async def test_construct_url_beta_protocol_explicit():
    """
    Test that realtime_protocol='beta' explicitly uses /openai/realtime.
    """
    from litellm.llms.azure.realtime.handler import AzureOpenAIRealtime

    handler = AzureOpenAIRealtime()
    url = handler._construct_url(
        api_base="https://my-endpoint.openai.azure.com",
        model="gpt-4o-realtime-preview",
        api_version="2024-10-01-preview",
        realtime_protocol="beta",
    )
    
    assert "/openai/realtime?" in url
    assert "/openai/v1/realtime" not in url


@pytest.mark.asyncio
async def test_construct_url_ga_protocol():
    """
    Test that realtime_protocol='GA' uses /openai/v1/realtime (GA path).
    GA path uses ?model= instead of ?api-version=&deployment= format.
    """
    from litellm.llms.azure.realtime.handler import AzureOpenAIRealtime

    handler = AzureOpenAIRealtime()
    url = handler._construct_url(
        api_base="https://my-endpoint.openai.azure.com",
        model="gpt-4o-realtime-preview",
        api_version="2024-10-01-preview",
        realtime_protocol="GA",
    )
    
    assert url.startswith("wss://my-endpoint.openai.azure.com/openai/v1/realtime?")
    assert "/openai/v1/realtime?" in url
    # Ensure it doesn't have both paths
    assert url.count("/realtime") == 1
    # GA path uses model= query param, not api-version and deployment
    assert "model=gpt-4o-realtime-preview" in url
    assert "api-version" not in url
    assert "deployment" not in url


@pytest.mark.asyncio
async def test_construct_url_v1_protocol():
    """
    Test that realtime_protocol='v1' also uses /openai/v1/realtime.
    """
    from litellm.llms.azure.realtime.handler import AzureOpenAIRealtime

    handler = AzureOpenAIRealtime()
    url = handler._construct_url(
        api_base="https://my-endpoint.openai.azure.com",
        model="gpt-4o-realtime-preview",
        api_version="2024-10-01-preview",
        realtime_protocol="v1",
    )
    
    assert "/openai/v1/realtime?" in url
    assert url.count("/realtime") == 1


@pytest.mark.asyncio
async def test_async_realtime_uses_ga_protocol_end_to_end():
    """
    Test that realtime_protocol='GA' flows through async_realtime to construct the correct URL.
    This is the end-to-end test ensuring the parameter is properly used.
    """
    from litellm.llms.azure.realtime.handler import AzureOpenAIRealtime

    handler = AzureOpenAIRealtime()
    api_base = "https://my-endpoint.openai.azure.com"
    api_key = "test-key"
    api_version = "2024-10-01-preview"
    model = "gpt-4o-realtime-preview"

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

    with patch("websockets.connect", return_value=DummyAsyncContextManager(mock_backend_ws)) as mock_ws_connect, \
         patch("litellm.llms.azure.realtime.handler.RealTimeStreaming") as mock_realtime_streaming:
        
        mock_streaming_instance = MagicMock()
        mock_realtime_streaming.return_value = mock_streaming_instance
        mock_streaming_instance.bidirectional_forward = AsyncMock()

        await handler.async_realtime(
            model=model,
            websocket=dummy_websocket,
            logging_obj=dummy_logging_obj,
            api_base=api_base,
            api_key=api_key,
            api_version=api_version,
            realtime_protocol="GA",
        )

        # Verify websockets.connect was called with GA URL
        mock_ws_connect.assert_called_once()
        called_url = mock_ws_connect.call_args[0][0]
        assert "/openai/v1/realtime" in called_url
        assert called_url.startswith("wss://")
        # GA path uses model= query param, not api-version and deployment
        assert "model=gpt-4o-realtime-preview" in called_url
        assert "api-version" not in called_url
        assert "deployment" not in called_url


@pytest.mark.asyncio
async def test_async_realtime_default_maintains_backwards_compatibility():
    """
    Test that not passing realtime_protocol maintains the original beta behavior.
    This ensures backwards compatibility for existing deployments.
    """
    from litellm.llms.azure.realtime.handler import AzureOpenAIRealtime

    handler = AzureOpenAIRealtime()
    api_base = "https://my-endpoint.openai.azure.com"
    api_key = "test-key"
    api_version = "2024-10-01-preview"
    model = "gpt-4o-realtime-preview"

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

    with patch("websockets.connect", return_value=DummyAsyncContextManager(mock_backend_ws)) as mock_ws_connect, \
         patch("litellm.llms.azure.realtime.handler.RealTimeStreaming") as mock_realtime_streaming:
        
        mock_streaming_instance = MagicMock()
        mock_realtime_streaming.return_value = mock_streaming_instance
        mock_streaming_instance.bidirectional_forward = AsyncMock()

        # Call without realtime_protocol parameter
        await handler.async_realtime(
            model=model,
            websocket=dummy_websocket,
            logging_obj=dummy_logging_obj,
            api_base=api_base,
            api_key=api_key,
            api_version=api_version,
        )

        # Verify it still uses the beta path
        called_url = mock_ws_connect.call_args[0][0]
        assert "/openai/realtime?" in called_url
        assert "/openai/v1/realtime" not in called_url


