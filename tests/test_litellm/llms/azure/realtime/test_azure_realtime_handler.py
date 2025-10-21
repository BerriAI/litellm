import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
        # Default should be None (unlimited) to match OpenAI's official agents SDK
        # https://github.com/openai/openai-agents-python/blob/cf1b933660e44fd37b4350c41febab8221801409/src/agents/realtime/openai_realtime.py#L235

        mock_realtime_streaming.assert_called_once()
        mock_streaming_instance.bidirectional_forward.assert_awaited_once()

