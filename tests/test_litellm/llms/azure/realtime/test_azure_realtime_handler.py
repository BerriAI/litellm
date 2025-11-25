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
async def test_construct_url_uses_legacy_realtime_by_default():
    """By default we should keep using `/openai/realtime` (beta behavior)."""

    from litellm.llms.azure.realtime.handler import AzureOpenAIRealtime

    handler = AzureOpenAIRealtime()
    api_base = "https://my-endpoint.openai.azure.com"
    api_version = "2024-10-01-preview"
    model = "gpt-4o-realtime-preview"

    url = handler._construct_url(api_base=api_base, model=model, api_version=api_version)

    assert url.startswith("wss://my-endpoint.openai.azure.com")
    assert "/openai/realtime" in url
    assert "/openai/v1/realtime" not in url


@pytest.mark.asyncio
async def test_construct_url_uses_v1_when_realtime_protocol_v1_or_ga():
    """Setting `realtime_protocol` to v1/GA should switch to `/openai/v1/realtime`."""

    from litellm.llms.azure.realtime.handler import AzureOpenAIRealtime

    api_base = "https://my-endpoint.openai.azure.com"
    api_version = "2024-10-01-preview"
    model = "gpt-4o-realtime-preview"

    # Helper to construct handler URL with a specific realtime_protocol.
    # We avoid mutating handler attributes directly since type checkers don't
    # know about `litellm_params` on this class. Instead, we patch the
    # `_get_realtime_protocol` helper which is what `_construct_url` uses.

    # v1 -> /openai/v1/realtime
    handler_v1 = AzureOpenAIRealtime()
    with patch.object(handler_v1, "_get_realtime_protocol", return_value="v1"):
        url_v1 = handler_v1._construct_url(api_base=api_base, model=model, api_version=api_version)
    assert "/openai/v1/realtime" in url_v1
    assert "/openai/realtime" not in url_v1

    # GA (case-insensitive) -> /openai/v1/realtime
    handler_ga = AzureOpenAIRealtime()
    with patch.object(handler_ga, "_get_realtime_protocol", return_value="v1"):
        url_ga = handler_ga._construct_url(api_base=api_base, model=model, api_version=api_version)
    assert "/openai/v1/realtime" in url_ga
    assert "/openai/realtime" not in url_ga

    # beta or any other value keeps legacy path
    handler_beta = AzureOpenAIRealtime()
    with patch.object(handler_beta, "_get_realtime_protocol", return_value="beta"):
        url_beta = handler_beta._construct_url(api_base=api_base, model=model, api_version=api_version)
    assert "/openai/realtime" in url_beta
    assert "/openai/v1/realtime" not in url_beta


