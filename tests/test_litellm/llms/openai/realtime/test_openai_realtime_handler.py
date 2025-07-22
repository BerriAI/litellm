import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

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
        api_base=api_base,     query_params = {
        "model": "gpt-4o-realtime-preview-2024-10-01",
    }
    )
    assert (
        url
        == f"wss://api.openai.com/v1/realtime"
    )


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
    # 'model' should be excluded from the query string
    assert url.startswith("wss://api.openai.com/v1/realtime?")
    assert "intent=chat" in url


import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.mark.asyncio
async def test_async_realtime_success():
    from litellm.llms.openai.realtime.handler import OpenAIRealtime
    from litellm.types.realtime import RealtimeQueryParams

    handler = OpenAIRealtime()
    api_base = "https://api.openai.com/v1"
    api_key = "test-key"
    model = "gpt-4o-realtime-preview-2024-10-01"
    query_params: RealtimeQueryParams = {"model": model, "intent": "chat"}

    dummy_websocket = MagicMock()
    dummy_logging_obj = MagicMock()

    # Patch websockets.connect and RealTimeStreaming
    with patch("litellm.llms.openai.realtime.handler.websockets.connect", new_callable=AsyncMock) as mock_ws_connect, \
         patch("litellm.llms.openai.realtime.handler.RealTimeStreaming") as mock_realtime_streaming:
        # Setup async context manager for websockets.connect
        mock_backend_ws = AsyncMock()
        mock_ws_connect.return_value.__aenter__.return_value = mock_backend_ws
        # Setup RealTimeStreaming mock
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

        mock_ws_connect.assert_awaited_once()
        mock_realtime_streaming.assert_called_once()
        mock_streaming_instance.bidirectional_forward.assert_awaited_once()
