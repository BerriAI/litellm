import asyncio
import json
import os

from dotenv import load_dotenv
from unittest.mock import AsyncMock

load_dotenv()
import os
import pytest
from fastapi.testclient import TestClient


websocket_request_body = {
    "type": "response.create",
    "response": {
        "modalities": ["text", "audio"],
        "instructions": "Hi",
    },
}

websocket_response_body = {
    "type": "response.done",
    "event_id": "event_B4mjnk5aUTJDtw9m8yaHE",
    "response": {
        "object": "realtime.response",
        "id": "resp_B4mjm7Y39HU6m67WlZVRm",
        "status": "completed",
        "status_details": None,
        "output": [
            {
                "id": "item_B4mjmFxnZ9mIgill7qKJJ",
                "object": "realtime.item",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "audio",
                        "transcript": "Hello! How can I assist you today?",
                    }
                ],
            }
        ],
        "conversation_id": "conv_B4mjmNtq17FqgJmPLyRqI",
        "modalities": ["audio", "text"],
        "voice": "alloy",
        "output_audio_format": "pcm16",
        "temperature": 0.8,
        "max_output_tokens": "inf",
        "usage": {
            "total_tokens": 92,
            "input_tokens": 5,
            "output_tokens": 87,
            "input_token_details": {
                "text_tokens": 5,
                "audio_tokens": 0,
                "cached_tokens": 0,
                "cached_tokens_details": {"text_tokens": 0, "audio_tokens": 0},
            },
            "output_token_details": {"text_tokens": 19, "audio_tokens": 68},
        },
        "metadata": None,
    },
}


@pytest.fixture
def client():
    from litellm.proxy.proxy_server import app, initialize

    filepath = os.path.dirname(os.path.abspath(__file__))
    config_fp = f"{filepath}/test_configs/test_openai_realtime.yaml"
    asyncio.run(initialize(config=config_fp))

    return TestClient(app)


class AsyncMockWebsocket(AsyncMock):
    async def send(self, *args, **kwargs):
        await asyncio.sleep(0.1)
        return {}

    async def recv(self, *args, **kwargs) -> bytes:
        await asyncio.sleep(0.1)
        return json.dumps(websocket_response_body).encode()


class AsyncMockClientWebsocket(AsyncMock):
    async def receive_text(self, *args, **kwargs):
        await asyncio.sleep(0.1)
        return json.dumps(websocket_request_body).encode()

    async def send_text(self, *args, **kwargs) -> bytes:
        return websocket_response_body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-4o-realtime-preview",
    ],
)
async def test_proxy_openai_realtime_cost_tracking(model, mocker, client):
    """
    Test OpenAI Realtime Tracking on Proxy server

    """
    import websockets
    import litellm

    backend_websocket = AsyncMockWebsocket()
    mock_connect = AsyncMock()
    mock_connect.__aenter__.return_value = backend_websocket
    mocker.patch(
        "litellm.llms.openai.realtime.handler.websockets.connect",
        return_value=mock_connect,
    )
    mock_update_database_function = mocker.patch(
        "litellm.litellm_core_utils.litellm_logging.Logging.async_websocket_success_handler"
    )
    mock_update_lagnfuse_function = mocker.patch(
        "litellm.litellm_core_utils.litellm_logging.Logging.websocket_success_handler"
    )

    litellm.set_verbose = True
    api_key = os.getenv("LITELLM_MASTER_KEY")

    with client.websocket_connect(
        f"/realtime?model={model}",
        headers={
            "Authorization": f"Bearer {api_key}",  # type: ignore
            "OpenAI-Beta": "realtime=v1",
        },
    ) as websocket:
        websocket.send_text(websocket_request_body)
        await asyncio.sleep(1)
    mock_update_database_function.assert_called()
    mock_update_lagnfuse_function.assert_called()
