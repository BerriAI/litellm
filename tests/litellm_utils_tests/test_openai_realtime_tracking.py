import asyncio
import json
import os

from dotenv import load_dotenv
from unittest.mock import AsyncMock

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

load_dotenv()
import os
import pytest


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
async def test_litellm_arealtime_openai_cost_tracking(model, mocker):
    import websockets
    import litellm

    client_websocket = AsyncMockClientWebsocket()
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
    user_api_key_auth = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN.value,
        api_key="sk-1234",
        user_id="1234",
    )
    user_api_key_dict = dict(
        user_api_key=user_api_key_auth.api_key,
        user_api_key_hash=user_api_key_auth.api_key,
        user_api_key_alias=user_api_key_auth.key_alias,
        user_api_key_team_id=user_api_key_auth.team_id,
        user_api_key_user_id=user_api_key_auth.user_id,
        user_api_key_org_id=user_api_key_auth.org_id,
        user_api_key_team_alias=user_api_key_auth.team_alias,
        user_api_key_end_user_id=user_api_key_auth.end_user_id,
        user_api_key_user_email=user_api_key_auth.user_email,
    )
    asyncio.create_task(
        litellm._arealtime(
            model=model, websocket=client_websocket, user_api_key_dict=user_api_key_dict
        )
    )
    await asyncio.sleep(1)
    mock_update_database_function.assert_called()
    mock_update_lagnfuse_function.assert_called()
