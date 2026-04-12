"""
Unit tests for VertexAIRealtimeConfig.

Validates:
- URL construction (regional and global)
- Auth headers (Bearer token + project header)
- Session setup message format
- Full text-in / text-out round-trip via RealTimeStreaming with a mocked
  WebSocket pair (no real network calls)
"""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
import websockets.exceptions  # registers websockets.exceptions on the websockets namespace

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.vertex_ai.realtime.transformation import VertexAIRealtimeConfig

# ---------------------------------------------------------------------------
# Config unit tests
# ---------------------------------------------------------------------------


def test_get_complete_url_regional():
    cfg = VertexAIRealtimeConfig(
        access_token="tok", project="my-proj", location="us-central1"
    )
    url = cfg.get_complete_url(api_base=None, model="gemini-2.0-flash-live-001")
    assert url == (
        "wss://us-central1-aiplatform.googleapis.com"
        "/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"
    )


def test_get_complete_url_global():
    cfg = VertexAIRealtimeConfig(
        access_token="tok", project="my-proj", location="global"
    )
    url = cfg.get_complete_url(api_base=None, model="gemini-2.0-flash-live-001")
    assert url == (
        "wss://aiplatform.googleapis.com"
        "/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"
    )


def test_get_complete_url_custom_api_base():
    cfg = VertexAIRealtimeConfig(
        access_token="tok", project="my-proj", location="us-central1"
    )
    url = cfg.get_complete_url(
        api_base="https://custom-gateway.example.com",
        model="gemini-2.0-flash-live-001",
    )
    assert url.startswith("wss://custom-gateway.example.com")
    assert "BidiGenerateContent" in url


def test_validate_environment_sets_bearer_and_project():
    cfg = VertexAIRealtimeConfig(
        access_token="mytoken", project="proj-123", location="us-central1"
    )
    headers = cfg.validate_environment(
        headers={}, model="gemini-2.0-flash-live-001", api_key=None
    )
    assert headers["Authorization"] == "Bearer mytoken"
    assert headers["x-goog-user-project"] == "proj-123"


def test_session_configuration_request_model_format():
    cfg = VertexAIRealtimeConfig(
        access_token="tok", project="my-proj", location="us-central1"
    )
    raw = cfg.session_configuration_request("gemini-2.0-flash-live-001")
    parsed = json.loads(raw)
    assert parsed["setup"]["model"] == (
        "projects/my-proj/locations/us-central1/publishers/google/models/gemini-2.0-flash-live-001"
    )


# ---------------------------------------------------------------------------
# Round-trip test: text-in / text-out via RealTimeStreaming
# ---------------------------------------------------------------------------

# Minimal Gemini BidiGenerateContent message sequence:
#   server → setupComplete
#   client → conversation.item.create  (OpenAI format, translated by config)
#   server → serverContent with modelTurn text delta
#   server → serverContent with generationComplete

SETUP_COMPLETE = json.dumps({"setupComplete": {}})

SERVER_TEXT_DELTA = json.dumps(
    {
        "serverContent": {
            "modelTurn": {
                "parts": [{"text": "Hello from Vertex AI!"}]
            }
        }
    }
)

# generationComplete fires RESPONSE_TEXT_DONE; turnComplete fires RESPONSE_DONE
# They must be separate messages (the transformer processes one top-level key per message).
SERVER_GENERATION_COMPLETE = json.dumps(
    {"serverContent": {"generationComplete": True}}
)

SERVER_TURN_COMPLETE = json.dumps(
    {"serverContent": {"turnComplete": True}}
)

# OpenAI-format text message the client sends
CLIENT_TEXT_MESSAGE = json.dumps(
    {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Say hello"}],
        },
    }
)


@pytest.mark.asyncio
async def test_vertex_realtime_text_in_text_out():
    """
    Simulate a full text-in / text-out session through RealTimeStreaming using
    VertexAIRealtimeConfig for message translation.  All I/O is mocked.
    """
    from litellm.litellm_core_utils.realtime_streaming import RealTimeStreaming

    cfg = VertexAIRealtimeConfig(
        access_token="fake-token",
        project="fake-project",
        location="us-central1",
    )

    # --- mock client WebSocket (FastAPI side) ---
    client_ws = MagicMock()
    client_ws.exceptions = MagicMock()
    client_ws.exceptions.ConnectionClosed = Exception

    sent_to_client: list[str] = []

    async def _client_send_text(data: str):
        sent_to_client.append(data)

    client_ws.send_text = AsyncMock(side_effect=_client_send_text)

    # Client sends one text message then raises to end the loop
    client_ws.receive_text = AsyncMock(
        side_effect=[CLIENT_TEXT_MESSAGE, Exception("client done")]
    )

    # --- mock backend WebSocket (Vertex AI side) ---
    backend_ws = MagicMock()

    upstream_messages = [
        SETUP_COMPLETE,
        SERVER_TEXT_DELTA,
        SERVER_GENERATION_COMPLETE,
        SERVER_TURN_COMPLETE,
    ]

    async def _backend_recv(decode=True):  # noqa: ARG001
        if not upstream_messages:
            # Signal normal connection close so the loop exits cleanly
            raise websockets.exceptions.ConnectionClosedOK(None, None)  # type: ignore[arg-type]
        return upstream_messages.pop(0)

    backend_ws.recv = AsyncMock(side_effect=_backend_recv)

    sent_to_backend: list[str] = []

    async def _backend_send(data: str):
        sent_to_backend.append(data)

    backend_ws.send = AsyncMock(side_effect=_backend_send)

    logging_obj = MagicMock()
    logging_obj.litellm_trace_id = "test-trace-id"
    logging_obj.pre_call = MagicMock()
    logging_obj.async_success_handler = AsyncMock()
    logging_obj.success_handler = MagicMock()

    streaming = RealTimeStreaming(
        websocket=client_ws,
        backend_ws=backend_ws,
        logging_obj=logging_obj,
        provider_config=cfg,
        model="gemini-2.0-flash-live-001",
    )

    # Run backend→client forwarding for the three queued messages, then stop.
    # We don't run client_ack_messages here to avoid the blocking receive loop.
    await streaming.backend_to_client_send_messages()

    # --- Assertions ---

    # session.created should have been forwarded to client
    session_created_msgs = [
        m for m in sent_to_client if '"session.created"' in m
    ]
    assert session_created_msgs, "Expected session.created to be sent to client"

    # At least one text delta should have been forwarded
    text_delta_msgs = [
        m for m in sent_to_client if '"response.text.delta"' in m
    ]
    assert text_delta_msgs, "Expected response.text.delta to be sent to client"

    # Verify the delta contains the model's text
    delta_obj = json.loads(text_delta_msgs[0])
    assert "Hello from Vertex AI!" in delta_obj.get("delta", "")

    # response.done should have been forwarded
    done_msgs = [m for m in sent_to_client if '"response.done"' in m]
    assert done_msgs, "Expected response.done to be sent to client"
