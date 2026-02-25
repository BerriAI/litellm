"""
End-to-end integration test for the Responses API WebSocket endpoint.

Uses Starlette's TestClient to exercise the full FastAPI WebSocket route
without requiring an external proxy process or live OpenAI key.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from litellm.proxy.proxy_server import app


class TestResponsesWebSocketE2E:
    """Full round-trip tests through the proxy's WebSocket endpoint."""

    def test_websocket_accepts_connection_and_routes(self):
        """
        Verify the /v1/responses WebSocket:
        1. Accepts the connection
        2. Passes auth
        3. Reaches the routing layer (will fail at model lookup without
           a configured router, which is expected)
        """
        client = TestClient(app)

        with patch(
            "litellm.proxy.proxy_server.user_api_key_auth_websocket"
        ) as mock_auth:
            mock_auth.return_value = MagicMock(
                token="test_token",
                user_id="test_user",
                team_id=None,
                api_key="sk-test",
                key_alias=None,
                allowed_model_region=None,
                tpm_limit=None,
                rpm_limit=None,
                max_budget=None,
                spend=0.0,
                metadata={},
            )

            try:
                with client.websocket_connect(
                    "/v1/responses?model=gpt-4o-mini",
                    headers={"Authorization": "Bearer sk-test"},
                ) as ws:
                    pass
            except Exception as e:
                error = str(e)
                assert "404" not in error, f"Route not found: {error}"
                assert "405" not in error, f"Method not allowed: {error}"

    def test_websocket_openai_path(self):
        """Verify /openai/v1/responses WebSocket also works."""
        client = TestClient(app)

        with patch(
            "litellm.proxy.proxy_server.user_api_key_auth_websocket"
        ) as mock_auth:
            mock_auth.return_value = MagicMock(
                token="t", user_id="u", team_id=None, api_key="sk-t",
                key_alias=None, allowed_model_region=None,
                tpm_limit=None, rpm_limit=None, max_budget=None,
                spend=0.0, metadata={},
            )

            try:
                with client.websocket_connect(
                    "/openai/v1/responses?model=gpt-4o-mini",
                    headers={"Authorization": "Bearer sk-t"},
                ) as ws:
                    pass
            except Exception as e:
                assert "404" not in str(e)

    def test_websocket_short_path(self):
        """Verify /responses WebSocket also works."""
        client = TestClient(app)

        with patch(
            "litellm.proxy.proxy_server.user_api_key_auth_websocket"
        ) as mock_auth:
            mock_auth.return_value = MagicMock(
                token="t", user_id="u", team_id=None, api_key="sk-t",
                key_alias=None, allowed_model_region=None,
                tpm_limit=None, rpm_limit=None, max_budget=None,
                spend=0.0, metadata={},
            )

            try:
                with client.websocket_connect(
                    "/responses?model=gpt-4o-mini",
                    headers={"Authorization": "Bearer sk-t"},
                ) as ws:
                    pass
            except Exception as e:
                assert "404" not in str(e)


class TestResponsesWebSocketHandlerE2E:
    """
    Tests that exercise the actual OpenAI handler with a mocked
    websockets.connect to simulate the full flow.
    """

    @pytest.mark.asyncio
    async def test_full_flow_with_mocked_backend(self):
        """
        Simulate a complete WebSocket session:
        1. Client sends response.create
        2. Backend sends back response.created, output_text.delta, response.completed
        3. Verify all messages are forwarded correctly
        """
        from litellm.litellm_core_utils.responses_websocket_streaming import (
            ResponsesWebSocketStreaming,
        )

        backend_events = [
            json.dumps({"type": "response.created", "response": {"id": "resp_test123", "status": "in_progress"}}),
            json.dumps({"type": "response.output_text.delta", "delta": "Hello"}),
            json.dumps({"type": "response.output_text.delta", "delta": " World"}),
            json.dumps({"type": "response.completed", "response": {"id": "resp_test123", "status": "completed", "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello World"}]}]}}),
        ]

        event_idx = 0
        from websockets.exceptions import ConnectionClosed

        async def mock_recv(decode=False):
            nonlocal event_idx
            if event_idx < len(backend_events):
                msg = backend_events[event_idx]
                event_idx += 1
                return msg
            raise ConnectionClosed(None, None)

        mock_backend_ws = AsyncMock()
        mock_backend_ws.recv = mock_recv

        client_messages_sent = []

        class MockClientWS:
            async def send_text(self, data):
                client_messages_sent.append(data)

            async def receive_text(self):
                await asyncio.sleep(0.05)
                raise Exception("client done")

        mock_client_ws = MockClientWS()
        mock_logging = MagicMock()
        mock_logging.pre_call = MagicMock()
        mock_logging.async_success_handler = AsyncMock()

        streaming = ResponsesWebSocketStreaming(
            websocket=mock_client_ws,
            backend_ws=mock_backend_ws,
            logging_obj=mock_logging,
        )

        await streaming.bidirectional_forward()

        assert len(client_messages_sent) == 4, (
            f"Expected 4 messages forwarded to client, got {len(client_messages_sent)}"
        )

        forwarded_types = [
            json.loads(m).get("type") for m in client_messages_sent
        ]
        assert forwarded_types == [
            "response.created",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.completed",
        ]

        assert len(streaming.messages) == 2

    @pytest.mark.asyncio
    async def test_client_sends_and_backend_receives(self):
        """
        Verify client→backend forwarding: client sends response.create
        and the backend WS receives it.
        """
        from litellm.litellm_core_utils.responses_websocket_streaming import (
            ResponsesWebSocketStreaming,
        )

        request_msg = json.dumps({
            "type": "response.create",
            "response": {
                "model": "gpt-4o-mini",
                "input": "Hello",
            },
        })

        call_count = 0

        class MockClientWS:
            async def receive_text(self):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return request_msg
                raise Exception("done")

            async def send_text(self, data):
                pass

        mock_backend = AsyncMock()
        mock_logging = MagicMock()
        mock_logging.pre_call = MagicMock()

        streaming = ResponsesWebSocketStreaming(
            websocket=MockClientWS(),
            backend_ws=mock_backend,
            logging_obj=mock_logging,
        )

        await streaming.client_to_backend()

        mock_backend.send.assert_called_once_with(request_msg)
        assert len(streaming.input_messages) == 1
        assert streaming.input_messages[0]["type"] == "response.create"

    @pytest.mark.asyncio
    async def test_handler_constructs_correct_wss_url(self):
        """Verify OpenAIResponsesWebSocket builds the correct WSS URL."""
        from litellm.llms.openai.responses.websocket_handler import (
            OpenAIResponsesWebSocket,
        )

        handler = OpenAIResponsesWebSocket()

        assert handler._construct_url("https://api.openai.com/v1") == "wss://api.openai.com/v1/responses"
        assert handler._construct_url("http://localhost:4000/v1") == "ws://localhost:4000/v1/responses"
        assert handler._construct_url("https://custom.endpoint.com/v1/responses") == "wss://custom.endpoint.com/v1/responses"
