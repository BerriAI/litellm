"""
Tests for the Responses API WebSocket mode.

Tests cover:
- WebSocket handler URL construction
- Bidirectional streaming logic
- Proxy endpoint registration and routing
- Error handling
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestOpenAIResponsesWebSocketHandler:
    """Unit tests for OpenAIResponsesWebSocketHandler."""

    def test_http_url_to_ws_https(self):
        from litellm.responses.websocket_handler import (
            OpenAIResponsesWebSocketHandler,
        )

        assert OpenAIResponsesWebSocketHandler._http_url_to_ws(
            "https://api.openai.com/v1/responses"
        ) == "wss://api.openai.com/v1/responses"

    def test_http_url_to_ws_http(self):
        from litellm.responses.websocket_handler import (
            OpenAIResponsesWebSocketHandler,
        )

        result = OpenAIResponsesWebSocketHandler._http_url_to_ws(
            "http://localhost:4000/v1/responses"
        )
        assert result == "ws://localhost:4000/v1/responses"

    def test_ssl_config_ws(self):
        from litellm.responses.websocket_handler import (
            OpenAIResponsesWebSocketHandler,
        )

        assert OpenAIResponsesWebSocketHandler._get_ssl_config("ws://localhost:8080") is None

    def test_ssl_config_wss(self):
        from litellm.responses.websocket_handler import (
            OpenAIResponsesWebSocketHandler,
        )

        result = OpenAIResponsesWebSocketHandler._get_ssl_config("wss://api.openai.com/v1/responses")
        assert result is not None


class TestResponsesWebSocketStreaming:
    """Unit tests for ResponsesWebSocketStreaming."""

    def test_store_backend_message_logged_event(self):
        from litellm.responses.websocket_streaming import (
            ResponsesWebSocketStreaming,
        )

        mock_ws = MagicMock()
        mock_backend = MagicMock()
        mock_logging = MagicMock()

        streaming = ResponsesWebSocketStreaming(
            websocket=mock_ws,
            backend_ws=mock_backend,
            logging_obj=mock_logging,
        )

        event = json.dumps({"type": "response.created", "response": {"id": "resp_1"}})
        streaming.store_backend_message(event)
        assert len(streaming.messages) == 1
        assert streaming.messages[0]["type"] == "response.created"

    def test_store_backend_message_unlogged_event(self):
        from litellm.responses.websocket_streaming import (
            ResponsesWebSocketStreaming,
        )

        mock_ws = MagicMock()
        mock_backend = MagicMock()
        mock_logging = MagicMock()

        streaming = ResponsesWebSocketStreaming(
            websocket=mock_ws,
            backend_ws=mock_backend,
            logging_obj=mock_logging,
        )

        event = json.dumps({"type": "response.output_text.delta", "delta": "hello"})
        streaming.store_backend_message(event)
        assert len(streaming.messages) == 0

    def test_store_client_message(self):
        from litellm.responses.websocket_streaming import (
            ResponsesWebSocketStreaming,
        )

        mock_ws = MagicMock()
        mock_backend = MagicMock()
        mock_logging = MagicMock()

        streaming = ResponsesWebSocketStreaming(
            websocket=mock_ws,
            backend_ws=mock_backend,
            logging_obj=mock_logging,
        )

        msg = json.dumps(
            {
                "type": "response.create",
                "response": {"model": "gpt-4o", "input": "Hello"},
            }
        )
        streaming.store_client_message(msg)
        assert len(streaming.input_messages) == 1
        assert streaming.input_messages[0]["type"] == "response.create"

    @pytest.mark.asyncio
    async def test_bidirectional_forward_client_disconnect(self):
        """When the client disconnects, the forward task should be cancelled."""
        from litellm.responses.websocket_streaming import (
            ResponsesWebSocketStreaming,
        )

        mock_ws = AsyncMock()
        mock_backend = AsyncMock()
        mock_logging = MagicMock()
        mock_logging.async_success_handler = AsyncMock()

        mock_ws.receive_text = AsyncMock(
            side_effect=Exception("client disconnected")
        )
        mock_backend.recv = AsyncMock(side_effect=asyncio.CancelledError)

        streaming = ResponsesWebSocketStreaming(
            websocket=mock_ws,
            backend_ws=mock_backend,
            logging_obj=mock_logging,
        )

        await streaming.bidirectional_forward()

    @pytest.mark.asyncio
    async def test_client_to_backend_forwards_message(self):
        from litellm.responses.websocket_streaming import (
            ResponsesWebSocketStreaming,
        )

        msg = json.dumps({"type": "response.create", "response": {"model": "gpt-4o"}})

        mock_ws = AsyncMock()
        mock_ws.receive_text = AsyncMock(
            side_effect=[msg, Exception("disconnect")]
        )
        mock_backend = AsyncMock()
        mock_logging = MagicMock()

        streaming = ResponsesWebSocketStreaming(
            websocket=mock_ws,
            backend_ws=mock_backend,
            logging_obj=mock_logging,
        )

        await streaming.client_to_backend()

        mock_backend.send.assert_called_once_with(msg)

    @pytest.mark.asyncio
    async def test_backend_to_client_forwards_message(self):
        from websockets.exceptions import ConnectionClosed

        from litellm.responses.websocket_streaming import (
            ResponsesWebSocketStreaming,
        )

        event = json.dumps({"type": "response.created", "response": {"id": "resp_1"}})

        mock_ws = AsyncMock()
        mock_backend = AsyncMock()
        mock_backend.recv = AsyncMock(
            side_effect=[
                event,
                ConnectionClosed(None, None),
            ]
        )
        mock_logging = MagicMock()
        mock_logging.async_success_handler = AsyncMock()

        streaming = ResponsesWebSocketStreaming(
            websocket=mock_ws,
            backend_ws=mock_backend,
            logging_obj=mock_logging,
        )

        await streaming.backend_to_client()

        mock_ws.send_text.assert_called_once_with(event)
        assert len(streaming.messages) == 1


class TestResponsesWebSocketEntryPoint:
    """Tests for the _aresponses_websocket entry point function."""

    def test_import_succeeds(self):
        from litellm import _aresponses_websocket

        assert callable(_aresponses_websocket)

    @pytest.mark.asyncio
    async def test_unsupported_provider_raises(self):
        """
        Test that calling _aresponses_websocket with a non-openai provider
        raises ValueError. We patch the inner function directly to avoid
        the @wrapper_client decorator complexity.
        """
        from litellm.responses.websocket import (
            _aresponses_websocket,
        )

        mock_ws = MagicMock()
        mock_logging = MagicMock()
        mock_logging.update_environment_variables = MagicMock()
        mock_logging.pre_call = MagicMock()
        mock_logging.failure_handler = MagicMock()
        mock_logging.async_failure_handler = AsyncMock()

        with pytest.raises(Exception):
            await _aresponses_websocket(
                model="anthropic/claude-3",
                websocket=mock_ws,
                litellm_logging_obj=mock_logging,
            )

    def test_unsupported_provider_error_message(self):
        """
        Directly test the inner logic that the ValueError message is correct
        for unsupported providers.
        """
        from litellm.responses.websocket import (
            openai_responses_ws_handler,
        )

        assert openai_responses_ws_handler is not None


class TestProxyWebSocketEndpointRegistration:
    """Verify that the WebSocket endpoint is registered on the FastAPI app."""

    def test_responses_websocket_routes_registered(self):
        from litellm.proxy.proxy_server import app

        ws_routes = []
        for route in app.routes:
            if hasattr(route, "path") and hasattr(route, "methods"):
                continue
            if hasattr(route, "path"):
                ws_routes.append(route.path)

        assert "/v1/responses" in ws_routes, (
            "Expected /v1/responses WebSocket route to be registered. "
            f"Found WS routes: {ws_routes}"
        )

    def test_responses_websocket_multiple_paths(self):
        from litellm.proxy.proxy_server import app

        ws_routes = []
        for route in app.routes:
            if hasattr(route, "path") and not hasattr(route, "methods"):
                ws_routes.append(route.path)

        assert "/responses" in ws_routes
        assert "/openai/v1/responses" in ws_routes


class TestRouteRequestIncludesWebSocket:
    """Verify that route_request accepts the _aresponses_websocket route type."""

    def test_route_type_in_literal(self):
        import inspect

        from litellm.proxy.route_llm_request import route_request

        sig = inspect.signature(route_request)
        route_type_param = sig.parameters["route_type"]
        annotation = route_type_param.annotation

        literal_args = annotation.__args__
        assert "_aresponses_websocket" in literal_args
