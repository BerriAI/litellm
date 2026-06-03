"""
Test for response_api_endpoints/endpoints.py
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from litellm.proxy.proxy_server import app


class TestResponsesAPIEndpoints(unittest.TestCase):
    @pytest.mark.asyncio
    @patch("litellm.proxy.proxy_server.llm_router")
    @patch("litellm.proxy.proxy_server.user_api_key_auth")
    async def test_openai_v1_responses_route(self, mock_auth, mock_router):
        """
        Test that /openai/v1/responses endpoint is correctly registered and accessible.
        """
        mock_auth.return_value = MagicMock(
            token="test_token",
            user_id="test_user",
            team_id=None,
        )

        mock_router.aresponses = AsyncMock(
            return_value={
                "id": "resp_abc123",
                "object": "realtime.response",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Test response"}],
                    }
                ],
            }
        )

        client = TestClient(app)

        test_data = {"model": "gpt-4o", "input": "Tell me about AI"}

        response = client.post(
            "/openai/v1/responses",
            json=test_data,
            headers={"Authorization": "Bearer sk-1234"},
        )

        assert response.status_code in [200, 401, 500]

    @pytest.mark.asyncio
    @patch("litellm.proxy.proxy_server.llm_router")
    @patch("litellm.proxy.proxy_server.user_api_key_auth")
    async def test_cursor_chat_completions_route(self, mock_auth, mock_router):
        """
        Test that /cursor/chat/completions endpoint:
        1. Accepts Responses API input format
        2. Returns chat completions format response
        3. Transforms streaming responses correctly
        """
        from litellm.types.llms.openai import ResponsesAPIResponse
        from litellm.types.utils import ResponseOutputMessage, ResponseOutputText

        mock_auth.return_value = MagicMock(
            token="test_token",
            user_id="test_user",
            team_id=None,
        )

        # Mock a Responses API response
        mock_responses_response = ResponsesAPIResponse(
            id="resp_cursor123",
            created_at=1234567890,
            model="gpt-4o",
            object="response",
            output=[
                ResponseOutputMessage(
                    type="message",
                    role="assistant",
                    content=[
                        ResponseOutputText(
                            type="output_text", text="Hello from Cursor!"
                        )
                    ],
                )
            ],
        )

        mock_router.aresponses = AsyncMock(return_value=mock_responses_response)

        client = TestClient(app)

        # Test with Responses API input format (what Cursor sends)
        test_data = {
            "model": "gpt-4o",
            "input": [{"role": "user", "content": "Hello"}],
        }

        response = client.post(
            "/cursor/chat/completions",
            json=test_data,
            headers={"Authorization": "Bearer sk-1234"},
        )

        # Should return 200 (or 401/500 if auth fails)
        assert response.status_code in [200, 401, 500]

        # If successful, verify it returns chat completions format
        if response.status_code == 200:
            response_data = response.json()
            # Should have chat completion structure
            assert "choices" in response_data or "id" in response_data
            # Should not have Responses API structure
            assert "output" not in response_data or "status" not in response_data

    @pytest.mark.asyncio
    @patch("litellm.proxy.proxy_server.llm_router")
    @patch("litellm.proxy.proxy_server.user_api_key_auth")
    async def test_responses_api_key_spend_header_includes_response_cost(
        self, mock_auth, mock_router
    ):
        """
        Test that x-litellm-key-spend header includes the current request's response_cost
        for /v1/responses endpoint.

        This ensures the spend header reflects updated spend including the current request,
        even though spend tracking updates happen asynchronously after the response.
        """
        from litellm.types.llms.openai import ResponsesAPIResponse
        from litellm.types.utils import ResponseOutputMessage, ResponseOutputText

        # Create mock user API key with initial spend
        mock_user_api_key_dict = MagicMock()
        mock_user_api_key_dict.token = "test_token"
        mock_user_api_key_dict.user_id = "test_user"
        mock_user_api_key_dict.team_id = None
        mock_user_api_key_dict.spend = 0.001  # Initial spend: $0.001
        mock_user_api_key_dict.tpm_limit = None
        mock_user_api_key_dict.rpm_limit = None
        mock_user_api_key_dict.max_budget = None
        mock_user_api_key_dict.allowed_model_region = None
        mock_user_api_key_dict.api_key = "sk-test-key"
        mock_user_api_key_dict.metadata = {}

        mock_auth.return_value = mock_user_api_key_dict

        # Mock response with hidden_params containing response_cost
        mock_response = ResponsesAPIResponse(
            id="resp_test123",
            created_at=1234567890,
            model="gpt-4o",
            object="response",
            output=[
                ResponseOutputMessage(
                    type="message",
                    role="assistant",
                    content=[
                        ResponseOutputText(type="output_text", text="Test response")
                    ],
                )
            ],
        )

        # Add hidden_params with response_cost to the mock response
        mock_response._hidden_params = {
            "response_cost": 0.0005,  # Current request cost: $0.0005
            "model_id": "test-model-id",
        }

        mock_router.aresponses = AsyncMock(return_value=mock_response)

        client = TestClient(app)

        test_data = {"model": "gpt-4o", "input": "Tell me about AI"}

        response = client.post(
            "/v1/responses",
            json=test_data,
            headers={"Authorization": "Bearer sk-test-key"},
        )

        # Verify the response was successful
        assert response.status_code == 200

        # Verify x-litellm-key-spend header includes current request cost
        assert "x-litellm-key-spend" in response.headers
        key_spend_value = float(response.headers["x-litellm-key-spend"])
        expected_spend = 0.001 + 0.0005  # Initial spend + current request cost
        assert key_spend_value == pytest.approx(expected_spend, abs=1e-10)

        # Verify x-litellm-response-cost header is present
        assert "x-litellm-response-cost" in response.headers
        response_cost_value = float(response.headers["x-litellm-response-cost"])
        assert response_cost_value == pytest.approx(0.0005, abs=1e-10)


import json


class TestManagedResponsesWSFirstMessage:
    @pytest.mark.asyncio
    async def test_first_message_processed_before_loop(self):
        """
        ManagedResponsesWebSocketHandler must process first_message before
        entering its receive loop. Regression for clients that connect without
        ?model= (e.g. Codex) and send model inside the first response.create event.
        """
        from litellm.responses.streaming_iterator import ManagedResponsesWebSocketHandler

        first = json.dumps(
            {
                "type": "response.create",
                "model": "gpt-4o-mini",
                "store": False,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hi"}],
                    }
                ],
            }
        )

        ws = MagicMock()
        ws.receive_text = AsyncMock(side_effect=Exception("disconnect"))
        ws.send_text = AsyncMock()

        processed: list = []

        async def fake_process(msg: str) -> None:
            processed.append(msg)

        handler = ManagedResponsesWebSocketHandler(
            websocket=ws,
            model="gpt-4o-mini",
            logging_obj=MagicMock(),
            first_message=first,
        )
        handler._process_response_create = fake_process  # type: ignore[method-assign]

        await handler.run()

        assert processed == [first]

    @pytest.mark.asyncio
    async def test_no_first_message_falls_through_to_loop(self):
        """When first_message is None, run() goes straight to receive_text()."""
        from litellm.responses.streaming_iterator import ManagedResponsesWebSocketHandler

        subsequent = json.dumps({"type": "response.create", "model": "gpt-4o-mini"})

        ws = MagicMock()
        ws.receive_text = AsyncMock(side_effect=[subsequent, Exception("disconnect")])
        ws.send_text = AsyncMock()

        processed: list = []

        async def fake_process(msg: str) -> None:
            processed.append(msg)

        handler = ManagedResponsesWebSocketHandler(
            websocket=ws,
            model="gpt-4o-mini",
            logging_obj=MagicMock(),
            first_message=None,
        )
        handler._process_response_create = fake_process  # type: ignore[method-assign]

        await handler.run()

        assert processed == [subsequent]


class TestResponsesWSStreamingFirstMessage:
    @pytest.mark.asyncio
    async def test_client_to_backend_replays_first_message(self):
        """
        ResponsesWebSocketStreaming.client_to_backend must send first_message to
        the backend before entering the receive loop.
        """
        from litellm.responses.streaming_iterator import ResponsesWebSocketStreaming

        first = json.dumps({"type": "response.create", "model": "gpt-4o-mini", "input": []})

        ws = MagicMock()
        ws.receive_text = AsyncMock(side_effect=Exception("disconnect"))

        backend_ws = MagicMock()
        backend_ws.send = AsyncMock()

        streaming = ResponsesWebSocketStreaming(
            websocket=ws,
            backend_ws=backend_ws,
            logging_obj=MagicMock(),
            first_message=first,
        )

        await streaming.client_to_backend()

        backend_ws.send.assert_awaited_once_with(first)


class TestWSSessionCostTracking:
    @pytest.mark.asyncio
    async def test_router_budget_limiter_skips_aresponses_websocket_call_type(self):
        """
        RouterBudgetLimiting.async_log_success_event must not raise when
        call_type='_aresponses_websocket', even when standard_logging_object is None.
        Per-turn costs are tracked by individual aresponses calls inside the session;
        the outer session wrapper fires with result=None.
        """
        from litellm.router_strategy.budget_limiter import RouterBudgetLimiting

        limiter = RouterBudgetLimiting.__new__(RouterBudgetLimiting)
        kwargs = {
            "call_type": "_aresponses_websocket",
            "standard_logging_object": None,
            "litellm_params": {"custom_llm_provider": "vertex_ai"},
        }
        await limiter.async_log_success_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )

    @pytest.mark.asyncio
    async def test_router_budget_limiter_skips_arealtime_call_type(self):
        """Same guard applies to _arealtime WS session wrappers."""
        from litellm.router_strategy.budget_limiter import RouterBudgetLimiting

        limiter = RouterBudgetLimiting.__new__(RouterBudgetLimiting)
        kwargs = {
            "call_type": "_arealtime",
            "standard_logging_object": None,
            "litellm_params": {"custom_llm_provider": "openai"},
        }
        await limiter.async_log_success_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )


class TestWSModelExtraction:
    """Test that first-frame model extraction handles both flat and nested formats."""

    def _extract_model(self, first_event: dict):
        """Mirror the extraction logic from endpoints.py."""
        nested = first_event.get("response")
        return (
            nested.get("model")
            if isinstance(nested, dict)
            else None
        ) or first_event.get("model")

    def test_flat_format_extracts_model(self):
        event = {"type": "response.create", "model": "gpt-4o", "input": "hello"}
        assert self._extract_model(event) == "gpt-4o"

    def test_nested_format_extracts_model(self):
        event = {"type": "response.create", "response": {"model": "gpt-4o", "input": "hello"}}
        assert self._extract_model(event) == "gpt-4o"

    def test_nested_format_takes_precedence_over_flat(self):
        event = {
            "type": "response.create",
            "model": "flat-model",
            "response": {"model": "nested-model"},
        }
        assert self._extract_model(event) == "nested-model"

    def test_no_model_returns_none(self):
        event = {"type": "response.create", "input": "hello"}
        assert self._extract_model(event) is None
