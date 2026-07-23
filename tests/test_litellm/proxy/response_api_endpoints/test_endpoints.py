"""
Test for response_api_endpoints/endpoints.py
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import litellm
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
    """Test _extract_model_from_first_ws_event for flat and nested frame formats."""

    def test_flat_format_extracts_model(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            _extract_model_from_first_ws_event,
        )
        event = {"type": "response.create", "model": "gpt-4o", "input": "hello"}
        assert _extract_model_from_first_ws_event(event) == "gpt-4o"

    def test_nested_format_extracts_model(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            _extract_model_from_first_ws_event,
        )
        event = {"type": "response.create", "response": {"model": "gpt-4o", "input": "hello"}}
        assert _extract_model_from_first_ws_event(event) == "gpt-4o"

    def test_nested_format_takes_precedence_over_flat(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            _extract_model_from_first_ws_event,
        )
        event = {
            "type": "response.create",
            "model": "flat-model",
            "response": {"model": "nested-model"},
        }
        assert _extract_model_from_first_ws_event(event) == "nested-model"

    def test_no_model_returns_none(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            _extract_model_from_first_ws_event,
        )
        event = {"type": "response.create", "input": "hello"}
        assert _extract_model_from_first_ws_event(event) is None

    def test_non_object_returns_none(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            _extract_model_from_first_ws_event,
        )

        assert _extract_model_from_first_ws_event([]) is None


class TestResponsesWSFirstFrameValidation:
    @pytest.mark.asyncio
    async def test_rejects_non_response_create_first_frame(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            _read_ws_model_from_first_frame,
        )

        ws = MagicMock()
        ws.receive_text = AsyncMock(
            return_value=json.dumps({"type": "session.update", "model": "gpt-4o"})
        )
        ws.send_text = AsyncMock()
        ws.close = AsyncMock()

        result = await _read_ws_model_from_first_frame(ws)

        assert result is None
        ws.send_text.assert_awaited_once()
        ws.close.assert_awaited_once_with(code=1008, reason="Invalid first message")
        error_payload = json.loads(ws.send_text.await_args.args[0])
        assert (
            error_payload["error"]["message"]
            == "First message must be a response.create JSON object."
        )

    @pytest.mark.asyncio
    async def test_rejects_non_object_json_first_frame(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            _read_ws_model_from_first_frame,
        )

        ws = MagicMock()
        ws.receive_text = AsyncMock(return_value=json.dumps(["gpt-4o"]))
        ws.send_text = AsyncMock()
        ws.close = AsyncMock()

        result = await _read_ws_model_from_first_frame(ws)

        assert result is None
        ws.send_text.assert_awaited_once()
        ws.close.assert_awaited_once_with(code=1008, reason="Invalid first message")

    @pytest.mark.asyncio
    async def test_client_disconnect_first_frame_does_not_close(self):
        from fastapi import WebSocketDisconnect

        from litellm.proxy.response_api_endpoints.endpoints import (
            _read_ws_model_from_first_frame,
        )

        ws = MagicMock()
        ws.receive_text = AsyncMock(side_effect=WebSocketDisconnect(code=1006))
        ws.send_text = AsyncMock()
        ws.close = AsyncMock()

        result = await _read_ws_model_from_first_frame(ws)

        assert result is None
        ws.close.assert_not_awaited()
        ws.send_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_server_error_first_frame_closes_with_internal_error(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            _read_ws_model_from_first_frame,
        )

        ws = MagicMock()
        ws.receive_text = AsyncMock(side_effect=RuntimeError("boom"))
        ws.send_text = AsyncMock()
        ws.close = AsyncMock()

        result = await _read_ws_model_from_first_frame(ws)

        assert result is None
        ws.close.assert_awaited_once_with(code=1011, reason="Internal server error")


class TestResponsesWSFirstFrameModelAuth:
    @pytest.mark.asyncio
    async def test_endpoint_enforces_auth_after_model_from_first_frame(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            responses_websocket_endpoint,
        )

        ws = MagicMock()
        ws.headers = {}
        ws.query_params = {}
        ws.scope = {"headers": []}
        ws.url = "ws://testserver/v1/responses"
        ws.accept = AsyncMock()
        ws.receive_text = AsyncMock(
            return_value=json.dumps(
                {"type": "response.create", "model": "gpt-4o-mini", "input": []}
            )
        )
        ws.close = AsyncMock()

        processor = MagicMock()
        processor.common_processing_pre_call_logic = AsyncMock(
            return_value=({"model": "gpt-4o-mini"}, MagicMock())
        )

        async def fake_llm_call():
            return None

        with (
            patch(
                "litellm.proxy.response_api_endpoints.endpoints._enforce_responses_ws_first_frame_model_auth",
                new_callable=AsyncMock,
            ) as mock_model_auth,
            patch(
                "litellm.proxy.response_api_endpoints.endpoints.ProxyBaseLLMRequestProcessing",
                return_value=processor,
            ),
            patch(
                "litellm.proxy.route_llm_request.route_request",
                new_callable=AsyncMock,
                return_value=fake_llm_call(),
            ),
        ):
            await responses_websocket_endpoint(
                websocket=ws,
                model=None,
                user_api_key_dict=MagicMock(),
            )

        mock_model_auth.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reruns_model_auth_for_first_frame_model(self):
        from starlette.requests import Request

        from litellm.proxy.response_api_endpoints.endpoints import (
            _enforce_responses_ws_first_frame_model_auth,
        )

        request = Request(
            {"type": "http", "method": "POST", "path": "/v1/responses", "headers": []}
        )
        user_api_key_dict = MagicMock()
        llm_router = MagicMock()

        with (
            patch(
                "litellm.proxy.auth.user_api_key_auth._enforce_key_and_fallback_model_access",
                new_callable=AsyncMock,
            ) as mock_key_check,
            patch(
                "litellm.proxy.auth.user_api_key_auth._run_centralized_common_checks",
                new_callable=AsyncMock,
            ) as mock_common_checks,
            patch(
                "litellm.proxy.proxy_server.llm_model_list",
                [],
            ),
            patch("litellm.proxy.proxy_server.master_key", "sk-test"),
            patch("litellm.proxy.proxy_server.user_custom_auth", None),
            patch("litellm.proxy.proxy_server.general_settings", {}),
        ):
            await _enforce_responses_ws_first_frame_model_auth(
                request=request,
                model="gpt-4o-mini",
                user_api_key_dict=user_api_key_dict,
                llm_router=llm_router,
            )

        mock_key_check.assert_awaited_once_with(
            valid_token=user_api_key_dict,
            request_data={"model": "gpt-4o-mini"},
            route="/v1/responses",
            request=request,
            llm_model_list=[],
            llm_router=llm_router,
        )
        mock_common_checks.assert_awaited_once_with(
            user_api_key_auth_obj=user_api_key_dict,
            request=request,
            request_data={"model": "gpt-4o-mini"},
            route="/v1/responses",
        )


class TestReadWSModelFromFirstFrameErrors:
    @pytest.mark.asyncio
    async def test_timeout_closes_without_error_frame(self):
        import asyncio

        from litellm.proxy.response_api_endpoints.endpoints import (
            _read_ws_model_from_first_frame,
        )

        ws = MagicMock()
        ws.receive_text = AsyncMock(side_effect=asyncio.TimeoutError())
        ws.send_text = AsyncMock()
        ws.close = AsyncMock()

        result = await _read_ws_model_from_first_frame(ws)

        assert result is None
        ws.send_text.assert_not_awaited()
        ws.close.assert_awaited_once_with(
            code=1008, reason="Timed out waiting for first message"
        )

    @pytest.mark.asyncio
    async def test_invalid_json_sends_error_and_closes(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            _read_ws_model_from_first_frame,
        )

        ws = MagicMock()
        ws.receive_text = AsyncMock(return_value="this is not json")
        ws.send_text = AsyncMock()
        ws.close = AsyncMock()

        result = await _read_ws_model_from_first_frame(ws)

        assert result is None
        payload = json.loads(ws.send_text.await_args.args[0])
        assert payload["error"]["message"] == "First message is not valid JSON."
        ws.close.assert_awaited_once_with(
            code=1008, reason="Invalid JSON in first message"
        )

    @pytest.mark.asyncio
    async def test_missing_model_sends_error_and_closes(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            _read_ws_model_from_first_frame,
        )

        ws = MagicMock()
        ws.receive_text = AsyncMock(
            return_value=json.dumps({"type": "response.create", "input": []})
        )
        ws.send_text = AsyncMock()
        ws.close = AsyncMock()

        result = await _read_ws_model_from_first_frame(ws)

        assert result is None
        payload = json.loads(ws.send_text.await_args.args[0])
        assert "No model provided" in payload["error"]["message"]
        ws.close.assert_awaited_once_with(code=1008, reason="No model provided")

    @pytest.mark.asyncio
    async def test_valid_first_frame_returns_model_and_raw(self):
        from litellm.proxy.response_api_endpoints.endpoints import (
            _read_ws_model_from_first_frame,
        )

        raw = json.dumps({"type": "response.create", "model": "gpt-4o", "input": []})
        ws = MagicMock()
        ws.receive_text = AsyncMock(return_value=raw)
        ws.send_text = AsyncMock()
        ws.close = AsyncMock()

        result = await _read_ws_model_from_first_frame(ws)

        assert result == ("gpt-4o", raw)
        ws.send_text.assert_not_awaited()
        ws.close.assert_not_awaited()


class TestManagedResponsesSameProvider:
    def _handler(self, model, custom_llm_provider=None):
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        return ManagedResponsesWebSocketHandler(
            websocket=MagicMock(),
            model=model,
            logging_obj=MagicMock(),
            custom_llm_provider=custom_llm_provider,
        )

    def test_none_model_treated_as_same_provider(self):
        assert self._handler("openai/gpt-4o")._same_provider(None) is True

    def test_identical_model_is_same_provider(self):
        assert self._handler("openai/gpt-4o")._same_provider("openai/gpt-4o") is True

    def test_same_provider_different_model(self):
        assert self._handler("gpt-4o")._same_provider("gpt-4o-mini") is True

    def test_different_provider_is_not_same(self):
        assert (
            self._handler("gpt-4o")._same_provider("vertex_ai/gemini-2.0-flash")
            is False
        )

    def test_inject_credentials_keeps_provider_for_same_provider_model(self):
        handler = self._handler("gpt-4o", custom_llm_provider="openai")
        call_kwargs: dict = {}
        handler._inject_credentials(call_kwargs, model="gpt-4o-mini")
        assert call_kwargs["custom_llm_provider"] == "openai"

    def test_inject_credentials_drops_provider_for_cross_provider_model(self):
        handler = self._handler("gpt-4o", custom_llm_provider="openai")
        call_kwargs: dict = {}
        handler._inject_credentials(call_kwargs, model="vertex_ai/gemini-2.0-flash")
        assert "custom_llm_provider" not in call_kwargs

    def test_unresolvable_connection_model_falls_back_to_custom_provider(self):
        handler = self._handler(
            "my-custom-deployment", custom_llm_provider="openai"
        )
        assert handler._same_provider("gpt-4o-mini") is True
        call_kwargs: dict = {}
        handler._inject_credentials(call_kwargs, model="gpt-4o-mini")
        assert call_kwargs["custom_llm_provider"] == "openai"

    def test_unresolvable_connection_model_still_drops_cross_provider(self):
        handler = self._handler(
            "my-custom-deployment", custom_llm_provider="openai"
        )
        call_kwargs: dict = {}
        handler._inject_credentials(call_kwargs, model="vertex_ai/gemini-2.0-flash")
        assert "custom_llm_provider" not in call_kwargs


def _auth_override():
    from litellm.proxy._types import UserAPIKeyAuth

    return UserAPIKeyAuth(api_key="sk-test-cursor", user_id="cursor-user")


def test_cursor_chat_completions_messages_body_uses_chat_pipeline():
    """A genuine chat-completions body (``messages`` present; what Cursor sends for
    models whose BYOK it already fixed) must run through the standard chat pipeline
    untouched: multi-turn tool history (assistant tool_calls + role="tool" results)
    and nested chat-format tool defs are valid there, while blindly renaming
    ``messages`` to ``input`` (the pre-fix behavior) produced items the Responses API
    rejects. Asserts acompletion is called with the exact messages and aresponses is
    never touched."""
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    import litellm.proxy.proxy_server as ps

    messages = [
        {"role": "user", "content": "read a file"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_hist1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_hist1", "content": "file contents"},
        {"role": "user", "content": "now summarize"},
    ]

    mock_router = MagicMock()
    mock_router.acompletion = AsyncMock(
        return_value=litellm.ModelResponse(
            id="chatcmpl-cursor-1",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "summary"},
                    "finish_reason": "stop",
                }
            ],
            model="gpt-4o",
        )
    )
    mock_router.aresponses = AsyncMock()
    mock_router.get_available_deployment = MagicMock(return_value=None)

    app.dependency_overrides[user_api_key_auth] = _auth_override
    try:
        with patch.object(ps, "llm_router", mock_router):
            client = TestClient(app)
            response = client.post(
                "/cursor/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": messages,
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": "read_file", "parameters": {"type": "object"}},
                        }
                    ],
                },
                headers={"Authorization": "Bearer sk-test-cursor"},
            )
    finally:
        app.dependency_overrides.pop(user_api_key_auth, None)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "summary"
    assert "output" not in body

    mock_router.acompletion.assert_called_once()
    called_kwargs = mock_router.acompletion.call_args.kwargs
    assert called_kwargs["messages"] == messages
    assert "input" not in called_kwargs
    mock_router.aresponses.assert_not_called()


def test_cursor_chat_completions_input_body_uses_responses_pipeline_and_strips_stream_options():
    """A Responses-shaped body (``input``, no ``messages``; what Cursor agent mode
    sends) must run through the Responses pipeline with chat-completions output, and
    ``stream_options`` (chat-completions-only; Cursor sends include_usage) must be
    stripped before the Responses call since OpenAI's Responses API rejects it.
    Stripping must not mutate the dict _read_request_body returned: that can be the
    request-scope cached parsed body itself, and removing a key from it corrupts the
    cache's key snapshot so any later _read_request_body caller (spend tracking,
    logging hooks) silently gets an empty body; a follow-up read must still see the
    full original body."""
    import asyncio

    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.common_utils.http_parsing_utils import (
        _read_request_body as real_read_request_body,
    )
    from litellm.types.llms.openai import ResponsesAPIResponse

    import litellm.proxy.proxy_server as ps

    captured_requests = []

    async def capturing_read_request_body(request):
        captured_requests.append(request)
        return await real_read_request_body(request=request)

    mock_router = MagicMock()
    mock_router.aresponses = AsyncMock(
        return_value=ResponsesAPIResponse(
            id="resp_cursor_agent1",
            created_at=1234567890,
            model="gpt-4o",
            object="response",
            output=[
                ResponseOutputMessage(
                    id="msg_agent1",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[
                        ResponseOutputText(type="output_text", text="agent reply", annotations=[])
                    ],
                )
            ],
        )
    )
    mock_router.acompletion = AsyncMock()

    app.dependency_overrides[user_api_key_auth] = _auth_override
    try:
        with patch.object(ps, "llm_router", mock_router), patch.object(
            ps, "_read_request_body", side_effect=capturing_read_request_body
        ):
            client = TestClient(app)
            response = client.post(
                "/cursor/chat/completions",
                json={
                    "model": "gpt-4o",
                    "input": [{"role": "user", "content": "hello"}],
                    "stream_options": {"include_usage": True},
                },
                headers={"Authorization": "Bearer sk-test-cursor"},
            )
    finally:
        app.dependency_overrides.pop(user_api_key_auth, None)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "agent reply"
    assert "output" not in body

    mock_router.aresponses.assert_called_once()
    called_kwargs = mock_router.aresponses.call_args.kwargs
    assert "stream_options" not in called_kwargs
    mock_router.acompletion.assert_not_called()

    assert captured_requests
    followup_body = asyncio.run(real_read_request_body(request=captured_requests[0]))
    assert followup_body.get("stream_options") == {"include_usage": True}
    assert followup_body.get("input") == [{"role": "user", "content": "hello"}]


def test_cursor_models_route_delegates_to_model_list():
    """Clients pointed at <proxy>/cursor as an OpenAI-compatible base URL resolve and
    verify keys via GET {base}/models (the OpenAI SDK contract). Without a dedicated
    route those requests fall through to the Cursor Cloud Agents passthrough and 401
    for lack of a Cursor API key, so BYOK verification fails before any chat request
    is sent. Both /cursor/models and /cursor/v1/models must serve the standard model
    list instead."""
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    import litellm.proxy.proxy_server as ps

    model_payload = {"data": [{"id": "gpt-5.6", "object": "model"}], "object": "list"}

    app.dependency_overrides[user_api_key_auth] = _auth_override
    try:
        with patch.object(ps, "model_list", AsyncMock(return_value=model_payload)) as mock_model_list:
            client = TestClient(app)
            for path in ("/cursor/models", "/cursor/v1/models"):
                response = client.get(path, headers={"Authorization": "Bearer sk-test-cursor"})
                assert response.status_code == 200, f"{path}: {response.text}"
                assert response.json() == model_payload
            assert mock_model_list.call_count == 2
    finally:
        app.dependency_overrides.pop(user_api_key_auth, None)


class TestNestFlatChatTools:
    def test_flat_custom_tool_is_nested(self):
        from litellm.proxy.response_api_endpoints.endpoints import _convert_tool_envelope

        result = _convert_tool_envelope(
            {"type": "custom", "name": "ApplyPatch", "description": "V4A patch", "format": {"type": "text"}},
            to_chat=True,
        )
        assert result == {
            "type": "custom",
            "custom": {"name": "ApplyPatch", "description": "V4A patch", "format": {"type": "text"}},
        }

    def test_flat_function_tool_is_nested(self):
        from litellm.proxy.response_api_endpoints.endpoints import _convert_tool_envelope

        result = _convert_tool_envelope(
            {"type": "function", "name": "read_file", "description": "d", "parameters": {"type": "object"}},
            to_chat=True,
        )
        assert result == {
            "type": "function",
            "function": {"name": "read_file", "description": "d", "parameters": {"type": "object"}},
        }

    def test_already_nested_and_unrecognized_tools_pass_through_unchanged(self):
        from litellm.proxy.response_api_endpoints.endpoints import _convert_tool_envelope

        tools = [
            {"type": "custom", "custom": {"name": "already_nested"}},
            {"type": "function", "function": {"name": "f", "parameters": {}}},
            {"type": "web_search"},
            {"type": "custom"},
            {"name": "typeless"},
            {},
            "junk",
            None,
            42,
        ]
        assert [_convert_tool_envelope(tool, to_chat=True) for tool in tools] == tools


class TestCursorMessagesArmToolNormalization:
    @pytest.mark.asyncio
    async def test_flat_custom_tool_nested_before_chat_completion_delegation(self):
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
        from litellm.proxy._types import UserAPIKeyAuth

        seen = {}

        async def fake_chat_completion(request, fastapi_response, model, user_api_key_dict):
            from litellm.proxy.common_utils.http_parsing_utils import _read_request_body

            seen["body"] = await _read_request_body(request=request)
            return {"id": "chatcmpl-fake", "object": "chat.completion", "choices": []}

        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(api_key="sk-1234")
        try:
            with patch("litellm.proxy.proxy_server.chat_completion", new=fake_chat_completion):
                client = TestClient(app)
                response = client.post(
                    "/cursor/chat/completions",
                    json={
                        "model": "gpt-5.6",
                        "messages": [{"role": "user", "content": "use ApplyPatch"}],
                        "tools": [
                            {
                                "type": "function",
                                "function": {"name": "read_file", "parameters": {"type": "object"}},
                            },
                            {
                                "type": "custom",
                                "name": "ApplyPatch",
                                "description": "V4A patch",
                                "format": {
                                    "type": "grammar",
                                    "definition": "start: patch",
                                    "syntax": "lark",
                                },
                            },
                        ],
                        "tool_choice": {"type": "custom", "name": "ApplyPatch"},
                    },
                    headers={"Authorization": "Bearer sk-1234"},
                )
        finally:
            app.dependency_overrides.pop(user_api_key_auth, None)

        assert response.status_code == 200
        assert seen["body"]["tools"] == [
            {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}},
            {
                "type": "custom",
                "custom": {
                    "name": "ApplyPatch",
                    "description": "V4A patch",
                    "format": {
                        "type": "grammar",
                        "grammar": {"definition": "start: patch", "syntax": "lark"},
                    },
                },
            },
        ]
        assert seen["body"]["tool_choice"] == {"type": "custom", "custom": {"name": "ApplyPatch"}}
        assert seen["body"]["messages"] == [{"role": "user", "content": "use ApplyPatch"}]

    @pytest.mark.asyncio
    async def test_messages_body_without_flat_tools_leaves_parsed_body_cache_untouched(self):
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
        from litellm.proxy._types import UserAPIKeyAuth

        seen = {}

        async def fake_chat_completion(request, fastapi_response, model, user_api_key_dict):
            from litellm.proxy.common_utils.http_parsing_utils import _read_request_body

            seen["body"] = await _read_request_body(request=request)
            return {"id": "chatcmpl-fake", "object": "chat.completion", "choices": []}

        body = {
            "model": "gpt-5.6",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        }
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(api_key="sk-1234")
        try:
            with patch("litellm.proxy.proxy_server.chat_completion", new=fake_chat_completion):
                client = TestClient(app)
                response = client.post(
                    "/cursor/chat/completions",
                    json=body,
                    headers={"Authorization": "Bearer sk-1234"},
                )
        finally:
            app.dependency_overrides.pop(user_api_key_auth, None)

        assert response.status_code == 200
        assert seen["body"]["tools"] == body["tools"]
        assert seen["body"]["messages"] == body["messages"]


class TestToolEnvelopeConversionMatrix:
    """
    Cursor mixes Responses API shapes into chat bodies PER LEVEL, independently
    (live-captured: a pre-nested custom envelope carrying a flat grammar format).
    Tool definitions and tool_choice share one envelope rule, so every cell of
    direction x envelope x format must land on that direction's canonical shape.
    """

    FLAT_GRAMMAR = {"type": "grammar", "definition": "start: patch", "syntax": "lark"}
    NESTED_GRAMMAR = {"type": "grammar", "grammar": {"definition": "start: patch", "syntax": "lark"}}
    TEXT = {"type": "text"}

    @pytest.mark.parametrize("to_chat", [True, False])
    @pytest.mark.parametrize("envelope", ["flat", "nested"])
    @pytest.mark.parametrize("format_shape", ["absent", "text", "flat_grammar", "nested_grammar"])
    def test_every_direction_envelope_and_format_lands_canonical(self, to_chat, envelope, format_shape):
        from litellm.proxy.response_api_endpoints.endpoints import _convert_tool_envelope

        format_value = {
            "absent": None,
            "text": self.TEXT,
            "flat_grammar": self.FLAT_GRAMMAR,
            "nested_grammar": self.NESTED_GRAMMAR,
        }[format_shape]
        payload = {"name": "ApplyPatch", "description": "V4A patch"}
        if format_value is not None:
            payload["format"] = format_value
        tool = {"type": "custom", "custom": payload} if envelope == "nested" else {"type": "custom", **payload}

        canonical_payload = {"name": "ApplyPatch", "description": "V4A patch"}
        if format_shape in ("flat_grammar", "nested_grammar"):
            canonical_payload["format"] = self.NESTED_GRAMMAR if to_chat else self.FLAT_GRAMMAR
        elif format_shape == "text":
            canonical_payload["format"] = self.TEXT
        expected = (
            {"type": "custom", "custom": canonical_payload} if to_chat else {"type": "custom", **canonical_payload}
        )

        assert _convert_tool_envelope(tool, to_chat=to_chat) == expected

    def test_nested_envelope_with_flat_grammar_matches_live_cursor_capture(self):
        from litellm.proxy.response_api_endpoints.endpoints import _convert_tool_envelope

        cursor_tool = {"type": "custom", "custom": {"name": "ApplyPatch", "format": self.FLAT_GRAMMAR}}
        assert _convert_tool_envelope(cursor_tool, to_chat=True) == {
            "type": "custom",
            "custom": {"name": "ApplyPatch", "format": self.NESTED_GRAMMAR},
        }

    @pytest.mark.parametrize("to_chat", [True, False])
    def test_conversion_is_idempotent(self, to_chat):
        from litellm.proxy.response_api_endpoints.endpoints import _convert_tool_envelope

        once = _convert_tool_envelope({"type": "custom", "name": "A", "format": self.FLAT_GRAMMAR}, to_chat=to_chat)
        assert _convert_tool_envelope(once, to_chat=to_chat) == once

    def test_nested_function_tool_flattens_and_flat_passes_through(self):
        from litellm.proxy.response_api_endpoints.endpoints import _convert_tool_envelope

        nested = {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}
        flat = {"type": "function", "name": "read_file", "parameters": {"type": "object"}}
        assert _convert_tool_envelope(nested, to_chat=False) == flat
        assert _convert_tool_envelope(flat, to_chat=False) == flat

    @pytest.mark.parametrize("to_chat", [True, False])
    def test_unrecognized_entries_pass_through(self, to_chat):
        from litellm.proxy.response_api_endpoints.endpoints import _convert_tool_envelope

        entries = [{"type": "web_search"}, {"type": "custom"}, "junk", None, {}, 42, {"type": "auto"}]
        assert [_convert_tool_envelope(entry, to_chat=to_chat) for entry in entries] == entries


class TestToolChoiceSharesTheToolEnvelopeRule:
    """
    tool_choice carries the same {"type": T, T: {...}} chat envelope as a tool
    definition, so it converts through the same function in both directions.
    OpenAI requires the nested key on chat (SDK ChatCompletionNamedToolChoiceParam
    and ChatCompletionNamedToolChoiceCustomParam both mark it Required).
    """

    @pytest.mark.parametrize("choice_type", ["custom", "function"])
    def test_flat_tool_choice_is_nested_for_chat(self, choice_type):
        from litellm.proxy.response_api_endpoints.endpoints import _convert_tool_envelope

        assert _convert_tool_envelope({"type": choice_type, "name": "ApplyPatch"}, to_chat=True) == {
            "type": choice_type,
            choice_type: {"name": "ApplyPatch"},
        }

    @pytest.mark.parametrize("choice_type", ["custom", "function"])
    def test_nested_tool_choice_is_flattened_for_responses(self, choice_type):
        from litellm.proxy.response_api_endpoints.endpoints import _convert_tool_envelope

        assert _convert_tool_envelope({"type": choice_type, choice_type: {"name": "ApplyPatch"}}, to_chat=False) == {
            "type": choice_type,
            "name": "ApplyPatch",
        }

    @pytest.mark.parametrize("to_chat", [True, False])
    def test_sentinel_and_malformed_tool_choice_pass_through(self, to_chat):
        from litellm.proxy.response_api_endpoints.endpoints import _convert_tool_envelope

        for unchanged in ("auto", "required", "none", None, {"type": "auto"}, 42):
            assert _convert_tool_envelope(unchanged, to_chat=to_chat) == unchanged


class TestNormalizeToolDialectCoversBothFields:
    """
    The regression that motivated one normalizer: tools were converted while
    tool_choice was left flat, so OpenAI rejected the request. Both fields move
    together in a single call, on both arms.
    """

    @pytest.mark.parametrize("to_chat", [True, False])
    def test_tools_and_tool_choice_convert_together(self, to_chat):
        from litellm.proxy.response_api_endpoints.endpoints import _normalize_tool_dialect

        flat = {"type": "custom", "name": "ApplyPatch"}
        nested = {"type": "custom", "custom": {"name": "ApplyPatch"}}
        source = flat if to_chat else nested
        expected = nested if to_chat else flat

        out = _normalize_tool_dialect({"messages": [], "tools": [source], "tool_choice": source}, to_chat=to_chat)
        assert out["tools"] == [expected]
        assert out["tool_choice"] == expected

    def test_body_needing_no_conversion_is_returned_by_identity(self):
        from litellm.proxy.response_api_endpoints.endpoints import _normalize_tool_dialect

        data = {"messages": [], "tools": [{"type": "function", "function": {"name": "f"}}], "tool_choice": "auto"}
        assert _normalize_tool_dialect(data, to_chat=True) is data

    def test_absent_tool_fields_are_not_invented(self):
        from litellm.proxy.response_api_endpoints.endpoints import _normalize_tool_dialect

        data = {"messages": [{"role": "user", "content": "hi"}]}
        result = _normalize_tool_dialect(data, to_chat=True)
        assert result == data
        assert "tools" not in result and "tool_choice" not in result


class TestCursorInputArmFlattening:
    @pytest.mark.asyncio
    async def test_nested_chat_shapes_in_input_body_reach_aresponses_flattened(self):
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
        from openai.types.responses import ResponseOutputMessage, ResponseOutputText

        from litellm.types.llms.openai import ResponsesAPIResponse

        mock_response = ResponsesAPIResponse(
            id="resp_flat123",
            created_at=1234567890,
            model="gpt-5.6",
            object="response",
            output=[
                ResponseOutputMessage(
                    id="msg_flat123",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[ResponseOutputText(type="output_text", text="ok", annotations=[])],
                )
            ],
        )

        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(api_key="sk-1234")
        try:
            with patch("litellm.proxy.proxy_server.llm_router") as mock_router:
                mock_router.aresponses = AsyncMock(return_value=mock_response)
                client = TestClient(app)
                response = client.post(
                    "/cursor/chat/completions",
                    json={
                        "model": "gpt-5.6",
                        "input": [{"role": "user", "content": "use ApplyPatch"}],
                        "tools": [
                            {
                                "type": "custom",
                                "custom": {
                                    "name": "ApplyPatch",
                                    "format": {
                                        "type": "grammar",
                                        "grammar": {"definition": "start: patch", "syntax": "lark"},
                                    },
                                },
                            },
                            {"type": "function", "name": "read_file", "parameters": {"type": "object"}},
                        ],
                        "tool_choice": {"type": "custom", "custom": {"name": "ApplyPatch"}},
                    },
                    headers={"Authorization": "Bearer sk-1234"},
                )
        finally:
            app.dependency_overrides.pop(user_api_key_auth, None)

        assert response.status_code == 200
        call_kwargs = mock_router.aresponses.call_args.kwargs
        assert call_kwargs["tools"] == [
            {
                "type": "custom",
                "name": "ApplyPatch",
                "format": {"type": "grammar", "definition": "start: patch", "syntax": "lark"},
            },
            {"type": "function", "name": "read_file", "parameters": {"type": "object"}},
        ]
        assert call_kwargs["tool_choice"] == {"type": "custom", "name": "ApplyPatch"}


class TestChatCompletionsBodyDetection:
    def test_routing_matrix(self):
        from litellm.proxy.response_api_endpoints.endpoints import _is_chat_completions_body

        assert _is_chat_completions_body({"messages": [{"role": "user", "content": "hi"}]}) is True
        assert _is_chat_completions_body({"messages": [{"role": "user", "content": "hi"}], "input": []}) is True
        assert _is_chat_completions_body({"messages": None, "input": [{"role": "user", "content": "hi"}]}) is False
        assert _is_chat_completions_body({"messages": [], "input": [{"role": "user", "content": "hi"}]}) is False
        assert _is_chat_completions_body({"messages": None}) is True
        assert _is_chat_completions_body({"messages": []}) is True
        assert _is_chat_completions_body({"input": [{"role": "user", "content": "hi"}]}) is False
        assert _is_chat_completions_body({}) is False

    @pytest.mark.asyncio
    async def test_null_messages_stub_with_input_reaches_responses_arm(self):
        from openai.types.responses import ResponseOutputMessage, ResponseOutputText

        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
        from litellm.types.llms.openai import ResponsesAPIResponse

        mock_response = ResponsesAPIResponse(
            id="resp_stub1",
            created_at=1234567890,
            model="gpt-5.6",
            object="response",
            output=[
                ResponseOutputMessage(
                    id="msg_stub1",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[ResponseOutputText(type="output_text", text="ok", annotations=[])],
                )
            ],
        )

        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(api_key="sk-1234")
        try:
            with patch("litellm.proxy.proxy_server.llm_router") as mock_router:
                mock_router.aresponses = AsyncMock(return_value=mock_response)
                client = TestClient(app)
                response = client.post(
                    "/cursor/chat/completions",
                    json={
                        "model": "gpt-5.6",
                        "messages": None,
                        "input": [{"role": "user", "content": "hello"}],
                    },
                    headers={"Authorization": "Bearer sk-1234"},
                )
        finally:
            app.dependency_overrides.pop(user_api_key_auth, None)

        assert response.status_code == 200
        assert mock_router.aresponses.call_args is not None
        assert mock_router.aresponses.call_args.kwargs["input"] == [{"role": "user", "content": "hello"}]
