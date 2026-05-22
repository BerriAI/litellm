"""
Mock tests for A2A endpoints.

Tests that invoke_agent_a2a properly integrates with add_litellm_data_to_request.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_invoke_agent_a2a_adds_litellm_data():
    """
    Test that invoke_agent_a2a calls add_litellm_data_to_request
    and the resulting data includes proxy_server_request.
    """
    from litellm.proxy._types import UserAPIKeyAuth

    # Track the data passed to add_litellm_data_to_request
    captured_data = {}

    async def mock_add_litellm_data(data, **kwargs):
        # Simulate what add_litellm_data_to_request does
        data["proxy_server_request"] = {
            "url": "http://localhost:4000/a2a/test-agent",
            "method": "POST",
            "headers": {},
            "body": dict(data),
        }
        captured_data.update(data)
        return data

    # Mock response from asend_message
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "jsonrpc": "2.0",
        "id": "test-id",
        "result": {"status": "success"},
    }

    # Mock agent
    mock_agent = MagicMock()
    mock_agent.agent_card_params = {
        "url": "http://backend-agent:10001",
        "name": "Test Agent",
    }
    mock_agent.litellm_params = None

    # Mock request
    mock_request = MagicMock()
    mock_request.json = AsyncMock(
        return_value={
            "jsonrpc": "2.0",
            "id": "test-id",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Hello"}],
                    "messageId": "msg-123",
                }
            },
        }
    )

    mock_user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key",
        user_id="test-user",
        team_id="test-team",
    )

    # Try to use real a2a.types if available, otherwise create realistic mocks
    # This test focuses on LiteLLM integration, not A2A protocol correctness,
    # but we want mocks that behave like the real types to catch usage issues
    try:
        from a2a.types import (
            MessageSendParams,
            SendMessageRequest,
            SendStreamingMessageRequest,
        )

        # Real types available - use them
        pass
    except ImportError:
        # Real types not available - create realistic mocks
        pass

        def make_mock_pydantic_class(name):
            """Create a mock class that behaves like a Pydantic model."""

            class MockPydanticClass:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)
                    # Store kwargs for model_dump() if needed
                    self._kwargs = kwargs

                def model_dump(self, mode="json", exclude_none=False):
                    """Mock model_dump method."""
                    result = dict(self._kwargs)
                    if exclude_none:
                        result = {k: v for k, v in result.items() if v is not None}
                    return result

            MockPydanticClass.__name__ = name
            return MockPydanticClass

        MessageSendParams = make_mock_pydantic_class("MessageSendParams")
        SendMessageRequest = make_mock_pydantic_class("SendMessageRequest")
        SendStreamingMessageRequest = make_mock_pydantic_class(
            "SendStreamingMessageRequest"
        )

    # Create a mock module for a2a.types
    mock_a2a_types = MagicMock()
    mock_a2a_types.MessageSendParams = MessageSendParams
    mock_a2a_types.SendMessageRequest = SendMessageRequest
    mock_a2a_types.SendStreamingMessageRequest = SendStreamingMessageRequest

    # Patch at the source modules
    # Note: add_litellm_data_to_request is called from common_request_processing,
    # so we need to patch it there, not at litellm_pre_call_utils
    with (
        patch(
            "litellm.proxy.agent_endpoints.a2a_endpoints._get_agent",
            return_value=mock_agent,
        ),
        patch(
            "litellm.proxy.common_request_processing.add_litellm_data_to_request",
            side_effect=mock_add_litellm_data,
        ) as mock_add_data,
        patch(
            "litellm.a2a_protocol.create_a2a_client",
            new_callable=AsyncMock,
        ),
        patch(
            "litellm.a2a_protocol.asend_message",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {},
        ),
        patch(
            "litellm.proxy.proxy_server.proxy_config",
            MagicMock(),
        ),
        patch(
            "litellm.proxy.proxy_server.version",
            "1.0.0",
        ),
        patch.dict(
            sys.modules,
            {"a2a": MagicMock(), "a2a.types": mock_a2a_types},
        ),
        patch(
            "litellm.a2a_protocol.main.A2A_SDK_AVAILABLE",
            True,
        ),
    ):
        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        mock_fastapi_response = MagicMock()

        await invoke_agent_a2a(
            agent_id="test-agent",
            request=mock_request,
            fastapi_response=mock_fastapi_response,
            user_api_key_dict=mock_user_api_key_dict,
        )

        # Verify add_litellm_data_to_request was called
        mock_add_data.assert_called_once()

        # Verify model and custom_llm_provider were set
        assert captured_data.get("model") == "a2a_agent/Test Agent"
        assert captured_data.get("custom_llm_provider") == "a2a_agent"

        # Verify proxy_server_request was added
        assert "proxy_server_request" in captured_data
        assert captured_data["proxy_server_request"]["method"] == "POST"


# ---------------------------------------------------------------------------
# Tests for Bug 1: context_id auto-injection (LIT-2955)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_asend_message_no_context_id_injection_without_trace():
    """
    When trace_id is None and the caller did not set context_id, asend_message
    must NOT inject a random UUID into the message.  Strict A2A agents reject
    unknown contextId values (A2A spec §3.4.1 -32054 Session not found).
    """
    from litellm.a2a_protocol.main import asend_message

    injected_context_ids = []

    async def _mock_execute(a2a_client, request, **kwargs):
        """Capture whatever context_id ends up on the message."""
        msg = request.params.message
        if isinstance(msg, dict):
            injected_context_ids.append(msg.get("context_id"))
        else:
            injected_context_ids.append(getattr(msg, "context_id", None))
        return MagicMock(
            model_dump=MagicMock(
                return_value={"jsonrpc": "2.0", "id": "r1", "result": {}}
            )
        )

    mock_a2a_client = MagicMock()
    mock_a2a_client._litellm_agent_card = None
    mock_a2a_client.agent_card = None

    # Build a minimal send-message request (as a mock so we don't need the SDK)
    mock_message = {
        "role": "user",
        "parts": [{"kind": "text", "text": "hi"}],
        "messageId": "m1",
    }
    mock_params = MagicMock()
    mock_params.message = mock_message
    mock_request = MagicMock()
    mock_request.id = "r1"
    mock_request.params = mock_params

    mock_response = MagicMock()
    mock_response.model_dump = MagicMock(
        return_value={"jsonrpc": "2.0", "id": "r1", "result": {}}
    )

    with (
        patch(
            "litellm.a2a_protocol.main._execute_a2a_send_with_retry",
            new_callable=AsyncMock,
            side_effect=_mock_execute,
        ),
        patch(
            "litellm.a2a_protocol.main._get_a2a_model_info",
            return_value="test-agent",
        ),
        patch(
            "litellm.a2a_protocol.main.A2ARequestUtils.calculate_usage_from_request_response",
            return_value=(0, 0, 0),
        ),
        patch(
            "litellm.a2a_protocol.main.LiteLLMSendMessageResponse.from_a2a_response",
            return_value=mock_response,
        ),
    ):
        await asend_message(
            a2a_client=mock_a2a_client,
            request=mock_request,
            litellm_params={},
            # No litellm_logging_obj => trace_id will be None
        )

    # context_id must not have been injected
    assert injected_context_ids == [
        None
    ], f"context_id was unexpectedly injected: {injected_context_ids}"


@pytest.mark.asyncio
async def test_asend_message_no_context_id_injection_even_with_logging_obj():
    """
    Even when a LiteLLM logging object with a trace_id IS present, asend_message
    must NOT inject the trace_id as context_id.  context_id (A2A session) and
    trace_id (LiteLLM distributed tracing) are distinct concepts; conflating them
    breaks strict agents.
    """
    from litellm.a2a_protocol.main import asend_message

    injected_context_ids = []

    async def _capture(a2a_client, request, **kwargs):
        msg = request.params.message
        if isinstance(msg, dict):
            injected_context_ids.append(msg.get("context_id"))
        else:
            injected_context_ids.append(getattr(msg, "context_id", None))
        return MagicMock(
            model_dump=MagicMock(
                return_value={"jsonrpc": "2.0", "id": "r1", "result": {}}
            )
        )

    mock_a2a_client = MagicMock()
    mock_a2a_client._litellm_agent_card = None
    mock_a2a_client.agent_card = None

    mock_message = {
        "role": "user",
        "parts": [{"kind": "text", "text": "hi"}],
        "messageId": "m2",
    }
    mock_params = MagicMock()
    mock_params.message = mock_message
    mock_request = MagicMock()
    mock_request.id = "r1"
    mock_request.params = mock_params

    mock_response = MagicMock()
    mock_response.model_dump = MagicMock(
        return_value={"jsonrpc": "2.0", "id": "r1", "result": {}}
    )

    _trace_id = "trace-abc-123"

    # Build a mock logging_obj that carries litellm_trace_id
    mock_logging_obj = MagicMock()
    mock_logging_obj.litellm_trace_id = _trace_id
    mock_logging_obj.model_call_details = {}

    with (
        patch(
            "litellm.a2a_protocol.main._execute_a2a_send_with_retry",
            new_callable=AsyncMock,
            side_effect=_capture,
        ),
        patch(
            "litellm.a2a_protocol.main._get_a2a_model_info",
            return_value="test-agent",
        ),
        patch(
            "litellm.a2a_protocol.main.A2ARequestUtils.calculate_usage_from_request_response",
            return_value=(0, 0, 0),
        ),
        patch(
            "litellm.a2a_protocol.main.LiteLLMSendMessageResponse.from_a2a_response",
            return_value=mock_response,
        ),
    ):
        await asend_message(
            a2a_client=mock_a2a_client,
            request=mock_request,
            litellm_params={},
            litellm_logging_obj=mock_logging_obj,
        )

    # trace_id must NOT be injected as context_id even when a logging_obj is present
    assert injected_context_ids == [
        None
    ], f"trace_id was unexpectedly injected as context_id: {injected_context_ids}"


# ---------------------------------------------------------------------------
# Tests for Bug 2: tasks/get pass-through (LIT-2955)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invoke_agent_a2a_tasks_get_forwarded():
    """
    tasks/get must be forwarded to the upstream agent URL and the response
    returned to the caller.  Previously LiteLLM returned -32601 Method not found.
    """
    from litellm.proxy._types import UserAPIKeyAuth

    expected_task = {
        "jsonrpc": "2.0",
        "id": "r2",
        "result": {
            "kind": "task",
            "id": "task-uuid",
            "status": {"state": "completed"},
        },
    }

    mock_agent = MagicMock()
    mock_agent.agent_id = "agent-uuid"
    mock_agent.agent_name = "mock-agent"
    mock_agent.agent_card_params = {
        "url": "http://backend-agent:9999",
        "name": "Mock Agent",
    }
    mock_agent.litellm_params = {}
    mock_agent.static_headers = {}
    mock_agent.extra_headers = []

    mock_request = MagicMock()
    mock_request.headers = {}
    mock_request.json = AsyncMock(
        return_value={
            "jsonrpc": "2.0",
            "id": "r2",
            "method": "tasks/get",
            "params": {"id": "task-uuid"},
        }
    )

    mock_user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test",
        user_id="user-1",
        team_id="team-1",
    )

    mock_a2a_types = MagicMock()

    async def mock_forward(agent_url, body, extra_headers=None):
        """Simulate a successful upstream tasks/get response."""
        from fastapi.responses import JSONResponse

        return JSONResponse(content=expected_task)

    with (
        patch(
            "litellm.proxy.agent_endpoints.a2a_endpoints._get_agent",
            return_value=mock_agent,
        ),
        patch(
            "litellm.proxy.agent_endpoints.a2a_endpoints._forward_jsonrpc_to_agent",
            side_effect=mock_forward,
        ) as mock_fwd,
        patch(
            "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.is_agent_allowed",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "litellm.proxy.common_request_processing.add_litellm_data_to_request",
            side_effect=lambda data, **kw: data,
        ),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {},
        ),
        patch(
            "litellm.proxy.proxy_server.proxy_config",
            MagicMock(),
        ),
        patch(
            "litellm.proxy.proxy_server.version",
            "1.0.0",
        ),
        patch.dict(
            sys.modules,
            {"a2a": MagicMock(), "a2a.types": mock_a2a_types},
        ),
        patch(
            "litellm.a2a_protocol.main.A2A_SDK_AVAILABLE",
            True,
        ),
    ):
        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        mock_fastapi_response = MagicMock()
        result = await invoke_agent_a2a(
            agent_id="agent-uuid",
            request=mock_request,
            fastapi_response=mock_fastapi_response,
            user_api_key_dict=mock_user_api_key_dict,
        )

    # Verify the forward helper was called (not the -32601 error path)
    mock_fwd.assert_called_once()
    call_kwargs = mock_fwd.call_args
    assert call_kwargs.kwargs["body"]["method"] == "tasks/get"
    assert call_kwargs.kwargs["body"]["params"] == {"id": "task-uuid"}


@pytest.mark.asyncio
async def test_forward_jsonrpc_to_agent_success():
    """
    _forward_jsonrpc_to_agent must POST the body to the agent URL and
    return its JSON response unchanged.
    """
    from fastapi.responses import JSONResponse

    from litellm.proxy.agent_endpoints.a2a_endpoints import _forward_jsonrpc_to_agent

    upstream_payload = {
        "jsonrpc": "2.0",
        "id": "r3",
        "result": {"kind": "task", "id": "t1", "status": {"state": "completed"}},
    }

    mock_http_response = MagicMock()
    mock_http_response.status_code = 200
    mock_http_response.json = MagicMock(return_value=upstream_payload)

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_http_response)

    mock_handler = MagicMock()
    mock_handler.client = mock_client

    from litellm.proxy.agent_endpoints.a2a_endpoints import _forward_jsonrpc_to_agent

    with patch(
        "litellm.proxy.agent_endpoints.a2a_endpoints.get_async_httpx_client",
        return_value=mock_handler,
    ):
        response = await _forward_jsonrpc_to_agent(
            agent_url="http://agent:9999",
            body={
                "jsonrpc": "2.0",
                "id": "r3",
                "method": "tasks/get",
                "params": {"id": "t1"},
            },
        )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 200
