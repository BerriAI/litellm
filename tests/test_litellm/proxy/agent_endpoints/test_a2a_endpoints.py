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
    with patch(
        "litellm.proxy.agent_endpoints.a2a_endpoints._get_agent",
        return_value=mock_agent,
    ), patch(
        "litellm.proxy.common_request_processing.add_litellm_data_to_request",
        side_effect=mock_add_litellm_data,
    ) as mock_add_data, patch(
        "litellm.a2a_protocol.create_a2a_client",
        new_callable=AsyncMock,
    ), patch(
        "litellm.a2a_protocol.asend_message",
        new_callable=AsyncMock,
        return_value=mock_response,
    ), patch(
        "litellm.proxy.proxy_server.general_settings",
        {},
    ), patch(
        "litellm.proxy.proxy_server.proxy_config",
        MagicMock(),
    ), patch(
        "litellm.proxy.proxy_server.version",
        "1.0.0",
    ), patch.dict(
        sys.modules,
        {"a2a": MagicMock(), "a2a.types": mock_a2a_types},
    ), patch(
        "litellm.a2a_protocol.main.A2A_SDK_AVAILABLE",
        True,
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
