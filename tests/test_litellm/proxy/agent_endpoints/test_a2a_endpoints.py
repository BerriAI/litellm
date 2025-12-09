"""
Mock tests for A2A endpoints.

Tests that invoke_agent_a2a properly integrates with add_litellm_data_to_request.
"""

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
    mock_request.json = AsyncMock(return_value={
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
    })

    mock_user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key",
        user_id="test-user",
        team_id="test-team",
    )

    # Patch at the source modules
    with patch(
        "litellm.proxy.agent_endpoints.a2a_endpoints._get_agent",
        return_value=mock_agent,
    ), patch(
        "litellm.proxy.litellm_pre_call_utils.add_litellm_data_to_request",
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
    ):
        from litellm.proxy.agent_endpoints.a2a_endpoints import invoke_agent_a2a

        mock_fastapi_response = MagicMock()

        result = await invoke_agent_a2a(
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
