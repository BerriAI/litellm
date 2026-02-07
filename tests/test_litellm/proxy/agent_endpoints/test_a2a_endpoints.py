"""
Mock tests for A2A endpoints.

Tests that invoke_agent_a2a properly integrates with ProxyBaseLLMRequestProcessing
for adding litellm data to requests.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest


@pytest.mark.asyncio
async def test_invoke_agent_a2a_adds_litellm_data():
    """
    Test that invoke_agent_a2a calls common_processing_pre_call_logic
    and the resulting data includes proxy_server_request.
    """
    from litellm.proxy._types import UserAPIKeyAuth

    # Track calls to common_processing_pre_call_logic
    processing_call_args = {}
    returned_data = {}

    async def mock_common_processing(
        request,
        general_settings,
        user_api_key_dict,
        proxy_logging_obj,
        proxy_config,
        route_type,
        version,
    ):
        # Capture the actual arguments passed to common_processing_pre_call_logic
        processing_call_args["request"] = request
        processing_call_args["general_settings"] = general_settings
        processing_call_args["user_api_key_dict"] = user_api_key_dict
        processing_call_args["route_type"] = route_type
        processing_call_args["version"] = version
        
        # Get the data from the processor instance
        data = mock_processor_instance.data
        
        # Simulate what common_processing_pre_call_logic does
        data["proxy_server_request"] = {
            "url": "http://localhost:4000/a2a/test-agent",
            "method": "POST",
            "headers": {},
            "body": dict(data),
        }
        
        # Store the returned data to verify endpoint uses it
        returned_data.update(data)
        mock_logging_obj = MagicMock()
        return data, mock_logging_obj

    # Mock response from asend_message
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "jsonrpc": "2.0",
        "id": "test-id",
        "result": {"status": "success"},
    }

    # Track what gets passed to asend_message
    asend_message_call_args = {}

    async def mock_asend_message(*args, **kwargs):
        asend_message_call_args["args"] = args
        asend_message_call_args["kwargs"] = kwargs
        return mock_response

    # Mock agent
    mock_agent = MagicMock()
    mock_agent.agent_card_params = {
        "url": "http://backend-agent:10001",
        "name": "Test Agent",
    }
    mock_agent.litellm_params = {}
    mock_agent.agent_id = "test-agent-id"

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
    try:
        from a2a.types import (
            MessageSendParams,
            SendMessageRequest,
            SendStreamingMessageRequest,
        )
    except ImportError:
        def make_mock_pydantic_class(name):
            """Create a mock class that behaves like a Pydantic model."""

            class MockPydanticClass:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)
                    self._kwargs = kwargs

                def model_dump(self, mode="json", exclude_none=False):
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

    # Create mock processor instance to capture data
    mock_processor_instance = MagicMock()
    mock_processor_instance.common_processing_pre_call_logic = AsyncMock(
        side_effect=mock_common_processing
    )

    def mock_processor_init(data):
        mock_processor_instance.data = data
        return mock_processor_instance

    # Patch at the source modules
    with patch(
        "litellm.proxy.agent_endpoints.a2a_endpoints._get_agent",
        return_value=mock_agent,
    ), patch(
        "litellm.proxy.agent_endpoints.a2a_endpoints.AgentRequestHandler.is_agent_allowed",
        new_callable=AsyncMock,
        return_value=True,
    ), patch(
        "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing",
        side_effect=mock_processor_init,
    ) as mock_processor_class, patch(
        "litellm.a2a_protocol.create_a2a_client",
        new_callable=AsyncMock,
    ), patch(
        "litellm.a2a_protocol.asend_message",
        new_callable=AsyncMock,
        side_effect=mock_asend_message,
    ), patch(
        "litellm.proxy.proxy_server.general_settings",
        {},
    ), patch(
        "litellm.proxy.proxy_server.proxy_config",
        MagicMock(),
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj",
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

        # Verify ProxyBaseLLMRequestProcessing was instantiated with data dict
        mock_processor_class.assert_called_once()
        init_call_args = mock_processor_class.call_args
        assert isinstance(init_call_args[0][0], dict), "Processor should be initialized with a dict"

        # Verify common_processing_pre_call_logic was called
        mock_processor_instance.common_processing_pre_call_logic.assert_called_once()
        
        # Verify the call included correct route_type and version
        assert processing_call_args.get("route_type") == "a2a_request"
        assert processing_call_args.get("version") == "1.0.0"

        # Verify model and custom_llm_provider were set in the data
        assert returned_data.get("model") == "a2a_agent/Test Agent"
        assert returned_data.get("custom_llm_provider") == "a2a_agent"

        # Verify proxy_server_request was added by common_processing_pre_call_logic
        assert "proxy_server_request" in returned_data
        assert returned_data["proxy_server_request"]["method"] == "POST"
        
        # Verify the data with proxy_server_request is what gets passed downstream
        # (The endpoint should use the returned data from common_processing_pre_call_logic)
        assert "metadata" in asend_message_call_args.get("kwargs", {}) or \
               any("proxy_server_request" in str(arg) for arg in asend_message_call_args.get("args", [])), \
               "Data from common_processing_pre_call_logic should be passed to asend_message"
