"""
Test Vertex AI Live API Passthrough Feature

This module tests the Vertex AI Live API WebSocket passthrough functionality,
including the logging handler, cost tracking, and WebSocket message processing.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy._types import UserAPIKeyAuth


class TestVertexAILivePassthroughIntegration:
    """Integration tests for Vertex AI Live passthrough functionality"""

    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket for testing"""
        websocket = AsyncMock()
        websocket.headers = {"authorization": "Bearer test-token"}
        websocket.client_state = MagicMock()
        websocket.client_state.DISCONNECTED = "disconnected"
        return websocket

    @pytest.fixture
    def mock_user_api_key(self):
        """Create a mock user API key"""
        return UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            team_id="test-team",
            user_role="customer",
        )

    @pytest.fixture
    def mock_logging_obj(self):
        """Create a mock logging object"""
        mock = MagicMock(spec=LiteLLMLoggingObj)
        mock.model_call_details = {}
        mock.response_cost_calculator.return_value = None
        return mock

    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.websocket_passthrough_request")
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router")
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.vertex_llm_base._ensure_access_token_async")
    @patch("litellm.proxy.proxy_server.proxy_logging_obj")
    @pytest.mark.asyncio
    async def test_vertex_ai_live_websocket_passthrough_route(
        self,
        mock_proxy_logging_obj,
        mock_ensure_access_token,
        mock_router,
        mock_websocket_passthrough,
        mock_websocket,
        mock_user_api_key,
        mock_logging_obj,
    ):
        """Test the Vertex AI Live WebSocket passthrough route"""
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            vertex_ai_live_websocket_passthrough,
        )

        # Mock the router methods
        mock_router.get_vertex_credentials.return_value = MagicMock(
            vertex_project="test-project",
            vertex_location="us-central1",
            vertex_credentials="test-credentials",
        )
        mock_router.set_default_vertex_config.return_value = None

        # Mock the access token async call
        mock_ensure_access_token.return_value = ("test-access-token", "test-project")

        # Mock the WebSocket passthrough request - it returns None, not an AsyncMock
        mock_websocket_passthrough.return_value = None

        # Test the route
        result = await vertex_ai_live_websocket_passthrough(
            websocket=mock_websocket, user_api_key_dict=mock_user_api_key
        )

        # Verify that the WebSocket passthrough was called
        mock_websocket_passthrough.assert_called_once()

        # Check the call arguments
        call_args = mock_websocket_passthrough.call_args
        assert call_args[1]["websocket"] == mock_websocket
        assert call_args[1]["user_api_key_dict"] == mock_user_api_key
        assert call_args[1]["endpoint"] == "/vertex_ai/live"

        # The result should be None since websocket_passthrough_request returns None
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__])
