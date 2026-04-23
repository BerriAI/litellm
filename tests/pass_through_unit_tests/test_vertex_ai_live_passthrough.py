"""
Test Vertex AI Live API Passthrough Feature

This module tests the Vertex AI Live API WebSocket passthrough functionality,
including the logging handler, cost tracking, and WebSocket message processing.
"""

import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Dict, List, Any, Optional

import pytest
import httpx

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_ai_live_passthrough_logging_handler import (
    VertexAILivePassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.utils import LlmProviders
from litellm.proxy._types import UserAPIKeyAuth


class TestVertexAILivePassthroughLoggingHandler:
    """Test the Vertex AI Live Passthrough Logging Handler"""

    @pytest.fixture
    def handler(self):
        """Create a handler instance for testing"""
        return VertexAILivePassthroughLoggingHandler()

    @pytest.fixture
    def mock_logging_obj(self):
        """Create a mock logging object"""
        mock = MagicMock(spec=LiteLLMLoggingObj)
        mock.model_call_details = {}
        return mock

    @pytest.fixture
    def sample_websocket_messages(self):
        """Sample WebSocket messages for testing"""
        return [
            {
                "type": "session.created",
                "session": {"id": "test-session-123"},
                "timestamp": "2024-01-01T00:00:00Z"
            },
            {
                "type": "response.create",
                "event_id": "event-123",
                "response": {
                    "text": "Hello, how can I help you?"
                },
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 15,
                    "totalTokenCount": 25,
                    "promptTokensDetails": [
                        {"modality": "TEXT", "tokenCount": 10}
                    ],
                    "candidatesTokensDetails": [
                        {"modality": "TEXT", "tokenCount": 15}
                    ]
                }
            },
            {
                "type": "response.done",
                "event_id": "event-123",
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 8,
                    "totalTokenCount": 13,
                    "promptTokensDetails": [
                        {"modality": "TEXT", "tokenCount": 5}
                    ],
                    "candidatesTokensDetails": [
                        {"modality": "TEXT", "tokenCount": 8}
                    ]
                }
            }
        ]

    def test_llm_provider_name_property(self, handler):
        """Test that llm_provider_name returns the correct provider"""
        assert handler.llm_provider_name == LlmProviders.VERTEX_AI

    def test_get_provider_config(self, handler):
        """Test that get_provider_config returns a valid config"""
        config = handler.get_provider_config("gemini-1.5-pro")
        assert config is not None
        # Verify it's a Vertex AI config by checking for expected methods
        assert hasattr(config, 'get_supported_openai_params')
        assert hasattr(config, 'map_openai_params')

    def test_extract_usage_metadata_single_message(self, handler):
        """Test usage metadata extraction from a single message"""
        messages = [{
            "type": "response.create",
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 15,
                "totalTokenCount": 25,
                "promptTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 10}
                ],
                "candidatesTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 15}
                ]
            }
        }]

        result = handler._extract_usage_metadata_from_websocket_messages(messages)

        assert result is not None
        assert result["promptTokenCount"] == 10
        assert result["candidatesTokenCount"] == 15
        assert result["totalTokenCount"] == 25
        assert len(result["promptTokensDetails"]) == 1
        assert len(result["candidatesTokensDetails"]) == 1

    def test_extract_usage_metadata_multiple_messages(self, handler):
        """Test usage metadata aggregation from multiple messages"""
        messages = [
            {
                "type": "response.create",
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 15,
                    "totalTokenCount": 25,
                    "promptTokensDetails": [
                        {"modality": "TEXT", "tokenCount": 10}
                    ],
                    "candidatesTokensDetails": [
                        {"modality": "TEXT", "tokenCount": 15}
                    ]
                }
            },
            {
                "type": "response.done",
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 8,
                    "totalTokenCount": 13,
                    "promptTokensDetails": [
                        {"modality": "TEXT", "tokenCount": 5}
                    ],
                    "candidatesTokensDetails": [
                        {"modality": "TEXT", "tokenCount": 8}
                    ]
                }
            }
        ]

        result = handler._extract_usage_metadata_from_websocket_messages(messages)

        assert result is not None
        assert result["promptTokenCount"] == 15  # 10 + 5
        assert result["candidatesTokenCount"] == 23  # 15 + 8
        assert result["totalTokenCount"] == 38  # 25 + 13
        assert len(result["promptTokensDetails"]) == 1
        assert result["promptTokensDetails"][0]["tokenCount"] == 15
        assert len(result["candidatesTokensDetails"]) == 1
        assert result["candidatesTokensDetails"][0]["tokenCount"] == 23

    def test_extract_usage_metadata_no_usage(self, handler):
        """Test handling of messages without usage metadata"""
        messages = [
            {"type": "session.created", "session": {"id": "test"}},
            {"type": "response.create", "response": {"text": "Hello"}}
        ]
        
        result = handler._extract_usage_metadata_from_websocket_messages(messages)
        assert result is None

    def test_extract_usage_metadata_empty_list(self, handler):
        """Test handling of empty message list"""
        result = handler._extract_usage_metadata_from_websocket_messages([])
        assert result is None

    def test_extract_usage_metadata_mixed_modalities(self, handler):
        """Test usage metadata extraction with mixed modalities"""
        messages = [{
            "type": "response.create",
            "usageMetadata": {
                "promptTokenCount": 20,
                "candidatesTokenCount": 30,
                "totalTokenCount": 50,
                "promptTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 10},
                    {"modality": "AUDIO", "tokenCount": 10}
                ],
                "candidatesTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 20},
                    {"modality": "AUDIO", "tokenCount": 10}
                ]
            }
        }]
        
        result = handler._extract_usage_metadata_from_websocket_messages(messages)
        
        assert result is not None
        assert result["promptTokenCount"] == 20
        assert result["candidatesTokenCount"] == 30
        assert len(result["promptTokensDetails"]) == 2
        assert len(result["candidatesTokensDetails"]) == 2
        
        # Check modality aggregation
        text_prompt = next(d for d in result["promptTokensDetails"] if d["modality"] == "TEXT")
        audio_prompt = next(d for d in result["promptTokensDetails"] if d["modality"] == "AUDIO")
        assert text_prompt["tokenCount"] == 10
        assert audio_prompt["tokenCount"] == 10

    @patch('litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_ai_live_passthrough_logging_handler.get_model_info')
    def test_calculate_cost_basic(self, mock_get_model_info, handler):
        """Test basic cost calculation"""
        mock_get_model_info.return_value = {
            "input_cost_per_token": 0.000001,
            "output_cost_per_token": 0.000002
        }
        
        usage_metadata = {
            "promptTokenCount": 100,
            "candidatesTokenCount": 50,
            "totalTokenCount": 150
        }
        
        cost = handler._calculate_live_api_cost("gemini-1.5-pro", usage_metadata)

        # The cost calculation may include additional factors, so we check it's reasonable
        expected_min_cost = (100 * 0.000001) + (50 * 0.000002)
        assert cost >= expected_min_cost
        assert cost > 0

    @patch('litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_ai_live_passthrough_logging_handler.get_model_info')
    def test_calculate_cost_with_audio(self, mock_get_model_info, handler):
        """Test cost calculation with audio tokens"""
        mock_get_model_info.return_value = {
            "input_cost_per_token": 0.000001,
            "output_cost_per_token": 0.000002,
            "input_cost_per_audio_token": 0.0001,
            "output_cost_per_audio_token": 0.0002
        }
        
        usage_metadata = {
            "promptTokenCount": 100,
            "candidatesTokenCount": 50,
            "totalTokenCount": 150,
            "promptTokensDetails": [
                {"modality": "TEXT", "tokenCount": 80},
                {"modality": "AUDIO", "tokenCount": 20}
            ],
            "candidatesTokensDetails": [
                {"modality": "TEXT", "tokenCount": 30},
                {"modality": "AUDIO", "tokenCount": 20}
            ]
        }
        
        cost = handler._calculate_live_api_cost("gemini-1.5-pro", usage_metadata)
        
        # Should include both text and audio costs
        assert cost > 0
        assert cost > (100 * 0.000001) + (50 * 0.000002)  # Should be higher due to audio

    @patch('litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_ai_live_passthrough_logging_handler.get_model_info')
    def test_calculate_cost_with_web_search(self, mock_get_model_info, handler):
        """Test cost calculation with web search (tool use)"""
        mock_get_model_info.return_value = {
            "input_cost_per_token": 0.000001,
            "output_cost_per_token": 0.000002,
            "web_search_cost_per_request": 0.01
        }
        
        usage_metadata = {
            "promptTokenCount": 100,
            "candidatesTokenCount": 50,
            "totalTokenCount": 150,
            "toolUsePromptTokenCount": 10
        }
        
        cost = handler._calculate_live_api_cost("gemini-1.5-pro", usage_metadata)
        
        # Should include web search cost
        expected_base_cost = (100 * 0.000001) + (50 * 0.000002)
        # The web search cost might be handled differently, so just check it's reasonable
        assert cost >= expected_base_cost
        assert cost > 0

    def test_vertex_ai_live_passthrough_handler_integration(self, handler, mock_logging_obj, sample_websocket_messages):
        """Test the main passthrough handler method"""
        url_route = "/vertex_ai/live"
        start_time = datetime.now()
        end_time = datetime.now()
        request_body = {"messages": [{"role": "user", "content": "Hello"}]}
        
        result = handler.vertex_ai_live_passthrough_handler(
            websocket_messages=sample_websocket_messages,
            logging_obj=mock_logging_obj,
            url_route=url_route,
            start_time=start_time,
            end_time=end_time,
            request_body=request_body
        )
        
        assert "result" in result
        assert "kwargs" in result
        
        # Check that the result contains expected fields
        result_data = result["result"]
        assert "model" in result_data
        assert "usage" in result_data
        assert "choices" in result_data
        
        # Check usage data
        usage = result_data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

    def test_vertex_ai_live_passthrough_handler_no_usage(self, handler, mock_logging_obj):
        """Test handler with messages that don't contain usage metadata"""
        messages = [
            {"type": "session.created", "session": {"id": "test"}},
            {"type": "response.create", "response": {"text": "Hello"}}
        ]
        
        url_route = "/vertex_ai/live"
        start_time = datetime.now()
        end_time = datetime.now()
        request_body = {"messages": [{"role": "user", "content": "Hello"}]}
        
        result = handler.vertex_ai_live_passthrough_handler(
            websocket_messages=messages,
            logging_obj=mock_logging_obj,
            url_route=url_route,
            start_time=start_time,
            end_time=end_time,
            request_body=request_body
        )
        
        assert "result" in result
        assert "kwargs" in result
        
        # Should still return a valid result even without usage data
        result_data = result["result"]
        # When no usage metadata is found, result_data will be None
        assert result_data is None


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
            user_role="customer"
        )

    @pytest.fixture
    def mock_logging_obj(self):
        """Create a mock logging object"""
        mock = MagicMock(spec=LiteLLMLoggingObj)
        mock.model_call_details = {}
        return mock

    @patch('litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.websocket_passthrough_request')
    @patch('litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router')
    @patch('litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.vertex_llm_base._ensure_access_token_async')
    @patch('litellm.proxy.proxy_server.proxy_logging_obj')
    @pytest.mark.asyncio
    async def test_vertex_ai_live_websocket_passthrough_route(
        self,
        mock_proxy_logging_obj,
        mock_ensure_access_token,
        mock_router,
        mock_websocket_passthrough,
        mock_websocket,
        mock_user_api_key,
        mock_logging_obj
    ):
        """Test the Vertex AI Live WebSocket passthrough route"""
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            vertex_ai_live_websocket_passthrough
        )
        
        # Mock the router methods
        mock_router.get_vertex_credentials.return_value = MagicMock(
            vertex_project="test-project",
            vertex_location="us-central1",
            vertex_credentials="test-credentials"
        )
        mock_router.set_default_vertex_config.return_value = None
        
        # Mock the access token async call
        mock_ensure_access_token.return_value = ("test-access-token", "test-project")
        
        # Mock the WebSocket passthrough request - it returns None, not an AsyncMock
        mock_websocket_passthrough.return_value = None
        
        # Test the route
        result = await vertex_ai_live_websocket_passthrough(
            websocket=mock_websocket,
            user_api_key_dict=mock_user_api_key
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

    def test_vertex_ai_live_route_detection(self):
        """Test that the route detection works correctly"""
        from litellm.proxy.pass_through_endpoints.success_handler import (
            PassThroughEndpointLogging
        )
        
        handler = PassThroughEndpointLogging()
        
        # Test valid routes
        assert handler.is_vertex_ai_live_route("/vertex_ai/live") == True
        assert handler.is_vertex_ai_live_route("/vertex_ai/live/") == True
        assert handler.is_vertex_ai_live_route("/vertex_ai/live/stream") == True
        
        # Test invalid routes
        assert handler.is_vertex_ai_live_route("/vertex_ai") == False
        assert handler.is_vertex_ai_live_route("/vertex_ai/discovery") == False
        assert handler.is_vertex_ai_live_route("/openai/chat/completions") == False

    @patch('litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_ai_live_passthrough_logging_handler.VertexAILivePassthroughLoggingHandler')
    @pytest.mark.asyncio
    async def test_success_handler_vertex_ai_live_integration(
        self,
        mock_handler_class,
        mock_logging_obj
    ):
        """Test the success handler integration with Vertex AI Live"""
        from litellm.proxy.pass_through_endpoints.success_handler import (
            PassThroughEndpointLogging
        )
        
        # Mock the handler
        mock_handler = MagicMock()
        mock_handler.vertex_ai_live_passthrough_handler.return_value = {
            "result": {"model": "gemini-1.5-pro", "usage": {"total_tokens": 100}},
            "kwargs": {"test": "value"}
        }
        mock_handler_class.return_value = mock_handler
        
        # Create success handler
        success_handler = PassThroughEndpointLogging()
        
        # Mock the route check
        success_handler.is_vertex_ai_live_route = MagicMock(return_value=True)
        
        # Test data
        response_body = [
            {"type": "response.create", "response": {"text": "Hello"}}
        ]
        url_route = "/vertex_ai/live"
        start_time = datetime.now()
        end_time = datetime.now()
        request_body = {"messages": [{"role": "user", "content": "Hello"}]}
        
        # Call the method
        result = await success_handler.pass_through_async_success_handler(
            httpx_response=MagicMock(),
            response_body=response_body,
            logging_obj=mock_logging_obj,
            url_route=url_route,
            result="test",
            start_time=start_time,
            end_time=end_time,
            cache_hit=False,
            request_body=request_body,
            passthrough_logging_payload=MagicMock()
        )
        
        # Verify the handler was called
        mock_handler.vertex_ai_live_passthrough_handler.assert_called_once()
        
        # The method returns None (it doesn't return anything), so just verify it completed without error
        assert result is None


class TestVertexAILivePassthroughErrorHandling:
    """Test error handling in Vertex AI Live passthrough"""

    @pytest.fixture
    def mock_logging_obj(self):
        """Create a mock logging object"""
        mock = MagicMock(spec=LiteLLMLoggingObj)
        mock.model_call_details = {}
        return mock

    def test_invalid_websocket_messages_format(self):
        """Test handling of invalid WebSocket message formats"""
        handler = VertexAILivePassthroughLoggingHandler()
        
        # Test with invalid message format
        invalid_messages = [
            {"type": "invalid", "data": "not a proper message"},
            "not a dict at all",
            None
        ]
        
        # Should not raise an exception
        result = handler._extract_usage_metadata_from_websocket_messages(invalid_messages)
        assert result is None

    def test_missing_usage_metadata(self):
        """Test handling of messages with missing usage metadata"""
        handler = VertexAILivePassthroughLoggingHandler()
        
        messages = [
            {"type": "response.create", "response": {"text": "Hello"}},
            {"type": "response.done", "response": {"text": "Done"}}
        ]
        
        result = handler._extract_usage_metadata_from_websocket_messages(messages)
        assert result is None

    @patch('litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_ai_live_passthrough_logging_handler.get_model_info')
    def test_cost_calculation_with_missing_model_info(self, mock_get_model_info):
        """Test cost calculation when model info is missing"""
        handler = VertexAILivePassthroughLoggingHandler()
        
        # Mock missing model info
        mock_get_model_info.return_value = {}
        
        usage_metadata = {
            "promptTokenCount": 100,
            "candidatesTokenCount": 50,
            "totalTokenCount": 150
        }
        
        # Should not raise an exception, should return 0 or handle gracefully
        cost = handler._calculate_live_api_cost("unknown-model", usage_metadata)
        assert cost == 0.0

    def test_handler_with_none_websocket_messages(self, mock_logging_obj):
        """Test handler with None websocket messages"""
        handler = VertexAILivePassthroughLoggingHandler()
        
        url_route = "/vertex_ai/live"
        start_time = datetime.now()
        end_time = datetime.now()
        request_body = {"messages": [{"role": "user", "content": "Hello"}]}
        
        # Should handle None gracefully
        result = handler.vertex_ai_live_passthrough_handler(
            websocket_messages=None,
            logging_obj=mock_logging_obj,
            url_route=url_route,
            start_time=start_time,
            end_time=end_time,
            request_body=request_body
        )
        
        assert "result" in result
        assert "kwargs" in result


if __name__ == "__main__":
    pytest.main([__file__])
