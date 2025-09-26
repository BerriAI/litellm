"""
Integration tests for Vertex AI Live API WebSocket passthrough

This module tests the end-to-end functionality of the Vertex AI Live API
WebSocket passthrough feature, including WebSocket connections, message
processing, and cost tracking.
"""

import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime
from typing import Dict, List, Any

import pytest
import httpx
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy.proxy_server import app
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_ai_live_passthrough_logging_handler import (
    VertexAILivePassthroughLoggingHandler,
)


class TestVertexAILivePassthroughIntegration:
    """Integration tests for Vertex AI Live passthrough"""

    @pytest.fixture
    def client(self):
        """Create a test client"""
        return TestClient(app)

    @pytest.fixture
    def mock_vertex_credentials(self):
        """Mock Vertex AI credentials"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            credentials = {
                "type": "service_account",
                "project_id": "test-project",
                "private_key_id": "test-key-id",
                "private_key": "-----BEGIN PRIVATE KEY-----\nMOCK_PRIVATE_KEY\n-----END PRIVATE KEY-----\n",
                "client_email": "test@test-project.iam.gserviceaccount.com",
                "client_id": "test-client-id",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
            json.dump(credentials, f)
            temp_file = f.name
        
        # Set environment variable
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file
        
        yield temp_file
        
        # Cleanup
        os.unlink(temp_file)
        if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

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
                    "text": "Hello! How can I help you today?",
                    "usage": {
                        "promptTokenCount": 15,
                        "candidatesTokenCount": 20,
                        "totalTokenCount": 35,
                        "promptTokensDetails": [
                            {"modality": "TEXT", "tokenCount": 15}
                        ],
                        "candidatesTokensDetails": [
                            {"modality": "TEXT", "tokenCount": 20}
                        ]
                    }
                }
            },
            {
                "type": "response.done",
                "event_id": "event-123",
                "response": {
                    "usage": {
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
            }
        ]

    def test_vertex_ai_live_route_registration(self, client):
        """Test that the Vertex AI Live route is properly registered"""
        # Check if the route exists in the app
        routes = [route.path for route in app.routes]
        assert "/vertex_ai/live" in routes

    @patch('litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.websocket_passthrough_request')
    @patch('litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router')
    def test_vertex_ai_live_websocket_connection(
        self, 
        mock_router, 
        mock_websocket_passthrough,
        client,
        mock_vertex_credentials
    ):
        """Test WebSocket connection to Vertex AI Live endpoint"""
        # Mock the router methods
        mock_router.get_vertex_credentials.return_value = MagicMock(
            vertex_project="test-project",
            vertex_location="us-central1",
            vertex_credentials="test-credentials"
        )
        mock_router.set_default_vertex_config.return_value = None
        
        # Mock the WebSocket passthrough request
        mock_websocket_passthrough.return_value = AsyncMock()
        
        # Test WebSocket connection
        with client.websocket_connect("/vertex_ai/live") as websocket:
            # Send a test message
            test_message = {
                "type": "session.create",
                "session": {
                    "modalities": ["TEXT"],
                    "instructions": "You are a helpful assistant."
                }
            }
            websocket.send_text(json.dumps(test_message))
            
            # The connection should be established without errors
            assert websocket is not None

    def test_vertex_ai_live_logging_handler_integration(self, sample_websocket_messages):
        """Test the logging handler with real WebSocket messages"""
        handler = VertexAILivePassthroughLoggingHandler()
        
        # Test usage metadata extraction
        usage_metadata = handler._extract_usage_metadata_from_websocket_messages(
            sample_websocket_messages
        )
        
        assert usage_metadata is not None
        assert usage_metadata["promptTokenCount"] == 20  # 15 + 5
        assert usage_metadata["candidatesTokenCount"] == 28  # 20 + 8
        assert usage_metadata["totalTokenCount"] == 48  # 35 + 13

    @patch('litellm.utils.get_model_info')
    def test_cost_calculation_integration(self, mock_get_model_info, sample_websocket_messages):
        """Test cost calculation with real usage data"""
        # Mock model info with realistic pricing
        mock_get_model_info.return_value = {
            "input_cost_per_token": 0.000001,
            "output_cost_per_token": 0.000002,
            "input_cost_per_audio_per_second": 0.0001,
            "output_cost_per_audio_per_second": 0.0002
        }
        
        handler = VertexAILivePassthroughLoggingHandler()
        
        # Extract usage metadata
        usage_metadata = handler._extract_usage_metadata_from_websocket_messages(
            sample_websocket_messages
        )
        
        # Calculate cost
        cost = handler._calculate_cost("gemini-1.5-pro", usage_metadata)
        
        # Verify cost calculation
        expected_cost = (20 * 0.000001) + (28 * 0.000002)
        assert cost == expected_cost
        assert cost > 0

    def test_multimodal_usage_tracking(self):
        """Test usage tracking with multiple modalities"""
        handler = VertexAILivePassthroughLoggingHandler()
        
        # Messages with mixed modalities
        multimodal_messages = [
            {
                "type": "response.create",
                "response": {
                    "usage": {
                        "promptTokenCount": 30,
                        "candidatesTokenCount": 25,
                        "totalTokenCount": 55,
                        "promptTokensDetails": [
                            {"modality": "TEXT", "tokenCount": 20},
                            {"modality": "AUDIO", "tokenCount": 10}
                        ],
                        "candidatesTokensDetails": [
                            {"modality": "TEXT", "tokenCount": 15},
                            {"modality": "AUDIO", "tokenCount": 10}
                        ]
                    }
                }
            }
        ]
        
        usage_metadata = handler._extract_usage_metadata_from_websocket_messages(
            multimodal_messages
        )
        
        assert usage_metadata is not None
        assert usage_metadata["promptTokenCount"] == 30
        assert usage_metadata["candidatesTokenCount"] == 25
        assert len(usage_metadata["promptTokensDetails"]) == 2
        assert len(usage_metadata["candidatesTokensDetails"]) == 2
        
        # Check modality details
        text_prompt = next(d for d in usage_metadata["promptTokensDetails"] if d["modality"] == "TEXT")
        audio_prompt = next(d for d in usage_metadata["promptTokensDetails"] if d["modality"] == "AUDIO")
        assert text_prompt["tokenCount"] == 20
        assert audio_prompt["tokenCount"] == 10

    def test_web_search_usage_tracking(self):
        """Test usage tracking with web search (tool use)"""
        handler = VertexAILivePassthroughLoggingHandler()
        
        # Messages with web search usage
        web_search_messages = [
            {
                "type": "response.create",
                "response": {
                    "usage": {
                        "promptTokenCount": 50,
                        "candidatesTokenCount": 30,
                        "totalTokenCount": 80,
                        "toolUsePromptTokenCount": 10,
                        "promptTokensDetails": [
                            {"modality": "TEXT", "tokenCount": 50}
                        ],
                        "candidatesTokensDetails": [
                            {"modality": "TEXT", "tokenCount": 30}
                        ]
                    }
                }
            }
        ]
        
        usage_metadata = handler._extract_usage_metadata_from_websocket_messages(
            web_search_messages
        )
        
        assert usage_metadata is not None
        assert usage_metadata["promptTokenCount"] == 50
        assert usage_metadata["candidatesTokenCount"] == 30
        assert usage_metadata["toolUsePromptTokenCount"] == 10

    @patch('litellm.utils.get_model_info')
    def test_web_search_cost_calculation(self, mock_get_model_info):
        """Test cost calculation with web search"""
        # Mock model info with web search pricing
        mock_get_model_info.return_value = {
            "input_cost_per_token": 0.000001,
            "output_cost_per_token": 0.000002,
            "web_search_cost_per_request": 0.01
        }
        
        handler = VertexAILivePassthroughLoggingHandler()
        
        usage_metadata = {
            "promptTokenCount": 100,
            "candidatesTokenCount": 50,
            "totalTokenCount": 150,
            "toolUsePromptTokenCount": 10
        }
        
        cost = handler._calculate_cost("gemini-1.5-pro", usage_metadata)
        
        # Should include web search cost
        expected_base_cost = (100 * 0.000001) + (50 * 0.000002)
        expected_web_search_cost = 0.01
        expected_total = expected_base_cost + expected_web_search_cost
        assert cost == expected_total

    def test_error_handling_invalid_messages(self):
        """Test error handling with invalid message formats"""
        handler = VertexAILivePassthroughLoggingHandler()
        
        # Test with various invalid message formats
        invalid_messages = [
            "not a dict",
            {"type": "invalid", "data": "incomplete"},
            None,
            [],
            {"type": "response.create"},  # Missing response field
            {"type": "response.create", "response": {}}  # Empty response
        ]
        
        # Should handle all cases gracefully
        for messages in invalid_messages:
            result = handler._extract_usage_metadata_from_websocket_messages(messages)
            assert result is None

    def test_empty_websocket_messages(self):
        """Test handling of empty WebSocket messages"""
        handler = VertexAILivePassthroughLoggingHandler()
        
        # Test with empty list
        result = handler._extract_usage_metadata_from_websocket_messages([])
        assert result is None
        
        # Test with None
        result = handler._extract_usage_metadata_from_websocket_messages(None)
        assert result is None

    @patch('litellm.utils.get_model_info')
    def test_missing_model_info_handling(self, mock_get_model_info):
        """Test handling when model info is missing or incomplete"""
        handler = VertexAILivePassthroughLoggingHandler()
        
        # Test with empty model info
        mock_get_model_info.return_value = {}
        
        usage_metadata = {
            "promptTokenCount": 100,
            "candidatesTokenCount": 50,
            "totalTokenCount": 150
        }
        
        cost = handler._calculate_cost("unknown-model", usage_metadata)
        assert cost == 0.0
        
        # Test with partial model info
        mock_get_model_info.return_value = {
            "input_cost_per_token": 0.000001
            # Missing output_cost_per_token
        }
        
        cost = handler._calculate_cost("partial-model", usage_metadata)
        # Should still calculate with available info
        assert cost >= 0

    def test_handler_with_mock_logging_obj(self, sample_websocket_messages):
        """Test the main handler method with a mock logging object"""
        handler = VertexAILivePassthroughLoggingHandler()
        mock_logging_obj = MagicMock()
        
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
        
        # Verify result structure
        assert "result" in result
        assert "kwargs" in result
        
        result_data = result["result"]
        assert "model" in result_data
        assert "usage" in result_data
        assert "choices" in result_data
        
        # Verify usage data
        usage = result_data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        
        # Verify aggregated usage
        assert usage["prompt_tokens"] == 20  # 15 + 5
        assert usage["completion_tokens"] == 28  # 20 + 8
        assert usage["total_tokens"] == 48  # 35 + 13


class TestVertexAILivePassthroughEndToEnd:
    """End-to-end tests for Vertex AI Live passthrough"""

    @pytest.fixture
    def mock_vertex_ai_live_api(self):
        """Mock the Vertex AI Live API responses"""
        with patch('websockets.asyncio.client.connect') as mock_connect:
            # Mock WebSocket connection
            mock_websocket = AsyncMock()
            mock_websocket.recv.side_effect = [
                json.dumps({
                    "type": "session.created",
                    "session": {"id": "test-session"}
                }),
                json.dumps({
                    "type": "response.create",
                    "response": {
                        "text": "Hello! How can I help you?",
                        "usage": {
                            "promptTokenCount": 10,
                            "candidatesTokenCount": 15,
                            "totalTokenCount": 25
                        }
                    }
                }),
                json.dumps({
                    "type": "response.done",
                    "response": {
                        "usage": {
                            "promptTokenCount": 5,
                            "candidatesTokenCount": 8,
                            "totalTokenCount": 13
                        }
                    }
                })
            ]
            mock_websocket.send = AsyncMock()
            mock_websocket.close = AsyncMock()
            
            mock_connect.return_value = mock_websocket
            yield mock_connect

    @pytest.mark.asyncio
    async def test_websocket_passthrough_flow(self, mock_vertex_ai_live_api):
        """Test the complete WebSocket passthrough flow"""
        from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
            websocket_passthrough_request
        )
        
        # Mock dependencies
        mock_websocket = MagicMock()
        mock_websocket.headers = {"authorization": "Bearer test-token"}
        mock_websocket.client_state = MagicMock()
        mock_websocket.client_state.DISCONNECTED = "disconnected"
        
        mock_user_api_key = MagicMock()
        mock_logging_obj = MagicMock()
        
        # Test the WebSocket passthrough
        await websocket_passthrough_request(
            websocket=mock_websocket,
            target="wss://test-vertex-ai-live-api.com/v1/stream",
            custom_headers={"Authorization": "Bearer test-token"},
            user_api_key_dict=mock_user_api_key,
            forward_headers=False,
            endpoint="/vertex_ai/live",
            accept_websocket=True,
            logging_obj=mock_logging_obj
        )
        
        # Verify that the WebSocket connection was established
        mock_vertex_ai_live_api.assert_called_once()

    def test_route_detection_in_success_handler(self):
        """Test that the success handler correctly detects Vertex AI Live routes"""
        from litellm.proxy.pass_through_endpoints.success_handler import (
            PassThroughEndpointLogging
        )
        
        handler = PassThroughEndpointLogging()
        
        # Test various route patterns
        test_routes = [
            "/vertex_ai/live",
            "/vertex_ai/live/",
            "/vertex_ai/live/stream",
            "/vertex_ai/live/chat",
            "/vertex_ai/live/v1/stream"
        ]
        
        for route in test_routes:
            assert handler.is_vertex_ai_live_route(route), f"Route {route} should be detected as Vertex AI Live"
        
        # Test non-Vertex AI Live routes
        non_live_routes = [
            "/vertex_ai",
            "/vertex_ai/discovery",
            "/vertex_ai/aiplatform",
            "/openai/chat/completions",
            "/anthropic/messages"
        ]
        
        for route in non_live_routes:
            assert not handler.is_vertex_ai_live_route(route), f"Route {route} should not be detected as Vertex AI Live"


if __name__ == "__main__":
    pytest.main([__file__])
