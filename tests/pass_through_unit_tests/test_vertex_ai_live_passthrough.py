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
                "timestamp": "2024-01-01T00:00:00Z",
            },
            {
                "type": "response.create",
                "event_id": "event-123",
                "response": {"text": "Hello, how can I help you?"},
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 15,
                    "totalTokenCount": 25,
                    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 10}],
                    "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 15}],
                },
            },
            {
                "type": "response.done",
                "event_id": "event-123",
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 8,
                    "totalTokenCount": 13,
                    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 5}],
                    "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 8}],
                },
            },
        ]

    def test_llm_provider_name_property(self, handler):
        """Test that llm_provider_name returns the correct provider"""
        assert handler.llm_provider_name == LlmProviders.VERTEX_AI

    def test_get_provider_config(self, handler):
        """Test that get_provider_config returns a valid config"""
        config = handler.get_provider_config("gemini-1.5-pro")
        assert config is not None
        # Verify it's a Vertex AI config by checking for expected methods
        assert hasattr(config, "get_supported_openai_params")
        assert hasattr(config, "map_openai_params")

    def test_extract_usage_metadata_single_message(self, handler):
        """Test usage metadata extraction from a single message"""
        messages = [
            {
                "type": "response.create",
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 15,
                    "totalTokenCount": 25,
                    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 10}],
                    "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 15}],
                },
            }
        ]

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
                    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 10}],
                    "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 15}],
                },
            },
            {
                "type": "response.done",
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 8,
                    "totalTokenCount": 13,
                    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 5}],
                    "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 8}],
                },
            },
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
            {"type": "response.create", "response": {"text": "Hello"}},
        ]

        result = handler._extract_usage_metadata_from_websocket_messages(messages)
        assert result is None

    def test_extract_usage_metadata_empty_list(self, handler):
        """Test handling of empty message list"""
        result = handler._extract_usage_metadata_from_websocket_messages([])
        assert result is None

    def test_extract_usage_metadata_mixed_modalities(self, handler):
        """Test usage metadata extraction with mixed modalities"""
        messages = [
            {
                "type": "response.create",
                "usageMetadata": {
                    "promptTokenCount": 20,
                    "candidatesTokenCount": 30,
                    "totalTokenCount": 50,
                    "promptTokensDetails": [
                        {"modality": "TEXT", "tokenCount": 10},
                        {"modality": "AUDIO", "tokenCount": 10},
                    ],
                    "candidatesTokensDetails": [
                        {"modality": "TEXT", "tokenCount": 20},
                        {"modality": "AUDIO", "tokenCount": 10},
                    ],
                },
            }
        ]

        result = handler._extract_usage_metadata_from_websocket_messages(messages)

        assert result is not None
        assert result["promptTokenCount"] == 20
        assert result["candidatesTokenCount"] == 30
        assert len(result["promptTokensDetails"]) == 2
        assert len(result["candidatesTokensDetails"]) == 2

        # Check modality aggregation
        text_prompt = next(
            d for d in result["promptTokensDetails"] if d["modality"] == "TEXT"
        )
        audio_prompt = next(
            d for d in result["promptTokensDetails"] if d["modality"] == "AUDIO"
        )
        assert text_prompt["tokenCount"] == 10
        assert audio_prompt["tokenCount"] == 10

    def test_usage_carries_every_modality(self, handler):
        """Regression: the Usage object reported only TEXT, so audio/image billed as nothing.

        prompt_tokens must be the full count and the details must name each modality,
        because the cost calculator prices audio and image from *_tokens_details.
        """
        usage_metadata = {
            "promptTokenCount": 1300,
            "candidatesTokenCount": 124,
            "totalTokenCount": 1424,
            "promptTokensDetails": [
                {"modality": "TEXT", "tokenCount": 13},
                {"modality": "AUDIO", "tokenCount": 127},
                {"modality": "IMAGE", "tokenCount": 1160},
            ],
            "candidatesTokensDetails": [
                {"modality": "TEXT", "tokenCount": 29},
                {"modality": "AUDIO", "tokenCount": 95},
            ],
        }

        usage = handler._create_usage_object_from_metadata(
            usage_metadata=usage_metadata, model="gemini-live-2.5-flash"
        )

        assert usage.prompt_tokens == 1300, "the full prompt count must survive, not just its text share"
        assert usage.completion_tokens == 124
        assert usage.prompt_tokens_details.text_tokens == 13
        assert usage.prompt_tokens_details.audio_tokens == 127
        assert usage.prompt_tokens_details.image_tokens == 1160
        assert usage.completion_tokens_details.text_tokens == 29
        assert usage.completion_tokens_details.audio_tokens == 95

    def test_usage_sums_repeated_modality_entries(self, handler):
        """A modality may appear more than once across aggregated turns; sum, don't overwrite."""
        usage = handler._create_usage_object_from_metadata(
            usage_metadata={
                "promptTokenCount": 40,
                "candidatesTokenCount": 0,
                "promptTokensDetails": [
                    {"modality": "IMAGE", "tokenCount": 10},
                    {"modality": "IMAGE", "tokenCount": 25},
                    {"modality": "TEXT", "tokenCount": 5},
                ],
            },
            model="gemini-live-2.5-flash",
        )
        assert usage.prompt_tokens_details.image_tokens == 35
        assert usage.prompt_tokens_details.text_tokens == 5

    @pytest.mark.parametrize(
        "label,prompt_details,candidate_details",
        [
            ("text only", [("TEXT", 6)], [("TEXT", 2)]),
            ("audio in", [("TEXT", 13), ("AUDIO", 127)], [("TEXT", 18)]),
            ("image in", [("TEXT", 10), ("IMAGE", 258)], [("TEXT", 24)]),
            ("frames in", [("TEXT", 11), ("IMAGE", 1032)], [("TEXT", 26)]),
            ("audio both ways", [("TEXT", 13), ("AUDIO", 127)], [("TEXT", 29), ("AUDIO", 95)]),
        ],
    )
    def test_live_session_bills_each_modality_at_its_own_rate(self, handler, label, prompt_details, candidate_details):
        """Every payload here is a real Vertex Live session's usageMetadata.

        Before the fix these billed the text share only, from 1x (text) to 55x under.
        The expected amount is derived from the entry's own rates rather than hardcoded,
        so this stays correct as prices move, and it is asserted exactly, so dropping a
        modality and double-charging one both fail.
        """
        from litellm.cost_calculator import completion_cost
        from litellm.types.utils import ModelResponse
        from litellm.utils import get_model_info

        model = "gemini-live-2.5-flash-preview-native-audio-09-2025"
        info = get_model_info(model=model, custom_llm_provider="vertex_ai")

        text_in = info["input_cost_per_token"]
        audio_in = info.get("input_cost_per_audio_token") or text_in
        image_in = info.get("input_cost_per_image_token") or text_in
        text_out = info["output_cost_per_token"]
        audio_out = info.get("output_cost_per_audio_token") or text_out
        rate_in = {"TEXT": text_in, "AUDIO": audio_in, "IMAGE": image_in}
        rate_out = {"TEXT": text_out, "AUDIO": audio_out}

        expected = sum(c * rate_in[m] for m, c in prompt_details) + sum(c * rate_out[m] for m, c in candidate_details)

        usage = handler._create_usage_object_from_metadata(
            usage_metadata={
                "promptTokenCount": sum(c for _, c in prompt_details),
                "candidatesTokenCount": sum(c for _, c in candidate_details),
                "promptTokensDetails": [{"modality": m, "tokenCount": c} for m, c in prompt_details],
                "candidatesTokensDetails": [{"modality": m, "tokenCount": c} for m, c in candidate_details],
            },
            model=model,
        )

        cost = completion_cost(
            completion_response=ModelResponse(
                id="x", object="chat.completion", created=0, model=model, usage=usage, choices=[]
            ),
            model=f"vertex_ai/{model}",
            custom_llm_provider="vertex_ai",
            call_type="acompletion",
        )

        assert cost == pytest.approx(expected, rel=1e-9), label

        text_only = sum(c for m, c in prompt_details if m == "TEXT") * text_in + sum(
            c for m, c in candidate_details if m == "TEXT"
        ) * text_out
        if any(m != "TEXT" for m, _ in prompt_details + candidate_details) and audio_in != text_in:
            assert cost > text_only, f"{label}: non-text modalities must add cost"

    def test_vertex_ai_live_passthrough_handler_integration(
        self, handler, mock_logging_obj, sample_websocket_messages
    ):
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
            request_body=request_body,
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

    def test_vertex_ai_live_passthrough_handler_no_usage(
        self, handler, mock_logging_obj
    ):
        """Test handler with messages that don't contain usage metadata"""
        messages = [
            {"type": "session.created", "session": {"id": "test"}},
            {"type": "response.create", "response": {"text": "Hello"}},
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
            request_body=request_body,
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
            user_role="customer",
        )

    @pytest.fixture
    def mock_logging_obj(self):
        """Create a mock logging object"""
        mock = MagicMock(spec=LiteLLMLoggingObj)
        mock.model_call_details = {}
        return mock

    @patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.websocket_passthrough_request"
    )
    @patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router"
    )
    @patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.vertex_llm_base._ensure_access_token_async"
    )
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

    def test_vertex_ai_live_route_detection(self):
        """Test that the route detection works correctly"""
        from litellm.proxy.pass_through_endpoints.success_handler import (
            PassThroughEndpointLogging,
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

    @patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_ai_live_passthrough_logging_handler.VertexAILivePassthroughLoggingHandler"
    )
    @pytest.mark.asyncio
    async def test_success_handler_vertex_ai_live_integration(
        self, mock_handler_class, mock_logging_obj
    ):
        """Test the success handler integration with Vertex AI Live"""
        from litellm.proxy.pass_through_endpoints.success_handler import (
            PassThroughEndpointLogging,
        )

        # Mock the handler
        mock_handler = MagicMock()
        mock_handler.vertex_ai_live_passthrough_handler.return_value = {
            "result": {"model": "gemini-1.5-pro", "usage": {"total_tokens": 100}},
            "kwargs": {"test": "value"},
        }
        mock_handler_class.return_value = mock_handler

        # Create success handler
        success_handler = PassThroughEndpointLogging()

        # Mock the route check
        success_handler.is_vertex_ai_live_route = MagicMock(return_value=True)

        # Test data
        response_body = [{"type": "response.create", "response": {"text": "Hello"}}]
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
            passthrough_logging_payload=MagicMock(),
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
            None,
        ]

        # Should not raise an exception
        result = handler._extract_usage_metadata_from_websocket_messages(
            invalid_messages
        )
        assert result is None

    def test_missing_usage_metadata(self):
        """Test handling of messages with missing usage metadata"""
        handler = VertexAILivePassthroughLoggingHandler()

        messages = [
            {"type": "response.create", "response": {"text": "Hello"}},
            {"type": "response.done", "response": {"text": "Done"}},
        ]

        result = handler._extract_usage_metadata_from_websocket_messages(messages)
        assert result is None

    def test_usage_without_modality_details(self):
        """Older payloads carry only the totals; fall back to them rather than reporting zero."""
        handler = VertexAILivePassthroughLoggingHandler()

        usage = handler._create_usage_object_from_metadata(
            usage_metadata={
                "promptTokenCount": 100,
                "candidatesTokenCount": 50,
                "totalTokenCount": 150,
            },
            model="unknown-model",
        )

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.prompt_tokens_details.audio_tokens is None
        assert usage.prompt_tokens_details.image_tokens is None

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
            request_body=request_body,
        )

        assert "result" in result
        assert "kwargs" in result


if __name__ == "__main__":
    pytest.main([__file__])
