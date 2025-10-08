import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../.."))  # Adds the parent directory to the system path

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.gemini_passthrough_logging_handler import (
    GeminiPassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    PassthroughStandardLoggingPayload,
)


class TestGeminiPassthroughLoggingHandler:
    """Test the Gemini passthrough logging handler for cost tracking."""

    def setup_method(self):
        """Set up test fixtures"""
        self.start_time = datetime.now()
        self.end_time = datetime.now()
        self.handler = GeminiPassthroughLoggingHandler()

        # Mock Gemini generateContent response
        self.mock_gemini_response = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello! How can I help you today?"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                    "safetyRatings": [
                        {"category": "HARM_CATEGORY_HARASSMENT", "probability": "NEGLIGIBLE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "NEGLIGIBLE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "probability": "NEGLIGIBLE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "NEGLIGIBLE"},
                    ],
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 8, "totalTokenCount": 18},
        }

    def _create_mock_httpx_response(self) -> httpx.Response:
        """Create a mock httpx.Response for testing"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.text = json.dumps(self.mock_gemini_response)
        mock_response.json.return_value = self.mock_gemini_response
        mock_response.headers = {"content-type": "application/json"}
        return mock_response

    def _create_mock_logging_obj(self) -> LiteLLMLoggingObj:
        """Create a mock logging object for testing"""
        mock_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {}
        mock_logging_obj.optional_params = {}
        mock_logging_obj.litellm_call_id = "test-call-id-123"
        return mock_logging_obj

    def _create_passthrough_logging_payload(self) -> PassthroughStandardLoggingPayload:
        """Create a mock passthrough logging payload for testing"""
        return PassthroughStandardLoggingPayload(
            url="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            request_body={"contents": [{"parts": [{"text": "Hello"}]}]},
            request_method="POST",
        )

    def test_is_gemini_route(self):
        """Test that Gemini routes are correctly identified"""
        from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging

        handler = PassThroughEndpointLogging()

        # Test generateContent endpoint
        assert (
            handler.is_gemini_route(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
                custom_llm_provider="gemini",
            )
            is True
        )

        # Test streamGenerateContent endpoint
        assert (
            handler.is_gemini_route(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent",
                custom_llm_provider="gemini",
            )
            is True
        )

        # Test non-Gemini endpoint
        assert (
            handler.is_gemini_route("https://api.openai.com/v1/chat/completions", custom_llm_provider="openai") is False
        )

    def test_extract_model_from_url(self):
        """Test that model is correctly extracted from Gemini URLs"""
        # Test generateContent endpoint
        model = GeminiPassthroughLoggingHandler.extract_model_from_url(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        )
        assert model == "gemini-1.5-flash"

        # Test streamGenerateContent endpoint
        model = GeminiPassthroughLoggingHandler.extract_model_from_url(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:streamGenerateContent"
        )
        assert model == "gemini-1.5-pro"

    @patch("litellm.completion_cost")
    @patch("litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload")
    def test_gemini_passthrough_handler_success(self, mock_get_standard_logging, mock_completion_cost):
        """Test successful cost tracking for Gemini generateContent endpoint"""
        # Arrange
        mock_completion_cost.return_value = 0.000045
        mock_get_standard_logging.return_value = {"test": "logging_payload"}

        mock_httpx_response = self._create_mock_httpx_response()
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()

        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gemini-1.5-flash",
        }

        # Act
        result = GeminiPassthroughLoggingHandler.gemini_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=self.mock_gemini_response,
            logging_obj=mock_logging_obj,
            url_route="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"contents": [{"parts": [{"text": "Hello"}]}]},
            **kwargs,
        )

        # Assert
        assert result is not None
        assert "result" in result
        assert "kwargs" in result
        assert result["kwargs"]["response_cost"] == 0.000045
        assert result["kwargs"]["model"] == "gemini-1.5-flash"
        assert result["kwargs"]["custom_llm_provider"] == "gemini"

        # Verify cost calculation was called
        mock_completion_cost.assert_called_once()

        # Verify logging object was updated
        assert mock_logging_obj.model_call_details["response_cost"] == 0.000045
        assert mock_logging_obj.model_call_details["model"] == "gemini-1.5-flash"
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "gemini"

    @patch("litellm.completion_cost")
    def test_gemini_passthrough_handler_streaming(self, mock_completion_cost):
        """Test cost tracking for Gemini streaming endpoint"""
        # Arrange
        mock_completion_cost.return_value = 0.000030

        # Mock streaming response chunks
        mock_chunks = [
            {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]},
            {"candidates": [{"content": {"parts": [{"text": " there!"}]}}]},
        ]

        mock_httpx_response = self._create_mock_httpx_response()
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()

        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gemini-1.5-flash",
        }

        # Act - Use generateContent URL since that's what the handler processes
        result = GeminiPassthroughLoggingHandler.gemini_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=mock_chunks,
            logging_obj=mock_logging_obj,
            url_route="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"contents": [{"parts": [{"text": "Hello"}]}]},
            **kwargs,
        )

        # Assert
        assert result is not None
        assert "result" in result
        assert "kwargs" in result
        assert result["kwargs"]["response_cost"] == 0.000030
        assert result["kwargs"]["model"] == "gemini-1.5-flash"
        assert result["kwargs"]["custom_llm_provider"] == "gemini"

        # Verify cost calculation was called
        mock_completion_cost.assert_called_once()

    def test_gemini_passthrough_handler_non_gemini_route(self):
        """Test that non-Gemini routes return None"""
        mock_httpx_response = self._create_mock_httpx_response()
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()

        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gpt-4o",
        }

        # Act
        result = GeminiPassthroughLoggingHandler.gemini_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=self.mock_gemini_response,
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/chat/completions",  # Non-Gemini route (no generateContent)
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
            **kwargs,
        )

        # Assert - the handler should return a dict with None result for non-Gemini routes
        assert result is not None
        assert result["result"] is None
        assert "kwargs" in result

    @pytest.mark.asyncio
    async def test_pass_through_success_handler_gemini_routing(self):
        """Test that the success handler correctly routes Gemini requests to the Gemini handler"""
        handler = PassThroughEndpointLogging()

        # Mock the logging object
        mock_logging_obj = self._create_mock_logging_obj()

        # Mock the _handle_logging method to capture the call
        handler._handle_logging = AsyncMock()

        # Mock httpx response
        mock_response = self._create_mock_httpx_response()

        # Create passthrough logging payload
        passthrough_logging_payload = self._create_passthrough_logging_payload()

        # Call the success handler with Gemini route and provider
        result = await handler.pass_through_async_success_handler(
            httpx_response=mock_response,
            response_body=self.mock_gemini_response,
            logging_obj=mock_logging_obj,
            url_route="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"contents": [{"parts": [{"text": "Hello"}]}]},
            passthrough_logging_payload=passthrough_logging_payload,
            custom_llm_provider="gemini",
        )

        # Assert - The success handler returns None on success (following the pattern from other tests)
        assert result is None

        # Verify that the logging object has the cost set (from Gemini handler)
        assert mock_logging_obj.model_call_details["response_cost"] is not None
        assert mock_logging_obj.model_call_details["model"] == "gemini-1.5-flash"
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "gemini"

        # Verify that _handle_logging was called with the correct kwargs
        handler._handle_logging.assert_called_once()
        call_kwargs = handler._handle_logging.call_args[1]
        assert call_kwargs["response_cost"] is not None
        assert call_kwargs["model"] == "gemini-1.5-flash"
        assert call_kwargs["custom_llm_provider"] == "gemini"
