import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler import (
    OpenAIPassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    PassthroughStandardLoggingPayload,
)


class TestOpenAIPassthroughLoggingHandler:
    """Test the OpenAI passthrough logging handler for cost tracking."""

    def setup_method(self):
        """Set up test fixtures"""
        self.start_time = datetime.now()
        self.end_time = datetime.now()
        self.handler = OpenAIPassthroughLoggingHandler()
        
        # Mock OpenAI chat completions response
        self.mock_openai_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4o-2024-08-06",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 15,
                "total_tokens": 35
            }
        }

    def _create_mock_logging_obj(self) -> LiteLLMLoggingObj:
        """Create a mock logging object"""
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {}
        return mock_logging_obj

    def _create_mock_httpx_response(self, response_data: dict = None) -> httpx.Response:
        """Create a mock httpx response"""
        if response_data is None:
            response_data = self.mock_openai_response
            
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.text = json.dumps(response_data)
        mock_response.json.return_value = response_data
        mock_response.headers = {"content-type": "application/json"}
        return mock_response

    def _create_passthrough_logging_payload(self, user: str = "test_user") -> PassthroughStandardLoggingPayload:
        """Create a mock passthrough logging payload"""
        return PassthroughStandardLoggingPayload(
            url="https://api.openai.com/v1/chat/completions",
            request_body={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
            request_method="POST",
        )

    def test_llm_provider_name(self):
        """Test that the handler returns the correct provider name"""
        assert self.handler.llm_provider_name == "openai"

    def test_get_provider_config(self):
        """Test that the handler returns an OpenAI config"""
        handler = OpenAIPassthroughLoggingHandler()
        config = handler.get_provider_config(model="gpt-4o")
        assert config is not None
        # Verify it's an OpenAI config by checking if it has the expected methods
        assert hasattr(config, 'transform_response')

    def test_is_openai_chat_completions_route(self):
        """Test OpenAI chat completions route detection"""
        # Positive cases
        assert OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route("https://api.openai.com/v1/chat/completions") == True
        assert OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route("https://openai.azure.com/v1/chat/completions") == True
        
        # Negative cases
        assert OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route("https://api.openai.com/v1/models") == False
        assert OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route("http://localhost:4000/openai/v1/chat/completions") == False
        assert OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route("https://api.anthropic.com/v1/messages") == False
        assert OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route("") == False

    def test_is_openai_image_generation_route(self):
        """Test OpenAI image generation route detection"""
        # Positive cases
        assert OpenAIPassthroughLoggingHandler.is_openai_image_generation_route("https://api.openai.com/v1/images/generations") == True
        assert OpenAIPassthroughLoggingHandler.is_openai_image_generation_route("https://openai.azure.com/v1/images/generations") == True
        
        # Negative cases
        assert OpenAIPassthroughLoggingHandler.is_openai_image_generation_route("https://api.openai.com/v1/chat/completions") == False
        assert OpenAIPassthroughLoggingHandler.is_openai_image_generation_route("https://api.openai.com/v1/images/edits") == False
        assert OpenAIPassthroughLoggingHandler.is_openai_image_generation_route("http://localhost:4000/openai/v1/images/generations") == False
        assert OpenAIPassthroughLoggingHandler.is_openai_image_generation_route("") == False

    def test_is_openai_image_editing_route(self):
        """Test OpenAI image editing route detection"""
        # Positive cases
        assert OpenAIPassthroughLoggingHandler.is_openai_image_editing_route("https://api.openai.com/v1/images/edits") == True
        assert OpenAIPassthroughLoggingHandler.is_openai_image_editing_route("https://openai.azure.com/v1/images/edits") == True
        
        # Negative cases
        assert OpenAIPassthroughLoggingHandler.is_openai_image_editing_route("https://api.openai.com/v1/chat/completions") == False
        assert OpenAIPassthroughLoggingHandler.is_openai_image_editing_route("https://api.openai.com/v1/images/generations") == False
        assert OpenAIPassthroughLoggingHandler.is_openai_image_editing_route("http://localhost:4000/openai/v1/images/edits") == False
        assert OpenAIPassthroughLoggingHandler.is_openai_image_editing_route("") == False

    def test_is_openai_responses_route(self):
        """Test OpenAI responses API route detection"""
        # Positive cases
        assert OpenAIPassthroughLoggingHandler.is_openai_responses_route("https://api.openai.com/v1/responses") == True
        assert OpenAIPassthroughLoggingHandler.is_openai_responses_route("https://openai.azure.com/v1/responses") == True
        assert OpenAIPassthroughLoggingHandler.is_openai_responses_route("https://api.openai.com/responses") == True
        
        # Negative cases
        assert OpenAIPassthroughLoggingHandler.is_openai_responses_route("https://api.openai.com/v1/chat/completions") == False
        assert OpenAIPassthroughLoggingHandler.is_openai_responses_route("https://api.openai.com/v1/images/generations") == False
        assert OpenAIPassthroughLoggingHandler.is_openai_responses_route("http://localhost:4000/openai/v1/responses") == False
        assert OpenAIPassthroughLoggingHandler.is_openai_responses_route("") == False

    @patch('litellm.completion_cost')
    @patch('litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload')
    def test_openai_passthrough_handler_success(self, mock_get_standard_logging, mock_completion_cost):
        """Test successful cost tracking for OpenAI chat completions"""
        # Arrange
        mock_completion_cost.return_value = 0.000045
        mock_get_standard_logging.return_value = {"test": "logging_payload"}
        
        mock_httpx_response = self._create_mock_httpx_response()
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()
        
        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gpt-4o",
        }

        # Act
        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=self.mock_openai_response,
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/chat/completions",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
            **kwargs
        )

        # Assert
        assert result is not None
        assert "result" in result
        assert "kwargs" in result
        assert result["kwargs"]["response_cost"] == 0.000045
        assert result["kwargs"]["model"] == "gpt-4o"
        assert result["kwargs"]["custom_llm_provider"] == "openai"
        
        # Verify cost calculation was called
        mock_completion_cost.assert_called_once()
        
        # Verify logging object was updated
        assert mock_logging_obj.model_call_details["response_cost"] == 0.000045
        assert mock_logging_obj.model_call_details["model"] == "gpt-4o"
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "openai"

    @patch('litellm.completion_cost')
    def test_openai_passthrough_handler_non_chat_completions(self, mock_completion_cost):
        """Test that non-chat-completions routes fall back to base handler"""
        # Arrange
        mock_httpx_response = self._create_mock_httpx_response()
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()
        
        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gpt-4o",
        }

        # Act - Use a non-chat-completions route
        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body={"id": "file-123", "object": "file"},
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/files",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"purpose": "fine-tune"},
            **kwargs
        )

        # Assert - Should fall back to base handler for non-chat-completions
        assert result is not None
        assert "result" in result
        assert "kwargs" in result
        # Cost calculation may be called by the base handler fallback
        # The important thing is that our specific OpenAI handler logic didn't run

    @patch('litellm.completion_cost')
    @patch('litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload')
    def test_openai_passthrough_handler_with_user_tracking(self, mock_get_standard_logging, mock_completion_cost):
        """Test cost tracking with user information"""
        # Arrange
        mock_completion_cost.return_value = 0.000123
        mock_get_standard_logging.return_value = {"test": "logging_payload"}
        
        mock_httpx_response = self._create_mock_httpx_response()
        mock_logging_obj = self._create_mock_logging_obj()
        
        # Create payload with user information
        passthrough_payload = PassthroughStandardLoggingPayload(
            url="https://api.openai.com/v1/chat/completions",
            request_body={
                "model": "gpt-4o", 
                "messages": [{"role": "user", "content": "Hello"}],
                "user": "test_user_123"
            },
            request_method="POST",
        )
        
        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gpt-4o",
        }

        # Act
        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=self.mock_openai_response,
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/chat/completions",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}], "user": "test_user_123"},
            **kwargs
        )

        # Assert
        assert result is not None
        assert "result" in result
        assert "kwargs" in result
        assert result["kwargs"]["response_cost"] == 0.000123
        
        # Verify user information is included in litellm_params
        assert "litellm_params" in result["kwargs"]
        assert "proxy_server_request" in result["kwargs"]["litellm_params"]
        assert "body" in result["kwargs"]["litellm_params"]["proxy_server_request"]
        assert result["kwargs"]["litellm_params"]["proxy_server_request"]["body"]["user"] == "test_user_123"

    @patch('litellm.completion_cost')
    def test_openai_passthrough_handler_cost_calculation_error(self, mock_completion_cost):
        """Test error handling in cost calculation"""
        # Arrange
        mock_completion_cost.side_effect = Exception("Cost calculation failed")
        
        mock_httpx_response = self._create_mock_httpx_response()
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()
        
        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gpt-4o",
        }

        # Act
        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=self.mock_openai_response,
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/chat/completions",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
            **kwargs
        )

        # Assert - Should fall back to base handler when cost calculation fails
        assert result is not None
        assert "result" in result
        assert "kwargs" in result

    def test_build_complete_streaming_response(self):
        """Test the streaming response builder (placeholder implementation)"""
        # This is a placeholder method that returns None for now
        result = self.handler._build_complete_streaming_response(
            all_chunks=["chunk1", "chunk2"],
            litellm_logging_obj=self._create_mock_logging_obj(),
            model="gpt-4o",
        )
        
        assert result is None  # Placeholder implementation

    @patch('litellm.completion_cost')
    @patch('litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload')
    def test_different_models_cost_tracking(self, mock_get_standard_logging, mock_completion_cost):
        """Test cost tracking for different OpenAI models"""
        # Arrange
        mock_get_standard_logging.return_value = {"test": "logging_payload"}
        
        test_cases = [
            ("gpt-4o", 0.000045),
            ("gpt-4o-mini", 0.000015),
            ("gpt-3.5-turbo", 0.000002),
        ]
        
        for model, expected_cost in test_cases:
            mock_completion_cost.return_value = expected_cost
            
            mock_httpx_response = self._create_mock_httpx_response()
            mock_httpx_response.json.return_value = {
                **self.mock_openai_response,
                "model": model
            }
            
            mock_logging_obj = self._create_mock_logging_obj()
            passthrough_payload = self._create_passthrough_logging_payload()
            
            kwargs = {
                "passthrough_logging_payload": passthrough_payload,
                "model": model,
            }

            # Act
            result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
                httpx_response=mock_httpx_response,
                response_body={**self.mock_openai_response, "model": model},
                logging_obj=mock_logging_obj,
                url_route="https://api.openai.com/v1/chat/completions",
                result="",
                start_time=self.start_time,
                end_time=self.end_time,
                cache_hit=False,
                request_body={"model": model, "messages": [{"role": "user", "content": "Hello"}]},
                **kwargs
            )

            # Assert
            assert result is not None
            assert "result" in result
            assert "kwargs" in result
            assert result["kwargs"]["response_cost"] == expected_cost
            assert result["kwargs"]["model"] == model
            assert result["kwargs"]["custom_llm_provider"] == "openai"

    def test_static_methods(self):
        """Test that static methods work correctly"""
        # Test static method calls
        assert OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route("https://api.openai.com/v1/chat/completions") == True
        # Test instance method
        handler = OpenAIPassthroughLoggingHandler()
        assert handler.get_provider_config("gpt-4o") is not None

    @patch('litellm.completion_cost')
    @patch('litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload')
    def test_azure_passthrough_tags_metadata_model_provider(self, mock_get_standard_logging, mock_completion_cost):
        """Test that tags, metadata, model, and custom_llm_provider are preserved for Azure passthrough in UI"""
        # Arrange
        mock_completion_cost.return_value = 0.000045
        mock_get_standard_logging.return_value = {"test": "logging_payload"}
        
        mock_httpx_response = self._create_mock_httpx_response()
        mock_logging_obj = self._create_mock_logging_obj()
        
        # Create payload with metadata tags
        passthrough_payload = PassthroughStandardLoggingPayload(
            url="https://openai.azure.com/v1/chat/completions",
            request_body={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}]
            },
            request_method="POST",
        )
        
        # Set up kwargs with existing litellm_params containing metadata tags
        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gpt-4o",
            "custom_llm_provider": "azure",  # Azure passthrough
            "litellm_params": {
                "metadata": {
                    "tags": ["production", "azure-deployment"],
                    "user_id": "user_123"
                },
                "proxy_server_request": {
                    "body": {
                        "user": "test_user"
                    }
                }
            }
        }

        # Act
        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=self.mock_openai_response,
            logging_obj=mock_logging_obj,
            url_route="https://openai.azure.com/v1/chat/completions",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
            **kwargs
        )

        # Assert - Verify tags, model, and custom_llm_provider are preserved
        assert result is not None
        assert "kwargs" in result
        
        # Verify model and custom_llm_provider are set correctly
        assert result["kwargs"]["model"] == "gpt-4o"
        assert result["kwargs"]["custom_llm_provider"] == "azure"  # Should preserve Azure, not default to "openai"
        assert result["kwargs"]["response_cost"] == 0.000045
        
        # Verify metadata tags are preserved in litellm_params
        assert "litellm_params" in result["kwargs"]
        assert "metadata" in result["kwargs"]["litellm_params"]
        assert "tags" in result["kwargs"]["litellm_params"]["metadata"]
        assert result["kwargs"]["litellm_params"]["metadata"]["tags"] == ["production", "azure-deployment"]
        assert result["kwargs"]["litellm_params"]["metadata"]["user_id"] == "user_123"
        
        # Verify logging object has correct values for UI display
        assert mock_logging_obj.model_call_details["model"] == "gpt-4o"
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "azure"
        assert mock_logging_obj.model_call_details["response_cost"] == 0.000045
        
        # Verify cost calculation was called with correct custom_llm_provider
        mock_completion_cost.assert_called_once()
        call_args = mock_completion_cost.call_args
        assert call_args[1]["custom_llm_provider"] == "azure"

    @patch('litellm.completion_cost')
    @patch('litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload')
    @patch('litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler.OpenAIPassthroughLoggingHandler.get_provider_config')
    def test_responses_api_cost_tracking(self, mock_get_provider_config, mock_get_standard_logging, mock_completion_cost):
        """Test cost tracking for responses API route"""
        # Arrange
        mock_completion_cost.return_value = 0.000050
        mock_get_standard_logging.return_value = {"test": "logging_payload"}
        
        # Mock the provider config's transform_response to return a valid ModelResponse
        from litellm import ModelResponse
        mock_model_response = ModelResponse(
            id="resp_abc123",
            model="gpt-4o-2024-08-06",
            choices=[{
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?"
                }
            }],
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 15,
                "total_tokens": 35
            }
        )
        
        mock_provider_config = MagicMock()
        mock_provider_config.transform_response.return_value = mock_model_response
        mock_get_provider_config.return_value = mock_provider_config
        
        # Mock responses API response
        mock_responses_response = {
            "id": "resp_abc123",
            "object": "response",
            "created": 1677652288,
            "model": "gpt-4o-2024-08-06",
            "output": [
                {
                    "type": "text",
                    "text": "Hello! How can I help you today?"
                }
            ],
            "usage": {
                "input_tokens": 20,
                "output_tokens": 15
            }
        }
        
        mock_httpx_response = self._create_mock_httpx_response(mock_responses_response)
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()
        
        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gpt-4o",
            "custom_llm_provider": "openai",
        }

        # Act
        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=mock_responses_response,
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/responses",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"model": "gpt-4o", "input": "Tell me about AI"},
            **kwargs
        )

        # Assert
        assert result is not None
        assert "result" in result
        assert "kwargs" in result
        assert result["kwargs"]["response_cost"] == 0.000050
        assert result["kwargs"]["model"] == "gpt-4o"
        assert result["kwargs"]["custom_llm_provider"] == "openai"
        
        # Verify cost calculation was called with responses call type
        mock_completion_cost.assert_called_once()
        call_args = mock_completion_cost.call_args
        assert call_args[1]["call_type"] == "responses"
        assert call_args[1]["model"] == "gpt-4o"
        assert call_args[1]["custom_llm_provider"] == "openai"
        
        # Verify logging object was updated
        assert mock_logging_obj.model_call_details["response_cost"] == 0.000050
        assert mock_logging_obj.model_call_details["model"] == "gpt-4o"
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "openai"


class TestOpenAIPassthroughIntegration:
    """Integration tests for OpenAI passthrough cost tracking"""

    def setup_method(self):
        """Set up test fixtures"""
        self.handler = PassThroughEndpointLogging()
        self.start_time = datetime.now()
        self.end_time = datetime.now()

    def _create_mock_logging_obj(self) -> LiteLLMLoggingObj:
        """Create a mock logging object"""
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {}
        return mock_logging_obj

    def _create_mock_httpx_response(self, response_data: dict = None) -> httpx.Response:
        """Create a mock httpx response"""
        if response_data is None:
            response_data = {"id": "test", "choices": [{"message": {"content": "Hello"}}]}
            
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.text = json.dumps(response_data)
        mock_response.json.return_value = response_data
        mock_response.headers = {"content-type": "application/json"}
        return mock_response

    def _create_passthrough_logging_payload(self, user: str = "test_user") -> PassthroughStandardLoggingPayload:
        """Create a mock passthrough logging payload"""
        return PassthroughStandardLoggingPayload(
            url="https://api.openai.com/v1/chat/completions",
            request_body={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
            request_method="POST",
        )

    def test_is_openai_route_detection(self):
        """Test OpenAI route detection in the main success handler"""
        # Positive cases
        assert self.handler.is_openai_route("https://api.openai.com/v1/chat/completions") == True
        assert self.handler.is_openai_route("https://openai.azure.com/v1/chat/completions") == True
        assert self.handler.is_openai_route("https://api.openai.com/v1/models") == True
        
        # Negative cases
        assert self.handler.is_openai_route("http://localhost:4000/openai/v1/chat/completions") == False
        assert self.handler.is_openai_route("https://api.anthropic.com/v1/messages") == False
        assert self.handler.is_openai_route("https://api.assemblyai.com/v2/transcript") == False
        assert self.handler.is_openai_route("") == False

    @patch('litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler.OpenAIPassthroughLoggingHandler.openai_passthrough_handler')
    @pytest.mark.asyncio
    async def test_success_handler_calls_openai_handler(self, mock_openai_handler):
        """Test that the success handler calls our OpenAI handler for OpenAI routes"""
        # Arrange
        mock_openai_handler.return_value = {
            "result": {"id": "chatcmpl-123"},
            "kwargs": {
                "response_cost": 0.000045,
                "model": "gpt-4o",
                "custom_llm_provider": "openai"
            }
        }
        
        mock_httpx_response = MagicMock(spec=httpx.Response)
        mock_httpx_response.text = '{"id": "chatcmpl-123", "choices": [{"message": {"content": "Hello"}}]}'
        
        mock_logging_obj = AsyncMock()
        mock_logging_obj.model_call_details = {}
        mock_logging_obj.async_success_handler = AsyncMock()
        
        passthrough_payload = PassthroughStandardLoggingPayload(
            url="https://api.openai.com/v1/chat/completions",
            request_body={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
            request_method="POST",
        )

        # Act
        result = await self.handler.pass_through_async_success_handler(
            httpx_response=mock_httpx_response,
            response_body={"id": "chatcmpl-123", "choices": [{"message": {"content": "Hello"}}]},
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/chat/completions",
            result="",
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
            passthrough_logging_payload=passthrough_payload,
        )

        # Assert
        mock_openai_handler.assert_called_once()
        # The success handler returns None on success, which is expected
        assert result is None

    @pytest.mark.asyncio
    async def test_success_handler_falls_back_for_non_openai_routes(self):
        """Test that non-OpenAI routes don't call our handler"""
        # Arrange
        mock_httpx_response = MagicMock(spec=httpx.Response)
        mock_httpx_response.text = '{"status": "success"}'
        mock_httpx_response.headers = {"content-type": "application/json"}
        
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {}
        
        passthrough_payload = PassthroughStandardLoggingPayload(
            url="https://api.anthropic.com/v1/messages",
            request_body={"model": "claude-3-sonnet", "messages": [{"role": "user", "content": "Hello"}]},
            request_method="POST",
        )

        # Mock the _handle_logging method to capture calls
        self.handler._handle_logging = AsyncMock()

        # Act
        result = await self.handler.pass_through_async_success_handler(
            httpx_response=mock_httpx_response,
            response_body={"status": "success"},
            logging_obj=mock_logging_obj,
            url_route="https://api.anthropic.com/v1/messages",
            result="",
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body={"model": "claude-3-sonnet", "messages": [{"role": "user", "content": "Hello"}]},
            passthrough_logging_payload=passthrough_payload,
        )

        # Assert - Should call the base handler, not our OpenAI handler
        self.handler._handle_logging.assert_called_once()

    @patch('litellm.cost_calculator.default_image_cost_calculator')
    def test_calculate_image_generation_cost(self, mock_image_cost_calculator):
        """Test image generation cost calculation"""
        # Arrange
        mock_image_cost_calculator.return_value = 0.040
        model = "dall-e-3"
        response_body = {
            "data": [
                {
                    "url": "https://example.com/image1.png",
                    "revised_prompt": "A beautiful sunset over the ocean"
                }
            ]
        }
        request_body = {
            "model": "dall-e-3",
            "prompt": "A beautiful sunset over the ocean",
            "n": 1,
            "size": "1024x1024",
            "quality": "standard"
        }

        # Act
        cost = OpenAIPassthroughLoggingHandler._calculate_image_generation_cost(
            model=model,
            response_body=response_body,
            request_body=request_body,
        )

        # Assert
        assert cost == 0.040
        mock_image_cost_calculator.assert_called_once_with(
            model=model,
            custom_llm_provider="openai",
            quality="standard",
            n=1,
            size="1024x1024",
            optional_params=request_body,
        )

    @patch('litellm.cost_calculator.default_image_cost_calculator')
    def test_calculate_image_editing_cost(self, mock_image_cost_calculator):
        """Test image editing cost calculation"""
        # Arrange
        mock_image_cost_calculator.return_value = 0.020
        model = "dall-e-2"
        response_body = {
            "data": [
                {
                    "url": "https://example.com/edited_image.png",
                    "revised_prompt": "A beautiful sunset over the ocean with added clouds"
                }
            ]
        }
        request_body = {
            "model": "dall-e-2",
            "prompt": "Add clouds to the sky",
            "n": 1,
            "size": "1024x1024"
        }

        # Act
        cost = OpenAIPassthroughLoggingHandler._calculate_image_editing_cost(
            model=model,
            response_body=response_body,
            request_body=request_body,
        )

        # Assert
        assert cost == 0.020
        mock_image_cost_calculator.assert_called_once_with(
            model=model,
            custom_llm_provider="openai",
            quality=None,  # Image editing doesn't have quality parameter
            n=1,
            size="1024x1024",
            optional_params=request_body,
        )

    def test_cost_calculation_preservation(self):
        """Test that manually calculated costs are preserved and not overridden."""
        # Create a logging object
        logging_obj = LiteLLMLoggingObj(
            model="dall-e-3",
            messages=[{"role": "user", "content": "Generate an image"}],
            stream=False,
            call_type="pass_through_endpoint",
            start_time=self.start_time,
            litellm_call_id="test_123",
            function_id="test_fn",
        )
        
        # Set a manually calculated cost in model_call_details
        test_cost = 0.040000
        logging_obj.model_call_details["response_cost"] = test_cost
        logging_obj.model_call_details["model"] = "dall-e-3"
        logging_obj.model_call_details["custom_llm_provider"] = "openai"
        
        # Create an ImageResponse with cost in _hidden_params
        from litellm.types.utils import ImageResponse
        image_response = ImageResponse(
            data=[{"url": "https://example.com/image.png"}],
            model="dall-e-3",
        )
        image_response._hidden_params = {"response_cost": test_cost}
        
        # Test the _response_cost_calculator method
        calculated_cost = logging_obj._response_cost_calculator(result=image_response)
        
        assert calculated_cost == test_cost, f"Expected {test_cost}, got {calculated_cost}"

    @patch('litellm.cost_calculator.default_image_cost_calculator')
    def test_openai_passthrough_handler_image_generation(self, mock_image_cost_calculator):
        """Test successful cost tracking for OpenAI image generation"""
        # Arrange
        mock_image_cost_calculator.return_value = 0.040
        
        mock_image_response = {
            "data": [
                {
                    "url": "https://example.com/image1.png",
                    "revised_prompt": "A beautiful sunset over the ocean"
                }
            ]
        }
        
        mock_httpx_response = self._create_mock_httpx_response(mock_image_response)
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()
        
        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "dall-e-3",
        }

        request_body = {
            "model": "dall-e-3",
            "prompt": "A beautiful sunset over the ocean",
            "n": 1,
            "size": "1024x1024",
            "quality": "standard"
        }

        # Act
        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=mock_image_response,
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/images/generations",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body=request_body,
            **kwargs
        )

        # Assert
        assert result is not None
        assert "result" in result
        assert "kwargs" in result
        assert result["kwargs"]["response_cost"] == 0.040
        assert result["kwargs"]["model"] == "dall-e-3"
        assert result["kwargs"]["custom_llm_provider"] == "openai"
        
        # Verify cost calculation was called
        mock_image_cost_calculator.assert_called_once()
        
        # Verify logging object was updated
        assert mock_logging_obj.model_call_details["response_cost"] == 0.040
        assert mock_logging_obj.model_call_details["model"] == "dall-e-3"
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "openai"

    @patch('litellm.cost_calculator.default_image_cost_calculator')
    def test_openai_passthrough_handler_image_editing(self, mock_image_cost_calculator):
        """Test successful cost tracking for OpenAI image editing"""
        # Arrange
        mock_image_cost_calculator.return_value = 0.020
        
        mock_image_response = {
            "data": [
                {
                    "url": "https://example.com/edited_image.png",
                    "revised_prompt": "A beautiful sunset over the ocean with added clouds"
                }
            ]
        }
        
        mock_httpx_response = self._create_mock_httpx_response(mock_image_response)
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()
        
        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "dall-e-2",
        }

        request_body = {
            "model": "dall-e-2",
            "prompt": "Add clouds to the sky",
            "n": 1,
            "size": "1024x1024"
        }

        # Act
        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=mock_image_response,
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/images/edits",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body=request_body,
            **kwargs
        )

        # Assert
        assert result is not None
        assert "result" in result
        assert "kwargs" in result
        assert result["kwargs"]["response_cost"] == 0.020
        assert result["kwargs"]["model"] == "dall-e-2"
        assert result["kwargs"]["custom_llm_provider"] == "openai"
        
        # Verify cost calculation was called
        mock_image_cost_calculator.assert_called_once()
        
        # Verify logging object was updated
        assert mock_logging_obj.model_call_details["response_cost"] == 0.020
        assert mock_logging_obj.model_call_details["model"] == "dall-e-2"
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "openai"


if __name__ == "__main__":
    pytest.main([__file__])
