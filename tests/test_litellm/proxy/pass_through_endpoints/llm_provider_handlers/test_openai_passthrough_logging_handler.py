import json
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler import (
    OpenAIPassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
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
                        "content": "Hello! How can I help you today?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
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
            request_body={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
            },
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
        assert hasattr(config, "transform_response")

    def test_is_openai_chat_completions_route(self):
        """Test OpenAI chat completions route detection"""
        # Positive cases
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route(
                "https://api.openai.com/v1/chat/completions"
            )
            == True
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route(
                "https://openai.azure.com/v1/chat/completions"
            )
            == True
        )

        # Negative cases
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route("https://api.openai.com/v1/models")
            == False
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route(
                "http://localhost:4000/openai/v1/chat/completions"
            )
            == False
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route("https://api.anthropic.com/v1/messages")
            == False
        )
        assert OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route("") == False

    def test_is_openai_image_generation_route(self):
        """Test OpenAI image generation route detection"""
        # Positive cases
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_image_generation_route(
                "https://api.openai.com/v1/images/generations"
            )
            == True
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_image_generation_route(
                "https://openai.azure.com/v1/images/generations"
            )
            == True
        )

        # Negative cases
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_image_generation_route(
                "https://api.openai.com/v1/chat/completions"
            )
            == False
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_image_generation_route("https://api.openai.com/v1/images/edits")
            == False
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_image_generation_route(
                "http://localhost:4000/openai/v1/images/generations"
            )
            == False
        )
        assert OpenAIPassthroughLoggingHandler.is_openai_image_generation_route("") == False

    def test_is_openai_image_editing_route(self):
        """Test OpenAI image editing route detection"""
        # Positive cases
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_image_editing_route("https://api.openai.com/v1/images/edits")
            == True
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_image_editing_route("https://openai.azure.com/v1/images/edits")
            == True
        )

        # Negative cases
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_image_editing_route("https://api.openai.com/v1/chat/completions")
            == False
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_image_editing_route(
                "https://api.openai.com/v1/images/generations"
            )
            == False
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_image_editing_route(
                "http://localhost:4000/openai/v1/images/edits"
            )
            == False
        )
        assert OpenAIPassthroughLoggingHandler.is_openai_image_editing_route("") == False

    def test_is_openai_responses_route(self):
        """Test OpenAI responses API route detection"""
        # Positive cases
        assert OpenAIPassthroughLoggingHandler.is_openai_responses_route("https://api.openai.com/v1/responses") == True
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_responses_route("https://openai.azure.com/v1/responses") == True
        )
        assert OpenAIPassthroughLoggingHandler.is_openai_responses_route("https://api.openai.com/responses") == True

        # Negative cases
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_responses_route("https://api.openai.com/v1/chat/completions")
            == False
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_responses_route("https://api.openai.com/v1/images/generations")
            == False
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_responses_route("http://localhost:4000/openai/v1/responses")
            == False
        )
        assert OpenAIPassthroughLoggingHandler.is_openai_responses_route("") == False

    def test_is_openai_embeddings_route(self):
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_embeddings_route("https://api.openai.com/v1/embeddings") is True
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_embeddings_route("https://openai.azure.com/v1/embeddings") is True
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_embeddings_route(
                "https://my-resource.cognitiveservices.azure.com/v1/embeddings"
            )
            is True
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_embeddings_route(
                "https://my-resource.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings"
            )
            is False
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_embeddings_route("https://api.openai.com/v1/chat/completions")
            is False
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_embeddings_route(
                "http://localhost:4000/openai_passthrough/v1/embeddings"
            )
            is False
        )
        assert OpenAIPassthroughLoggingHandler.is_openai_embeddings_route("") is False

    def test_is_openai_route_recognizes_cognitiveservices_azure_com(self):
        """Azure OpenAI resources created via the newer "Azure AI Foundry" /
        Cognitive Services pathway live on `*.cognitiveservices.azure.com`
        subdomains rather than the older `openai.azure.com`. The
        is_openai_*_route methods must recognize both Azure subdomains so
        cost tracking applies regardless of which Azure naming the user's
        resource happens to be on.
        """
        cognitive_chat = "https://my-resource.cognitiveservices.azure.com/v1/chat/completions"
        cognitive_images_gen = "https://my-resource.cognitiveservices.azure.com/v1/images/generations"
        cognitive_images_edit = "https://my-resource.cognitiveservices.azure.com/v1/images/edits"
        cognitive_responses = "https://my-resource.cognitiveservices.azure.com/v1/responses"
        cognitive_embeddings = "https://my-resource.cognitiveservices.azure.com/v1/embeddings"

        assert OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route(cognitive_chat) is True
        assert OpenAIPassthroughLoggingHandler.is_openai_image_generation_route(cognitive_images_gen) is True
        assert OpenAIPassthroughLoggingHandler.is_openai_image_editing_route(cognitive_images_edit) is True
        assert OpenAIPassthroughLoggingHandler.is_openai_responses_route(cognitive_responses) is True
        assert OpenAIPassthroughLoggingHandler.is_openai_embeddings_route(cognitive_embeddings) is True

        # Cross-route negatives still hold for cognitiveservices hosts.
        assert OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route(cognitive_responses) is False
        assert OpenAIPassthroughLoggingHandler.is_openai_responses_route(cognitive_chat) is False
        assert OpenAIPassthroughLoggingHandler.is_openai_embeddings_route(cognitive_chat) is False

    @patch("litellm.completion_cost")
    @patch("litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload")
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
            request_body={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            **kwargs,
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

    @patch("litellm.completion_cost")
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
            **kwargs,
        )

        # Assert - Should fall back to base handler for non-chat-completions
        assert result is not None
        assert "result" in result
        assert "kwargs" in result
        # Cost calculation may be called by the base handler fallback
        # The important thing is that our specific OpenAI handler logic didn't run

    @patch("litellm.completion_cost")
    @patch("litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload")
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
                "user": "test_user_123",
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
            request_body={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
                "user": "test_user_123",
            },
            **kwargs,
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

    @patch("litellm.completion_cost")
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
            request_body={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            **kwargs,
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

    @patch(f"{OpenAIPassthroughLoggingHandler.__module__}.get_standard_logging_object_payload")
    @patch("litellm.completion_cost", return_value=3.3e-06)
    def test_streaming_responses_cost_uses_completed_response(self, mock_completion_cost, mock_get_standard_logging):
        response_id = "resp_PROOFSENTINEL0123456789abcdef"
        completed_event = {
            "type": "response.completed",
            "sequence_number": 8,
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": 1786374786,
                "status": "completed",
                "model": "gpt-4o-mini-2024-07-18",
                "output": [
                    {
                        "id": "msg_abc",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "OK",
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 14,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 2,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 16,
                },
                "error": None,
                "incomplete_details": None,
                "instructions": None,
                "metadata": {},
                "parallel_tool_calls": True,
                "temperature": 1.0,
                "tool_choice": "auto",
                "tools": [],
                "top_p": 1.0,
            },
        }
        logging_obj = self._create_mock_logging_obj()

        result = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=MagicMock(),
            url_route="https://api.openai.com/v1/responses",
            request_body={"model": "gpt-4o-mini", "stream": True},
            endpoint_type=MagicMock(),
            start_time=self.start_time,
            all_chunks=[f"data: {json.dumps(completed_event)}", "data: [DONE]"],
            end_time=self.end_time,
        )

        response = result["result"]
        assert response.id == response_id
        assert response.model == "gpt-4o-mini-2024-07-18"
        assert response.usage.input_tokens == 14
        assert response.usage.output_tokens == 2
        assert result["kwargs"]["response_cost"] == 3.3e-06
        assert result["kwargs"]["standard_logging_object"] is mock_get_standard_logging.return_value
        mock_completion_cost.assert_called_once_with(
            completion_response=response,
            model="gpt-4o-mini",
            custom_llm_provider="openai",
            call_type="responses",
        )

    @patch(f"{OpenAIPassthroughLoggingHandler.__module__}.get_standard_logging_object_payload")
    @patch("litellm.completion_cost", return_value=2.1e-06)
    def test_streaming_responses_incomplete_event_is_billed(self, mock_completion_cost, mock_get_standard_logging):
        response_id = "resp_INCOMPLETESENTINEL0123456789ab"
        incomplete_event = {
            "type": "response.incomplete",
            "sequence_number": 5,
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": 1786374786,
                "status": "incomplete",
                "model": "gpt-4o-mini-2024-07-18",
                "output": [
                    {
                        "id": "msg_abc",
                        "type": "message",
                        "status": "incomplete",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "OK",
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 14,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 32,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 46,
                },
                "error": None,
                "incomplete_details": {"reason": "max_output_tokens"},
                "instructions": None,
                "metadata": {},
                "parallel_tool_calls": True,
                "temperature": 1.0,
                "tool_choice": "auto",
                "tools": [],
                "top_p": 1.0,
            },
        }
        logging_obj = self._create_mock_logging_obj()

        result = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=MagicMock(),
            url_route="https://api.openai.com/v1/responses",
            request_body={"model": "gpt-4o-mini", "stream": True},
            endpoint_type=MagicMock(),
            start_time=self.start_time,
            all_chunks=[f"data: {json.dumps(incomplete_event)}"],
            end_time=self.end_time,
        )

        response = result["result"]
        assert response.id == response_id
        assert response.status == "incomplete"
        assert response.usage.output_tokens == 32
        assert result["kwargs"]["response_cost"] == 2.1e-06
        assert result["kwargs"]["standard_logging_object"] is mock_get_standard_logging.return_value
        mock_completion_cost.assert_called_once_with(
            completion_response=response,
            model="gpt-4o-mini",
            custom_llm_provider="openai",
            call_type="responses",
        )

    @patch(f"{OpenAIPassthroughLoggingHandler.__module__}.get_standard_logging_object_payload")
    @patch("litellm.completion_cost", return_value=1.4e-06)
    def test_streaming_responses_failed_event_is_billed(self, mock_completion_cost, mock_get_standard_logging):
        response_id = "resp_FAILEDSENTINEL0123456789abcd"
        failed_event = {
            "type": "response.failed",
            "sequence_number": 4,
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": 1786374786,
                "status": "failed",
                "model": "gpt-4o-mini-2024-07-18",
                "output": [],
                "usage": {
                    "input_tokens": 14,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 7,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 21,
                },
                "error": {"code": "server_error", "message": "The model failed to generate a response"},
                "incomplete_details": None,
                "instructions": None,
                "metadata": {},
                "parallel_tool_calls": True,
                "temperature": 1.0,
                "tool_choice": "auto",
                "tools": [],
                "top_p": 1.0,
            },
        }
        logging_obj = self._create_mock_logging_obj()

        result = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=MagicMock(),
            url_route="https://api.openai.com/v1/responses",
            request_body={"model": "gpt-4o-mini", "stream": True},
            endpoint_type=MagicMock(),
            start_time=self.start_time,
            all_chunks=[f"data: {json.dumps(failed_event)}"],
            end_time=self.end_time,
        )

        response = result["result"]
        assert response.id == response_id
        assert response.status == "failed"
        assert response.usage.total_tokens == 21
        assert result["kwargs"]["response_cost"] == 1.4e-06
        assert result["kwargs"]["standard_logging_object"] is mock_get_standard_logging.return_value
        mock_completion_cost.assert_called_once_with(
            completion_response=response,
            model="gpt-4o-mini",
            custom_llm_provider="openai",
            call_type="responses",
        )

    @patch(f"{OpenAIPassthroughLoggingHandler.__module__}.get_standard_logging_object_payload", return_value=None)
    @patch("litellm.completion_cost", return_value=3.3e-06)
    def test_streaming_responses_none_payload_is_not_attached(self, mock_completion_cost, mock_get_standard_logging):
        completed_event = {
            "type": "response.completed",
            "sequence_number": 8,
            "response": {
                "id": "resp_NONEPAYLOADSENTINEL0123456789",
                "object": "response",
                "created_at": 1786374786,
                "status": "completed",
                "model": "gpt-4o-mini-2024-07-18",
                "output": [],
                "usage": {
                    "input_tokens": 14,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 2,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 16,
                },
                "error": None,
                "incomplete_details": None,
                "instructions": None,
                "metadata": {},
                "parallel_tool_calls": True,
                "temperature": 1.0,
                "tool_choice": "auto",
                "tools": [],
                "top_p": 1.0,
            },
        }
        logging_obj = self._create_mock_logging_obj()

        result = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=MagicMock(),
            url_route="https://api.openai.com/v1/responses",
            request_body={"model": "gpt-4o-mini", "stream": True},
            endpoint_type=MagicMock(),
            start_time=self.start_time,
            all_chunks=[f"data: {json.dumps(completed_event)}", "data: [DONE]"],
            end_time=self.end_time,
        )

        assert "standard_logging_object" not in result["kwargs"]
        assert result["kwargs"]["response_cost"] == 3.3e-06

    @patch(f"{OpenAIPassthroughLoggingHandler.__module__}.get_standard_logging_object_payload")
    @patch("litellm.completion_cost")
    def test_streaming_responses_without_completed_event_returns_none(
        self, mock_completion_cost, mock_get_standard_logging
    ):
        logging_obj = self._create_mock_logging_obj()

        result = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=MagicMock(),
            url_route="https://api.openai.com/v1/responses",
            request_body={"model": "gpt-4o-mini", "stream": True},
            endpoint_type=MagicMock(),
            start_time=self.start_time,
            all_chunks=[
                'data: {"type": "response.created", "sequence_number": 0}',
                'data: {"type": "response.output_text.delta", "sequence_number": 1, "delta": "OK"}',
            ],
            end_time=self.end_time,
        )

        assert result == {"result": None, "kwargs": {}}
        mock_completion_cost.assert_not_called()

    @patch("litellm.completion_cost")
    @patch("litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload")
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
                "model": model,
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
                request_body={
                    "model": model,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
                **kwargs,
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
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route(
                "https://api.openai.com/v1/chat/completions"
            )
            == True
        )
        # Test instance method
        handler = OpenAIPassthroughLoggingHandler()
        assert handler.get_provider_config("gpt-4o") is not None

    @patch("litellm.completion_cost")
    @patch("litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload")
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
                "messages": [{"role": "user", "content": "Hello"}],
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
                    "user_id": "user_123",
                },
                "proxy_server_request": {"body": {"user": "test_user"}},
            },
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
            request_body={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            **kwargs,
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
        assert result["kwargs"]["litellm_params"]["metadata"]["tags"] == [
            "production",
            "azure-deployment",
        ]
        assert result["kwargs"]["litellm_params"]["metadata"]["user_id"] == "user_123"

        # Verify logging object has correct values for UI display
        assert mock_logging_obj.model_call_details["model"] == "gpt-4o"
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "azure"
        assert mock_logging_obj.model_call_details["response_cost"] == 0.000045

        # Verify cost calculation was called with correct custom_llm_provider
        mock_completion_cost.assert_called_once()
        call_args = mock_completion_cost.call_args
        assert call_args[1]["custom_llm_provider"] == "azure"

    @patch("litellm.completion_cost")
    @patch("litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload")
    @patch("litellm.llms.openai.responses.transformation.OpenAIResponsesAPIConfig.transform_response_api_response")
    def test_responses_api_cost_tracking(
        self,
        mock_transform_responses,
        mock_get_standard_logging,
        mock_completion_cost,
    ):
        """Test cost tracking for responses API route.

        Mocks the Responses-API transformer (the dedicated one this branch
        of the handler dispatches into post-fix) so we can assert the
        downstream cost-calculation contract without depending on the
        real transformer's full behavior.
        """
        # Arrange
        mock_completion_cost.return_value = 0.000050
        mock_get_standard_logging.return_value = {"test": "logging_payload"}

        # Mock the Responses transformer's return — a ResponsesAPIResponse
        # carrying the usage fields downstream cost-calc expects.
        from litellm.types.llms.openai import ResponsesAPIResponse

        mock_responses_api_response = ResponsesAPIResponse.model_construct(
            id="resp_abc123",
            object="response",
            created_at=1677652288,
            model="gpt-4o-2024-08-06",
            status="completed",
            output=[],
            usage={
                "input_tokens": 20,
                "output_tokens": 15,
                "total_tokens": 35,
            },
        )
        mock_transform_responses.return_value = mock_responses_api_response

        # Mock responses API response
        mock_responses_response = {
            "id": "resp_abc123",
            "object": "response",
            "created": 1677652288,
            "model": "gpt-4o-2024-08-06",
            "output": [{"type": "text", "text": "Hello! How can I help you today?"}],
            "usage": {"input_tokens": 20, "output_tokens": 15},
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
            **kwargs,
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

    @patch("litellm.completion_cost")
    @patch("litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload")
    def test_responses_api_uses_responses_transformer_not_chat_completions(
        self, mock_get_standard_logging, mock_completion_cost
    ):
        """Regression test for the Responses-API cost-tracking dispatch bug.

        BUG: the `elif is_responses:` branch in `openai_passthrough_handler`
        was calling `OpenAIConfig.transform_response` (the chat-completions
        transformer) on a Responses API payload. Chat-completions
        transform_response expects `choices: [...]` in the raw response;
        the Responses API uses `output: [...]` and `usage.input_tokens` /
        `usage.output_tokens` (not `prompt_tokens` / `completion_tokens`).
        The result was a KeyError 'choices' inside
        `convert_to_model_response_object`, swallowed by the surrounding
        try/except, and the SpendLogs row was written with zero tokens
        and zero spend.

        FIX: use the dedicated `OpenAIResponsesAPIConfig.transform_response_api_response`
        for the Responses branch.

        This test exercises the REAL transformer (no mocked
        `get_provider_config`) so that running it against the un-fixed
        handler raises and running it against the fixed handler succeeds.
        """
        mock_completion_cost.return_value = 0.000050
        mock_get_standard_logging.return_value = {"test": "logging_payload"}

        # A real-shaped Azure / OpenAI Responses API payload — NO `choices`,
        # uses `output` and `usage.input_tokens` / `usage.output_tokens`.
        responses_api_body = {
            "id": "resp_abc123",
            "object": "response",
            "created_at": 1677652288,
            "model": "gpt-4o-2024-08-06",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Hello!",
                        }
                    ],
                }
            ],
            "usage": {
                "input_tokens": 20,
                "output_tokens": 15,
                "total_tokens": 35,
            },
        }

        mock_httpx_response = self._create_mock_httpx_response(responses_api_body)
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()

        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gpt-4o",
            "custom_llm_provider": "openai",
        }

        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=responses_api_body,
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/responses",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"model": "gpt-4o", "input": "Tell me about AI"},
            **kwargs,
        )

        # Pre-fix this assertion fails — the handler swallows the
        # KeyError raised by the chat-completions transformer and falls
        # back to the passthrough_chat_handler which yields a different
        # response_cost value. Post-fix, the Responses transformer
        # succeeds and we get the mocked 0.000050.
        assert result is not None
        assert result["kwargs"]["response_cost"] == 0.000050
        assert result["kwargs"]["model"] == "gpt-4o"

        # `completion_cost` must be called with the responses call type
        # and a `ResponsesAPIResponse` (not a `ModelResponse`).
        mock_completion_cost.assert_called_once()
        call_kwargs = mock_completion_cost.call_args[1]
        assert call_kwargs["call_type"] == "responses"

        from litellm.types.llms.openai import ResponsesAPIResponse

        assert isinstance(call_kwargs["completion_response"], ResponsesAPIResponse), (
            "completion_response must be a ResponsesAPIResponse; passing a "
            "chat-completions ModelResponse means the Responses transformer "
            "isn't being used and we're back in the bug."
        )


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
            response_data = {
                "id": "test",
                "choices": [{"message": {"content": "Hello"}}],
            }

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
            request_body={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            request_method="POST",
        )

    def test_is_openai_route_detection(self):
        """Test OpenAI route detection in the main success handler"""
        # Positive cases
        assert self.handler.is_openai_route("https://api.openai.com/v1/chat/completions") == True
        assert self.handler.is_openai_route("https://openai.azure.com/v1/chat/completions") == True
        assert self.handler.is_openai_route("https://api.openai.com/v1/models") == True
        # Azure OpenAI on the shared Cognitive Services domain, identified by an
        # OpenAI-style path segment.
        assert (
            self.handler.is_openai_route("https://my-resource.cognitiveservices.azure.com/v1/chat/completions") == True
        )

        # Negative cases
        assert self.handler.is_openai_route("http://localhost:4000/openai/v1/chat/completions") == False
        assert self.handler.is_openai_route("https://api.anthropic.com/v1/messages") == False
        assert self.handler.is_openai_route("https://api.assemblyai.com/v2/transcript") == False
        # Non-OpenAI Azure Cognitive Services share the `cognitiveservices.azure.com`
        # domain but must NOT be classified as OpenAI routes (no OpenAI path segment).
        assert (
            self.handler.is_openai_route("https://my-resource.cognitiveservices.azure.com/speechtotext/v3.1/recognize")
            == False
        )
        assert (
            self.handler.is_openai_route("https://my-resource.cognitiveservices.azure.com/vision/v3.2/analyze") == False
        )
        # A look-alike domain that merely contains an OpenAI host as a substring
        # must be rejected by the suffix-based hostname match.
        assert (
            self.handler.is_openai_route("https://cognitiveservices.azure.com.attacker.example/v1/chat/completions")
            == False
        )
        assert self.handler.is_openai_route("") == False

    def test_is_supported_openai_endpoint_includes_responses_api(self):
        """Regression test for the outer dispatch gate.

        `_is_supported_openai_endpoint` is the gate that decides whether the
        OpenAI handler runs for a given URL. Before this gate accepted the
        Responses API, calls to `/v1/responses` would fail the gate and the
        handler's `elif is_responses:` branch was unreachable in the live
        success-handler pipeline — every Responses-API call landed in
        `LiteLLM_SpendLogs` with zero tokens / zero spend even though the
        handler had a Responses branch internally.

        This test exercises the dispatch decision directly so future
        refactors of `_is_supported_openai_endpoint` can't silently
        remove Responses from the OR-chain without a test failure.
        """
        # Responses must be supported on api.openai.com and openai.azure.com.
        assert self.handler._is_supported_openai_endpoint("https://api.openai.com/v1/responses") is True
        assert self.handler._is_supported_openai_endpoint("https://openai.azure.com/v1/responses") is True
        # The other supported endpoints stay supported (no regression).
        assert self.handler._is_supported_openai_endpoint("https://api.openai.com/v1/chat/completions") is True
        assert self.handler._is_supported_openai_endpoint("https://api.openai.com/v1/images/generations") is True
        assert self.handler._is_supported_openai_endpoint("https://api.openai.com/v1/images/edits") is True
        # Unsupported OpenAI endpoints (e.g. /v1/models) still return False.
        assert self.handler._is_supported_openai_endpoint("https://api.openai.com/v1/models") is False
        assert (
            self.handler._is_supported_openai_endpoint(
                "https://my-resource.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings"
            )
            is False
        )

    def test_is_supported_openai_endpoint_includes_embeddings(self):
        assert self.handler._is_supported_openai_endpoint("https://api.openai.com/v1/embeddings") is True
        assert self.handler._is_supported_openai_endpoint("https://openai.azure.com/v1/embeddings") is True

    def test_is_cohere_route_does_not_match_openai_embeddings(self):
        assert self.handler.is_cohere_route("https://api.cohere.com/v1/embed") is True
        assert self.handler.is_cohere_route("https://api.cohere.com/v2/chat") is True
        assert self.handler.is_cohere_route("https://api.openai.com/v1/embeddings") is False
        assert self.handler.is_cohere_route("https://api.cohere.com/v1/rerank") is False
        assert self.handler.is_cohere_route("http://localhost:4000/openai_passthrough/v1/embeddings") is False

    @patch("litellm.completion_cost")
    @patch("litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload")
    def test_openai_passthrough_handler_embeddings_sets_response_cost(
        self, mock_get_standard_logging, mock_completion_cost
    ):
        mock_completion_cost.return_value = 2.8e-07
        mock_get_standard_logging.return_value = {"test": "logging_payload"}

        response_body = {
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2],
                }
            ],
            "usage": {"prompt_tokens": 14, "total_tokens": 14},
        }
        mock_httpx_response = self._create_mock_httpx_response(response_body)
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = PassthroughStandardLoggingPayload(
            url="https://api.openai.com/v1/embeddings",
            request_body={
                "model": "text-embedding-3-small",
                "input": "PROOF_SENTINEL_TEXT",
            },
            request_method="POST",
        )
        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "litellm_params": {},
        }

        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=response_body,
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/embeddings",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={
                "model": "text-embedding-3-small",
                "input": "PROOF_SENTINEL_TEXT",
            },
            **kwargs,
        )

        assert result["result"] is not None
        assert result["kwargs"]["response_cost"] == 2.8e-07
        assert result["kwargs"]["model"] == "text-embedding-3-small"
        assert result["kwargs"]["custom_llm_provider"] == "openai"
        assert result["result"]._hidden_params["response_cost"] == 2.8e-07
        mock_completion_cost.assert_called_once()
        assert mock_completion_cost.call_args.kwargs["call_type"] == "aembedding"
        assert mock_logging_obj.model_call_details["response_cost"] == 2.8e-07

    @patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler.OpenAIPassthroughLoggingHandler.passthrough_chat_handler"
    )
    @patch("litellm.completion_cost")
    def test_openai_passthrough_handler_embeddings_without_model_falls_back(
        self, mock_completion_cost, mock_chat_handler
    ):
        mock_chat_handler.return_value = {"result": None, "kwargs": {}}
        response_body = {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=self._create_mock_httpx_response(response_body),
            response_body=response_body,
            logging_obj=self._create_mock_logging_obj(),
            url_route="https://api.openai.com/v1/embeddings",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"input": "PROOF_SENTINEL_TEXT"},
            passthrough_logging_payload=PassthroughStandardLoggingPayload(
                url="https://api.openai.com/v1/embeddings",
                request_body={"input": "PROOF_SENTINEL_TEXT"},
                request_method="POST",
            ),
        )
        mock_completion_cost.assert_not_called()
        mock_chat_handler.assert_called_once()
        assert result == {"result": None, "kwargs": {}}

    def test_openai_passthrough_handler_embeddings_unmapped_model_logs_zero_cost(self):
        response_body = {
            "object": "list",
            "model": "lit5787-unmapped-embeddings-deployment",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "usage": {"prompt_tokens": 9, "total_tokens": 9},
        }
        mock_logging_obj = self._create_mock_logging_obj()
        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=self._create_mock_httpx_response(response_body),
            response_body=response_body,
            logging_obj=mock_logging_obj,
            url_route="https://my-resource.openai.azure.com/openai/v1/embeddings",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={
                "model": "lit5787-unmapped-embeddings-deployment",
                "input": "spend probe",
            },
            passthrough_logging_payload=PassthroughStandardLoggingPayload(
                url="https://my-resource.openai.azure.com/openai/v1/embeddings",
                request_body={
                    "model": "lit5787-unmapped-embeddings-deployment",
                    "input": "spend probe",
                },
                request_method="POST",
            ),
            litellm_params={},
        )

        assert result["result"] is not None
        assert result["result"].usage.prompt_tokens == 9
        assert result["kwargs"]["response_cost"] == 0.0
        assert result["kwargs"]["model"] == "lit5787-unmapped-embeddings-deployment"
        assert result["result"]._hidden_params["response_cost"] == 0.0
        assert mock_logging_obj.model_call_details["response_cost"] == 0.0

    def test_openai_passthrough_handler_embeddings_error_skips_chat_fallback(self):
        response_body = {
            "object": "list",
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 9, "total_tokens": 9},
        }
        kwargs_in = {
            "passthrough_logging_payload": PassthroughStandardLoggingPayload(
                url="https://api.openai.com/v1/embeddings",
                request_body={"model": "text-embedding-3-small", "input": "spend probe"},
                request_method="POST",
            ),
            "litellm_params": {},
        }
        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=self._create_mock_httpx_response(response_body),
            response_body=response_body,
            logging_obj=self._create_mock_logging_obj(),
            url_route="https://api.openai.com/v1/embeddings",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"model": "text-embedding-3-small", "input": "spend probe"},
            **kwargs_in,
        )

        assert result["result"] is None
        assert result["kwargs"]["passthrough_logging_payload"] == kwargs_in["passthrough_logging_payload"]

    @patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler.OpenAIPassthroughLoggingHandler.openai_passthrough_handler"
    )
    @pytest.mark.asyncio
    async def test_success_handler_dispatches_embeddings_to_openai_handler(self, mock_openai_handler):
        mock_openai_handler.return_value = {
            "result": {"object": "list"},
            "kwargs": {
                "response_cost": 2.8e-07,
                "model": "text-embedding-3-small",
                "custom_llm_provider": "openai",
            },
        }

        mock_httpx_response = MagicMock(spec=httpx.Response)
        mock_httpx_response.text = (
            '{"object":"list","model":"text-embedding-3-small",'
            '"data":[{"object":"embedding","index":0,"embedding":[0.1]}],'
            '"usage":{"prompt_tokens":14,"total_tokens":14}}'
        )

        mock_logging_obj = AsyncMock()
        mock_logging_obj.model_call_details = {}
        mock_logging_obj.async_success_handler = AsyncMock()

        passthrough_payload = PassthroughStandardLoggingPayload(
            url="https://api.openai.com/v1/embeddings",
            request_body={
                "model": "text-embedding-3-small",
                "input": "PROOF_SENTINEL_TEXT",
            },
            request_method="POST",
        )

        await self.handler.pass_through_async_success_handler(
            httpx_response=mock_httpx_response,
            response_body={
                "object": "list",
                "model": "text-embedding-3-small",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                "usage": {"prompt_tokens": 14, "total_tokens": 14},
            },
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/embeddings",
            result="",
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body={
                "model": "text-embedding-3-small",
                "input": "PROOF_SENTINEL_TEXT",
            },
            passthrough_logging_payload=passthrough_payload,
        )

        mock_openai_handler.assert_called_once()
        assert mock_openai_handler.call_args.kwargs["url_route"] == "https://api.openai.com/v1/embeddings"

    @patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler.OpenAIPassthroughLoggingHandler.openai_passthrough_handler"
    )
    @pytest.mark.asyncio
    async def test_success_handler_dispatches_responses_api_to_openai_handler(self, mock_openai_handler):
        """End-to-end dispatch test for the Responses API path.

        Pre-fix: `_is_supported_openai_endpoint` returned False for
        `/v1/responses` URLs, so the OpenAI handler was never called.
        This test would fail (mock never invoked) on the un-fixed
        success_handler — passes only when the dispatch gate accepts
        Responses URLs.
        """
        mock_openai_handler.return_value = {
            "result": {"id": "resp_abc123"},
            "kwargs": {
                "response_cost": 0.0001,
                "model": "gpt-4o",
                "custom_llm_provider": "openai",
            },
        }

        mock_httpx_response = MagicMock(spec=httpx.Response)
        mock_httpx_response.text = (
            '{"id": "resp_abc123", "object": "response", '
            '"output": [], "usage": {"input_tokens": 5, "output_tokens": 3}}'
        )

        mock_logging_obj = AsyncMock()
        mock_logging_obj.model_call_details = {}
        mock_logging_obj.async_success_handler = AsyncMock()

        passthrough_payload = PassthroughStandardLoggingPayload(
            url="https://api.openai.com/v1/responses",
            request_body={"model": "gpt-4o", "input": "Hello"},
            request_method="POST",
        )

        await self.handler.pass_through_async_success_handler(
            httpx_response=mock_httpx_response,
            response_body={
                "id": "resp_abc123",
                "object": "response",
                "output": [],
                "usage": {"input_tokens": 5, "output_tokens": 3},
            },
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/responses",
            result="",
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body={"model": "gpt-4o", "input": "Hello"},
            passthrough_logging_payload=passthrough_payload,
        )

        # The OpenAI handler MUST have been invoked. Pre-fix the dispatch
        # gate filtered Responses URLs out and the mock was never called.
        mock_openai_handler.assert_called_once()
        # And we can verify it was dispatched with the Responses URL.
        call_kwargs = mock_openai_handler.call_args.kwargs
        assert call_kwargs["url_route"] == "https://api.openai.com/v1/responses"

    @patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler.OpenAIPassthroughLoggingHandler.openai_passthrough_handler"
    )
    @pytest.mark.asyncio
    async def test_success_handler_calls_openai_handler(self, mock_openai_handler):
        """Test that the success handler calls our OpenAI handler for OpenAI routes"""
        # Arrange
        mock_openai_handler.return_value = {
            "result": {"id": "chatcmpl-123"},
            "kwargs": {
                "response_cost": 0.000045,
                "model": "gpt-4o",
                "custom_llm_provider": "openai",
            },
        }

        mock_httpx_response = MagicMock(spec=httpx.Response)
        mock_httpx_response.text = '{"id": "chatcmpl-123", "choices": [{"message": {"content": "Hello"}}]}'

        mock_logging_obj = AsyncMock()
        mock_logging_obj.model_call_details = {}
        mock_logging_obj.async_success_handler = AsyncMock()

        passthrough_payload = PassthroughStandardLoggingPayload(
            url="https://api.openai.com/v1/chat/completions",
            request_body={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            request_method="POST",
        )

        # Act
        result = await self.handler.pass_through_async_success_handler(
            httpx_response=mock_httpx_response,
            response_body={
                "id": "chatcmpl-123",
                "choices": [{"message": {"content": "Hello"}}],
            },
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/chat/completions",
            result="",
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
            },
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
            request_body={
                "model": "claude-3-sonnet",
                "messages": [{"role": "user", "content": "Hello"}],
            },
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
            request_body={
                "model": "claude-3-sonnet",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            passthrough_logging_payload=passthrough_payload,
        )

        # Assert - Should call the base handler, not our OpenAI handler
        self.handler._handle_logging.assert_called_once()

    @patch("litellm.cost_calculator.default_image_cost_calculator")
    def test_calculate_image_generation_cost(self, mock_image_cost_calculator):
        """Test image generation cost calculation"""
        # Arrange
        mock_image_cost_calculator.return_value = 0.040
        model = "dall-e-3"
        response_body = {
            "data": [
                {
                    "url": "https://example.com/image1.png",
                    "revised_prompt": "A beautiful sunset over the ocean",
                }
            ]
        }
        request_body = {
            "model": "dall-e-3",
            "prompt": "A beautiful sunset over the ocean",
            "n": 1,
            "size": "1024x1024",
            "quality": "standard",
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

    @patch("litellm.cost_calculator.default_image_cost_calculator")
    def test_calculate_image_editing_cost(self, mock_image_cost_calculator):
        """Test image editing cost calculation"""
        # Arrange
        mock_image_cost_calculator.return_value = 0.020
        model = "dall-e-2"
        response_body = {
            "data": [
                {
                    "url": "https://example.com/edited_image.png",
                    "revised_prompt": "A beautiful sunset over the ocean with added clouds",
                }
            ]
        }
        request_body = {
            "model": "dall-e-2",
            "prompt": "Add clouds to the sky",
            "n": 1,
            "size": "1024x1024",
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

    @patch("litellm.cost_calculator.default_image_cost_calculator")
    def test_openai_passthrough_handler_image_generation(self, mock_image_cost_calculator):
        """Test successful cost tracking for OpenAI image generation"""
        # Arrange
        mock_image_cost_calculator.return_value = 0.040

        mock_image_response = {
            "data": [
                {
                    "url": "https://example.com/image1.png",
                    "revised_prompt": "A beautiful sunset over the ocean",
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
            "quality": "standard",
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
            **kwargs,
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

    @patch("litellm.cost_calculator.default_image_cost_calculator")
    def test_openai_passthrough_handler_image_editing(self, mock_image_cost_calculator):
        """Test successful cost tracking for OpenAI image editing"""
        # Arrange
        mock_image_cost_calculator.return_value = 0.020

        mock_image_response = {
            "data": [
                {
                    "url": "https://example.com/edited_image.png",
                    "revised_prompt": "A beautiful sunset over the ocean with added clouds",
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
            "size": "1024x1024",
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
            **kwargs,
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


class TestOpenAIPassthroughResponsesStreamingSpendLog:
    """A streamed OpenAI-passthrough `/v1/responses` call must write a priced spend
    log row (#36523).

    `_handle_logging_openai_collected_chunks` received `url_route` and ignored it, so
    a Responses SSE stream was reassembled by the chat-completions chunk parser. The
    assembled object carried a synthetic `chatcmpl-` id, `model=None` and zero usage,
    and the spend row landed at zero tokens and zero spend while the identical
    buffered call priced exactly.
    """

    RESPONSE_ID = "resp_0c72ddebf05f8751"
    MODEL_MAP_KEY = "gpt-4o-mini-2024-07-18"
    INPUT_TOKENS = 14
    OUTPUT_TOKENS = 2

    def setup_method(self):
        self.start_time = datetime.now()
        self.end_time = datetime.now()
        rates = litellm.model_cost[self.MODEL_MAP_KEY]
        self.expected_spend = (
            self.INPUT_TOKENS * rates["input_cost_per_token"]
            + self.OUTPUT_TOKENS * rates["output_cost_per_token"]
        )

    def _responses_stream_chunks(self) -> List[str]:
        """Usage arrives only on the terminal `response.completed` event, nested under
        `response` as `input_tokens` / `output_tokens`. No event carries a top-level
        `id`, `model` or `usage`."""
        response_body = {
            "id": self.RESPONSE_ID,
            "object": "response",
            "created_at": 1786374786,
            "model": self.MODEL_MAP_KEY,
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": {},
            "parallel_tool_calls": True,
            "temperature": 1.0,
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1.0,
        }
        created_event = {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**response_body, "status": "in_progress", "output": [], "usage": None},
        }
        delta_event = {
            "type": "response.output_text.delta",
            "sequence_number": 3,
            "item_id": "msg_abc",
            "output_index": 0,
            "content_index": 0,
            "delta": "OK",
        }
        completed_event = {
            "type": "response.completed",
            "sequence_number": 8,
            "response": {
                **response_body,
                "status": "completed",
                "output": [
                    {
                        "id": "msg_abc",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "OK", "annotations": []}],
                    }
                ],
                "usage": {
                    "input_tokens": self.INPUT_TOKENS,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": self.OUTPUT_TOKENS,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": self.INPUT_TOKENS + self.OUTPUT_TOKENS,
                },
            },
        }
        return [
            f"data: {json.dumps(created_event)}",
            f"data: {json.dumps(delta_event)}",
            f"data: {json.dumps(completed_event)}",
            "data: [DONE]",
        ]

    def _logging_obj(self) -> LiteLLMLoggingObj:
        logging_obj = LiteLLMLoggingObj(
            model="gpt-4o-mini",
            messages=[],
            stream=True,
            call_type="pass_through_endpoint",
            start_time=self.start_time,
            litellm_call_id="323dfe4f-2741-4473-b6e2-000000000000",
            function_id="1234",
        )
        logging_obj.model_call_details["custom_llm_provider"] = "openai"
        return logging_obj

    def test_streamed_responses_passthrough_spend_log_is_priced(self):
        """The spend row books the same tokens, spend and `resp_` id as the buffered call."""
        result = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
            litellm_logging_obj=self._logging_obj(),
            passthrough_success_handler_obj=None,
            url_route="https://api.openai.com/v1/responses",
            request_body={
                "model": "gpt-4o-mini",
                "input": "Say hi in one word.",
                "max_output_tokens": 16,
                "stream": True,
            },
            endpoint_type=None,
            start_time=self.start_time,
            all_chunks=self._responses_stream_chunks(),
            end_time=self.end_time,
        )

        spend_log_row = get_logging_payload(
            kwargs=result["kwargs"],
            response_obj=result["result"],
            start_time=self.start_time,
            end_time=self.end_time,
        )

        assert spend_log_row["prompt_tokens"] == self.INPUT_TOKENS
        assert spend_log_row["completion_tokens"] == self.OUTPUT_TOKENS
        assert spend_log_row["total_tokens"] == self.INPUT_TOKENS + self.OUTPUT_TOKENS
        assert spend_log_row["spend"] == self.expected_spend
        assert spend_log_row["request_id"] == self.RESPONSE_ID
        assert spend_log_row["model"] == "gpt-4o-mini"

        row_metadata = json.loads(spend_log_row["metadata"])
        assert row_metadata["model_map_information"]["model_map_key"] == self.MODEL_MAP_KEY


class TestOpenAIPassthroughEmbeddingsSpendLog:
    """An OpenAI-passthrough `/v1/embeddings` call must write a priced spend log row
    (#36646).

    `_is_supported_openai_endpoint` ORed four route predicates, none of which matched
    `/v1/embeddings`, so the dispatcher never entered the OpenAI handler and billable
    embedding tokens produced no spend row at all, under-enforcing key budgets.
    """

    EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
    MODEL = "text-embedding-3-small"
    PROMPT_TOKENS = 14
    CALL_ID = "11024cc3-b143-4a63-954a-ec06081df768"

    def setup_method(self):
        self.start_time = datetime.now()
        self.end_time = datetime.now()
        self.expected_spend = self.PROMPT_TOKENS * litellm.model_cost[self.MODEL]["input_cost_per_token"]
        self.response_body = {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.0, 1.0]}],
            "model": self.MODEL,
            "usage": {"prompt_tokens": self.PROMPT_TOKENS, "total_tokens": self.PROMPT_TOKENS},
        }
        self.request_body = {"model": self.MODEL, "input": "hello"}

    def _create_mock_httpx_response(self) -> httpx.Response:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.text = json.dumps(self.response_body)
        mock_response.json.return_value = self.response_body
        mock_response.headers = {"content-type": "application/json"}
        return mock_response

    def _logging_obj(self) -> LiteLLMLoggingObj:
        logging_obj = LiteLLMLoggingObj(
            model=self.MODEL,
            messages=[],
            stream=False,
            call_type="pass_through_endpoint",
            start_time=self.start_time,
            litellm_call_id=self.CALL_ID,
            function_id="1234",
        )
        logging_obj.model_call_details["passthrough_logging_payload"] = PassthroughStandardLoggingPayload(
            url=self.EMBEDDINGS_URL,
            request_body=self.request_body,
            request_method="POST",
        )
        return logging_obj

    def test_embeddings_passthrough_spend_log_is_priced(self):
        """The dispatched call books prompt tokens and cost onto the spend row."""
        dispatched = PassThroughEndpointLogging().normalize_llm_passthrough_logging_payload(
            httpx_response=self._create_mock_httpx_response(),
            response_body=self.response_body,
            request_body=self.request_body,
            logging_obj=self._logging_obj(),
            url_route=self.EMBEDDINGS_URL,
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            custom_llm_provider="openai",
            litellm_call_id=self.CALL_ID,
            litellm_params={},
            call_type="pass_through_endpoint",
        )

        assert dispatched["standard_logging_response_object"] is not None
        assert dispatched["kwargs"]["response_cost"] == self.expected_spend

        spend_log_row = get_logging_payload(
            kwargs=dispatched["kwargs"],
            response_obj=dispatched["standard_logging_response_object"],
            start_time=self.start_time,
            end_time=self.end_time,
        )

        assert spend_log_row["prompt_tokens"] == self.PROMPT_TOKENS
        assert spend_log_row["total_tokens"] == self.PROMPT_TOKENS
        assert spend_log_row["spend"] == self.expected_spend
        assert spend_log_row["model"] == self.MODEL
        assert spend_log_row["custom_llm_provider"] == "openai"
        assert spend_log_row["request_id"] == self.CALL_ID


if __name__ == "__main__":
    pytest.main([__file__])
