from dataclasses import dataclass, field, replace
import asyncio
import contextlib
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from types import MappingProxyType, SimpleNamespace
from typing import Any, Dict, Final, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from prisma.errors import TableNotFoundError, UniqueViolationError
from pydantic import BaseModel, ConfigDict


import litellm
from litellm.proxy import proxy_server
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import EnterpriseCallbackControls
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.token_counter import high_detail_image_token_upper_bound
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler import (
    OpenAIPassthroughLoggingHandler,
    count_relayed_prompt_tokens,
)
from litellm.proxy._types import SpendLogsPayload, UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db.db_lookup_gate import DBLookupDeadlineExceeded
from litellm.redis.stores.auth_cache import AuthObject
from litellm.interactions.background_cost_polling import PollSchedule
from litellm.proxy.pass_through_endpoints.background_response_creator import (
    BackgroundResponseCreateContext,
    PrismaBackgroundResponseCreatorStore,
)
from litellm.proxy.pass_through_endpoints.background_response_polling import (
    BACKGROUND_RESPONSE_POLL_LEASE_SECONDS,
    BACKGROUND_RESPONSE_POLL_SCHEDULE,
    BackgroundResponsePoller,
    BackgroundResponsePollRequest,
    Fetched,
    FetchOutcome,
)
from litellm.proxy.pass_through_endpoints.background_response_settlement import (
    PrismaBackgroundResponseSettlementStore,
)
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.proxy.pass_through_endpoints.managed_id_codec import new_managed_id
from litellm.proxy.pass_through_endpoints.streaming_handler import PassThroughStreamContext, PassThroughStreamingHandler
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import TeamCallbackWiring
from litellm.proxy.pass_through_endpoints.success_handler import (
    CreatorLoggingWiring,
    PassThroughEndpointLogging,
    proxy_creator_logging_wiring,
)
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.caching.caching import DualCache
from litellm.proxy.hooks.parallel_request_limiter import PROXY_MaxParallelRequestsHandler
from litellm.proxy.utils import InternalUsageCache
from litellm.types.utils import ModelResponse, Usage
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    EndpointType,
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
            custom_pricing=False,
            router_model_id=None,
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
            custom_pricing=False,
            router_model_id=None,
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
            custom_pricing=False,
            router_model_id=None,
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
        assert result["result"].hidden_params["response_cost"] == 2.8e-07
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
        anthropic_message = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-sonnet",
            "content": [{"type": "text", "text": "Hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        mock_httpx_response = MagicMock(spec=httpx.Response)
        mock_httpx_response.text = json.dumps(anthropic_message)
        mock_httpx_response.json.return_value = anthropic_message
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
        self.handler.handle_logging = AsyncMock()

        # Act
        result = await self.handler.pass_through_async_success_handler(
            httpx_response=mock_httpx_response,
            response_body=anthropic_message,
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
        self.handler.handle_logging.assert_called_once()

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
            model_info=None,
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
            model_info=None,
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

        # Test the response_cost_calculator method
        calculated_cost = logging_obj.response_cost_calculator(result=image_response)

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
        assert result["result"].hidden_params["response_cost"] == 0.040

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
        assert result["result"].hidden_params["response_cost"] == 0.020

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

    def _expected_spend(self) -> float:
        rates = litellm.model_cost[self.MODEL_MAP_KEY]
        return self.INPUT_TOKENS * rates["input_cost_per_token"] + self.OUTPUT_TOKENS * rates["output_cost_per_token"]

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

    def test_streamed_responses_passthrough_spend_log_is_priced(self, local_model_cost_map):
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
        assert spend_log_row["spend"] == pytest.approx(self._expected_spend())
        assert spend_log_row["request_id"] == self.RESPONSE_ID
        assert spend_log_row["model"] == "gpt-4o-mini"

        row_metadata = json.loads(spend_log_row["metadata"])
        assert row_metadata["model_map_information"]["model_map_key"] == self.MODEL_MAP_KEY

    def test_resumed_responses_stream_spend_log_is_labeled_with_the_streamed_model(self, local_model_cost_map):
        result: Final = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
            litellm_logging_obj=self._logging_obj(),
            passthrough_success_handler_obj=None,
            url_route=f"https://api.openai.com/v1/responses/{self.RESPONSE_ID}?stream=true",
            request_body={},
            endpoint_type=None,
            start_time=self.start_time,
            all_chunks=self._responses_stream_chunks(),
            end_time=self.end_time,
        )

        spend_log_row: Final = get_logging_payload(
            kwargs=result["kwargs"],
            response_obj=result["result"],
            start_time=self.start_time,
            end_time=self.end_time,
        )

        assert spend_log_row["model"] == self.MODEL_MAP_KEY
        assert spend_log_row["spend"] == pytest.approx(self._expected_spend())


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
        self.response_body = {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.0, 1.0]}],
            "model": self.MODEL,
            "usage": {"prompt_tokens": self.PROMPT_TOKENS, "total_tokens": self.PROMPT_TOKENS},
        }
        self.request_body = {"model": self.MODEL, "input": "hello"}

    def _expected_spend(self) -> float:
        return self.PROMPT_TOKENS * litellm.model_cost[self.MODEL]["input_cost_per_token"]

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

    def test_embeddings_passthrough_spend_log_is_priced(self, local_model_cost_map):
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
        assert dispatched["kwargs"]["response_cost"] == pytest.approx(self._expected_spend())

        spend_log_row = get_logging_payload(
            kwargs=dispatched["kwargs"],
            response_obj=dispatched["standard_logging_response_object"],
            start_time=self.start_time,
            end_time=self.end_time,
        )

        assert spend_log_row["prompt_tokens"] == self.PROMPT_TOKENS
        assert spend_log_row["total_tokens"] == self.PROMPT_TOKENS
        assert spend_log_row["spend"] == pytest.approx(self._expected_spend())
        assert spend_log_row["model"] == self.MODEL
        assert spend_log_row["custom_llm_provider"] == "openai"
        assert spend_log_row["request_id"] == self.CALL_ID


RESPONSES_URL: Final = "https://api.openai.com/v1/responses"
STORED_RESPONSE_ID: Final = "resp_080b10ff5d740c69006ad3de732e5087"
DISPATCH_TIME: Final = datetime(2026, 10, 17, 13, 47, 3)
RESPONSES_USAGE: Final = {
    "input_tokens": 11,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 5,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 16,
}
CREATE_REQUEST: Final = {"model": "gpt-5.4-mini", "input": "Say hi in one word"}
COMPACT_REQUEST: Final = {"model": "gpt-5.4-mini", "input": "Remember the number 42."}
COMPACTION: Final = {
    "id": STORED_RESPONSE_ID,
    "object": "response.compaction",
    "created_at": 1792270025,
    "output": [{"id": "cmp_082492f9", "type": "compaction", "encrypted_content": "gAAAA"}],
    "usage": RESPONSES_USAGE,
}
AGENTS_SESSIONS_URL: Final = "https://api.openai.com/v1/agents/sessions"
ASSISTANT_MESSAGE: Final = {
    "id": "msg_080b10ff5d740c69",
    "type": "message",
    "status": "completed",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "Hi", "annotations": []}],
}


def _stored_response(status: str, background: bool, usage: Mapping[str, object] | None) -> dict[str, object]:
    return {
        "id": STORED_RESPONSE_ID,
        "object": "response",
        "created_at": 1792269939,
        "status": status,
        "background": background,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "model": "gpt-5.4-mini-2026-03-17",
        "output": [ASSISTANT_MESSAGE] if usage else [],
        "parallel_tool_calls": True,
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 0.98,
        "usage": usage,
    }


def _agent_session(agent_model: str | None) -> dict[str, object]:
    agent: Final = (
        {}
        if agent_model is None
        else {
            "agent": {"id": "agent_6a0b0f8e", "model": agent_model, "name": None, "service_tier": "auto", "tools": []}
        }
    )
    return {
        "id": "sess_6a0b0f8e2c1d4e5f",
        "object": "agent.session",
        "created_at": 1792270025,
        "last_active_at": 1792270025,
        "status": "in_progress",
        "metadata": {},
        "required_actions": [],
        "vault_ids": [],
        "usage": None,
        **agent,
    }


@dataclass(frozen=True, slots=True)
class _LoggedCall:
    spend_row: SpendLogsPayload
    callback_response_id: str


class _CallbackPayloadId(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str


class _SpendRowRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.call: Final[asyncio.Future[_LoggedCall]] = asyncio.get_running_loop().create_future()

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.call.set_result(
            _LoggedCall(
                spend_row=get_logging_payload(
                    kwargs=kwargs, response_obj=response_obj, start_time=start_time, end_time=end_time
                ),
                callback_response_id=_CallbackPayloadId.model_validate(kwargs.get("standard_logging_object")).id,
            )
        )


class _SettlementTable:
    def __init__(self) -> None:
        self.settled_response_ids: Final[set[str]] = set()

    async def create_many(self, data: Sequence[Mapping[str, str]], *, skip_duplicates: bool) -> int:
        new_ids: Final = {row["response_id"] for row in data} - self.settled_response_ids
        if not skip_duplicates and len(new_ids) < len(data):
            raise UniqueViolationError({"user_facing_error": {"error_code": "P2002"}})
        self.settled_response_ids.update(new_ids)
        return len(new_ids)

    async def delete_many(self, *, where: Mapping[str, str]) -> int:
        released: Final = where["response_id"] in self.settled_response_ids
        self.settled_response_ids.discard(where["response_id"])
        return int(released)


class _UnreleasableSettlementTable(_SettlementTable):
    async def delete_many(self, *, where: Mapping[str, str]) -> int:
        raise ConnectionResetError("server closed the connection unexpectedly")


class _UnreachableSettlementTable:
    async def create_many(self, data: Sequence[Mapping[str, str]], *, skip_duplicates: bool) -> int:
        raise ConnectionResetError("server closed the connection unexpectedly")


@dataclass(frozen=True, slots=True)
class _StalledSettlementTable:
    table: _SettlementTable

    async def create_many(self, data: Sequence[Mapping[str, str]], *, skip_duplicates: bool) -> int:
        await asyncio.sleep(3)
        return await self.table.create_many(data, skip_duplicates=skip_duplicates)


@dataclass(frozen=True, slots=True)
class _SlowAckSettlementTable:
    table: _SettlementTable
    inserted: asyncio.Event

    async def create_many(self, data: Sequence[Mapping[str, str]], *, skip_duplicates: bool) -> int:
        inserted_count: Final = await self.table.create_many(data, skip_duplicates=skip_duplicates)
        self.inserted.set()
        await asyncio.sleep(0.3)
        return inserted_count


def _settling_passthrough_logging(
    table: _SettlementTable | _UnreachableSettlementTable | _StalledSettlementTable | _SlowAckSettlementTable | None,
    timeout_seconds: float = 0.05,
) -> PassThroughEndpointLogging:
    return PassThroughEndpointLogging(
        background_response_settlement=PrismaBackgroundResponseSettlementStore(
            table=lambda: table, timeout_seconds=timeout_seconds
        )
    )


def _completed_stream_bytes(response: Mapping[str, object]) -> list[bytes]:
    completed: Final = {"type": "response.completed", "sequence_number": 9, "response": response}
    return [f"event: response.completed\ndata: {json.dumps(completed)}\n\n".encode()]


class TestOpenAIPassthroughResponsesSubRouteBilling:
    def _dispatch(
        self,
        method: str,
        sub_route: str,
        request_body: dict[str, object],
        response_body: dict[str, object],
        base_url: str = RESPONSES_URL,
    ):
        url_route: Final = f"{base_url}{sub_route}"
        logging_obj: Final = LiteLLMLoggingObj(
            model="gpt-5.4-mini",
            messages=[],
            stream=False,
            call_type="pass_through_endpoint",
            start_time=DISPATCH_TIME,
            litellm_call_id="6c1f3a52-93d4-4b1e-9f0e-6f2f4f6b9a10",
            function_id="1234",
        )
        return PassThroughEndpointLogging().normalize_llm_passthrough_logging_payload(
            httpx_response=httpx.Response(200, json=response_body, request=httpx.Request(method, url_route)),
            response_body=response_body,
            request_body=request_body,
            logging_obj=logging_obj,
            url_route=url_route,
            result="",
            start_time=DISPATCH_TIME,
            end_time=DISPATCH_TIME,
            cache_hit=False,
            custom_llm_provider="openai",
            litellm_params={},
            call_type="pass_through_endpoint",
        )

    @pytest.mark.parametrize(
        ("method", "sub_route", "request_body", "response_body"),
        [
            pytest.param(
                "POST", "", CREATE_REQUEST, _stored_response("completed", False, RESPONSES_USAGE), id="create"
            ),
            pytest.param(
                "POST",
                "/",
                CREATE_REQUEST,
                _stored_response("completed", False, RESPONSES_USAGE),
                id="create_trailing_slash",
            ),
            pytest.param("POST", "/compact", COMPACT_REQUEST, COMPACTION, id="compact"),
            pytest.param("POST", "/%2563ompact", COMPACT_REQUEST, COMPACTION, id="percent_encoded_compact"),
            pytest.param(
                "GET",
                f"/{STORED_RESPONSE_ID}",
                {},
                _stored_response("completed", True, RESPONSES_USAGE),
                id="background_retrieve_completed",
            ),
            pytest.param(
                "POST",
                f"/{STORED_RESPONSE_ID}/cancel",
                {},
                _stored_response("cancelled", True, RESPONSES_USAGE),
                id="background_cancel",
            ),
        ],
    )
    def test_billed_responses_call_records_cost(
        self,
        local_model_cost_map,
        method: str,
        sub_route: str,
        request_body: dict[str, object],
        response_body: dict[str, object],
    ):
        dispatched: Final = self._dispatch(method, sub_route, request_body, response_body)

        assert dispatched["standard_logging_response_object"] is not None
        assert dispatched["kwargs"]["response_cost"] > 0

    @pytest.mark.parametrize(
        ("method", "sub_route", "request_body", "response_body"),
        [
            pytest.param(
                "POST",
                "/input_tokens",
                {"model": "gpt-5.4-mini", "input": "hi"},
                {"object": "response.input_tokens", "input_tokens": 7},
                id="input_tokens",
            ),
            pytest.param(
                "GET",
                f"/{STORED_RESPONSE_ID}",
                {},
                _stored_response("completed", False, RESPONSES_USAGE),
                id="retrieve",
            ),
            pytest.param(
                "GET",
                f"/{STORED_RESPONSE_ID}/input_items",
                {},
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "msg_074176e3",
                            "type": "message",
                            "status": "completed",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Say hi in one word"}],
                        }
                    ],
                    "first_id": "msg_074176e3",
                    "last_id": "msg_074176e3",
                    "has_more": False,
                },
                id="input_items",
            ),
            pytest.param(
                "DELETE",
                f"/{STORED_RESPONSE_ID}",
                {},
                {"id": STORED_RESPONSE_ID, "object": "response.deleted", "deleted": True},
                id="delete",
            ),
            pytest.param(
                "POST",
                f"/{STORED_RESPONSE_ID}/cancel",
                {},
                _stored_response("cancelled", True, None),
                id="background_cancel_before_any_output",
            ),
        ],
    )
    def test_unbilled_responses_sub_route_records_no_cost_and_no_error(
        self,
        local_model_cost_map,
        caplog: pytest.LogCaptureFixture,
        method: str,
        sub_route: str,
        request_body: dict[str, object],
        response_body: dict[str, object],
    ):
        with caplog.at_level(logging.WARNING):
            dispatched: Final = self._dispatch(method, sub_route, request_body, response_body)

        assert not dispatched["kwargs"].get("response_cost")
        assert [record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR] == []

    @staticmethod
    def _recorded_logging_obj(recorder: _SpendRowRecorder, litellm_call_id: str, stream: bool) -> LiteLLMLoggingObj:
        logging_obj: Final = LiteLLMLoggingObj(
            model="unknown",
            messages=[],
            stream=stream,
            call_type="pass_through_endpoint",
            start_time=DISPATCH_TIME,
            litellm_call_id=litellm_call_id,
            function_id="1234",
            dynamic_async_success_callbacks=[recorder],
        )
        logging_obj.update_environment_variables(
            model="unknown", user="unknown", optional_params={}, litellm_params={}, call_type="pass_through_endpoint"
        )
        return logging_obj

    async def _logged_stream(
        self, url_route: str, request_body: dict[str, object], sse_events: tuple[bytes, ...], litellm_call_id: str
    ) -> _LoggedCall:
        recorder: Final = _SpendRowRecorder()
        await PassThroughStreamingHandler.route_streaming_logging_to_handler(
            litellm_logging_obj=self._recorded_logging_obj(recorder, litellm_call_id, stream=True),
            passthrough_success_handler_obj=PassThroughEndpointLogging(),
            url_route=url_route,
            request_body=request_body,
            endpoint_type=EndpointType.OPENAI,
            start_time=DISPATCH_TIME,
            raw_bytes=sse_events,
            end_time=DISPATCH_TIME,
        )
        return await asyncio.wait_for(recorder.call, timeout=5)

    async def _logged_call(
        self,
        method: str,
        sub_route: str,
        request_body: dict[str, object],
        response_body: dict[str, object],
        litellm_call_id: str,
        base_url: str = RESPONSES_URL,
        *,
        passthrough_logging: PassThroughEndpointLogging | None = None,
    ) -> _LoggedCall:
        recorder: Final = _SpendRowRecorder()
        await self._handled_call(
            method,
            f"{base_url}{sub_route}",
            request_body,
            response_body,
            self._recorded_logging_obj(recorder, litellm_call_id, stream=False),
            passthrough_logging or PassThroughEndpointLogging(),
        )
        return await asyncio.wait_for(recorder.call, timeout=5)

    @staticmethod
    async def _handled_call(
        method: str,
        url_route: str,
        request_body: dict[str, object],
        response_body: dict[str, object],
        logging_obj: LiteLLMLoggingObj,
        passthrough_logging: PassThroughEndpointLogging,
        finalize_before_dispatch: bool = False,
    ) -> None:
        await passthrough_logging.pass_through_async_success_handler(
            httpx_response=httpx.Response(200, json=response_body, request=httpx.Request(method, url_route)),
            response_body=response_body,
            logging_obj=logging_obj,
            url_route=url_route,
            result="",
            start_time=DISPATCH_TIME,
            end_time=DISPATCH_TIME,
            cache_hit=False,
            request_body=request_body,
            passthrough_logging_payload=PassthroughStandardLoggingPayload(
                url=url_route, request_body=request_body, request_method=method
            ),
            custom_llm_provider="openai",
            finalize_before_dispatch=finalize_before_dispatch,
        )

    async def _streamed_call(
        self,
        sub_route: str,
        request_body: dict[str, object],
        raw_bytes: list[bytes],
        litellm_call_id: str,
        passthrough_logging: PassThroughEndpointLogging,
    ) -> _LoggedCall:
        recorder: Final = _SpendRowRecorder()
        logging_obj: Final = self._recorded_logging_obj(recorder, litellm_call_id, stream=True)
        logging_obj.model_call_details["custom_llm_provider"] = "openai"

        await PassThroughStreamingHandler.route_streaming_logging_to_handler(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=passthrough_logging,
            url_route=f"{RESPONSES_URL}{sub_route}",
            request_body=request_body,
            endpoint_type=EndpointType.OPENAI,
            start_time=DISPATCH_TIME,
            raw_bytes=raw_bytes,
            end_time=DISPATCH_TIME,
        )
        return await asyncio.wait_for(recorder.call, timeout=5)

    def _failed_stream_logging_obj(self, litellm_call_id: str) -> LiteLLMLoggingObj:
        logging_obj: Final = self._recorded_logging_obj(_SpendRowRecorder(), litellm_call_id, stream=True)
        logging_obj.model_call_details["custom_llm_provider"] = "openai"
        return logging_obj

    @staticmethod
    async def _failed_stream(
        logging_obj: LiteLLMLoggingObj,
        sub_route: str,
        raw_bytes: list[bytes],
        passthrough_logging: PassThroughEndpointLogging,
    ) -> None:
        await PassThroughStreamingHandler.schedule_stream_failure_logging(
            litellm_logging_obj=logging_obj,
            endpoint_type=EndpointType.OPENAI,
            request_body={},
            raw_bytes=raw_bytes,
            exception=httpx.RemoteProtocolError("peer closed connection without sending complete message body"),
            stream_context=PassThroughStreamContext(passthrough_logging, f"{RESPONSES_URL}{sub_route}", DISPATCH_TIME),
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "sub_route", "response_body"),
        [
            pytest.param(
                "GET",
                f"/{STORED_RESPONSE_ID}",
                _stored_response("completed", True, RESPONSES_USAGE),
                id="retrieve_then_retrieve",
            ),
            pytest.param(
                "POST",
                f"/{STORED_RESPONSE_ID}/cancel",
                _stored_response("cancelled", True, RESPONSES_USAGE),
                id="retrieve_then_cancel",
            ),
        ],
    )
    async def test_a_background_response_read_twice_bills_only_the_first_read(
        self, local_model_cost_map, method: str, sub_route: str, response_body: dict[str, object]
    ):
        passthrough_logging: Final = _settling_passthrough_logging(_SettlementTable())
        later_call_id: Final = "e4a1c7b2-3d58-4f9e-8b06-91c2d7f3a5e8"

        first: Final = await self._logged_call(
            "GET",
            f"/{STORED_RESPONSE_ID}",
            {},
            _stored_response("completed", True, RESPONSES_USAGE),
            "2f9d6b13-8c4e-4a71-b5d0-3e7a1c9f2b64",
            passthrough_logging=passthrough_logging,
        )
        later: Final = await self._logged_call(
            method, sub_route, {}, response_body, later_call_id, passthrough_logging=passthrough_logging
        )

        assert first.spend_row["spend"] > 0
        assert first.spend_row["total_tokens"] > 0
        assert first.spend_row["request_id"].startswith(STORED_RESPONSE_ID)
        assert not later.spend_row["spend"]
        assert not later.spend_row["total_tokens"]
        assert later.spend_row["request_id"] == later_call_id

    @pytest.mark.asyncio
    async def test_the_proxy_database_keeps_a_background_response_billed_once(self, local_model_cost_map, monkeypatch):
        settlement_table: Final = _SettlementTable()
        monkeypatch.setattr(
            proxy_server,
            "prisma_client",
            SimpleNamespace(
                db=SimpleNamespace(
                    litellm_backgroundresponsesettlement=settlement_table,
                    litellm_backgroundresponsecreator=_CreatorTable(),
                )
            ),
        )
        passthrough_logging: Final = PassThroughEndpointLogging()
        finished: Final = _stored_response("completed", True, RESPONSES_USAGE)

        first: Final = await self._logged_call(
            "GET",
            f"/{STORED_RESPONSE_ID}",
            {},
            finished,
            "5a9c2e71-3f4b-4d86-a1e7-c8b6d3f05e92",
            passthrough_logging=passthrough_logging,
        )
        second: Final = await self._logged_call(
            "GET",
            f"/{STORED_RESPONSE_ID}",
            {},
            finished,
            "b2d7f4a8-6e1c-4a39-8f5b-0c9e3a7d1b64",
            passthrough_logging=passthrough_logging,
        )

        assert first.spend_row["spend"] > 0
        assert not second.spend_row["spend"]
        assert settlement_table.settled_response_ids == {STORED_RESPONSE_ID}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "retrieved_id",
        [
            pytest.param(STORED_RESPONSE_ID, id="raw_id"),
            pytest.param(new_managed_id("openai", STORED_RESPONSE_ID), id="managed_id"),
        ],
    )
    async def test_a_streamed_background_create_bills_once_and_its_later_retrieve_does_not(
        self, local_model_cost_map, retrieved_id: str
    ):
        passthrough_logging: Final = _settling_passthrough_logging(_SettlementTable())
        finished: Final = _stored_response("completed", True, RESPONSES_USAGE)

        streamed_create: Final = await self._streamed_call(
            "",
            {**CREATE_REQUEST, "background": True, "stream": True},
            _completed_stream_bytes(finished),
            "8d2f4a61-7b3c-4e95-a0d8-c5e1f9b7234a",
            passthrough_logging,
        )
        retrieve: Final = await self._logged_call(
            "GET",
            f"/{STORED_RESPONSE_ID}",
            {},
            {**finished, "id": retrieved_id},
            "4b7e9c25-1a6d-4f38-9e02-d8c3a5f1b760",
            passthrough_logging=passthrough_logging,
        )

        assert streamed_create.spend_row["spend"] > 0
        assert not retrieve.spend_row["spend"]

    @pytest.mark.asyncio
    async def test_a_finished_read_cancelled_while_it_settles_still_bills_and_its_later_read_does_not(
        self, local_model_cost_map
    ):
        settled: Final = asyncio.Event()
        passthrough_logging: Final = _settling_passthrough_logging(
            _SlowAckSettlementTable(_SettlementTable(), settled), timeout_seconds=2
        )
        finished: Final = _stored_response("completed", True, RESPONSES_USAGE)
        recorder: Final = _SpendRowRecorder()
        cancelled_read: Final = asyncio.create_task(
            self._handled_call(
                "GET",
                f"{RESPONSES_URL}/{STORED_RESPONSE_ID}",
                {},
                finished,
                self._recorded_logging_obj(recorder, "6c3e8a14-2f7b-4d95-b1e0-a8d4c6f2e735", stream=False),
                passthrough_logging,
                finalize_before_dispatch=True,
            )
        )
        await settled.wait()
        cancelled_read.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_read
        first: Final = await asyncio.wait_for(recorder.call, timeout=5)
        later: Final = await self._logged_call(
            "GET",
            f"/{STORED_RESPONSE_ID}",
            {},
            finished,
            "d1a7f3c9-5e2b-4806-9c4d-3b8e6f0a2d17",
            passthrough_logging=passthrough_logging,
        )

        assert first.spend_row["spend"] > 0
        assert not later.spend_row["spend"]

    @pytest.mark.asyncio
    async def test_a_background_create_stream_cut_before_it_finishes_leaves_the_bill_to_the_finished_retrieve(
        self, local_model_cost_map
    ):
        passthrough_logging: Final = _settling_passthrough_logging(_SettlementTable())
        created: Final = {
            "type": "response.created",
            "sequence_number": 0,
            "response": _stored_response("in_progress", True, None),
        }

        await self._streamed_call(
            "",
            {**CREATE_REQUEST, "background": True, "stream": True},
            [f"event: response.created\ndata: {json.dumps(created)}\n\n".encode()],
            "d4b8e2f6-9a1c-4573-8e6d-1f7c3a5b9e20",
            passthrough_logging,
        )
        retrieve: Final = await self._logged_call(
            "GET",
            f"/{STORED_RESPONSE_ID}",
            {},
            _stored_response("completed", True, RESPONSES_USAGE),
            "f1a6c3e8-4d2b-4b97-a5e0-7c9d2f4b6a13",
            passthrough_logging=passthrough_logging,
        )

        assert retrieve.spend_row["spend"] > 0

    @pytest.mark.asyncio
    async def test_a_streamed_foreground_create_bills_without_recording_a_settlement(self, local_model_cost_map):
        settlement_table: Final = _SettlementTable()

        streamed_create: Final = await self._streamed_call(
            "",
            {**CREATE_REQUEST, "stream": True},
            _completed_stream_bytes(_stored_response("completed", False, RESPONSES_USAGE)),
            "3e8b1f57-6c2a-4d94-b7e0-9a5c4f2d816b",
            _settling_passthrough_logging(settlement_table),
        )

        assert streamed_create.spend_row["spend"] > 0
        assert not settlement_table.settled_response_ids

    @pytest.mark.asyncio
    async def test_a_resumed_stream_after_a_billed_retrieve_does_not_bill_again(self, local_model_cost_map):
        passthrough_logging: Final = _settling_passthrough_logging(_SettlementTable())
        finished: Final = _stored_response("completed", True, RESPONSES_USAGE)

        retrieve: Final = await self._logged_call(
            "GET",
            f"/{STORED_RESPONSE_ID}",
            {},
            finished,
            "6c3a8e14-5f2b-4d97-b1e6-a7d4c2f9e853",
            passthrough_logging=passthrough_logging,
        )
        resumed_stream: Final = await self._streamed_call(
            f"/{STORED_RESPONSE_ID}?stream=true",
            {},
            _completed_stream_bytes(finished),
            "a9e5d3c7-2b8f-4146-8d0a-f3b6e1c4d295",
            passthrough_logging,
        )

        assert retrieve.spend_row["spend"] > 0
        assert retrieve.spend_row["total_tokens"] > 0
        assert not resumed_stream.spend_row["spend"]
        assert not resumed_stream.spend_row["total_tokens"]
        assert resumed_stream.spend_row["request_id"] != retrieve.spend_row["request_id"]

    @pytest.mark.asyncio
    async def test_a_resumed_stream_that_fails_after_it_finished_leaves_the_charge_to_its_next_retrieve(
        self, local_model_cost_map
    ):
        passthrough_logging: Final = _settling_passthrough_logging(_SettlementTable())
        finished: Final = _stored_response("completed", True, RESPONSES_USAGE)

        await self._failed_stream(
            self._failed_stream_logging_obj("5d9a3f72-1e8b-4c46-a7d0-b3f6e2c9a581"),
            f"/{STORED_RESPONSE_ID}?stream=true",
            _completed_stream_bytes(finished),
            passthrough_logging,
        )
        await GLOBAL_LOGGING_WORKER.flush()
        first_retrieve: Final = await self._logged_call(
            "GET",
            f"/{STORED_RESPONSE_ID}",
            {},
            finished,
            "c1e7b4a9-6f2d-4835-9b0e-a8d5f3c2e176",
            passthrough_logging=passthrough_logging,
        )
        later_retrieve: Final = await self._logged_call(
            "GET",
            f"/{STORED_RESPONSE_ID}",
            {},
            finished,
            "7a3e9c15-4b8d-4f62-a0e7-d5c2b9f1e384",
            passthrough_logging=passthrough_logging,
        )

        assert first_retrieve.spend_row["spend"] > 0
        assert not later_retrieve.spend_row["spend"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "table",
        [
            pytest.param(None, id="no_database"),
            pytest.param(_UnreachableSettlementTable(), id="database_error"),
            pytest.param(_StalledSettlementTable(_SettlementTable()), id="database_stalled"),
        ],
    )
    async def test_finished_reads_bill_as_before_when_their_settlement_cannot_be_recorded(
        self, local_model_cost_map, table: _UnreachableSettlementTable | _StalledSettlementTable | None
    ):
        passthrough_logging: Final = _settling_passthrough_logging(table)
        finished: Final = _stored_response("completed", True, RESPONSES_USAGE)

        first: Final = await self._logged_call(
            "GET",
            f"/{STORED_RESPONSE_ID}",
            {},
            finished,
            "1d6b4f82-9e3a-4c75-a2d8-b5f7e0c3a916",
            passthrough_logging=passthrough_logging,
        )
        second: Final = await self._logged_call(
            "GET",
            f"/{STORED_RESPONSE_ID}",
            {},
            finished,
            "7f2c9a35-4e1d-4b68-9c07-e6a3d8b5f142",
            passthrough_logging=passthrough_logging,
        )

        assert first.spend_row["spend"] > 0
        assert second.spend_row["spend"] == pytest.approx(first.spend_row["spend"])

    @pytest.mark.asyncio
    async def test_unbilled_retrieve_spend_log_keeps_the_response_model(self, local_model_cost_map):
        response_body: Final = _stored_response("completed", False, RESPONSES_USAGE)

        spend_log_row: Final = (
            await self._logged_call(
                "GET", f"/{STORED_RESPONSE_ID}", {}, response_body, "0f6f3d2e-6a51-4c4e-9a43-6f1b0d1e9c21"
            )
        ).spend_row

        assert spend_log_row["model"] == response_body["model"]
        assert not spend_log_row["spend"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("request_agent", "response_agent_model", "logged_model"),
        [
            pytest.param({"agent_id": "agent_6a0b0f8e"}, "gpt-5.4-mini", "gpt-5.4-mini", id="saved_agent"),
            pytest.param({"agent": {"model": "gpt-5.4-mini"}}, None, "gpt-5.4-mini", id="request_agent_only"),
            pytest.param({"agent": {"model": "gpt-5.4"}}, "gpt-5.4-mini", "gpt-5.4-mini", id="response_agent_wins"),
        ],
    )
    async def test_unbilled_agents_session_spend_log_names_the_agent_model(
        self, request_agent: dict[str, object], response_agent_model: str | None, logged_model: str
    ):
        spend_log_row: Final = (
            await self._logged_call(
                "POST",
                "",
                {"environment": {"type": "none"}, **request_agent},
                _agent_session(response_agent_model),
                "2f7c9e1a-4b3d-4a6e-8c5f-1d0e9b7a3c62",
                AGENTS_SESSIONS_URL,
            )
        ).spend_row

        assert spend_log_row["model"] == logged_model
        assert not spend_log_row["spend"]

    @pytest.mark.asyncio
    async def test_unbilled_saved_agent_spend_log_names_the_agent_model(self):
        saved_agent: Final = {"id": "agent_6a0b0f8e", "object": "agent", "model": "gpt-5.4-mini", "tools": []}

        spend_log_row: Final = (
            await self._logged_call(
                "POST",
                "",
                {"model": "gpt-5.4-mini", "instructions": "Answer in one word"},
                saved_agent,
                "8d2e4f6a-1b3c-4d5e-9f7a-2c4e6a8b0d13",
                "https://api.openai.com/v1/agents",
            )
        ).spend_row

        assert spend_log_row["model"] == saved_agent["model"]
        assert not spend_log_row["spend"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("request_body", "session_agent_model", "logged_model"),
        [
            pytest.param({}, "gpt-5.4-mini", "gpt-5.4-mini", id="session_event_names_the_model"),
            pytest.param({"agent": {"model": "gpt-5.4"}}, None, "gpt-5.4", id="request_agent_fallback"),
        ],
    )
    async def test_streamed_agents_session_spend_log_names_the_agent_model(
        self, request_body: dict[str, object], session_agent_model: str | None, logged_model: str
    ):
        session_event: Final = {"type": "agent.session.in_progress", "session": _agent_session(session_agent_model)}
        sse_events: Final = (
            f"event: agent.session.in_progress\ndata: {json.dumps(session_event)}\n\n".encode(),
            b'event: agent.turn.completed\ndata: {"type": "agent.turn.completed", "usage": null}\n\n',
        )

        spend_log_row: Final = (
            await self._logged_stream(
                f"{AGENTS_SESSIONS_URL}/sess_6a0b0f8e2c1d4e5f/events",
                request_body,
                sse_events,
                "5b7d9f1a-3c5e-4a7b-8d9f-0e2a4c6b8d35",
            )
        ).spend_row

        assert spend_log_row["model"] == logged_model
        assert not spend_log_row["spend"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "sub_route", "response_body"),
        [
            pytest.param(
                "GET",
                f"/{STORED_RESPONSE_ID}",
                _stored_response("completed", True, RESPONSES_USAGE),
                id="background_retrieve_completed",
            ),
            pytest.param(
                "POST",
                f"/{STORED_RESPONSE_ID}/cancel",
                _stored_response("cancelled", True, RESPONSES_USAGE),
                id="background_cancel",
            ),
        ],
    )
    async def test_billed_background_read_spend_log_survives_next_to_its_create_and_poll_rows(
        self, local_model_cost_map, method: str, sub_route: str, response_body: dict[str, object]
    ):
        create: Final = await self._logged_call(
            "POST",
            "",
            {**CREATE_REQUEST, "background": True},
            _stored_response("queued", True, None),
            "3a4f0c1e-2b7d-4c55-8e0e-0d7a51d9f6a1",
        )
        poll: Final = await self._logged_call(
            "GET",
            f"/{STORED_RESPONSE_ID}",
            {},
            _stored_response("in_progress", True, None),
            "5d8c2f71-0e4b-4a9d-b3c6-7f1e2a9d0c44",
        )
        read: Final = await self._logged_call(
            method, sub_route, {}, response_body, "9b2e6d40-71c3-4f8a-a5d2-4e3c8f0b7d12"
        )

        assert read.spend_row["request_id"] not in {create.spend_row["request_id"], poll.spend_row["request_id"]}
        assert read.spend_row["request_id"].startswith(STORED_RESPONSE_ID)
        assert read.spend_row["spend"] > 0
        assert not poll.spend_row["spend"]
        assert read.callback_response_id == STORED_RESPONSE_ID

    @pytest.mark.asyncio
    async def test_streamed_background_create_and_its_finished_read_share_one_spend_row(self, local_model_cost_map):
        finished: Final = _stored_response("completed", True, RESPONSES_USAGE)
        stream_logging_obj: Final = LiteLLMLoggingObj(
            model="gpt-5.4-mini",
            messages=[],
            stream=True,
            call_type="pass_through_endpoint",
            start_time=DISPATCH_TIME,
            litellm_call_id="7e3b9a14-5c2d-4f60-8d1a-2b6c0e9f4a37",
            function_id="1234",
        )
        stream_logging_obj.model_call_details["custom_llm_provider"] = "openai"
        streamed_create: Final = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
            litellm_logging_obj=stream_logging_obj,
            passthrough_success_handler_obj=None,
            url_route=RESPONSES_URL,
            request_body={**CREATE_REQUEST, "background": True, "stream": True},
            endpoint_type=None,
            start_time=DISPATCH_TIME,
            all_chunks=[
                f"data: {json.dumps({'type': 'response.completed', 'sequence_number': 1, 'response': finished})}",
                "data: [DONE]",
            ],
            end_time=DISPATCH_TIME,
        )
        create_row: Final = get_logging_payload(
            kwargs=streamed_create["kwargs"],
            response_obj=streamed_create["result"],
            start_time=DISPATCH_TIME,
            end_time=DISPATCH_TIME,
        )

        read: Final = await self._logged_call(
            "GET", f"/{STORED_RESPONSE_ID}", {}, finished, "1c8d5e2a-9f47-4b3e-a6d0-5e7f2c1b8a93"
        )

        assert create_row["spend"] > 0
        assert read.spend_row["request_id"] == create_row["request_id"]
        assert read.callback_response_id == STORED_RESPONSE_ID

    @pytest.mark.parametrize(
        "base_url",
        [
            pytest.param("https://my-resource.openai.azure.com/openai/v1/responses", id="azure_openai_v1"),
            pytest.param("https://my-resource.cognitiveservices.azure.com/openai/responses", id="azure_openai_native"),
        ],
    )
    def test_azure_responses_sub_routes_bill_like_openai(self, local_model_cost_map, base_url: str):
        retrieve: Final = self._dispatch(
            "GET", f"/{STORED_RESPONSE_ID}", {}, _stored_response("completed", False, RESPONSES_USAGE), base_url
        )
        input_tokens: Final = self._dispatch(
            "POST",
            "/input_tokens",
            {"model": "gpt-5.4-mini", "input": "hi"},
            {"object": "response.input_tokens", "input_tokens": 7},
            base_url,
        )
        background_retrieve: Final = self._dispatch(
            "GET", f"/{STORED_RESPONSE_ID}", {}, _stored_response("completed", True, RESPONSES_USAGE), base_url
        )
        compact: Final = self._dispatch("POST", "/compact", COMPACT_REQUEST, COMPACTION, base_url)

        assert not retrieve["kwargs"].get("response_cost")
        assert not input_tokens["kwargs"].get("response_cost")
        assert background_retrieve["kwargs"]["response_cost"] > 0
        assert compact["kwargs"]["response_cost"] > 0


CREATE_CALL_ID: Final = "0d4b7e19-6a2c-4f83-9e51-2c8a7f3b6d40"
READ_CALL_ID: Final = "c71e5a93-2f0d-4b68-a4c9-8e3d1b7f2a05"
BACKGROUND_COST_REQUEST_ID: Final = f"{STORED_RESPONSE_ID}_background_cost"
CREATOR_METADATA: Final = {
    "user_api_key": "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",
    "user_api_key_team_id": "team-creator",
    "user_api_key_user_id": "user-creator",
    "user_api_key_budget_reservation": {"reserved_cost": 0.5, "counter_keys": ["spend:key:creator"]},
}
READER_METADATA: Final = {
    "user_api_key": "4a5ec0ad6c1e9b3fd0a8b2c7e91f3d6a5b4c2e1f0a9b8c7d6e5f4a3b2c1d0e9f",
    "user_api_key_team_id": "team-reader",
    "user_api_key_user_id": "user-reader",
}
UPSTREAM_AUTHORIZATION: Final = "Bearer upstream-openai-key"
CREATOR_RECORDED_AT: Final = datetime(2026, 10, 17, 13, 47, 4, tzinfo=timezone.utc)
FAST_POLLS: Final = PollSchedule(initial_interval_seconds=0.001, max_interval_seconds=0.001, timeout_seconds=5.0)
HALF_SECOND_POLLS: Final = PollSchedule(initial_interval_seconds=0.5, max_interval_seconds=0.5, timeout_seconds=5.0)
LEASE_LAPSED_AT: Final = CREATOR_RECORDED_AT + timedelta(seconds=BACKGROUND_RESPONSE_POLL_LEASE_SECONDS + 2)
RESUMED_POLLS: Final = PollSchedule(
    initial_interval_seconds=0.001,
    max_interval_seconds=0.001,
    timeout_seconds=BACKGROUND_RESPONSE_POLL_SCHEDULE.timeout_seconds,
)


@dataclass(frozen=True, slots=True)
class _CreatorRow:
    response_id: str
    upstream_url: str
    create_context: object
    created_at: datetime
    poll_lease_expires_at: object


def _lease_open(row: _CreatorRow, where: Mapping[str, object]) -> bool:
    expiries: Final = where.get("OR")
    if not isinstance(expiries, Sequence):
        return True
    lease: Final = row.poll_lease_expires_at
    return any(
        lease is None if expiry["poll_lease_expires_at"] is None else lease < expiry["poll_lease_expires_at"]["lt"]
        for expiry in expiries
    )


def _creator_context(metadata: Mapping[str, object]) -> BackgroundResponseCreateContext:
    return BackgroundResponseCreateContext.model_validate(
        {
            "model": "gpt-5.4-mini",
            "custom_llm_provider": "openai",
            "litellm_call_id": CREATE_CALL_ID,
            "litellm_trace_id": CREATE_CALL_ID,
            "start_time": CREATOR_RECORDED_AT,
            "metadata": metadata,
            "key_or_team_logging": True,
        }
    )


class _CreatorTable:
    def __init__(self, lookup_failures: int = 0) -> None:
        self.rows: Final[dict[str, _CreatorRow]] = {}
        self.forgotten: Final = asyncio.Event()
        self.lookup_failures: Final = lookup_failures
        self.failed_lookups: Final[list[str]] = []

    async def create(self, *, data: Mapping[str, object]) -> _CreatorRow:
        stored_context: Final = data["create_context"]
        row: Final = _CreatorRow(
            response_id=str(data["response_id"]),
            upstream_url=str(data["upstream_url"]),
            create_context=json.loads(json.dumps(getattr(stored_context, "data", stored_context))),
            created_at=CREATOR_RECORDED_AT,
            poll_lease_expires_at=data.get("poll_lease_expires_at"),
        )
        self.rows[row.response_id] = row
        return row

    async def find_unique(self, *, where: Mapping[str, str]) -> _CreatorRow | None:
        if len(self.failed_lookups) < self.lookup_failures:
            self.failed_lookups.append(where["response_id"])
            raise ConnectionResetError("server closed the connection unexpectedly")
        return self.rows.get(where["response_id"])

    async def find_many(self, *, where: Mapping[str, Mapping[str, datetime]]) -> list[_CreatorRow]:
        recent: Final = (row for row in self.rows.values() if row.created_at > where["created_at"]["gt"])
        return [row for row in recent if _lease_open(row, where)]

    async def update_many(self, *, data: Mapping[str, datetime], where: Mapping[str, object]) -> int:
        row: Final = self.rows.get(str(where["response_id"]))
        if row is None or not _lease_open(row, where):
            return 0
        self.rows[row.response_id] = replace(row, poll_lease_expires_at=data["poll_lease_expires_at"])
        return 1

    async def delete_many(self, *, where: Mapping[str, object]) -> int:
        if "created_at" in where:
            return 0
        removed: Final = self.rows.pop(str(where["response_id"]), None)
        self.forgotten.set()
        return 0 if removed is None else 1


class _UnmigratedCreatorTable(_CreatorTable):
    async def find_unique(self, *, where: Mapping[str, str]) -> _CreatorRow | None:
        raise TableNotFoundError(data={"user_facing_error": {"meta": {"table": "LiteLLM_BackgroundResponseCreator"}}})


class _StalledKeyCache:
    async def async_get_cache(self, **kwargs: object) -> None:
        await asyncio.Event().wait()


class _DeadlineKeyCache:
    async def async_get_cache(self, **kwargs: object) -> None:
        raise DBLookupDeadlineExceeded("key", 0.1)


class _ScriptedFetch:
    def __init__(self) -> None:
        self.outcomes: Final[asyncio.Queue[FetchOutcome]] = asyncio.Queue()
        self.requests: Final[list[BackgroundResponsePollRequest]] = []

    async def __call__(self, request: BackgroundResponsePollRequest) -> FetchOutcome:
        self.requests.append(request)
        return await self.outcomes.get()


@dataclass(frozen=True, slots=True)
class _DispatchedRow:
    metadata: Mapping[str, object]
    spend_row: SpendLogsPayload
    start_time: datetime
    end_time: datetime
    logging_obj: LiteLLMLoggingObj


class _DispatchedSpendRows:
    def __init__(self) -> None:
        self.rows: Final[list[_DispatchedRow]] = []

    async def __call__(
        self,
        *,
        logging_obj: LiteLLMLoggingObj,
        standard_logging_response_object: object,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        **kwargs: object,
    ) -> None:
        await PassThroughEndpointLogging().handle_logging(
            logging_obj=logging_obj,
            standard_logging_response_object=standard_logging_response_object,
            result=result,
            start_time=start_time,
            end_time=end_time,
            cache_hit=cache_hit,
            **kwargs,
        )
        await GLOBAL_LOGGING_WORKER.flush()
        self.rows.append(
            _DispatchedRow(
                metadata=dict(logging_obj.model_call_details["litellm_params"]["metadata"]),
                spend_row=get_logging_payload(
                    kwargs=logging_obj.model_call_details,
                    response_obj=standard_logging_response_object,
                    start_time=start_time,
                    end_time=end_time,
                ),
                start_time=start_time,
                end_time=end_time,
                logging_obj=logging_obj,
            )
        )

    def spend_rows(self) -> tuple[SpendLogsPayload, ...]:
        return tuple(row.spend_row for row in self.rows)

    def logged_for(self, team_id: str) -> tuple[SpendLogsPayload, ...]:
        return tuple(row for row in self.spend_rows() if row["team_id"] == team_id)

    def billed(self) -> tuple[_DispatchedRow, ...]:
        return tuple(row for row in self.rows if row.spend_row["spend"])


@dataclass(frozen=True, slots=True)
class _FirstClaimWaitsUntilASecondClaimLoses:
    table: _SettlementTable
    first_claim_inserted: asyncio.Event
    second_claim_lost: asyncio.Event

    async def create_many(self, data: Sequence[Mapping[str, str]], *, skip_duplicates: bool) -> int:
        inserted: Final = await self.table.create_many(data, skip_duplicates=skip_duplicates)
        if not inserted:
            self.second_claim_lost.set()
            return inserted
        self.first_claim_inserted.set()
        await self.second_claim_lost.wait()
        return inserted

    async def delete_many(self, *, where: Mapping[str, str]) -> int:
        return await self.table.delete_many(where=where)


@dataclass(frozen=True, slots=True)
class _ClaimCommitsWhenAllowed:
    table: _SettlementTable
    insert_started: asyncio.Event = field(default_factory=asyncio.Event)
    commit: asyncio.Event = field(default_factory=asyncio.Event)
    released: asyncio.Event = field(default_factory=asyncio.Event)

    async def create_many(self, data: Sequence[Mapping[str, str]], *, skip_duplicates: bool) -> int:
        inserted: Final = await self.table.create_many(data, skip_duplicates=skip_duplicates)
        self.insert_started.set()
        await self.commit.wait()
        return inserted

    async def delete_many(self, *, where: Mapping[str, str]) -> int:
        deleted: Final = await self.table.delete_many(where=where)
        self.released.set()
        return deleted

    async def commit_and_wait_for_release(self) -> None:
        self.commit.set()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.released.wait(), timeout=1)


class _ClaimUnrecordedOnCall:
    def __init__(self, table: _SettlementTable, failing_call: int) -> None:
        self.table: Final = table
        self.failing_call: Final = failing_call
        self.calls: Final[list[Sequence[Mapping[str, str]]]] = []

    async def create_many(self, data: Sequence[Mapping[str, str]], *, skip_duplicates: bool) -> int:
        self.calls.append(data)
        if len(self.calls) == self.failing_call:
            raise ConnectionResetError("server closed the connection unexpectedly")
        return await self.table.create_many(data, skip_duplicates=skip_duplicates)

    async def delete_many(self, *, where: Mapping[str, str]) -> int:
        return await self.table.delete_many(where=where)


class _ClaimAcksLateOnCall:
    def __init__(self, table: _SettlementTable, late_call: int, ack_delay_seconds: float) -> None:
        self.table: Final = table
        self.late_call: Final = late_call
        self.ack_delay_seconds: Final = ack_delay_seconds
        self.calls: Final[list[Sequence[Mapping[str, str]]]] = []

    async def create_many(self, data: Sequence[Mapping[str, str]], *, skip_duplicates: bool) -> int:
        self.calls.append(data)
        call: Final = len(self.calls)
        inserted: Final = await self.table.create_many(data, skip_duplicates=skip_duplicates)
        if call == self.late_call:
            await asyncio.sleep(self.ack_delay_seconds)
        return inserted

    async def delete_many(self, *, where: Mapping[str, str]) -> int:
        return await self.table.delete_many(where=where)


class _CreatorDispatchFailsAfterTheCreate:
    def __init__(self, rows: _DispatchedSpendRows, refusals: int) -> None:
        self.rows: Final = rows
        self.refusals: Final = refusals
        self.refused: Final[list[str]] = []

    async def __call__(
        self,
        *,
        logging_obj: LiteLLMLoggingObj,
        standard_logging_response_object: object,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        **kwargs: object,
    ) -> None:
        creator_team: Final = CREATOR_METADATA["user_api_key_team_id"]
        team_id: Final = logging_obj.model_call_details["litellm_params"]["metadata"].get("user_api_key_team_id")
        if team_id == creator_team and self.rows.logged_for(creator_team) and len(self.refused) < self.refusals:
            self.refused.append(logging_obj.litellm_call_id)
            raise ConnectionResetError("spend log queue closed")
        await self.rows(
            logging_obj=logging_obj,
            standard_logging_response_object=standard_logging_response_object,
            result=result,
            start_time=start_time,
            end_time=end_time,
            cache_hit=cache_hit,
            **kwargs,
        )


def _creator_billing_passthrough_logging(
    rows: _DispatchedSpendRows | _CreatorDispatchFailsAfterTheCreate,
    settlements: (
        _SettlementTable
        | _FirstClaimWaitsUntilASecondClaimLoses
        | _ClaimUnrecordedOnCall
        | _ClaimAcksLateOnCall
        | _ClaimCommitsWhenAllowed
    ),
    creators: _CreatorTable,
    fetch: _ScriptedFetch,
    polls: bool,
    now: datetime = CREATOR_RECORDED_AT + timedelta(seconds=1),
    schedule: PollSchedule = FAST_POLLS,
    creator_logging_wiring: CreatorLoggingWiring = proxy_creator_logging_wiring,
    settlement_timeout_seconds: float = 5,
) -> PassThroughEndpointLogging:
    return PassThroughEndpointLogging(
        creator_logging_wiring=creator_logging_wiring,
        log_dispatch=rows,
        background_response_settlement=PrismaBackgroundResponseSettlementStore(
            table=lambda: settlements, timeout_seconds=settlement_timeout_seconds
        ),
        background_response_creators=PrismaBackgroundResponseCreatorStore(table=lambda: creators, timeout_seconds=5),
        background_response_poller=BackgroundResponsePoller(
            fetch=fetch, schedule=schedule, database_configured=lambda: polls, now=lambda: now
        ),
    )


def _running_poll(passthrough_logging: PassThroughEndpointLogging) -> "asyncio.Task[None]":
    (poll,) = passthrough_logging.background_response_poller.active.values()
    return poll


def _identified_logging_obj(
    litellm_call_id: str,
    metadata: Mapping[str, object],
    stream: bool = False,
    callbacks: tuple[CustomLogger, ...] = (),
    litellm_params: Mapping[str, object] = MappingProxyType({}),
) -> LiteLLMLoggingObj:
    logging_obj: Final = LiteLLMLoggingObj(
        model="unknown",
        messages=[],
        stream=stream,
        call_type="pass_through_endpoint",
        start_time=DISPATCH_TIME,
        litellm_call_id=litellm_call_id,
        function_id="1245",
        dynamic_async_success_callbacks=list(callbacks),
    )
    logging_obj.update_environment_variables(
        model="unknown",
        user="unknown",
        optional_params={},
        litellm_params={**litellm_params, "metadata": dict(metadata)},
        call_type="pass_through_endpoint",
    )
    return logging_obj


async def _passed_through(
    passthrough_logging: PassThroughEndpointLogging,
    logging_obj: LiteLLMLoggingObj,
    upstream_request: httpx.Request,
    url_route: str,
    request_body: dict[str, object],
    response_body: dict[str, object],
    custom_llm_provider: str | None = "openai",
    finalize_before_dispatch: bool = True,
) -> None:
    await passthrough_logging.pass_through_async_success_handler(
        httpx_response=httpx.Response(200, json=response_body, request=upstream_request),
        response_body=response_body,
        logging_obj=logging_obj,
        url_route=url_route,
        result="",
        start_time=DISPATCH_TIME,
        end_time=DISPATCH_TIME,
        cache_hit=False,
        request_body=request_body,
        passthrough_logging_payload=PassthroughStandardLoggingPayload(
            url=url_route, request_body=request_body, request_method=upstream_request.method
        ),
        custom_llm_provider=custom_llm_provider,
        finalize_before_dispatch=finalize_before_dispatch,
    )


async def _background_create(
    passthrough_logging: PassThroughEndpointLogging,
    responses_url: str = RESPONSES_URL,
    response_id: str = STORED_RESPONSE_ID,
    logging_obj: LiteLLMLoggingObj | None = None,
) -> None:
    request_body: Final = {**CREATE_REQUEST, "background": True}
    await _passed_through(
        passthrough_logging,
        logging_obj or _identified_logging_obj(CREATE_CALL_ID, CREATOR_METADATA),
        httpx.Request("POST", responses_url, headers={"authorization": UPSTREAM_AUTHORIZATION}, json=request_body),
        responses_url,
        request_body,
        {**_stored_response("queued", True, None), "id": response_id},
    )


async def _finished_read(
    passthrough_logging: PassThroughEndpointLogging,
    custom_llm_provider: str | None = "openai",
    finalize_before_dispatch: bool = True,
) -> None:
    url_route: Final = f"{RESPONSES_URL}/{STORED_RESPONSE_ID}"
    await _passed_through(
        passthrough_logging,
        _identified_logging_obj(READ_CALL_ID, READER_METADATA),
        httpx.Request("GET", url_route, headers={"authorization": UPSTREAM_AUTHORIZATION}),
        url_route,
        {},
        _stored_response("completed", True, RESPONSES_USAGE),
        custom_llm_provider=custom_llm_provider,
        finalize_before_dispatch=finalize_before_dispatch,
    )


async def _reader_paid_spend() -> float:
    rows: Final = _DispatchedSpendRows()
    await _finished_read(
        _creator_billing_passthrough_logging(rows, _SettlementTable(), _CreatorTable(), _ScriptedFetch(), polls=False)
    )
    return rows.rows[0].spend_row["spend"]


async def _first_fetch(fetch: _ScriptedFetch) -> None:
    while not fetch.requests:
        await asyncio.sleep(0.01)


async def _poll_settled(creators: _CreatorTable) -> None:
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(creators.forgotten.wait(), timeout=5)


async def _resumed_poll_headers(api_base: str) -> Mapping[str, str]:
    return {"authorization": "Bearer resumed-openai-key"}


class _PROXY_SpendLogger(CustomLogger):
    pass


def _assert_billed_to_the_creator(row: _DispatchedRow, create_row: SpendLogsPayload, spend: float) -> None:
    assert row.spend_row["team_id"] == CREATOR_METADATA["user_api_key_team_id"]
    assert row.spend_row["user"] == CREATOR_METADATA["user_api_key_user_id"]
    assert row.spend_row["api_key"] == create_row["api_key"]
    assert row.spend_row["request_id"] == BACKGROUND_COST_REQUEST_ID
    assert row.spend_row["spend"] == pytest.approx(spend)
    assert spend > 0


class TestOpenAIPassthroughBackgroundResponseCreatorBilling:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "read_route_provider",
        [pytest.param("openai", id="openai_passthrough_route"), pytest.param(None, id="configured_route")],
    )
    async def test_a_finished_read_by_another_key_bills_the_key_that_created_the_response(
        self, read_route_provider: str | None, local_model_cost_map
    ):
        rows: Final = _DispatchedSpendRows()
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows, _SettlementTable(), _CreatorTable(), _ScriptedFetch(), polls=False
        )

        await _background_create(passthrough_logging)
        await _finished_read(passthrough_logging, custom_llm_provider=read_route_provider)

        create_row: Final = rows.spend_rows()[0]
        (read_row,) = rows.logged_for(READER_METADATA["user_api_key_team_id"])
        assert len(rows.billed()) == 1
        _assert_billed_to_the_creator(rows.billed()[0], create_row, await _reader_paid_spend())
        assert read_row["request_id"] == READ_CALL_ID
        assert read_row["team_id"] == READER_METADATA["user_api_key_team_id"]
        assert not read_row["spend"]

    @pytest.mark.asyncio
    async def test_a_background_response_nobody_reads_is_billed_once_to_its_creator(self, local_model_cost_map):
        rows: Final = _DispatchedSpendRows()
        creators: Final = _CreatorTable()
        fetch: Final = _ScriptedFetch()
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows, _SettlementTable(), creators, fetch, polls=True
        )

        await _background_create(passthrough_logging)
        fetch.outcomes.put_nowait(Fetched(body=_stored_response("in_progress", True, None)))
        fetch.outcomes.put_nowait(Fetched(body=_stored_response("completed", True, RESPONSES_USAGE)))

        await _poll_settled(creators)
        assert len(rows.billed()) == 1
        _assert_billed_to_the_creator(rows.billed()[0], rows.spend_rows()[0], await _reader_paid_spend())
        assert [str(request.url) for request in fetch.requests] == [f"{RESPONSES_URL}/{STORED_RESPONSE_ID}"] * 2
        assert ("authorization", UPSTREAM_AUTHORIZATION) in fetch.requests[0].headers
        assert not {"content-length", "content-type"} & {name.lower() for name, _ in fetch.requests[0].headers}

    @pytest.mark.asyncio
    async def test_a_read_after_the_poll_billed_the_creator_is_not_billed_again(self, local_model_cost_map):
        rows: Final = _DispatchedSpendRows()
        creators: Final = _CreatorTable()
        fetch: Final = _ScriptedFetch()
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows, _SettlementTable(), creators, fetch, polls=True
        )
        await _background_create(passthrough_logging)
        fetch.outcomes.put_nowait(Fetched(body=_stored_response("completed", True, RESPONSES_USAGE)))
        await _poll_settled(creators)

        await _finished_read(passthrough_logging)

        read_row: Final = rows.spend_rows()[-1]
        assert [row.spend_row["team_id"] for row in rows.billed()] == [CREATOR_METADATA["user_api_key_team_id"]]
        assert read_row["request_id"] == READ_CALL_ID
        assert not read_row["spend"]

    @pytest.mark.asyncio
    async def test_a_read_that_claims_while_the_poll_loses_the_claim_still_bills_the_creator_once(
        self, local_model_cost_map
    ):
        rows: Final = _DispatchedSpendRows()
        creators: Final = _CreatorTable()
        fetch: Final = _ScriptedFetch()
        first_claim_inserted: Final = asyncio.Event()
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows,
            _FirstClaimWaitsUntilASecondClaimLoses(_SettlementTable(), first_claim_inserted, asyncio.Event()),
            creators,
            fetch,
            polls=True,
        )
        await _background_create(passthrough_logging)
        poll: Final = _running_poll(passthrough_logging)

        read: Final = asyncio.create_task(_finished_read(passthrough_logging))
        await asyncio.wait_for(first_claim_inserted.wait(), timeout=5)
        fetch.outcomes.put_nowait(Fetched(body=_stored_response("completed", True, RESPONSES_USAGE)))
        await asyncio.wait_for(asyncio.gather(read, poll), timeout=5)

        create_row: Final = rows.spend_rows()[0]
        (read_row,) = rows.logged_for(READER_METADATA["user_api_key_team_id"])
        assert len(rows.billed()) == 1
        _assert_billed_to_the_creator(rows.billed()[0], create_row, await _reader_paid_spend())
        assert not read_row["spend"]
        assert creators.rows == {}

    @pytest.mark.asyncio
    async def test_a_restarted_proxy_polls_only_creators_on_its_configured_openai_base(self, local_model_cost_map):
        rows: Final = _DispatchedSpendRows()
        settlements: Final = _SettlementTable()
        creators: Final = _CreatorTable()
        other_base_response_id: Final = "resp_0c5e9a71f2b84d36a1e07b9c4d2f8a63"
        before_restart: Final = _creator_billing_passthrough_logging(
            rows, settlements, creators, _ScriptedFetch(), polls=False
        )
        await _background_create(before_restart)
        await _background_create(
            before_restart, "https://my-resource.openai.azure.com/openai/v1/responses", other_base_response_id
        )
        create_row: Final = rows.spend_rows()[0]
        fetch: Final = _ScriptedFetch()
        after_restart: Final = _creator_billing_passthrough_logging(
            rows, settlements, creators, fetch, polls=True, now=LEASE_LAPSED_AT, schedule=RESUMED_POLLS
        )

        polls: Final = await after_restart.sweep_background_response_polls(
            api_base="https://api.openai.com/", poll_headers=_resumed_poll_headers
        )
        fetch.outcomes.put_nowait(Fetched(body=_stored_response("completed", True, RESPONSES_USAGE)))
        await asyncio.wait_for(asyncio.gather(*polls), timeout=5)

        assert len(rows.billed()) == 1
        _assert_billed_to_the_creator(rows.billed()[0], create_row, await _reader_paid_spend())
        assert [str(request.url) for request in fetch.requests] == [f"{RESPONSES_URL}/{STORED_RESPONSE_ID}"]
        assert fetch.requests[0].headers == (("authorization", "Bearer resumed-openai-key"),)
        assert set(creators.rows) == {other_base_response_id}

    @pytest.mark.asyncio
    async def test_the_creator_bill_does_not_settle_the_create_budget_reservation_again(self, local_model_cost_map):
        rows: Final = _DispatchedSpendRows()
        creators: Final = _CreatorTable()
        fetch: Final = _ScriptedFetch()
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows, _SettlementTable(), creators, fetch, polls=True
        )

        await _background_create(passthrough_logging)
        fetch.outcomes.put_nowait(Fetched(body=_stored_response("completed", True, RESPONSES_USAGE)))

        await _poll_settled(creators)
        create, bill = rows.rows
        assert create.metadata["user_api_key_budget_reservation"] == CREATOR_METADATA["user_api_key_budget_reservation"]
        assert bill.spend_row["team_id"] == CREATOR_METADATA["user_api_key_team_id"]
        assert "user_api_key_budget_reservation" not in bill.metadata

    @pytest.mark.asyncio
    async def test_a_poll_that_finds_the_response_ended_without_usage_bills_nobody(self, local_model_cost_map):
        rows: Final = _DispatchedSpendRows()
        creators: Final = _CreatorTable()
        fetch: Final = _ScriptedFetch()
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows, _SettlementTable(), creators, fetch, polls=True
        )
        await _background_create(passthrough_logging)

        fetch.outcomes.put_nowait(Fetched(body=_stored_response("cancelled", True, None)))

        await _poll_settled(creators)
        assert len(fetch.requests) == 1
        assert creators.rows == {}
        assert rows.billed() == ()

    @pytest.mark.asyncio
    async def test_only_a_create_that_leaves_a_background_response_running_records_its_creator(
        self, local_model_cost_map
    ):
        creators: Final = _CreatorTable()
        fetch: Final = _ScriptedFetch()
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            _DispatchedSpendRows(), _SettlementTable(), creators, fetch, polls=True
        )
        foreground: Final = {**CREATE_REQUEST, "background": False}
        background: Final = {**CREATE_REQUEST, "background": True}
        creates: Final = (
            (foreground, {**_stored_response("completed", False, RESPONSES_USAGE), "id": "resp_foreground"}),
            (background, {**_stored_response("completed", True, RESPONSES_USAGE), "id": "resp_finished_at_create"}),
            (background, {**_stored_response("queued", True, None), "id": "resp_a/../b"}),
            (background, {**_stored_response("queued", True, None), "id": STORED_RESPONSE_ID}),
        )

        for request_body, response_body in creates:
            await _passed_through(
                passthrough_logging,
                _identified_logging_obj(CREATE_CALL_ID, CREATOR_METADATA),
                httpx.Request("POST", RESPONSES_URL, json=request_body),
                RESPONSES_URL,
                request_body,
                response_body,
            )

        assert set(creators.rows) == {STORED_RESPONSE_ID}
        fetch.outcomes.put_nowait(Fetched(body=_stored_response("cancelled", True, None)))
        await _poll_settled(creators)
        assert [str(request.url) for request in fetch.requests] == [f"{RESPONSES_URL}/{STORED_RESPONSE_ID}"]

    @pytest.mark.asyncio
    async def test_a_creator_row_this_version_cannot_read_leaves_the_reader_billed(self, local_model_cost_map):
        rows: Final = _DispatchedSpendRows()
        creators: Final = _CreatorTable()
        await creators.create(
            data={"response_id": STORED_RESPONSE_ID, "upstream_url": RESPONSES_URL, "create_context": {"v": 2}}
        )

        await _finished_read(
            _creator_billing_passthrough_logging(rows, _SettlementTable(), creators, _ScriptedFetch(), polls=False)
        )

        (read_row,) = rows.spend_rows()
        assert read_row["team_id"] == READER_METADATA["user_api_key_team_id"]
        assert read_row["request_id"] == BACKGROUND_COST_REQUEST_ID
        assert read_row["spend"] > 0

    @pytest.mark.asyncio
    async def test_a_finished_read_with_no_recorded_creator_bills_the_reader(self, local_model_cost_map):
        rows: Final = _DispatchedSpendRows()

        await _finished_read(
            _creator_billing_passthrough_logging(
                rows, _SettlementTable(), _CreatorTable(), _ScriptedFetch(), polls=False
            )
        )

        (read_row,) = rows.spend_rows()
        assert read_row["request_id"] == BACKGROUND_COST_REQUEST_ID
        assert read_row["team_id"] == READER_METADATA["user_api_key_team_id"]
        assert read_row["user"] == READER_METADATA["user_api_key_user_id"]
        assert read_row["spend"] > 0

    @pytest.mark.asyncio
    async def test_a_finished_read_before_the_creator_table_is_migrated_bills_the_reader(self, local_model_cost_map):
        rows: Final = _DispatchedSpendRows()

        await _finished_read(
            _creator_billing_passthrough_logging(
                rows, _SettlementTable(), _UnmigratedCreatorTable(), _ScriptedFetch(), polls=False
            )
        )

        (read_row,) = rows.spend_rows()
        assert read_row["team_id"] == READER_METADATA["user_api_key_team_id"]
        assert read_row["spend"] > 0

    @pytest.mark.asyncio
    async def test_an_azure_background_create_records_no_creator_and_its_finished_read_bills_the_reader(
        self, local_model_cost_map
    ):
        rows: Final = _DispatchedSpendRows()
        creators: Final = _CreatorTable()
        fetch: Final = _ScriptedFetch()
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows, _SettlementTable(), creators, fetch, polls=True
        )
        azure_responses_url: Final = "https://my-resource.openai.azure.com/openai/v1/responses"
        read_url: Final = f"{azure_responses_url}/{STORED_RESPONSE_ID}"
        create_body: Final = {**CREATE_REQUEST, "background": True}

        await _passed_through(
            passthrough_logging,
            _identified_logging_obj(CREATE_CALL_ID, CREATOR_METADATA),
            httpx.Request("POST", azure_responses_url, json=create_body),
            azure_responses_url,
            create_body,
            {**_stored_response("queued", True, None), "id": STORED_RESPONSE_ID},
            custom_llm_provider="azure",
        )
        await _passed_through(
            passthrough_logging,
            _identified_logging_obj(READ_CALL_ID, READER_METADATA),
            httpx.Request("GET", read_url),
            read_url,
            {},
            _stored_response("completed", True, RESPONSES_USAGE),
            custom_llm_provider="azure",
        )

        assert creators.rows == {}
        assert fetch.requests == []
        (read_row,) = rows.logged_for(READER_METADATA["user_api_key_team_id"])
        assert read_row["spend"] > 0
        assert not rows.logged_for(CREATOR_METADATA["user_api_key_team_id"])[1:]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "read_route_provider",
        [pytest.param("azure", id="azure_passthrough_route"), pytest.param(None, id="configured_route")],
    )
    async def test_an_azure_finished_read_bills_the_reader_without_reading_the_creator_table(
        self, read_route_provider: str | None, local_model_cost_map
    ):
        rows: Final = _DispatchedSpendRows()
        creators: Final = _CreatorTable(lookup_failures=1)
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows, _SettlementTable(), creators, _ScriptedFetch(), polls=False
        )
        read_url: Final = f"https://my-resource.openai.azure.com/openai/v1/responses/{STORED_RESPONSE_ID}"

        await _passed_through(
            passthrough_logging,
            _identified_logging_obj(READ_CALL_ID, READER_METADATA),
            httpx.Request("GET", read_url),
            read_url,
            {},
            _stored_response("completed", True, RESPONSES_USAGE),
            custom_llm_provider=read_route_provider,
        )

        assert creators.failed_lookups == []
        (read_row,) = rows.logged_for(READER_METADATA["user_api_key_team_id"])
        assert read_row["spend"] > 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "read_route_provider",
        [pytest.param("azure", id="azure_passthrough_route"), pytest.param(None, id="configured_route")],
    )
    async def test_an_azure_finished_streamed_read_bills_the_reader_without_reading_the_creator_table(
        self, read_route_provider: str | None, local_model_cost_map
    ):
        creators: Final = _CreatorTable(lookup_failures=1)
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            _DispatchedSpendRows(), _SettlementTable(), creators, _ScriptedFetch(), polls=False
        )
        recorder: Final = _SpendRowRecorder()
        reader: Final = _identified_logging_obj(READ_CALL_ID, READER_METADATA, stream=True, callbacks=(recorder,))
        if read_route_provider is not None:
            reader.model_call_details["custom_llm_provider"] = read_route_provider

        await PassThroughStreamingHandler.route_streaming_logging_to_handler(
            litellm_logging_obj=reader,
            passthrough_success_handler_obj=passthrough_logging,
            url_route=f"https://my-resource.openai.azure.com/openai/v1/responses/{STORED_RESPONSE_ID}?stream=true",
            request_body={},
            endpoint_type=EndpointType.OPENAI,
            start_time=DISPATCH_TIME,
            raw_bytes=_completed_stream_bytes(_stored_response("completed", True, RESPONSES_USAGE)),
            end_time=DISPATCH_TIME,
        )
        streamed_read: Final = await asyncio.wait_for(recorder.call, timeout=5)

        assert creators.failed_lookups == []
        assert streamed_read.spend_row["team_id"] == READER_METADATA["user_api_key_team_id"]
        assert streamed_read.spend_row["spend"] > 0

    @pytest.mark.asyncio
    async def test_a_poll_claim_the_database_drops_is_retried_and_bills_the_creator_once(
        self, local_model_cost_map
    ):
        rows: Final = _DispatchedSpendRows()
        creators: Final = _CreatorTable()
        fetch: Final = _ScriptedFetch()
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows, _ClaimUnrecordedOnCall(_SettlementTable(), failing_call=1), creators, fetch, polls=True
        )
        await _background_create(passthrough_logging)
        fetch.outcomes.put_nowait(Fetched(body=_stored_response("completed", True, RESPONSES_USAGE)))
        fetch.outcomes.put_nowait(Fetched(body=_stored_response("completed", True, RESPONSES_USAGE)))

        await _poll_settled(creators)
        assert len(fetch.requests) == 2
        assert len(rows.billed()) == 1
        _assert_billed_to_the_creator(rows.billed()[0], rows.spend_rows()[0], await _reader_paid_spend())

        await _finished_read(passthrough_logging)

        (read_row,) = rows.logged_for(READER_METADATA["user_api_key_team_id"])
        assert len(rows.billed()) == 1
        assert not read_row["spend"]

    @pytest.mark.asyncio
    async def test_a_poll_claim_the_database_drops_after_a_read_billed_the_creator_does_not_bill_again(
        self, local_model_cost_map
    ):
        rows: Final = _DispatchedSpendRows()
        creators: Final = _CreatorTable()
        fetch: Final = _ScriptedFetch()
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows, _ClaimUnrecordedOnCall(_SettlementTable(), failing_call=2), creators, fetch, polls=True
        )
        await _background_create(passthrough_logging)
        poll: Final = _running_poll(passthrough_logging)
        await _finished_read(passthrough_logging)

        fetch.outcomes.put_nowait(Fetched(body=_stored_response("completed", True, RESPONSES_USAGE)))
        fetch.outcomes.put_nowait(Fetched(body=_stored_response("completed", True, RESPONSES_USAGE)))

        await asyncio.wait_for(poll, timeout=5)
        assert len(rows.billed()) == 1
        _assert_billed_to_the_creator(rows.billed()[0], rows.spend_rows()[0], await _reader_paid_spend())

    @pytest.mark.asyncio
    async def test_a_read_claim_the_database_commits_after_its_timeout_still_lets_the_poll_bill_the_creator(
        self, local_model_cost_map
    ):
        rows: Final = _DispatchedSpendRows()
        creators: Final = _CreatorTable()
        fetch: Final = _ScriptedFetch()
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows,
            _ClaimAcksLateOnCall(_SettlementTable(), late_call=1, ack_delay_seconds=0.05),
            creators,
            fetch,
            polls=True,
            schedule=HALF_SECOND_POLLS,
            settlement_timeout_seconds=0.01,
        )
        await _background_create(passthrough_logging)
        await _finished_read(passthrough_logging)
        fetch.outcomes.put_nowait(Fetched(body=_stored_response("completed", True, RESPONSES_USAGE)))

        await _poll_settled(creators)

        assert len(rows.billed()) == 1
        _assert_billed_to_the_creator(rows.billed()[0], rows.spend_rows()[0], await _reader_paid_spend())

    @pytest.mark.asyncio
    async def test_a_poll_claim_the_database_commits_after_its_timeout_is_retried_and_bills_the_creator(
        self, local_model_cost_map
    ):
        rows: Final = _DispatchedSpendRows()
        creators: Final = _CreatorTable()
        fetch: Final = _ScriptedFetch()
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows,
            _ClaimAcksLateOnCall(_SettlementTable(), late_call=1, ack_delay_seconds=0.05),
            creators,
            fetch,
            polls=True,
            schedule=HALF_SECOND_POLLS,
            settlement_timeout_seconds=0.01,
        )
        await _background_create(passthrough_logging)
        fetch.outcomes.put_nowait(Fetched(body=_stored_response("completed", True, RESPONSES_USAGE)))
        fetch.outcomes.put_nowait(Fetched(body=_stored_response("completed", True, RESPONSES_USAGE)))

        await _poll_settled(creators)

        assert len(fetch.requests) == 2
        assert len(rows.billed()) == 1
        _assert_billed_to_the_creator(rows.billed()[0], rows.spend_rows()[0], await _reader_paid_spend())

    @pytest.mark.asyncio
    async def test_a_late_poll_claim_that_lost_to_a_read_keeps_the_read_claim_so_a_later_read_is_not_billed(
        self, local_model_cost_map
    ):
        rows: Final = _DispatchedSpendRows()
        creators: Final = _CreatorTable()
        fetch: Final = _ScriptedFetch()
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows,
            _ClaimAcksLateOnCall(_SettlementTable(), late_call=2, ack_delay_seconds=0.05),
            creators,
            fetch,
            polls=True,
            schedule=HALF_SECOND_POLLS,
            settlement_timeout_seconds=0.01,
        )
        await _background_create(passthrough_logging)
        poll: Final = _running_poll(passthrough_logging)
        await asyncio.wait_for(_first_fetch(fetch), timeout=5)
        await _finished_read(passthrough_logging)
        fetch.outcomes.put_nowait(Fetched(body=_stored_response("completed", True, RESPONSES_USAGE)))
        await asyncio.wait_for(poll, timeout=5)

        await _finished_read(passthrough_logging)

        assert len(rows.billed()) == 1
        _assert_billed_to_the_creator(rows.billed()[0], rows.spend_rows()[0], await _reader_paid_spend())
        assert len(rows.logged_for(READER_METADATA["user_api_key_team_id"])) == 2

    @pytest.mark.asyncio
    async def test_a_poll_cancelled_while_its_claim_is_in_flight_gives_the_claim_back_so_a_read_bills_the_creator(
        self, local_model_cost_map
    ):
        rows: Final = _DispatchedSpendRows()
        settlements: Final = _ClaimCommitsWhenAllowed(_SettlementTable())
        fetch: Final = _ScriptedFetch()
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows, settlements, _CreatorTable(), fetch, polls=True
        )
        await _background_create(passthrough_logging)
        poll: Final = _running_poll(passthrough_logging)
        fetch.outcomes.put_nowait(Fetched(body=_stored_response("completed", True, RESPONSES_USAGE)))
        await asyncio.wait_for(settlements.insert_started.wait(), timeout=5)
        poll.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(poll, timeout=5)
        await settlements.commit_and_wait_for_release()

        await _finished_read(passthrough_logging)

        assert len(rows.billed()) == 1
        _assert_billed_to_the_creator(rows.billed()[0], rows.spend_rows()[0], await _reader_paid_spend())

    @pytest.mark.asyncio
    async def test_a_read_cancelled_while_its_claim_is_in_flight_gives_the_claim_back_so_the_next_read_pays(
        self, local_model_cost_map
    ):
        rows: Final = _DispatchedSpendRows()
        settlements: Final = _ClaimCommitsWhenAllowed(_SettlementTable())
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows, settlements, _CreatorTable(), _ScriptedFetch(), polls=False
        )
        read: Final = asyncio.ensure_future(_finished_read(passthrough_logging, finalize_before_dispatch=False))
        await asyncio.wait_for(settlements.insert_started.wait(), timeout=5)
        read.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(read, timeout=5)
        await settlements.commit_and_wait_for_release()

        await _finished_read(passthrough_logging)

        assert len(rows.billed()) == 1
        assert rows.billed()[0].spend_row["team_id"] == READER_METADATA["user_api_key_team_id"]
        assert rows.billed()[0].spend_row["spend"] == pytest.approx(await _reader_paid_spend())

    @pytest.mark.asyncio
    async def test_the_creator_bill_spans_the_run_under_its_own_logging_id_so_it_never_overwrites_the_create_log(
        self, local_model_cost_map
    ):
        rows: Final = _DispatchedSpendRows()
        creators: Final = _CreatorTable()
        fetch: Final = _ScriptedFetch()
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows, _SettlementTable(), creators, fetch, polls=True
        )
        created_at: Final = DISPATCH_TIME + timedelta(seconds=1)
        completed_at: Final = DISPATCH_TIME + timedelta(seconds=42)
        await _background_create(passthrough_logging)
        fetch.outcomes.put_nowait(
            Fetched(
                body={
                    **_stored_response("completed", True, RESPONSES_USAGE),
                    "created_at": int(created_at.timestamp()),
                    "completed_at": int(completed_at.timestamp()),
                }
            )
        )

        await _poll_settled(creators)

        create, bill = rows.rows
        assert bill.spend_row["spend"] > 0
        assert (bill.start_time, bill.end_time) == (created_at, completed_at)
        timing: Final = bill.logging_obj.v2_call_facts(bill.end_time).timing
        assert (timing.start, timing.end) == (created_at.timestamp(), completed_at.timestamp())
        assert litellm.utils.get_logging_id(
            bill.start_time, bill.logging_obj.model_call_details["standard_logging_object"]
        ) != litellm.utils.get_logging_id(
            create.start_time, create.logging_obj.model_call_details["standard_logging_object"]
        )

    @pytest.mark.asyncio
    async def test_a_background_create_that_finishes_at_create_is_not_billed_again_by_a_read(
        self, local_model_cost_map
    ):
        rows: Final = _DispatchedSpendRows()
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows, _SettlementTable(), _CreatorTable(), _ScriptedFetch(), polls=False
        )
        request_body: Final = {**CREATE_REQUEST, "background": True}
        await _passed_through(
            passthrough_logging,
            _identified_logging_obj(CREATE_CALL_ID, CREATOR_METADATA),
            httpx.Request("POST", RESPONSES_URL, json=request_body),
            RESPONSES_URL,
            request_body,
            _stored_response("completed", True, RESPONSES_USAGE),
        )

        await _finished_read(passthrough_logging)

        (read_row,) = rows.logged_for(READER_METADATA["user_api_key_team_id"])
        assert [row.spend_row["team_id"] for row in rows.billed()] == [CREATOR_METADATA["user_api_key_team_id"]]
        assert rows.billed()[0].spend_row["spend"] == pytest.approx(await _reader_paid_spend())
        assert not read_row["spend"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "read_route_provider",
        [pytest.param("openai", id="openai_passthrough_route"), pytest.param(None, id="configured_route")],
    )
    async def test_a_finished_streamed_read_by_another_key_bills_the_key_that_created_the_response(
        self, read_route_provider: str | None, local_model_cost_map
    ):
        rows: Final = _DispatchedSpendRows()
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows, _SettlementTable(), _CreatorTable(), _ScriptedFetch(), polls=False
        )
        await _background_create(passthrough_logging)
        recorder: Final = _SpendRowRecorder()
        reader: Final = _identified_logging_obj(READ_CALL_ID, READER_METADATA, stream=True, callbacks=(recorder,))
        if read_route_provider is not None:
            reader.model_call_details["custom_llm_provider"] = read_route_provider

        await PassThroughStreamingHandler.route_streaming_logging_to_handler(
            litellm_logging_obj=reader,
            passthrough_success_handler_obj=passthrough_logging,
            url_route=f"{RESPONSES_URL}/{STORED_RESPONSE_ID}?stream=true",
            request_body={},
            endpoint_type=EndpointType.OPENAI,
            start_time=DISPATCH_TIME,
            raw_bytes=_completed_stream_bytes(_stored_response("completed", True, RESPONSES_USAGE)),
            end_time=DISPATCH_TIME,
        )
        streamed_read: Final = await asyncio.wait_for(recorder.call, timeout=5)

        assert len(rows.billed()) == 1
        _assert_billed_to_the_creator(rows.billed()[0], rows.spend_rows()[0], await _reader_paid_spend())
        assert streamed_read.spend_row["team_id"] == READER_METADATA["user_api_key_team_id"]
        assert not streamed_read.spend_row["spend"]

    @pytest.mark.asyncio
    async def test_a_creator_bill_that_fails_on_a_read_is_billed_once_by_a_later_read(self, local_model_cost_map):
        rows: Final = _DispatchedSpendRows()
        creators: Final = _CreatorTable()
        dispatch: Final = _CreatorDispatchFailsAfterTheCreate(rows, refusals=1)
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            dispatch, _SettlementTable(), creators, _ScriptedFetch(), polls=False
        )
        await _background_create(passthrough_logging)

        await _finished_read(passthrough_logging)

        (read_row,) = rows.logged_for(READER_METADATA["user_api_key_team_id"])
        assert dispatch.refused == [CREATE_CALL_ID]
        assert read_row["request_id"] == READ_CALL_ID
        assert not read_row["spend"]
        assert rows.billed() == ()

        await _finished_read(passthrough_logging)

        assert len(rows.billed()) == 1
        _assert_billed_to_the_creator(rows.billed()[0], rows.spend_rows()[0], await _reader_paid_spend())
        assert not any(row["spend"] for row in rows.logged_for(READER_METADATA["user_api_key_team_id"]))

    @pytest.mark.asyncio
    async def test_a_failed_creator_bill_whose_claim_cannot_be_released_answers_every_read_and_bills_nobody(
        self, local_model_cost_map
    ):
        rows: Final = _DispatchedSpendRows()
        creators: Final = _CreatorTable()
        dispatch: Final = _CreatorDispatchFailsAfterTheCreate(rows, refusals=1)
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            dispatch, _UnreleasableSettlementTable(), creators, _ScriptedFetch(), polls=False
        )
        await _background_create(passthrough_logging)

        await _finished_read(passthrough_logging)
        await _finished_read(passthrough_logging)

        assert dispatch.refused == [CREATE_CALL_ID]
        assert len(rows.logged_for(READER_METADATA["user_api_key_team_id"])) == 2
        assert not any(row["spend"] for row in rows.logged_for(READER_METADATA["user_api_key_team_id"]))
        assert rows.billed() == ()

    @pytest.mark.asyncio
    async def test_a_creator_bill_that_fails_at_the_poll_is_retried_and_bills_the_creator_once(
        self, local_model_cost_map
    ):
        rows: Final = _DispatchedSpendRows()
        creators: Final = _CreatorTable()
        fetch: Final = _ScriptedFetch()
        dispatch: Final = _CreatorDispatchFailsAfterTheCreate(rows, refusals=1)
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            dispatch, _SettlementTable(), creators, fetch, polls=True
        )
        await _background_create(passthrough_logging)

        fetch.outcomes.put_nowait(Fetched(body=_stored_response("completed", True, RESPONSES_USAGE)))
        fetch.outcomes.put_nowait(Fetched(body=_stored_response("completed", True, RESPONSES_USAGE)))

        await _poll_settled(creators)
        assert dispatch.refused == [CREATE_CALL_ID]
        assert len(fetch.requests) == 2
        assert len(rows.billed()) == 1
        _assert_billed_to_the_creator(rows.billed()[0], rows.spend_rows()[0], await _reader_paid_spend())

    @pytest.mark.asyncio
    async def test_the_bill_leaves_the_creators_in_flight_request_holding_its_legacy_limiter_slot(
        self, local_model_cost_map
    ):
        rows: Final = _DispatchedSpendRows()
        creators: Final = _CreatorTable()
        fetch: Final = _ScriptedFetch()
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows, _SettlementTable(), creators, fetch, polls=True
        )
        await _background_create(
            passthrough_logging,
            logging_obj=_identified_logging_obj(
                CREATE_CALL_ID, {**CREATOR_METADATA, "user_api_key_model_max_budget": {}}
            ),
        )
        fetch.outcomes.put_nowait(Fetched(body=_stored_response("completed", True, RESPONSES_USAGE)))
        await _poll_settled(creators)
        (bill,) = rows.billed()
        billed_at: Final = datetime(2026, 10, 17, 13, 48, 30)
        limiter: Final = PROXY_MaxParallelRequestsHandler(
            internal_usage_cache=InternalUsageCache(DualCache()), clock=lambda: billed_at
        )
        bucket: Final = f"{CREATOR_METADATA['user_api_key']}::{billed_at.strftime('%Y-%m-%d-%H-%M')}::request_count"
        await limiter.internal_usage_cache.async_set_cache(
            key=bucket,
            value={"current_requests": 1, "current_tpm": 0, "current_rpm": 1},
            litellm_parent_otel_span=None,
        )

        await limiter.async_log_success_event(
            kwargs=bill.logging_obj.model_call_details,
            response_obj=ModelResponse(usage=Usage(prompt_tokens=8, completion_tokens=4, total_tokens=12)),
            start_time=billed_at,
            end_time=billed_at,
        )

        assert await limiter.internal_usage_cache.async_get_cache(key=bucket, litellm_parent_otel_span=None) == {
            "current_requests": 1,
            "current_tpm": 12,
            "current_rpm": 1,
        }

    @pytest.mark.asyncio
    async def test_a_read_whose_claim_the_database_drops_bills_nobody_and_the_poll_bills_the_creator_once(
        self, local_model_cost_map
    ):
        rows: Final = _DispatchedSpendRows()
        creators: Final = _CreatorTable()
        fetch: Final = _ScriptedFetch()
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows, _ClaimUnrecordedOnCall(_SettlementTable(), failing_call=1), creators, fetch, polls=True
        )
        await _background_create(passthrough_logging)

        await _finished_read(passthrough_logging)

        (read_row,) = rows.logged_for(READER_METADATA["user_api_key_team_id"])
        assert not read_row["spend"]
        assert rows.billed() == ()

        fetch.outcomes.put_nowait(Fetched(body=_stored_response("completed", True, RESPONSES_USAGE)))
        await _poll_settled(creators)

        assert len(rows.billed()) == 1
        _assert_billed_to_the_creator(rows.billed()[0], rows.spend_rows()[0], await _reader_paid_spend())

    @pytest.mark.asyncio
    async def test_the_bill_of_a_no_log_create_skips_external_loggers_but_runs_the_proxy_spend_logger(
        self, local_model_cost_map
    ):
        rows: Final = _DispatchedSpendRows()
        creators: Final = _CreatorTable()
        fetch: Final = _ScriptedFetch()
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows, _SettlementTable(), creators, fetch, polls=True
        )
        create: Final = _identified_logging_obj(
            CREATE_CALL_ID,
            CREATOR_METADATA,
            litellm_params={
                "no-log": True,
                "proxy_server_request": {"headers": {"x-litellm-disable-callbacks": "Langfuse, datadog"}},
            },
        )
        await _background_create(passthrough_logging, logging_obj=create)

        fetch.outcomes.put_nowait(Fetched(body=_stored_response("completed", True, RESPONSES_USAGE)))
        await _poll_settled(creators)

        (bill,) = rows.billed()
        bill_params: Final = bill.logging_obj.model_call_details["litellm_params"]
        _assert_billed_to_the_creator(bill, rows.spend_rows()[0], await _reader_paid_spend())
        assert not bill.logging_obj.should_run_callback(CustomLogger(), bill_params, "async_success_handler")
        assert bill.logging_obj.should_run_callback(_PROXY_SpendLogger(), bill_params, "async_success_handler")
        assert set(
            EnterpriseCallbackControls.get_disabled_callbacks(
                bill_params, bill.logging_obj.standard_callback_dynamic_params
            )
        ) == set(
            EnterpriseCallbackControls.get_disabled_callbacks(
                create.model_call_details["litellm_params"], create.standard_callback_dynamic_params
            )
        )

    @pytest.mark.asyncio
    async def test_a_read_that_cannot_look_up_the_creator_bills_nobody_and_a_later_read_bills_the_creator_once(
        self, local_model_cost_map
    ):
        rows: Final = _DispatchedSpendRows()
        creators: Final = _CreatorTable(lookup_failures=1)
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows, _SettlementTable(), creators, _ScriptedFetch(), polls=False
        )
        await _background_create(passthrough_logging)

        await _finished_read(passthrough_logging)

        assert creators.failed_lookups == [STORED_RESPONSE_ID]
        assert not rows.billed()

        await _finished_read(passthrough_logging)

        assert len(rows.billed()) == 1
        _assert_billed_to_the_creator(rows.billed()[0], rows.spend_rows()[0], await _reader_paid_spend())

    @pytest.mark.asyncio
    async def test_the_bill_of_a_create_with_team_logging_goes_to_the_creator_keys_team_loggers(
        self, local_model_cost_map
    ):
        rows: Final = _DispatchedSpendRows()
        creators: Final = _CreatorTable()
        fetch: Final = _ScriptedFetch()
        creator_team_logger: Final = _SpendRowRecorder()

        async def creator_key_wiring(context: BackgroundResponseCreateContext) -> TeamCallbackWiring | None:
            if context.metadata.get("user_api_key") != CREATOR_METADATA["user_api_key"]:
                return None
            return TeamCallbackWiring(success_callbacks=[creator_team_logger])

        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows, _SettlementTable(), creators, fetch, polls=True, creator_logging_wiring=creator_key_wiring
        )
        await _background_create(
            passthrough_logging,
            logging_obj=_identified_logging_obj(CREATE_CALL_ID, CREATOR_METADATA, callbacks=(CustomLogger(),)),
        )

        fetch.outcomes.put_nowait(Fetched(body=_stored_response("completed", True, RESPONSES_USAGE)))
        await _poll_settled(creators)
        team_logged: Final = await asyncio.wait_for(creator_team_logger.call, timeout=5)

        (bill,) = rows.billed()
        _assert_billed_to_the_creator(bill, rows.spend_rows()[0], await _reader_paid_spend())
        assert team_logged.spend_row["team_id"] == CREATOR_METADATA["user_api_key_team_id"]
        assert team_logged.spend_row["spend"] == pytest.approx(bill.spend_row["spend"])

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("create_has_team_logging", "external_loggers_get_the_bill"),
        [pytest.param(True, False, id="team_logging"), pytest.param(False, True, id="global_logging_only")],
    )
    async def test_the_bill_whose_creator_key_cannot_be_loaded_keeps_team_logs_out_of_global_loggers(
        self, local_model_cost_map, create_has_team_logging: bool, external_loggers_get_the_bill: bool
    ):
        rows: Final = _DispatchedSpendRows()
        creators: Final = _CreatorTable()
        fetch: Final = _ScriptedFetch()

        async def creator_key_not_found(context: BackgroundResponseCreateContext) -> TeamCallbackWiring | None:
            return None

        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows, _SettlementTable(), creators, fetch, polls=True, creator_logging_wiring=creator_key_not_found
        )
        create_callbacks: Final = (CustomLogger(),) if create_has_team_logging else ()
        await _background_create(
            passthrough_logging,
            logging_obj=_identified_logging_obj(CREATE_CALL_ID, CREATOR_METADATA, callbacks=create_callbacks),
        )

        fetch.outcomes.put_nowait(Fetched(body=_stored_response("completed", True, RESPONSES_USAGE)))
        await _poll_settled(creators)

        (bill,) = rows.billed()
        bill_params: Final = bill.logging_obj.model_call_details["litellm_params"]
        _assert_billed_to_the_creator(bill, rows.spend_rows()[0], await _reader_paid_spend())
        assert (
            bill.logging_obj.should_run_callback(CustomLogger(), bill_params, "async_success_handler")
            is external_loggers_get_the_bill
        )
        assert bill.logging_obj.should_run_callback(_PROXY_SpendLogger(), bill_params, "async_success_handler")

    @pytest.mark.asyncio
    async def test_the_proxy_wiring_sends_the_bill_to_the_logging_the_creator_key_has_now(self, monkeypatch):
        key_cache: Final = UserApiKeyCache()
        key_cache.set_cache(
            AuthObject.virtual_key(CREATOR_METADATA["user_api_key"]),
            UserAPIKeyAuth(
                token=CREATOR_METADATA["user_api_key"],
                metadata={
                    "logging": [
                        {
                            "callback_name": "langfuse",
                            "callback_type": "success",
                            "callback_vars": {"langfuse_public_key": "pk-lf-creator", "langfuse_secret_key": "sk-lf"},
                        }
                    ]
                },
            ),
        )
        monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace())
        monkeypatch.setattr(proxy_server, "user_api_key_cache", key_cache)

        wiring: Final = await proxy_creator_logging_wiring(_creator_context(CREATOR_METADATA))

        assert wiring is not None
        assert wiring.success_callbacks == ["langfuse"]
        assert wiring.logging_kwargs is not None
        assert wiring.logging_kwargs["langfuse_public_key"] == "pk-lf-creator"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("metadata", "prisma_client"),
        [
            pytest.param(CREATOR_METADATA, SimpleNamespace(), id="key_not_loadable"),
            pytest.param({"user_api_key_team_id": "team-creator"}, SimpleNamespace(), id="no_creator_key"),
        ],
    )
    async def test_the_proxy_wiring_has_no_logging_when_the_creator_key_cannot_be_resolved(
        self, monkeypatch, metadata: Mapping[str, object], prisma_client: SimpleNamespace | None
    ):
        monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
        monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())

        assert await proxy_creator_logging_wiring(_creator_context(metadata)) is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "key_cache",
        [pytest.param(_StalledKeyCache(), id="stalled"), pytest.param(_DeadlineKeyCache(), id="past_deadline")],
    )
    async def test_the_proxy_wiring_defers_the_bill_when_the_creator_key_lookup_times_out(
        self, monkeypatch, key_cache: _StalledKeyCache | _DeadlineKeyCache
    ):
        monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace())
        monkeypatch.setattr(proxy_server, "user_api_key_cache", key_cache)

        resolving: Final = asyncio.ensure_future(
            proxy_creator_logging_wiring(_creator_context(CREATOR_METADATA), timeout_seconds=0.05)
        )
        done, _ = await asyncio.wait({resolving}, timeout=2)
        resolving.cancel()

        assert resolving in done
        assert isinstance(resolving.exception(), (TimeoutError, asyncio.TimeoutError))

    @pytest.mark.asyncio
    async def test_the_creator_bill_ends_when_the_response_completed(self, local_model_cost_map):
        rows: Final = _DispatchedSpendRows()
        creators: Final = _CreatorTable()
        fetch: Final = _ScriptedFetch()
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows, _SettlementTable(), creators, fetch, polls=True
        )
        completed_at: Final = DISPATCH_TIME + timedelta(seconds=42)
        await _background_create(passthrough_logging)

        fetch.outcomes.put_nowait(
            Fetched(
                body={
                    **_stored_response("completed", True, RESPONSES_USAGE),
                    "completed_at": int(completed_at.timestamp()),
                }
            )
        )
        await _poll_settled(creators)

        (bill,) = rows.billed()
        assert bill.end_time == completed_at

    @pytest.mark.asyncio
    async def test_the_creator_bill_never_ends_before_the_create_started(self, local_model_cost_map):
        rows: Final = _DispatchedSpendRows()
        creators: Final = _CreatorTable()
        fetch: Final = _ScriptedFetch()
        passthrough_logging: Final = _creator_billing_passthrough_logging(
            rows, _SettlementTable(), creators, fetch, polls=True
        )
        await _background_create(passthrough_logging)

        fetch.outcomes.put_nowait(
            Fetched(
                body={
                    **_stored_response("completed", True, RESPONSES_USAGE),
                    "completed_at": int((DISPATCH_TIME - timedelta(seconds=42)).timestamp()),
                }
            )
        )
        await _poll_settled(creators)

        (bill,) = rows.billed()
        assert bill.end_time > DISPATCH_TIME


DECISIONS_URL: Final = "https://api.openai.com/v1/decisions"
DECISIONS_MODEL: Final = "gpt-6-luna"
DECISIONS_INPUT_TOKENS: Final = 1000
DECISIONS_OUTPUT_TOKENS: Final = 7
DECISIONS_REQUEST: Final = {
    "input": "The package arrived crushed and the screen is cracked.",
    "questions": [{"type": "predicate", "name": "is_defect", "instructions": "Is the product defective?"}],
}
DECISION: Final = {
    "model": DECISIONS_MODEL,
    "answers": [{"type": "predicate", "name": "is_defect", "probability": 0.97}],
    "usage": {
        "input_tokens": DECISIONS_INPUT_TOKENS,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": DECISIONS_OUTPUT_TOKENS,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": DECISIONS_INPUT_TOKENS + DECISIONS_OUTPUT_TOKENS,
    },
}


class TestOpenAIPassthroughDecisionsSpendLog:
    @staticmethod
    async def _spend_row(method: str, url_route: str, request_body: dict[str, object]) -> SpendLogsPayload:
        logged: Final = await TestOpenAIPassthroughResponsesSubRouteBilling()._logged_call(
            method, "", request_body, DECISION, "4e9b2c71-8a3d-4f5e-b6c0-1d2e3f4a5b6c", url_route
        )
        return logged.spend_row

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("url_route", "requested_model"),
        [
            pytest.param(DECISIONS_URL, DECISIONS_MODEL, id="openai"),
            pytest.param(f"{DECISIONS_URL}/", DECISIONS_MODEL, id="trailing_slash"),
            pytest.param(DECISIONS_URL, "luna-prod-alias", id="unpriced_alias_billed_as_served_model"),
        ],
    )
    async def test_decisions_passthrough_spend_log_is_priced_like_native_decisions(
        self, local_model_cost_map, url_route: str, requested_model: str
    ):
        decisions_rates: Final = litellm.model_cost[DECISIONS_MODEL]

        spend_log_row: Final = await self._spend_row(
            "POST", url_route, {**DECISIONS_REQUEST, "model": requested_model}
        )

        assert spend_log_row["prompt_tokens"] == DECISIONS_INPUT_TOKENS
        assert spend_log_row["completion_tokens"] == DECISIONS_OUTPUT_TOKENS
        assert spend_log_row["spend"] == pytest.approx(
            DECISIONS_INPUT_TOKENS * decisions_rates["input_cost_per_token_decisions"]
            + DECISIONS_OUTPUT_TOKENS * decisions_rates["output_cost_per_token_decisions"]
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "url_route"),
        [
            pytest.param("GET", DECISIONS_URL, id="list"),
            pytest.param("GET", f"{DECISIONS_URL}/dec_6a0b0f8e", id="retrieve"),
            pytest.param("POST", f"{DECISIONS_URL}/dec_6a0b0f8e/cancel", id="sub_route"),
        ],
    )
    async def test_decisions_reads_and_sub_routes_are_not_billed(
        self, local_model_cost_map, caplog: pytest.LogCaptureFixture, method: str, url_route: str
    ):
        with caplog.at_level(logging.WARNING):
            spend_log_row: Final = await self._spend_row(
                method, url_route, {**DECISIONS_REQUEST, "model": DECISIONS_MODEL}
            )

        assert not spend_log_row["spend"]
        assert [record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR] == []


if __name__ == "__main__":
    pytest.main([__file__])


ONE_PIXEL_PNG_DATA_URL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
UNREACHABLE_IMAGE_URL = "http://127.0.0.1:9/doc.png"
TEXT_ONLY_MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "Describe this"}]}]


def _image_messages(url: str, detail: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this"},
                {"type": "image_url", "image_url": {"url": url, "detail": detail}},
            ],
        }
    ]


def test_count_relayed_prompt_tokens_counts_a_data_url_image_exactly():
    messages = _image_messages(ONE_PIXEL_PNG_DATA_URL, "high")

    assert count_relayed_prompt_tokens("gpt-4.1-mini", messages) == litellm.token_counter(
        model="gpt-4.1-mini", messages=messages
    )


def test_count_relayed_prompt_tokens_keeps_a_low_detail_remote_image_at_the_base_count():
    messages = _image_messages(UNREACHABLE_IMAGE_URL, "low")

    assert count_relayed_prompt_tokens("gpt-4.1-mini", messages) == litellm.token_counter(
        model="gpt-4.1-mini", messages=messages
    )
    assert count_relayed_prompt_tokens("gpt-4.1-mini", messages) < high_detail_image_token_upper_bound()


def test_count_relayed_prompt_tokens_charges_only_the_remote_high_detail_image_at_the_upper_bound():
    messages = _image_messages(UNREACHABLE_IMAGE_URL, "high")

    assert count_relayed_prompt_tokens("gpt-4.1-mini", messages) == (
        litellm.token_counter(model="gpt-4.1-mini", messages=TEXT_ONLY_MESSAGES) + high_detail_image_token_upper_bound()
    )


@pytest.mark.parametrize("scheme", ["HTTPS://", "Http://"])
def test_count_relayed_prompt_tokens_charges_an_uppercase_scheme_remote_high_detail_image_at_the_upper_bound(scheme):
    messages = _image_messages(scheme + UNREACHABLE_IMAGE_URL.split("://", 1)[1], "high")

    assert count_relayed_prompt_tokens("gpt-4.1-mini", messages) == (
        litellm.token_counter(model="gpt-4.1-mini", messages=TEXT_ONLY_MESSAGES) + high_detail_image_token_upper_bound()
    )


NEGOTIATED_INPUT_COST: Final = 3.3e-7
NEGOTIATED_OUTPUT_COST: Final = 7.7e-6
CHAT_URL: Final = "https://api.openai.com/v1/chat/completions"
CHAT_REQUEST: Final = {"model": "gpt-5.4-mini", "messages": [{"role": "user", "content": "hi"}]}
CHAT_USAGE: Final = {"prompt_tokens": 1000, "completion_tokens": 1000, "total_tokens": 2000}
CHAT_CHOICE: Final = {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
EMBEDDINGS_URL: Final = "https://api.openai.com/v1/embeddings"
EMBEDDINGS_REQUEST: Final = {"model": "text-embedding-3-small", "input": "hello"}
EMBEDDINGS_RESPONSE: Final = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.0, 1.0]}],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 1000, "total_tokens": 1000},
}


def _chat(object_type: str, **fields: object) -> dict[str, object]:
    return {"id": "chatcmpl-1", "object": object_type, "created": 1, "model": "gpt-5.4-mini", **fields}


def _sse(*events: Mapping[str, object]) -> tuple[bytes, ...]:
    return tuple(f"data: {json.dumps(event)}\n\n".encode() for event in events)


def _negotiated_deployment_logging_obj(model: str, stream: bool, recorder: _SpendRowRecorder) -> LiteLLMLoggingObj:
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import openai_deployment_pricing_model_info

    pricing: Final = {"input_cost_per_token": NEGOTIATED_INPUT_COST, "output_cost_per_token": NEGOTIATED_OUTPUT_COST}
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {"model": f"openai/{model}", "api_key": "sk-test"},
                "model_info": {"id": "negotiated-openai-deployment", **pricing},
            }
        ]
    )
    model_info: Final = openai_deployment_pricing_model_info("https://api.openai.com/", router, {"model": model}, None)
    assert model_info is not None, "the passthrough found no OpenAI deployment to price this model"
    return _identified_logging_obj("negotiated-price-call", {"model_info": dict(model_info)}, stream, (recorder,))


async def _billed_spend(url: str, request_body: dict[str, object], upstream: object) -> float:
    recorder: Final = _SpendRowRecorder()
    stream: Final = isinstance(upstream, tuple)
    logging_obj: Final = _negotiated_deployment_logging_obj(str(request_body["model"]), stream, recorder)
    if stream:
        await PassThroughStreamingHandler.route_streaming_logging_to_handler(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=PassThroughEndpointLogging(),
            url_route=url,
            request_body=request_body,
            endpoint_type=EndpointType.OPENAI,
            start_time=DISPATCH_TIME,
            raw_bytes=list(upstream),
            end_time=DISPATCH_TIME,
        )
    else:
        await _passed_through(
            PassThroughEndpointLogging(), logging_obj, httpx.Request("POST", url), url, request_body, upstream
        )
    return (await asyncio.wait_for(recorder.call, timeout=5)).spend_row["spend"]


@pytest.mark.parametrize(
    ("url", "request_body", "upstream", "input_tokens", "output_tokens"),
    [
        pytest.param(
            CHAT_URL,
            CHAT_REQUEST,
            _chat("chat.completion", choices=[CHAT_CHOICE], usage=CHAT_USAGE),
            1000,
            1000,
            id="chat",
        ),
        pytest.param(
            CHAT_URL,
            {**CHAT_REQUEST, "stream": True},
            _sse(
                _chat("chat.completion.chunk", choices=[{**CHAT_CHOICE, "delta": CHAT_CHOICE["message"]}]),
                _chat("chat.completion.chunk", choices=[], usage=CHAT_USAGE),
            ),
            1000,
            1000,
            id="chat-streamed",
        ),
        pytest.param(
            RESPONSES_URL, CREATE_REQUEST, _stored_response("completed", False, RESPONSES_USAGE), 11, 5, id="responses"
        ),
        pytest.param(
            RESPONSES_URL,
            {**CREATE_REQUEST, "stream": True},
            _sse(
                {
                    "type": "response.completed",
                    "sequence_number": 9,
                    "response": _stored_response("completed", False, RESPONSES_USAGE),
                }
            ),
            11,
            5,
            id="responses-streamed",
        ),
        pytest.param(EMBEDDINGS_URL, EMBEDDINGS_REQUEST, EMBEDDINGS_RESPONSE, 1000, 0, id="embeddings"),
    ],
)
@pytest.mark.asyncio
async def test_openai_passthrough_bills_the_matching_deployment_negotiated_price(
    local_model_cost_map, url, request_body, upstream, input_tokens, output_tokens
):
    spend: Final = await _billed_spend(url, request_body, upstream)

    assert spend == pytest.approx(input_tokens * NEGOTIATED_INPUT_COST + output_tokens * NEGOTIATED_OUTPUT_COST), (
        f"passthrough billed {spend} instead of the deployment's negotiated rates"
    )
