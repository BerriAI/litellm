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

    def _create_passthrough_logging_payload(
        self, user: str = "test_user"
    ) -> PassthroughStandardLoggingPayload:
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
            OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route(
                "https://api.openai.com/v1/models"
            )
            == False
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route(
                "http://localhost:4000/openai/v1/chat/completions"
            )
            == False
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route(
                "https://api.anthropic.com/v1/messages"
            )
            == False
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route("")
            == False
        )

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
            OpenAIPassthroughLoggingHandler.is_openai_image_generation_route(
                "https://api.openai.com/v1/images/edits"
            )
            == False
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_image_generation_route(
                "http://localhost:4000/openai/v1/images/generations"
            )
            == False
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_image_generation_route("")
            == False
        )

    def test_is_openai_image_editing_route(self):
        """Test OpenAI image editing route detection"""
        # Positive cases
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_image_editing_route(
                "https://api.openai.com/v1/images/edits"
            )
            == True
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_image_editing_route(
                "https://openai.azure.com/v1/images/edits"
            )
            == True
        )

        # Negative cases
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_image_editing_route(
                "https://api.openai.com/v1/chat/completions"
            )
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
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_image_editing_route("") == False
        )

    def test_is_openai_responses_route(self):
        """Test OpenAI responses API route detection"""
        # Positive cases
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_responses_route(
                "https://api.openai.com/v1/responses"
            )
            == True
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_responses_route(
                "https://openai.azure.com/v1/responses"
            )
            == True
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_responses_route(
                "https://api.openai.com/responses"
            )
            == True
        )

        # Negative cases
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_responses_route(
                "https://api.openai.com/v1/chat/completions"
            )
            == False
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_responses_route(
                "https://api.openai.com/v1/images/generations"
            )
            == False
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_responses_route(
                "http://localhost:4000/openai/v1/responses"
            )
            == False
        )
        assert OpenAIPassthroughLoggingHandler.is_openai_responses_route("") == False

    def test_is_openai_route_recognizes_cognitiveservices_azure_com(self):
        """Azure OpenAI resources created via the newer "Azure AI Foundry" /
        Cognitive Services pathway live on `*.cognitiveservices.azure.com`
        subdomains rather than the older `openai.azure.com`. All four
        is_openai_*_route methods must recognize both Azure subdomains so
        cost tracking applies regardless of which Azure naming the user's
        resource happens to be on.
        """
        cognitive_chat = (
            "https://my-resource.cognitiveservices.azure.com/v1/chat/completions"
        )
        cognitive_images_gen = (
            "https://my-resource.cognitiveservices.azure.com/v1/images/generations"
        )
        cognitive_images_edit = (
            "https://my-resource.cognitiveservices.azure.com/v1/images/edits"
        )
        cognitive_responses = (
            "https://my-resource.cognitiveservices.azure.com/v1/responses"
        )

        assert (
            OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route(
                cognitive_chat
            )
            is True
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_image_generation_route(
                cognitive_images_gen
            )
            is True
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_image_editing_route(
                cognitive_images_edit
            )
            is True
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_responses_route(
                cognitive_responses
            )
            is True
        )

        # Cross-route negatives still hold for cognitiveservices hosts.
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route(
                cognitive_responses
            )
            is False
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_responses_route(cognitive_chat)
            is False
        )

    def test_is_openai_chat_completions_route_classic_azure_deployment_path(self):
        """Classic Azure OpenAI chat completions are shaped
        `/openai/deployments/{deployment-id}/chat/completions?api-version=...`
        — no `/v1/` segment. Requiring `/v1/chat/completions` missed them
        entirely, so those calls were billed upstream and recorded as $0.
        """
        for url in (
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-01",
            "https://my-resource.cognitiveservices.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-01",
            # Deployment ids may contain dots/dashes.
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4.1-mini/chat/completions?api-version=2024-10-21",
        ):
            assert (
                OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route(url)
                is True
            ), url

        # The `/v1/` form keeps working.
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route(
                "https://my-resource.openai.azure.com/openai/v1/chat/completions"
            )
            is True
        )

        # Other classic-deployment operations are not chat completions.
        for url in (
            "https://my-resource.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2024-02-01",
            "https://my-resource.openai.azure.com/openai/deployments/dall-e-3/images/generations?api-version=2024-02-01",
        ):
            assert (
                OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route(url)
                is False
            ), url

    def test_is_openai_image_generation_route_classic_azure_deployment_path(self):
        """Classic Azure DALL-E / gpt-image calls are shaped
        `/openai/deployments/{deployment-id}/images/generations?api-version=...`
        — no `/v1/` segment. Requiring `/v1/images/generations` missed them
        entirely, so every Azure image generation was billed against our Azure
        account and recorded as $0. Same defect already fixed for chat.
        """
        for url in (
            "https://my-resource.openai.azure.com/openai/deployments/dall-e-3/images/generations?api-version=2024-02-01",
            "https://my-resource.cognitiveservices.azure.com/openai/deployments/gpt-image-1/images/generations?api-version=2025-04-01",
            # Deployment ids may contain dots/dashes.
            "https://my-resource.openai.azure.com/openai/deployments/dall-e-3.0/images/generations?api-version=2024-02-01",
        ):
            assert (
                OpenAIPassthroughLoggingHandler.is_openai_image_generation_route(url)
                is True
            ), url

        # The `/v1/` form keeps working.
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_image_generation_route(
                "https://my-resource.openai.azure.com/openai/v1/images/generations"
            )
            is True
        )

        # Other classic-deployment operations are not image generation.
        for url in (
            "https://my-resource.openai.azure.com/openai/deployments/dall-e-3/images/edits?api-version=2024-02-01",
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-01",
        ):
            assert (
                OpenAIPassthroughLoggingHandler.is_openai_image_generation_route(url)
                is False
            ), url

    def test_is_openai_image_editing_route_classic_azure_deployment_path(self):
        """Classic Azure image *edits* have the same shape defect as
        generations — see the sibling test.
        """
        for url in (
            "https://my-resource.openai.azure.com/openai/deployments/gpt-image-1/images/edits?api-version=2025-04-01",
            "https://my-resource.cognitiveservices.azure.com/openai/deployments/gpt-image-1/images/edits?api-version=2025-04-01",
        ):
            assert (
                OpenAIPassthroughLoggingHandler.is_openai_image_editing_route(url)
                is True
            ), url

        # The `/v1/` form keeps working.
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_image_editing_route(
                "https://my-resource.openai.azure.com/openai/v1/images/edits"
            )
            is True
        )

        # Generations are not edits.
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_image_editing_route(
                "https://my-resource.openai.azure.com/openai/deployments/dall-e-3/images/generations?api-version=2024-02-01"
            )
            is False
        )

    def test_classic_azure_image_generation_is_costed_not_zero(self):
        """End-to-end: a classic Azure image generation must produce a real cost.

        Recognising the route is necessary but not sufficient — the classic
        Azure surface carries the model *only* in the URL (the request body has
        no `model`, and an image response body echoes none), so the model must
        be resolved from the deployment segment or the call still records $0.
        """
        url = "https://my-resource.openai.azure.com/openai/deployments/dall-e-3/images/generations?api-version=2024-02-01"
        response_body = {
            "created": 1700000000,
            "data": [{"url": "https://example.com/generated.png"}],
        }
        mock_httpx_response = self._create_mock_httpx_response(response_body)
        mock_logging_obj = self._create_mock_logging_obj()

        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=response_body,
            logging_obj=mock_logging_obj,
            url_route=url,
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={
                "prompt": "a cat",
                "size": "1024x1024",
                "quality": "standard",
                "n": 1,
            },
        )

        # Model comes from the deployment segment, since neither body has one.
        assert result["kwargs"]["model"] == "dall-e-3"
        assert result["kwargs"]["response_cost"] > 0
        # The pass-through spend path reads cost from model_call_details.
        assert mock_logging_obj.model_call_details["response_cost"] > 0

    def test_is_openai_responses_route_matches_whole_segment_only(self):
        """`is_openai_responses_route` used a bare `"/responses" in path`
        containment test, which matched a `responses` segment anywhere on an
        in-scope host — including sibling resources that merely start with the
        word, and unrelated nested routes. Those were then costed with the
        Responses-API transformer, which mis-parses them.
        """
        # Every surface that really serves the Responses API still matches.
        for url in (
            "https://api.openai.com/v1/responses",
            "https://api.openai.com/responses",
            "https://my-resource.openai.azure.com/openai/responses?api-version=preview",
            "https://my-resource.openai.azure.com/openai/v1/responses",
        ):
            assert (
                OpenAIPassthroughLoggingHandler.is_openai_responses_route(url) is True
            ), url

        # Newly rejected: partial-word segments, unrelated nesting — and item
        # routes. Item responses ECHO the original usage block, so costing them
        # re-bills the full generation on every retrieve/cancel; a method gate
        # alone would not stop `POST .../{id}/cancel`.
        for url in (
            "https://api.openai.com/v1/responses_archive",
            "https://api.openai.com/v1/evals/responses",
            "https://api.openai.com/v1/containers/responses",
            "https://api.openai.com/v1/responses/resp_123",
            "https://api.openai.com/v1/responses/resp_123/cancel",
            "https://api.openai.com/v1/responses/resp_123/input_items",
        ):
            assert (
                OpenAIPassthroughLoggingHandler.is_openai_responses_route(url) is False
            ), url

    def test_is_openai_chat_completions_route_rejects_lookalike_hosts(self):
        """Host matching is suffix-based, not substring-based."""
        for url in (
            "https://openai.azure.com.attacker.example/openai/deployments/gpt-4o/chat/completions",
            "https://cognitiveservices.azure.com.attacker.example/v1/chat/completions",
            "https://api.openai.com.attacker.example/v1/chat/completions",
        ):
            assert (
                OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route(url)
                is False
            ), url

    @patch("litellm.completion_cost")
    @patch(
        "litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload"
    )
    def test_openai_passthrough_handler_success(
        self, mock_get_standard_logging, mock_completion_cost
    ):
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
    def test_openai_passthrough_handler_non_chat_completions(
        self, mock_completion_cost
    ):
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
    @patch(
        "litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload"
    )
    def test_openai_passthrough_handler_with_user_tracking(
        self, mock_get_standard_logging, mock_completion_cost
    ):
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
        assert (
            result["kwargs"]["litellm_params"]["proxy_server_request"]["body"]["user"]
            == "test_user_123"
        )

    @patch("litellm.completion_cost")
    def test_openai_passthrough_handler_cost_calculation_error(
        self, mock_completion_cost
    ):
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

    @patch("litellm.completion_cost")
    @patch(
        "litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload"
    )
    def test_different_models_cost_tracking(
        self, mock_get_standard_logging, mock_completion_cost
    ):
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
    @patch(
        "litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload"
    )
    def test_azure_passthrough_tags_metadata_model_provider(
        self, mock_get_standard_logging, mock_completion_cost
    ):
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
        assert (
            result["kwargs"]["custom_llm_provider"] == "azure"
        )  # Should preserve Azure, not default to "openai"
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
    @patch(
        "litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload"
    )
    @patch(
        "litellm.llms.openai.responses.transformation.OpenAIResponsesAPIConfig.transform_response_api_response"
    )
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
    @patch(
        "litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload"
    )
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


    def test_is_openai_embeddings_route(self):
        """Test OpenAI embeddings route detection.

        Both the OpenAI-v1 surface and the classic Azure deployment path
        (`/openai/deployments/{deployment}/embeddings`, which carries no
        `/v1/` segment) must be recognised, or those calls record $0.
        """
        # Positive cases
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_embeddings_route(
                "https://api.openai.com/v1/embeddings"
            )
            is True
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_embeddings_route(
                "https://myresource.openai.azure.com/openai/v1/embeddings"
            )
            is True
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_embeddings_route(
                "https://myresource.openai.azure.com/openai/deployments/emb/embeddings?api-version=2024-02-01"
            )
            is True
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_embeddings_route(
                "https://myresource.cognitiveservices.azure.com/openai/v1/embeddings"
            )
            is True
        )

        # Negative cases — other endpoints and non-OpenAI hosts
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_embeddings_route(
                "https://api.openai.com/v1/chat/completions"
            )
            is False
        )
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_embeddings_route(
                "https://api.anthropic.com/v1/embeddings"
            )
            is False
        )
        assert OpenAIPassthroughLoggingHandler.is_openai_embeddings_route("") is False

    def test_openai_passthrough_handler_embeddings_cost_tracking(self):
        """Regression test: pass-through embeddings must not record $0.

        BUG: the handler had no embeddings predicate at all, so
        `/v1/embeddings` fell through the supported-endpoint check and
        returned `result: None` with no `response_cost`. We billed upstream
        for the tokens and wrote a zero-spend row against the calling
        virtual key.

        This exercises the REAL cost calculator (no mock) so the assertion
        pins actual pricing behavior, not just that some number was set.
        """
        import litellm

        prompt_tokens = 1000
        embeddings_response_body = {
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            },
        }

        mock_httpx_response = self._create_mock_httpx_response(
            embeddings_response_body
        )
        mock_logging_obj = self._create_mock_logging_obj()

        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=embeddings_response_body,
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/embeddings",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"model": "text-embedding-3-small", "input": "hello"},
            passthrough_logging_payload=None,
            litellm_params={},
            custom_llm_provider="openai",
        )

        expected_cost = (
            litellm.model_cost["text-embedding-3-small"]["input_cost_per_token"]
            * prompt_tokens
        )
        assert expected_cost > 0, "test fixture model must have non-zero pricing"

        assert result is not None
        assert result["kwargs"]["response_cost"] == pytest.approx(expected_cost)
        assert result["kwargs"]["model"] == "text-embedding-3-small"
        assert mock_logging_obj.model_call_details["response_cost"] == pytest.approx(
            expected_cost
        )

        # The response object must be an EmbeddingResponse carrying the usage,
        # with the cost pinned in _hidden_params so downstream logging does not
        # recalculate it (same contract as the image branches).
        from litellm.types.utils import EmbeddingResponse

        embedding_response = result["result"]
        assert isinstance(embedding_response, EmbeddingResponse)
        assert embedding_response.usage.prompt_tokens == prompt_tokens
        assert embedding_response.usage.completion_tokens == 0
        assert embedding_response._hidden_params["response_cost"] == pytest.approx(
            expected_cost
        )

    def test_responses_api_streaming_cost_tracking(self):
        """Regression test: streamed Responses API calls must not record $0.

        BUG: `_handle_logging_openai_collected_chunks` reassembled every
        stream with `OpenAIChatCompletionResponseIterator` +
        `stream_chunk_builder`, which understand chat-completion chunks only.
        A streamed Responses call emits `response.*` typed events instead, so
        parsing yielded nothing, the handler bailed with `result: None`, and
        the SpendLogs row was written with zero spend — even though
        non-streaming Responses calls were costed correctly.

        FIX: detect the terminal `response.completed` event and route its
        `response` object through the same
        `_build_responses_api_response_and_cost` path the non-streaming
        branch uses.

        Uses the REAL cost calculator so the numbers are meaningful.
        """
        import litellm

        input_tokens = 1000
        output_tokens = 500
        completed_response = {
            "id": "resp_abc123",
            "object": "response",
            "created_at": 1677652288,
            "model": "gpt-4o-2024-08-06",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello!"}],
                }
            ],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }

        # A realistic Responses event stream: the SSE `event:` line and its
        # `data:` payload arrive as separate entries because
        # `_convert_raw_bytes_to_str_lines` splits on newlines.
        all_chunks = [
            "event: response.created",
            "data: "
            + json.dumps({"type": "response.created", "response": {"id": "resp_abc123"}}),
            "event: response.output_text.delta",
            "data: " + json.dumps({"type": "response.output_text.delta", "delta": "Hello!"}),
            "event: response.completed",
            "data: "
            + json.dumps({"type": "response.completed", "response": completed_response}),
            "data: [DONE]",
        ]

        mock_logging_obj = self._create_mock_logging_obj()

        result = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
            litellm_logging_obj=mock_logging_obj,
            passthrough_success_handler_obj=None,
            url_route="https://api.openai.com/v1/responses",
            request_body={"model": "gpt-4o", "stream": True},
            endpoint_type=None,
            start_time=self.start_time,
            all_chunks=all_chunks,
            end_time=self.end_time,
        )

        model_pricing = litellm.model_cost["gpt-4o"]
        expected_cost = (
            model_pricing["input_cost_per_token"] * input_tokens
            + model_pricing["output_cost_per_token"] * output_tokens
        )
        assert expected_cost > 0, "test fixture model must have non-zero pricing"

        # Pre-fix this is the failing assertion: the chat-chunk reassembly
        # returns None, so the handler returns `{"result": None, "kwargs": {}}`
        # and there is no response_cost at all.
        assert result["kwargs"].get("response_cost") == pytest.approx(expected_cost)
        assert mock_logging_obj.model_call_details["response_cost"] == pytest.approx(
            expected_cost
        )

        from litellm.types.llms.openai import ResponsesAPIResponse

        assert isinstance(result["result"], ResponsesAPIResponse), (
            "streamed Responses calls must reassemble into a "
            "ResponsesAPIResponse via the shared Responses costing path"
        )

    def test_responses_streaming_detection_leaves_chat_streams_alone(self):
        """The Responses event detector must not hijack chat-completion streams.

        Chat chunks carry no `type: response.completed`, so detection returns
        None and the existing `stream_chunk_builder` path still runs — this
        guards the regression risk of the streaming fix.
        """
        chat_chunks = [
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-123",
                    "object": "chat.completion.chunk",
                    "created": 1677652288,
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "Hello"},
                            "finish_reason": None,
                        }
                    ],
                }
            ),
            "data: [DONE]",
        ]

        assert (
            OpenAIPassthroughLoggingHandler._extract_responses_api_completed_response(
                all_chunks=chat_chunks
            )
            is None
        )

        # And a Responses stream IS detected, returning the completed
        # `response` object (not the event envelope).
        responses_chunks = [
            "event: response.completed",
            "data: "
            + json.dumps(
                {
                    "type": "response.completed",
                    "response": {"id": "resp_1", "usage": {"input_tokens": 5}},
                }
            ),
        ]
        detected = (
            OpenAIPassthroughLoggingHandler._extract_responses_api_completed_response(
                all_chunks=responses_chunks
            )
        )
        assert detected is not None
        assert detected["id"] == "resp_1"


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

    def _create_passthrough_logging_payload(
        self, user: str = "test_user"
    ) -> PassthroughStandardLoggingPayload:
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
        assert (
            self.handler.is_openai_route("https://api.openai.com/v1/chat/completions")
            == True
        )
        assert (
            self.handler.is_openai_route("https://openai.azure.com/v1/chat/completions")
            == True
        )
        assert self.handler.is_openai_route("https://api.openai.com/v1/models") == True
        # Azure OpenAI on the shared Cognitive Services domain, identified by an
        # OpenAI-style path segment.
        assert (
            self.handler.is_openai_route(
                "https://my-resource.cognitiveservices.azure.com/v1/chat/completions"
            )
            == True
        )

        # Negative cases
        assert (
            self.handler.is_openai_route(
                "http://localhost:4000/openai/v1/chat/completions"
            )
            == False
        )
        assert (
            self.handler.is_openai_route("https://api.anthropic.com/v1/messages")
            == False
        )
        assert (
            self.handler.is_openai_route("https://api.assemblyai.com/v2/transcript")
            == False
        )
        # Non-OpenAI Azure Cognitive Services share the `cognitiveservices.azure.com`
        # domain but must NOT be classified as OpenAI routes (no OpenAI path segment).
        assert (
            self.handler.is_openai_route(
                "https://my-resource.cognitiveservices.azure.com/speechtotext/v3.1/recognize"
            )
            == False
        )
        assert (
            self.handler.is_openai_route(
                "https://my-resource.cognitiveservices.azure.com/vision/v3.2/analyze"
            )
            == False
        )
        # A look-alike domain that merely contains an OpenAI host as a substring
        # must be rejected by the suffix-based hostname match.
        assert (
            self.handler.is_openai_route(
                "https://cognitiveservices.azure.com.attacker.example/v1/chat/completions"
            )
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
        assert (
            self.handler._is_supported_openai_endpoint(
                "https://api.openai.com/v1/responses"
            )
            is True
        )
        assert (
            self.handler._is_supported_openai_endpoint(
                "https://openai.azure.com/v1/responses"
            )
            is True
        )
        # The other supported endpoints stay supported (no regression).
        assert (
            self.handler._is_supported_openai_endpoint(
                "https://api.openai.com/v1/chat/completions"
            )
            is True
        )
        assert (
            self.handler._is_supported_openai_endpoint(
                "https://api.openai.com/v1/images/generations"
            )
            is True
        )
        assert (
            self.handler._is_supported_openai_endpoint(
                "https://api.openai.com/v1/images/edits"
            )
            is True
        )
        # Unsupported OpenAI endpoints (e.g. /v1/models) still return False.
        assert (
            self.handler._is_supported_openai_endpoint(
                "https://api.openai.com/v1/models"
            )
            is False
        )

    def test_is_supported_openai_endpoint_includes_embeddings(self):
        """Regression test for the embeddings dispatch gate.

        `_is_supported_openai_endpoint` allow-listed exactly four shapes
        (chat, image generation, image editing, responses). Embeddings were
        absent, so `/v1/embeddings` never reached the handler and every
        pass-through embedding call was billed upstream but recorded as $0
        against the calling virtual key.
        """
        assert (
            self.handler._is_supported_openai_endpoint(
                "https://api.openai.com/v1/embeddings"
            )
            is True
        )
        assert (
            self.handler._is_supported_openai_endpoint(
                "https://myresource.openai.azure.com/openai/v1/embeddings"
            )
            is True
        )
        assert (
            self.handler._is_supported_openai_endpoint(
                "https://myresource.openai.azure.com/openai/deployments/emb/embeddings?api-version=2024-02-01"
            )
            is True
        )
        # Still not a blanket allow: unrelated OpenAI routes stay unsupported.
        assert (
            self.handler._is_supported_openai_endpoint(
                "https://api.openai.com/v1/models"
            )
            is False
        )

    @patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler.OpenAIPassthroughLoggingHandler.openai_passthrough_handler"
    )
    @pytest.mark.asyncio
    async def test_success_handler_dispatches_responses_api_to_openai_handler(
        self, mock_openai_handler
    ):
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
        mock_httpx_response.text = (
            '{"id": "chatcmpl-123", "choices": [{"message": {"content": "Hello"}}]}'
        )

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

        assert (
            calculated_cost == test_cost
        ), f"Expected {test_cost}, got {calculated_cost}"

    @patch("litellm.cost_calculator.default_image_cost_calculator")
    def test_openai_passthrough_handler_image_generation(
        self, mock_image_cost_calculator
    ):
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


if __name__ == "__main__":
    pytest.main([__file__])


class TestOpenAICompatibleProviderScope:
    """`is_openai_route` / the `is_openai_*_route` helpers must key off
    `custom_llm_provider`, not a hardcoded hostname allow-list.

    Pass-through forwards a request upstream using OUR credentials. The OpenAI
    handler's cost math is already provider-agnostic — it transforms the
    response with the OpenAI config and hands the usage to `completion_cost`
    together with the provider. But entry to that handler was gated on a
    hardcoded tuple of hostnames (`api.openai.com` + the two Azure domains), so
    every other OpenAI-compatible upstream (Fireworks, Groq, Together, ...) was
    never dispatched and recorded $0 while still billing us.
    """

    def setup_method(self):
        self.success_handler = PassThroughEndpointLogging()

    FIREWORKS_CHAT = "https://api.fireworks.ai/inference/v1/chat/completions"
    GROQ_CHAT = "https://api.groq.com/openai/v1/chat/completions"

    @pytest.mark.parametrize(
        "url, provider",
        [
            (FIREWORKS_CHAT, "fireworks_ai"),
            (GROQ_CHAT, "groq"),
            ("https://api.together.xyz/v1/chat/completions", "together_ai"),
            ("https://api.deepseek.com/v1/chat/completions", "deepseek"),
        ],
    )
    def test_openai_compatible_providers_are_in_scope(self, url, provider):
        """An OpenAI-compatible provider is in scope regardless of hostname."""
        assert self.success_handler.is_openai_route(url, provider) is True
        assert (
            self.success_handler._is_supported_openai_endpoint(url, provider) is True
        )

    def test_fireworks_is_in_scope_without_a_configured_provider(self):
        """Generic pass-throughs (`general_settings.pass_through_endpoints`)
        have no `custom_llm_provider` field at all, so the hostname must still
        place a known OpenAI-protocol upstream in scope."""
        assert self.success_handler.is_openai_route(self.FIREWORKS_CHAT) is True
        assert (
            self.success_handler._is_supported_openai_endpoint(self.FIREWORKS_CHAT)
            is True
        )

    @pytest.mark.parametrize(
        "url",
        [
            "https://api.openai.com/v1/chat/completions",
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions",
            "https://my-resource.cognitiveservices.azure.com/v1/chat/completions",
        ],
    )
    def test_hostname_path_still_works_without_a_provider(self, url):
        """No regression: OpenAI/Azure keep matching on hostname alone."""
        assert self.success_handler.is_openai_route(url) is True
        assert self.success_handler._is_supported_openai_endpoint(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://my-resource.cognitiveservices.azure.com/speechtotext/v3.1/transcriptions",
            "https://my-resource.cognitiveservices.azure.com/vision/v3.2/analyze",
        ],
    )
    def test_azure_path_marker_guard_survives_a_provider_label(self, url):
        """The shared Azure domains also host Speech / Vision / Language. A
        `custom_llm_provider: azure` label must NOT let those bypass the
        `/openai/` `/v1/` path-marker guard and be costed as chat completions."""
        assert self.success_handler.is_openai_route(url, "azure") is False
        assert self.success_handler.is_openai_route(url) is False

    def test_non_openai_providers_stay_out_of_scope(self):
        """Providers with their own dedicated handler must not be swallowed."""
        for provider in ("anthropic", "gemini", "vertex_ai", "cohere", "cursor"):
            assert (
                self.success_handler.is_openai_route(
                    "https://api.anthropic.com/v1/messages", provider
                )
                is False
            ), provider

    def test_unknown_host_without_a_provider_stays_out_of_scope(self):
        """The provider is what widens the scope — an unlabelled unknown host
        must not silently become an OpenAI route."""
        assert (
            self.success_handler.is_openai_route(
                "https://api.example.com/v1/chat/completions"
            )
            is False
        )

    def test_fireworks_lookalike_host_is_rejected(self):
        """Suffix matching, not a substring test."""
        assert (
            self.success_handler.is_openai_route(
                "https://api.fireworks.ai.attacker.example/v1/chat/completions"
            )
            is False
        )

    @pytest.mark.parametrize(
        "predicate, url",
        [
            ("is_openai_chat_completions_route", FIREWORKS_CHAT),
            (
                "is_openai_embeddings_route",
                "https://api.fireworks.ai/inference/v1/embeddings",
            ),
            (
                "is_openai_responses_route",
                "https://api.fireworks.ai/inference/v1/responses",
            ),
        ],
    )
    def test_route_predicates_accept_a_provider(self, predicate, url):
        """Each `is_openai_*_route` helper takes `custom_llm_provider`."""
        assert getattr(OpenAIPassthroughLoggingHandler, predicate)(
            url, "fireworks_ai"
        ) is True


class TestFireworksPassthroughCostTracking:
    """Fireworks pass-through must resolve to a real price, not $0.

    Two defects: (a) Fireworks prices are keyed `fireworks_ai/accounts/...` in
    the price map while a native response echoes the bare `accounts/...` id, and
    the handler defaulted `custom_llm_provider` to "openai" — which makes
    `completion_cost` raise "model isn't mapped yet", swallowed by the handler,
    so the call was billed upstream and recorded at $0. (b) Streamed Fireworks
    responses were classified `EndpointType.GENERIC`, which costs nothing.
    """

    FIREWORKS_MODEL = "accounts/fireworks/models/deepseek-v3"
    FIREWORKS_CHAT_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
    PROMPT_TOKENS = 1000
    COMPLETION_TOKENS = 500

    def setup_method(self):
        self.start_time = datetime.now()
        self.end_time = datetime.now()
        self.response_body = {
            "id": "chatcmpl-fw-1",
            "object": "chat.completion",
            "created": 1677652288,
            "model": self.FIREWORKS_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": self.PROMPT_TOKENS,
                "completion_tokens": self.COMPLETION_TOKENS,
                "total_tokens": self.PROMPT_TOKENS + self.COMPLETION_TOKENS,
            },
        }

    def _expected_cost(self) -> float:
        """Price straight from the map, so a price refresh cannot break this."""
        import litellm

        entry = litellm.model_cost["fireworks_ai/" + self.FIREWORKS_MODEL]
        return (
            self.PROMPT_TOKENS * entry["input_cost_per_token"]
            + self.COMPLETION_TOKENS * entry["output_cost_per_token"]
        )

    def _mock_httpx_response(self) -> httpx.Response:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.text = json.dumps(self.response_body)
        mock_response.json.return_value = self.response_body
        mock_response.headers = {"content-type": "application/json"}
        return mock_response

    def _mock_logging_obj(self) -> LiteLLMLoggingObj:
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {}
        return mock_logging_obj

    def _run_handler(self, custom_llm_provider=None):
        kwargs = {
            "passthrough_logging_payload": PassthroughStandardLoggingPayload(
                url=self.FIREWORKS_CHAT_URL,
                request_body={"model": self.FIREWORKS_MODEL},
                request_method="POST",
            ),
            "litellm_params": {},
        }
        if custom_llm_provider is not None:
            kwargs["custom_llm_provider"] = custom_llm_provider
        return OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=self._mock_httpx_response(),
            response_body=self.response_body,
            logging_obj=self._mock_logging_obj(),
            url_route=self.FIREWORKS_CHAT_URL,
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={
                "model": self.FIREWORKS_MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
            },
            **kwargs,
        )

    def test_bare_fireworks_model_id_resolves_to_a_real_price(self):
        """The map is keyed `fireworks_ai/accounts/...`; the wire id is bare."""
        result = self._run_handler(custom_llm_provider="fireworks_ai")

        assert result["kwargs"]["custom_llm_provider"] == "fireworks_ai"
        assert result["kwargs"]["response_cost"] > 0, "Fireworks recorded $0"
        assert result["kwargs"]["response_cost"] == pytest.approx(
            self._expected_cost()
        )

    def test_provider_is_inferred_when_the_passthrough_declares_none(self):
        """A generic pass-through carries no `custom_llm_provider`, and
        defaulting to "openai" made the price lookup raise and record $0."""
        result = self._run_handler(custom_llm_provider=None)

        assert result["kwargs"]["custom_llm_provider"] == "fireworks_ai"
        assert result["kwargs"]["response_cost"] == pytest.approx(
            self._expected_cost()
        )

    def test_prefixed_fireworks_model_id_is_normalized(self):
        """`fireworks_ai/accounts/...` must not double up into
        `fireworks_ai/fireworks_ai/accounts/...` when it reaches the calculator."""
        from litellm.proxy.pass_through_endpoints.common_utils import (
            normalize_fireworks_model_id,
        )

        assert (
            normalize_fireworks_model_id("fireworks_ai/" + self.FIREWORKS_MODEL)
            == self.FIREWORKS_MODEL
        )
        assert normalize_fireworks_model_id(self.FIREWORKS_MODEL) == self.FIREWORKS_MODEL
        # Non-Fireworks models are returned untouched.
        assert normalize_fireworks_model_id("gpt-4o") == "gpt-4o"
        assert normalize_fireworks_model_id("azure/my-deployment") == "azure/my-deployment"

    def test_unmapped_fireworks_model_falls_back_to_the_parameter_size_tier(self):
        """LiteLLM's Fireworks calculator maps an unknown id to a size tier
        (`fireworks-ai-above-16b`, ...). Route through it rather than $0."""
        import litellm

        unmapped = "accounts/fireworks/models/some-unreleased-70b"
        assert unmapped not in litellm.model_cost
        assert "fireworks_ai/" + unmapped not in litellm.model_cost

        self.FIREWORKS_MODEL = unmapped
        self.response_body["model"] = unmapped
        result = self._run_handler(custom_llm_provider="fireworks_ai")

        tier = litellm.model_cost["fireworks-ai-above-16b"]
        assert result["kwargs"]["response_cost"] == pytest.approx(
            self.PROMPT_TOKENS * tier["input_cost_per_token"]
            + self.COMPLETION_TOKENS * tier["output_cost_per_token"]
        )

    def test_openai_costing_is_unchanged(self):
        """No regression: an OpenAI route with no provider still prices as
        OpenAI, not as an inferred third party."""
        import litellm

        self.FIREWORKS_MODEL = "gpt-4o"
        self.response_body["model"] = "gpt-4o"
        self.FIREWORKS_CHAT_URL = "https://api.openai.com/v1/chat/completions"
        result = self._run_handler(custom_llm_provider=None)

        entry = litellm.model_cost["gpt-4o"]
        assert result["kwargs"]["custom_llm_provider"] == "openai"
        assert result["kwargs"]["response_cost"] == pytest.approx(
            self.PROMPT_TOKENS * entry["input_cost_per_token"]
            + self.COMPLETION_TOKENS * entry["output_cost_per_token"]
        )

    def test_streamed_fireworks_chunks_are_costed(self):
        """Streaming path: reassembled chunks must price via Fireworks."""
        import litellm

        chunk_template = {
            "id": "chatcmpl-fw-stream",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": self.FIREWORKS_MODEL,
        }
        all_chunks = [
            json.dumps(
                {
                    **chunk_template,
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant", "content": "Hi"}}
                    ],
                }
            ),
            json.dumps(
                {
                    **chunk_template,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": self.PROMPT_TOKENS,
                        "completion_tokens": self.COMPLETION_TOKENS,
                        "total_tokens": self.PROMPT_TOKENS + self.COMPLETION_TOKENS,
                    },
                }
            ),
        ]

        logging_obj = self._mock_logging_obj()
        result = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=PassThroughEndpointLogging(),
            url_route=self.FIREWORKS_CHAT_URL,
            request_body={"model": self.FIREWORKS_MODEL},
            endpoint_type=EndpointType.OPENAI,
            start_time=self.start_time,
            all_chunks=all_chunks,
            end_time=self.end_time,
        )

        assert result["kwargs"]["custom_llm_provider"] == "fireworks_ai"
        assert result["kwargs"]["response_cost"] > 0, "streamed Fireworks recorded $0"
        assert result["kwargs"]["response_cost"] == pytest.approx(
            self._expected_cost()
        )
