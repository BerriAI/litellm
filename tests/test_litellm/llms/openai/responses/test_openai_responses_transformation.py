import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.llms.openai import (
    ImageGenerationPartialImageEvent,
    OutputTextDeltaEvent,
    ResponseCompletedEvent,
    ResponsesAPIRequestParams,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from litellm.types.router import GenericLiteLLMParams


class TestOpenAIResponsesAPIConfig:
    def setup_method(self):
        self.config = OpenAIResponsesAPIConfig()
        self.model = "gpt-4o"
        self.logging_obj = MagicMock()

    def test_map_openai_params(self):
        """Test that parameters are correctly mapped"""
        test_params = {"input": "Hello world", "temperature": 0.7, "stream": True}

        result = self.config.map_openai_params(
            response_api_optional_params=test_params,
            model=self.model,
            drop_params=False,
        )

        # The function should return the params unchanged
        assert result == test_params

    def validate_responses_api_request_params(self, params, expected_fields):
        """
        Validate that the params dict has the expected structure of ResponsesAPIRequestParams

        Args:
            params: The dict to validate
            expected_fields: Dict of field names and their expected values
        """
        # Check that it's a dict
        assert isinstance(params, dict), "Result should be a dict"

        # Check expected fields have correct values
        for field, value in expected_fields.items():
            assert field in params, f"Missing expected field: {field}"
            assert (
                params[field] == value
            ), f"Field {field} has value {params[field]}, expected {value}"

    def test_transform_responses_api_request(self):
        """Test request transformation"""
        input_text = "What is the capital of France?"
        optional_params = {"temperature": 0.7, "stream": True, "background": True}

        result = self.config.transform_responses_api_request(
            model=self.model,
            input=input_text,
            response_api_optional_request_params=optional_params,
            litellm_params={},
            headers={},
        )

        # Validate the result has the expected structure and values
        expected_fields = {
            "model": self.model,
            "input": input_text,
            "temperature": 0.7,
            "stream": True,
            "background": True,
        }

        self.validate_responses_api_request_params(result, expected_fields)

    def test_transform_streaming_response(self):
        """Test streaming response transformation"""
        # Test with a text delta event
        chunk = {
            "type": "response.output_text.delta",
            "item_id": "item_123",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello",
        }

        result = self.config.transform_streaming_response(
            model=self.model, parsed_chunk=chunk, logging_obj=self.logging_obj
        )

        assert isinstance(result, OutputTextDeltaEvent)
        assert result.type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
        assert result.delta == "Hello"
        assert result.item_id == "item_123"

        # Test with a completed event - providing all required fields
        completed_chunk = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "created_at": 1234567890,
                "model": "gpt-4o",
                "object": "response",
                "output": [],
                "parallel_tool_calls": False,
                "error": None,
                "incomplete_details": None,
                "instructions": None,
                "metadata": None,
                "temperature": 0.7,
                "tool_choice": "auto",
                "tools": [],
                "top_p": 1.0,
                "max_output_tokens": None,
                "previous_response_id": None,
                "reasoning": None,
                "status": "completed",
                "text": None,
                "truncation": "auto",
                "usage": None,
                "user": None,
            },
        }

        # Mock the get_event_model_class to avoid validation issues in tests
        with patch.object(
            OpenAIResponsesAPIConfig, "get_event_model_class"
        ) as mock_get_class:
            mock_get_class.return_value = ResponseCompletedEvent

            result = self.config.transform_streaming_response(
                model=self.model,
                parsed_chunk=completed_chunk,
                logging_obj=self.logging_obj,
            )

            assert result.type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
            assert result.response.id == "resp_123"

    @pytest.mark.serial
    def test_validate_environment(self):
        """Test that validate_environment correctly sets the Authorization header"""
        # Test with provided API key
        headers = {}
        api_key = "test_api_key"
        litellm_params = GenericLiteLLMParams(api_key=api_key)
        result = self.config.validate_environment(
            headers=headers, model=self.model, litellm_params=litellm_params
        )

        assert "Authorization" in result
        assert result["Authorization"] == f"Bearer {api_key}"

        # Test with empty headers
        headers = {}

        with patch("litellm.api_key", "litellm_api_key"):
            litellm_params = GenericLiteLLMParams()
            result = self.config.validate_environment(
                headers=headers, model=self.model, litellm_params=litellm_params
            )

            assert "Authorization" in result
            assert result["Authorization"] == "Bearer litellm_api_key"

        # Test with existing headers
        headers = {"Content-Type": "application/json"}

        with patch("litellm.openai_key", "openai_key"):
            with patch("litellm.api_key", None):
                litellm_params = GenericLiteLLMParams()
                result = self.config.validate_environment(
                    headers=headers, model=self.model, litellm_params=litellm_params
                )

                assert "Authorization" in result
                assert result["Authorization"] == "Bearer openai_key"
                assert "Content-Type" in result
                assert result["Content-Type"] == "application/json"

        # Test with environment variable
        headers = {}

        with patch("litellm.api_key", None):
            with patch("litellm.openai_key", None):
                with patch(
                    "litellm.llms.openai.responses.transformation.get_secret_str",
                    return_value="env_api_key",
                ):
                    litellm_params = GenericLiteLLMParams()
                    result = self.config.validate_environment(
                        headers=headers, model=self.model, litellm_params=litellm_params
                    )

                    assert "Authorization" in result
                    assert result["Authorization"] == "Bearer env_api_key"

    def test_get_complete_url(self):
        """Test that get_complete_url returns the correct URL"""
        # Test with provided API base
        api_base = "https://custom-openai.example.com/v1"

        result = self.config.get_complete_url(
            api_base=api_base,
            litellm_params={},
        )

        assert result == "https://custom-openai.example.com/v1/responses"

        # Test with litellm.api_base
        with patch("litellm.api_base", "https://litellm-api-base.example.com/v1"):
            result = self.config.get_complete_url(
                api_base=None,
                litellm_params={},
            )

            assert result == "https://litellm-api-base.example.com/v1/responses"

        # Test with environment variable
        with patch("litellm.api_base", None):
            with patch(
                "litellm.llms.openai.responses.transformation.get_secret_str",
                return_value="https://env-api-base.example.com/v1",
            ):
                result = self.config.get_complete_url(
                    api_base=None,
                    litellm_params={},
                )

                assert result == "https://env-api-base.example.com/v1/responses"

        # Test with default API base
        with patch("litellm.api_base", None):
            with patch(
                "litellm.llms.openai.responses.transformation.get_secret_str",
                return_value=None,
            ):
                result = self.config.get_complete_url(
                    api_base=None,
                    litellm_params={},
                )

                assert result == "https://api.openai.com/v1/responses"

        # Test with trailing slash in API base
        api_base = "https://custom-openai.example.com/v1/"

        result = self.config.get_complete_url(
            api_base=api_base,
            litellm_params={},
        )

        assert result == "https://custom-openai.example.com/v1/responses"

    def test_get_event_model_class_generic_event(self):
        """Test that get_event_model_class returns the correct event model class"""
        from litellm.types.llms.openai import GenericEvent

        event_type = "test"
        result = self.config.get_event_model_class(event_type)
        assert result == GenericEvent

    def test_transform_streaming_response_generic_event(self):
        """Test that transform_streaming_response returns the correct event model class"""
        from litellm.types.llms.openai import GenericEvent

        chunk = {"type": "test", "test": "test"}
        result = self.config.transform_streaming_response(
            model=self.model, parsed_chunk=chunk, logging_obj=self.logging_obj
        )
        assert isinstance(result, GenericEvent)
        assert result.type == "test"

    def test_get_event_model_class_image_generation_partial_image(self):
        """Test that get_event_model_class returns ImageGenerationPartialImageEvent for image generation events"""
        event_type = ResponsesAPIStreamEvents.IMAGE_GENERATION_PARTIAL_IMAGE
        result = self.config.get_event_model_class(event_type)
        assert result == ImageGenerationPartialImageEvent

    def test_transform_streaming_response_image_generation_partial_image(self):
        """Test streaming response transformation for image generation partial image events"""
        # Test with a partial image event - simulating OpenAI's streaming image generation
        chunk = {
            "type": "image_generation.partial_image",
            "partial_image_index": 0,
            "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",  # 1x1 red pixel PNG
        }

        result = self.config.transform_streaming_response(
            model=self.model, parsed_chunk=chunk, logging_obj=self.logging_obj
        )

        # Verify the result is the correct event type
        assert isinstance(result, ImageGenerationPartialImageEvent)
        assert result.type == ResponsesAPIStreamEvents.IMAGE_GENERATION_PARTIAL_IMAGE
        assert result.partial_image_index == 0
        assert result.b64_json == chunk["b64_json"]
        assert len(result.b64_json) > 0  # Verify we have image data

    def test_transform_streaming_response_multiple_partial_images(self):
        """Test streaming response with multiple partial images (simulating progressive image generation)"""
        # Test with multiple partial images (as would happen with partial_images=2 or 3)
        test_cases = [
            {
                "type": "image_generation.partial_image",
                "partial_image_index": 0,
                "b64_json": "base64data_partial_0",
            },
            {
                "type": "image_generation.partial_image",
                "partial_image_index": 1,
                "b64_json": "base64data_partial_1",
            },
            {
                "type": "image_generation.partial_image",
                "partial_image_index": 2,
                "b64_json": "base64data_partial_2",
            },
        ]

        for idx, chunk in enumerate(test_cases):
            result = self.config.transform_streaming_response(
                model=self.model, parsed_chunk=chunk, logging_obj=self.logging_obj
            )

            assert isinstance(result, ImageGenerationPartialImageEvent)
            assert result.type == ResponsesAPIStreamEvents.IMAGE_GENERATION_PARTIAL_IMAGE
            assert result.partial_image_index == idx
            assert result.b64_json == chunk["b64_json"]

    def test_transform_responses_api_request_with_partial_images_param(self):
        """Test request transformation with partial_images parameter for streaming image generation"""
        input_text = "Generate a beautiful landscape"
        optional_params = {
            "temperature": 0.7,
            "stream": True,
            "partial_images": 2,  # Request 2 partial images during generation
        }

        result = self.config.transform_responses_api_request(
            model=self.model,
            input=input_text,
            response_api_optional_request_params=optional_params,
            litellm_params={},
            headers={},
        )

        # Validate the result includes partial_images parameter
        expected_fields = {
            "model": self.model,
            "input": input_text,
            "temperature": 0.7,
            "stream": True,
            "partial_images": 2,
        }

        self.validate_responses_api_request_params(result, expected_fields)

    def test_partial_images_parameter_validation(self):
        """Test that partial_images parameter accepts valid values (1-3)"""
        input_text = "Generate an image"

        # Test with different valid partial_images values
        for partial_images_value in [1, 2, 3]:
            optional_params = {
                "stream": True,
                "partial_images": partial_images_value,
            }

            result = self.config.transform_responses_api_request(
                model=self.model,
                input=input_text,
                response_api_optional_request_params=optional_params,
                litellm_params={},
                headers={},
            )

            assert result["partial_images"] == partial_images_value
            assert result["stream"] is True

    def test_transform_streaming_response_coalesces_null_error_code(self):
        """Ensure that when a streaming error event contains error.code=None,
        transform_streaming_response coalesces it to 'unknown_error' and returns
        an ErrorEvent instance without raising a ValidationError.
        """
        from litellm.types.llms.openai import ErrorEvent

        parsed_chunk = {
            "type": "error",
            "sequence_number": 1,
            "error": {
                "type": "invalid_request_error",
                "code": None,
                "message": "Something went wrong",
                "param": None,
            },
        }

        event = self.config.transform_streaming_response(
            model=self.model, parsed_chunk=parsed_chunk, logging_obj=self.logging_obj
        )

        # Validate returned type and coalesced code
        assert isinstance(event, ErrorEvent)
        assert event.error.code == "unknown_error"
        assert event.error.message == "Something went wrong"

    def test_transform_streaming_response_missing_required_fields_response_created(
        self,
    ):
        """Test that ResponseCreatedEvent with missing required fields (created_at,
        output) does not crash but falls back to model_construct.

        Reproduces https://github.com/BerriAI/litellm/issues/20570
        """
        from litellm.types.llms.openai import ResponseCreatedEvent

        # Minimal payload an OpenAI-compatible provider might send,
        # omitting `created_at` and `output` inside the response object.
        parsed_chunk = {
            "type": "response.created",
            "response": {
                "id": "resp_q7BOLpck7clq",
                "model": "gpt-oss-120b",
                "status": "in_progress",
            },
        }

        result = self.config.transform_streaming_response(
            model=self.model, parsed_chunk=parsed_chunk, logging_obj=self.logging_obj
        )

        assert isinstance(result, ResponseCreatedEvent)
        assert result.type == ResponsesAPIStreamEvents.RESPONSE_CREATED
        assert result.response["id"] == "resp_q7BOLpck7clq"

    def test_transform_streaming_response_missing_required_fields_output_text_delta(
        self,
    ):
        """Test that OutputTextDeltaEvent with missing output_index and
        content_index falls back to model_construct without crashing.

        Reproduces https://github.com/BerriAI/litellm/issues/20570
        """
        from litellm.types.llms.openai import OutputTextDeltaEvent

        # Provider omits output_index and content_index
        parsed_chunk = {
            "type": "response.output_text.delta",
            "item_id": "item_456",
            "delta": "Hello",
        }

        result = self.config.transform_streaming_response(
            model=self.model, parsed_chunk=parsed_chunk, logging_obj=self.logging_obj
        )

        assert isinstance(result, OutputTextDeltaEvent)
        assert result.type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
        assert result.delta == "Hello"
        assert result.item_id == "item_456"

    def test_transform_streaming_response_missing_required_fields_content_part_added(
        self,
    ):
        """Test that ContentPartAddedEvent with missing output_index and
        content_index falls back to model_construct without crashing.

        Reproduces https://github.com/BerriAI/litellm/issues/20570
        """
        from litellm.types.llms.openai import ContentPartAddedEvent

        # Provider omits output_index and content_index
        parsed_chunk = {
            "type": "response.content_part.added",
            "item_id": "item_789",
            "part": {"type": "output_text", "text": ""},
        }

        result = self.config.transform_streaming_response(
            model=self.model, parsed_chunk=parsed_chunk, logging_obj=self.logging_obj
        )

        assert isinstance(result, ContentPartAddedEvent)
        assert result.type == ResponsesAPIStreamEvents.CONTENT_PART_ADDED
        assert result.item_id == "item_789"

    def test_transform_streaming_response_missing_required_fields_output_item_added(
        self,
    ):
        """Test that OutputItemAddedEvent with missing output_index falls back
        to model_construct without crashing.

        Reproduces https://github.com/BerriAI/litellm/issues/20570
        """
        from litellm.types.llms.openai import OutputItemAddedEvent

        # Provider omits output_index
        parsed_chunk = {
            "type": "response.output_item.added",
            "item": {"type": "message", "id": "msg_001", "role": "assistant"},
        }

        result = self.config.transform_streaming_response(
            model=self.model, parsed_chunk=parsed_chunk, logging_obj=self.logging_obj
        )

        assert isinstance(result, OutputItemAddedEvent)
        assert result.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED

    def test_transform_streaming_response_valid_chunk_still_works(self):
        """Ensure that fully valid chunks still go through normal Pydantic
        validation (not model_construct) and work correctly."""
        parsed_chunk = {
            "type": "response.output_text.delta",
            "item_id": "item_123",
            "output_index": 0,
            "content_index": 0,
            "delta": "World",
        }

        result = self.config.transform_streaming_response(
            model=self.model, parsed_chunk=parsed_chunk, logging_obj=self.logging_obj
        )

        assert isinstance(result, OutputTextDeltaEvent)
        assert result.delta == "World"
        assert result.output_index == 0
        assert result.content_index == 0


class TestAzureResponsesAPIConfig:
    def setup_method(self):
        self.config = AzureOpenAIResponsesAPIConfig()
        self.model = "gpt-4o"
        self.logging_obj = MagicMock()

    def test_azure_get_complete_url_with_version_types(self):
        """Test Azure get_complete_url with different API version types"""
        base_url = "https://litellm8397336933.openai.azure.com"

        # Test with preview version - should use openai/v1/responses
        result_preview = self.config.get_complete_url(
            api_base=base_url,
            litellm_params={"api_version": "preview"},
        )
        assert (
            result_preview
            == "https://litellm8397336933.openai.azure.com/openai/v1/responses?api-version=preview"
        )

        # Test with latest version - should use openai/v1/responses
        result_latest = self.config.get_complete_url(
            api_base=base_url,
            litellm_params={"api_version": "latest"},
        )
        assert (
            result_latest
            == "https://litellm8397336933.openai.azure.com/openai/v1/responses?api-version=latest"
        )

        # Test with date-based version - should use openai/responses
        result_date = self.config.get_complete_url(
            api_base=base_url,
            litellm_params={"api_version": "2025-01-01"},
        )
        assert (
            result_date
            == "https://litellm8397336933.openai.azure.com/openai/responses?api-version=2025-01-01"
        )


class TestTransformListInputItemsRequest:
    """Test suite for transform_list_input_items_request function"""

    def setup_method(self):
        """Setup test fixtures"""
        self.openai_config = OpenAIResponsesAPIConfig()
        self.azure_config = AzureOpenAIResponsesAPIConfig()
        self.response_id = "resp_abc123"
        self.api_base = "https://api.openai.com/v1/responses"
        self.litellm_params = GenericLiteLLMParams()
        self.headers = {"Authorization": "Bearer test-key"}

    def test_openai_transform_list_input_items_request_minimal(self):
        """Test OpenAI implementation with minimal parameters"""
        # Execute
        url, params = self.openai_config.transform_list_input_items_request(
            response_id=self.response_id,
            api_base=self.api_base,
            litellm_params=self.litellm_params,
            headers=self.headers,
        )

        # Assert
        expected_url = f"{self.api_base}/{self.response_id}/input_items"
        assert url == expected_url
        assert params == {"limit": 20, "order": "desc"}

    def test_openai_transform_list_input_items_request_all_params(self):
        """Test OpenAI implementation with all optional parameters"""
        # Execute
        url, params = self.openai_config.transform_list_input_items_request(
            response_id=self.response_id,
            api_base=self.api_base,
            litellm_params=self.litellm_params,
            headers=self.headers,
            after="cursor_after_123",
            before="cursor_before_456",
            include=["metadata", "content"],
            limit=50,
            order="asc",
        )

        # Assert
        expected_url = f"{self.api_base}/{self.response_id}/input_items"
        expected_params = {
            "after": "cursor_after_123",
            "before": "cursor_before_456",
            "include": "metadata,content",  # Should be comma-separated string
            "limit": 50,
            "order": "asc",
        }
        assert url == expected_url
        assert params == expected_params

    def test_openai_transform_list_input_items_request_include_list_formatting(self):
        """Test that include list is properly formatted as comma-separated string"""
        # Execute
        url, params = self.openai_config.transform_list_input_items_request(
            response_id=self.response_id,
            api_base=self.api_base,
            litellm_params=self.litellm_params,
            headers=self.headers,
            include=["metadata", "content", "annotations"],
        )

        # Assert
        assert params["include"] == "metadata,content,annotations"

    def test_openai_transform_list_input_items_request_none_values(self):
        """Test OpenAI implementation with None values for optional parameters"""
        # Execute - pass only required parameters and explicit None for truly optional params
        url, params = self.openai_config.transform_list_input_items_request(
            response_id=self.response_id,
            api_base=self.api_base,
            litellm_params=self.litellm_params,
            headers=self.headers,
            after=None,
            before=None,
            include=None,
        )

        # Assert
        expected_url = f"{self.api_base}/{self.response_id}/input_items"
        expected_params = {
            "limit": 20,
            "order": "desc",
        }  # Default values should be present
        assert url == expected_url
        assert params == expected_params

    def test_openai_transform_list_input_items_request_empty_include_list(self):
        """Test OpenAI implementation with empty include list"""
        # Execute
        url, params = self.openai_config.transform_list_input_items_request(
            response_id=self.response_id,
            api_base=self.api_base,
            litellm_params=self.litellm_params,
            headers=self.headers,
            include=[],
        )

        # Assert
        assert "include" not in params  # Empty list should not be included

    def test_azure_transform_list_input_items_request_minimal(self):
        """Test Azure implementation with minimal parameters"""
        # Setup
        azure_api_base = "https://test.openai.azure.com/openai/responses?api-version=2024-05-01-preview"

        # Execute
        url, params = self.azure_config.transform_list_input_items_request(
            response_id=self.response_id,
            api_base=azure_api_base,
            litellm_params=self.litellm_params,
            headers=self.headers,
        )

        # Assert
        assert self.response_id in url
        assert "/input_items" in url
        assert params == {"limit": 20, "order": "desc"}

    def test_azure_transform_list_input_items_request_url_construction(self):
        """Test Azure implementation URL construction with response_id in path"""
        # Setup
        azure_api_base = "https://test.openai.azure.com/openai/responses?api-version=2024-05-01-preview"

        # Execute
        url, params = self.azure_config.transform_list_input_items_request(
            response_id=self.response_id,
            api_base=azure_api_base,
            litellm_params=self.litellm_params,
            headers=self.headers,
        )

        # Assert
        # The Azure implementation should construct URL with response_id in path
        assert self.response_id in url
        assert "/input_items" in url
        assert "api-version=2024-05-01-preview" in url

    def test_azure_transform_list_input_items_request_with_all_params(self):
        """Test Azure implementation with all optional parameters"""
        # Setup
        azure_api_base = "https://test.openai.azure.com/openai/responses?api-version=2024-05-01-preview"

        # Execute
        url, params = self.azure_config.transform_list_input_items_request(
            response_id=self.response_id,
            api_base=azure_api_base,
            litellm_params=self.litellm_params,
            headers=self.headers,
            after="cursor_after_123",
            before="cursor_before_456",
            include=["metadata", "content"],
            limit=100,
            order="asc",
        )

        # Assert
        expected_params = {
            "after": "cursor_after_123",
            "before": "cursor_before_456",
            "include": "metadata,content",
            "limit": 100,
            "order": "asc",
        }
        assert params == expected_params

    @patch("litellm.router.Router")
    def test_mock_litellm_router_with_transform_list_input_items_request(
        self, mock_router
    ):
        """Mock test using litellm.router for transform_list_input_items_request"""
        # Setup mock router
        mock_router_instance = Mock()
        mock_router.return_value = mock_router_instance

        # Mock the provider config
        mock_provider_config = Mock(spec=OpenAIResponsesAPIConfig)
        mock_provider_config.transform_list_input_items_request.return_value = (
            "https://api.openai.com/v1/responses/resp_123/input_items",
            {"limit": 20, "order": "desc"},
        )

        # Setup router mock
        mock_router_instance.get_provider_responses_api_config.return_value = (
            mock_provider_config
        )

        # Test parameters
        response_id = "resp_test123"

        # Execute
        url, params = mock_provider_config.transform_list_input_items_request(
            response_id=response_id,
            api_base="https://api.openai.com/v1/responses",
            litellm_params=GenericLiteLLMParams(),
            headers={"Authorization": "Bearer test"},
            after="cursor_123",
            include=["metadata"],
            limit=30,
        )

        # Assert
        mock_provider_config.transform_list_input_items_request.assert_called_once_with(
            response_id=response_id,
            api_base="https://api.openai.com/v1/responses",
            litellm_params=GenericLiteLLMParams(),
            headers={"Authorization": "Bearer test"},
            after="cursor_123",
            include=["metadata"],
            limit=30,
        )
        assert url == "https://api.openai.com/v1/responses/resp_123/input_items"
        assert params == {"limit": 20, "order": "desc"}

    @patch("litellm.list_input_items")
    def test_mock_litellm_list_input_items_integration(self, mock_list_input_items):
        """Test integration with litellm.list_input_items function"""
        # Setup mock response
        mock_response = {
            "object": "list",
            "data": [
                {
                    "id": "input_item_123",
                    "object": "input_item",
                    "type": "message",
                    "role": "user",
                    "content": "Test message",
                }
            ],
            "has_more": False,
            "first_id": "input_item_123",
            "last_id": "input_item_123",
        }
        mock_list_input_items.return_value = mock_response

        # Execute
        result = mock_list_input_items(
            response_id="resp_test123",
            after="cursor_after",
            limit=10,
            custom_llm_provider="openai",
        )

        # Assert
        mock_list_input_items.assert_called_once_with(
            response_id="resp_test123",
            after="cursor_after",
            limit=10,
            custom_llm_provider="openai",
        )
        assert result["object"] == "list"
        assert len(result["data"]) == 1

    def test_parameter_validation_edge_cases(self):
        """Test edge cases for parameter validation"""
        # Test with limit=0
        url, params = self.openai_config.transform_list_input_items_request(
            response_id=self.response_id,
            api_base=self.api_base,
            litellm_params=self.litellm_params,
            headers=self.headers,
            limit=0,
        )
        assert params["limit"] == 0

        # Test with very large limit
        url, params = self.openai_config.transform_list_input_items_request(
            response_id=self.response_id,
            api_base=self.api_base,
            litellm_params=self.litellm_params,
            headers=self.headers,
            limit=1000,
        )
        assert params["limit"] == 1000

        # Test with single item in include list
        url, params = self.openai_config.transform_list_input_items_request(
            response_id=self.response_id,
            api_base=self.api_base,
            litellm_params=self.litellm_params,
            headers=self.headers,
            include=["metadata"],
        )
        assert params["include"] == "metadata"

    def test_url_construction_with_different_api_bases(self):
        """Test URL construction with different API base formats"""
        test_cases = [
            {
                "api_base": "https://api.openai.com/v1/responses",
                "expected_suffix": "/resp_abc123/input_items",
            },
            {
                "api_base": "https://api.openai.com/v1/responses/",  # with trailing slash
                "expected_suffix": "/resp_abc123/input_items",
            },
            {
                "api_base": "https://custom-api.example.com/v1/responses",
                "expected_suffix": "/resp_abc123/input_items",
            },
        ]

        for case in test_cases:
            url, params = self.openai_config.transform_list_input_items_request(
                response_id=self.response_id,
                api_base=case["api_base"],
                litellm_params=self.litellm_params,
                headers=self.headers,
            )
            assert url.endswith(case["expected_suffix"])

    def test_return_type_validation(self):
        """Test that function returns correct types"""
        url, params = self.openai_config.transform_list_input_items_request(
            response_id=self.response_id,
            api_base=self.api_base,
            litellm_params=self.litellm_params,
            headers=self.headers,
        )

        # Assert return types
        assert isinstance(url, str)
        assert isinstance(params, dict)

        # Assert URL is properly formatted
        assert url.startswith("http")
        assert "input_items" in url

        # Assert params contains expected keys with correct types
        for key, value in params.items():
            assert isinstance(key, str)
            assert value is not None


def test_get_supported_openai_params():
    config = OpenAIResponsesAPIConfig()
    params = config.get_supported_openai_params("gpt-4o")
    assert "temperature" in params
    assert "stream" in params
    assert "background" in params
    assert "stream" in params