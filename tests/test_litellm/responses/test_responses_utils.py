import base64
import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.responses.utils import ResponseAPILoggingUtils, ResponsesAPIRequestUtils
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams
from litellm.types.utils import Usage


class TestResponsesAPIRequestUtils:
    def test_get_optional_params_responses_api(self):
        """Test that optional parameters are correctly processed for responses API"""
        # Setup
        model = "gpt-4o"
        config = OpenAIResponsesAPIConfig()
        optional_params = ResponsesAPIOptionalRequestParams(
            {
                "temperature": 0.7,
                "max_output_tokens": 100,
                "prompt": {"id": "pmpt_123"},
            }
        )

        # Execute
        result = ResponsesAPIRequestUtils.get_optional_params_responses_api(
            model=model,
            responses_api_provider_config=config,
            response_api_optional_params=optional_params,
        )

        # Assert
        assert result == optional_params
        assert "temperature" in result
        assert result["temperature"] == 0.7
        assert "max_output_tokens" in result
        assert result["max_output_tokens"] == 100
        assert "prompt" in result
        assert result["prompt"] == {"id": "pmpt_123"}

    def test_get_optional_params_responses_api_unsupported_param(self):
        """Test that unsupported parameters raise an error"""
        # Setup
        model = "gpt-4o"
        config = OpenAIResponsesAPIConfig()
        optional_params = ResponsesAPIOptionalRequestParams(
            {"temperature": 0.7, "unsupported_param": "value"}
        )

        # Execute and Assert
        with pytest.raises(litellm.UnsupportedParamsError) as excinfo:
            ResponsesAPIRequestUtils.get_optional_params_responses_api(
                model=model,
                responses_api_provider_config=config,
                response_api_optional_params=optional_params,
            )

        assert "unsupported_param" in str(excinfo.value)
        assert model in str(excinfo.value)

    def test_get_requested_response_api_optional_param(self):
        """Test filtering parameters to only include those in ResponsesAPIOptionalRequestParams"""
        # Setup
        params = {
            "temperature": 0.7,
            "max_output_tokens": 100,
            "prompt": {"id": "pmpt_456"},
            "invalid_param": "value",
            "model": "gpt-4o",  # This is not in ResponsesAPIOptionalRequestParams
        }

        # Execute
        result = ResponsesAPIRequestUtils.get_requested_response_api_optional_param(
            params
        )

        # Assert
        assert "temperature" in result
        assert "max_output_tokens" in result
        assert "invalid_param" not in result
        assert "model" not in result
        assert result["temperature"] == 0.7
        assert result["max_output_tokens"] == 100
        assert result["prompt"] == {"id": "pmpt_456"}

    def test_decode_previous_response_id_to_original_previous_response_id(self):
        """Test decoding a LiteLLM encoded previous_response_id to the original previous_response_id"""
        # Setup
        test_provider = "openai"
        test_model_id = "gpt-4o"
        original_response_id = "resp_abc123"

        # Use the helper method to build an encoded response ID
        encoded_id = ResponsesAPIRequestUtils._build_responses_api_response_id(
            custom_llm_provider=test_provider,
            model_id=test_model_id,
            response_id=original_response_id,
        )

        # Execute
        result = ResponsesAPIRequestUtils.decode_previous_response_id_to_original_previous_response_id(
            encoded_id
        )

        # Assert
        assert result == original_response_id

        # Test with a non-encoded ID
        plain_id = "resp_xyz789"
        result_plain = ResponsesAPIRequestUtils.decode_previous_response_id_to_original_previous_response_id(
            plain_id
        )
        assert result_plain == plain_id

    def test_update_responses_api_response_id_with_model_id_handles_dict(self):
        """Ensure _update_responses_api_response_id_with_model_id works with dict input"""
        responses_api_response = {"id": "resp_abc123"}
        litellm_metadata = {"model_info": {"id": "gpt-4o"}}
        updated = ResponsesAPIRequestUtils._update_responses_api_response_id_with_model_id(
            responses_api_response=responses_api_response,
            custom_llm_provider="openai",
            litellm_metadata=litellm_metadata,
        )
        assert updated["id"] != "resp_abc123"
        decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(updated["id"])
        assert decoded.get("response_id") == "resp_abc123"
        assert decoded.get("model_id") == "gpt-4o"
        assert decoded.get("custom_llm_provider") == "openai"


class TestResponseAPILoggingUtils:
    def test_is_response_api_usage_true(self):
        """Test identification of Response API usage format"""
        # Setup
        usage = {"input_tokens": 10, "output_tokens": 20}

        # Execute
        result = ResponseAPILoggingUtils._is_response_api_usage(usage)

        # Assert
        assert result is True

    def test_is_response_api_usage_false(self):
        """Test identification of non-Response API usage format"""
        # Setup
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

        # Execute
        result = ResponseAPILoggingUtils._is_response_api_usage(usage)

        # Assert
        assert result is False

    def test_transform_response_api_usage_to_chat_usage(self):
        """Test transformation from Response API usage to Chat usage format"""
        # Setup
        usage = {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens_details": {"reasoning_tokens": 5},
        }

        # Execute
        result = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(
            usage
        )

        # Assert
        assert isinstance(result, Usage)
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 20
        assert result.total_tokens == 30
        assert result.prompt_tokens_details and result.prompt_tokens_details.cached_tokens == 2

    def test_transform_response_api_usage_with_none_values(self):
        """Test transformation handles None values properly"""
        # Setup
        usage = {
            "input_tokens": 0,  # Changed from None to 0
            "output_tokens": 20,
            "total_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 5},
        }

        # Execute
        result = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(
            usage
        )

        # Assert
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 20
        assert result.total_tokens == 20
