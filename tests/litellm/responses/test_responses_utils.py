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


class TestResponsesAPIRequestUtils:
    def test_get_optional_params_responses_api(self):
        """Test that optional parameters are correctly processed for responses API"""
        # Setup
        model = "gpt-4o"
        config = OpenAIResponsesAPIConfig()
        optional_params = ResponsesAPIOptionalRequestParams(
            {"temperature": 0.7, "max_output_tokens": 100}
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
