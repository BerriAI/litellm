import json
import uuid
from datetime import datetime
from unittest.mock import Mock, patch

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.bedrock_passthrough_logging_handler import (
    BedrockPassthroughLoggingHandler,
)
from litellm.types.utils import ModelResponse, Usage


class TestBedrockPassthroughLoggingHandler:
    @pytest.fixture
    def mock_httpx_response(self):
        """Mock httpx.Response object"""
        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.headers = {"content-type": "application/json"}
        return response

    @pytest.fixture
    def mock_anthropic_bedrock_response(self):
        """Mock Anthropic Claude response from Bedrock (actual format)"""
        return {
            "metrics": {"latencyMs": 1145},
            "output": {
                "message": {
                    "content": [{"text": "Hello! I'm Claude, an AI assistant created by Anthropic."}],
                    "role": "assistant"
                }
            },
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 15,
                "totalTokens": 25
            }
        }

    @pytest.fixture
    def mock_titan_bedrock_response(self):
        """Mock Amazon Titan response from Bedrock"""
        return {
            "inputTextTokenCount": 8,
            "results": [
                {
                    "outputText": "This is a response from Amazon Titan.",
                    "tokenCount": 12,
                    "completionReason": "FINISH",
                }
            ],
        }

    @pytest.fixture
    def mock_logging_obj(self):
        """Mock LiteLLM logging object"""
        logging_obj = Mock(spec=LiteLLMLoggingObj)
        logging_obj.model_call_details = {}
        logging_obj.litellm_call_id = "test-call-id-123"
        return logging_obj

    @patch("litellm.completion_cost")
    def test_anthropic_bedrock_response_transformation(
        self, mock_completion_cost, mock_httpx_response, mock_anthropic_bedrock_response, mock_logging_obj
    ):
        """Test transformation of Anthropic Claude responses from Bedrock"""
        mock_completion_cost.return_value = 0.00025
        mock_httpx_response.json.return_value = mock_anthropic_bedrock_response
        mock_httpx_response.status_code = 200
        mock_httpx_response.text = json.dumps(mock_anthropic_bedrock_response)

        start_time = datetime.now()
        end_time = datetime.now()
        model = "anthropic.claude-3-sonnet-20240229-v1:0"

        kwargs = {
            "model": model,
            "passthrough_logging_payload": {
                "url": "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-sonnet-20240229-v1:0/invoke",
                "request_body": {"modelId": model, "body": {"messages": [{"role": "user", "content": "Hello"}]}},
                "request_method": "POST",
            },
        }

        result = BedrockPassthroughLoggingHandler.bedrock_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=mock_anthropic_bedrock_response,
            logging_obj=mock_logging_obj,
            url_route="/model/anthropic.claude-3-sonnet-20240229-v1:0/invoke",
            result="",
            start_time=start_time,
            end_time=end_time,
            cache_hit=False,
            **kwargs,
        )

        # Verify result structure
        assert "result" in result
        assert "kwargs" in result
        assert isinstance(result["result"], ModelResponse)

        # Verify model response content
        model_response = result["result"]
        assert model_response.model == model
        assert len(model_response.choices) == 1
        assert model_response.choices[0].message.content == "Hello! I'm Claude, an AI assistant created by Anthropic."
        assert model_response.choices[0].message.role == "assistant"
        assert model_response.choices[0].finish_reason == "stop"  # end_turn maps to stop
        assert model_response.id == "test-call-id-123"

        # Verify usage
        assert model_response.usage.prompt_tokens == 10
        assert model_response.usage.completion_tokens == 15
        assert model_response.usage.total_tokens == 25

        # Verify kwargs are updated with cost and model info
        assert result["kwargs"]["model"] == model
        assert result["kwargs"]["custom_llm_provider"] == "bedrock"
        assert result["kwargs"]["response_cost"] == 0.00025
        # Verify logging object was updated
        assert mock_logging_obj.model_call_details["model"] == model
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "bedrock"

    @patch("litellm.completion_cost")
    def test_titan_bedrock_response_transformation(
        self, mock_completion_cost, mock_httpx_response, mock_titan_bedrock_response, mock_logging_obj
    ):
        """Test transformation of Amazon Titan responses from Bedrock"""
        mock_completion_cost.return_value = 0.00012
        mock_httpx_response.json.return_value = mock_titan_bedrock_response
        mock_httpx_response.status_code = 200
        mock_httpx_response.text = json.dumps(mock_titan_bedrock_response)

        start_time = datetime.now()
        end_time = datetime.now()
        model = "amazon.titan-text-lite-v1"

        kwargs = {
            "model": model,
            "passthrough_logging_payload": {
                "url": "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.titan-text-lite-v1/invoke",
                "request_body": {"modelId": model, "body": {"inputText": "Hello"}},
                "request_method": "POST",
            },
        }

        result = BedrockPassthroughLoggingHandler.bedrock_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=mock_titan_bedrock_response,
            logging_obj=mock_logging_obj,
            url_route="/model/amazon.titan-text-lite-v1/invoke",
            result="",
            start_time=start_time,
            end_time=end_time,
            cache_hit=False,
            **kwargs,
        )

        # Verify result structure
        assert isinstance(result["result"], ModelResponse)

        # Verify model response content
        model_response = result["result"]
        assert model_response.model == model
        assert len(model_response.choices) == 1
        assert model_response.choices[0].message.content == "This is a response from Amazon Titan."
        assert model_response.choices[0].finish_reason == "stop"  # FINISH maps to stop

        # Verify usage information is extracted properly by LiteLLM transformations
        # Note: LiteLLM may estimate tokens differently than the raw response
        assert model_response.usage.prompt_tokens > 0
        assert model_response.usage.completion_tokens > 0
        assert model_response.usage.total_tokens > 0

    @patch("litellm.completion_cost")
    def test_generic_bedrock_response_transformation(
        self, mock_completion_cost, mock_httpx_response, mock_logging_obj
    ):
        """Test transformation of Amazon Titan (generic) response format"""
        mock_completion_cost.return_value = 0.00008

        start_time = datetime.now()
        end_time = datetime.now()
        model = "amazon.titan-text-express-v1"

        # Use Titan response format for generic test since unknown models default to Titan parsing
        response_body = {
            "inputTextTokenCount": 5,
            "results": [
                {
                    "outputText": "This is a generic response from Titan.",
                    "tokenCount": 8,
                    "completionReason": "FINISH",
                }
            ],
        }

        mock_httpx_response.json.return_value = response_body
        mock_httpx_response.status_code = 200
        mock_httpx_response.text = json.dumps(response_body)

        kwargs = {
            "model": model,
            "passthrough_logging_payload": {
                "url": "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.titan-text-express-v1/invoke",
                "request_body": {"modelId": model, "body": {"prompt": "Hello"}},
                "request_method": "POST",
            },
        }

        result = BedrockPassthroughLoggingHandler.bedrock_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=response_body,
            logging_obj=mock_logging_obj,
            url_route="/model/amazon.titan-text-express-v1/invoke",
            result="",
            start_time=start_time,
            end_time=end_time,
            cache_hit=False,
            **kwargs,
        )

        # Verify result structure
        assert isinstance(result["result"], ModelResponse)

        # Verify model response content
        model_response = result["result"]
        assert model_response.model == model
        assert len(model_response.choices) == 1
        assert model_response.choices[0].message.content == "This is a generic response from Titan."

    @patch("litellm.completion_cost")
    def test_cost_calculation(
        self, mock_completion_cost, mock_httpx_response, mock_anthropic_bedrock_response, mock_logging_obj
    ):
        """Test that cost calculation is called correctly"""
        mock_completion_cost.return_value = 0.00015
        mock_httpx_response.json.return_value = mock_anthropic_bedrock_response
        mock_httpx_response.status_code = 200
        mock_httpx_response.text = json.dumps(mock_anthropic_bedrock_response)

        start_time = datetime.now()
        end_time = datetime.now()
        model = "anthropic.claude-3-sonnet-20240229-v1:0"

        kwargs = {"model": model}

        result = BedrockPassthroughLoggingHandler.bedrock_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=mock_anthropic_bedrock_response,
            logging_obj=mock_logging_obj,
            url_route="/model/anthropic.claude-3-sonnet-20240229-v1:0/invoke",
            result="",
            start_time=start_time,
            end_time=end_time,
            cache_hit=False,
            **kwargs,
        )

        # Verify cost calculation was called
        mock_completion_cost.assert_called_once()
        assert result["kwargs"]["response_cost"] == 0.00015

    def test_error_handling(
        self, mock_httpx_response, mock_logging_obj
    ):
        """Test error handling when response transformation fails"""
        start_time = datetime.now()
        end_time = datetime.now()
        model = "anthropic.claude-3-sonnet-20240229-v1:0"

        # Malformed response that will cause transformation to fail
        malformed_response = {"invalid": "response"}
        mock_httpx_response.json.return_value = malformed_response
        mock_httpx_response.status_code = 500
        mock_httpx_response.text = "Internal Server Error"

        kwargs = {"model": model}

        # Should raise an exception due to malformed response
        with pytest.raises(Exception):
            BedrockPassthroughLoggingHandler.bedrock_passthrough_handler(
                httpx_response=mock_httpx_response,
                response_body=malformed_response,
                logging_obj=mock_logging_obj,
                url_route="/model/anthropic.claude-3-sonnet-20240229-v1:0/invoke",
                result="original_result",
                start_time=start_time,
                end_time=end_time,
                cache_hit=False,
                **kwargs,
            )

    def test_user_extraction_from_metadata(self):
        """Test user extraction from metadata"""
        metadata_with_user = {
            "user": "test_user_456"
        }

        empty_metadata = {}

        assert BedrockPassthroughLoggingHandler._get_user_from_metadata(metadata_with_user) == "test_user_456"
        assert BedrockPassthroughLoggingHandler._get_user_from_metadata(empty_metadata) is None

    @patch("litellm.completion_cost")
    def test_user_metadata_handling(
        self, mock_completion_cost, mock_httpx_response, mock_anthropic_bedrock_response, mock_logging_obj
    ):
        """Test that user metadata is properly handled"""
        mock_completion_cost.return_value = 0.00025
        mock_httpx_response.json.return_value = mock_anthropic_bedrock_response
        mock_httpx_response.status_code = 200
        mock_httpx_response.text = json.dumps(mock_anthropic_bedrock_response)

        start_time = datetime.now()
        end_time = datetime.now()
        model = "anthropic.claude-3-sonnet-20240229-v1:0"

        passthrough_logging_payload = {
            "url": "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-sonnet-20240229-v1:0/invoke",
            "request_body": {"modelId": model},
            "request_method": "POST",
        }

        kwargs = {
            "model": model,
            "passthrough_logging_payload": passthrough_logging_payload,
            "metadata": {"user": "test_user"},
        }

        result = BedrockPassthroughLoggingHandler.bedrock_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=mock_anthropic_bedrock_response,
            logging_obj=mock_logging_obj,
            url_route="/model/anthropic.claude-3-sonnet-20240229-v1:0/invoke",
            result="",
            start_time=start_time,
            end_time=end_time,
            cache_hit=False,
            **kwargs,
        )

        # Verify user was properly set in litellm_params
        litellm_params = result["kwargs"]["litellm_params"]
        assert litellm_params["proxy_server_request"]["body"]["user"] == "test_user"

        # Verify other kwargs
        assert result["kwargs"]["model"] == model
        assert result["kwargs"]["custom_llm_provider"] == "bedrock"
        assert result["kwargs"]["response_cost"] == 0.00025
