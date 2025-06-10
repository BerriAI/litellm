import json
import uuid
from datetime import datetime
from unittest.mock import Mock, patch

import httpx
import pytest

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
        """Mock Anthropic Claude response from Bedrock"""
        return {
            "content": [{"text": "Hello! I'm Claude, an AI assistant created by Anthropic."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 15},
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
        return logging_obj

    def test_anthropic_bedrock_response_transformation(
        self, mock_httpx_response, mock_anthropic_bedrock_response, mock_logging_obj
    ):
        """Test transformation of Anthropic Claude responses from Bedrock"""
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
        assert model_response.choices[0].finish_reason == "end_turn"
        
        # Verify usage
        assert model_response.usage.prompt_tokens == 10
        assert model_response.usage.completion_tokens == 15
        assert model_response.usage.total_tokens == 25

        # Verify kwargs are updated with cost and model info
        assert result["kwargs"]["model"] == model
        assert result["kwargs"]["custom_llm_provider"] == "bedrock"
        assert "response_cost" in result["kwargs"]

    def test_titan_bedrock_response_transformation(
        self, mock_httpx_response, mock_titan_bedrock_response, mock_logging_obj
    ):
        """Test transformation of Amazon Titan responses from Bedrock"""
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
        assert model_response.choices[0].finish_reason == "FINISH"
        
        # Verify usage
        assert model_response.usage.prompt_tokens == 8
        assert model_response.usage.completion_tokens == 12
        assert model_response.usage.total_tokens == 20

    def test_generic_bedrock_response_transformation(
        self, mock_httpx_response, mock_logging_obj
    ):
        """Test transformation of unknown/generic Bedrock model responses"""
        start_time = datetime.now()
        end_time = datetime.now()
        model = "unknown.model-v1"

        # Generic response format
        response_body = {
            "text": "This is a generic response from an unknown model.",
        }

        kwargs = {
            "model": model,
            "passthrough_logging_payload": {
                "url": "https://bedrock-runtime.us-east-1.amazonaws.com/model/unknown.model-v1/invoke",
                "request_body": {"modelId": model, "body": {"prompt": "Hello"}},
                "request_method": "POST",
            },
        }

        result = BedrockPassthroughLoggingHandler.bedrock_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=response_body,
            logging_obj=mock_logging_obj,
            url_route="/model/unknown.model-v1/invoke",
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
        assert model_response.choices[0].message.content == "This is a generic response from an unknown model."

    @patch("litellm.completion_cost")
    def test_cost_calculation(
        self, mock_completion_cost, mock_httpx_response, mock_anthropic_bedrock_response, mock_logging_obj
    ):
        """Test that cost calculation is called correctly"""
        mock_completion_cost.return_value = 0.00015
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

        kwargs = {"model": model}

        result = BedrockPassthroughLoggingHandler.bedrock_passthrough_handler(
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

        # Verify graceful error handling - should return None result
        assert result["result"] is None
        assert "kwargs" in result

    def test_route_detection(self):
        """Test Bedrock route detection logic"""
        # Valid Bedrock routes
        assert BedrockPassthroughLoggingHandler._should_log_request(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-sonnet-20240229-v1:0/invoke"
        )
        assert BedrockPassthroughLoggingHandler._should_log_request(
            "https://bedrock-agent-runtime.us-west-2.amazonaws.com/agents/agent-id/sessions/session-id/text"
        )

        # Invalid routes
        assert not BedrockPassthroughLoggingHandler._should_log_request(
            "https://api.anthropic.com/v1/messages"
        )
        assert not BedrockPassthroughLoggingHandler._should_log_request(
            "https://api.openai.com/v1/chat/completions"
        )

    def test_user_extraction_from_metadata(self):
        """Test user extraction from metadata"""
        metadata_with_user_id = {
            "user_api_key_user_id": "test_user_123",
            "other_field": "value"
        }
        
        metadata_with_user = {
            "user": "another_user_456"
        }
        
        empty_metadata = {}

        assert BedrockPassthroughLoggingHandler._get_user_from_metadata(metadata_with_user_id) == "test_user_123"
        assert BedrockPassthroughLoggingHandler._get_user_from_metadata(metadata_with_user) == "another_user_456"
        assert BedrockPassthroughLoggingHandler._get_user_from_metadata(empty_metadata) is None

    def test_passthrough_logging_payload_update(
        self, mock_httpx_response, mock_anthropic_bedrock_response, mock_logging_obj
    ):
        """Test that passthrough logging payload is properly updated"""
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
            "metadata": {"user_api_key_user_id": "test_user"},
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

        # Verify passthrough logging payload was updated
        updated_payload = result["kwargs"]["passthrough_logging_payload"]
        assert updated_payload["user"] == "test_user"
        assert updated_payload["model"] == model
        assert updated_payload["custom_llm_provider"] == "bedrock"
        assert "response_cost" in updated_payload