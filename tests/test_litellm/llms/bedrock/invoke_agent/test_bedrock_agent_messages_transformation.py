"""
Tests for AmazonInvokeAgentMessagesConfig - Anthropic /v1/messages API
interface for Bedrock managed agents (InvokeAgent API).
"""

import base64
import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.bedrock.messages.invoke_agent_transformation import (
    AmazonInvokeAgentMessagesConfig,
)
from litellm.types.router import GenericLiteLLMParams


class TestAmazonInvokeAgentMessagesConfig:
    """Test suite for AmazonInvokeAgentMessagesConfig"""

    @pytest.fixture
    def config(self):
        """Create a test instance"""
        return AmazonInvokeAgentMessagesConfig()

    @pytest.fixture
    def sample_messages(self):
        """Sample Anthropic messages format"""
        return [
            {"role": "user", "content": "Hello, how can you help me?"},
            {"role": "assistant", "content": "I can help with various tasks."},
            {"role": "user", "content": "What is the weather like?"},
        ]

    @pytest.fixture
    def sample_messages_with_content_blocks(self):
        """Sample messages with content block lists"""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First part."},
                    {"type": "text", "text": "Second part."},
                ],
            },
        ]

    @pytest.fixture
    def sample_events(self):
        """Sample parsed AWS events"""
        return [
            {
                "headers": {"event_type": "chunk"},
                "payload": {
                    "bytes": base64.b64encode("Hello ".encode("utf-8")).decode("utf-8")
                },
            },
            {
                "headers": {"event_type": "chunk"},
                "payload": {
                    "bytes": base64.b64encode("world!".encode("utf-8")).decode("utf-8")
                },
            },
            {
                "headers": {"event_type": "trace"},
                "payload": {
                    "trace": {
                        "preProcessingTrace": {
                            "modelInvocationOutput": {
                                "metadata": {
                                    "usage": {"inputTokens": 10, "outputTokens": 20}
                                }
                            }
                        },
                        "orchestrationTrace": {
                            "modelInvocationInput": {
                                "foundationModel": "anthropic.claude-v2"
                            }
                        },
                    }
                },
            },
        ]

    def test_should_return_supported_params_empty(self, config):
        """should return empty list for supported params (agents have no Anthropic-compatible params)"""
        result = config.get_supported_anthropic_messages_params(
            model="agent/TEST123/ALIAS456"
        )
        assert result == []

    def test_should_transform_request_from_anthropic_messages(
        self, config, sample_messages
    ):
        """should transform Anthropic messages format into InvokeAgent request"""
        result = config.transform_anthropic_messages_request(
            model="agent/TEST123/ALIAS456",
            messages=sample_messages,
            anthropic_messages_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert result["inputText"] == "What is the weather like?"
        assert result["enableTrace"] is True

    def test_should_extract_last_user_message_with_content_blocks(
        self, config, sample_messages_with_content_blocks
    ):
        """should handle messages with content block lists"""
        result = config.transform_anthropic_messages_request(
            model="agent/TEST123/ALIAS456",
            messages=sample_messages_with_content_blocks,
            anthropic_messages_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert result["inputText"] == "First part.\nSecond part."

    def test_should_fallback_to_last_message_when_no_user_message(self, config):
        """should fallback to last message content when no user message found"""
        messages = [
            {"role": "assistant", "content": "I'll help you."},
        ]

        result = config.transform_anthropic_messages_request(
            model="agent/TEST123/ALIAS456",
            messages=messages,
            anthropic_messages_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert result["inputText"] == "I'll help you."

    def test_should_transform_response_to_anthropic_format(self, config, sample_events):
        """should transform InvokeAgent event stream response to Anthropic Messages format"""
        raw_content = b"mock_event_stream_data"
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.content = raw_content
        mock_response.status_code = 200

        with patch.object(
            config._invoke_agent_config,
            "_parse_aws_event_stream",
            return_value=sample_events,
        ):
            result = config.transform_anthropic_messages_response(
                model="agent/TEST123/ALIAS456",
                raw_response=mock_response,
                logging_obj=MagicMock(),
            )

        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["stop_reason"] == "end_turn"
        assert result["model"] == "anthropic.claude-v2"

        content = result["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Hello world!"

        usage = result["usage"]
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 20

    def test_should_use_model_name_when_trace_has_no_model(self, config):
        """should use the original model name when trace events have no foundationModel"""
        events = [
            {
                "headers": {"event_type": "chunk"},
                "payload": {
                    "bytes": base64.b64encode("test".encode("utf-8")).decode("utf-8")
                },
            },
        ]

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.content = b"mock_data"
        mock_response.status_code = 200

        with patch.object(
            config._invoke_agent_config,
            "_parse_aws_event_stream",
            return_value=events,
        ):
            result = config.transform_anthropic_messages_response(
                model="agent/TEST123/ALIAS456",
                raw_response=mock_response,
                logging_obj=MagicMock(),
            )

        assert result["model"] == "agent/TEST123/ALIAS456"

    def test_should_raise_on_parse_error(self, config):
        """should raise BedrockError when event stream parsing fails"""
        from litellm.llms.bedrock.common_utils import BedrockError

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.content = b"invalid_data"
        mock_response.status_code = 200

        with patch.object(
            config._invoke_agent_config,
            "_parse_aws_event_stream",
            side_effect=Exception("parse error"),
        ):
            with pytest.raises(BedrockError, match="Error processing response"):
                config.transform_anthropic_messages_response(
                    model="agent/TEST123/ALIAS456",
                    raw_response=mock_response,
                    logging_obj=MagicMock(),
                )

    def test_should_generate_unique_message_id(self, config, sample_events):
        """should generate a unique msg_ prefixed ID for each response"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.content = b"mock_data"
        mock_response.status_code = 200

        with patch.object(
            config._invoke_agent_config,
            "_parse_aws_event_stream",
            return_value=sample_events,
        ):
            result1 = config.transform_anthropic_messages_response(
                model="agent/TEST123/ALIAS456",
                raw_response=mock_response,
                logging_obj=MagicMock(),
            )
            result2 = config.transform_anthropic_messages_response(
                model="agent/TEST123/ALIAS456",
                raw_response=mock_response,
                logging_obj=MagicMock(),
            )

        assert result1["id"].startswith("msg_")
        assert result2["id"].startswith("msg_")
        assert result1["id"] != result2["id"]

    def test_should_handle_empty_response(self, config):
        """should handle empty event stream gracefully"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.content = b""
        mock_response.status_code = 200

        with patch.object(
            config._invoke_agent_config,
            "_parse_aws_event_stream",
            return_value=[],
        ):
            result = config.transform_anthropic_messages_response(
                model="agent/TEST123/ALIAS456",
                raw_response=mock_response,
                logging_obj=MagicMock(),
            )

        assert result["content"][0]["text"] == ""
        assert result["usage"]["input_tokens"] == 0
        assert result["usage"]["output_tokens"] == 0

    def test_should_validate_environment_returns_headers_and_api_base(self, config):
        """should return headers and api_base unchanged"""
        headers = {"Authorization": "Bearer test"}
        api_base = "https://example.com"

        result_headers, result_api_base = (
            config.validate_anthropic_messages_environment(
                headers=headers,
                model="agent/TEST123/ALIAS456",
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base=api_base,
            )
        )

        assert result_headers == headers
        assert result_api_base == api_base


class TestBedrockAgentMessagesRouting:
    """Test that the bedrock agent route correctly resolves to the messages config."""

    def test_should_return_messages_config_for_agent_route(self):
        """should return AmazonInvokeAgentMessagesConfig for agent/ models"""
        from litellm.llms.bedrock.common_utils import BedrockModelInfo

        config = BedrockModelInfo.get_bedrock_provider_config_for_messages_api(
            model="agent/AGENT123/ALIAS456"
        )

        assert config is not None
        assert isinstance(config, AmazonInvokeAgentMessagesConfig)

    def test_should_return_correct_route_type(self):
        """should return 'agent' route for agent/ model prefix"""
        from litellm.llms.bedrock.common_utils import BedrockModelInfo

        route = BedrockModelInfo.get_bedrock_route("agent/AGENT123/ALIAS456")
        assert route == "agent"

    def test_should_not_return_agent_config_for_non_agent_models(self):
        """should not return agent config for regular claude models"""
        from litellm.llms.bedrock.common_utils import BedrockModelInfo

        config = BedrockModelInfo.get_bedrock_provider_config_for_messages_api(
            model="anthropic.claude-3-sonnet-20240229-v1:0"
        )

        assert config is None or not isinstance(config, AmazonInvokeAgentMessagesConfig)
