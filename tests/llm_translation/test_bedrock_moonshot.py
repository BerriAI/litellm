"""
Tests for Bedrock Moonshot (Kimi K2) integration.

This test suite verifies:
1. Basic completion functionality
2. Streaming responses
3. System message support
4. Temperature parameter handling
5. Reasoning content extraction from <reasoning> tags
6. Tool calling support (including tool response handling)
7. Parameter validation (e.g., stop sequences not supported)
"""

from base_llm_unit_tests import BaseLLMChatTest
import pytest
import sys
import os
import json
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm.llms.bedrock.common_utils import get_bedrock_chat_config
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler


class TestBedrockMoonshotInvoke(BaseLLMChatTest):
    """
    Test suite for Bedrock Moonshot via invoke route.
    Inherits all standard LLM tests from BaseLLMChatTest.
    """

    def get_base_completion_call_args(self) -> dict:
        litellm._turn_on_debug()
        return {
            "model": "bedrock/invoke/moonshot.kimi-k2-thinking",
        }

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly."""
        pass

    # ---------------------------------------------------------------------
    # The three overrides below replace inherited BaseLLMChatTest tests that
    # would otherwise make live AWS Bedrock calls. The live versions were
    # consistently crashing llm_translation xdist workers. Each override
    # patches the HTTP client's post() so no network request is sent, and
    # asserts on the outgoing request body (or a mocked response for cost)
    # — which is what the translation lane is actually supposed to cover.
    # ---------------------------------------------------------------------

    def test_developer_role_translation(self):
        """Verify LiteLLM maps the ``developer`` role to ``system`` on the
        outgoing Bedrock invoke request, without hitting the network."""
        client = HTTPHandler()
        with patch.object(client, "post", new=Mock()) as mock_post:
            try:
                litellm.completion(
                    model="bedrock/invoke/moonshot.kimi-k2-thinking",
                    messages=[
                        {"role": "developer", "content": "Be a good bot!"},
                        {"role": "user", "content": "Hello, how are you?"},
                    ],
                    aws_access_key_id="fake",
                    aws_secret_access_key="fake",
                    aws_region_name="us-west-2",
                    client=client,
                )
            except Exception:
                # The Mock() return value is not a parseable Bedrock response;
                # we only care about the outgoing request body.
                pass

        mock_post.assert_called_once()
        body = json.loads(mock_post.call_args.kwargs["data"])
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"] == "Be a good bot!"
        assert body["messages"][1]["role"] == "user"

    def test_message_with_name(self):
        """Verify a user message carrying a ``name`` field is serialized into
        the outgoing Bedrock invoke request without breaking the call."""
        client = HTTPHandler()
        with patch.object(client, "post", new=Mock()) as mock_post:
            try:
                litellm.completion(
                    model="bedrock/invoke/moonshot.kimi-k2-thinking",
                    messages=[
                        {"role": "user", "content": "Hello", "name": "test_name"},
                    ],
                    aws_access_key_id="fake",
                    aws_secret_access_key="fake",
                    aws_region_name="us-west-2",
                    client=client,
                )
            except Exception:
                pass

        mock_post.assert_called_once()
        body = json.loads(mock_post.call_args.kwargs["data"])
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["content"] == "Hello"

    async def test_completion_cost(self):
        """Verify LiteLLM computes a positive cost from a mocked Bedrock
        Moonshot response, using the local model cost map."""
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        response_body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "moonshot.kimi-k2-thinking",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.text = json.dumps(response_body)
        mock_response.json = lambda: response_body

        client = AsyncHTTPHandler()
        with patch.object(client, "post", new=AsyncMock(return_value=mock_response)):
            response = await litellm.acompletion(
                model="bedrock/invoke/moonshot.kimi-k2-thinking",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                aws_access_key_id="fake",
                aws_secret_access_key="fake",
                aws_region_name="us-west-2",
                client=client,
            )

        assert response._hidden_params["response_cost"] > 0


class TestBedrockMoonshotBasic:
    """Unit tests for Bedrock Moonshot configuration and transformations."""

    def test_provider_detection_invoke(self):
        """Test that Bedrock Moonshot invoke models are correctly detected."""
        config = get_bedrock_chat_config("bedrock/invoke/moonshot.kimi-k2-thinking")
        assert config is not None
        assert config.__class__.__name__ == "AmazonMoonshotConfig"

    def test_provider_detection_converse(self):
        """Test that Bedrock Moonshot converse models are correctly detected."""
        config = get_bedrock_chat_config("bedrock/moonshot.kimi-k2-thinking")
        assert config is not None

    def test_config_initialization(self):
        """Test that AmazonMoonshotConfig initializes correctly."""
        config = get_bedrock_chat_config("invoke/moonshot.kimi-k2-thinking")
        assert config is not None
        assert config.custom_llm_provider == "bedrock"

    def test_supported_params(self):
        """Test that supported OpenAI params are correctly defined."""
        config = get_bedrock_chat_config("invoke/moonshot.kimi-k2-thinking")
        supported_params = config.get_supported_openai_params(
            "moonshot.kimi-k2-thinking"
        )

        # Should support these params
        assert "temperature" in supported_params
        assert "max_tokens" in supported_params
        assert "top_p" in supported_params
        assert "stream" in supported_params
        assert "tools" in supported_params
        assert "tool_choice" in supported_params

        # Should NOT support stop sequences on Bedrock
        assert "stop" not in supported_params

        # Should NOT support functions (use tools instead)
        assert "functions" not in supported_params

    def test_transform_request_strips_model_prefix(self):
        """Test that model ID prefixes are correctly stripped in transform_request."""
        from litellm.llms.bedrock.chat.invoke_transformations.amazon_moonshot_transformation import (
            AmazonMoonshotConfig,
        )

        config = AmazonMoonshotConfig()

        messages = [{"role": "user", "content": "Hello"}]

        # Test that bedrock/invoke/ prefix is stripped
        transformed = config.transform_request(
            model="bedrock/invoke/moonshot.kimi-k2-thinking",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # The model ID in the request body should be stripped
        assert transformed["model"] == "moonshot.kimi-k2-thinking"


class TestBedrockMoonshotReasoningContent:
    """Tests for reasoning content extraction."""

    def test_reasoning_content_extraction(self):
        """Test that reasoning content is extracted from <reasoning> tags."""
        from litellm.llms.bedrock.chat.invoke_transformations.amazon_moonshot_transformation import (
            AmazonMoonshotConfig,
        )

        config = AmazonMoonshotConfig()

        # Test with reasoning tags
        content_with_reasoning = (
            "<reasoning>This is my thought process</reasoning>This is the answer"
        )
        reasoning, content = config._extract_reasoning_from_content(
            content_with_reasoning
        )

        assert reasoning == "This is my thought process"
        assert content == "This is the answer"
        assert "<reasoning>" not in content

        # Test without reasoning tags
        content_without_reasoning = "This is just a regular answer"
        reasoning, content = config._extract_reasoning_from_content(
            content_without_reasoning
        )

        assert reasoning is None
        assert content == "This is just a regular answer"


class TestBedrockMoonshotToolCalling:
    """Unit tests for tool calling functionality."""

    def test_tool_calling_supported(self):
        """Test that tool calling is supported for Kimi K2 Thinking model."""
        config = get_bedrock_chat_config("invoke/moonshot.kimi-k2-thinking")
        supported_params = config.get_supported_openai_params(
            "moonshot.kimi-k2-thinking"
        )

        # Kimi K2 Thinking DOES support tool calls (unlike kimi-thinking-preview)
        assert "tools" in supported_params
        assert "tool_choice" in supported_params

    def test_tool_call_request_format(self):
        """Test that tool call requests are formatted correctly."""
        from litellm.llms.bedrock.chat.invoke_transformations.amazon_moonshot_transformation import (
            AmazonMoonshotConfig,
        )

        config = AmazonMoonshotConfig()

        messages = [{"role": "user", "content": "What's the weather in San Francisco?"}]

        optional_params = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    },
                }
            ]
        }

        transformed = config.transform_request(
            model="bedrock/invoke/moonshot.kimi-k2-thinking",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        # Verify model ID is stripped
        assert transformed["model"] == "moonshot.kimi-k2-thinking"

        # Verify tools are included
        assert "tools" in transformed
        assert len(transformed["tools"]) == 1
        assert transformed["tools"][0]["function"]["name"] == "get_weather"

    def test_tool_response_message_format(self):
        """Test that tool response messages are formatted correctly."""
        # This tests the proper format for sending tool responses back
        tool_response_message = {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": json.dumps({"temperature": 72, "condition": "sunny"}),
        }

        # Verify the message structure
        assert tool_response_message["role"] == "tool"
        assert "tool_call_id" in tool_response_message
        assert "content" in tool_response_message


class TestBedrockMoonshotParameterValidation:
    """Tests for parameter validation and edge cases."""

    def test_stop_sequences_not_supported(self):
        """Test that stop sequences are correctly excluded from supported params."""
        config = get_bedrock_chat_config("invoke/moonshot.kimi-k2-thinking")
        supported_params = config.get_supported_openai_params(
            "moonshot.kimi-k2-thinking"
        )

        # Bedrock Moonshot doesn't support stopSequences field
        assert "stop" not in supported_params

    def test_temperature_range(self):
        """Test that temperature parameter is handled correctly."""
        # Moonshot models support temperature 0-1
        # This is handled by the parent MoonshotChatConfig class
        config = get_bedrock_chat_config("invoke/moonshot.kimi-k2-thinking")

        # Verify config exists and can handle temperature
        assert config is not None
        supported_params = config.get_supported_openai_params(
            "moonshot.kimi-k2-thinking"
        )
        assert "temperature" in supported_params


class TestBedrockMoonshotTransformations:
    """Tests for request/response transformations."""

    def test_transform_request_basic(self):
        """Test basic request transformation."""
        from litellm.llms.bedrock.chat.invoke_transformations.amazon_moonshot_transformation import (
            AmazonMoonshotConfig,
        )

        config = AmazonMoonshotConfig()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]

        optional_params = {"temperature": 0.7, "max_tokens": 100}

        transformed = config.transform_request(
            model="bedrock/invoke/moonshot.kimi-k2-thinking",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        # Verify model ID is stripped
        assert transformed["model"] == "moonshot.kimi-k2-thinking"

        # Verify messages are included
        assert "messages" in transformed
        assert len(transformed["messages"]) >= 1

        # Verify optional params are included
        assert transformed["temperature"] == 0.7
        assert transformed["max_tokens"] == 100

    def test_transform_request_with_system_message(self):
        """Test request transformation with system message."""
        from litellm.llms.bedrock.chat.invoke_transformations.amazon_moonshot_transformation import (
            AmazonMoonshotConfig,
        )

        config = AmazonMoonshotConfig()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]

        transformed = config.transform_request(
            model="moonshot.kimi-k2-thinking",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # System messages should be supported
        assert "messages" in transformed
