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
from typing import Optional
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
    # The overrides below replace inherited BaseLLMChatTest tests that would
    # otherwise make live AWS Bedrock calls. The live versions were
    # consistently crashing llm_translation xdist workers. Each override
    # patches the HTTP client's post() so no network request is sent, and
    # asserts on the outgoing request body (and, where needed, parses a
    # canned response) — which is what the translation lane is actually
    # supposed to cover.
    # ---------------------------------------------------------------------

    @staticmethod
    def _make_moonshot_response(content: str = "Hi!") -> Mock:
        """Build a Mock httpx.Response that AmazonMoonshotConfig.transform_response
        (which delegates to MoonshotChatConfig → OpenAI) can parse."""
        body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "moonshot.kimi-k2-thinking",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.text = json.dumps(body)
        mock_resp.json = lambda: body
        return mock_resp

    def _invoke_with_mocked_post(
        self,
        *,
        messages: list,
        extra_kwargs: Optional[dict] = None,
        response_content: str = "Hi!",
    ) -> "tuple[Mock, object]":
        """Run a sync litellm.completion() with HTTPHandler.post patched to
        return a canned moonshot response. Returns (mock_post, response)."""
        client = HTTPHandler()
        mock_resp = self._make_moonshot_response(content=response_content)
        with patch.object(
            client, "post", new=Mock(return_value=mock_resp)
        ) as mock_post:
            response = litellm.completion(
                model="bedrock/invoke/moonshot.kimi-k2-thinking",
                messages=messages,
                aws_access_key_id="fake",
                aws_secret_access_key="fake",
                aws_region_name="us-west-2",
                client=client,
                **(extra_kwargs or {}),
            )
        return mock_post, response

    def test_developer_role_translation(self):
        """Verify LiteLLM maps the ``developer`` role to ``system`` on the
        outgoing Bedrock invoke request, without hitting the network."""
        mock_post, response = self._invoke_with_mocked_post(
            messages=[
                {"role": "developer", "content": "Be a good bot!"},
                {"role": "user", "content": "Hello, how are you?"},
            ],
        )
        mock_post.assert_called_once()
        body = json.loads(mock_post.call_args.kwargs["data"])
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"] == "Be a good bot!"
        assert body["messages"][1]["role"] == "user"
        assert response.choices[0].message.content is not None

    def test_message_with_name(self):
        """Verify a user message carrying a ``name`` field is serialized into
        the outgoing Bedrock invoke request without breaking the call."""
        mock_post, response = self._invoke_with_mocked_post(
            messages=[{"role": "user", "content": "Hello", "name": "test_name"}],
        )
        mock_post.assert_called_once()
        body = json.loads(mock_post.call_args.kwargs["data"])
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["content"] == "Hello"
        assert response is not None

    def test_content_list_handling(self):
        """Verify the inherited content-list-handling test passes against a
        mocked moonshot response (no network)."""
        mock_post, response = self._invoke_with_mocked_post(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello, how are you?"}],
                }
            ],
        )
        mock_post.assert_called_once()
        assert response.choices[0].message.content is not None

    def test_pydantic_model_input(self):
        """Verify a completion call with a pydantic ``Message`` as input does
        not raise and produces a parseable response."""
        from litellm import Message

        mock_post, response = self._invoke_with_mocked_post(
            messages=[Message(content="Hello, how are you?", role="user")],
        )
        mock_post.assert_called_once()
        assert response is not None

    @pytest.mark.parametrize("response_format", [{"type": "text"}])
    def test_response_format_type_text_with_tool_calls_no_tool_choice(
        self, response_format
    ):
        """Verify response_format + tools + drop_params sends a valid request
        and produces a response object."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        mock_post, response = self._invoke_with_mocked_post(
            messages=[
                {"role": "user", "content": "What's the weather like in Boston today?"}
            ],
            extra_kwargs={
                "response_format": response_format,
                "tools": tools,
                "drop_params": True,
            },
        )
        mock_post.assert_called_once()
        body = json.loads(mock_post.call_args.kwargs["data"])
        assert "tools" in body
        assert body["tools"][0]["function"]["name"] == "get_current_weather"
        assert response is not None

    def test_streaming(self):
        """Verify stream=True routes to the invoke-with-response-stream
        endpoint with the messages body. Iteration of the stream itself is
        not exercised here — moonshot streaming delegates to the OpenAI
        parser and is covered by the OpenAI test suite.

        Note: bedrock invoke streaming cannot be intercepted by patching
        the caller-supplied client, because ``CustomStreamWrapper.fetch_sync_stream``
        at streaming_handler.py invokes the stored ``make_call`` partial with
        ``client=litellm.module_level_client``, which overrides any client the
        caller passed. Patch ``make_sync_call`` at its import site in
        ``base_invoke_transformation`` so we observe the exact kwargs the
        partial was built with at stream-wrapper construction time.
        """
        from litellm.utils import CustomStreamWrapper

        captured: dict = {}

        def fake_make_sync_call(**kwargs):
            captured.update(kwargs)
            # Return an empty iterator so the stream wrapper's iteration
            # doesn't try to parse real bytes.
            return iter([])

        with patch(
            "litellm.llms.bedrock.chat.invoke_transformations."
            "base_invoke_transformation.make_sync_call",
            new=fake_make_sync_call,
        ):
            response = litellm.completion(
                model="bedrock/invoke/moonshot.kimi-k2-thinking",
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Hello, how are you?"}],
                    }
                ],
                stream=True,
                aws_access_key_id="fake",
                aws_secret_access_key="fake",
                aws_region_name="us-west-2",
            )
            assert isinstance(response, CustomStreamWrapper)
            # Trigger fetch_sync_stream → make_call(...) → fake_make_sync_call.
            try:
                next(iter(response))
            except StopIteration:
                pass

        assert captured, "make_sync_call was never invoked"
        assert captured["api_base"].endswith("/invoke-with-response-stream")
        body = json.loads(captured["data"])
        # Bedrock invoke does not put stream=true in the body (the URL
        # carries the streaming flag); verify the user message is present.
        assert body["messages"][0]["role"] == "user"

    async def test_completion_cost(self):
        """Verify LiteLLM computes a positive cost from a mocked Bedrock
        Moonshot response, using the local model cost map."""
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        mock_response = self._make_moonshot_response()
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
