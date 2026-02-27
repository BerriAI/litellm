"""
Tests for OpenAI GPT transformation (litellm/llms/openai/chat/gpt_transformation.py)
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.openai.chat.gpt_transformation import (
    OpenAIChatCompletionStreamingHandler,
    OpenAIGPTConfig,
)


class TestOpenAIGPTConfig:
    """Tests for OpenAIGPTConfig class"""

    def setup_method(self):
        self.config = OpenAIGPTConfig()

    def test_user_param_supported_for_regular_models(self):
        """Test that 'user' param is in supported params for regular OpenAI models."""
        supported_params = self.config.get_supported_openai_params("gpt-4o")
        assert "user" in supported_params

        supported_params = self.config.get_supported_openai_params("gpt-4.1-mini")
        assert "user" in supported_params

    def test_user_param_supported_for_responses_api_models(self):
        """Test that 'user' param is in supported params for responses API models.

        Regression test for: https://github.com/BerriAI/litellm/issues/17633
        When using model="openai/responses/gpt-4.1", the 'user' parameter should
        be included in supported params so it reaches OpenAI and SpendLogs.
        """
        # responses/gpt-4.1-mini should support 'user' just like gpt-4.1-mini
        supported_params = self.config.get_supported_openai_params("responses/gpt-4.1-mini")
        assert "user" in supported_params

        supported_params = self.config.get_supported_openai_params("responses/gpt-4o")
        assert "user" in supported_params

        supported_params = self.config.get_supported_openai_params("responses/gpt-4.1")
        assert "user" in supported_params

    def test_model_normalization_for_responses_prefix(self):
        """Test that models with 'responses/' prefix are normalized correctly.

        The fix normalizes 'responses/gpt-4.1' to 'gpt-4.1' when checking
        if the model is in the list of supported OpenAI models.
        """
        # Both should have the same supported params
        regular_params = self.config.get_supported_openai_params("gpt-4.1-mini")
        responses_params = self.config.get_supported_openai_params("responses/gpt-4.1-mini")

        # 'user' should be in both
        assert "user" in regular_params
        assert "user" in responses_params

    def test_base_params_always_included(self):
        """Test that base params are always included regardless of model."""
        base_expected_params = [
            "frequency_penalty",
            "max_tokens",
            "temperature",
            "top_p",
            "stream",
            "tools",
            "tool_choice",
        ]

        supported_params = self.config.get_supported_openai_params("responses/gpt-4.1-mini")

        for param in base_expected_params:
            assert param in supported_params, f"Expected '{param}' in supported params"

    def test_prompt_cache_key_supported(self):
        """Test that 'prompt_cache_key' is in supported params for OpenAI chat completion models.

        OpenAI's Chat Completions API supports prompt_cache_key for cache routing optimization.
        """
        supported_params = self.config.get_supported_openai_params("gpt-4.1-nano")
        assert "prompt_cache_key" in supported_params

        supported_params = self.config.get_supported_openai_params("gpt-4.1")
        assert "prompt_cache_key" in supported_params


class TestGetOptionalParamsIntegration:
    """Integration tests using litellm.get_optional_params()"""

    def test_user_in_optional_params_for_responses_model(self):
        """Test that 'user' ends up in optional_params when using responses API models.

        Regression test for: https://github.com/BerriAI/litellm/issues/17633
        This verifies the full flow through get_optional_params().
        """
        from litellm.utils import get_optional_params

        # Test with responses model
        optional_params = get_optional_params(
            model="responses/gpt-4.1-mini",
            custom_llm_provider="openai",
            user="test-user-123",
        )
        assert optional_params.get("user") == "test-user-123"

    def test_user_in_optional_params_for_regular_model(self):
        """Test that 'user' ends up in optional_params for regular OpenAI models."""
        from litellm.utils import get_optional_params

        optional_params = get_optional_params(
            model="gpt-4o",
            custom_llm_provider="openai",
            user="test-user-456",
        )
        assert optional_params.get("user") == "test-user-456"

    def test_user_param_consistency_between_regular_and_responses(self):
        """Test that 'user' param behavior is consistent between regular and responses models."""
        from litellm.utils import get_optional_params

        regular_params = get_optional_params(
            model="gpt-4.1-mini",
            custom_llm_provider="openai",
            user="my-end-user",
        )

        responses_params = get_optional_params(
            model="responses/gpt-4.1-mini",
            custom_llm_provider="openai",
            user="my-end-user",
        )

        # Both should include user
        assert regular_params.get("user") == "my-end-user"
        assert responses_params.get("user") == "my-end-user"


class TestOpenAIChatCompletionStreamingHandler:
    """Tests for OpenAIChatCompletionStreamingHandler.chunk_parser()"""

    def test_chunk_parser_preserves_usage(self):
        """
        Test that chunk_parser preserves the usage field from streaming chunks.

        """
        handler = OpenAIChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        usage_chunk = {
            "id": "gen-123",
            "created": 1234567890,
            "model": "openai/gpt-4o-mini",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
            "usage": {
                "prompt_tokens": 13797,
                "completion_tokens": 350,
                "total_tokens": 14147,
            },
        }

        result = handler.chunk_parser(usage_chunk)

        assert result.usage is not None
        assert result.usage.prompt_tokens == 13797
        assert result.usage.completion_tokens == 350
        assert result.usage.total_tokens == 14147

    def test_chunk_parser_without_usage(self):
        """Test that chunk_parser works normally for chunks without usage."""
        handler = OpenAIChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        chunk = {
            "id": "gen-123",
            "created": 1234567890,
            "model": "openai/gpt-4o-mini",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello"},
                    "finish_reason": None,
                }
            ],
        }

        result = handler.chunk_parser(chunk)

        assert result.id == "gen-123"
        assert result.choices[0].delta.content == "Hello"
        assert not hasattr(result, "usage") or result.usage is None

    def test_chunk_parser_maps_reasoning_to_reasoning_content(self):
        """
        Test that chunk_parser maps 'reasoning' field to 'reasoning_content'.

        Some OpenAI-compatible providers (e.g., GLM-5, hosted_vllm) return
        delta.reasoning, but LiteLLM expects delta.reasoning_content.

        Regression test for: Streaming responses with delta.reasoning field
        coming back empty when using openai/ or hosted_vllm/ providers.
        """
        handler = OpenAIChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        # Simulate a chunk with reasoning field (as returned by GLM-5)
        chunk = {
            "id": "chatcmpl-8e3d624de9b12528",
            "object": "chat.completion.chunk",
            "created": 1771411455,
            "model": "glm-5",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning": "The capital of France",
                        "role": None,
                    },
                    "finish_reason": None,
                }
            ],
        }

        # Parse the chunk
        parsed_chunk = handler.chunk_parser(chunk)

        # Verify that reasoning was mapped to reasoning_content
        assert parsed_chunk.choices[0].delta.reasoning_content == "The capital of France"
        # Verify that the original 'reasoning' field was removed
        assert not hasattr(parsed_chunk.choices[0].delta, "reasoning")

    def test_chunk_parser_reasoning_field_not_present(self):
        """
        Test that chunks without reasoning field still work correctly.
        """
        handler = OpenAIChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        # Simulate a chunk without reasoning field
        chunk = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1769511767,
            "model": "gpt-4o",
            "choices": [
                {
                    "delta": {
                        "content": "Regular content",
                        "role": "assistant",
                    },
                    "finish_reason": None,
                    "index": 0,
                }
            ],
        }

        # Parse the chunk
        parsed_chunk = handler.chunk_parser(chunk)

        # Verify that content is present
        assert parsed_chunk.choices[0].delta.content == "Regular content"
        assert parsed_chunk.choices[0].delta.role == "assistant"
        # Verify that reasoning_content is not set (it should be deleted by Delta.__init__)
        assert not hasattr(parsed_chunk.choices[0].delta, "reasoning_content")


class TestPromptCacheKeyIntegration:
    """Tests for prompt_cache_key support"""

    def test_prompt_cache_key_in_optional_params(self):
        """Test that 'prompt_cache_key' flows through get_optional_params for OpenAI models."""
        from litellm.utils import get_optional_params

        optional_params = get_optional_params(
            model="gpt-4.1-nano",
            custom_llm_provider="openai",
            prompt_cache_key="test-cache-key-123",
        )
        assert optional_params.get("prompt_cache_key") == "test-cache-key-123"


class TestPromptCacheParams:
    """Tests for prompt_cache_key and prompt_cache_retention support."""

    def setup_method(self):
        self.config = OpenAIGPTConfig()

    def test_prompt_cache_key_in_supported_params(self):
        """Test that prompt_cache_key is in supported params for OpenAI models."""
        supported_params = self.config.get_supported_openai_params("gpt-4o")
        assert "prompt_cache_key" in supported_params

    def test_prompt_cache_retention_in_supported_params(self):
        """Test that prompt_cache_retention is in supported params for OpenAI models."""
        supported_params = self.config.get_supported_openai_params("gpt-4o")
        assert "prompt_cache_retention" in supported_params

    def test_prompt_cache_params_passed_through(self):
        """Test that prompt_cache_key and prompt_cache_retention are passed through by map_openai_params."""
        optional_params = self.config.map_openai_params(
            non_default_params={
                "prompt_cache_key": "my-cache-key",
                "prompt_cache_retention": "24h",
            },
            optional_params={},
            model="gpt-4o",
            drop_params=False,
        )
        assert optional_params.get("prompt_cache_key") == "my-cache-key"
        assert optional_params.get("prompt_cache_retention") == "24h"


class TestNormalizeToolCallIds:
    """
    Tests for OpenAIGPTConfig._normalize_tool_call_ids().

    Regression tests for: https://github.com/BerriAI/litellm/issues/22317
    Some OpenAI-compatible providers (e.g. MiniMax via OpenRouter) return tool
    call IDs like ``call_function_jlv0n7uyomle_1`` which contain underscores and
    exceed OpenAI's 9-character limit.  When those IDs are forwarded in a
    subsequent request to a strict provider (e.g. Mistral) the request is
    rejected with a 400 BadRequestError.
    """

    def setup_method(self):
        self.config = OpenAIGPTConfig()

    def test_compliant_ids_unchanged(self):
        """IDs that already satisfy ^[a-zA-Z0-9]{1,64}$ must not be modified."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "abc123xyz", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "abc123xyz", "content": "result"},
        ]
        result = self.config._normalize_tool_call_ids(messages)
        assert result[0]["tool_calls"][0]["id"] == "abc123xyz"
        assert result[1]["tool_call_id"] == "abc123xyz"

    def test_non_compliant_id_is_remapped(self):
        """IDs with underscores/hyphens or wrong length must be remapped."""
        bad_id = "call_function_jlv0n7uyomle_1"
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": bad_id, "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": bad_id, "content": "result"},
        ]
        result = self.config._normalize_tool_call_ids(messages)
        new_id = result[0]["tool_calls"][0]["id"]
        assert new_id != bad_id
        # Must satisfy the strict format
        import re
        assert re.match(r"^[a-zA-Z0-9]{1,64}$", new_id)
        # Assistant and tool messages must use the same new ID
        assert result[1]["tool_call_id"] == new_id

    def test_remapping_is_deterministic(self):
        """The same original ID must always produce the same normalised ID."""
        bad_id = "call_function_jlv0n7uyomle_1"
        messages = lambda: [  # noqa: E731
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": bad_id, "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": bad_id, "content": "result"},
        ]
        id_first = self.config._normalize_tool_call_ids(messages())[0]["tool_calls"][0]["id"]
        id_second = self.config._normalize_tool_call_ids(messages())[0]["tool_calls"][0]["id"]
        assert id_first == id_second

    def test_multiple_tool_calls_each_remapped_consistently(self):
        """Each distinct non-compliant ID gets its own deterministic mapping."""
        bad_id_1 = "call_function_aaaa_1"
        bad_id_2 = "call_function_bbbb_2"
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": bad_id_1, "type": "function", "function": {"name": "f1", "arguments": "{}"}},
                    {"id": bad_id_2, "type": "function", "function": {"name": "f2", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": bad_id_1, "content": "result1"},
            {"role": "tool", "tool_call_id": bad_id_2, "content": "result2"},
        ]
        result = self.config._normalize_tool_call_ids(messages)
        new_id_1 = result[0]["tool_calls"][0]["id"]
        new_id_2 = result[0]["tool_calls"][1]["id"]
        assert new_id_1 != bad_id_1
        assert new_id_2 != bad_id_2
        assert new_id_1 != new_id_2
        assert result[1]["tool_call_id"] == new_id_1
        assert result[2]["tool_call_id"] == new_id_2

    def test_messages_without_tool_calls_unaffected(self):
        """Plain user/assistant messages must pass through unchanged."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        result = self.config._normalize_tool_call_ids(messages)
        assert result == messages

    def test_transform_request_normalizes_ids(self):
        """transform_request must normalise non-compliant IDs end-to-end."""
        bad_id = "call_function_jlv0n7uyomle_1"
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": bad_id, "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": bad_id, "content": "sunny"},
        ]
        result = self.config.transform_request(
            model="gpt-4o",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        out_messages = result["messages"]
        new_id = out_messages[0]["tool_calls"][0]["id"]
        import re
        assert re.match(r"^[a-zA-Z0-9]{1,64}$", new_id)
        assert out_messages[1]["tool_call_id"] == new_id
