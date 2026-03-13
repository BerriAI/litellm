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
from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config


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


class TestGPT5ReasoningEffortPreservation:
    """Tests for GPT-5 reasoning_effort dict preservation for Responses API."""

    def setup_method(self):
        self.config = OpenAIGPT5Config()

    def test_reasoning_effort_string_preserved(self):
        """Test that reasoning_effort as string is preserved."""
        non_default_params = {"reasoning_effort": "high"}
        optional_params = {}
        
        self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gpt-5.4",
            drop_params=False,
        )
        
        # String format should be preserved
        assert non_default_params.get("reasoning_effort") == "high"

    def test_reasoning_effort_dict_with_only_effort_normalized(self):
        """Test that reasoning_effort dict with only 'effort' key is normalized to string."""
        non_default_params = {"reasoning_effort": {"effort": "high"}}
        optional_params = {}
        
        self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gpt-5.4",
            drop_params=False,
        )
        
        # Dict with only 'effort' should be normalized to string
        assert non_default_params.get("reasoning_effort") == "high"

    def test_reasoning_effort_dict_with_summary_preserved(self):
        """Test that reasoning_effort dict with 'summary' field is preserved for Responses API.
        
        Regression test for: User reported that summary field was being dropped when
        routing to Responses API. The dict format with additional fields should be
        preserved so it can be properly handled by the Responses API transformation.
        """
        non_default_params = {"reasoning_effort": {"effort": "high", "summary": "detailed"}}
        optional_params = {}
        
        self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gpt-5.4",
            drop_params=False,
        )
        
        # Dict with additional fields should be preserved
        assert non_default_params.get("reasoning_effort") == {"effort": "high", "summary": "detailed"}
        assert isinstance(non_default_params.get("reasoning_effort"), dict)
        assert non_default_params["reasoning_effort"]["effort"] == "high"
        assert non_default_params["reasoning_effort"]["summary"] == "detailed"

    def test_reasoning_effort_dict_with_generate_summary_preserved(self):
        """Test that reasoning_effort dict with 'generate_summary' field is preserved."""
        non_default_params = {"reasoning_effort": {"effort": "medium", "generate_summary": "auto"}}
        optional_params = {}
        
        self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gpt-5.4",
            drop_params=False,
        )
        
        # Dict with additional fields should be preserved
        assert non_default_params.get("reasoning_effort") == {"effort": "medium", "generate_summary": "auto"}
        assert isinstance(non_default_params.get("reasoning_effort"), dict)

    def test_reasoning_effort_dict_with_all_fields_preserved(self):
        """Test that reasoning_effort dict with all fields is preserved."""
        non_default_params = {
            "reasoning_effort": {
                "effort": "high",
                "summary": "detailed",
                "generate_summary": "concise"
            }
        }
        optional_params = {}
        
        self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gpt-5.4",
            drop_params=False,
        )
        
        # Dict with all fields should be preserved
        reasoning = non_default_params.get("reasoning_effort")
        assert isinstance(reasoning, dict)
        assert reasoning["effort"] == "high"
        assert reasoning["summary"] == "detailed"
        assert reasoning["generate_summary"] == "concise"

    def test_reasoning_effort_dict_xhigh_triggers_validation(self):
        """xhigh-dict: effective effort is extracted for model-support validation.
        
        When reasoning_effort={"effort": "xhigh", "summary": "detailed"} is passed to a model
        that doesn't support xhigh (e.g. gpt-5.1), the xhigh guard must fire.
        """
        import litellm

        non_default_params = {"reasoning_effort": {"effort": "xhigh", "summary": "detailed"}}
        optional_params = {}

        with pytest.raises(litellm.utils.UnsupportedParamsError):
            self.config.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model="gpt-5.1",
                drop_params=False,
            )

    def test_reasoning_effort_dict_xhigh_dropped_when_requested(self):
        """xhigh-dict with drop_params=True: reasoning_effort is dropped."""
        non_default_params = {"reasoning_effort": {"effort": "xhigh", "summary": "detailed"}}
        optional_params = {}

        self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gpt-5.1",
            drop_params=True,
        )

        assert "reasoning_effort" not in non_default_params

    def test_reasoning_effort_dict_none_dropped_for_gpt5_4_with_tools(self):
        """none-dict with tools on gpt-5.4: reasoning_effort is dropped."""
        tools = [{"type": "function", "function": {"name": "test", "description": "test"}}]
        non_default_params = {"reasoning_effort": {"effort": "none", "summary": "detailed"}, "tools": tools}
        optional_params = {}

        self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gpt-5.4",
            drop_params=False,
        )

        assert "reasoning_effort" not in non_default_params
        assert non_default_params.get("tools") == tools

    def test_reasoning_effort_dict_none_treated_as_none_for_sampling(self):
        """none-dict: {"effort": "none", "summary": "detailed"} allows logprobs/top_p.
        
        Sampling-param guard should NOT fire; logprobs should be kept.
        """
        non_default_params = {
            "reasoning_effort": {"effort": "none", "summary": "detailed"},
            "logprobs": True,
        }
        optional_params = {}

        self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gpt-5.1",
            drop_params=False,
        )

        assert non_default_params.get("reasoning_effort") == {"effort": "none", "summary": "detailed"}
        assert non_default_params.get("logprobs") is True

    def test_reasoning_effort_dict_none_allows_temperature(self):
        """none-dict: {"effort": "none", "summary": "detailed"} allows non-default temperature."""
        non_default_params = {
            "reasoning_effort": {"effort": "none", "summary": "detailed"},
            "temperature": 0.5,
        }
        optional_params = {}

        self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gpt-5.1",
            drop_params=False,
        )

        assert optional_params.get("temperature") == 0.5
        assert non_default_params.get("reasoning_effort") == {"effort": "none", "summary": "detailed"}
