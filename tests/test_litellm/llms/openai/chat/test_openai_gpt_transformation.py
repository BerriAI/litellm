"""
Tests for OpenAI GPT transformation (litellm/llms/openai/chat/gpt_transformation.py)
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm import utils
from litellm.llms.openai.openai import OpenAIChatCompletion
from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config
from litellm.llms.openai.chat.gpt_transformation import (
    OpenAIChatCompletionStreamingHandler,
    OpenAIGPTConfig,
)
from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config


class _OpenAIChatCompletionResponse:
    def model_dump(self):
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }


def test_custom_openai_drop_params_removes_openai_sdk_unsupported_params():
    captured_data = {}
    original_drop_params = litellm.drop_params
    litellm.drop_params = True

    def fake_get_openai_client(*args, **kwargs):
        return SimpleNamespace(
            api_key="test-key",
            _base_url=SimpleNamespace(_uri_reference="http://example.com/v1"),
        )

    def fake_openai_chat_completion_request(self, openai_client, data, timeout, logging_obj):
        captured_data.update(data)
        return {}, _OpenAIChatCompletionResponse()

    try:
        with patch.object(OpenAIChatCompletion, "_get_openai_client", fake_get_openai_client), patch.object(
            OpenAIChatCompletion,
            "make_sync_openai_chat_completion_request",
            fake_openai_chat_completion_request,
        ):
            litellm.completion(
                model="custom_openai/test-model",
                api_base="http://example.com/v1",
                api_key="test-key",
                messages=[{"role": "user", "content": "hi"}],
                output_config={"effort": "high"},
                context_management={"edits": "clear"},
                thinking={"type": "enabled"},
                extra_body={"kept": True},
            )
    finally:
        litellm.drop_params = original_drop_params

    assert "output_config" not in captured_data
    assert "context_management" not in captured_data
    assert "thinking" not in captured_data
    assert captured_data["extra_body"] == {"kept": True}


def test_custom_openai_sdk_param_filter_keeps_params_when_drop_disabled():
    optional_params = {
        "temperature": 0.2,
        "thinking": {"type": "enabled"},
    }
    original_drop_params = litellm.drop_params
    litellm.drop_params = False

    try:
        filtered_params = utils._drop_unsupported_openai_sdk_params(
            optional_params=optional_params,
            supported_params=["temperature"],
            drop_params=False,
        )
    finally:
        litellm.drop_params = original_drop_params

    assert filtered_params == optional_params


def test_custom_openai_sdk_param_filter_allows_missing_extra_body():
    original_drop_params = litellm.drop_params
    litellm.drop_params = True

    try:
        filtered_params = utils._drop_unsupported_openai_sdk_params(
            optional_params={
                "temperature": 0.2,
                "thinking": {"type": "enabled"},
            },
            supported_params=["temperature"],
            drop_params=True,
        )
    finally:
        litellm.drop_params = original_drop_params

    assert filtered_params == {"temperature": 0.2}


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
        supported_params = self.config.get_supported_openai_params(
            "responses/gpt-4.1-mini"
        )
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
        responses_params = self.config.get_supported_openai_params(
            "responses/gpt-4.1-mini"
        )

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

        supported_params = self.config.get_supported_openai_params(
            "responses/gpt-4.1-mini"
        )

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

    def test_chunk_parser_raises_on_in_body_error_payload(self):
        """vLLM/sglang return HTTP 200 streams whose body carries the error,
        e.g. data: {"error": {..., "code": 400}}. chunk_parser must surface it
        as a provider error instead of parsing an empty chunk that silently
        ends the stream (https://github.com/BerriAI/litellm/issues/25492)."""
        from litellm.llms.openai.common_utils import OpenAIError

        handler = OpenAIChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        error_chunk = {
            "error": {
                "object": "error",
                "message": "The model is not multimodal. Please remove image inputs.",
                "type": "BadRequestError",
                "param": None,
                "code": 400,
            }
        }

        with pytest.raises(OpenAIError) as excinfo:
            handler.chunk_parser(error_chunk)

        assert excinfo.value.status_code == 400
        assert "not multimodal" in excinfo.value.message

    def test_chunk_parser_error_payload_without_usable_code_maps_to_500(self):
        """OpenAI-style error payloads may carry a string code (e.g.
        "invalid_api_key") or none at all; those must map to 500, not crash."""
        from litellm.llms.openai.common_utils import OpenAIError

        handler = OpenAIChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        with pytest.raises(OpenAIError) as excinfo:
            handler.chunk_parser(
                {"error": {"message": "engine crashed", "code": "server_error"}}
            )
        assert excinfo.value.status_code == 500
        assert "engine crashed" in excinfo.value.message

        with pytest.raises(OpenAIError) as excinfo:
            handler.chunk_parser({"error": "plain string error"})
        assert excinfo.value.status_code == 500
        assert "plain string error" in excinfo.value.message

        with pytest.raises(OpenAIError) as excinfo:
            handler.chunk_parser({"error": {"type": "overloaded", "code": 503}})
        assert excinfo.value.status_code == 503
        assert excinfo.value.message == '{"type": "overloaded", "code": 503}'

    def test_chunk_parser_tolerates_null_error_field(self):
        """A chunk that carries "error": null alongside real data must parse
        normally, not raise."""
        handler = OpenAIChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        chunk = {
            "id": "gen-123",
            "created": 1234567890,
            "model": "openai/gpt-4o-mini",
            "object": "chat.completion.chunk",
            "error": None,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello"},
                    "finish_reason": None,
                }
            ],
        }

        result = handler.chunk_parser(chunk)
        assert result.choices[0].delta.content == "Hello"

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
        assert (
            parsed_chunk.choices[0].delta.reasoning_content == "The capital of France"
        )
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

    def test_chunk_parser_without_id_field(self):
        """
        Test that chunk_parser works when chunk is missing the 'id' field.

        Some OpenAI-compatible providers (e.g., MiniMax) return streaming chunks
        without an 'id' field in certain cases. This should not raise KeyError.

        Regression test for: KeyError: 'id' when using MiniMax m2.5 model
        """
        handler = OpenAIChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        # Simulate a chunk without 'id' field (as returned by MiniMax)
        chunk = {
            "object": "chat.completion.chunk",
            "created": 1769511767,
            "model": "minimax/m2.5",
            "choices": [
                {
                    "delta": {
                        "content": "Hello",
                        "role": "assistant",
                    },
                    "finish_reason": None,
                    "index": 0,
                }
            ],
        }

        # Parse the chunk - should not raise KeyError
        parsed_chunk = handler.chunk_parser(chunk)

        # Verify that content is present and id was auto-generated
        assert parsed_chunk.choices[0].delta.content == "Hello"
        assert parsed_chunk.choices[0].delta.role == "assistant"
        # ModelResponseStream auto-generates an id when None is passed
        assert parsed_chunk.id is not None


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

    def test_reasoning_effort_dict_with_summary_normalized(self):
        """Test that reasoning_effort dict with 'summary' is normalized for Chat Completions API.

        map_openai_params normalizes all dicts to string. Full dict is restored in main.py
        when routing to Responses API (test_gpt_5_4_responses_bridge_preserves_reasoning_summary_dict).
        """
        non_default_params = {
            "reasoning_effort": {"effort": "high", "summary": "detailed"}
        }
        optional_params = {}

        self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gpt-5.4",
            drop_params=False,
        )

        # Dict is normalized to string for Chat Completions API
        assert non_default_params.get("reasoning_effort") == "high"

    def test_reasoning_effort_dict_with_generate_summary_normalized(self):
        """Test that reasoning_effort dict with 'generate_summary' is normalized for Chat Completions API."""
        non_default_params = {
            "reasoning_effort": {"effort": "medium", "generate_summary": "auto"}
        }
        optional_params = {}

        self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gpt-5.4",
            drop_params=False,
        )

        # Dict is normalized to string for Chat Completions API
        assert non_default_params.get("reasoning_effort") == "medium"

    def test_reasoning_effort_dict_with_all_fields_normalized(self):
        """Test that reasoning_effort dict with all fields is normalized to effort string."""
        non_default_params = {
            "reasoning_effort": {
                "effort": "high",
                "summary": "detailed",
                "generate_summary": "concise",
            }
        }
        optional_params = {}

        self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gpt-5.4",
            drop_params=False,
        )

        # Dict is normalized to string for Chat Completions API
        assert non_default_params.get("reasoning_effort") == "high"

    def test_reasoning_effort_dict_xhigh_triggers_validation(self):
        """xhigh-dict: effective effort is extracted for model-support validation.

        When reasoning_effort={"effort": "xhigh", "summary": "detailed"} is passed to a model
        that doesn't support xhigh (e.g. gpt-5.1), the xhigh guard must fire.
        """
        import litellm

        non_default_params = {
            "reasoning_effort": {"effort": "xhigh", "summary": "detailed"}
        }
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
        non_default_params = {
            "reasoning_effort": {"effort": "xhigh", "summary": "detailed"}
        }
        optional_params = {}

        self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gpt-5.1",
            drop_params=True,
        )

        assert "reasoning_effort" not in non_default_params

    def test_reasoning_effort_dict_none_passed_through_for_gpt5_4_with_tools(self):
        """none-dict with tools on gpt-5.4: reasoning_effort is passed through (routing to Responses at completion level)."""
        tools = [
            {"type": "function", "function": {"name": "test", "description": "test"}}
        ]
        non_default_params = {
            "reasoning_effort": {"effort": "none", "summary": "detailed"},
            "tools": tools,
        }
        optional_params = {}

        self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gpt-5.4",
            drop_params=False,
        )

        # Normalized to "none", passed through; routing to Responses API happens at completion()
        assert non_default_params.get("reasoning_effort") == "none"
        assert non_default_params.get("tools") == tools

    def test_reasoning_effort_dict_none_treated_as_none_for_sampling(self):
        """none-dict: {"effort": "none", "summary": "detailed"} allows logprobs/top_p.

        effective_effort='none' is used for sampling guard; logprobs should be kept.
        Dict is normalized to "none" for Chat Completions API.
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

        assert non_default_params.get("reasoning_effort") == "none"
        assert non_default_params.get("logprobs") is True

    def test_reasoning_effort_dict_none_allows_temperature(self):
        """none-dict: {"effort": "none", "summary": "detailed"} allows non-default temperature.

        effective_effort='none' is used for temperature guard. Dict is normalized to "none".
        """
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
        assert non_default_params.get("reasoning_effort") == "none"


class TestCacheControlPreservationForCustomEndpoint:
    """
    Regression tests for https://github.com/BerriAI/litellm/issues/30319

    The AnthropicCacheControlHook injects cache_control when a user passes
    cache_control_injection_points, but the base OpenAIGPTConfig used to strip
    it unconditionally, making the feature a guaranteed no-op for the generic
    openai provider pointed at a cache_control-aware endpoint (a LiteLLM proxy,
    vLLM, an Anthropic-compatible gateway). cache_control must survive there
    while still being stripped for real api.openai.com.
    """

    def setup_method(self):
        self.config = OpenAIGPTConfig()

    @pytest.fixture(autouse=True)
    def _clean_openai_base_env(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        monkeypatch.setattr(litellm, "api_base", None, raising=False)

    @staticmethod
    def _cache_controlled_messages():
        return [
            {
                "role": "system",
                "content": "You are helpful.",
                "cache_control": {"type": "ephemeral"},
            },
            {
                "role": "user",
                "content": "Hello",
                "cache_control": {"type": "ephemeral"},
            },
        ]

    def _transform(self, custom_llm_provider, api_base, optional_params=None):
        return self.config.transform_request(
            model="claude-sonnet-4",
            messages=self._cache_controlled_messages(),
            optional_params=optional_params or {},
            litellm_params={
                "custom_llm_provider": custom_llm_provider,
                "api_base": api_base,
            },
            headers={},
        )

    def test_predicate_openai_provider_custom_api_base_preserves(self):
        assert (
            self.config._should_preserve_cache_control_for_endpoint(
                "openai", "http://localhost:4000/v1"
            )
            is True
        )

    def test_predicate_real_openai_no_api_base_strips(self):
        assert (
            self.config._should_preserve_cache_control_for_endpoint("openai", None)
            is False
        )

    def test_predicate_explicit_openai_host_strips(self):
        assert (
            self.config._should_preserve_cache_control_for_endpoint(
                "openai", "https://api.openai.com/v1"
            )
            is False
        )

    def test_predicate_non_openai_provider_strips(self):
        assert (
            self.config._should_preserve_cache_control_for_endpoint(
                "deepseek", "https://api.deepseek.com"
            )
            is False
        )

    def test_predicate_resolves_openai_base_url_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:4000/v1")
        assert (
            self.config._should_preserve_cache_control_for_endpoint("openai", None)
            is True
        )

    def test_predicate_resolves_openai_api_base_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_BASE", "http://localhost:4000/v1")
        assert (
            self.config._should_preserve_cache_control_for_endpoint("openai", None)
            is True
        )

    def test_predicate_lookalike_host_is_not_treated_as_openai(self):
        assert (
            self.config._should_preserve_cache_control_for_endpoint(
                "openai", "https://api.openai.com.evil.example/v1"
            )
            is True
        )

    def test_predicate_openai_subdomain_strips(self):
        assert (
            self.config._should_preserve_cache_control_for_endpoint(
                "openai", "https://eu.api.openai.com/v1"
            )
            is False
        )

    def test_transform_request_preserves_for_custom_api_base(self):
        body = self._transform("openai", "http://localhost:4000/v1")
        assert all("cache_control" in m for m in body["messages"])

    def test_transform_request_strips_for_real_openai(self):
        body = self._transform("openai", None)
        assert all("cache_control" not in m for m in body["messages"])

    def test_transform_request_strips_for_non_openai_provider(self):
        body = self._transform("fireworks_ai", "https://api.fireworks.ai/inference/v1")
        assert all("cache_control" not in m for m in body["messages"])

    def test_transform_request_preserves_tool_cache_control(self):
        tools = [
            {
                "type": "function",
                "function": {"name": "f", "parameters": {}},
                "cache_control": {"type": "ephemeral"},
            }
        ]
        body = self._transform(
            "openai", "http://localhost:4000/v1", optional_params={"tools": tools}
        )
        assert "cache_control" in body["tools"][0]

    @pytest.mark.asyncio
    async def test_async_transform_request_preserves_for_custom_api_base(self):
        body = await self.config.async_transform_request(
            model="claude-sonnet-4",
            messages=self._cache_controlled_messages(),
            optional_params={},
            litellm_params={
                "custom_llm_provider": "openai",
                "api_base": "http://localhost:4000/v1",
            },
            headers={},
        )
        assert all("cache_control" in m for m in body["messages"])

    @pytest.mark.asyncio
    async def test_async_transform_request_strips_for_real_openai(self):
        body = await self.config.async_transform_request(
            model="gpt-4o",
            messages=self._cache_controlled_messages(),
            optional_params={},
            litellm_params={"custom_llm_provider": "openai", "api_base": None},
            headers={},
        )
        assert all("cache_control" not in m for m in body["messages"])
