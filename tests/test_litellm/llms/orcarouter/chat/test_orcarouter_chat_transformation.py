import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.orcarouter.chat.transformation import (
    OrcaRouterChatCompletionStreamingHandler,
    OrcaRouterConfig,
    OrcaRouterException,
)


class TestOrcaRouterChatCompletionStreamingHandler:
    def test_chunk_parser_successful(self):
        handler = OrcaRouterChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        chunk = {
            "id": "test_id",
            "created": 1234567890,
            "model": "test_model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "choices": [
                {"delta": {"content": "test content", "reasoning": "test reasoning"}}
            ],
        }

        result = handler.chunk_parser(chunk)

        assert result.id == "test_id"
        assert result.object == "chat.completion.chunk"
        assert result.created == 1234567890
        assert result.model == "test_model"
        assert result.usage.prompt_tokens == chunk["usage"]["prompt_tokens"]
        assert result.usage.completion_tokens == chunk["usage"]["completion_tokens"]
        assert result.usage.total_tokens == chunk["usage"]["total_tokens"]
        assert len(result.choices) == 1
        assert result.choices[0]["delta"]["reasoning_content"] == "test reasoning"

    def test_chunk_parser_error_response(self):
        handler = OrcaRouterChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        error_chunk = {
            "error": {
                "message": "test error",
                "code": 400,
                "metadata": {"key": "value"},
                "user_id": "test_user",
            }
        }

        with pytest.raises(OrcaRouterException) as exc_info:
            handler.chunk_parser(error_chunk)

        assert "Message: test error" in str(exc_info.value)
        assert exc_info.value.status_code == 400

    def test_chunk_parser_key_error(self):
        handler = OrcaRouterChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        invalid_chunk = {"incomplete": "data"}

        with pytest.raises(OrcaRouterException) as exc_info:
            handler.chunk_parser(invalid_chunk)

        assert "KeyError" in str(exc_info.value)
        assert exc_info.value.status_code == 400


class TestOrcaRouterExtraBody:
    def test_map_openai_params_extracts_models_and_route(self):
        params = OrcaRouterConfig().map_openai_params(
            non_default_params={
                "models": ["openai/gpt-5", "openai/gpt-5-mini"],
                "route": "fallback",
                "temperature": 0.5,
            },
            optional_params={},
            model="orcarouter/auto",
            drop_params=False,
        )
        assert params["extra_body"] == {
            "models": ["openai/gpt-5", "openai/gpt-5-mini"],
            "route": "fallback",
        }
        assert params["temperature"] == 0.5

    def test_map_openai_params_no_extra_body_when_no_orcarouter_fields(self):
        params = OrcaRouterConfig().map_openai_params(
            non_default_params={"temperature": 0.5},
            optional_params={},
            model="orcarouter/openai/gpt-5",
            drop_params=False,
        )
        assert "extra_body" not in params

    def test_transform_request_merges_extra_body_to_top_level(self):
        request = OrcaRouterConfig().transform_request(
            model="orcarouter/openai/gpt-5",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={
                "extra_body": {"models": ["openai/gpt-5"], "route": "fallback"}
            },
            litellm_params={},
            headers={},
        )
        assert request["models"] == ["openai/gpt-5"]
        assert request["route"] == "fallback"
        assert "extra_body" not in request

    def test_extra_body_cannot_override_guarded_fields(self):
        """Security boundary: a caller's `extra_body` must not be able to
        clobber `model`, `messages`, or other guarded request fields. Only
        the OrcaRouter routing allowlist (`models`, `route`) survives."""
        request = OrcaRouterConfig().transform_request(
            model="orcarouter/openai/gpt-5",
            messages=[{"role": "user", "content": "original-user-input"}],
            optional_params={
                "extra_body": {
                    "model": "anthropic/claude-opus-4.7",
                    "messages": [{"role": "user", "content": "hijack-attempt"}],
                    "tools": [{"type": "function", "function": {"name": "exfil"}}],
                    "stream": True,
                    "models": ["openai/gpt-5", "openai/gpt-4o"],
                    "route": "fallback",
                }
            },
            litellm_params={},
            headers={},
        )
        # Allowlisted routing fields surface
        assert request["models"] == ["openai/gpt-5", "openai/gpt-4o"]
        assert request["route"] == "fallback"
        # Guarded fields keep their original values, not the attacker's
        assert request["model"] == "orcarouter/openai/gpt-5"
        assert request["messages"] == [
            {"role": "user", "content": "original-user-input"}
        ]
        # Non-allowlisted extras are dropped entirely
        assert "tools" not in request
        assert "stream" not in request

    def test_no_usage_cost_injection(self):
        """OrcaRouter doesn't return usage.cost; we must not inject usage:{include:true}."""
        request = OrcaRouterConfig().transform_request(
            model="orcarouter/openai/gpt-5",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert "usage" not in request

    def test_native_router_name_gets_orcarouter_prefix_restored(self):
        """`auto` (bare router name, prefix-stripped by LiteLLM) → API sees `orcarouter/auto`."""
        request = OrcaRouterConfig().transform_request(
            model="auto",  # post-strip form from get_llm_provider
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert request["model"] == "orcarouter/auto"

    def test_vendor_namespaced_model_passes_through_unchanged(self):
        """`openai/gpt-5` (has /) → API sees `openai/gpt-5` (no prefix added)."""
        request = OrcaRouterConfig().transform_request(
            model="openai/gpt-5",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert request["model"] == "openai/gpt-5"


class TestOrcaRouterReasoningPerVendor:
    def test_anthropic_reasoning_effort_converts_to_thinking_block(self):
        params = OrcaRouterConfig().map_openai_params(
            non_default_params={"reasoning_effort": "high"},
            optional_params={},
            model="orcarouter/anthropic/claude-opus-4.7",
            drop_params=False,
        )
        assert "reasoning_effort" not in params
        assert params["thinking"] == {"type": "enabled", "budget_tokens": 8192}

    def test_anthropic_reasoning_effort_medium_default_budget(self):
        params = OrcaRouterConfig().map_openai_params(
            non_default_params={"reasoning_effort": "medium"},
            optional_params={},
            model="orcarouter/anthropic/claude-opus-4.7",
            drop_params=False,
        )
        assert params["thinking"]["budget_tokens"] == 4096

    def test_anthropic_explicit_thinking_preserved(self):
        params = OrcaRouterConfig().map_openai_params(
            non_default_params={
                "thinking": {"type": "enabled", "budget_tokens": 16000}
            },
            optional_params={},
            model="orcarouter/anthropic/claude-opus-4.7",
            drop_params=False,
        )
        assert params["thinking"] == {"type": "enabled", "budget_tokens": 16000}

    def test_openai_uses_reasoning_effort_top_level(self):
        params = OrcaRouterConfig().map_openai_params(
            non_default_params={"reasoning_effort": "high"},
            optional_params={},
            model="orcarouter/openai/gpt-5",
            drop_params=False,
        )
        assert params["reasoning_effort"] == "high"
        assert "thinking" not in params

    def test_openai_thinking_block_dropped(self):
        """OpenAI doesn't accept thinking block; drop it."""
        params = OrcaRouterConfig().map_openai_params(
            non_default_params={"thinking": {"type": "enabled", "budget_tokens": 4096}},
            optional_params={},
            model="orcarouter/openai/gpt-5",
            drop_params=False,
        )
        assert "thinking" not in params

    def test_deepseek_reasoner_drops_all_reasoning_fields(self):
        params = OrcaRouterConfig().map_openai_params(
            non_default_params={
                "reasoning_effort": "high",
                "thinking": {"type": "enabled", "budget_tokens": 4096},
            },
            optional_params={},
            model="orcarouter/deepseek/deepseek-reasoner",
            drop_params=False,
        )
        assert "reasoning_effort" not in params
        assert "thinking" not in params


class TestOrcaRouterModelQuirks:
    def test_opus47_drops_temperature_and_top_k(self):
        params = OrcaRouterConfig().map_openai_params(
            non_default_params={"temperature": 0.5, "top_k": 40},
            optional_params={},
            model="orcarouter/anthropic/claude-opus-4.7",
            drop_params=False,
        )
        assert "temperature" not in params
        assert "top_k" not in params

    def test_gpt5_family_drops_temperature(self):
        for model_id in [
            "orcarouter/openai/gpt-5",
            "orcarouter/openai/gpt-5-mini",
            "orcarouter/openai/gpt-5-nano",
            "orcarouter/openai/gpt-5.2-pro",
        ]:
            params = OrcaRouterConfig().map_openai_params(
                non_default_params={"temperature": 0.7},
                optional_params={},
                model=model_id,
                drop_params=False,
            )
            assert "temperature" not in params, f"failed for {model_id}"

    def test_gpt4_family_drops_top_k(self):
        for model_id in [
            "orcarouter/openai/gpt-4o",
            "orcarouter/openai/gpt-4o-mini",
            "orcarouter/openai/gpt-4.1",
            "orcarouter/openai/gpt-4.1-mini",
            "orcarouter/openai/gpt-4.1-nano",
            "orcarouter/openai/gpt-4-turbo",
        ]:
            params = OrcaRouterConfig().map_openai_params(
                non_default_params={"top_k": 40, "temperature": 0.7},
                optional_params={},
                model=model_id,
                drop_params=False,
            )
            assert "top_k" not in params, f"failed for {model_id}"
            assert (
                params.get("temperature") == 0.7
            ), f"unrelated field dropped for {model_id}"

    def test_grok43_drops_penalty_fields(self):
        params = OrcaRouterConfig().map_openai_params(
            non_default_params={
                "presence_penalty": 0.5,
                "frequency_penalty": 0.5,
                "temperature": 0.5,
            },
            optional_params={},
            model="orcarouter/grok/grok-4.3",
            drop_params=False,
        )
        assert "presence_penalty" not in params
        assert "frequency_penalty" not in params
        assert params["temperature"] == 0.5

    def test_kimi_k26_forces_temperature_and_top_p(self):
        params = OrcaRouterConfig().map_openai_params(
            non_default_params={"temperature": 0.5, "top_p": 0.5},
            optional_params={},
            model="orcarouter/kimi/kimi-k2.6",
            drop_params=False,
        )
        assert params["temperature"] == 1
        assert params["top_p"] == 0.95


class TestOrcaRouterCacheControl:
    def test_cache_control_supported_for_claude(self):
        config = OrcaRouterConfig()
        assert config._supports_cache_control_in_content("anthropic/claude-opus-4.7")
        assert config._supports_cache_control_in_content(
            "orcarouter/anthropic/claude-opus-4.7"
        )

    def test_cache_control_supported_for_gemini(self):
        assert OrcaRouterConfig()._supports_cache_control_in_content(
            "google/gemini-3-flash-preview"
        )

    def test_cache_control_supported_for_minimax_glm_zai(self):
        config = OrcaRouterConfig()
        assert config._supports_cache_control_in_content("minimax/minimax-m2.7")
        assert config._supports_cache_control_in_content("z-ai/glm-4.5")

    def test_cache_control_not_supported_for_openai(self):
        assert not OrcaRouterConfig()._supports_cache_control_in_content("openai/gpt-5")

    def test_cache_control_moved_to_last_content_block(self):
        config = OrcaRouterConfig()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": "second"},
                ],
                "cache_control": {"type": "ephemeral"},
            }
        ]
        transformed = config._move_cache_control_to_content(messages)
        assert "cache_control" not in transformed[0]
        assert "cache_control" not in transformed[0]["content"][0]
        assert transformed[0]["content"][1]["cache_control"] == {"type": "ephemeral"}

    def test_cache_control_string_content_converted_to_block(self):
        config = OrcaRouterConfig()
        messages = [
            {
                "role": "user",
                "content": "hello",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        transformed = config._move_cache_control_to_content(messages)
        assert transformed[0]["content"] == [
            {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}
        ]


class TestOrcaRouterTransformResponse:
    """Smoke-test the response-parsing path on an OrcaRouter-shaped payload.

    `OrcaRouterConfig` inherits `transform_response` from `OpenAIGPTConfig`,
    so this is intentionally a thin integration check: confirm the inherited
    parser handles an OrcaRouter chat-completion response (which is
    OpenAI-compatible) without raising and populates content + usage.
    """

    def test_transform_response_parses_orcarouter_payload(self):
        from unittest.mock import Mock

        import httpx

        from litellm.types.utils import ModelResponse

        config = OrcaRouterConfig()
        raw = Mock(spec=httpx.Response)
        raw.status_code = 200
        raw.text = ""
        raw.headers = {}
        raw.json.return_value = {
            "id": "or-gen-abc",
            "object": "chat.completion",
            "created": 1779190000,
            "model": "openai/gpt-5",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 1,
                "total_tokens": 9,
            },
        }

        logging_obj = Mock()
        result = config.transform_response(
            model="orcarouter/openai/gpt-5",
            raw_response=raw,
            model_response=ModelResponse(),
            logging_obj=logging_obj,
            request_data={},
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert result.choices[0].message.content == "hello"
        assert result.usage.prompt_tokens == 8
        assert result.usage.completion_tokens == 1
        assert result.usage.total_tokens == 9
