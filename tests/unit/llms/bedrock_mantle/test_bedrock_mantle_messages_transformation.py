"""
Unit tests for the bedrock_mantle native Anthropic Messages route.

Mantle serves its Claude models only on `/anthropic/v1/messages` (the OpenAI
paths reject them), so `bedrock_mantle/anthropic.claude-*` requests on
/v1/messages must hit that endpoint directly instead of the chat-completions
bridge. Rust sends these calls; the wire request is tested under tests/test_litellm_rust/messages.
These tests lock the dispatcher gate, the URL derivation from the OpenAI-surface base that
get_llm_provider pre-fills, and the request transform count_tokens reads.
"""

from typing import Final

import pytest

import litellm
from litellm.constants import BEDROCK_MIN_THINKING_BUDGET_TOKENS
from litellm.llms.bedrock_mantle.messages.transformation import (
    BedrockMantleAnthropicMessagesConfig,
    build_mantle_native_messages_url,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager

MESSAGES_PATH = "/anthropic/v1/messages"


@pytest.fixture(autouse=True)
def _no_ambient_mantle_env(monkeypatch):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)
    monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
    monkeypatch.delenv("BEDROCK_MANTLE_REGION", raising=False)
    monkeypatch.delenv("AWS_REGION_NAME", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)


_SSE_EVENTS = (
    (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_stream",
                "type": "message",
                "role": "assistant",
                "model": "anthropic.claude-sonnet-5",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        },
    ),
    ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
    (
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "pong"}},
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
    ("message_stop", {"type": "message_stop"}),
)


class TestDispatch:
    def test_claude_models_get_the_native_messages_config(self):
        config = ProviderConfigManager.get_provider_anthropic_messages_config(
            model="anthropic.claude-sonnet-5", provider=litellm.LlmProviders.BEDROCK_MANTLE
        )
        assert isinstance(config, BedrockMantleAnthropicMessagesConfig)
        assert config.custom_llm_provider == "bedrock_mantle"

    @pytest.mark.parametrize("model", ["openai.gpt-5.6-sol", "openai.gpt-oss-120b-1:0", "google.gemma-4-31b"])
    def test_non_claude_models_keep_the_bridge(self, model):
        assert (
            ProviderConfigManager.get_provider_anthropic_messages_config(
                model=model, provider=litellm.LlmProviders.BEDROCK_MANTLE
            )
            is None
        )


class TestURL:
    @pytest.mark.parametrize(
        "api_base",
        [
            "https://bedrock-mantle.us-east-1.api.aws/v1",
            "https://bedrock-mantle.us-east-1.api.aws/openai/v1",
            "https://bedrock-mantle.us-east-1.api.aws/openai/v1/",
            "https://bedrock-mantle.us-east-1.api.aws",
            "https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages",
        ],
    )
    def test_prefilled_openai_base_becomes_the_messages_endpoint(self, api_base):
        url = build_mantle_native_messages_url(api_base, {"aws_region_name": "us-east-1"})
        assert url == f"https://bedrock-mantle.us-east-1.api.aws{MESSAGES_PATH}"

    def test_aws_region_name_wins_over_the_prefilled_host_region(self):
        url = build_mantle_native_messages_url(
            "https://bedrock-mantle.us-east-1.api.aws/v1", {"aws_region_name": "us-east-2"}
        )
        assert url == f"https://bedrock-mantle.us-east-2.api.aws{MESSAGES_PATH}"

    def test_host_region_is_used_when_no_region_param(self):
        url = build_mantle_native_messages_url("https://bedrock-mantle.eu-west-1.api.aws/v1", {})
        assert url == f"https://bedrock-mantle.eu-west-1.api.aws{MESSAGES_PATH}"

    def test_custom_host_is_preserved(self):
        url = build_mantle_native_messages_url("https://vpce-abc.bedrock-mantle.example.com/v1", {})
        assert url == f"https://vpce-abc.bedrock-mantle.example.com{MESSAGES_PATH}"

    def test_env_base_is_used_without_api_base(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_MANTLE_API_BASE", "https://mantle-proxy.internal/openai/v1")
        assert build_mantle_native_messages_url(None, {}) == f"https://mantle-proxy.internal{MESSAGES_PATH}"

    def test_default_host_comes_from_mantle_region_env(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_MANTLE_REGION", "ap-northeast-1")
        assert (
            build_mantle_native_messages_url(None, {})
            == f"https://bedrock-mantle.ap-northeast-1.api.aws{MESSAGES_PATH}"
        )

    def test_config_get_complete_url_reads_litellm_params(self):
        config = BedrockMantleAnthropicMessagesConfig()
        url = config.get_complete_url(
            api_base="https://bedrock-mantle.us-east-1.api.aws/v1",
            api_key=None,
            model="anthropic.claude-sonnet-5",
            optional_params={},
            litellm_params={"aws_region_name": "us-west-2"},
        )
        assert url == f"https://bedrock-mantle.us-west-2.api.aws{MESSAGES_PATH}"


class TestRequestBody:
    def test_body_carries_model_and_stream_but_not_the_invoke_version(self):
        config = BedrockMantleAnthropicMessagesConfig()
        body = config.transform_anthropic_messages_request(
            model="anthropic.claude-sonnet-5",
            messages=[{"role": "user", "content": "ping"}],
            anthropic_messages_optional_request_params={"max_tokens": 8, "stream": True},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert body["model"] == "anthropic.claude-sonnet-5"
        assert body["stream"] is True
        assert body["max_tokens"] == 8
        assert "anthropic_version" not in body

    def test_body_omits_stream_when_not_streaming(self):
        config = BedrockMantleAnthropicMessagesConfig()
        body = config.transform_anthropic_messages_request(
            model="anthropic.claude-sonnet-5",
            messages=[{"role": "user", "content": "ping"}],
            anthropic_messages_optional_request_params={"max_tokens": 8},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert "stream" not in body

    @pytest.mark.parametrize(
        ("model", "thinking_params", "expected_thinking", "expected_edit_types"),
        [
            pytest.param(
                "anthropic.claude-3-5-haiku-20241022-v1:0",
                {"thinking": {"type": "adaptive"}},
                None,
                ["clear_tool_uses_20250919"],
                id="no-thinking-support-drops-edit",
            ),
            pytest.param(
                "anthropic.claude-3-5-haiku-20241022-v1:0",
                {},
                None,
                ["clear_thinking_20251015", "clear_tool_uses_20250919"],
                id="caller-edit-without-thinking-forwarded",
            ),
            pytest.param(
                "anthropic.claude-opus-4-5-20251101-v1:0",
                {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
                {"type": "enabled", "budget_tokens": BEDROCK_MIN_THINKING_BUDGET_TOKENS},
                ["clear_thinking_20251015", "clear_tool_uses_20250919"],
                id="thinking-added-back-keeps-edit",
            ),
            pytest.param(
                "anthropic.claude-opus-4-5-20251101-v1:0",
                {"max_tokens": BEDROCK_MIN_THINKING_BUDGET_TOKENS, "reasoning_effort": "none"},
                None,
                ["clear_tool_uses_20250919"],
                id="max-tokens-too-small-to-add-thinking-back-drops-edit",
            ),
            pytest.param(
                "anthropic.claude-mythos-preview",
                {},
                None,
                ["clear_thinking_20251015", "clear_tool_uses_20250919"],
                id="always-on-model-keeps-edit",
            ),
            pytest.param(
                "anthropic.claude-sonnet-4-6",
                {"thinking": {"type": "adaptive"}},
                {"type": "adaptive"},
                ["clear_thinking_20251015", "clear_tool_uses_20250919"],
                id="adaptive-thinking-keeps-edit",
            ),
        ],
    )
    def test_clear_thinking_edit_dropped_only_with_dropped_thinking(
        self,
        model: str,
        thinking_params: dict[str, object],
        expected_thinking: dict[str, object] | None,
        expected_edit_types: list[str],
    ) -> None:
        body: Final = BedrockMantleAnthropicMessagesConfig().transform_anthropic_messages_request(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            anthropic_messages_optional_request_params={
                "max_tokens": 32000,
                **thinking_params,
                "context_management": {
                    "edits": [{"type": "clear_thinking_20251015", "keep": "all"}, {"type": "clear_tool_uses_20250919"}]
                },
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert body.get("thinking") == expected_thinking
        assert [edit["type"] for edit in body["context_management"]["edits"]] == expected_edit_types

    def test_disabled_thinking_without_edits_is_untouched(self) -> None:
        body: Final = BedrockMantleAnthropicMessagesConfig().transform_anthropic_messages_request(
            model="anthropic.claude-3-5-haiku-20241022-v1:0",
            messages=[{"role": "user", "content": "ping"}],
            anthropic_messages_optional_request_params={"max_tokens": 8, "thinking": {"type": "disabled"}},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert body["thinking"] == {"type": "disabled"}

    def test_body_hoists_wrapped_defer_loading_and_drops_custom(self):
        schema: Final = {"type": "object", "properties": {}}
        body = BedrockMantleAnthropicMessagesConfig().transform_anthropic_messages_request(
            model="anthropic.claude-sonnet-5",
            messages=[{"role": "user", "content": "ping"}],
            anthropic_messages_optional_request_params={
                "max_tokens": 8,
                "tools": [{"name": "Read", "input_schema": schema, "custom": {"defer_loading": True}}],
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert body["tools"] == [{"name": "Read", "input_schema": schema, "defer_loading": True}]
