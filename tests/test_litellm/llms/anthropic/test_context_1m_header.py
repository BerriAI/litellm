from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.anthropic.common_utils import (
    context_1m_requested,
    strip_context_1m_suffix,
)
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)


class TestAnthropicContext1mBetaHeader:
    def test_strip_context_1m_suffix_is_pure(self):
        assert strip_context_1m_suffix("claude-sonnet-4-6[1m]") == "claude-sonnet-4-6"
        assert strip_context_1m_suffix("claude-sonnet-4-6[1M]") == "claude-sonnet-4-6"
        assert strip_context_1m_suffix("claude-sonnet-4-6") == "claude-sonnet-4-6"
        assert strip_context_1m_suffix("gpt-4o[1m]") == "gpt-4o"

    def test_context_1m_requested_from_model_or_stashed_original(self):
        assert context_1m_requested(model="claude-sonnet-4-6[1m]")
        assert context_1m_requested(
            model="claude-sonnet-4-6",
            litellm_params={"_original_model": "claude-sonnet-4-6[1M]"},
        )
        assert not context_1m_requested(model="claude-sonnet-4-6")
        assert not context_1m_requested(model="gpt-4o")

    def test_get_anthropic_headers_with_context_1m(self):
        headers = AnthropicConfig().get_anthropic_headers(api_key="sk-test", context_1m_supported=True)
        assert "context-1m-2025-08-07" in headers.get("anthropic-beta", "")

    def test_get_anthropic_headers_without_context_1m(self):
        headers = AnthropicConfig().get_anthropic_headers(api_key="sk-test", context_1m_supported=False)
        assert "context-1m-2025-08-07" not in headers.get("anthropic-beta", "")

    def test_validate_environment_adds_context_1m_for_suffixed_model(self):
        headers = AnthropicConfig().validate_environment(
            headers={},
            model="claude-sonnet-4-6[1m]",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            api_key="sk-test",
        )
        assert "context-1m-2025-08-07" in headers.get("anthropic-beta", "")

    def test_validate_environment_no_context_1m_without_suffix(self):
        headers = AnthropicConfig().validate_environment(
            headers={},
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            api_key="sk-test",
        )
        assert "context-1m-2025-08-07" not in headers.get("anthropic-beta", "")

    def test_validate_environment_uses_original_model_from_litellm_params(self):
        headers = AnthropicConfig().validate_environment(
            headers={},
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={"_original_model": "claude-sonnet-4-6[1M]"},
            api_key="sk-test",
        )
        assert "context-1m-2025-08-07" in headers.get("anthropic-beta", "")

    def test_update_headers_adds_context_1m_with_original_model(self):
        headers = AnthropicConfig().update_headers_with_optional_anthropic_beta(
            headers={},
            optional_params={"_original_model": "claude-sonnet-4-6[1m]"},
        )
        assert "context-1m-2025-08-07" in headers.get("anthropic-beta", "")

    def test_update_headers_no_context_1m_without_original_model(self):
        headers = AnthropicConfig().update_headers_with_optional_anthropic_beta(
            headers={},
            optional_params={},
        )
        assert "context-1m-2025-08-07" not in headers.get("anthropic-beta", "")

    def test_pass_through_adds_context_1m(self):
        headers = AnthropicMessagesConfig._update_headers_with_anthropic_beta(
            headers={},
            optional_params={"_original_model": "claude-sonnet-4-6[1m]"},
        )
        assert "context-1m-2025-08-07" in headers.get("anthropic-beta", "")

    def test_pass_through_no_context_1m_without_suffix(self):
        headers = AnthropicMessagesConfig._update_headers_with_anthropic_beta(
            headers={},
            optional_params={},
        )
        assert "context-1m-2025-08-07" not in headers.get("anthropic-beta", "")

    def test_pass_through_adds_context_1m_from_model_without_original_model(self):
        headers, _ = AnthropicMessagesConfig().validate_anthropic_messages_environment(
            headers={},
            model="claude-sonnet-4-6[1m]",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            api_key="sk-test",
        )
        assert "context-1m-2025-08-07" in headers.get("anthropic-beta", "")

    def test_pass_through_transform_strips_1m_from_upstream_model(self):
        result = AnthropicMessagesConfig().transform_anthropic_messages_request(
            model="claude-sonnet-4-6[1m]",
            messages=[{"role": "user", "content": "hi"}],
            anthropic_messages_optional_request_params={"max_tokens": 16},
            litellm_params={},
            headers={},
        )
        assert result["model"] == "claude-sonnet-4-6"

    def test_chat_transform_request_strips_1m_and_injects_header(self):
        headers = {}
        result = AnthropicConfig().transform_request(
            model="claude-sonnet-4-6[1m]",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"max_tokens": 16},
            litellm_params={},
            headers=headers,
        )
        assert result["model"] == "claude-sonnet-4-6"
        assert "context-1m-2025-08-07" in headers.get("anthropic-beta", "")

    def test_chat_transform_request_uppercase_suffix_strips_and_injects_header(self):
        headers = {}
        result = AnthropicConfig().transform_request(
            model="claude-sonnet-4-6[1M]",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"max_tokens": 16},
            litellm_params={},
            headers=headers,
        )
        assert result["model"] == "claude-sonnet-4-6"
        assert "context-1m-2025-08-07" in headers.get("anthropic-beta", "")

    def test_proxy_does_not_globally_strip_non_anthropic_suffix(self):
        from litellm.proxy import route_llm_request

        assert not hasattr(route_llm_request, "stash_and_strip_context_1m_model_suffix")
