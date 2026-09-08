from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.proxy.route_llm_request import stash_and_strip_context_1m_model_suffix


class TestAnthropicContext1mBetaHeader:
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

    def test_route_strips_suffix_and_stashes_original(self):
        data = {"model": "claude-sonnet-4-6[1m]"}
        stash_and_strip_context_1m_model_suffix(data)
        assert data["model"] == "claude-sonnet-4-6"
        assert data["_original_model"] == "claude-sonnet-4-6[1m]"

    def test_route_strips_uppercase_suffix(self):
        data = {"model": "claude-sonnet-4-6[1M]"}
        stash_and_strip_context_1m_model_suffix(data)
        assert data["model"] == "claude-sonnet-4-6"
        assert data["_original_model"] == "claude-sonnet-4-6[1M]"

    def test_route_leaves_unsuffixed_model_alone(self):
        data = {"model": "claude-sonnet-4-6"}
        stash_and_strip_context_1m_model_suffix(data)
        assert data == {"model": "claude-sonnet-4-6"}
