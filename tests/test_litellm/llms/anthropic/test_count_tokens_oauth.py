"""
Tests for Anthropic CountTokens API OAuth token handling.

Verifies that get_required_headers() correctly handles OAuth tokens
(sk-ant-oat*) by delegating to optionally_handle_anthropic_oauth().

Regression test for https://github.com/BerriAI/litellm/issues/22040
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from litellm.llms.anthropic.count_tokens.transformation import (
    AnthropicCountTokensConfig,
)

# Fake tokens for testing (not real secrets)
FAKE_OAUTH_TOKEN = "sk-ant-oat01-fake-token-for-testing-123456789abcdef"
FAKE_REGULAR_KEY = "sk-ant-api03-regular-key-for-testing-123456789"


class TestCountTokensOAuthHeaders:
    """Tests that count_tokens headers are correct for both regular and OAuth keys."""

    def test_regular_api_key_uses_x_api_key(self):
        """Regular API keys should be sent via x-api-key header."""
        config = AnthropicCountTokensConfig()
        headers = config.get_required_headers(FAKE_REGULAR_KEY)

        assert headers["x-api-key"] == FAKE_REGULAR_KEY
        assert "authorization" not in headers

    def test_oauth_key_uses_bearer_authorization(self):
        """OAuth tokens (sk-ant-oat*) should be sent via Authorization: Bearer."""
        config = AnthropicCountTokensConfig()
        headers = config.get_required_headers(FAKE_OAUTH_TOKEN)

        assert headers.get("authorization") == f"Bearer {FAKE_OAUTH_TOKEN}"
        assert "x-api-key" not in headers

    def test_oauth_key_sets_oauth_beta_header(self):
        """OAuth tokens should trigger the anthropic-beta oauth header."""
        config = AnthropicCountTokensConfig()
        headers = config.get_required_headers(FAKE_OAUTH_TOKEN)

        assert "oauth-2025-04-20" in headers.get("anthropic-beta", "")

    def test_regular_key_preserves_token_counting_beta(self):
        """Regular keys should keep the token-counting beta header."""
        config = AnthropicCountTokensConfig()
        headers = config.get_required_headers(FAKE_REGULAR_KEY)

        assert "token-counting" in headers.get("anthropic-beta", "")

    def test_headers_always_have_content_type(self):
        """Both regular and OAuth paths should have Content-Type."""
        config = AnthropicCountTokensConfig()

        for key in [FAKE_REGULAR_KEY, FAKE_OAUTH_TOKEN]:
            headers = config.get_required_headers(key)
            assert headers["Content-Type"] == "application/json"

    def test_headers_always_have_anthropic_version(self):
        """Both paths should have anthropic-version."""
        config = AnthropicCountTokensConfig()

        for key in [FAKE_REGULAR_KEY, FAKE_OAUTH_TOKEN]:
            headers = config.get_required_headers(key)
            assert headers["anthropic-version"] == "2023-06-01"

    def test_oauth_key_preserves_token_counting_beta(self):
        """OAuth tokens must preserve the token-counting beta alongside the OAuth beta."""
        config = AnthropicCountTokensConfig()
        headers = config.get_required_headers(FAKE_OAUTH_TOKEN)

        beta_value = headers.get("anthropic-beta", "")
        assert "token-counting" in beta_value, f"token-counting beta missing from OAuth headers: {beta_value}"
        assert "oauth-2025-04-20" in beta_value, f"oauth beta missing from OAuth headers: {beta_value}"


class TestCountTokensUsesWorkloadIdentity:
    """A federated deployment holds no static key. Without minting one, count_tokens returns None
    and the caller silently falls back to the local tokenizer, so the number a federated
    deployment reports would never come from Anthropic."""

    @pytest.mark.asyncio
    async def test_a_federated_deployment_mints_and_counts(self, monkeypatch):
        from litellm.llms.anthropic.count_tokens import token_counter as token_counter_module

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        minted = "sk-ant-oat01-minted-for-count"

        async def fake_mint(_params, _api_base, _model):
            return minted

        monkeypatch.setattr(token_counter_module, "aget_anthropic_wif_token", fake_mint, raising=False)
        monkeypatch.setattr("litellm.llms.anthropic.wif.aget_anthropic_wif_token", fake_mint, raising=False)

        seen: dict[str, object] = {}

        async def fake_request(**kwargs):
            seen.update(kwargs)
            return {"input_tokens": 42}

        monkeypatch.setattr(
            token_counter_module.anthropic_count_tokens_handler,
            "handle_count_tokens_request",
            fake_request,
            raising=False,
        )

        result = await token_counter_module.AnthropicTokenCounter().count_tokens(
            model_to_use="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
            deployment={
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5",
                    "anthropic_federation_rule_id": "fdrl_x",
                    "anthropic_organization_id": "org-x",
                }
            },
            request_model="claude-sonnet-4-5",
        )

        assert result is not None
        assert result.total_tokens == 42
        assert seen["api_key"] == minted

    @pytest.mark.asyncio
    async def test_an_auth_token_deployment_never_mints(self, monkeypatch):
        from litellm.llms.anthropic.count_tokens import token_counter as token_counter_module

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "bearer-token-for-testing")
        mint_calls: list[str] = []

        async def fake_mint(_params, _api_base, model):
            mint_calls.append(model)
            return "sk-ant-oat01-should-not-be-minted"

        monkeypatch.setattr("litellm.llms.anthropic.wif.aget_anthropic_wif_token", fake_mint, raising=False)

        result = await token_counter_module.AnthropicTokenCounter().count_tokens(
            model_to_use="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
            deployment={
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5",
                    "anthropic_federation_rule_id": "fdrl_x",
                    "anthropic_organization_id": "org-x",
                }
            },
            request_model="claude-sonnet-4-5",
        )

        assert result is None
        assert mint_calls == []

    @pytest.mark.asyncio
    async def test_a_failed_mint_degrades_like_an_anthropic_error(self, monkeypatch):
        import litellm
        from litellm.llms.anthropic.count_tokens import token_counter as token_counter_module

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

        async def failing_mint(_params, _api_base, model):
            raise litellm.AuthenticationError(
                message="federation_rule_id is not a well-formed fdrl_ tagged ID",
                llm_provider="anthropic",
                model=model,
            )

        monkeypatch.setattr("litellm.llms.anthropic.wif.aget_anthropic_wif_token", failing_mint, raising=False)

        result = await token_counter_module.AnthropicTokenCounter().count_tokens(
            model_to_use="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
            deployment={
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5",
                    "anthropic_federation_rule_id": "not-a-rule",
                    "anthropic_organization_id": "org-x",
                }
            },
            request_model="claude-sonnet-4-5",
        )

        assert result is not None
        assert result.error is True
        assert result.status_code == 401
        assert result.total_tokens == 0
        assert "fdrl_" in (result.error_message or "")
