"""
Tests for Anthropic CountTokens API OAuth token handling.

Verifies that get_required_headers() correctly handles OAuth tokens
(sk-ant-oat*) by delegating to optionally_handle_anthropic_oauth().

Regression test for https://github.com/BerriAI/litellm/issues/22040
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

import httpx
import pytest
import respx

import litellm
from litellm.llms.anthropic.count_tokens.token_counter import AnthropicTokenCounter
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
        assert (
            "token-counting" in beta_value
        ), f"token-counting beta missing from OAuth headers: {beta_value}"
        assert (
            "oauth-2025-04-20" in beta_value
        ), f"oauth beta missing from OAuth headers: {beta_value}"


@pytest.mark.asyncio
async def test_a_global_anthropic_key_counts_like_chat_would_authenticate(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "anthropic_key", FAKE_REGULAR_KEY)

    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://api.anthropic.com/v1/messages/count_tokens").mock(
            return_value=httpx.Response(200, json={"input_tokens": 13})
        )
        result = await AnthropicTokenCounter().count_tokens(
            model_to_use="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
            deployment={"litellm_params": {"model": "anthropic/claude-sonnet-4-5"}},
            request_model="claude-sonnet-4-5",
        )

    assert result is not None
    assert result.total_tokens == 13
    assert route.calls.last.request.headers["x-api-key"] == FAKE_REGULAR_KEY


@pytest.mark.asyncio
async def test_no_anthropic_key_returns_a_401_count_error(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(litellm, "anthropic_key", None)
    monkeypatch.setattr(litellm, "api_key", None)

    result = await AnthropicTokenCounter().count_tokens(
        model_to_use="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "hi"}],
        contents=None,
        deployment={"litellm_params": {"model": "anthropic/claude-sonnet-4-5"}},
        request_model="claude-sonnet-4-5",
    )

    assert result is not None
    assert result.error is True
    assert result.status_code == 401
    assert result.total_tokens == 0
