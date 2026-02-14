"""
Tests for Anthropic OAuth token handling for Claude Code Max integration.
"""

import os
import sys

# Add litellm to path
sys.path.insert(0, os.path.abspath("../../../../.."))

# Fake OAuth token for testing (not a real secret)
FAKE_OAUTH_TOKEN = "sk-ant-oat01-fake-token-for-testing-123456789abcdef"


def test_oauth_detection_in_common_utils():
    """Test 1: OAuth token detection in common_utils"""
    from litellm.llms.anthropic.common_utils import optionally_handle_anthropic_oauth

    headers = {"authorization": f"Bearer {FAKE_OAUTH_TOKEN}"}
    updated_headers, extracted_api_key = optionally_handle_anthropic_oauth(headers, None)

    assert extracted_api_key == FAKE_OAUTH_TOKEN
    assert updated_headers["anthropic-beta"] == "oauth-2025-04-20"
    assert updated_headers["anthropic-dangerous-direct-browser-access"] == "true"


def test_oauth_integration_in_validate_environment():
    """Test 2: OAuth integration in AnthropicConfig validate_environment"""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    config = AnthropicModelInfo()
    headers = {"authorization": f"Bearer {FAKE_OAUTH_TOKEN}"}

    updated_headers = config.validate_environment(
        headers=headers,
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )

    assert updated_headers["x-api-key"] == FAKE_OAUTH_TOKEN
    assert updated_headers["anthropic-dangerous-direct-browser-access"] == "true"


def test_oauth_detection_in_messages_transformation():
    """Test 3: OAuth detection in messages transformation"""
    from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
        AnthropicMessagesConfig,
    )

    config = AnthropicMessagesConfig()
    headers = {"authorization": f"Bearer {FAKE_OAUTH_TOKEN}"}

    updated_headers, _ = config.validate_anthropic_messages_environment(
        headers=headers,
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )

    assert updated_headers["x-api-key"] == FAKE_OAUTH_TOKEN
    assert "oauth-2025-04-20" in updated_headers["anthropic-beta"]
    assert updated_headers["anthropic-dangerous-direct-browser-access"] == "true"


def test_regular_api_keys_still_work():
    """Test 4: Regular API keys still work (regression test)"""
    from litellm.llms.anthropic.common_utils import optionally_handle_anthropic_oauth

    regular_key = "sk-ant-api03-regular-key-123"
    headers = {"authorization": f"Bearer {regular_key}"}

    updated_headers, extracted_api_key = optionally_handle_anthropic_oauth(headers, regular_key)

    # Regular key should be unchanged
    assert extracted_api_key == regular_key
    # OAuth headers should NOT be added
    assert "anthropic-dangerous-direct-browser-access" not in updated_headers

class TestIsAnthropicOAuthKey:
    """Tests for is_anthropic_oauth_key helper function."""

    def test_oauth_token_raw(self):
        """Raw OAuth token should be detected."""
        from litellm.llms.anthropic.common_utils import is_anthropic_oauth_key

        assert is_anthropic_oauth_key("sk-ant-oat01-abc123") is True
        assert is_anthropic_oauth_key("sk-ant-oat02-xyz789") is True

    def test_oauth_token_bearer_format(self):
        """Bearer-prefixed OAuth token should be detected."""
        from litellm.llms.anthropic.common_utils import is_anthropic_oauth_key

        assert is_anthropic_oauth_key("Bearer sk-ant-oat01-abc123") is True
        assert is_anthropic_oauth_key("Bearer sk-ant-oat02-xyz789") is True

    def test_non_oauth_tokens(self):
        """Non-OAuth values should return False."""
        from litellm.llms.anthropic.common_utils import is_anthropic_oauth_key

        assert is_anthropic_oauth_key(None) is False
        assert is_anthropic_oauth_key("") is False
        assert is_anthropic_oauth_key("sk-ant-api01-abc123") is False
        assert is_anthropic_oauth_key("Bearer sk-ant-api01-abc123") is False

    def test_case_sensitivity(self):
        """OAuth prefix matching should be case-sensitive."""
        from litellm.llms.anthropic.common_utils import is_anthropic_oauth_key

        assert is_anthropic_oauth_key("sk-ant-OAT01-abc123") is False
        assert is_anthropic_oauth_key("SK-ANT-OAT01-abc123") is False

    def test_just_prefix(self):
        """Just the prefix with no suffix should still match."""
        from litellm.llms.anthropic.common_utils import is_anthropic_oauth_key

        assert is_anthropic_oauth_key("sk-ant-oat") is True


class TestProxyOAuthHeaderForwarding:
    """Tests for proxy-layer OAuth header preservation and forwarding."""

    def test_clean_headers_preserves_oauth_authorization(self):
        """clean_headers should preserve Authorization header with OAuth tokens."""
        from starlette.datastructures import Headers

        from litellm.proxy.litellm_pre_call_utils import clean_headers

        raw_headers = Headers(
            raw=[
                (b"authorization", f"Bearer {FAKE_OAUTH_TOKEN}".encode()),
                (b"content-type", b"application/json"),
            ]
        )
        cleaned = clean_headers(raw_headers)

        assert "authorization" in cleaned
        assert cleaned["authorization"] == f"Bearer {FAKE_OAUTH_TOKEN}"
        assert cleaned["content-type"] == "application/json"

    def test_clean_headers_strips_non_oauth_authorization(self):
        """clean_headers should strip Authorization header with regular API keys."""
        from starlette.datastructures import Headers

        from litellm.proxy.litellm_pre_call_utils import clean_headers

        raw_headers = Headers(
            raw=[
                (b"authorization", b"Bearer sk-regular-key-123"),
                (b"content-type", b"application/json"),
            ]
        )
        cleaned = clean_headers(raw_headers)

        assert "authorization" not in cleaned
        assert cleaned["content-type"] == "application/json"

    def test_add_provider_specific_headers_forwards_oauth(self):
        """add_provider_specific_headers_to_request should forward OAuth Authorization
        as a ProviderSpecificHeader scoped to Anthropic-compatible providers."""
        from litellm.proxy.litellm_pre_call_utils import (
            add_provider_specific_headers_to_request,
        )

        data: dict = {}
        headers = {
            "authorization": f"Bearer {FAKE_OAUTH_TOKEN}",
            "content-type": "application/json",
        }

        add_provider_specific_headers_to_request(data=data, headers=headers)

        assert "provider_specific_header" in data
        psh = data["provider_specific_header"]
        assert "anthropic" in psh["custom_llm_provider"]
        assert "bedrock" in psh["custom_llm_provider"]
        assert "vertex_ai" in psh["custom_llm_provider"]
        assert psh["extra_headers"]["authorization"] == f"Bearer {FAKE_OAUTH_TOKEN}"

    def test_add_provider_specific_headers_ignores_non_oauth(self):
        """add_provider_specific_headers_to_request should not create a
        ProviderSpecificHeader for non-OAuth Authorization headers."""
        from litellm.proxy.litellm_pre_call_utils import (
            add_provider_specific_headers_to_request,
        )

        data: dict = {}
        headers = {
            "authorization": "Bearer sk-regular-key-123",
            "content-type": "application/json",
        }

        add_provider_specific_headers_to_request(data=data, headers=headers)

        assert "provider_specific_header" not in data

    def test_add_provider_specific_headers_combines_anthropic_and_oauth(self):
        """When both anthropic-beta and OAuth Authorization are present, both
        should be included in the ProviderSpecificHeader."""
        from litellm.proxy.litellm_pre_call_utils import (
            add_provider_specific_headers_to_request,
        )

        data: dict = {}
        headers = {
            "authorization": f"Bearer {FAKE_OAUTH_TOKEN}",
            "anthropic-beta": "oauth-2025-04-20",
            "content-type": "application/json",
        }

        add_provider_specific_headers_to_request(data=data, headers=headers)

        assert "provider_specific_header" in data
        psh = data["provider_specific_header"]
        assert psh["extra_headers"]["authorization"] == f"Bearer {FAKE_OAUTH_TOKEN}"
        assert psh["extra_headers"]["anthropic-beta"] == "oauth-2025-04-20"
