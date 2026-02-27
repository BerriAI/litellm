"""
Tests for Anthropic OAuth token handling in common_utils.

Verifies that OAuth tokens (sk-ant-oat*) are sent via Authorization: Bearer
instead of x-api-key, per Anthropic's OAuth specification.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

# Fake OAuth token for testing (not a real secret)
FAKE_OAUTH_TOKEN = "sk-ant-oat01-fake-token-for-testing-123456789abcdef"
FAKE_REGULAR_KEY = "sk-ant-api03-regular-key-for-testing-123456789"


class TestOptionallyHandleAnthropicOAuth:
    """Tests for optionally_handle_anthropic_oauth function."""

    def test_oauth_token_in_authorization_header(self):
        """OAuth token in Authorization header should be detected and headers set correctly."""
        from litellm.llms.anthropic.common_utils import (
            optionally_handle_anthropic_oauth,
        )

        headers = {"authorization": f"Bearer {FAKE_OAUTH_TOKEN}"}
        updated_headers, extracted_api_key = optionally_handle_anthropic_oauth(
            headers, None
        )

        assert extracted_api_key == FAKE_OAUTH_TOKEN
        assert updated_headers["anthropic-beta"] == "oauth-2025-04-20"
        assert updated_headers["anthropic-dangerous-direct-browser-access"] == "true"
        assert "x-api-key" not in updated_headers

    def test_oauth_token_in_api_key_directly(self):
        """OAuth token passed as api_key should set Authorization: Bearer header."""
        from litellm.llms.anthropic.common_utils import (
            optionally_handle_anthropic_oauth,
        )

        headers = {}
        updated_headers, returned_api_key = optionally_handle_anthropic_oauth(
            headers, FAKE_OAUTH_TOKEN
        )

        assert returned_api_key == FAKE_OAUTH_TOKEN
        assert updated_headers["authorization"] == f"Bearer {FAKE_OAUTH_TOKEN}"
        assert updated_headers["anthropic-beta"] == "oauth-2025-04-20"
        assert updated_headers["anthropic-dangerous-direct-browser-access"] == "true"
        assert "x-api-key" not in updated_headers

    def test_oauth_removes_existing_x_api_key(self):
        """When OAuth is detected, any existing x-api-key should be removed."""
        from litellm.llms.anthropic.common_utils import (
            optionally_handle_anthropic_oauth,
        )

        headers = {"x-api-key": FAKE_OAUTH_TOKEN}
        updated_headers, _ = optionally_handle_anthropic_oauth(
            headers, FAKE_OAUTH_TOKEN
        )

        assert "x-api-key" not in updated_headers
        assert updated_headers["authorization"] == f"Bearer {FAKE_OAUTH_TOKEN}"

    def test_regular_api_key_unchanged(self):
        """Regular API keys (non-OAuth) should pass through unmodified."""
        from litellm.llms.anthropic.common_utils import (
            optionally_handle_anthropic_oauth,
        )

        headers = {}
        updated_headers, returned_api_key = optionally_handle_anthropic_oauth(
            headers, FAKE_REGULAR_KEY
        )

        assert returned_api_key == FAKE_REGULAR_KEY
        assert "authorization" not in updated_headers
        assert "anthropic-dangerous-direct-browser-access" not in updated_headers
        assert "anthropic-beta" not in updated_headers

    def test_regular_key_in_authorization_header(self):
        """Non-OAuth token in Authorization header should not trigger OAuth handling."""
        from litellm.llms.anthropic.common_utils import (
            optionally_handle_anthropic_oauth,
        )

        headers = {"authorization": f"Bearer {FAKE_REGULAR_KEY}"}
        updated_headers, returned_api_key = optionally_handle_anthropic_oauth(
            headers, FAKE_REGULAR_KEY
        )

        assert returned_api_key == FAKE_REGULAR_KEY
        assert "anthropic-dangerous-direct-browser-access" not in updated_headers

    def test_none_api_key_no_error(self):
        """None api_key with empty headers should not raise errors."""
        from litellm.llms.anthropic.common_utils import (
            optionally_handle_anthropic_oauth,
        )

        headers = {}
        updated_headers, returned_api_key = optionally_handle_anthropic_oauth(
            headers, None
        )

        assert returned_api_key is None
        assert "authorization" not in updated_headers


class TestGetAnthropicHeaders:
    """Tests for get_anthropic_headers method with OAuth support."""

    def test_oauth_token_uses_authorization_bearer(self):
        """OAuth token should produce Authorization: Bearer header, not x-api-key."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        headers = config.get_anthropic_headers(
            api_key=FAKE_OAUTH_TOKEN,
            computer_tool_used=False,
            prompt_caching_set=False,
            pdf_used=False,
            is_vertex_request=False,
        )

        assert headers["authorization"] == f"Bearer {FAKE_OAUTH_TOKEN}"
        assert headers["anthropic-dangerous-direct-browser-access"] == "true"
        assert "oauth-2025-04-20" in headers.get("anthropic-beta", "")
        assert "x-api-key" not in headers

    def test_regular_key_uses_x_api_key(self):
        """Regular API key should produce x-api-key header, not Authorization."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        headers = config.get_anthropic_headers(
            api_key=FAKE_REGULAR_KEY,
            computer_tool_used=False,
            prompt_caching_set=False,
            pdf_used=False,
            is_vertex_request=False,
        )

        assert headers["x-api-key"] == FAKE_REGULAR_KEY
        assert "authorization" not in headers
        assert "anthropic-dangerous-direct-browser-access" not in headers

    def test_oauth_includes_standard_headers(self):
        """OAuth path should still include standard Anthropic headers."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        headers = config.get_anthropic_headers(
            api_key=FAKE_OAUTH_TOKEN,
            computer_tool_used=False,
            prompt_caching_set=False,
            pdf_used=False,
            is_vertex_request=False,
        )

        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["accept"] == "application/json"
        assert headers["content-type"] == "application/json"


class TestValidateEnvironmentOAuth:
    """Tests for validate_environment with OAuth tokens."""

    def test_oauth_via_authorization_header(self):
        """validate_environment should produce correct headers for OAuth tokens."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        headers = {"authorization": f"Bearer {FAKE_OAUTH_TOKEN}"}

        updated_headers = config.validate_environment(
            headers=headers,
            model="claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )

        assert updated_headers["authorization"] == f"Bearer {FAKE_OAUTH_TOKEN}"
        assert updated_headers["anthropic-dangerous-direct-browser-access"] == "true"
        assert "oauth-2025-04-20" in updated_headers.get("anthropic-beta", "")
        assert "x-api-key" not in updated_headers

    def test_oauth_via_api_key_param(self):
        """validate_environment with OAuth token as api_key should use Bearer auth."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        headers = {}

        updated_headers = config.validate_environment(
            headers=headers,
            model="claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=FAKE_OAUTH_TOKEN,
            api_base=None,
        )

        assert updated_headers["authorization"] == f"Bearer {FAKE_OAUTH_TOKEN}"
        assert updated_headers["anthropic-dangerous-direct-browser-access"] == "true"
        assert "x-api-key" not in updated_headers

    def test_regular_key_via_api_key_param(self):
        """validate_environment with regular API key should use x-api-key."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        headers = {}

        updated_headers = config.validate_environment(
            headers=headers,
            model="claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=FAKE_REGULAR_KEY,
            api_base=None,
        )

        assert updated_headers["x-api-key"] == FAKE_REGULAR_KEY
        assert "authorization" not in updated_headers
        assert "anthropic-dangerous-direct-browser-access" not in updated_headers


class TestPassthroughOAuth:
    """Tests for passthrough messages endpoint with OAuth tokens."""

    def test_passthrough_oauth_no_x_api_key(self):
        """Passthrough endpoint should not add x-api-key for OAuth tokens."""
        from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
            AnthropicMessagesConfig,
        )

        config = AnthropicMessagesConfig()
        headers = {"authorization": f"Bearer {FAKE_OAUTH_TOKEN}"}

        updated_headers, _ = config.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )

        assert "oauth-2025-04-20" in updated_headers.get("anthropic-beta", "")
        assert updated_headers["anthropic-dangerous-direct-browser-access"] == "true"
        assert "x-api-key" not in updated_headers

    def test_passthrough_regular_key_uses_x_api_key(self):
        """Passthrough endpoint should still use x-api-key for regular API keys."""
        from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
            AnthropicMessagesConfig,
        )

        config = AnthropicMessagesConfig()
        headers = {}

        updated_headers, _ = config.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=FAKE_REGULAR_KEY,
            api_base=None,
        )

        assert updated_headers["x-api-key"] == FAKE_REGULAR_KEY
        assert "authorization" not in updated_headers


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

    def test_clean_headers_forwards_anthropic_api_key_when_enabled(self):
        """clean_headers should preserve x-api-key when forward_llm_provider_auth_headers=True."""
        from starlette.datastructures import Headers

        from litellm.proxy.litellm_pre_call_utils import clean_headers

        raw_headers = Headers(
            raw=[
                (b"authorization", b"Bearer sk-proxy-auth"),
                (b"x-api-key", b"sk-ant-api03-test-key"),
                (b"content-type", b"application/json"),
            ]
        )
        cleaned = clean_headers(raw_headers, forward_llm_provider_auth_headers=True)

        # x-api-key should be preserved when flag is True
        assert "x-api-key" in cleaned
        assert cleaned["x-api-key"] == "sk-ant-api03-test-key"
        # Authorization (proxy auth) should still be stripped
        assert "authorization" not in cleaned
        assert cleaned["content-type"] == "application/json"

    def test_clean_headers_strips_anthropic_api_key_when_disabled(self):
        """clean_headers should strip x-api-key when forward_llm_provider_auth_headers=False (default)."""
        from starlette.datastructures import Headers

        from litellm.proxy.litellm_pre_call_utils import clean_headers

        raw_headers = Headers(
            raw=[
                (b"x-api-key", b"sk-ant-api03-test-key"),
                (b"content-type", b"application/json"),
            ]
        )
        cleaned = clean_headers(raw_headers, forward_llm_provider_auth_headers=False)

        # x-api-key should be stripped by default
        assert "x-api-key" not in cleaned
        assert cleaned["content-type"] == "application/json"

    def test_clean_headers_forwards_google_api_key_when_enabled(self):
        """clean_headers should preserve x-goog-api-key when forward_llm_provider_auth_headers=True."""
        from starlette.datastructures import Headers

        from litellm.proxy.litellm_pre_call_utils import clean_headers

        raw_headers = Headers(
            raw=[
                (b"x-goog-api-key", b"google-api-key-123"),
                (b"content-type", b"application/json"),
            ]
        )
        cleaned = clean_headers(raw_headers, forward_llm_provider_auth_headers=True)

        assert "x-goog-api-key" in cleaned
        assert cleaned["x-goog-api-key"] == "google-api-key-123"
        assert cleaned["content-type"] == "application/json"

    def test_clean_headers_preserves_oauth_regardless_of_forward_flag(self):
        """clean_headers should always preserve OAuth tokens regardless of forward_llm_provider_auth_headers."""
        from starlette.datastructures import Headers

        from litellm.proxy.litellm_pre_call_utils import clean_headers

        raw_headers = Headers(
            raw=[
                (b"authorization", f"Bearer {FAKE_OAUTH_TOKEN}".encode()),
                (b"content-type", b"application/json"),
            ]
        )
        
        # Should preserve OAuth even with flag=False
        cleaned_without_flag = clean_headers(raw_headers, forward_llm_provider_auth_headers=False)
        assert "authorization" in cleaned_without_flag
        assert cleaned_without_flag["authorization"] == f"Bearer {FAKE_OAUTH_TOKEN}"
        
        # Should also preserve OAuth with flag=True
        cleaned_with_flag = clean_headers(raw_headers, forward_llm_provider_auth_headers=True)
        assert "authorization" in cleaned_with_flag
        assert cleaned_with_flag["authorization"] == f"Bearer {FAKE_OAUTH_TOKEN}"

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
