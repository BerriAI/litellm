"""
Tests for Anthropic authentication and environment variable handling in common_utils.

Verifies that:
- OAuth tokens (sk-ant-oat*) produce Authorization: Bearer headers with OAuth beta flags.
- Regular API keys produce x-api-key headers.
- ANTHROPIC_AUTH_TOKEN produces Authorization: Bearer headers,
  matching the official Anthropic SDK behavior.
- ANTHROPIC_BASE_URL is used as a fallback for base URL resolution.
- ANTHROPIC_API_KEY / ANTHROPIC_API_BASE take precedence over their aliases.
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

# Fake tokens for testing (not real secrets)
FAKE_OAUTH_TOKEN = "sk-ant-oat01-fake-token-for-testing-123456789abcdef"
FAKE_REGULAR_KEY = "sk-ant-api03-regular-key-for-testing-123456789"
FAKE_AUTH_TOKEN = "sk-ant-aut01-fake-auth-token-for-testing-123456789"


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

    def test_custom_api_base_uses_bearer_header(self):
        """Custom api_base and non-standard API key should produce Authorization: Bearer header when opted in."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        headers = config.get_anthropic_headers(
            api_key="my-custom-ollama-token",
            computer_tool_used=False,
            prompt_caching_set=False,
            pdf_used=False,
            is_vertex_request=False,
            api_base="https://ollama.com/",
            use_bearer_for_custom_base=True,
        )

        assert headers["authorization"] == "Bearer my-custom-ollama-token"
        assert "x-api-key" not in headers

    def test_custom_api_base_uses_bearer_header_already_starts_with_bearer(self):
        """If the key already starts with Bearer and Bearer opt-in is enabled, use it directly."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        headers = config.get_anthropic_headers(
            api_key="Bearer my-custom-ollama-token",
            computer_tool_used=False,
            prompt_caching_set=False,
            pdf_used=False,
            is_vertex_request=False,
            api_base="https://ollama.com/",
            use_bearer_for_custom_base=True,
        )

        assert headers["authorization"] == "Bearer my-custom-ollama-token"
        assert "x-api-key" not in headers

    def test_custom_api_base_uses_x_api_key_when_standard_key(self):
        """If the key is standard sk-ant- key, use x-api-key even with custom api_base."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        headers = config.get_anthropic_headers(
            api_key=FAKE_REGULAR_KEY,
            computer_tool_used=False,
            prompt_caching_set=False,
            pdf_used=False,
            is_vertex_request=False,
            api_base="https://ollama.com/",
        )

        assert headers["x-api-key"] == FAKE_REGULAR_KEY
        assert "authorization" not in headers

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

    def test_custom_api_base_via_param(self):
        """validate_environment uses Bearer when use_bearer_for_custom_base is set in litellm_params."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        headers = {}

        updated_headers = config.validate_environment(
            headers=headers,
            model="claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={"use_bearer_for_custom_base": True},
            api_key="custom-api-key",
            api_base="https://custom-gateway.com",
        )

        assert updated_headers["authorization"] == "Bearer custom-api-key"
        assert "x-api-key" not in updated_headers

    def test_custom_api_base_via_litellm_params(self):
        """validate_environment uses Bearer when api_base and use_bearer_for_custom_base are in litellm_params."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        headers = {}

        updated_headers = config.validate_environment(
            headers=headers,
            model="claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={"api_base": "https://custom-gateway.com", "use_bearer_for_custom_base": True},
            api_key="custom-api-key",
            api_base=None,
        )

        assert updated_headers["authorization"] == "Bearer custom-api-key"
        assert "x-api-key" not in updated_headers
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

    def test_passthrough_custom_api_base_uses_bearer_when_configured(self):
        from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
            AnthropicMessagesConfig,
        )

        config = AnthropicMessagesConfig()

        updated_headers, _ = config.validate_anthropic_messages_environment(
            headers={},
            model="anthropic/claude-fable-5",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={
                "api_base": "https://api.cloudflare.com/client/v4/accounts/test/ai",
                "use_bearer_for_custom_base": True,
            },
            api_key="custom-api-key",
            api_base=None,
        )

        assert updated_headers["authorization"] == "Bearer custom-api-key"
        assert "x-api-key" not in updated_headers


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
        cleaned_without_flag = clean_headers(
            raw_headers, forward_llm_provider_auth_headers=False
        )
        assert "authorization" in cleaned_without_flag
        assert cleaned_without_flag["authorization"] == f"Bearer {FAKE_OAUTH_TOKEN}"

        # Should also preserve OAuth with flag=True
        cleaned_with_flag = clean_headers(
            raw_headers, forward_llm_provider_auth_headers=True
        )
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

    def test_clean_headers_forwards_x_api_key_when_authenticated_with_litellm_key(self):
        """clean_headers should forward x-api-key when user authenticated with x-litellm-api-key and forward_llm_provider_auth_headers=True."""
        from starlette.datastructures import Headers

        from litellm.proxy.litellm_pre_call_utils import clean_headers

        raw_headers = Headers(
            raw=[
                (b"x-litellm-api-key", b"sk-litellm-proxy-key"),
                (b"x-api-key", b"sk-ant-api03-client-key"),
                (b"content-type", b"application/json"),
            ]
        )
        cleaned = clean_headers(
            raw_headers,
            forward_llm_provider_auth_headers=True,
            authenticated_with_header="x-litellm-api-key",
        )

        # x-api-key should be forwarded (it's a provider key, not used for auth)
        assert "x-api-key" in cleaned
        assert cleaned["x-api-key"] == "sk-ant-api03-client-key"
        # x-litellm-api-key should be excluded (special header)
        assert "x-litellm-api-key" not in cleaned
        assert cleaned["content-type"] == "application/json"

    def test_clean_headers_excludes_x_api_key_when_used_for_auth(self):
        """clean_headers should exclude x-api-key when it was used for LiteLLM authentication."""
        from starlette.datastructures import Headers

        from litellm.proxy.litellm_pre_call_utils import clean_headers

        raw_headers = Headers(
            raw=[
                (b"x-api-key", b"sk-litellm-proxy-key"),
                (b"content-type", b"application/json"),
            ]
        )
        cleaned = clean_headers(raw_headers, authenticated_with_header="x-api-key")

        # x-api-key should be excluded (was used for LiteLLM auth)
        assert "x-api-key" not in cleaned
        assert cleaned["content-type"] == "application/json"

    def test_clean_headers_forwards_x_api_key_when_authenticated_with_authorization(
        self,
    ):
        """clean_headers should forward x-api-key when user authenticated with Authorization header and forward_llm_provider_auth_headers=True."""
        from starlette.datastructures import Headers

        from litellm.proxy.litellm_pre_call_utils import clean_headers

        raw_headers = Headers(
            raw=[
                (b"authorization", b"Bearer sk-litellm-proxy-key"),
                (b"x-api-key", b"sk-ant-api03-client-key"),
                (b"content-type", b"application/json"),
            ]
        )
        cleaned = clean_headers(
            raw_headers,
            forward_llm_provider_auth_headers=True,
            authenticated_with_header="authorization",
        )

        # x-api-key should be forwarded (it's a provider key, not used for auth)
        assert "x-api-key" in cleaned
        assert cleaned["x-api-key"] == "sk-ant-api03-client-key"
        # authorization should be excluded (was used for auth, not OAuth)
        assert "authorization" not in cleaned
        assert cleaned["content-type"] == "application/json"

    def test_clean_headers_x_api_key_without_authenticated_header_param(self):
        """clean_headers should exclude x-api-key when authenticated_with_header is None."""
        from starlette.datastructures import Headers

        from litellm.proxy.litellm_pre_call_utils import clean_headers

        raw_headers = Headers(
            raw=[
                (b"x-api-key", b"sk-ant-api03-key"),
                (b"content-type", b"application/json"),
            ]
        )
        cleaned = clean_headers(raw_headers, authenticated_with_header=None)

        # x-api-key should be excluded (no authenticated_with_header means we can't determine)
        assert "x-api-key" not in cleaned
        assert cleaned["content-type"] == "application/json"

    def test_clean_headers_forwards_x_api_key_with_forward_flag_and_litellm_auth(
        self,
    ):
        """clean_headers should forward x-api-key when both forward_llm_provider_auth_headers=True
        and authenticated_with_header indicates different header was used for auth."""
        from starlette.datastructures import Headers

        from litellm.proxy.litellm_pre_call_utils import clean_headers

        raw_headers = Headers(
            raw=[
                (b"x-litellm-api-key", b"sk-litellm-proxy-key"),
                (b"x-api-key", b"sk-ant-api03-client-key"),
                (b"x-goog-api-key", b"google-key-123"),
                (b"content-type", b"application/json"),
            ]
        )
        cleaned = clean_headers(
            raw_headers,
            forward_llm_provider_auth_headers=True,
            authenticated_with_header="x-litellm-api-key",
        )

        # x-api-key should be forwarded (provider key, not used for auth)
        assert "x-api-key" in cleaned
        assert cleaned["x-api-key"] == "sk-ant-api03-client-key"
        # x-goog-api-key should also be forwarded (forward flag is True)
        assert "x-goog-api-key" in cleaned
        assert cleaned["x-goog-api-key"] == "google-key-123"
        # x-litellm-api-key should be excluded (special header)
        assert "x-litellm-api-key" not in cleaned
        assert cleaned["content-type"] == "application/json"

    def test_clean_headers_authorization_not_forwarded_when_used_for_litellm_auth(
        self,
    ):
        """Authorization Bearer (LiteLLM key) must never be forwarded to the LLM provider.

        When a user sends their LiteLLM key as 'Authorization: Bearer sk-1234' and
        forward_llm_provider_auth_headers=True, the Authorization header must be stripped
        — not sent to Anthropic as if it were an Anthropic API key.
        """
        from starlette.datastructures import Headers

        from litellm.proxy.litellm_pre_call_utils import clean_headers

        raw_headers = Headers(
            raw=[
                (b"authorization", b"Bearer sk-1234-litellm-proxy-key"),
                (b"x-api-key", b"sk-ant-api03-real-anthropic-key"),
                (b"content-type", b"application/json"),
            ]
        )
        # Authorization was the header used for LiteLLM auth
        cleaned = clean_headers(
            raw_headers,
            forward_llm_provider_auth_headers=True,
            authenticated_with_header="authorization",
        )

        # Authorization must NOT be forwarded — it was used for proxy auth
        assert "authorization" not in cleaned
        assert "Authorization" not in cleaned
        # x-api-key should be forwarded (it's the real Anthropic key, auth was via Authorization)
        assert "x-api-key" in cleaned
        assert cleaned["x-api-key"] == "sk-ant-api03-real-anthropic-key"
        assert cleaned["content-type"] == "application/json"

    def test_clean_headers_oauth_authorization_forwarded_when_not_used_for_litellm_auth(
        self,
    ):
        """OAuth Authorization header IS forwarded when x-litellm-api-key was used for proxy auth."""

        from starlette.datastructures import Headers

        from litellm.proxy.litellm_pre_call_utils import clean_headers

        oauth_token = "Bearer claude-gODtUFO8RoSnClWTtHKFJg"

        raw_headers = Headers(
            raw=[
                (b"x-litellm-api-key", b"sk-litellm-proxy-key"),
                (b"authorization", oauth_token.encode()),
                (b"content-type", b"application/json"),
            ]
        )
        # x-litellm-api-key was used for LiteLLM auth; Authorization carries the Anthropic OAuth token
        with patch(
            "litellm.llms.anthropic.common_utils.is_anthropic_oauth_key",
            return_value=True,
        ):
            cleaned = clean_headers(
                raw_headers,
                forward_llm_provider_auth_headers=True,
                authenticated_with_header="x-litellm-api-key",
            )

        # OAuth Authorization should be forwarded (not used for proxy auth)
        assert "authorization" in cleaned
        assert cleaned["authorization"] == oauth_token
        # Proxy key must be stripped
        assert "x-litellm-api-key" not in cleaned


class TestGetAnthropicHeadersWithAuthToken:
    """Tests for get_anthropic_headers with auth_token parameter."""

    def test_auth_token_uses_bearer_header(self):
        """auth_token should produce Authorization: Bearer header."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        headers = config.get_anthropic_headers(
            api_key=None,
            auth_token=FAKE_AUTH_TOKEN,
            computer_tool_used=False,
            prompt_caching_set=False,
            pdf_used=False,
            is_vertex_request=False,
        )

        assert headers["authorization"] == f"Bearer {FAKE_AUTH_TOKEN}"
        assert "x-api-key" not in headers
        # auth_token should NOT set OAuth-specific flags
        assert "anthropic-dangerous-direct-browser-access" not in headers

    def test_auth_token_includes_standard_headers(self):
        """auth_token path should include standard Anthropic headers."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        headers = config.get_anthropic_headers(
            api_key=None,
            auth_token=FAKE_AUTH_TOKEN,
            computer_tool_used=False,
            prompt_caching_set=False,
            pdf_used=False,
            is_vertex_request=False,
        )

        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["accept"] == "application/json"
        assert headers["content-type"] == "application/json"

    def test_api_key_takes_precedence_over_auth_token(self):
        """When both api_key and auth_token are provided, api_key wins."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        headers = config.get_anthropic_headers(
            api_key=FAKE_REGULAR_KEY,
            auth_token=FAKE_AUTH_TOKEN,
            computer_tool_used=False,
            prompt_caching_set=False,
            pdf_used=False,
            is_vertex_request=False,
        )

        assert headers["x-api-key"] == FAKE_REGULAR_KEY
        assert "authorization" not in headers


class TestValidateEnvironmentAuthToken:
    """Tests for validate_environment with auth_token resolution."""

    def test_auth_token_env_var_produces_bearer_header(self):
        """validate_environment should use Bearer auth when only ANTHROPIC_AUTH_TOKEN is set."""
        from unittest.mock import patch as mock_patch

        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        with mock_patch.dict(
            "os.environ",
            {"ANTHROPIC_AUTH_TOKEN": FAKE_AUTH_TOKEN},
            clear=True,
        ):
            headers = config.validate_environment(
                headers={},
                model="claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base=None,
            )

        assert headers["authorization"] == f"Bearer {FAKE_AUTH_TOKEN}"
        assert "x-api-key" not in headers
        assert "anthropic-dangerous-direct-browser-access" not in headers

    def test_api_key_param_takes_precedence_over_auth_token_env_var(self):
        """validate_environment should prefer explicit api_key over ANTHROPIC_AUTH_TOKEN."""
        from unittest.mock import patch as mock_patch

        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        with mock_patch.dict(
            "os.environ",
            {"ANTHROPIC_AUTH_TOKEN": FAKE_AUTH_TOKEN},
            clear=True,
        ):
            headers = config.validate_environment(
                headers={},
                model="claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},
                litellm_params={},
                api_key=FAKE_REGULAR_KEY,
                api_base=None,
            )

        assert headers["x-api-key"] == FAKE_REGULAR_KEY
        assert "authorization" not in headers

    def test_raises_when_no_credentials(self):
        """validate_environment should raise when neither API key nor auth token is available."""
        from unittest.mock import patch as mock_patch

        import pytest

        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        with mock_patch.dict("os.environ", {}, clear=True):
            with pytest.raises(
                Exception, match="ANTHROPIC_API_KEY.*ANTHROPIC_AUTH_TOKEN"
            ):
                config.validate_environment(
                    headers={},
                    model="claude-sonnet-4-5-20250929",
                    messages=[{"role": "user", "content": "Hello"}],
                    optional_params={},
                    litellm_params={},
                    api_key=None,
                    api_base=None,
                )

    def test_resolves_api_key_from_env_when_param_is_none(self):
        """validate_environment should resolve ANTHROPIC_API_KEY from env when api_key param is None."""
        from unittest.mock import patch as mock_patch

        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        with mock_patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": FAKE_REGULAR_KEY},
            clear=True,
        ):
            headers = config.validate_environment(
                headers={},
                model="claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base=None,
            )

        assert headers["x-api-key"] == FAKE_REGULAR_KEY
        assert "authorization" not in headers


class TestGetAuthToken:
    """Tests for AnthropicModelInfo.get_auth_token() static method."""

    def test_returns_env_var_value(self):
        """get_auth_token returns the ANTHROPIC_AUTH_TOKEN env var value."""
        from unittest.mock import patch as mock_patch

        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        with mock_patch.dict(
            "os.environ", {"ANTHROPIC_AUTH_TOKEN": FAKE_AUTH_TOKEN}, clear=True
        ):
            assert AnthropicModelInfo.get_auth_token() == FAKE_AUTH_TOKEN

    def test_returns_none_when_not_set(self):
        """get_auth_token returns None when ANTHROPIC_AUTH_TOKEN is not set."""
        from unittest.mock import patch as mock_patch

        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        with mock_patch.dict("os.environ", {}, clear=True):
            assert AnthropicModelInfo.get_auth_token() is None

    def test_explicit_param_takes_precedence(self):
        """Explicit auth_token param takes precedence over env var."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        explicit_token = "sk-ant-aut01-explicit-token-override-123456789"
        assert AnthropicModelInfo.get_auth_token(explicit_token) == explicit_token


class TestGetAuthHeader:
    """Tests for AnthropicModelInfo.get_auth_header() centralized helper."""

    def test_returns_x_api_key_when_api_key_provided(self):
        """Explicit api_key param should return x-api-key header."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        result = AnthropicModelInfo.get_auth_header(api_key=FAKE_REGULAR_KEY)
        assert result == {"x-api-key": FAKE_REGULAR_KEY}

    def test_returns_x_api_key_from_env(self):
        """ANTHROPIC_API_KEY env var should return x-api-key header."""
        from unittest.mock import patch as mock_patch

        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        with mock_patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": FAKE_REGULAR_KEY},
            clear=True,
        ):
            result = AnthropicModelInfo.get_auth_header()
            assert result == {"x-api-key": FAKE_REGULAR_KEY}

    def test_returns_bearer_from_auth_token_env(self):
        """ANTHROPIC_AUTH_TOKEN env var should return Authorization: Bearer header."""
        from unittest.mock import patch as mock_patch

        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        with mock_patch.dict(
            "os.environ",
            {"ANTHROPIC_AUTH_TOKEN": FAKE_AUTH_TOKEN},
            clear=True,
        ):
            result = AnthropicModelInfo.get_auth_header()
            assert result == {"authorization": f"Bearer {FAKE_AUTH_TOKEN}"}

    def test_api_key_takes_precedence_over_auth_token(self):
        """ANTHROPIC_API_KEY should take precedence over ANTHROPIC_AUTH_TOKEN."""
        from unittest.mock import patch as mock_patch

        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        with mock_patch.dict(
            "os.environ",
            {
                "ANTHROPIC_API_KEY": FAKE_REGULAR_KEY,
                "ANTHROPIC_AUTH_TOKEN": FAKE_AUTH_TOKEN,
            },
            clear=True,
        ):
            result = AnthropicModelInfo.get_auth_header()
            assert result == {"x-api-key": FAKE_REGULAR_KEY}

    def test_explicit_api_key_overrides_env_auth_token(self):
        """Explicit api_key param should override ANTHROPIC_AUTH_TOKEN env var."""
        from unittest.mock import patch as mock_patch

        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        with mock_patch.dict(
            "os.environ",
            {"ANTHROPIC_AUTH_TOKEN": FAKE_AUTH_TOKEN},
            clear=True,
        ):
            result = AnthropicModelInfo.get_auth_header(api_key=FAKE_REGULAR_KEY)
            assert result == {"x-api-key": FAKE_REGULAR_KEY}

    def test_returns_none_when_no_credentials(self):
        """Should return None when neither api_key nor auth_token is available."""
        from unittest.mock import patch as mock_patch

        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        with mock_patch.dict("os.environ", {}, clear=True):
            result = AnthropicModelInfo.get_auth_header()
            assert result is None

    def test_oauth_token_uses_bearer_not_x_api_key(self):
        """OAuth token (sk-ant-oat*) should return Authorization: Bearer, not x-api-key."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        result = AnthropicModelInfo.get_auth_header(api_key=FAKE_OAUTH_TOKEN)
        assert result == {"authorization": f"Bearer {FAKE_OAUTH_TOKEN}"}

    def test_oauth_token_from_env_uses_bearer(self):
        """OAuth token in ANTHROPIC_API_KEY env var should return Authorization: Bearer."""
        from unittest.mock import patch as mock_patch

        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        with mock_patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": FAKE_OAUTH_TOKEN},
            clear=True,
        ):
            result = AnthropicModelInfo.get_auth_header()
            assert result == {"authorization": f"Bearer {FAKE_OAUTH_TOKEN}"}

    def test_custom_api_base_get_auth_header_uses_bearer(self):
        """Non-standard API key and custom api_base returns Bearer when use_bearer_for_custom_base=True."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        result = AnthropicModelInfo.get_auth_header(api_key="my-custom-key", api_base="https://custom-gateway.com", use_bearer_for_custom_base=True)
        assert result == {"authorization": "Bearer my-custom-key"}

    def test_custom_api_base_get_auth_header_uses_x_api_key_when_standard(self):
        """Standard sk-ant- key with custom api_base should still return x-api-key."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        result = AnthropicModelInfo.get_auth_header(api_key=FAKE_REGULAR_KEY, api_base="https://custom-gateway.com")
        assert result == {"x-api-key": FAKE_REGULAR_KEY}


class TestGetApiBaseFallbackChain:
    """Tests for AnthropicModelInfo.get_api_base() fallback to ANTHROPIC_BASE_URL."""

    def test_explicit_param_takes_precedence(self):
        """Explicit api_base param takes precedence over all env vars."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        assert (
            AnthropicModelInfo.get_api_base("https://explicit.example.com")
            == "https://explicit.example.com"
        )

    def test_defaults_to_anthropic_api(self):
        """get_api_base returns the default Anthropic API base when no env vars are set."""
        from unittest.mock import patch as mock_patch

        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        with mock_patch.dict("os.environ", {}, clear=True):
            assert AnthropicModelInfo.get_api_base() == "https://api.anthropic.com"

    def test_api_base_env_preferred_over_base_url_env(self):
        """ANTHROPIC_API_BASE takes precedence over ANTHROPIC_BASE_URL."""
        from unittest.mock import patch as mock_patch

        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        with mock_patch.dict(
            "os.environ",
            {
                "ANTHROPIC_API_BASE": "https://api-base.example.com",
                "ANTHROPIC_BASE_URL": "https://base-url.example.com",
            },
            clear=True,
        ):
            assert AnthropicModelInfo.get_api_base() == "https://api-base.example.com"

    def test_falls_back_to_base_url_env(self):
        """get_api_base falls back to ANTHROPIC_BASE_URL when ANTHROPIC_API_BASE is not set."""
        from unittest.mock import patch as mock_patch

        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        with mock_patch.dict(
            "os.environ",
            {"ANTHROPIC_BASE_URL": "https://base-url.example.com"},
            clear=True,
        ):
            assert AnthropicModelInfo.get_api_base() == "https://base-url.example.com"


class TestPassthroughAuthToken:
    """Tests for passthrough messages endpoint with ANTHROPIC_AUTH_TOKEN."""

    def test_passthrough_auth_token_uses_bearer_header(self):
        """Passthrough endpoint should use Bearer auth when only ANTHROPIC_AUTH_TOKEN is set."""
        from unittest.mock import patch as mock_patch

        from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
            AnthropicMessagesConfig,
        )

        config = AnthropicMessagesConfig()
        with mock_patch.dict(
            "os.environ", {"ANTHROPIC_AUTH_TOKEN": FAKE_AUTH_TOKEN}, clear=True
        ):
            updated_headers, _ = config.validate_anthropic_messages_environment(
                headers={},
                model="claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base=None,
            )

        assert updated_headers["authorization"] == f"Bearer {FAKE_AUTH_TOKEN}"
        assert "x-api-key" not in updated_headers
        assert "anthropic-dangerous-direct-browser-access" not in updated_headers

    def test_passthrough_api_key_takes_precedence(self):
        """Passthrough endpoint should prefer ANTHROPIC_API_KEY over ANTHROPIC_AUTH_TOKEN."""
        from unittest.mock import patch as mock_patch

        from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
            AnthropicMessagesConfig,
        )

        config = AnthropicMessagesConfig()
        with mock_patch.dict(
            "os.environ",
            {
                "ANTHROPIC_API_KEY": FAKE_REGULAR_KEY,
                "ANTHROPIC_AUTH_TOKEN": FAKE_AUTH_TOKEN,
            },
            clear=True,
        ):
            updated_headers, _ = config.validate_anthropic_messages_environment(
                headers={},
                model="claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base=None,
            )

        assert updated_headers["x-api-key"] == FAKE_REGULAR_KEY
        assert "authorization" not in updated_headers

    def test_passthrough_get_complete_url_honours_base_url_env(self):
        """get_complete_url should use ANTHROPIC_BASE_URL when api_base is None."""
        from unittest.mock import patch as mock_patch

        from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
            AnthropicMessagesConfig,
        )

        config = AnthropicMessagesConfig()
        with mock_patch.dict(
            "os.environ",
            {"ANTHROPIC_BASE_URL": "https://custom.example.com"},
            clear=True,
        ):
            url = config.get_complete_url(
                api_base=None,
                api_key=FAKE_REGULAR_KEY,
                model="claude-sonnet-4-5-20250929",
                optional_params={},
                litellm_params={},
            )

        assert url == "https://custom.example.com/v1/messages"


class TestAnthropicThinkingSignatureSelfHeal:
    """Helpers for retrying after invalid encrypted thinking signatures."""

    def test_is_anthropic_invalid_thinking_signature_error_positive(self):
        from litellm.llms.anthropic.common_utils import (
            is_anthropic_invalid_thinking_signature_error,
        )

        raw = (
            '{"type":"error","error":{"type":"invalid_request_error",'
            '"message":"messages.3.content.3: Invalid `signature` in `thinking` block"},'
            '"request_id":"req_011Ca2EtQDxp7x6RGUY2jVn9"}'
        )
        assert is_anthropic_invalid_thinking_signature_error(raw) is True

    def test_is_anthropic_invalid_thinking_signature_error_negative(self):
        from litellm.llms.anthropic.common_utils import (
            is_anthropic_invalid_thinking_signature_error,
        )

        assert is_anthropic_invalid_thinking_signature_error("") is False
        assert (
            is_anthropic_invalid_thinking_signature_error("rate limit exceeded")
            is False
        )

    def test_strip_thinking_blocks_from_anthropic_messages(self):
        from litellm.llms.anthropic.common_utils import (
            strip_thinking_blocks_from_anthropic_messages,
        )

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "plan", "signature": "sig"},
                    {"type": "text", "text": "hello"},
                ],
            },
        ]
        out = strip_thinking_blocks_from_anthropic_messages(messages)
        assert len(out) == 2
        assert out[0] == messages[0]
        assert len(out[1]["content"]) == 1
        assert out[1]["content"][0]["type"] == "text"
        assert messages[1]["content"][0]["type"] == "thinking"

    def test_strip_thinking_blocks_drops_message_when_only_thinking_blocks(self):
        from litellm.llms.anthropic.common_utils import (
            strip_thinking_blocks_from_anthropic_messages,
        )

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "plan", "signature": "sig"},
                ],
            },
        ]
        out = strip_thinking_blocks_from_anthropic_messages(messages)
        assert len(out) == 1
        assert out[0]["role"] == "user"

    def test_strip_thinking_blocks_from_anthropic_messages_request_dict(self):
        from litellm.llms.anthropic.common_utils import (
            strip_thinking_blocks_from_anthropic_messages_request_dict,
        )

        data = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "x",
                            "signature": "y",
                        },
                    ],
                }
            ],
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        }
        strip_thinking_blocks_from_anthropic_messages_request_dict(data)
        assert "thinking" not in data
        assert data["messages"] == []

    def test_strip_empty_text_blocks_from_anthropic_messages(self):
        """Covers #22930.  The core regression scenario: an assistant message
        with an empty text block alongside ``tool_use`` loses the empty block
        and keeps the ``tool_use``; a whole message that reduces to no blocks
        is dropped; whitespace-only text counts as empty; the caller's list
        is never mutated."""
        from litellm.llms.anthropic.common_utils import (
            strip_empty_text_blocks_from_anthropic_messages,
        )

        tu = {"type": "tool_use", "id": "x", "name": "Bash", "input": {}}
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "  \n "}, tu]},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        ]
        out = strip_empty_text_blocks_from_anthropic_messages(msgs)
        assert len(out) == 2 and out[0] is msgs[0]
        assert [b["type"] for b in out[1]["content"]] == ["tool_use"]
        assert len(msgs[1]["content"]) == 2  # caller's content unchanged

    def test_strip_empty_text_blocks_preserves_thinking_blocks(self):
        from litellm.llms.anthropic.common_utils import (
            strip_empty_text_blocks_from_anthropic_messages,
        )

        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "plan", "signature": "sig"},
                    {"type": "text", "text": ""},
                ],
            }
        ]
        out = strip_empty_text_blocks_from_anthropic_messages(msgs)
        assert [b["type"] for b in out[0]["content"]] == ["thinking"]

    def test_strip_empty_text_blocks_treats_null_text_as_empty(self):
        from litellm.llms.anthropic.common_utils import (
            strip_empty_text_blocks_from_anthropic_messages,
        )

        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": None},
                    {"type": "tool_result", "tool_use_id": "x", "content": "y"},
                ],
            }
        ]
        out = strip_empty_text_blocks_from_anthropic_messages(msgs)
        assert [b["type"] for b in out[0]["content"]] == ["tool_result"]

    def test_strip_empty_text_blocks_treats_missing_text_key_as_empty(self):
        from litellm.llms.anthropic.common_utils import (
            strip_empty_text_blocks_from_anthropic_messages,
        )

        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text"},
                    {"type": "tool_result", "tool_use_id": "x", "content": "y"},
                ],
            }
        ]
        out = strip_empty_text_blocks_from_anthropic_messages(msgs)
        assert [b["type"] for b in out[0]["content"]] == ["tool_result"]

    def test_strip_empty_text_blocks_leaves_non_empty_text_alone(self):
        from litellm.llms.anthropic.common_utils import (
            strip_empty_text_blocks_from_anthropic_messages,
        )

        msgs = [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]
        out = strip_empty_text_blocks_from_anthropic_messages(msgs)
        assert out[0] is msgs[0]  # untouched messages keep identity

    def test_strip_empty_text_blocks_treats_non_string_text_value_as_empty(self):
        from litellm.llms.anthropic.common_utils import (
            strip_empty_text_blocks_from_anthropic_messages,
        )

        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": 123},
                    {"type": "tool_result", "tool_use_id": "x", "content": "y"},
                ],
            }
        ]
        out = strip_empty_text_blocks_from_anthropic_messages(msgs)
        assert [b["type"] for b in out[0]["content"]] == ["tool_result"]

    def test_sanitize_tool_use_ids_in_anthropic_messages(self):
        from litellm.llms.anthropic.common_utils import (
            sanitize_tool_use_ids_in_anthropic_messages,
        )

        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "functions.Bash:0",
                        "name": "Bash",
                        "input": {},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "functions.Bash:0",
                        "content": "ok",
                    }
                ],
            },
        ]
        out = sanitize_tool_use_ids_in_anthropic_messages(msgs)
        assert out[0]["content"][0]["id"] == "functions_Bash_0"
        assert out[1]["content"][0]["tool_use_id"] == "functions_Bash_0"
        assert msgs[0]["content"][0]["id"] == "functions.Bash:0"

    def test_normalize_anthropic_tool_use_id_strips_thought_signature(self):
        from litellm.litellm_core_utils.prompt_templates.factory import (
            THOUGHT_SIGNATURE_SEPARATOR,
        )
        from litellm.llms.anthropic.common_utils import normalize_anthropic_tool_use_id

        base = "call_abc123"
        sig = "CiIBDDnWx+/a=="
        assert (
            normalize_anthropic_tool_use_id(f"{base}{THOUGHT_SIGNATURE_SEPARATOR}{sig}")
            == base
        )

    def test_anthropic_messages_config_http_retry_helpers(self):
        import httpx

        from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
            AnthropicMessagesConfig,
        )

        config = AnthropicMessagesConfig()
        assert config.max_retry_on_anthropic_messages_http_error == 2

        req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        err_text = (
            '{"type":"error","error":{"type":"invalid_request_error",'
            '"message":"messages.3.content.3: Invalid `signature` in `thinking` block"},'
            '"request_id":"req_011Ca2EtQDxp7x6RGUY2jVn9"}'
        )
        resp = httpx.Response(400, request=req, text=err_text)
        err = httpx.HTTPStatusError("bad", request=req, response=resp)
        assert config.should_retry_anthropic_messages_on_http_error(err, {}) is True

        resp_bad = httpx.Response(400, request=req, text="rate limit exceeded")
        err_bad = httpx.HTTPStatusError("bad", request=req, response=resp_bad)
        assert (
            config.should_retry_anthropic_messages_on_http_error(err_bad, {}) is False
        )

        resp_500 = httpx.Response(500, request=req, text=err_text)
        err_500 = httpx.HTTPStatusError("bad", request=req, response=resp_500)
        assert (
            config.should_retry_anthropic_messages_on_http_error(err_500, {}) is False
        )

        data = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "x",
                            "signature": "y",
                        },
                    ],
                }
            ],
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        }
        config.transform_anthropic_messages_request_on_http_error(err, data)
        assert "thinking" not in data
        assert data["messages"] == []


@pytest.fixture
def local_model_cost_map(monkeypatch):
    """Force the bundled backup cost map so detection doesn't depend on the
    network-fetched ``main`` copy (which lacks this branch's flags until merge)."""
    import litellm

    original = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original
        litellm.get_model_info.cache_clear()


class TestClaudeOpus48AdaptiveThinking:
    """Opus 4.8 requires adaptive thinking (``thinking.type='adaptive'`` +
    ``output_config.effort``). Detection is driven by the
    ``supports_adaptive_thinking`` cost-map flag, resolved through provider
    prefixes. Before the fix the Bedrock entries lacked the flag and the lookup
    didn't strip the ``us.anthropic.``/``invoke/`` prefixes, so a
    ``bedrock/us.anthropic.claude-opus-4-8`` call sent the legacy
    ``thinking.type='enabled'`` shape and Bedrock rejected it (issue #29188)."""

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-8",
            "anthropic/claude-opus-4-8",
            "anthropic.claude-opus-4-8",
            "bedrock/us.anthropic.claude-opus-4-8",
            "bedrock/invoke/us.anthropic.claude-opus-4-8",
            "bedrock/eu.anthropic.claude-opus-4-8",
            "vertex_ai/claude-opus-4-8",
            "azure_ai/claude-opus-4-8",
            "anthropic.claude-opus-4-8-20251201-v1:0",
            "bedrock/invoke/global.anthropic.claude-opus-4-8-20251201-v1:0",
        ],
    )
    def test_adaptive_thinking_detected_for_opus_4_8(self, local_model_cost_map, model):
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        assert AnthropicModelInfo._is_adaptive_thinking_model(model, "anthropic") is True

    def test_resolver_reads_flag_through_bedrock_invoke_prefix(
        self, local_model_cost_map
    ):
        """The resolver fix: ``bedrock/invoke/...`` resolves to the flagged
        Bedrock entry. Pure ``_supports_factory`` without prefix-stripping
        returns False here, which is why the data-only fix alone was not enough."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        assert (
            AnthropicModelInfo._supports_model_capability(
                "bedrock/invoke/us.anthropic.claude-opus-4-8",
                "supports_adaptive_thinking",
                "anthropic",
            )
            is True
        )

    @pytest.mark.parametrize(
        "model",
        [
            "claude-fable-5",
            "anthropic.claude-fable-5",
            "us.anthropic.claude-fable-5",
            "bedrock/invoke/us.anthropic.claude-fable-5",
            "vertex_ai/claude-fable-5",
        ],
    )
    def test_adaptive_thinking_detected_for_fable_5(self, local_model_cost_map, model):
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        assert AnthropicModelInfo._is_adaptive_thinking_model(model, "anthropic") is True

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-6",
            "us.anthropic.claude-opus-4-7",
            "bedrock/invoke/us.anthropic.claude-opus-4-7",
            "bedrock/invoke/global.anthropic.claude-opus-4-7-v1:0",
            "global.anthropic.claude-sonnet-4-6-v1:0",
            "bedrock/invoke/us.anthropic.claude-opus-4-6-v1:0",
            "anthropic.claude-opus-4-6-v1",
            "bedrock/us.anthropic.claude-sonnet-4-6",
            "us.anthropic.claude-sonnet-4-6",
            "vertex_ai/claude-opus-4-6",
            "azure_ai/claude-sonnet-4-6",
            "claude-sonnet-4-6-20260219",
            "us.anthropic.claude-sonnet-4-6-20251101-v1:0",
            "bedrock/invoke/us.anthropic.claude-sonnet-4-6-20251101-v1:0",
            "claude-sonnet-4.6",
        ],
    )
    def test_adaptive_thinking_detected_for_opus_4_6_4_7_and_sonnet_4_6(
        self, local_model_cost_map, model
    ):
        """Opus 4.6/4.7 and Sonnet 4.6 carry the ``supports_adaptive_thinking`` flag,
        so detection holds purely from the cost map with no version-rule
        fallback. Each alias form the Bedrock/anthropic paths see resolves to a flagged
        base entry through candidate normalization: provider/region prefixes, a
        Bedrock ``-v1:0`` version suffix (stripped fully for 4.7/4.8 keys or to ``-v1``
        for the 4.6 key), a dated release suffix (``-20260219``), a combined
        ``-<date>-v1:0`` suffix (the real Bedrock id shape), and a dotted family
        version (``4.6`` -> ``4-6``)."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        assert AnthropicModelInfo._is_adaptive_thinking_model(model, "anthropic") is True

    @pytest.mark.parametrize(
        "model",
        [
            "us.anthropic.claude-fable-preview",
            "claude-fable-preview",
        ],
    )
    def test_unmapped_aliases_without_parseable_version_stay_non_adaptive(
        self, local_model_cost_map, model
    ):
        """An alias absent from the map, not matched by any ``fallback_generalizations``
        rule, and without any parseable family version stays non-adaptive. ``fable``
        without a major version matches neither the core-family 4.6+ gate nor the
        family-agnostic 5+ gate, so neither the cost map nor the declarative rule marks
        it adaptive."""
        import litellm
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        assert model not in litellm.model_cost
        assert AnthropicModelInfo._is_adaptive_thinking_model(model, "anthropic") is False

    @pytest.mark.parametrize(
        "model",
        [
            "bedrock/invoke/us.anthropic.claude-opus-4-9",
            "vertex_ai/claude-sonnet-5-0",
            "us.anthropic.claude-opus-5-2",
            "us.anthropic.claude-opus-6-1",
            "claude-opus-5-0",
            "claude-opus-4-10",
            "claude-opus-4-8-some-future-suffix",
            "claude-fable-5-preview",
            "us.anthropic.claude-fable-5-preview",
        ],
    )
    def test_adaptive_thinking_version_fallback_for_unmapped_high_versions(
        self, local_model_cost_map, model
    ):
        """Provider-prefixed or suffixed Claude names that resolve to no mapped entry
        still resolve to adaptive when the id carries claude-<family>- at version 4.6
        or higher, bare 5+ majors included. The version gate is the declarative
        ``claude-adaptive-thinking`` rule, so 5.x, 6.x and any later family are covered
        with no code change."""
        import litellm
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        assert model not in litellm.model_cost
        assert AnthropicModelInfo._is_adaptive_thinking_model(model, "anthropic") is True

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-0",
            "us.anthropic.claude-opus-4-0",
            "bedrock/invoke/us.anthropic.claude-opus-4-5",
            "us.anthropic.claude-opus-4-20250514",
        ],
    )
    def test_adaptive_thinking_not_detected_for_unmapped_low_versions(
        self, local_model_cost_map, model
    ):
        """Unmapped Claude names below 4.6 stay non-adaptive through the declarative path.
        The eight-digit dated Opus 4.0 id (``...-4-20250514``) is the date-safety case: the
        version rule caps the minor at two digits, so the date is not misread as a >= 4.6
        minor. The anchored pricing rule still resolves these for cost, just without the
        adaptive flag."""
        import litellm
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        assert model not in litellm.model_cost
        assert AnthropicModelInfo._is_adaptive_thinking_model(model, "anthropic") is False

    @pytest.mark.parametrize(
        "model",
        ["claude-opus-4-5", "claude-3-7-sonnet", "claude-3-5-haiku-20241022"],
    )
    def test_non_adaptive_models_not_detected(self, local_model_cost_map, model):
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        assert AnthropicModelInfo._is_adaptive_thinking_model(model, "anthropic") is False


class TestDefaultSuffixAdaptiveThinking:
    """@default-suffixed Vertex AI model names (e.g. vertex_ai/claude-opus-4-8@default)
    must resolve as adaptive thinking. Before the fix, _model_map_lookup_candidates
    never stripped the @default suffix, so the lookup fell through to the bare
    model name without @default, which may or may not have the flag, and for
    provider-prefixed forms the lookup always missed (issue #31760)."""

    @pytest.mark.parametrize(
        "model",
        [
            "vertex_ai/claude-opus-4-8@default",
            "vertex_ai/claude-sonnet-4-6@default",
            "vertex_ai/claude-opus-4-7@default",
            "vertex_ai/claude-opus-4-6@default",
            "vertex_ai/claude-fable-5@default",
        ],
    )
    def test_default_suffix_models_are_adaptive_thinking(
        self, local_model_cost_map, model: str
    ) -> None:
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        assert AnthropicModelInfo._is_adaptive_thinking_model(model, "anthropic") is True, (
            f"{model} not classified as adaptive thinking. "
            "Check _model_map_lookup_candidates strips @default suffix."
        )

    @pytest.mark.parametrize(
        "model,expected_bare",
        [
            ("vertex_ai/claude-opus-4-8@default", "claude-opus-4-8"),
            ("vertex_ai/claude-sonnet-4-6@default", "claude-sonnet-4-6"),
        ],
    )
    def test_lookup_candidates_include_bare_name(
        self, model: str, expected_bare: str
    ) -> None:
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        candidates = AnthropicModelInfo._model_map_lookup_candidates(model)
        assert expected_bare in candidates, (
            f"Expected '{expected_bare}' in candidates for '{model}', got: {candidates}"
        )


class TestCapabilityProbeUsesCallerProvider:
    """``_supports_model_capability`` must probe under the caller's real provider
    namespace instead of a pinned ``"anthropic"``. With the pin, the exact Bedrock
    cost-map entry for ``global.anthropic.claude-opus-4-8`` was rejected by the
    provider match and the anthropic-scoped fallback rule answered instead, so
    flipping ``supports_adaptive_thinking`` on the exact entry changed nothing and
    the documented "exact entry beats rule" precedence was silently violated."""

    BEDROCK_MODEL = "global.anthropic.claude-opus-4-8"

    def test_exact_bedrock_entry_flag_is_authoritative_for_bedrock_caller(
        self, local_model_cost_map, monkeypatch
    ):
        import litellm
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        assert (
            AnthropicModelInfo._is_adaptive_thinking_model(self.BEDROCK_MODEL, "bedrock")
            is True
        )

        monkeypatch.setitem(
            litellm.model_cost[self.BEDROCK_MODEL], "supports_adaptive_thinking", False
        )
        litellm.get_model_info.cache_clear()

        assert (
            AnthropicModelInfo._is_adaptive_thinking_model(self.BEDROCK_MODEL, "bedrock")
            is False
        )

    def test_native_anthropic_probe_still_reads_anthropic_entry(
        self, local_model_cost_map, monkeypatch
    ):
        import litellm
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        monkeypatch.setitem(
            litellm.model_cost[self.BEDROCK_MODEL], "supports_adaptive_thinking", False
        )
        litellm.get_model_info.cache_clear()

        assert (
            AnthropicModelInfo._is_adaptive_thinking_model("claude-opus-4-8", "anthropic")
            is True
        )
