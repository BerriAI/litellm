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
        from unittest.mock import patch

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
            {"ANTHROPIC_API_KEY": FAKE_REGULAR_KEY, "ANTHROPIC_AUTH_TOKEN": FAKE_AUTH_TOKEN},
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
