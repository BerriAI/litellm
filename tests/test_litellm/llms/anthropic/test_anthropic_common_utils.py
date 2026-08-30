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

import json
import os
import sys
import threading
from types import SimpleNamespace
from typing import Final
from unittest.mock import patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from litellm.proxy._types import SpecialHeaders  # noqa: E402  # sys.path must be patched before importing litellm

# Fake tokens for testing (not real secrets)
FAKE_OAUTH_TOKEN = "sk-ant-oat01-fake-token-for-testing-123456789abcdef"
FAKE_REGULAR_KEY = "sk-ant-api03-regular-key-for-testing-123456789"
FAKE_AUTH_TOKEN = "sk-ant-aut01-fake-auth-token-for-testing-123456789"


class TestOptionallyHandleAnthropicOAuth:
    """Tests for optionally_handle_anthropic_oauth function."""

    @pytest.mark.parametrize("header_name", ["authorization", "Authorization", "AUTHORIZATION"])
    def test_oauth_token_in_authorization_header(self, header_name):
        """OAuth token in Authorization header should be detected and headers set correctly."""
        from litellm.llms.anthropic.common_utils import (
            optionally_handle_anthropic_oauth,
        )

        headers = {header_name: f"Bearer {FAKE_OAUTH_TOKEN}"}
        updated_headers, extracted_api_key = optionally_handle_anthropic_oauth(headers, None)

        assert extracted_api_key == FAKE_OAUTH_TOKEN
        assert updated_headers["anthropic-beta"] == "oauth-2025-04-20"
        assert updated_headers["anthropic-dangerous-direct-browser-access"] == "true"
        assert "x-api-key" not in updated_headers
        assert [name for name in updated_headers if name.lower() == "authorization"] == ["authorization"]
        assert updated_headers["authorization"] == f"Bearer {FAKE_OAUTH_TOKEN}"

    @pytest.mark.parametrize("api_key_header_name", ["x-api-key", "X-Api-Key"])
    def test_oauth_removes_x_api_key_any_casing(self, api_key_header_name):
        """When OAuth wins, a client x-api-key header is removed whatever its casing."""
        from litellm.llms.anthropic.common_utils import (
            optionally_handle_anthropic_oauth,
        )

        headers = {api_key_header_name: FAKE_REGULAR_KEY, "Authorization": f"Bearer {FAKE_OAUTH_TOKEN}"}
        updated_headers, extracted_api_key = optionally_handle_anthropic_oauth(headers, None)

        assert extracted_api_key == FAKE_OAUTH_TOKEN
        assert [name for name in updated_headers if name.lower() == "x-api-key"] == []
        assert [name for name in updated_headers if name.lower() == "authorization"] == ["authorization"]
        assert updated_headers["authorization"] == f"Bearer {FAKE_OAUTH_TOKEN}"

    def test_oauth_token_in_api_key_directly(self):
        """OAuth token passed as api_key should set Authorization: Bearer header."""
        from litellm.llms.anthropic.common_utils import (
            optionally_handle_anthropic_oauth,
        )

        headers = {}
        updated_headers, returned_api_key = optionally_handle_anthropic_oauth(headers, FAKE_OAUTH_TOKEN)

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
        updated_headers, _ = optionally_handle_anthropic_oauth(headers, FAKE_OAUTH_TOKEN)

        assert "x-api-key" not in updated_headers
        assert updated_headers["authorization"] == f"Bearer {FAKE_OAUTH_TOKEN}"

    def test_regular_api_key_unchanged(self):
        """Regular API keys (non-OAuth) should pass through unmodified."""
        from litellm.llms.anthropic.common_utils import (
            optionally_handle_anthropic_oauth,
        )

        headers = {}
        updated_headers, returned_api_key = optionally_handle_anthropic_oauth(headers, FAKE_REGULAR_KEY)

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
        updated_headers, returned_api_key = optionally_handle_anthropic_oauth(headers, FAKE_REGULAR_KEY)

        assert returned_api_key == FAKE_REGULAR_KEY
        assert "anthropic-dangerous-direct-browser-access" not in updated_headers

    def test_none_api_key_no_error(self):
        """None api_key with empty headers should not raise errors."""
        from litellm.llms.anthropic.common_utils import (
            optionally_handle_anthropic_oauth,
        )

        headers = {}
        updated_headers, returned_api_key = optionally_handle_anthropic_oauth(headers, None)

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
        as a ProviderSpecificHeader scoped to Anthropic and nothing else."""
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
        assert psh["custom_llm_provider"] == "anthropic"
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
        reach Anthropic."""
        from litellm.litellm_core_utils.get_provider_specific_headers import (
            ProviderSpecificHeaderUtils,
        )
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
        anthropic_headers = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header=data["provider_specific_header"],
            custom_llm_provider="anthropic",
        )
        assert anthropic_headers["authorization"] == f"Bearer {FAKE_OAUTH_TOKEN}"
        assert anthropic_headers["anthropic-beta"] == "oauth-2025-04-20"

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
            with pytest.raises(Exception, match=r"ANTHROPIC_API_KEY.*ANTHROPIC_AUTH_TOKEN"):
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

        with mock_patch.dict("os.environ", {"ANTHROPIC_AUTH_TOKEN": FAKE_AUTH_TOKEN}, clear=True):
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
        """OAuth token (sk-ant-oat*) should return Authorization: Bearer with the
        mandatory oauth beta, not x-api-key."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        result = AnthropicModelInfo.get_auth_header(api_key=FAKE_OAUTH_TOKEN)
        assert result == {
            "authorization": f"Bearer {FAKE_OAUTH_TOKEN}",
            "anthropic-beta": "oauth-2025-04-20",
        }

    def test_oauth_token_from_env_uses_bearer(self):
        """OAuth token in ANTHROPIC_API_KEY env var should return Authorization: Bearer
        with the mandatory oauth beta."""
        from unittest.mock import patch as mock_patch

        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        with mock_patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": FAKE_OAUTH_TOKEN},
            clear=True,
        ):
            result = AnthropicModelInfo.get_auth_header()
            assert result == {
                "authorization": f"Bearer {FAKE_OAUTH_TOKEN}",
                "anthropic-beta": "oauth-2025-04-20",
            }

    def test_custom_api_base_get_auth_header_uses_bearer(self):
        """Non-standard API key and custom api_base returns Bearer when use_bearer_for_custom_base=True."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        result = AnthropicModelInfo.get_auth_header(
            api_key="my-custom-key", api_base="https://custom-gateway.com", use_bearer_for_custom_base=True
        )
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

        assert AnthropicModelInfo.get_api_base("https://explicit.example.com") == "https://explicit.example.com"

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
        with mock_patch.dict("os.environ", {"ANTHROPIC_AUTH_TOKEN": FAKE_AUTH_TOKEN}, clear=True):
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

    def test_passthrough_missing_credentials_raises_authentication_error(self):
        """Passthrough endpoint should raise locally instead of forwarding an unauthenticated request."""
        from unittest.mock import patch as mock_patch

        import litellm
        from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
            AnthropicMessagesConfig,
        )

        config = AnthropicMessagesConfig()
        with mock_patch.dict("os.environ", {}, clear=True):
            with pytest.raises(litellm.AuthenticationError, match="Missing Anthropic API Key"):
                config.validate_anthropic_messages_environment(
                    headers={},
                    model="claude-sonnet-4-5-20250929",
                    messages=[{"role": "user", "content": "Hello"}],
                    optional_params={},
                    litellm_params={},
                    api_key=None,
                    api_base=None,
                )

    @pytest.mark.parametrize("header_name", ["x-api-key", "X-Api-Key", "X-API-KEY"])
    def test_passthrough_client_x_api_key_header_is_kept(self, header_name):
        """A client-forwarded x-api-key header, whatever its casing, should satisfy validation without env credentials."""
        from unittest.mock import patch as mock_patch

        from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
            AnthropicMessagesConfig,
        )

        config = AnthropicMessagesConfig()
        with mock_patch.dict("os.environ", {}, clear=True):
            updated_headers, _ = config.validate_anthropic_messages_environment(
                headers={header_name: FAKE_REGULAR_KEY},
                model="claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base=None,
            )

        assert [name for name in updated_headers if name.lower() == "x-api-key"] == [header_name]
        assert updated_headers[header_name] == FAKE_REGULAR_KEY

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
    """Helpers for retrying after invalid thinking blocks in replayed history:
    invalid encrypted signatures, and blocks with empty thinking text."""

    def test_is_anthropic_invalid_thinking_block_error_positive(self):
        from litellm.llms.anthropic.common_utils import (
            is_anthropic_invalid_thinking_block_error,
        )

        raw = (
            '{"type":"error","error":{"type":"invalid_request_error",'
            '"message":"messages.3.content.3: Invalid `signature` in `thinking` block"},'
            '"request_id":"req_011Ca2EtQDxp7x6RGUY2jVn9"}'
        )
        assert is_anthropic_invalid_thinking_block_error(raw) is True

    def test_is_anthropic_invalid_thinking_block_error_positive_bedrock(self):
        from litellm.llms.anthropic.common_utils import (
            is_anthropic_invalid_thinking_block_error,
        )

        # Real user-reported Bedrock scenario
        raw = '{"message":"messages.2.content.0.thinking.signature.str: Input should be a valid string"}'
        assert is_anthropic_invalid_thinking_block_error(raw) is True

    def test_is_anthropic_invalid_thinking_block_error_positive_vertex(self):
        from litellm.llms.anthropic.common_utils import (
            is_anthropic_invalid_thinking_block_error,
        )

        raw = "messages.4.content.1.thinking.signature.str: Input should be a valid string"
        assert is_anthropic_invalid_thinking_block_error(raw) is True

    def test_is_anthropic_invalid_thinking_block_error_negative(self):
        from litellm.llms.anthropic.common_utils import (
            is_anthropic_invalid_thinking_block_error,
        )

        assert is_anthropic_invalid_thinking_block_error("") is False
        assert is_anthropic_invalid_thinking_block_error("rate limit exceeded") is False
        assert is_anthropic_invalid_thinking_block_error("invalid_request_error: model not found") is False
        assert is_anthropic_invalid_thinking_block_error("thinking signature is malformed") is False

    def test_is_anthropic_invalid_thinking_block_error_positive_empty_thinking(self):
        """LIT-6357: replayed history holding {"type": "thinking", "thinking": ""}
        (produced when a non-Anthropic reasoning model's turn is bridged to the
        Anthropic surface with no reasoning text) 400s with a message that names
        no signature, so the pre-rename matcher missed it and the strip-and-retry
        never fired. Raw string captured live on 2026-08-27."""
        from litellm.llms.anthropic.common_utils import (
            is_anthropic_invalid_thinking_block_error,
        )

        raw = (
            '{"type":"error","error":{"type":"invalid_request_error",'
            '"message":"messages.1.content.0.thinking: each thinking block must contain thinking"},'
            '"request_id":"req_011CeUTxhJj2rTUkK61qtbJ8"}'
        )
        assert is_anthropic_invalid_thinking_block_error(raw) is True

    def test_is_empty_thinking_block(self):
        from litellm.llms.anthropic.common_utils import is_empty_thinking_block

        assert is_empty_thinking_block({"type": "thinking", "thinking": ""}) is True
        assert is_empty_thinking_block({"type": "thinking", "thinking": " \n\t "}) is True
        assert is_empty_thinking_block({"type": "thinking", "thinking": None}) is True
        assert is_empty_thinking_block({"type": "thinking"}) is True
        assert is_empty_thinking_block({"type": "thinking", "thinking": "", "signature": "sig_abc"}) is True
        assert is_empty_thinking_block({"type": "thinking", "thinking": "plan", "signature": "sig"}) is False
        assert is_empty_thinking_block({"type": "redacted_thinking", "data": "opaque"}) is False
        assert is_empty_thinking_block({"type": "text", "text": ""}) is False
        assert is_empty_thinking_block("not a dict") is False

    def test_is_empty_unsigned_thinking_block(self):
        """Emit-side predicate: a signature-only block must be kept (Bedrock
        Converse adaptive thinking emits empty text with only a signature, and
        the client needs it to replay reasoning in tool-use turns); only an
        empty block with nothing to preserve is droppable."""
        from litellm.llms.anthropic.common_utils import is_empty_unsigned_thinking_block

        assert is_empty_unsigned_thinking_block({"type": "thinking", "thinking": ""}) is True
        assert is_empty_unsigned_thinking_block({"type": "thinking", "thinking": " \n\t "}) is True
        assert is_empty_unsigned_thinking_block({"type": "thinking"}) is True
        assert is_empty_unsigned_thinking_block({"type": "thinking", "thinking": "", "signature": ""}) is True
        assert is_empty_unsigned_thinking_block({"type": "thinking", "thinking": "", "signature": "sig_abc"}) is False
        assert is_empty_unsigned_thinking_block({"type": "thinking", "thinking": " ", "signature": "sig_abc"}) is False
        assert is_empty_unsigned_thinking_block({"type": "thinking", "thinking": "plan"}) is False
        assert is_empty_unsigned_thinking_block({"type": "redacted_thinking", "data": "opaque"}) is False
        assert is_empty_unsigned_thinking_block("not a dict") is False

    def test_strip_empty_content_blocks_drops_empty_thinking_blocks(self):
        """LIT-6357 ingestion half: an assistant tool-loop turn carrying an
        empty (even signed) thinking block keeps its tool_use blocks and loses
        the poison; whitespace-only counts as empty; a non-empty thinking block
        and redacted_thinking are untouched."""
        from litellm.llms.anthropic.common_utils import (
            strip_empty_content_blocks_from_anthropic_messages,
        )

        tu = {"type": "tool_use", "id": "toolu_01A", "name": "get_weather", "input": {"city": "Paris"}}
        msgs = [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "", "signature": "sig_abc"}, tu],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": " \n "},
                    {"type": "thinking", "thinking": "real plan", "signature": "sig"},
                    {"type": "redacted_thinking", "data": "opaque"},
                ],
            },
            {"role": "assistant", "content": [{"type": "thinking", "thinking": ""}]},
        ]
        out = strip_empty_content_blocks_from_anthropic_messages(msgs)
        assert len(out) == 3
        assert [b["type"] for b in out[1]["content"]] == ["tool_use"]
        assert [b["type"] for b in out[2]["content"]] == ["thinking", "redacted_thinking"]
        assert out[2]["content"][0]["thinking"] == "real plan"
        assert len(msgs[1]["content"]) == 2

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

    def test_strip_empty_content_blocks_from_anthropic_messages(self):
        """Covers #22930.  The core regression scenario: an assistant message
        with an empty text block alongside ``tool_use`` loses the empty block
        and keeps the ``tool_use``; a whole message that reduces to no blocks
        is dropped; whitespace-only text counts as empty; the caller's list
        is never mutated."""
        from litellm.llms.anthropic.common_utils import (
            strip_empty_content_blocks_from_anthropic_messages,
        )

        tu = {"type": "tool_use", "id": "x", "name": "Bash", "input": {}}
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "  \n "}, tu]},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        ]
        out = strip_empty_content_blocks_from_anthropic_messages(msgs)
        assert len(out) == 2 and out[0] is msgs[0]
        assert [b["type"] for b in out[1]["content"]] == ["tool_use"]
        assert len(msgs[1]["content"]) == 2  # caller's content unchanged

    def test_strip_empty_text_blocks_preserves_thinking_blocks(self):
        from litellm.llms.anthropic.common_utils import (
            strip_empty_content_blocks_from_anthropic_messages,
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
        out = strip_empty_content_blocks_from_anthropic_messages(msgs)
        assert [b["type"] for b in out[0]["content"]] == ["thinking"]

    def test_strip_empty_text_blocks_treats_null_text_as_empty(self):
        from litellm.llms.anthropic.common_utils import (
            strip_empty_content_blocks_from_anthropic_messages,
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
        out = strip_empty_content_blocks_from_anthropic_messages(msgs)
        assert [b["type"] for b in out[0]["content"]] == ["tool_result"]

    def test_strip_empty_text_blocks_treats_missing_text_key_as_empty(self):
        from litellm.llms.anthropic.common_utils import (
            strip_empty_content_blocks_from_anthropic_messages,
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
        out = strip_empty_content_blocks_from_anthropic_messages(msgs)
        assert [b["type"] for b in out[0]["content"]] == ["tool_result"]

    def test_strip_empty_text_blocks_leaves_non_empty_text_alone(self):
        from litellm.llms.anthropic.common_utils import (
            strip_empty_content_blocks_from_anthropic_messages,
        )

        msgs = [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]
        out = strip_empty_content_blocks_from_anthropic_messages(msgs)
        assert out[0] is msgs[0]  # untouched messages keep identity

    def test_strip_empty_text_blocks_treats_non_string_text_value_as_empty(self):
        from litellm.llms.anthropic.common_utils import (
            strip_empty_content_blocks_from_anthropic_messages,
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
        out = strip_empty_content_blocks_from_anthropic_messages(msgs)
        assert [b["type"] for b in out[0]["content"]] == ["tool_result"]

    def test_flatten_unencrypted_web_search_results_keeps_snippet_evidence(self):
        from litellm.llms.anthropic.common_utils import (
            flatten_unencrypted_web_search_results_in_anthropic_messages,
        )

        msgs = [
            {"role": "user", "content": "latest litellm version?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "srvtoolu_1",
                        "name": "web_search",
                        "input": {"query": "latest litellm version"},
                    },
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_1",
                        "content": [
                            {
                                "type": "web_search_result",
                                "url": "https://github.com/BerriAI/litellm/releases",
                                "title": "Releases",
                                "page_age": None,
                                "encrypted_content": "",
                                "snippet": "Latest release v1.95.0",
                            }
                        ],
                    },
                    {"type": "text", "text": "v1.95.0"},
                ],
            },
        ]

        out = flatten_unencrypted_web_search_results_in_anthropic_messages(msgs)

        assert out[0] is msgs[0]
        assert [b["type"] for b in out[1]["content"]] == ["text", "text"]
        flattened = out[1]["content"][0]["text"]
        assert "Web search results for 'latest litellm version':" in flattened
        assert "URL: https://github.com/BerriAI/litellm/releases" in flattened
        assert "Snippet: Latest release v1.95.0" in flattened
        assert msgs[1]["content"][0]["type"] == "server_tool_use"

    @pytest.mark.parametrize("results", [[], None], ids=["empty_list", "search_raised"])
    def test_flatten_unencrypted_web_search_results_flattens_a_resultless_search(self, results):
        """A search that found nothing, or that raised, still has to be flattened.

        Both cases reach the client as ``content: []``, and leaving that block in
        place ships an unsupported tag to Bedrock on the next turn just as surely
        as a populated one does.
        """
        from litellm.integrations.websearch_interception.transformation import (
            WebSearchTransformation,
        )
        from litellm.llms.anthropic.common_utils import (
            flatten_unencrypted_web_search_results_in_anthropic_messages,
        )

        block = WebSearchTransformation.build_web_search_tool_result_block(
            tool_use_id="srvtoolu_1",
            search_response=None if results is None else SimpleNamespace(results=results),
        )
        assert block["content"] == [], "fixture drifted from what the interceptor emits"

        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "srvtoolu_1",
                        "name": "web_search",
                        "input": {"query": "who won"},
                    },
                    block,
                    {"type": "text", "text": "I could not find that."},
                ],
            }
        ]

        out = flatten_unencrypted_web_search_results_in_anthropic_messages(msgs)

        assert [b["type"] for b in out[0]["content"]] == ["text", "text"]
        assert out[0]["content"][0]["text"] == ("Web search results for 'who won':\n\nNo results were returned.")

    @pytest.mark.parametrize("results", [[SimpleNamespace(title="Rome", url="u", snippet="s", date=None)], []])
    def test_flatten_unencrypted_web_search_results_is_idempotent(self, results):
        """Flattening twice must equal flattening once.

        The agentic loop re-enters the same entry point for its follow-up call and
        hands it the original history, so this runs again on already-flattened
        messages once per iteration. A pass that appended instead of replacing
        would duplicate the evidence on every loop.
        """
        from litellm.integrations.websearch_interception.transformation import (
            WebSearchTransformation,
        )
        from litellm.llms.anthropic.common_utils import (
            flatten_unencrypted_web_search_results_in_anthropic_messages,
        )

        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "when"}},
                    WebSearchTransformation.build_web_search_tool_result_block(
                        tool_use_id="srvtoolu_1",
                        search_response=SimpleNamespace(results=results),
                    ),
                    {"type": "text", "text": "753 BC."},
                ],
            }
        ]

        once = flatten_unencrypted_web_search_results_in_anthropic_messages(msgs)
        twice = flatten_unencrypted_web_search_results_in_anthropic_messages(once)

        assert [b["type"] for b in once[0]["content"]] == ["text", "text"]
        assert json.dumps(twice) == json.dumps(once)

    def test_flatten_unencrypted_web_search_results_preserves_real_anthropic_blocks(self):
        from litellm.llms.anthropic.common_utils import (
            flatten_unencrypted_web_search_results_in_anthropic_messages,
        )

        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "srvtoolu_1",
                        "name": "web_search",
                        "input": {"query": "q"},
                    },
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_1",
                        "content": [
                            {
                                "type": "web_search_result",
                                "url": "https://example.com",
                                "title": "Example",
                                "page_age": None,
                                "encrypted_content": "EqgfCioIARgBIiQ4",
                            }
                        ],
                    },
                ],
            }
        ]

        out = flatten_unencrypted_web_search_results_in_anthropic_messages(msgs)

        assert out[0] is msgs[0]

    def test_flatten_unencrypted_web_search_results_leaves_error_blocks_alone(self):
        from litellm.llms.anthropic.common_utils import (
            flatten_unencrypted_web_search_results_in_anthropic_messages,
        )

        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_1",
                        "content": {
                            "type": "web_search_tool_result_error",
                            "error_code": "max_uses_exceeded",
                        },
                    }
                ],
            }
        ]

        out = flatten_unencrypted_web_search_results_in_anthropic_messages(msgs)

        assert out[0] is msgs[0]

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
        assert normalize_anthropic_tool_use_id(f"{base}{THOUGHT_SIGNATURE_SEPARATOR}{sig}") == base

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
        assert config.should_retry_anthropic_messages_on_http_error(err_bad, {}) is False

        resp_500 = httpx.Response(500, request=req, text=err_text)
        err_500 = httpx.HTTPStatusError("bad", request=req, response=resp_500)
        assert config.should_retry_anthropic_messages_on_http_error(err_500, {}) is False

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

    def test_resolver_reads_flag_through_bedrock_invoke_prefix(self, local_model_cost_map):
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
    def test_adaptive_thinking_detected_for_opus_4_6_4_7_and_sonnet_4_6(self, local_model_cost_map, model):
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
    def test_unmapped_aliases_without_parseable_version_stay_non_adaptive(self, local_model_cost_map, model):
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
    def test_adaptive_thinking_version_fallback_for_unmapped_high_versions(self, local_model_cost_map, model):
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
    def test_adaptive_thinking_not_detected_for_unmapped_low_versions(self, local_model_cost_map, model):
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
    def test_default_suffix_models_are_adaptive_thinking(self, local_model_cost_map, model: str) -> None:
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        assert AnthropicModelInfo._is_adaptive_thinking_model(model, "anthropic") is True, (
            f"{model} not classified as adaptive thinking. Check _model_map_lookup_candidates strips @default suffix."
        )

    @pytest.mark.parametrize(
        "model,expected_bare",
        [
            ("vertex_ai/claude-opus-4-8@default", "claude-opus-4-8"),
            ("vertex_ai/claude-sonnet-4-6@default", "claude-sonnet-4-6"),
        ],
    )
    def test_lookup_candidates_include_bare_name(self, model: str, expected_bare: str) -> None:
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        candidates = AnthropicModelInfo._model_map_lookup_candidates(model)
        assert expected_bare in candidates, f"Expected '{expected_bare}' in candidates for '{model}', got: {candidates}"


class TestCapabilityProbeUsesCallerProvider:
    """``_supports_model_capability`` must probe under the caller's real provider
    namespace instead of a pinned ``"anthropic"``. With the pin, the exact Bedrock
    cost-map entry for ``global.anthropic.claude-opus-4-8`` was rejected by the
    provider match and the anthropic-scoped fallback rule answered instead, so
    flipping ``supports_adaptive_thinking`` on the exact entry changed nothing and
    the documented "exact entry beats rule" precedence was silently violated."""

    BEDROCK_MODEL = "global.anthropic.claude-opus-4-8"

    def test_exact_bedrock_entry_flag_is_authoritative_for_bedrock_caller(self, local_model_cost_map, monkeypatch):
        import litellm
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        assert AnthropicModelInfo._is_adaptive_thinking_model(self.BEDROCK_MODEL, "bedrock") is True

        monkeypatch.setitem(litellm.model_cost[self.BEDROCK_MODEL], "supports_adaptive_thinking", False)
        litellm.get_model_info.cache_clear()

        assert AnthropicModelInfo._is_adaptive_thinking_model(self.BEDROCK_MODEL, "bedrock") is False

    def test_native_anthropic_probe_still_reads_anthropic_entry(self, local_model_cost_map, monkeypatch):
        import litellm
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        monkeypatch.setitem(litellm.model_cost[self.BEDROCK_MODEL], "supports_adaptive_thinking", False)
        litellm.get_model_info.cache_clear()

        assert AnthropicModelInfo._is_adaptive_thinking_model("claude-opus-4-8", "anthropic") is True


def test_create_anthropic_model_list_response_shape():
    from litellm.llms.anthropic.common_utils import (
        create_anthropic_model_list_response,
    )

    response = create_anthropic_model_list_response(
        [
            {"id": "claude-opus-4-6", "object": "model", "created": 0, "owned_by": "openai"},
            {"id": "gpt-4o", "object": "model", "created": 0, "owned_by": "openai"},
            {"id": "claude-haiku-4-5", "object": "model", "created": 0, "owned_by": "openai"},
        ]
    )

    assert "object" not in response
    assert response["has_more"] is False
    assert response["first_id"] == "claude-opus-4-6"
    assert response["last_id"] == "claude-haiku-4-5"
    assert [m["id"] for m in response["data"]] == [
        "claude-opus-4-6",
        "gpt-4o",
        "claude-haiku-4-5",
    ]
    for entry in response["data"]:
        assert entry["type"] == "model"
        assert entry["display_name"] == entry["id"]
        # ISO 8601 with a Z suffix, as the Anthropic Models API returns.
        assert entry["created_at"].endswith("Z")
        assert "+00:00" not in entry["created_at"]
        assert entry["max_input_tokens"] is None
        assert entry["max_tokens"] is None


def test_create_anthropic_model_list_response_carries_token_limits():
    """max_input_tokens and max_tokens are nullable in the Anthropic Models shape,
    not optional, so both keys are emitted for every entry and carry null when the
    limit is unknown."""
    from litellm.llms.anthropic.common_utils import (
        create_anthropic_model_list_response,
    )

    response = create_anthropic_model_list_response(
        [
            {
                "id": "claude-opus-4-6",
                "object": "model",
                "created": 0,
                "owned_by": "openai",
                "max_input_tokens": 200000,
                "max_output_tokens": 64000,
            },
            {
                "id": "input-only",
                "object": "model",
                "created": 0,
                "owned_by": "openai",
                "max_input_tokens": 8192,
            },
            {"id": "unknown-limits", "object": "model", "created": 0, "owned_by": "openai"},
        ]
    )

    opus, input_only, unknown = response["data"]
    assert opus["max_input_tokens"] == 200000
    assert opus["max_tokens"] == 64000
    assert "max_output_tokens" not in opus
    assert input_only["max_input_tokens"] == 8192
    assert input_only["max_tokens"] is None
    assert unknown["max_input_tokens"] is None
    assert unknown["max_tokens"] is None
    for entry in response["data"]:
        assert "max_input_tokens" in entry
        assert "max_tokens" in entry


def test_create_anthropic_model_list_response_empty():
    from litellm.llms.anthropic.common_utils import (
        create_anthropic_model_list_response,
    )

    response = create_anthropic_model_list_response([])

    assert response["data"] == []
    assert response["has_more"] is False
    assert response["first_id"] is None
    assert response["last_id"] is None


# --------------------------------------------------------------------------- #
# Workload identity federation wiring (issue #28607)
# --------------------------------------------------------------------------- #

FAKE_MINTED_TOKEN = "sk-ant-oat01-wif-minted-token-for-testing-abc123"

ANTHROPIC_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_BASE",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_FEDERATION_RULE_ID",
    "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_SERVICE_ACCOUNT_ID",
    "ANTHROPIC_WORKSPACE_ID",
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
    "ANTHROPIC_IDENTITY_TOKEN",
    "LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS",
)

WIF_ENV = {
    "ANTHROPIC_FEDERATION_RULE_ID": "fdrl_wire1",
    "ANTHROPIC_ORGANIZATION_ID": "org-wire-1",
    "ANTHROPIC_IDENTITY_TOKEN": "inline-wire-jwt",
}

PROXY_CREDENTIAL_HEADER_NAMES = sorted(SpecialHeaders.litellm_credential_header_names())


class RecordingPoster:
    def __init__(self, response):
        self.requests = []
        self.thread_ids = []
        self._response = response

    def post(self, url, *, content, headers, timeout):
        self.requests.append((url, content, dict(headers)))
        self.thread_ids.append(threading.get_ident())
        return self._response


@pytest.fixture
def clean_anthropic_env(monkeypatch):
    for name in ANTHROPIC_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def wif_engine(monkeypatch, clean_anthropic_env):
    """Route the wiring's WIF tier through a fresh engine (never the module
    singleton, to avoid cross-test cache pollution) and count its consultations."""
    import httpx

    from litellm.llms.anthropic import common_utils as anthropic_common_utils
    from litellm.llms.anthropic.wif import get_anthropic_wif_token
    from litellm.llms.base_llm.auth.token_exchange import JwtBearerTokenExchangeEngine

    poster = RecordingPoster(
        httpx.Response(
            200,
            json={"access_token": FAKE_MINTED_TOKEN, "token_type": "Bearer", "expires_in": 3600},
        )
    )
    engine = JwtBearerTokenExchangeEngine(poster=poster)
    calls = []

    def with_injected_engine(litellm_params, api_base, model):
        calls.append(model)
        return get_anthropic_wif_token(litellm_params, api_base, model, engine)

    monkeypatch.setattr(anthropic_common_utils, "get_anthropic_wif_token", with_injected_engine)
    return poster, calls


@pytest.fixture
def wif_async_engine(monkeypatch, clean_anthropic_env):
    """Route both WIF facades through one fresh engine; the poster records the
    thread each exchange ran on and sync-facade consultations are counted so
    async tests can prove the mint went through the async seam, off the loop."""
    import httpx

    from litellm.llms.anthropic import common_utils as anthropic_common_utils
    from litellm.llms.anthropic.wif import aget_anthropic_wif_token, get_anthropic_wif_token
    from litellm.llms.base_llm.auth.token_exchange import JwtBearerTokenExchangeEngine

    poster = RecordingPoster(
        httpx.Response(
            200,
            json={"access_token": FAKE_MINTED_TOKEN, "token_type": "Bearer", "expires_in": 3600},
        )
    )
    engine = JwtBearerTokenExchangeEngine(poster=poster)
    sync_calls = []

    def sync_shim(litellm_params, api_base, model):
        sync_calls.append(model)
        return get_anthropic_wif_token(litellm_params, api_base, model, engine)

    async def async_shim(litellm_params, api_base, model):
        return await aget_anthropic_wif_token(litellm_params, api_base, model, engine)

    monkeypatch.setattr(anthropic_common_utils, "get_anthropic_wif_token", sync_shim)
    monkeypatch.setattr(anthropic_common_utils, "aget_anthropic_wif_token", async_shim)
    return poster, sync_calls


def _validate_chat_environment(api_key=None):
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    return AnthropicModelInfo().validate_environment(
        headers={},
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        api_key=api_key,
        api_base=None,
    )


class TestWifTierPrecedence:
    """WIF is the LOWEST credential tier: any api_key / auth_token source must
    win without the engine ever being consulted."""

    def _set_wif_env(self, monkeypatch):
        for name, value in WIF_ENV.items():
            monkeypatch.setenv(name, value)

    def test_explicit_api_key_beats_wif(self, monkeypatch, wif_engine):
        poster, calls = wif_engine
        self._set_wif_env(monkeypatch)

        headers = _validate_chat_environment(api_key=FAKE_REGULAR_KEY)

        assert headers["x-api-key"] == FAKE_REGULAR_KEY
        assert calls == []
        assert poster.requests == []

    def test_api_key_env_beats_wif(self, monkeypatch, wif_engine):
        poster, calls = wif_engine
        self._set_wif_env(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_REGULAR_KEY)

        headers = _validate_chat_environment()

        assert headers["x-api-key"] == FAKE_REGULAR_KEY
        assert calls == []
        assert poster.requests == []

    def test_auth_token_env_beats_wif(self, monkeypatch, wif_engine):
        poster, calls = wif_engine
        self._set_wif_env(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", FAKE_AUTH_TOKEN)

        headers = _validate_chat_environment()

        assert headers["authorization"] == f"Bearer {FAKE_AUTH_TOKEN}"
        assert calls == []
        assert poster.requests == []

    def test_wif_alone_mints_once(self, monkeypatch, wif_engine):
        poster, calls = wif_engine
        self._set_wif_env(monkeypatch)

        headers = _validate_chat_environment()

        assert headers["authorization"] == f"Bearer {FAKE_MINTED_TOKEN}"
        assert calls == ["claude-sonnet-4-5"]
        assert len(poster.requests) == 1
        assert poster.requests[0][0] == "https://api.anthropic.com/v1/oauth/token"


class TestWifZeroBehaviorChange:
    def test_unconfigured_raises_same_authentication_error(self, clean_anthropic_env):
        """No WIF config and no keys: same AuthenticationError as today (message
        extended, type and provider identical)."""
        import litellm

        with pytest.raises(litellm.AuthenticationError) as exc_info:
            _validate_chat_environment()

        assert exc_info.value.llm_provider == "anthropic"
        assert "ANTHROPIC_API_KEY" in exc_info.value.message
        assert "ANTHROPIC_AUTH_TOKEN" in exc_info.value.message
        assert "ANTHROPIC_FEDERATION_RULE_ID" in exc_info.value.message
        assert "ANTHROPIC_ORGANIZATION_ID" in exc_info.value.message
        assert "ANTHROPIC_SERVICE_ACCOUNT_ID" in exc_info.value.message
        assert "ANTHROPIC_IDENTITY_TOKEN_FILE" in exc_info.value.message


class TestWifHeaderContract:
    def test_minted_token_headers(self, monkeypatch, wif_engine):
        for name, value in WIF_ENV.items():
            monkeypatch.setenv(name, value)

        headers = _validate_chat_environment()

        assert headers["authorization"] == f"Bearer {FAKE_MINTED_TOKEN}"
        assert "oauth-2025-04-20" in headers["anthropic-beta"]
        assert "x-api-key" not in headers
        assert "anthropic-dangerous-direct-browser-access" not in headers

    def test_consumer_oat_key_keeps_dangerous_header(self, clean_anthropic_env):
        """Regression: user-supplied consumer sk-ant-oat keys keep today's behavior."""
        headers = _validate_chat_environment(api_key=FAKE_OAUTH_TOKEN)

        assert headers["authorization"] == f"Bearer {FAKE_OAUTH_TOKEN}"
        assert headers["anthropic-dangerous-direct-browser-access"] == "true"
        assert "oauth-2025-04-20" in headers["anthropic-beta"]


class TestMergeAnthropicBetaHeaders:
    """The Skills surface accepted a list-valued anthropic-beta before it shared this helper,
    so the helper has to keep taking one: .split() on a list is an AttributeError."""

    def test_list_valued_existing_header_is_merged(self):
        from litellm.llms.anthropic.common_utils import merge_anthropic_beta_headers

        assert merge_anthropic_beta_headers(["skills-2025-10-02", "files-api-2025-04-14"], "oauth-2025-04-20") == (
            "files-api-2025-04-14,oauth-2025-04-20,skills-2025-10-02"
        )

    def test_list_and_comma_string_forms_agree(self):
        from litellm.llms.anthropic.common_utils import merge_anthropic_beta_headers

        as_list = merge_anthropic_beta_headers(["a", "b"], "c")
        as_string = merge_anthropic_beta_headers("a,b", "c")
        assert as_list == as_string == "a,b,c"

    def test_skills_validate_environment_accepts_a_list_header(self, monkeypatch):
        """End of the regression: the Skills surface itself must not raise on the list form."""
        from litellm.llms.anthropic.skills.transformation import AnthropicSkillsConfig

        monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_REGULAR_KEY)

        headers = AnthropicSkillsConfig().validate_environment(
            headers={"anthropic-beta": ["files-api-2025-04-14"]},
            litellm_params=None,
        )

        assert "files-api-2025-04-14" in headers["anthropic-beta"]
        assert isinstance(headers["anthropic-beta"], str)


class TestWifServerOwnedAuthHeaderStrip:
    """A WIF-minted token must never ride alongside a caller-supplied credential
    header, but that stripping must fire only when a mint actually happened."""

    def test_mint_strips_caller_supplied_x_api_key(self, monkeypatch, wif_engine):
        """Security regression: without the strip, a caller-forwarded x-api-key
        would sit next to the server-minted Authorization on the outgoing request."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        poster, _ = wif_engine
        for name, value in WIF_ENV.items():
            monkeypatch.setenv(name, value)
        caller_key = "sk-ant-CALLER-SUPPLIED"

        headers = AnthropicModelInfo().validate_environment(
            headers={"x-api-key": caller_key},
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )

        assert headers["authorization"] == f"Bearer {FAKE_MINTED_TOKEN}"
        assert "x-api-key" not in headers
        assert caller_key not in headers.values()
        assert len(poster.requests) == 1

    def test_server_owned_set_is_every_proxy_credential_header(self):
        """The strip list must track the proxy's own key-header list, not a hand-rolled
        pair: every header user_api_key_auth accepts a LiteLLM key in must be here."""
        from litellm.llms.anthropic.common_utils import _SERVER_OWNED_AUTH_HEADERS

        assert _SERVER_OWNED_AUTH_HEADERS == SpecialHeaders.litellm_credential_header_names()
        assert {"x-litellm-api-key", "api-key", "x-goog-api-key"} < _SERVER_OWNED_AUTH_HEADERS

    @pytest.mark.parametrize("header_name", PROXY_CREDENTIAL_HEADER_NAMES)
    def test_mint_strips_every_proxy_credential_header(self, monkeypatch, wif_engine, header_name):
        """A LiteLLM virtual key arrives in any of the proxy's accepted key headers; once
        a mint happened none of them may reach Anthropic in any header slot."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        for name, value in WIF_ENV.items():
            monkeypatch.setenv(name, value)
        caller_key = "sk-litellm-CALLER-VIRTUAL-KEY"

        headers = AnthropicModelInfo().validate_environment(
            headers={header_name.title(): caller_key, "user-agent": "caller/1.0"},
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )

        assert headers["authorization"] == f"Bearer {FAKE_MINTED_TOKEN}"
        assert header_name == "authorization" or header_name not in {name.lower() for name in headers}
        assert all(caller_key not in value for value in headers.values())
        assert headers["user-agent"] == "caller/1.0"

    @pytest.mark.parametrize("header_name", PROXY_CREDENTIAL_HEADER_NAMES)
    def test_skills_surface_strips_caller_credentials_too(self, monkeypatch, wif_engine, header_name):
        """Skills builds its own headers as well; every minting surface needs the same strip."""
        from litellm.llms.anthropic.skills.transformation import AnthropicSkillsConfig

        for name, value in WIF_ENV.items():
            monkeypatch.setenv(name, value)
        caller_key = "sk-litellm-CALLER-VIRTUAL-KEY"

        headers = AnthropicSkillsConfig().validate_environment(
            headers={header_name.title(): caller_key, "user-agent": "caller/1.0"},
            litellm_params=None,
        )

        assert headers["authorization"] == f"Bearer {FAKE_MINTED_TOKEN}"
        assert header_name == "authorization" or header_name not in {name.lower() for name in headers}
        assert all(caller_key not in value for value in headers.values())
        assert headers["user-agent"] == "caller/1.0"

    def test_passthrough_honors_a_case_variant_caller_key_instead_of_minting(self, monkeypatch, wif_engine):
        """The passthrough surface hands the caller's own credential upstream rather than minting.
        That check was case-sensitive, so X-Api-Key slipped past it and the caller's key would have
        travelled beside a minted Bearer."""
        from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
            AnthropicMessagesConfig,
        )

        for name, value in WIF_ENV.items():
            monkeypatch.setenv(name, value)
        poster, _ = wif_engine
        caller_key = "sk-ant-CALLER-SUPPLIED"

        headers, _ = AnthropicMessagesConfig().validate_anthropic_messages_environment(
            headers={"X-Api-Key": caller_key},
            model="claude-sonnet-4-5",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )

        assert headers["X-Api-Key"] == caller_key
        assert "authorization" not in {name.lower() for name in headers}
        assert len(poster.requests) == 0

    @pytest.mark.parametrize("header_name", PROXY_CREDENTIAL_HEADER_NAMES)
    def test_batches_surface_strips_caller_credentials_too(self, monkeypatch, wif_engine, header_name):
        """Batches builds its own headers on the create path, so it needs the same strip: the
        handler's retrieve path passes none, but this entry point takes the caller's."""
        from litellm.llms.anthropic.batches.transformation import AnthropicBatchesConfig

        for name, value in WIF_ENV.items():
            monkeypatch.setenv(name, value)
        caller_key = "sk-litellm-CALLER-VIRTUAL-KEY"

        headers = AnthropicBatchesConfig().validate_environment(
            headers={header_name.title(): caller_key, "user-agent": "caller/1.0"},
            model="claude-sonnet-4-5",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )

        assert headers["authorization"] == f"Bearer {FAKE_MINTED_TOKEN}"
        assert header_name == "authorization" or header_name not in {name.lower() for name in headers}
        assert all(caller_key not in value for value in headers.values())
        assert headers["user-agent"] == "caller/1.0"

    @pytest.mark.parametrize("header_name", PROXY_CREDENTIAL_HEADER_NAMES)
    def test_files_surface_strips_caller_credentials_too(self, monkeypatch, wif_engine, header_name):
        """The files surface builds its own headers, so it needs the same strip the chat surface
        has: without it a minted federation Bearer travels beside the caller's own credential."""
        from litellm.llms.anthropic.files.transformation import AnthropicFilesConfig

        for name, value in WIF_ENV.items():
            monkeypatch.setenv(name, value)
        caller_key = "sk-litellm-CALLER-VIRTUAL-KEY"

        headers = AnthropicFilesConfig().validate_environment(
            headers={header_name.title(): caller_key, "user-agent": "caller/1.0"},
            model="claude-sonnet-4-5",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )

        assert headers["authorization"] == f"Bearer {FAKE_MINTED_TOKEN}"
        assert header_name == "authorization" or header_name not in {name.lower() for name in headers}
        assert all(caller_key not in value for value in headers.values())
        assert headers["user-agent"] == "caller/1.0"

    def test_no_mint_preserves_caller_supplied_authorization(self, monkeypatch, clean_anthropic_env):
        """No-regression: LiteLLM deliberately lets a caller-forwarded credential
        header ride alongside a statically configured ANTHROPIC_API_KEY, because the
        two occupy different header slots (x-api-key vs authorization) when the key
        isn't OAuth-shaped. The strip must stay conditional on an actual WIF mint."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_REGULAR_KEY)
        caller_authorization = "Bearer caller-forwarded-downstream-token"

        headers = AnthropicModelInfo().validate_environment(
            headers={"authorization": caller_authorization},
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )

        assert headers["x-api-key"] == FAKE_REGULAR_KEY
        assert headers["authorization"] == caller_authorization


class TestWifResolvedApiKeyThreading:
    """Regression for the resolved_api_key local (formerly a rebind of the api_key
    parameter): a minted token must reach the outgoing headers on both the sync
    validate_environment path and the async aget_auth_header path, never a stale
    None left over from the original unresolved parameter."""

    def test_validate_environment_carries_minted_token(self, monkeypatch, wif_engine):
        for name, value in WIF_ENV.items():
            monkeypatch.setenv(name, value)

        headers = _validate_chat_environment()

        assert "authorization" in headers
        assert headers["authorization"] not in (None, "Bearer None")
        assert headers["authorization"] == f"Bearer {FAKE_MINTED_TOKEN}"

    @pytest.mark.asyncio
    async def test_aget_auth_header_carries_minted_token(self, monkeypatch, wif_async_engine):
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        for name, value in WIF_ENV.items():
            monkeypatch.setenv(name, value)

        result = await AnthropicModelInfo.aget_auth_header(allow_workload_identity=True)

        assert result is not None
        assert result["authorization"] not in (None, "Bearer None")
        assert result["authorization"] == f"Bearer {FAKE_MINTED_TOKEN}"


class TestGetAuthHeaderBetas:
    def test_oat_branch_carries_oauth_beta(self, clean_anthropic_env):
        """The pre-existing bug: the oat branch returned a bare Bearer without the
        mandatory oauth beta."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        result = AnthropicModelInfo.get_auth_header(api_key=FAKE_OAUTH_TOKEN)

        assert result == {
            "authorization": f"Bearer {FAKE_OAUTH_TOKEN}",
            "anthropic-beta": "oauth-2025-04-20",
        }

    def test_wif_fallback_returns_bearer_and_beta(self, monkeypatch, wif_engine):
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        poster, calls = wif_engine
        for name, value in WIF_ENV.items():
            monkeypatch.setenv(name, value)

        result = AnthropicModelInfo.get_auth_header(allow_workload_identity=True)

        assert result == {
            "authorization": f"Bearer {FAKE_MINTED_TOKEN}",
            "anthropic-beta": "oauth-2025-04-20",
        }
        assert len(poster.requests) == 1

    def test_no_credentials_still_returns_none(self, clean_anthropic_env):
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        assert AnthropicModelInfo.get_auth_header() is None


class TestFilesBatchesBetaMerge:
    """Regression for the anthropic-beta clobber (files) and drop (batches):
    a Bearer oat auth header must keep the oauth beta AND gain the surface beta."""

    def test_files_merges_oauth_and_files_betas(self, clean_anthropic_env):
        from litellm.llms.anthropic.files.transformation import AnthropicFilesConfig

        headers = AnthropicFilesConfig().validate_environment(
            headers={},
            model="",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=FAKE_OAUTH_TOKEN,
        )

        assert headers["authorization"] == f"Bearer {FAKE_OAUTH_TOKEN}"
        betas = set(headers["anthropic-beta"].split(","))
        assert {"oauth-2025-04-20", "files-api-2025-04-14"} <= betas

    def test_batches_merges_oauth_and_batches_betas(self, clean_anthropic_env):
        from litellm.llms.anthropic.batches.transformation import AnthropicBatchesConfig

        headers = AnthropicBatchesConfig().validate_environment(
            headers={},
            model="",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=FAKE_OAUTH_TOKEN,
        )

        assert headers["authorization"] == f"Bearer {FAKE_OAUTH_TOKEN}"
        betas = set(headers["anthropic-beta"].split(","))
        assert {"oauth-2025-04-20", "message-batches-2024-09-24"} <= betas

    def test_files_preserves_caller_supplied_beta(self, clean_anthropic_env):
        """Regression: files did a two-way merge that dropped the client's own
        anthropic-beta; it must three-way merge exactly like batches."""
        from litellm.llms.anthropic.files.transformation import AnthropicFilesConfig

        headers = AnthropicFilesConfig().validate_environment(
            headers={"anthropic-beta": "context-1m-2025-08-07"},
            model="",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=FAKE_OAUTH_TOKEN,
        )

        betas = set(headers["anthropic-beta"].split(","))
        assert {"context-1m-2025-08-07", "oauth-2025-04-20", "files-api-2025-04-14"} <= betas


class TestMessagesEnvAuthBetaMerge:
    def test_client_beta_survives_env_auth_injection(self, monkeypatch, clean_anthropic_env):
        """Regression: headers.update(auth_header) silently clobbered the client's
        anthropic-beta on the native /v1/messages route."""
        from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
            AnthropicMessagesConfig,
        )

        monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_OAUTH_TOKEN)

        headers, _ = AnthropicMessagesConfig().validate_anthropic_messages_environment(
            headers={"anthropic-beta": "context-1m-2025-08-07"},
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
        )

        assert headers["authorization"] == f"Bearer {FAKE_OAUTH_TOKEN}"
        betas = set(headers["anthropic-beta"].split(","))
        assert {"context-1m-2025-08-07", "oauth-2025-04-20"} <= betas


WIF_PARAMS_ONLY = {
    "anthropic_federation_rule_id": "fdrl_params",
    "anthropic_organization_id": "org-params",
    "anthropic_identity_token": "oidc/env/WIF_PARAMS_TEST_TOKEN",
}


class TestWifLitellmParamsPlumbing:
    """Per-deployment anthropic_* litellm_params must reach the WIF tier on every
    surface that has them, not only chat."""

    @pytest.fixture(autouse=True)
    def _inline_identity_token(self, monkeypatch):
        monkeypatch.setenv("WIF_PARAMS_TEST_TOKEN", "params-jwt")

    def test_files_mints_from_litellm_params(self, wif_engine):
        from litellm.llms.anthropic.files.transformation import AnthropicFilesConfig

        poster, _ = wif_engine
        headers = AnthropicFilesConfig().validate_environment(
            headers={},
            model="",
            messages=[],
            optional_params={},
            litellm_params=dict(WIF_PARAMS_ONLY),
        )

        assert headers["authorization"] == f"Bearer {FAKE_MINTED_TOKEN}"
        assert len(poster.requests) == 1

    def test_batches_mints_from_litellm_params(self, wif_engine):
        from litellm.llms.anthropic.batches.transformation import AnthropicBatchesConfig

        poster, _ = wif_engine
        headers = AnthropicBatchesConfig().validate_environment(
            headers={},
            model="",
            messages=[],
            optional_params={},
            litellm_params=dict(WIF_PARAMS_ONLY),
        )

        assert headers["authorization"] == f"Bearer {FAKE_MINTED_TOKEN}"
        assert len(poster.requests) == 1

    def test_skills_mints_from_litellm_params(self, wif_engine):
        from litellm.llms.anthropic.skills.transformation import AnthropicSkillsConfig
        from litellm.types.router import GenericLiteLLMParams

        poster, _ = wif_engine
        headers = AnthropicSkillsConfig().validate_environment(
            headers={},
            litellm_params=GenericLiteLLMParams(
                anthropic_federation_rule_id=WIF_PARAMS_ONLY["anthropic_federation_rule_id"],
                anthropic_organization_id=WIF_PARAMS_ONLY["anthropic_organization_id"],
                anthropic_identity_token=WIF_PARAMS_ONLY["anthropic_identity_token"],
            ),
        )

        assert headers["authorization"] == f"Bearer {FAKE_MINTED_TOKEN}"
        assert len(poster.requests) == 1

    def test_messages_mints_from_litellm_params(self, wif_engine):
        from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
            AnthropicMessagesConfig,
        )

        poster, _ = wif_engine
        headers, _ = AnthropicMessagesConfig().validate_anthropic_messages_environment(
            headers={},
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params=dict(WIF_PARAMS_ONLY),
        )

        assert headers["authorization"] == f"Bearer {FAKE_MINTED_TOKEN}"
        assert len(poster.requests) == 1


class TestWifTokenUrlParity:
    """Both credential tiers must derive the SAME clean token URL from any form of
    the deployment base; a mismatch also duplicates mints because token_url is in
    the engine cache key."""

    @pytest.mark.parametrize(
        "configured_base",
        [
            "https://gw.example.com",
            "https://gw.example.com/",
            "https://gw.example.com/v1/messages",
            "https://gw.example.com/v1/messages/",
        ],
    )
    def test_both_tiers_share_one_clean_token_url(self, monkeypatch, wif_engine, configured_base):
        # This is about deriving one URL from many spellings of the same base, not about which
        # hosts an operator trusts with org-scoped credentials, so the private host is allowlisted.
        monkeypatch.setenv("LITELLM_ANTHROPIC_WIF_ALLOWED_HOSTS", "gw.example.com")
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        poster, _ = wif_engine
        for name, value in WIF_ENV.items():
            monkeypatch.setenv(name, value)

        AnthropicModelInfo().validate_environment(
            headers={},
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={"api_base": configured_base},
            api_key=None,
            api_base=None,
        )
        AnthropicModelInfo.get_auth_header(api_base=configured_base, allow_workload_identity=True)

        assert [url for (url, _, _) in poster.requests] == ["https://gw.example.com/v1/oauth/token"]


class TestWifAsyncSeam:
    """Async callers must resolve the WIF tier through the async facade so a cold
    mint never blocks the event loop."""

    @pytest.mark.asyncio
    async def test_aget_auth_header_runs_exchange_off_event_loop(self, monkeypatch, wif_async_engine):
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        poster, sync_calls = wif_async_engine
        for name, value in WIF_ENV.items():
            monkeypatch.setenv(name, value)

        result = await AnthropicModelInfo.aget_auth_header(allow_workload_identity=True)

        assert result == {
            "authorization": f"Bearer {FAKE_MINTED_TOKEN}",
            "anthropic-beta": "oauth-2025-04-20",
        }
        assert sync_calls == []
        assert poster.thread_ids == [poster.thread_ids[0]]
        assert poster.thread_ids[0] != threading.get_ident()

    @pytest.mark.asyncio
    async def test_avalidate_messages_environment_mints_off_loop(self, wif_async_engine, monkeypatch):
        from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
            AnthropicMessagesConfig,
        )

        poster, sync_calls = wif_async_engine
        monkeypatch.setenv("WIF_PARAMS_TEST_TOKEN", "params-jwt")

        headers, _ = await AnthropicMessagesConfig().avalidate_anthropic_messages_environment(
            headers={},
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params=dict(WIF_PARAMS_ONLY),
        )

        assert headers["authorization"] == f"Bearer {FAKE_MINTED_TOKEN}"
        assert sync_calls == []
        assert poster.thread_ids[0] != threading.get_ident()

    @pytest.mark.asyncio
    async def test_avalidate_delegates_to_subclass_sync_override(self):
        """A provider subclass that only overrides the sync method must keep its
        behavior when the handler goes through the async variant."""
        from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
            AnthropicMessagesConfig,
        )

        class MarkerConfig(AnthropicMessagesConfig):
            def validate_anthropic_messages_environment(
                self,
                headers,
                model,
                messages,
                optional_params,
                litellm_params,
                api_key=None,
                api_base=None,
            ):
                return {"x-marker": "sync"}, api_base

        headers, api_base = await MarkerConfig().avalidate_anthropic_messages_environment(
            headers={},
            model="claude-sonnet-4-5",
            messages=[],
            optional_params={},
            litellm_params={},
            api_base="https://marker.example.com",
        )

        assert headers == {"x-marker": "sync"}
        assert api_base == "https://marker.example.com"

    @pytest.mark.asyncio
    async def test_base_default_avalidate_delegates_to_sync(self):
        from litellm.llms.base_llm.anthropic_messages.transformation import (
            BaseAnthropicMessagesConfig,
        )

        class SyncOnlyConfig(BaseAnthropicMessagesConfig):
            def validate_anthropic_messages_environment(
                self,
                headers,
                model,
                messages,
                optional_params,
                litellm_params,
                api_key=None,
                api_base=None,
            ):
                return {"x-sync-only": "1"}, api_base

            def get_complete_url(self, api_base, api_key, model, optional_params, litellm_params, stream=None):
                return api_base or ""

            def get_supported_anthropic_messages_params(self, model):
                return []

            def transform_anthropic_messages_request(
                self, model, messages, anthropic_messages_optional_request_params, litellm_params, headers
            ):
                return {}

            def transform_anthropic_messages_response(self, model, raw_response, logging_obj):
                raise NotImplementedError

        headers, _ = await SyncOnlyConfig().avalidate_anthropic_messages_environment(
            headers={},
            model="m",
            messages=[],
            optional_params={},
            litellm_params={},
        )

        assert headers == {"x-sync-only": "1"}


class TestWifRespxEndToEnd:
    def test_completion_mints_and_never_leaks_config(self, monkeypatch, tmp_path, clean_anthropic_env):
        """Drives the REAL kwargs funnel through litellm.completion: the mint hits
        /v1/oauth/token, the data plane carries the minted Bearer + oauth beta, and
        NONE of the six anthropic_* keys leak into the /v1/messages body."""
        import httpx
        import respx

        import litellm
        from litellm.llms.anthropic import common_utils as anthropic_common_utils
        from litellm.llms.anthropic.wif import get_anthropic_wif_token
        from litellm.llms.base_llm.auth.token_exchange import JwtBearerTokenExchangeEngine

        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.setattr(litellm, "anthropic_key", None)
        monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", str(tmp_path))
        token_file = tmp_path / "identity-token"
        token_file.write_text("e2e-oidc-assertion", encoding="utf-8")

        engine = JwtBearerTokenExchangeEngine()
        monkeypatch.setattr(
            anthropic_common_utils,
            "get_anthropic_wif_token",
            lambda litellm_params, api_base, model: get_anthropic_wif_token(litellm_params, api_base, model, engine),
        )

        wif_kwarg_names: Final = (
            "anthropic_federation_rule_id",
            "anthropic_organization_id",
            "anthropic_service_account_id",
            "anthropic_workspace_id",
            "anthropic_identity_token_file",
            "anthropic_identity_token",
        )
        anthropic_response = {
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "Hello from WIF"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 4},
        }

        with respx.mock:
            token_route = respx.post("https://api.anthropic.com/v1/oauth/token").mock(
                return_value=httpx.Response(
                    200,
                    json={"access_token": FAKE_MINTED_TOKEN, "token_type": "Bearer", "expires_in": 3600},
                )
            )
            messages_route = respx.post("https://api.anthropic.com/v1/messages").mock(
                return_value=httpx.Response(200, json=anthropic_response)
            )
            response = litellm.completion(
                model="anthropic/claude-sonnet-4-5",
                messages=[{"role": "user", "content": "hi"}],
                anthropic_federation_rule_id="fdrl_e2e",
                anthropic_organization_id="org-e2e",
                anthropic_service_account_id="svcacct_e2e",
                anthropic_workspace_id="wrkspc_e2e",
                anthropic_identity_token_file=str(token_file),
                anthropic_identity_token="oidc/env/UNUSED_FALLBACK",
            )

        assert response.choices[0].message.content == "Hello from WIF"
        assert token_route.call_count == 1
        exchange_body = json.loads(token_route.calls[0].request.content)
        assert exchange_body["assertion"] == "e2e-oidc-assertion"
        assert exchange_body["federation_rule_id"] == "fdrl_e2e"

        data_request = messages_route.calls[0].request
        assert data_request.headers["authorization"] == f"Bearer {FAKE_MINTED_TOKEN}"
        assert "oauth-2025-04-20" in data_request.headers["anthropic-beta"]
        assert "x-api-key" not in data_request.headers
        assert "anthropic-dangerous-direct-browser-access" not in data_request.headers
        data_body = json.loads(data_request.content)
        for key in wif_kwarg_names:
            assert key not in data_body

    def test_completion_with_trailing_slash_api_base_mints_at_clean_token_url(
        self, monkeypatch, tmp_path, clean_anthropic_env
    ):
        """Regression: a trailing-slash api_base defeated the endswith check in
        main.py AND the removesuffix surgery, sending the exchange POST to
        .../v1/messages/v1/oauth/token (404)."""
        import httpx
        import respx

        import litellm
        from litellm.llms.anthropic import common_utils as anthropic_common_utils
        from litellm.llms.anthropic.wif import get_anthropic_wif_token
        from litellm.llms.base_llm.auth.token_exchange import JwtBearerTokenExchangeEngine

        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.setattr(litellm, "anthropic_key", None)
        monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", str(tmp_path))
        token_file = tmp_path / "identity-token"
        token_file.write_text("e2e-oidc-assertion", encoding="utf-8")

        engine = JwtBearerTokenExchangeEngine()
        monkeypatch.setattr(
            anthropic_common_utils,
            "get_anthropic_wif_token",
            lambda litellm_params, api_base, model: get_anthropic_wif_token(litellm_params, api_base, model, engine),
        )

        anthropic_response = {
            "id": "msg_02",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "Hello from WIF"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 4},
        }

        with respx.mock:
            token_route = respx.post("https://api.anthropic.com/v1/oauth/token").mock(
                return_value=httpx.Response(
                    200,
                    json={"access_token": FAKE_MINTED_TOKEN, "token_type": "Bearer", "expires_in": 3600},
                )
            )
            messages_route = respx.post(url__regex=r"https://api\.anthropic\.com/v1/messages.*").mock(
                return_value=httpx.Response(200, json=anthropic_response)
            )
            response = litellm.completion(
                model="anthropic/claude-sonnet-4-5",
                messages=[{"role": "user", "content": "hi"}],
                api_base="https://api.anthropic.com/v1/messages/",
                anthropic_federation_rule_id="fdrl_e2e",
                anthropic_organization_id="org-e2e",
                anthropic_identity_token_file=str(token_file),
            )

        assert response.choices[0].message.content == "Hello from WIF"
        assert token_route.call_count == 1
        assert messages_route.calls[0].request.headers["authorization"] == f"Bearer {FAKE_MINTED_TOKEN}"

    def test_get_auth_header_with_litellm_params_mints_via_real_engine(
        self, monkeypatch, tmp_path, clean_anthropic_env
    ):
        import httpx
        import respx

        from litellm.llms.anthropic import common_utils as anthropic_common_utils
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo
        from litellm.llms.anthropic.wif import get_anthropic_wif_token
        from litellm.llms.base_llm.auth.token_exchange import JwtBearerTokenExchangeEngine

        monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", str(tmp_path))
        token_file = tmp_path / "identity-token"
        token_file.write_text("e2e-oidc-assertion", encoding="utf-8")

        engine = JwtBearerTokenExchangeEngine()
        monkeypatch.setattr(
            anthropic_common_utils,
            "get_anthropic_wif_token",
            lambda litellm_params, api_base, model: get_anthropic_wif_token(litellm_params, api_base, model, engine),
        )

        with respx.mock:
            token_route = respx.post("https://api.anthropic.com/v1/oauth/token").mock(
                return_value=httpx.Response(
                    200,
                    json={"access_token": FAKE_MINTED_TOKEN, "token_type": "Bearer", "expires_in": 3600},
                )
            )
            result = AnthropicModelInfo.get_auth_header(
                allow_workload_identity=True,
                litellm_params={
                    "anthropic_federation_rule_id": "fdrl_e2e",
                    "anthropic_organization_id": "org-e2e",
                    "anthropic_identity_token_file": str(token_file),
                },
            )

        assert result == {
            "authorization": f"Bearer {FAKE_MINTED_TOKEN}",
            "anthropic-beta": "oauth-2025-04-20",
        }
        assert token_route.call_count == 1
        exchange_body = json.loads(token_route.calls[0].request.content)
        assert exchange_body["federation_rule_id"] == "fdrl_e2e"


class TestWifProviderAllowlist:
    """A federation token is an Anthropic-org credential, and the exchange POSTs the workload's OIDC
    assertion to the deployment's own api_base host. Providers that subclass the Anthropic config for
    their own endpoints must therefore never reach the WIF tier, even when it is configured purely
    through ANTHROPIC_* environment variables."""

    @staticmethod
    def _env_only_wif(monkeypatch) -> None:  # noqa: D401
        monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_prod")
        monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "org-prod-uuid")
        monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN", "oidc/env/WIF_TEST_JWT")
        monkeypatch.setenv("WIF_TEST_JWT", "jwt-assertion-value")

    def test_vertex_anthropic_never_mints_or_sends_the_assertion(self, monkeypatch, wif_engine):
        from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import (
            VertexAIAnthropicConfig,
        )

        import litellm

        poster, calls = wif_engine
        self._env_only_wif(monkeypatch)

        with pytest.raises(litellm.AuthenticationError):
            VertexAIAnthropicConfig().validate_environment(
                headers={},
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "hi"}],
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base="https://us-east5-aiplatform.googleapis.com/v1/projects/p/locations/us-east5",
            )

        assert calls == []
        assert poster.requests == []

    def test_anthropic_itself_still_mints(self, monkeypatch, wif_engine):
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        poster, calls = wif_engine
        self._env_only_wif(monkeypatch)

        headers = AnthropicConfig().validate_environment(
            headers={},
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )

        assert headers["authorization"] == f"Bearer {FAKE_MINTED_TOKEN}"
        assert len(poster.requests) == 1

    def test_auth_header_facade_defaults_to_refusing_to_mint(self, monkeypatch, clean_anthropic_env):
        """The facade is reachable from provider code that has nothing to do with Anthropic, so a
        caller must state that it authenticates against Anthropic's own API."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        self._env_only_wif(monkeypatch)

        assert AnthropicModelInfo.get_auth_header(None) is None
        assert AnthropicModelInfo.get_auth_header(None, allow_workload_identity=False) is None

    def test_eligibility_is_not_inherited_by_a_new_subclass(self):
        """A provider added later by subclassing the Anthropic config must not inherit the right to
        mint an Anthropic-org credential against its own host."""
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig
        from litellm.llms.anthropic.common_utils import config_allows_workload_identity

        class NewCompatibleProvider(AnthropicConfig):
            pass

        assert config_allows_workload_identity(AnthropicConfig()) is True
        assert config_allows_workload_identity(NewCompatibleProvider()) is False

    def test_model_discovery_gates_on_the_instance(self, monkeypatch, wif_engine):
        """get_models is inherited, so it must consult the instance rather than trusting its caller."""
        from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import (
            VertexAIAnthropicConfig,
        )

        poster, calls = wif_engine
        self._env_only_wif(monkeypatch)

        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            VertexAIAnthropicConfig().get_models(
                api_base="https://us-east5-aiplatform.googleapis.com/v1/projects/p/locations/us-east5"
            )

        assert poster.requests == []


def _models_page_response(page: dict, status_code: int = 200):
    import httpx

    return httpx.Response(status_code, json=page, request=httpx.Request("GET", "https://api.anthropic.com/v1/models"))


class RecordingModelsClient:
    """Records every call and answers with the queued responses in order, cycling the last one
    once exhausted so a runaway pagination loop degrades to a repeated page rather than an
    IndexError, letting the page-cap test observe the cap firing instead of a test bug."""

    def __init__(self, pages: list[dict] | None = None, responses=None):
        self.calls = []
        self._responses = responses if responses is not None else [_models_page_response(page) for page in pages]

    def get(self, url, headers=None, params=None, follow_redirects=None, timeout=None):
        self.calls.append(SimpleNamespace(url=url, headers=headers, params=params, follow_redirects=follow_redirects))
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[index]


class TestModelDiscovery:
    """AnthropicModelInfo.get_models / discover_models: pagination, redirect refusal, and
    sanitized errors on the upstream Anthropic /v1/models call itself (issue #28607 gap: a
    WIF source configured in litellm_params, rather than the environment, could not
    discover)."""

    @pytest.mark.parametrize(
        "configured_base", ["https://api.anthropic.com/v1", "https://api.anthropic.com/v1/messages"]
    )
    def test_discovery_does_not_double_the_version_segment(self, monkeypatch, clean_anthropic_env, configured_base):
        """Regression: /v1/models is appended here, so a base an operator already wrote as
        .../v1 (or the chat URL they copied) would be asked for /v1/v1/models and 404."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_REGULAR_KEY)
        client = RecordingModelsClient([{"data": [{"id": "claude-a"}], "has_more": False, "last_id": "claude-a"}])
        monkeypatch.setattr("litellm.module_level_client", client)

        models = AnthropicModelInfo().get_models(api_base=configured_base)

        assert models == ["anthropic/claude-a"]
        assert client.calls[0].url == "https://api.anthropic.com/v1/models"

    def test_get_models_paginates_via_has_more_and_last_id(self, monkeypatch, clean_anthropic_env):
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_REGULAR_KEY)
        client = RecordingModelsClient(
            [
                {"data": [{"id": "claude-a"}, {"id": "claude-b"}], "has_more": True, "last_id": "claude-b"},
                {"data": [{"id": "claude-c"}], "has_more": False, "last_id": "claude-c"},
            ]
        )
        monkeypatch.setattr("litellm.module_level_client", client)

        models = AnthropicModelInfo().get_models(api_base="https://api.anthropic.com")

        assert models == ["anthropic/claude-a", "anthropic/claude-b", "anthropic/claude-c"]
        assert len(client.calls) == 2
        assert client.calls[0].url == "https://api.anthropic.com/v1/models"
        assert client.calls[1].url == "https://api.anthropic.com/v1/models?after_id=claude-b"

    def test_paginated_fetch_survives_the_real_http_client(self, monkeypatch, clean_anthropic_env):
        """Regression: the second page is fetched through the real HTTPHandler, which merges the
        URL's query string into the params mapping by mutating it. Handing that client a
        read-only mapping raised AttributeError, so discovery blew up for any org holding more
        models than one page, while the stubbed client here never exercised the mutation."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_REGULAR_KEY)
        requested: Final = []  # mutable-ok: a test spy recording the URLs the client was asked for

        def respond(request: httpx.Request) -> httpx.Response:
            requested.append(str(request.url))
            first: Final = "after_id" not in request.url.params
            return httpx.Response(
                200,
                json={
                    "data": [{"id": "claude-a"}] if first else [{"id": "claude-b"}],
                    "has_more": first,
                    "last_id": "claude-a" if first else "claude-b",
                },
            )

        handler: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
        monkeypatch.setattr("litellm.module_level_client", handler)

        models = AnthropicModelInfo().get_models(api_base="https://api.anthropic.com")

        assert models == ["anthropic/claude-a", "anthropic/claude-b"]
        assert requested == [
            "https://api.anthropic.com/v1/models",
            "https://api.anthropic.com/v1/models?after_id=claude-a",
        ]

    def test_get_models_refuses_to_follow_redirects(self, monkeypatch, clean_anthropic_env):
        """Only the configured api_base is validated, so a redirected /v1/models must not be
        allowed to replay the credential to an unvalidated origin -- same rule already applied
        to the WIF token exchange itself."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_REGULAR_KEY)
        client = RecordingModelsClient([{"data": [], "has_more": False, "last_id": None}])
        monkeypatch.setattr("litellm.module_level_client", client)

        AnthropicModelInfo().get_models(api_base="https://api.anthropic.com")

        assert client.calls[0].follow_redirects is False

    def test_get_models_page_cap_stops_a_runaway_has_more(self, monkeypatch, clean_anthropic_env):
        from litellm.llms.anthropic.common_utils import (
            _MODEL_LIST_PAGE_CAP,
            AnthropicModelInfo,
        )

        monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_REGULAR_KEY)
        client = RecordingModelsClient([{"data": [{"id": "claude-loop"}], "has_more": True, "last_id": "claude-loop"}])
        monkeypatch.setattr("litellm.module_level_client", client)

        with pytest.raises(Exception, match="did not terminate"):
            AnthropicModelInfo().get_models(api_base="https://api.anthropic.com")

        assert len(client.calls) == _MODEL_LIST_PAGE_CAP

    def test_get_models_error_is_sanitized_not_raw_response_text(self, monkeypatch, clean_anthropic_env):
        """A failed discovery call must never echo the raw response body verbatim -- only the
        structured error message, so an unrelated/oversized/reflected body is not surfaced."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_REGULAR_KEY)
        reflected_payload = "<script>evil()</script>" * 50
        client = RecordingModelsClient(
            responses=[
                _models_page_response(
                    {
                        "type": "error",
                        "error": {
                            "type": "authentication_error",
                            "message": "invalid x-api-key",
                            "reflected": reflected_payload,
                        },
                    },
                    status_code=401,
                )
            ]
        )
        monkeypatch.setattr("litellm.module_level_client", client)

        with pytest.raises(Exception, match="invalid x-api-key") as exc_info:  # noqa: B017, PT011  # the callee raises a bare Exception; match pins the sanitized text
            AnthropicModelInfo().get_models(api_base="https://api.anthropic.com")

        assert "invalid x-api-key" in str(exc_info.value)
        assert reflected_payload not in str(exc_info.value)

    def test_discover_models_threads_litellm_params_into_wif(self, monkeypatch, wif_engine):
        """The gap this phase fixes: get_models only ever saw api_key/api_base, so a WIF source
        configured in litellm_params (rather than ANTHROPIC_* env vars) could not discover."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        poster, calls = wif_engine
        client = RecordingModelsClient([{"data": [{"id": "claude-wif"}], "has_more": False, "last_id": None}])
        monkeypatch.setattr("litellm.module_level_client", client)
        monkeypatch.setenv("DISC_JWT", "jwt-assertion-value")

        models = AnthropicModelInfo().discover_models(
            litellm_params={
                "anthropic_federation_rule_id": "fdrl_disc",
                "anthropic_organization_id": "org-disc",
                "anthropic_identity_token": "oidc/env/DISC_JWT",
            }
        )

        assert models == ["anthropic/claude-wif"]
        assert len(poster.requests) == 1
        assert client.calls[0].headers["authorization"] == f"Bearer {FAKE_MINTED_TOKEN}"

    def test_discover_models_without_litellm_params_behaves_like_get_models(self, monkeypatch, clean_anthropic_env):
        """No litellm_params (the wildcard-discovery call shape) must fall back to the
        env-only resolution get_models has always used -- zero behavior change for that path."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_REGULAR_KEY)
        client = RecordingModelsClient([{"data": [{"id": "claude-env"}], "has_more": False, "last_id": None}])
        monkeypatch.setattr("litellm.module_level_client", client)

        models = AnthropicModelInfo().discover_models(litellm_params=None)

        assert models == ["anthropic/claude-env"]
        assert client.calls[0].headers["x-api-key"] == FAKE_REGULAR_KEY

    def test_discover_models_explicit_api_key_beats_wif(self, monkeypatch, wif_engine):
        """Same precedence discover_models must honor as every other Anthropic auth surface:
        WIF is the lowest tier."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        poster, calls = wif_engine
        client = RecordingModelsClient([{"data": [], "has_more": False, "last_id": None}])
        monkeypatch.setattr("litellm.module_level_client", client)

        AnthropicModelInfo().discover_models(
            litellm_params={
                "api_key": FAKE_REGULAR_KEY,
                "anthropic_federation_rule_id": "fdrl_disc",
                "anthropic_organization_id": "org-disc",
                "anthropic_identity_token": "oidc/env/DISC_JWT",
            }
        )

        assert client.calls[0].headers["x-api-key"] == FAKE_REGULAR_KEY
        assert calls == []
        assert poster.requests == []


class TestWifExchangeTransportHardening:
    def test_token_exchange_client_does_not_follow_redirects(self):
        """Only the initial token URL is validated, so a 3xx must not be allowed to replay the
        assertion to an origin that was never checked."""
        from litellm.llms.base_llm.auth.token_exchange import _HttpxSyncTokenPoster

        handler = _HttpxSyncTokenPoster()._handler_instance()

        assert handler.client.follow_redirects is False


class TestWifParamsAreNotClientSettable:
    def test_every_minting_param_is_server_owned(self):
        """Each of these selects which server-side secret is read, or the scope it is minted for.
        The workspace id was once carved out here as inert; it is not. It is the scope of the
        minted org credential, and the router merges request kwargs over deployment params, so a
        caller who set it picked the scope instead of the administrator."""
        from litellm.proxy.auth.auth_utils import _ANTHROPIC_WIF_UNCONDITIONAL_BANNED
        from litellm.types.utils import anthropic_wif_litellm_params

        assert set(_ANTHROPIC_WIF_UNCONDITIONAL_BANNED) == set(anthropic_wif_litellm_params)


class TestWifServerOwnedParamsAreUnconditional:
    """The minting fields choose which server-side secret is read and, with api_base, where it goes,
    so no client-side credential opt-in may re-enable them."""

    @staticmethod
    def _body(param: str) -> dict:
        return {"model": "claude-sonnet-5", param: "oidc/env/SOME_SERVER_SECRET"}

    @pytest.mark.parametrize(
        "param",
        [
            "anthropic_identity_token",
            "anthropic_identity_token_file",
            "anthropic_federation_rule_id",
            "anthropic_organization_id",
            "anthropic_service_account_id",
            # Phase 1 identity-source selection and its two variants' fields: each one
            # selects a server-side secret or a destination (a signing key, a client
            # secret, a token endpoint), so every one joins the same unconditional ban.
            "anthropic_identity_source",
            "anthropic_issuer_url",
            "anthropic_issuer_subject",
            "anthropic_issuer_audience",
            "anthropic_issuer_ttl_seconds",
            "anthropic_issuer_signing_key_ref",
            "anthropic_keycloak_token_url",
            "anthropic_keycloak_client_id",
            "anthropic_keycloak_auth_method",
            "anthropic_keycloak_client_secret_ref",
            "anthropic_keycloak_scope",
        ],
    )
    def test_rejected_even_with_proxy_wide_opt_in(self, param: str):
        from litellm.proxy.auth.auth_utils import is_request_body_safe

        with pytest.raises(ValueError, match="server-owned workload identity federation"):
            is_request_body_safe(
                request_body=self._body(param),
                general_settings={"allow_client_side_credentials": True},
                llm_router=None,
                model="claude-sonnet-5",
            )

    def test_rejected_inside_nested_litellm_params(self):
        from litellm.proxy.auth.auth_utils import is_request_body_safe

        with pytest.raises(ValueError, match="server-owned workload identity federation"):
            is_request_body_safe(
                request_body={"model": "claude-sonnet-5", "litellm_params": self._body("anthropic_identity_token")},
                general_settings={"allow_client_side_credentials": True},
                llm_router=None,
                model="claude-sonnet-5",
            )

    def test_workspace_id_is_refused_from_a_request_body(self):
        """Regression, proven live against Anthropic before this was closed: a caller-supplied
        workspace id reached the token endpoint, which answered "workspace_id is not a well-formed
        wrkspc_ tagged ID", i.e. the caller's value had become the scope of the minted credential.
        router.py merges request kwargs OVER deployment params, so it also beat the configured one."""
        from litellm.proxy.auth.auth_utils import is_request_body_safe

        with pytest.raises(Exception, match="server-owned workload identity federation parameter"):
            is_request_body_safe(
                request_body={"model": "claude-sonnet-5", "anthropic_workspace_id": "wrkspc_abc"},
                general_settings={},
                llm_router=None,
                model="claude-sonnet-5",
            )

    def test_refusal_points_bedrock_callers_at_their_own_spelling(self):
        """Banning this spelling must not read as "no workspace selection anywhere": the Bedrock
        Claude Platform route takes workspace_id/aws_workspace_id, neither of which is a
        federation parameter, so the error names them."""
        from litellm.proxy.auth.auth_utils import is_request_body_safe

        with pytest.raises(Exception, match="workspace_id or aws_workspace_id"):
            is_request_body_safe(
                request_body={"model": "claude-sonnet-5", "anthropic_workspace_id": "wrkspc_abc"},
                general_settings={},
                llm_router=None,
                model="claude-sonnet-5",
            )

    def test_bedrock_workspace_spellings_are_untouched(self):
        """The Bedrock route's own spellings stay settable, which is what keeps this ban from
        removing a pre-existing capability."""
        from litellm.proxy.auth.auth_utils import is_request_body_safe

        for spelling in ("workspace_id", "aws_workspace_id"):
            assert (
                is_request_body_safe(
                    request_body={"model": "claude-sonnet-5", spelling: "wrkspc_abc"},
                    general_settings={},
                    llm_router=None,
                    model="claude-sonnet-5",
                )
                is True
            )


class TestWifDisabledOnClientRedirectedBase:
    def test_the_sentinel_survives_the_kwargs_funnel(self):
        """Setting the sentinel is only half of it. get_litellm_params rebuilds litellm_params from
        kwargs, so a field it does not carry is dropped on the way and the deployment federates for
        the caller-chosen base after all."""
        from litellm.litellm_core_utils.get_litellm_params import FORWARDED_KWARGS_KEYS
        from litellm.router_utils.clientside_credential_handler import (
            DISABLE_WORKLOAD_IDENTITY_PARAM,
        )

        assert DISABLE_WORKLOAD_IDENTITY_PARAM in FORWARDED_KWARGS_KEYS

    def test_the_sentinel_is_not_client_settable(self):
        """It is server-owned in both directions: a caller must not be able to set it, and must not
        be able to clear it either."""
        from litellm.router_utils.clientside_credential_handler import (
            DISABLE_WORKLOAD_IDENTITY_PARAM,
        )
        from litellm.types.router import reject_server_owned_wif_params

        with pytest.raises(ValueError, match=DISABLE_WORKLOAD_IDENTITY_PARAM):
            reject_server_owned_wif_params({DISABLE_WORKLOAD_IDENTITY_PARAM: False})

    def test_base_override_clears_wif_and_sets_the_sentinel(self):
        """A federation token minted for a client-chosen api_base would send the workload's assertion,
        and then the minted bearer, to that host."""
        from litellm.llms.anthropic.wif import resolve_anthropic_wif_params
        from litellm.router_utils.clientside_credential_handler import (
            DISABLE_WORKLOAD_IDENTITY_PARAM,
            get_dynamic_litellm_params,
        )

        admin_deployment = {
            "model": "anthropic/claude-sonnet-5",
            "anthropic_federation_rule_id": "fdrl_admin",
            "anthropic_organization_id": "org-admin",
            "anthropic_identity_token": "oidc/env/WIF_TEST_JWT",
        }

        redirected = get_dynamic_litellm_params(
            litellm_params=dict(admin_deployment),
            request_kwargs={"api_base": "https://not-anthropic.example"},
        )

        assert redirected[DISABLE_WORKLOAD_IDENTITY_PARAM] is True
        assert "anthropic_federation_rule_id" not in redirected
        assert resolve_anthropic_wif_params(redirected) is None

    def test_sentinel_blocks_env_var_configured_federation(self, monkeypatch):
        """Environment-configured federation cannot be cleared out of a dict, so the sentinel is what
        stops it on a redirected deployment."""
        from litellm.llms.anthropic.wif import resolve_anthropic_wif_params
        from litellm.router_utils.clientside_credential_handler import DISABLE_WORKLOAD_IDENTITY_PARAM

        monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_env")
        monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "org-env")
        monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN", "oidc/env/WIF_TEST_JWT")
        monkeypatch.setenv("WIF_TEST_JWT", "jwt-assertion-value")

        assert resolve_anthropic_wif_params({}) is not None
        assert resolve_anthropic_wif_params({DISABLE_WORKLOAD_IDENTITY_PARAM: True}) is None

    def test_base_override_clears_internal_issuer_fields(self):
        """Same failure mode the legacy-path test above guards against, for the internal_issuer
        identity source: a signing_key_ref resolved for a client-chosen api_base would mint an
        assertion, and then a bearer token, for that host."""
        from litellm.llms.anthropic.wif import resolve_anthropic_wif_params
        from litellm.router_utils.clientside_credential_handler import (
            DISABLE_WORKLOAD_IDENTITY_PARAM,
            get_dynamic_litellm_params,
        )

        admin_deployment = {
            "model": "anthropic/claude-sonnet-5",
            "anthropic_federation_rule_id": "fdrl_admin",
            "anthropic_organization_id": "org-admin",
            "anthropic_identity_source": "internal_issuer",
            "anthropic_issuer_url": "https://issuer.internal.example",
            "anthropic_issuer_subject": "workload-a",
            "anthropic_issuer_signing_key_ref": "oidc/env/ISSUER_SIGNING_KEY_PEM",
        }

        redirected = get_dynamic_litellm_params(
            litellm_params=dict(admin_deployment),
            request_kwargs={"api_base": "https://not-anthropic.example"},
        )

        assert redirected[DISABLE_WORKLOAD_IDENTITY_PARAM] is True
        assert "anthropic_identity_source" not in redirected
        assert "anthropic_issuer_signing_key_ref" not in redirected
        assert resolve_anthropic_wif_params(redirected) is None

    def test_base_override_clears_keycloak_fields(self):
        """Same as the internal_issuer case above, for the keycloak identity source: a
        client_secret_ref resolved for a client-chosen api_base must not follow it there."""
        from litellm.llms.anthropic.wif import resolve_anthropic_wif_params
        from litellm.router_utils.clientside_credential_handler import (
            DISABLE_WORKLOAD_IDENTITY_PARAM,
            get_dynamic_litellm_params,
        )

        admin_deployment = {
            "model": "anthropic/claude-sonnet-5",
            "anthropic_federation_rule_id": "fdrl_admin",
            "anthropic_organization_id": "org-admin",
            "anthropic_identity_source": "keycloak",
            "anthropic_keycloak_token_url": "https://keycloak.internal.example/realms/r/protocol/openid-connect/token",
            "anthropic_keycloak_client_id": "litellm",
            "anthropic_keycloak_client_secret_ref": "oidc/env/KEYCLOAK_CLIENT_SECRET",
        }

        redirected = get_dynamic_litellm_params(
            litellm_params=dict(admin_deployment),
            request_kwargs={"api_base": "https://not-anthropic.example"},
        )

        assert redirected[DISABLE_WORKLOAD_IDENTITY_PARAM] is True
        assert "anthropic_identity_source" not in redirected
        assert "anthropic_keycloak_client_secret_ref" not in redirected
        assert resolve_anthropic_wif_params(redirected) is None
