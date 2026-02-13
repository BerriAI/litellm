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
