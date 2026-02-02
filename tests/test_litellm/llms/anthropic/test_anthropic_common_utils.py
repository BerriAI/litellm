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
    """Test 1: OAuth token detection in common_utils returns marker"""
    from litellm.llms.anthropic.common_utils import optionally_handle_anthropic_oauth

    headers = {"authorization": f"Bearer {FAKE_OAUTH_TOKEN}"}
    updated_headers, extracted_api_key = optionally_handle_anthropic_oauth(headers, None)

    # With the marker approach, we return a marker not the actual token
    assert extracted_api_key == "_oauth_", "Should return OAuth marker instead of token"
    assert updated_headers["anthropic-beta"] == "oauth-2025-04-20"
    assert updated_headers["anthropic-dangerous-direct-browser-access"] == "true"
    # The original Authorization header should remain intact
    assert headers["authorization"] == f"Bearer {FAKE_OAUTH_TOKEN}"


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

    # x-api-key should be set to the marker, NOT the OAuth token
    assert updated_headers["x-api-key"] == "_oauth_", "x-api-key should be marker, not OAuth token"
    assert updated_headers["anthropic-dangerous-direct-browser-access"] == "true"


def test_oauth_detection_in_messages_transformation():
    """Test 3: OAuth detection in messages transformation should not set x-api-key"""
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

    # IMPORTANT: x-api-key should NOT be set when using OAuth
    # This prevents OAuth token leakage to third-party providers
    assert "x-api-key" not in updated_headers, "x-api-key should NOT be set for OAuth requests"
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


def test_is_anthropic_oauth_key_edge_cases():
    """Test 5: is_anthropic_oauth_key() with various edge cases"""
    from litellm.llms.anthropic.common_utils import is_anthropic_oauth_key

    # OAuth tokens should return True
    assert is_anthropic_oauth_key("sk-ant-oat01-abc123") is True
    assert is_anthropic_oauth_key("sk-ant-oat02-xyz789") is True
    assert is_anthropic_oauth_key("Bearer sk-ant-oat01-abc123") is True
    assert is_anthropic_oauth_key("Bearer sk-ant-oat02-xyz789") is True

    # Non-OAuth should return False
    assert is_anthropic_oauth_key(None) is False
    assert is_anthropic_oauth_key("") is False
    assert is_anthropic_oauth_key("sk-ant-api01-abc123") is False
    assert is_anthropic_oauth_key("sk-ant-api02-xyz789") is False
    assert is_anthropic_oauth_key("Bearer sk-ant-api01-abc123") is False
    assert is_anthropic_oauth_key("Bearer sk-ant-api02-xyz789") is False

    # Just the prefix (edge case - starts with sk-ant-oat)
    assert is_anthropic_oauth_key("sk-ant-oat") is True

    # Case sensitivity (lowercase should not match)
    assert is_anthropic_oauth_key("sk-ant-OAT01-abc123") is False
    assert is_anthropic_oauth_key("SK-ANT-OAT01-abc123") is False