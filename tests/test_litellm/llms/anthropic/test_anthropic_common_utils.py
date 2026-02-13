"""
Tests for Anthropic OAuth token handling for Claude Code Max integration.
"""

import os
import sys

from litellm.llms.anthropic.common_utils import is_anthropic_oauth_key

# Add litellm to path
sys.path.insert(0, os.path.abspath("../../../../.."))

# Fake OAuth token for testing (not a real secret)
FAKE_OAUTH_TOKEN = "sk-ant-oat01-fake-token-for-testing-123456789abcdef"


def test_oauth_detection_in_common_utils():
    """Test 1: OAuth token detection in common_utils"""
    from litellm.llms.anthropic.common_utils import optionally_handle_anthropic_oauth

    headers = {"authorization": f"Bearer {FAKE_OAUTH_TOKEN}"}
    updated_headers, extracted_api_key = optionally_handle_anthropic_oauth(headers, None)

    assert is_anthropic_oauth_key(extracted_api_key)
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

    # OAuth should use Authorization header, not x-api-key
    assert "x-api-key" not in updated_headers
    assert "authorization" in updated_headers
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

    # OAuth should use Authorization header, not x-api-key
    assert "x-api-key" not in updated_headers
    assert "authorization" in updated_headers
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


def test_is_anthropic_oauth_key_formats():
    """Test 5: is_anthropic_oauth_key handles both raw and Bearer formats"""
    # Raw token format (after extraction from Authorization header)
    assert is_anthropic_oauth_key(FAKE_OAUTH_TOKEN) is True

    # Bearer token format (from Authorization header)
    assert is_anthropic_oauth_key(f"Bearer {FAKE_OAUTH_TOKEN}") is True

    # Case insensitive Bearer
    assert is_anthropic_oauth_key(f"bearer {FAKE_OAUTH_TOKEN}") is True
    assert is_anthropic_oauth_key(f"BEARER {FAKE_OAUTH_TOKEN}") is True

    # Regular API keys should return False
    assert is_anthropic_oauth_key("sk-ant-api03-regular-key-123") is False
    assert is_anthropic_oauth_key("Bearer sk-ant-api03-regular-key-123") is False

    # Edge cases
    assert is_anthropic_oauth_key(None) is False
    assert is_anthropic_oauth_key("") is False
    assert is_anthropic_oauth_key("Bearer ") is False


def test_oauth_with_direct_api_key():
    """Test 6: OAuth token passed directly as api_key (not via Authorization header)"""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    config = AnthropicModelInfo()
    headers = {}

    updated_headers = config.validate_environment(
        headers=headers,
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        api_key=FAKE_OAUTH_TOKEN,  # OAuth token passed directly
        api_base=None,
    )

    # OAuth should use Authorization header, not x-api-key
    assert "x-api-key" not in updated_headers
    assert "authorization" in updated_headers
    assert updated_headers["authorization"] == f"Bearer {FAKE_OAUTH_TOKEN}"
    # OAuth beta headers should also be set
    assert "oauth-2025-04-20" in updated_headers["anthropic-beta"]
    assert updated_headers["anthropic-dangerous-direct-browser-access"] == "true"


def test_proxy_forwards_oauth_authorization_header():
    """Test 7: Proxy _get_forwardable_headers forwards OAuth Authorization header"""
    from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup

    # OAuth token in Authorization header should be forwarded
    oauth_headers = {"authorization": f"Bearer {FAKE_OAUTH_TOKEN}"}
    forwarded = LiteLLMProxyRequestSetup._get_forwardable_headers(oauth_headers)
    assert "authorization" in forwarded
    assert forwarded["authorization"] == f"Bearer {FAKE_OAUTH_TOKEN}"

    # Regular API key in Authorization header should NOT be forwarded
    regular_headers = {"authorization": "Bearer sk-ant-api03-regular-key-123"}
    forwarded = LiteLLMProxyRequestSetup._get_forwardable_headers(regular_headers)
    assert "authorization" not in forwarded

    # x-* headers should still be forwarded
    x_headers = {"x-custom-header": "value", "authorization": "Bearer sk-regular"}
    forwarded = LiteLLMProxyRequestSetup._get_forwardable_headers(x_headers)
    assert "x-custom-header" in forwarded
    assert "authorization" not in forwarded


def test_oauth_case_insensitive_authorization_header():
    """Test 8: OAuth detection works with different Authorization header cases"""
    from litellm.llms.anthropic.common_utils import optionally_handle_anthropic_oauth

    # Test with capital 'A' (how HTTP headers typically come through)
    headers_capital = {"Authorization": f"Bearer {FAKE_OAUTH_TOKEN}"}
    updated_headers, extracted_api_key = optionally_handle_anthropic_oauth(headers_capital, None)
    assert is_anthropic_oauth_key(extracted_api_key)
    assert updated_headers["anthropic-beta"] == "oauth-2025-04-20"

    # Test with lowercase (our test default)
    headers_lower = {"authorization": f"Bearer {FAKE_OAUTH_TOKEN}"}
    updated_headers, extracted_api_key = optionally_handle_anthropic_oauth(headers_lower, None)
    assert is_anthropic_oauth_key(extracted_api_key)

    # Test with mixed case
    headers_mixed = {"AUTHORIZATION": f"Bearer {FAKE_OAUTH_TOKEN}"}
    updated_headers, extracted_api_key = optionally_handle_anthropic_oauth(headers_mixed, None)
    assert is_anthropic_oauth_key(extracted_api_key)


def test_oauth_passthrough_endpoint_headers():
    """Test 9: Pass-through endpoint correctly builds OAuth headers"""
    from litellm.llms.anthropic.common_utils import (
        optionally_handle_anthropic_oauth,
        set_anthropic_headers,
    )

    # Simulate pass-through endpoint flow with OAuth token
    incoming_headers = {"authorization": f"Bearer {FAKE_OAUTH_TOKEN}"}
    oauth_headers, oauth_api_key = optionally_handle_anthropic_oauth(
        headers=incoming_headers, api_key=None
    )

    # Should extract OAuth API key
    assert oauth_api_key is not None
    assert oauth_api_key == FAKE_OAUTH_TOKEN

    # Build custom headers like pass-through endpoint does
    custom_headers = set_anthropic_headers(oauth_api_key)
    custom_headers["anthropic-dangerous-direct-browser-access"] = oauth_headers.get(
        "anthropic-dangerous-direct-browser-access", "true"
    )

    # Should use Authorization header, not x-api-key
    assert "x-api-key" not in custom_headers
    assert "authorization" in custom_headers
    assert custom_headers["authorization"] == f"Bearer {FAKE_OAUTH_TOKEN}"
    assert custom_headers["anthropic-dangerous-direct-browser-access"] == "true"


def test_regular_key_passthrough_endpoint_headers():
    """Test 10: Pass-through endpoint uses x-api-key for regular keys"""
    from litellm.llms.anthropic.common_utils import (
        optionally_handle_anthropic_oauth,
        set_anthropic_headers,
    )

    # Simulate pass-through endpoint flow with no OAuth token in request
    incoming_headers = {}
    oauth_headers, oauth_api_key = optionally_handle_anthropic_oauth(
        headers=incoming_headers, api_key=None
    )

    # No OAuth token found
    assert oauth_api_key is None

    # When no OAuth token, use regular x-api-key format
    regular_key = "sk-ant-api03-regular-key"
    custom_headers = set_anthropic_headers(regular_key)

    # Should use x-api-key, not authorization
    assert "authorization" not in custom_headers
    assert "x-api-key" in custom_headers
    assert custom_headers["x-api-key"] == regular_key