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


def test_cache_beta_headers_filtered():
    """Test filtering of invalid cache beta headers"""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    config = AnthropicModelInfo()

    # Test filtering single invalid cache header
    result = config._get_user_anthropic_beta_headers("prompt-caching-scope-2026-01-05")
    assert result is None or "prompt-caching-scope-2026-01-05" not in result

    # Test filtering invalid cache header from a list
    result = config._get_user_anthropic_beta_headers(
        "prompt-caching-scope-2026-01-05,max-tokens-3-5-sonnet-2022-07-15"
    )
    assert result is not None
    assert "prompt-caching-scope-2026-01-05" not in result
    assert "max-tokens-3-5-sonnet-2022-07-15" in result

    # Test that valid headers are preserved
    result = config._get_user_anthropic_beta_headers("computer-use-2024-10-22")
    assert result is not None
    assert "computer-use-2024-10-22" in result

    # Test multiple valid and invalid cache headers
    result = config._get_user_anthropic_beta_headers(
        "computer-use-2024-10-22,prompt-caching-scope-2026-01-05,max-tokens-3-5-sonnet-2022-07-15"
    )
    assert result is not None
    assert "prompt-caching-scope-2026-01-05" not in result
    assert "computer-use-2024-10-22" in result
    assert "max-tokens-3-5-sonnet-2022-07-15" in result


def test_cache_beta_headers_integration():
    """Test that invalid cache beta headers are filtered in validate_environment"""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    config = AnthropicModelInfo()
    headers = {
        "anthropic-beta": "prompt-caching-scope-2026-01-05,computer-use-2024-10-22"
    }

    updated_headers = config.validate_environment(
        headers=headers,
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        api_key="sk-ant-test-key",
        api_base=None,
    )

    # The invalid cache beta header should be filtered out
    anthropic_beta = updated_headers.get("anthropic-beta", "")
    assert "prompt-caching-scope-2026-01-05" not in anthropic_beta
    # Valid header should still be present
    assert "computer-use-2024-10-22" in anthropic_beta