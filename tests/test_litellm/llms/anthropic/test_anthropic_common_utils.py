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
    """Test filtering of cache beta headers for Vertex and Bedrock only"""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    config = AnthropicModelInfo()

    # Test that cache headers are NOT filtered for direct Anthropic API calls
    result = config._get_user_anthropic_beta_headers(
        "prompt-caching-scope-2026-01-05",
        is_vertex_request=False,
        is_bedrock_request=False
    )
    assert result is not None
    assert "prompt-caching-scope-2026-01-05" in result

    # Test filtering for Vertex requests
    result = config._get_user_anthropic_beta_headers(
        "prompt-caching-scope-2026-01-05,max-tokens-3-5-sonnet-2022-07-15",
        is_vertex_request=True,
        is_bedrock_request=False
    )
    assert result is not None
    assert "prompt-caching-scope-2026-01-05" not in result
    assert "max-tokens-3-5-sonnet-2022-07-15" in result

    # Test filtering for Bedrock requests
    result = config._get_user_anthropic_beta_headers(
        "prompt-caching-scope-2026-01-05,computer-use-2024-10-22",
        is_vertex_request=False,
        is_bedrock_request=True
    )
    assert result is not None
    assert "prompt-caching-scope-2026-01-05" not in result
    assert "computer-use-2024-10-22" in result

    # Test that valid headers are preserved for direct Anthropic calls
    result = config._get_user_anthropic_beta_headers(
        "computer-use-2024-10-22,prompt-caching-scope-2026-01-05",
        is_vertex_request=False,
        is_bedrock_request=False
    )
    assert result is not None
    assert "computer-use-2024-10-22" in result
    assert "prompt-caching-scope-2026-01-05" in result  # NOT filtered for direct Anthropic


def test_cache_beta_headers_integration():
    """Test that cache beta headers are filtered only for Vertex/Bedrock in validate_environment"""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    config = AnthropicModelInfo()
    headers = {
        "anthropic-beta": "prompt-caching-scope-2026-01-05,computer-use-2024-10-22"
    }

    # Test direct Anthropic API - cache headers should NOT be filtered
    updated_headers = config.validate_environment(
        headers=headers.copy(),
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        api_key="sk-ant-test-key",
        api_base=None,
    )
    anthropic_beta = updated_headers.get("anthropic-beta", "")
    assert "prompt-caching-scope-2026-01-05" in anthropic_beta  # NOT filtered for direct Anthropic
    assert "computer-use-2024-10-22" in anthropic_beta

    # Test Vertex request - NO user beta headers are sent (Vertex doesn't support them)
    updated_headers = config.validate_environment(
        headers=headers.copy(),
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"is_vertex_request": True},
        litellm_params={},
        api_key="sk-ant-test-key",
        api_base=None,
    )
    anthropic_beta = updated_headers.get("anthropic-beta", "")
    # Vertex doesn't support any user beta headers (except web_search which isn't in this test)
    assert "prompt-caching-scope-2026-01-05" not in anthropic_beta
    assert "computer-use-2024-10-22" not in anthropic_beta
    assert anthropic_beta == ""  # No beta headers for Vertex without web_search

    # Test Bedrock request - cache headers should be filtered, but other headers preserved
    updated_headers = config.validate_environment(
        headers=headers.copy(),
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"is_bedrock_request": True},
        litellm_params={},
        api_key="sk-ant-test-key",
        api_base=None,
    )
    anthropic_beta = updated_headers.get("anthropic-beta", "")
    assert "prompt-caching-scope-2026-01-05" not in anthropic_beta  # Cache header filtered for Bedrock
    assert "computer-use-2024-10-22" in anthropic_beta  # Other headers preserved for Bedrock