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


def test_custom_user_agent_via_parameter():
    """Test 5: Custom User-Agent via custom_user_agent parameter"""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    config = AnthropicModelInfo()
    custom_agent = "Claude Code/1.0"

    updated_headers = config.validate_environment(
        headers={},
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"custom_user_agent": custom_agent},
        litellm_params={},
        api_key="sk-ant-api03-test-key",
        api_base=None,
    )

    assert updated_headers["User-Agent"] == custom_agent


def test_custom_user_agent_via_env_var():
    """Test 6: Custom User-Agent via ANTHROPIC_USER_AGENT environment variable"""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    config = AnthropicModelInfo()
    custom_agent = "Claude Code/1.0"

    # Set environment variable
    os.environ["ANTHROPIC_USER_AGENT"] = custom_agent

    try:
        updated_headers = config.validate_environment(
            headers={},
            model="claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key="sk-ant-api03-test-key",
            api_base=None,
        )

        assert updated_headers["User-Agent"] == custom_agent
    finally:
        # Clean up
        del os.environ["ANTHROPIC_USER_AGENT"]


def test_custom_user_agent_parameter_priority():
    """Test 7: custom_user_agent parameter takes priority over environment variable"""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    config = AnthropicModelInfo()
    param_agent = "Claude Code/2.0"
    env_agent = "Claude Code/1.0"

    # Set environment variable
    os.environ["ANTHROPIC_USER_AGENT"] = env_agent

    try:
        updated_headers = config.validate_environment(
            headers={},
            model="claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={"custom_user_agent": param_agent},
            litellm_params={},
            api_key="sk-ant-api03-test-key",
            api_base=None,
        )

        # Parameter should take priority over environment variable
        assert updated_headers["User-Agent"] == param_agent
    finally:
        # Clean up
        del os.environ["ANTHROPIC_USER_AGENT"]


def test_no_custom_user_agent():
    """Test 8: Default User-Agent is preserved when no custom agent is provided"""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    config = AnthropicModelInfo()
    default_agent = "litellm/1.0.0"

    updated_headers = config.validate_environment(
        headers={"User-Agent": default_agent},
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        api_key="sk-ant-api03-test-key",
        api_base=None,
    )

    # Default User-Agent should be preserved
    assert updated_headers["User-Agent"] == default_agent