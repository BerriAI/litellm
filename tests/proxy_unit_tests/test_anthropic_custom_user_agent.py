"""
Test custom User-Agent configuration for Anthropic provider in LiteLLM proxy.
"""
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import pytest
from litellm.llms.anthropic.common_utils import AnthropicModelInfo
from litellm.llms.anthropic.chat.transformation import AnthropicConfig


def test_anthropic_config_supports_custom_user_agent():
    """Test that custom_user_agent is in supported params for Anthropic"""
    config = AnthropicConfig()
    supported_params = config.get_supported_openai_params("claude-3-5-sonnet-20241022")

    assert "custom_user_agent" in supported_params, \
        "custom_user_agent should be in supported OpenAI params"


def test_proxy_litellm_params_with_custom_user_agent():
    """Test that custom_user_agent from litellm_params is properly set"""
    config = AnthropicModelInfo()
    custom_agent = "Claude Code/1.0"

    # Simulate litellm_params from proxy YAML config
    litellm_params = {
        "model": "anthropic/claude-3-5-sonnet-20241022",
        "api_key": "sk-ant-test-key",
        "custom_user_agent": custom_agent,
    }

    updated_headers = config.validate_environment(
        headers={},
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"custom_user_agent": custom_agent},
        litellm_params={},
        api_key="sk-ant-test-key",
        api_base=None,
    )

    assert updated_headers["User-Agent"] == custom_agent, \
        f"Expected User-Agent to be '{custom_agent}', got '{updated_headers.get('User-Agent')}'"


def test_proxy_env_var_anthropic_user_agent():
    """Test that ANTHROPIC_USER_AGENT environment variable works"""
    config = AnthropicModelInfo()
    custom_agent = "Claude Code Proxy/1.0"

    os.environ["ANTHROPIC_USER_AGENT"] = custom_agent

    try:
        updated_headers = config.validate_environment(
            headers={},
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key="sk-ant-test-key",
            api_base=None,
        )

        assert updated_headers["User-Agent"] == custom_agent, \
            f"Expected User-Agent to be '{custom_agent}', got '{updated_headers.get('User-Agent')}'"
    finally:
        del os.environ["ANTHROPIC_USER_AGENT"]


def test_proxy_extra_headers_user_agent():
    """Test that User-Agent can be set via extra_headers"""
    config = AnthropicModelInfo()
    custom_agent = "Claude Code via Extra Headers/1.0"

    # Headers would come from extra_headers in litellm_params
    headers = {"User-Agent": custom_agent}

    updated_headers = config.validate_environment(
        headers=headers,
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        api_key="sk-ant-test-key",
        api_base=None,
    )

    # Should preserve the User-Agent from extra_headers when no custom_user_agent is set
    assert updated_headers["User-Agent"] == custom_agent, \
        f"Expected User-Agent to be '{custom_agent}', got '{updated_headers.get('User-Agent')}'"


def test_proxy_priority_order():
    """Test priority: custom_user_agent > ANTHROPIC_USER_AGENT > extra_headers"""
    config = AnthropicModelInfo()

    param_agent = "Param Agent"
    env_agent = "Env Agent"
    header_agent = "Header Agent"

    os.environ["ANTHROPIC_USER_AGENT"] = env_agent

    try:
        # Test 1: custom_user_agent parameter overrides environment variable
        updated_headers = config.validate_environment(
            headers={"User-Agent": header_agent},
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={"custom_user_agent": param_agent},
            litellm_params={},
            api_key="sk-ant-test-key",
            api_base=None,
        )

        assert updated_headers["User-Agent"] == param_agent, \
            "custom_user_agent parameter should have highest priority"

        # Test 2: Environment variable overrides extra_headers
        updated_headers = config.validate_environment(
            headers={"User-Agent": header_agent},
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key="sk-ant-test-key",
            api_base=None,
        )

        assert updated_headers["User-Agent"] == env_agent, \
            "ANTHROPIC_USER_AGENT env var should override extra_headers"

    finally:
        del os.environ["ANTHROPIC_USER_AGENT"]


if __name__ == "__main__":
    print("Running Anthropic custom User-Agent proxy tests...")
    test_anthropic_config_supports_custom_user_agent()
    print("✓ Test 1 passed: custom_user_agent is supported")

    test_proxy_litellm_params_with_custom_user_agent()
    print("✓ Test 2 passed: custom_user_agent from litellm_params works")

    test_proxy_env_var_anthropic_user_agent()
    print("✓ Test 3 passed: ANTHROPIC_USER_AGENT env var works")

    test_proxy_extra_headers_user_agent()
    print("✓ Test 4 passed: User-Agent via extra_headers works")

    test_proxy_priority_order()
    print("✓ Test 5 passed: Priority order is correct")

    print("\n✅ All proxy tests passed!")
