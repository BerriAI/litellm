"""
Tests for Claude Code Native Provider

This module tests the Claude Code Native provider, a variant of Anthropic
with specific headers and system prompt requirements.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.claude_code_native.transformation import ClaudeCodeNativeConfig


class TestClaudeCodeNativeConfig:
    """Test suite for ClaudeCodeNativeConfig class."""

    def test_provider_name(self):
        """Test that the provider returns the correct custom_llm_provider name."""
        config = ClaudeCodeNativeConfig()
        assert config.custom_llm_provider == "claude_code_native"

    def test_required_headers_constant(self):
        """Test that required headers are correctly defined."""
        assert hasattr(ClaudeCodeNativeConfig, "REQUIRED_HEADERS")
        assert isinstance(ClaudeCodeNativeConfig.REQUIRED_HEADERS, dict)
        assert "anthropic-beta" in ClaudeCodeNativeConfig.REQUIRED_HEADERS
        assert "anthropic-version" in ClaudeCodeNativeConfig.REQUIRED_HEADERS
        assert ClaudeCodeNativeConfig.REQUIRED_HEADERS["anthropic-beta"] == "oauth-2025-04-20"
        assert ClaudeCodeNativeConfig.REQUIRED_HEADERS["anthropic-version"] == "2023-06-01"

    def test_system_prompt_constant(self):
        """Test that the system prompt is correctly defined."""
        assert hasattr(ClaudeCodeNativeConfig, "CLAUDE_CODE_SYSTEM_PROMPT")
        assert isinstance(ClaudeCodeNativeConfig.CLAUDE_CODE_SYSTEM_PROMPT, str)
        assert "Claude Code" in ClaudeCodeNativeConfig.CLAUDE_CODE_SYSTEM_PROMPT
        assert ClaudeCodeNativeConfig.CLAUDE_CODE_SYSTEM_PROMPT == "You are Claude Code, Anthropic's official CLI for Claude."

    def test_validate_environment_adds_required_headers(self):
        """Test that validate_environment adds required headers."""
        config = ClaudeCodeNativeConfig()
        result = config.validate_environment(
            headers={"x-api-key": "test-key"},
            model="claude-4-5-sonnet",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-api-key",
        )

        # Check that required headers are present
        assert result.get("anthropic-beta") == "oauth-2025-04-20"
        assert result.get("anthropic-version") == "2023-06-01"
        # Should use Authorization: Bearer instead of x-api-key
        assert result.get("Authorization") == "Bearer test-api-key"
        assert "x-api-key" not in result

    def test_update_headers_with_optional_anthropic_beta_preserves_required(self):
        """Test that update_headers_with_optional_anthropic_beta preserves required beta header."""
        config = ClaudeCodeNativeConfig()
        headers = {"anthropic-beta": "some-other-beta"}
        
        result = config.update_headers_with_optional_anthropic_beta(
            headers=headers,
            optional_params={"output_format": {"type": "json_schema"}}
        )
        
        # Should contain both the required beta and the one added by parent
        assert "oauth-2025-04-20" in result["anthropic-beta"]
        assert "structured-outputs-2025-11-13" in result["anthropic-beta"]

    def test_validate_environment_headers_take_priority(self):
        """Test that provider headers take priority over user extra_headers."""
        config = ClaudeCodeNativeConfig()
        result = config.validate_environment(
            headers={"x-api-key": "test-key"},
            model="claude-4-5-sonnet",
            messages=[],
            optional_params={
                "extra_headers": {
                    "anthropic-beta": "user-beta-value",
                    "anthropic-version": "user-version",
                    "x-custom": "custom-value",
                }
            },
            litellm_params={},
            api_key="test-api-key",
        )

        # Provider headers should take priority for required keys
        assert result.get("anthropic-beta") == "oauth-2025-04-20", "Provider beta should override user beta"
        assert result.get("anthropic-version") == "2023-06-01", "Provider version should override user version"
        # Custom header should still be included
        assert result.get("x-custom") == "custom-value", "Custom header should be preserved"

    def test_transform_request_prepends_system_message(self):
        """Test that transform_request prepends the Claude Code system message."""
        config = ClaudeCodeNativeConfig()
        messages = [{"role": "user", "content": "Hello"}]

        result = config.transform_request(
            model="claude-4-5-sonnet",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Check that system message is present
        assert "system" in result
        assert isinstance(result["system"], list)

        # Check that the first system message is the Claude Code prompt
        first_system_message = result["system"][0]
        assert first_system_message.get("type") == "text"
        assert (
            first_system_message.get("text")
            == ClaudeCodeNativeConfig.CLAUDE_CODE_SYSTEM_PROMPT
        )

    def test_transform_request_prepends_before_existing_system_messages(self):
        """Test that transform_request prepends Claude Code prompt before existing system messages."""
        config = ClaudeCodeNativeConfig()
        messages = [
            {"role": "system", "content": "Existing system message"},
            {"role": "user", "content": "Hello"},
        ]

        result = config.transform_request(
            model="claude-4-5-sonnet",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Check that system messages are present
        assert "system" in result
        assert isinstance(result["system"], list)
        assert len(result["system"]) == 2

        # First should be Claude Code prompt
        assert result["system"][0].get("text") == ClaudeCodeNativeConfig.CLAUDE_CODE_SYSTEM_PROMPT

        # Second should be original system message (extracted from messages)
        assert result["system"][1].get("text") == "Existing system message"

    def test_transform_request_with_system_in_optional_params(self):
        """Test that transform_request handles system messages already in optional_params."""
        config = ClaudeCodeNativeConfig()
        messages = [{"role": "user", "content": "Hello"}]

        result = config.transform_request(
            model="claude-4-5-sonnet",
            messages=messages,
            optional_params={
                "system": [{"type": "text", "text": "Optional params system"}]
            },
            litellm_params={},
            headers={},
        )

        # Check that system messages are present
        assert "system" in result
        assert isinstance(result["system"], list)
        assert len(result["system"]) == 2

        # First should be Claude Code prompt
        assert result["system"][0].get("text") == ClaudeCodeNativeConfig.CLAUDE_CODE_SYSTEM_PROMPT

        # Second should be the one from optional_params
        assert result["system"][1].get("text") == "Optional params system"

    def test_transform_request_with_string_system_in_optional_params(self):
        """Test that transform_request handles string system in optional_params."""
        config = ClaudeCodeNativeConfig()
        messages = [{"role": "user", "content": "Hello"}]

        result = config.transform_request(
            model="claude-4-5-sonnet",
            messages=messages,
            optional_params={"system": "String system message"},
            litellm_params={},
            headers={},
        )

        # Check that system messages are present
        assert "system" in result
        assert isinstance(result["system"], list)
        assert len(result["system"]) >= 1

        # First should be Claude Code prompt
        assert result["system"][0].get("text") == ClaudeCodeNativeConfig.CLAUDE_CODE_SYSTEM_PROMPT

        # String system message may be converted or preserved depending on parent class behavior
        # The key requirement is that Claude Code prompt is first


class TestClaudeCodeNativeChatCompletion:
    """Test suite for ClaudeCodeNativeChatCompletion class."""

    def test_chat_completion_import(self):
        """Test that ClaudeCodeNativeChatCompletion can be imported."""
        from litellm.llms.claude_code_native.completion import (
            ClaudeCodeNativeChatCompletion,
        )

        # Verify class can be instantiated
        completion = ClaudeCodeNativeChatCompletion()
        assert completion is not None

    def test_chat_completion_inherits_from_anthropic(self):
        """Test that ClaudeCodeNativeChatCompletion inherits from AnthropicChatCompletion."""
        from litellm.llms.anthropic.chat.handler import AnthropicChatCompletion
        from litellm.llms.claude_code_native.completion import (
            ClaudeCodeNativeChatCompletion,
        )

        # Verify inheritance
        assert issubclass(ClaudeCodeNativeChatCompletion, AnthropicChatCompletion)


def test_completion_factory_function():
    """Test that the completion factory function works."""
    from litellm.llms.claude_code_native.completion import completion

    result = completion()
    assert result is not None
    assert hasattr(result, "completion")
