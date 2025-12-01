"""
Unit tests for Claude Code Chat Configuration and Transformation.

Tests message transformation, image handling, and configuration.
"""
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.claude_code.chat.transformation import ClaudeCodeChatConfig, ClaudeCodeError


class TestClaudeCodeChatConfig:
    """Tests for ClaudeCodeChatConfig class."""

    def test_custom_llm_provider(self):
        """Test that custom_llm_provider returns 'claude_code'."""
        config = ClaudeCodeChatConfig()
        assert config.custom_llm_provider == "claude_code"

    def test_get_supported_openai_params(self):
        """Test that supported OpenAI params are returned correctly."""
        config = ClaudeCodeChatConfig()
        params = config.get_supported_openai_params(model="claude-sonnet-4-5-20250929")

        assert "max_tokens" in params
        assert "temperature" in params
        assert "messages" in params
        assert "stream" in params
        assert "tools" in params
        assert "tool_choice" in params

    def test_get_claude_code_path_default(self):
        """Test that default claude code path is 'claude' when no custom path is set."""
        # Reset class-level attribute to ensure default behavior
        ClaudeCodeChatConfig.claude_code_path = None
        config = ClaudeCodeChatConfig()
        assert config.get_claude_code_path() == "claude"

    def test_get_claude_code_path_custom(self):
        """Test that custom claude code path is used when explicitly set."""
        config = ClaudeCodeChatConfig(claude_code_path="/custom/path/claude")
        assert config.get_claude_code_path() == "/custom/path/claude"
        # Reset for subsequent tests
        ClaudeCodeChatConfig.claude_code_path = None

    def test_get_disabled_tools_string(self):
        """Test that disabled tools are returned as comma-separated string."""
        config = ClaudeCodeChatConfig()
        disabled_tools = config.get_disabled_tools_string()

        assert "Task" in disabled_tools
        assert "Bash" in disabled_tools
        assert "Read" in disabled_tools
        assert "Edit" in disabled_tools
        assert "," in disabled_tools

    def test_disabled_tools_list(self):
        """Test that DISABLED_TOOLS contains expected tools."""
        config = ClaudeCodeChatConfig()

        expected_tools = [
            "Task", "Bash", "Glob", "Grep", "LS", "exit_plan_mode",
            "Read", "Edit", "MultiEdit", "Write", "NotebookRead",
            "NotebookEdit", "WebFetch", "TodoRead", "TodoWrite", "WebSearch",
        ]

        for tool in expected_tools:
            assert tool in config.DISABLED_TOOLS

    def test_map_openai_params_max_tokens(self):
        """Test mapping of max_tokens parameter."""
        config = ClaudeCodeChatConfig()
        optional_params = {}
        non_default_params = {"max_tokens": 1000}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="claude-sonnet-4-5-20250929",
        )

        assert result["max_tokens"] == 1000

    def test_map_openai_params_temperature(self):
        """Test mapping of temperature parameter."""
        config = ClaudeCodeChatConfig()
        optional_params = {}
        non_default_params = {"temperature": 0.7}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="claude-sonnet-4-5-20250929",
        )

        assert result["temperature"] == 0.7

    def test_map_openai_params_stream(self):
        """Test mapping of stream parameter."""
        config = ClaudeCodeChatConfig()
        optional_params = {}
        non_default_params = {"stream": True}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="claude-sonnet-4-5-20250929",
        )

        assert result["stream"] == True


class TestMessageTransformation:
    """Tests for message transformation logic."""

    def test_transform_simple_text_message(self):
        """Test transformation of simple text message."""
        config = ClaudeCodeChatConfig()
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]

        system_prompt, transformed = config.transform_messages_to_claude_code_format(messages)

        assert system_prompt == ""
        assert len(transformed) == 1
        assert transformed[0]["role"] == "user"
        assert transformed[0]["content"] == "Hello, how are you?"

    def test_transform_system_message_extraction(self):
        """Test that system message is extracted as system prompt."""
        config = ClaudeCodeChatConfig()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ]

        system_prompt, transformed = config.transform_messages_to_claude_code_format(messages)

        assert system_prompt == "You are a helpful assistant."
        assert len(transformed) == 1
        assert transformed[0]["role"] == "user"

    def test_transform_multi_turn_conversation(self):
        """Test transformation of multi-turn conversation."""
        config = ClaudeCodeChatConfig()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]

        system_prompt, transformed = config.transform_messages_to_claude_code_format(messages)

        assert len(transformed) == 3
        assert transformed[0]["role"] == "user"
        assert transformed[1]["role"] == "assistant"
        assert transformed[2]["role"] == "user"

    def test_transform_content_string(self):
        """Test _transform_content with string content."""
        config = ClaudeCodeChatConfig()
        result = config._transform_content("Hello world")
        assert result == "Hello world"

    def test_transform_content_text_block(self):
        """Test _transform_content with text content block."""
        config = ClaudeCodeChatConfig()
        content = [{"type": "text", "text": "Hello world"}]
        result = config._transform_content(content)

        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Hello world"

    def test_transform_content_anthropic_image(self):
        """Test _transform_content with Anthropic native image format."""
        config = ClaudeCodeChatConfig()
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                }
            }
        ]
        result = config._transform_content(content)

        # Anthropic format should pass through directly
        assert len(result) == 1
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "base64"

    def test_transform_content_openai_image_url_base64(self):
        """Test _transform_content converts OpenAI image_url base64 to Anthropic format."""
        config = ClaudeCodeChatConfig()
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD"
                }
            }
        ]
        result = config._transform_content(content)

        assert len(result) == 1
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "base64"
        assert result[0]["source"]["media_type"] == "image/jpeg"
        assert result[0]["source"]["data"] == "/9j/4AAQSkZJRgABAQEASABIAAD"

    def test_transform_content_openai_image_url_http(self):
        """Test _transform_content converts OpenAI image_url HTTP to Anthropic URL format."""
        config = ClaudeCodeChatConfig()
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image.jpg"
                }
            }
        ]
        result = config._transform_content(content)

        assert len(result) == 1
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "url"
        assert result[0]["source"]["url"] == "https://example.com/image.jpg"

    def test_transform_content_mixed_text_and_image(self):
        """Test _transform_content with mixed text and image content."""
        config = ClaudeCodeChatConfig()
        content = [
            {"type": "text", "text": "What is in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image.jpg"
                }
            }
        ]
        result = config._transform_content(content)

        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image"

    def test_transform_content_system_message_content_blocks(self):
        """Test transformation of system message with content blocks."""
        config = ClaudeCodeChatConfig()
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are helpful."},
                    {"type": "text", "text": "Be concise."}
                ]
            },
            {"role": "user", "content": "Hello"}
        ]

        system_prompt, transformed = config.transform_messages_to_claude_code_format(messages)

        assert system_prompt == "You are helpful.\nBe concise."
        assert len(transformed) == 1


class TestTransformRequest:
    """Tests for transform_request method."""

    def test_transform_request_basic(self):
        """Test basic request transformation."""
        config = ClaudeCodeChatConfig()
        messages = [
            {"role": "user", "content": "Hello"}
        ]

        result = config.transform_request(
            model="claude-sonnet-4-5-20250929",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        assert result["model"] == "claude-sonnet-4-5-20250929"
        assert result["system_prompt"] == ""
        assert len(result["messages"]) == 1
        assert "optional_params" in result

    def test_transform_request_with_system_prompt(self):
        """Test request transformation with system prompt."""
        config = ClaudeCodeChatConfig()
        messages = [
            {"role": "system", "content": "You are a pirate."},
            {"role": "user", "content": "Hello"}
        ]

        result = config.transform_request(
            model="claude-sonnet-4-5-20250929",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        assert result["system_prompt"] == "You are a pirate."


class TestValidateEnvironment:
    """Tests for validate_environment method."""

    def test_validate_environment_returns_empty_dict(self):
        """Test that validate_environment returns empty dict (no API key needed)."""
        config = ClaudeCodeChatConfig()

        result = config.validate_environment(
            headers={},
            model="claude-sonnet-4-5-20250929",
            messages=[],
            optional_params={},
            litellm_params={},
        )

        assert result == {}


class TestClaudeCodeError:
    """Tests for ClaudeCodeError exception class."""

    def test_claude_code_error_creation(self):
        """Test that ClaudeCodeError can be created with required attributes."""
        error = ClaudeCodeError(
            status_code=500,
            message="Test error message",
            headers={"Content-Type": "application/json"},
        )

        assert error.status_code == 500
        assert error.message == "Test error message"

    def test_get_error_class(self):
        """Test that get_error_class returns ClaudeCodeError."""
        config = ClaudeCodeChatConfig()

        error = config.get_error_class(
            error_message="Test error",
            status_code=400,
            headers={},
        )

        assert isinstance(error, ClaudeCodeError)
        assert error.status_code == 400
        assert error.message == "Test error"


class TestGetConfig:
    """Tests for get_config class method."""

    def test_get_config_excludes_disabled_tools(self):
        """Test that get_config excludes DISABLED_TOOLS from config dict."""
        config_dict = ClaudeCodeChatConfig.get_config()

        assert "DISABLED_TOOLS" not in config_dict

    def test_get_config_excludes_private_attributes(self):
        """Test that get_config excludes private attributes."""
        config_dict = ClaudeCodeChatConfig.get_config()

        for key in config_dict:
            assert not key.startswith("_")
