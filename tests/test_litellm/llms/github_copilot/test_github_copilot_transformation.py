import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import httpx
import pytest
from respx import MockRouter

import litellm

# Import at the top to make the patch work correctly
import litellm.llms.github_copilot.chat.transformation
from litellm import Choices, Message, ModelResponse, Usage, acompletion, completion
from litellm.exceptions import AuthenticationError
from litellm.llms.github_copilot.authenticator import Authenticator
from litellm.llms.github_copilot.chat.transformation import GithubCopilotConfig
from litellm.llms.github_copilot.common_utils import (
    APIKeyExpiredError,
    GetAccessTokenError,
    GetAPIKeyError,
    GetDeviceCodeError,
    RefreshAPIKeyError,
)


def test_github_copilot_config_get_openai_compatible_provider_info():
    """Test the GitHub Copilot configuration provider info retrieval."""

    config = GithubCopilotConfig()

    # Mock the authenticator to avoid actual API calls
    mock_api_key = "gh.test-key-123456789"
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = mock_api_key
    # Test with dynamic endpoint
    config.authenticator.get_api_base.return_value = "https://api.enterprise.githubcopilot.com"

    # Test with default values
    model = "github_copilot/gpt-4"
    (
        api_base,
        dynamic_api_key,
        custom_llm_provider,
    ) = config._get_openai_compatible_provider_info(
        model=model,
        api_base=None,
        api_key=None,
        custom_llm_provider="github_copilot",
    )

    assert api_base == "https://api.enterprise.githubcopilot.com"
    assert dynamic_api_key == mock_api_key
    assert custom_llm_provider == "github_copilot"

    # Test fallback to default if no dynamic endpoint
    config.authenticator.get_api_base.return_value = None
    (
        api_base,
        dynamic_api_key,
        custom_llm_provider,
    ) = config._get_openai_compatible_provider_info(
        model=model,
        api_base=None,
        api_key=None,
        custom_llm_provider="github_copilot",
    )
    assert api_base == "https://api.githubcopilot.com"

    # Test with authentication failure
    config.authenticator.get_api_key.side_effect = GetAPIKeyError(
        message="Failed to get API key",
        status_code=401,
    )

    with pytest.raises(AuthenticationError) as excinfo:
        config._get_openai_compatible_provider_info(
            model=model,
            api_base=None,
            api_key=None,
            custom_llm_provider="github_copilot",
        )

    assert "Failed to get API key" in str(excinfo.value)


@patch("litellm.llms.github_copilot.authenticator.Authenticator.get_api_key")
@patch("litellm.llms.openai.openai.OpenAIChatCompletion.completion")
def test_completion_github_copilot_mock_response(mock_completion, mock_get_api_key):
    """Test the completion function with GitHub Copilot provider."""

    # Mock the API key return value
    mock_api_key = "gh.test-key-123456789"
    mock_get_api_key.return_value = mock_api_key

    # Mock completion response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello, I'm GitHub Copilot!"
    mock_completion.return_value = mock_response

    # Test non-streaming completion
    messages = [
        {"role": "system", "content": "You're GitHub Copilot, an AI assistant."},
        {"role": "user", "content": "Hello, who are you?"},
    ]

    # Create a properly formatted headers dictionary
    headers = {
        "editor-version": "Neovim/0.9.0",
        "Copilot-Integration-Id": "vscode-chat",
    }

    response = completion(
        model="github_copilot/gpt-4",
        messages=messages,
        extra_headers=headers,
    )

    assert response is not None

    # Verify the get_api_key call was made (can be called multiple times)
    assert mock_get_api_key.call_count >= 1

    # Verify the completion call was made with the expected params
    mock_completion.assert_called_once()
    args, kwargs = mock_completion.call_args

    # Check that the proper authorization header is set
    assert "headers" in kwargs
    # Check that the model name is correctly formatted
    assert (
        kwargs.get("model") == "gpt-4"
    )  # Model name should be without provider prefix
    assert kwargs.get("messages") == messages


def test_transform_messages_disable_copilot_system_to_assistant(monkeypatch):
    """Test that system messages are converted to assistant unless disable_copilot_system_to_assistant is True."""
    import litellm
    from litellm.llms.github_copilot.chat.transformation import GithubCopilotConfig

    # Save original value
    original_flag = litellm.disable_copilot_system_to_assistant
    try:
        # Case 1: Flag is False (default, conversion happens)
        litellm.disable_copilot_system_to_assistant = False
        config = GithubCopilotConfig()
        messages = [
            {"role": "system", "content": "System message."},
            {"role": "user", "content": "User message."},
        ]
        out = config._transform_messages([m.copy() for m in messages], model="github_copilot/gpt-4")
        assert out[0]["role"] == "assistant"
        assert out[1]["role"] == "user"

        # Case 2: Flag is True (conversion does not happen)
        litellm.disable_copilot_system_to_assistant = True
        out = config._transform_messages([m.copy() for m in messages], model="github_copilot/gpt-4")
        assert out[0]["role"] == "system"
        assert out[1]["role"] == "user"

        # Case 3: Flag is False again (conversion happens)
        litellm.disable_copilot_system_to_assistant = False
        out = config._transform_messages([m.copy() for m in messages], model="github_copilot/gpt-4")
        assert out[0]["role"] == "assistant"
        assert out[1]["role"] == "user"
    finally:
        # Restore original value
        litellm.disable_copilot_system_to_assistant = original_flag


def test_x_initiator_header_user_request():
    """Test that user-only messages result in X-Initiator: user header"""
    config = GithubCopilotConfig()
    
    # Mock the authenticator
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123"
    config.authenticator.get_api_base.return_value = None

    messages = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    
    headers = config.validate_environment(
        headers={},
        model="github_copilot/gpt-4",
        messages=messages,
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )
    
    assert headers["X-Initiator"] == "user"


def test_x_initiator_header_agent_request_with_assistant():
    """Test that messages with assistant role result in X-Initiator: agent header"""
    config = GithubCopilotConfig()
    
    # Mock the authenticator
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123"
    config.authenticator.get_api_base.return_value = None

    messages = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "assistant", "content": "I can help you."},
    ]
    
    headers = config.validate_environment(
        headers={},
        model="github_copilot/gpt-4", 
        messages=messages,
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )
    
    assert headers["X-Initiator"] == "agent"


def test_x_initiator_header_agent_request_with_tool():
    """Test that messages with tool role result in X-Initiator: agent header"""
    config = GithubCopilotConfig()
    
    # Mock the authenticator
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123"
    config.authenticator.get_api_base.return_value = None

    messages = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "tool", "content": "Tool response.", "tool_call_id": "123"},
    ]
    
    headers = config.validate_environment(
        headers={},
        model="github_copilot/gpt-4", 
        messages=messages,
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )
    
    assert headers["X-Initiator"] == "agent"


def test_x_initiator_header_mixed_messages_with_agent_roles():
    """Test that mixed messages with agent roles (assistant/tool) result in X-Initiator: agent header"""
    config = GithubCopilotConfig()
    
    # Mock the authenticator  
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123"
    config.authenticator.get_api_base.return_value = None

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Previous response."},
        {"role": "user", "content": "Follow up question."},
    ]
    
    headers = config.validate_environment(
        headers={},
        model="github_copilot/gpt-4",
        messages=messages, 
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )
    
    assert headers["X-Initiator"] == "agent"


def test_x_initiator_header_user_only_messages():
    """Test that user + system only messages result in X-Initiator: user header"""
    config = GithubCopilotConfig()
    
    # Mock the authenticator  
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123"
    config.authenticator.get_api_base.return_value = None

    messages = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "Follow up question."},
    ]
    
    headers = config.validate_environment(
        headers={},
        model="github_copilot/gpt-4",
        messages=messages, 
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )
    
    assert headers["X-Initiator"] == "user"


def test_x_initiator_header_empty_messages():
    """Test that empty messages result in X-Initiator: user header"""
    config = GithubCopilotConfig()
    
    # Mock the authenticator
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123"
    config.authenticator.get_api_base.return_value = None

    messages = []
    
    headers = config.validate_environment(
        headers={},
        model="github_copilot/gpt-4",
        messages=messages,
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )
    
    assert headers["X-Initiator"] == "user"


def test_x_initiator_header_system_only_messages():
    """Test that system-only messages result in X-Initiator: user header"""
    config = GithubCopilotConfig()
    
    # Mock the authenticator
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123"
    config.authenticator.get_api_base.return_value = None

    messages = [
        {"role": "system", "content": "You are an assistant."},
    ]
    
    headers = config.validate_environment(
        headers={},
        model="github_copilot/gpt-4",
        messages=messages,
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )
    
    assert headers["X-Initiator"] == "user"


def test_get_supported_openai_params_claude_model():
    """Test that Claude models with extended thinking support have thinking and reasoning parameters."""
    config = GithubCopilotConfig()
    
    # Test Claude 4 model supports thinking and reasoning_effort parameters
    supported_params = config.get_supported_openai_params("claude-sonnet-4-20250514")
    assert "thinking" in supported_params
    assert "reasoning_effort" in supported_params
    
    # Test Claude 3-7 model supports thinking and reasoning_effort parameters
    supported_params_claude37 = config.get_supported_openai_params("claude-3-7-sonnet-20250219")
    assert "thinking" in supported_params_claude37
    assert "reasoning_effort" in supported_params_claude37
    
    # Test Claude 3.5 model does NOT support thinking parameters (no extended thinking)
    supported_params_claude35 = config.get_supported_openai_params("claude-3.5-sonnet")
    assert "thinking" not in supported_params_claude35
    assert "reasoning_effort" not in supported_params_claude35
    
    # Test non-Claude model doesn't include thinking parameters but may include reasoning_effort
    supported_params_gpt = config.get_supported_openai_params("gpt-4o")
    assert "thinking" not in supported_params_gpt
    # gpt-4o should NOT have reasoning_effort (not a reasoning model)
    assert "reasoning_effort" not in supported_params_gpt
    
    # Test O-series reasoning models include reasoning_effort but not thinking
    supported_params_o3 = config.get_supported_openai_params("o3-mini")
    assert "thinking" not in supported_params_o3
    # o3-mini should have reasoning_effort (it's an O-series reasoning model)
    assert "reasoning_effort" in supported_params_o3


def test_get_supported_openai_params_case_insensitive():
    """Test that Claude model detection is case-insensitive for models with extended thinking."""
    config = GithubCopilotConfig()
    
    # Test uppercase Claude 4 model with full model name
    supported_params_upper = config.get_supported_openai_params("CLAUDE-SONNET-4-20250514")
    assert "thinking" in supported_params_upper
    assert "reasoning_effort" in supported_params_upper
    
    # Test mixed case Claude 3-7 model (has extended thinking) with full model name
    supported_params_mixed = config.get_supported_openai_params("Claude-3-7-Sonnet-20250219")
    assert "thinking" in supported_params_mixed
    assert "reasoning_effort" in supported_params_mixed
    
    # Test that Claude 3.5 models don't have thinking support (case insensitive)
    supported_params_35 = config.get_supported_openai_params("CLAUDE-3.5-SONNET")
    assert "thinking" not in supported_params_35
    assert "reasoning_effort" not in supported_params_35

def test_copilot_vision_request_header_with_image():
    """Test that Copilot-Vision-Request header is added when messages contain images"""
    config = GithubCopilotConfig()
    
    # Mock the authenticator
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123"
    config.authenticator.get_api_base.return_value = None

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,abc123"}
                }
            ]
        }
    ]
    
    headers = config.validate_environment(
        headers={},
        model="github_copilot/gpt-4-vision-preview",
        messages=messages,
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )
    
    assert headers["Copilot-Vision-Request"] == "true"
    assert headers["X-Initiator"] == "user"


def test_copilot_vision_request_header_text_only():
    """Test that Copilot-Vision-Request header is not added for text-only messages"""
    config = GithubCopilotConfig()
    
    # Mock the authenticator
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123"
    config.authenticator.get_api_base.return_value = None

    messages = [
        {"role": "user", "content": "Just a text message"},
    ]
    
    headers = config.validate_environment(
        headers={},
        model="github_copilot/gpt-4",
        messages=messages,
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )
    
    assert "Copilot-Vision-Request" not in headers
    assert headers["X-Initiator"] == "user"


def test_copilot_vision_request_header_with_type_image_url():
    """Test that Copilot-Vision-Request header is added for content with type: image_url"""
    config = GithubCopilotConfig()

    # Mock the authenticator
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123"
    config.authenticator.get_api_base.return_value = None

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
            ]
        }
    ]

    headers = config.validate_environment(
        headers={},
        model="github_copilot/gpt-4-vision-preview",
        messages=messages,
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )

    assert headers["Copilot-Vision-Request"] == "true"
    assert headers["X-Initiator"] == "user"


# ==================== Tool Result Consolidation Tests ====================


def test_consolidate_openai_tool_messages_no_duplicates():
    """Test that messages without duplicate tool_call_ids are unchanged"""
    config = GithubCopilotConfig()

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "function": {"name": "tool1", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "Result 1"},
        {"role": "tool", "tool_call_id": "call_2", "content": "Result 2"},
    ]

    result = config._consolidate_openai_tool_messages(messages)

    # Messages should be unchanged
    assert len(result) == 4
    assert result[2]["tool_call_id"] == "call_1"
    assert result[2]["content"] == "Result 1"
    assert result[3]["tool_call_id"] == "call_2"
    assert result[3]["content"] == "Result 2"


def test_consolidate_openai_tool_messages_with_duplicates():
    """Test that duplicate tool messages with same tool_call_id are consolidated"""
    config = GithubCopilotConfig()

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "function": {"name": "tool1", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "Result part 1"},
        {"role": "tool", "tool_call_id": "call_1", "content": "Result part 2"},
        {"role": "tool", "tool_call_id": "call_1", "content": "Result part 3"},
    ]

    result = config._consolidate_openai_tool_messages(messages)

    # Should have 3 messages: user, assistant, and one consolidated tool message
    assert len(result) == 3
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"
    assert result[2]["role"] == "tool"
    assert result[2]["tool_call_id"] == "call_1"
    assert "Result part 1" in result[2]["content"]
    assert "Result part 2" in result[2]["content"]
    assert "Result part 3" in result[2]["content"]


def test_consolidate_openai_tool_messages_mixed():
    """Test consolidation with mixed duplicate and unique tool messages"""
    config = GithubCopilotConfig()

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "tool", "tool_call_id": "call_1", "content": "Result 1a"},
        {"role": "tool", "tool_call_id": "call_1", "content": "Result 1b"},
        {"role": "tool", "tool_call_id": "call_2", "content": "Result 2"},
    ]

    result = config._consolidate_openai_tool_messages(messages)

    # Should have 3 messages: user, consolidated call_1, and call_2
    assert len(result) == 3
    assert result[0]["role"] == "user"
    assert result[1]["tool_call_id"] == "call_1"
    assert "Result 1a" in result[1]["content"]
    assert "Result 1b" in result[1]["content"]
    assert result[2]["tool_call_id"] == "call_2"
    assert result[2]["content"] == "Result 2"


def test_consolidate_anthropic_tool_results_no_duplicates():
    """Test that Anthropic-style messages without duplicate tool_use_ids are unchanged"""
    config = GithubCopilotConfig()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Result 1"},
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": "Result 2"},
            ]
        }
    ]

    result = config._consolidate_anthropic_tool_results(messages)

    # Messages should be unchanged
    assert len(result) == 1
    assert len(result[0]["content"]) == 2
    assert result[0]["content"][0]["tool_use_id"] == "toolu_1"
    assert result[0]["content"][1]["tool_use_id"] == "toolu_2"


def test_consolidate_anthropic_tool_results_with_duplicates():
    """Test that Anthropic-style tool_result blocks with same tool_use_id are consolidated"""
    config = GithubCopilotConfig()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Result part 1"},
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Result part 2"},
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Result part 3"},
            ]
        }
    ]

    result = config._consolidate_anthropic_tool_results(messages)

    # Should have 1 message with 1 consolidated tool_result
    assert len(result) == 1
    assert len(result[0]["content"]) == 1
    assert result[0]["content"][0]["tool_use_id"] == "toolu_1"
    assert "Result part 1" in result[0]["content"][0]["content"]
    assert "Result part 2" in result[0]["content"][0]["content"]
    assert "Result part 3" in result[0]["content"][0]["content"]


def test_consolidate_anthropic_tool_results_mixed_content():
    """Test consolidation with mixed tool_results and other content types"""
    config = GithubCopilotConfig()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here are the results:"},
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Result 1a"},
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Result 1b"},
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": "Result 2"},
            ]
        }
    ]

    result = config._consolidate_anthropic_tool_results(messages)

    # Should have 1 message with text + 2 tool_results (one consolidated, one original)
    assert len(result) == 1
    content = result[0]["content"]
    assert len(content) == 3  # text + consolidated toolu_1 + toolu_2

    # First should be the text
    assert content[0]["type"] == "text"

    # Second should be consolidated toolu_1
    assert content[1]["type"] == "tool_result"
    assert content[1]["tool_use_id"] == "toolu_1"
    assert "Result 1a" in content[1]["content"]
    assert "Result 1b" in content[1]["content"]

    # Third should be toolu_2
    assert content[2]["type"] == "tool_result"
    assert content[2]["tool_use_id"] == "toolu_2"
    assert content[2]["content"] == "Result 2"


def test_merge_tool_results_string_content():
    """Test merging tool_results with string content"""
    config = GithubCopilotConfig()

    tool_results = [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Part 1"},
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Part 2"},
    ]

    merged = config._merge_tool_results(tool_results)

    assert merged["type"] == "tool_result"
    assert merged["tool_use_id"] == "toolu_1"
    assert merged["content"] == "Part 1\nPart 2"


def test_merge_tool_results_list_content():
    """Test merging tool_results with list content"""
    config = GithubCopilotConfig()

    tool_results = [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "Part 1"}]},
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "Part 2"}]},
    ]

    merged = config._merge_tool_results(tool_results)

    assert merged["type"] == "tool_result"
    assert merged["tool_use_id"] == "toolu_1"
    assert isinstance(merged["content"], list)
    assert len(merged["content"]) == 2
    assert merged["content"][0]["text"] == "Part 1"
    assert merged["content"][1]["text"] == "Part 2"


def test_merge_tool_results_preserves_is_error():
    """Test that is_error flag is preserved when any result has it True"""
    config = GithubCopilotConfig()

    tool_results = [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Success"},
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Error occurred", "is_error": True},
    ]

    merged = config._merge_tool_results(tool_results)

    assert merged["is_error"] is True


def test_merge_tool_results_preserves_cache_control():
    """Test that cache_control is preserved from first result"""
    config = GithubCopilotConfig()

    tool_results = [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Part 1", "cache_control": {"type": "ephemeral"}},
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Part 2"},
    ]

    merged = config._merge_tool_results(tool_results)

    assert "cache_control" in merged
    assert merged["cache_control"]["type"] == "ephemeral"


def test_transform_messages_consolidates_tool_results():
    """Test that _transform_messages consolidates duplicate tool results"""
    import litellm

    config = GithubCopilotConfig()

    # Save original value
    original_flag = litellm.disable_copilot_system_to_assistant
    try:
        litellm.disable_copilot_system_to_assistant = True  # Don't modify system messages

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "function": {"name": "tool1", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "Part 1"},
            {"role": "tool", "tool_call_id": "call_1", "content": "Part 2"},
        ]

        result = config._transform_messages(messages, model="github_copilot/claude-sonnet-4")

        # Should consolidate duplicate tool messages
        assert len(result) == 3
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "call_1"
        assert "Part 1" in result[2]["content"]
        assert "Part 2" in result[2]["content"]
    finally:
        litellm.disable_copilot_system_to_assistant = original_flag


def test_consolidate_tool_results_no_tool_messages():
    """Test that messages without any tool content are unchanged"""
    config = GithubCopilotConfig()

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    result = config._consolidate_tool_results(messages)

    # Messages should be unchanged
    assert len(result) == 2
    assert result[0]["content"] == "Hello"
    assert result[1]["content"] == "Hi there!"


def test_merge_tool_messages_with_name():
    """Test that name is preserved when merging tool messages"""
    config = GithubCopilotConfig()

    tool_messages = [
        {"role": "tool", "tool_call_id": "call_1", "content": "Part 1", "name": "my_tool"},
        {"role": "tool", "tool_call_id": "call_1", "content": "Part 2"},
    ]

    merged = config._merge_tool_messages(tool_messages)

    assert merged["name"] == "my_tool"


def test_consolidate_anthropic_tool_results_no_content_list():
    """Test that messages with non-list content are unchanged"""
    config = GithubCopilotConfig()

    messages = [
        {"role": "user", "content": "Just a string content"},
        {"role": "assistant", "content": "Response"},
    ]

    result = config._consolidate_anthropic_tool_results(messages)

    # Messages should be unchanged
    assert result == messages
