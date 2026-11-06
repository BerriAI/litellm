"""
Test that Anthropic guardrail write-back preserves tool_result.is_error and system output_config.

Regression tests for https://github.com/BerriAI/litellm/issues/50269
"""

from litellm.litellm_core_utils.prompt_templates.factory import convert_to_anthropic_tool_result
from litellm.llms.anthropic.chat.guardrail_translation.handler import AnthropicMessagesHandler
from litellm.types.llms.openai import ChatCompletionToolMessage


def test_convert_to_anthropic_tool_result_preserves_is_error_true():
    """Tool message with is_error=true must survive OpenAI->Anthropic conversion."""
    tool_message: ChatCompletionToolMessage = {
        "role": "tool",
        "tool_call_id": "toolu_01abc",
        "content": "cat: file.txt: No such file or directory",
        "is_error": True,
    }

    result = convert_to_anthropic_tool_result(tool_message)

    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "toolu_01abc"
    assert result["content"] == "cat: file.txt: No such file or directory"
    assert result.get("is_error") is True


def test_convert_to_anthropic_tool_result_preserves_is_error_false():
    """Tool message with is_error=false must survive OpenAI->Anthropic conversion."""
    tool_message: ChatCompletionToolMessage = {
        "role": "tool",
        "tool_call_id": "toolu_02xyz",
        "content": "success output",
        "is_error": False,
    }

    result = convert_to_anthropic_tool_result(tool_message)

    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "toolu_02xyz"
    assert result["content"] == "success output"
    assert result.get("is_error") is False


def test_convert_to_anthropic_tool_result_no_is_error():
    """Tool message without is_error field should not add one."""
    tool_message: ChatCompletionToolMessage = {
        "role": "tool",
        "tool_call_id": "toolu_03def",
        "content": "normal output",
    }

    result = convert_to_anthropic_tool_result(tool_message)

    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "toolu_03def"
    assert result["content"] == "normal output"
    assert "is_error" not in result


def test_openai_system_message_to_anthropic_preserves_output_config():
    """System message with output_config must survive OpenAI->Anthropic conversion."""
    openai_system_message = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "Environment update",
            }
        ],
        "output_config": {"effort": "medium"},
    }

    result = AnthropicMessagesHandler._openai_system_message_to_anthropic(openai_system_message)

    assert result is not None
    assert result["role"] == "system"
    assert isinstance(result["content"], list)
    assert len(result["content"]) == 1
    assert result["content"][0]["type"] == "text"
    assert result["content"][0]["text"] == "Environment update"
    assert result.get("output_config") == {"effort": "medium"}


def test_openai_system_message_to_anthropic_preserves_cache_control():
    """System message with cache_control must still be preserved."""
    openai_system_message = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "Cached content",
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }

    result = AnthropicMessagesHandler._openai_system_message_to_anthropic(openai_system_message)

    assert result is not None
    assert result["role"] == "system"
    assert isinstance(result["content"], list)
    assert len(result["content"]) == 1
    assert result["content"][0]["type"] == "text"
    assert result["content"][0]["text"] == "Cached content"
    assert result["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_openai_system_message_to_anthropic_preserves_both_fields():
    """System message with both output_config and cache_control must preserve both."""
    openai_system_message = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "Complex system message",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "output_config": {"effort": "high"},
    }

    result = AnthropicMessagesHandler._openai_system_message_to_anthropic(openai_system_message)

    assert result is not None
    assert result["role"] == "system"
    assert isinstance(result["content"], list)
    assert result["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert result.get("output_config") == {"effort": "high"}


def test_openai_system_message_to_anthropic_string_content_with_output_config():
    """System message with string content and output_config must preserve output_config."""
    openai_system_message = {
        "role": "system",
        "content": "Simple system prompt",
        "output_config": {"effort": "low"},
    }

    result = AnthropicMessagesHandler._openai_system_message_to_anthropic(openai_system_message)

    assert result is not None
    assert result["role"] == "system"
    assert result["content"] == "Simple system prompt"
    assert result.get("output_config") == {"effort": "low"}
