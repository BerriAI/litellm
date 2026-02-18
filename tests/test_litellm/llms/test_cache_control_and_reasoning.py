"""
Test cache_control and reasoning parameter support for MiniMax, GLM/ZAI, and OpenRouter.

This test file verifies the fixes for Issue #19923:
- cache_control is preserved (not stripped) for MiniMax, GLM, and OpenRouter variants
- thinking parameter is supported for reasoning-capable models
- Model metadata correctly reflects capabilities
"""
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.llms.minimax.chat.transformation import MinimaxChatConfig
from litellm.llms.openrouter.chat.transformation import OpenrouterConfig
from litellm.llms.zai.chat.transformation import ZAIChatConfig


def test_minimax_preserves_cache_control_in_messages():
    """MiniMax should NOT strip cache_control from messages."""
    config = MinimaxChatConfig()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
            "cache_control": {"type": "ephemeral"},
        },
        {
            "role": "user",
            "content": "Hello, world!",
        },
    ]

    transformed_messages, _ = config.remove_cache_control_flag_from_messages_and_tools(
        model="minimax/MiniMax-M2.1", messages=messages
    )

    # cache_control should be preserved
    assert transformed_messages[0].get("cache_control") == {"type": "ephemeral"}


def test_minimax_preserves_cache_control_in_tools():
    """MiniMax should NOT strip cache_control from tools."""
    config = MinimaxChatConfig()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {"type": "object", "properties": {}},
            },
            "cache_control": {"type": "ephemeral"},
        }
    ]

    _, transformed_tools = config.remove_cache_control_flag_from_messages_and_tools(
        model="minimax/MiniMax-M2.1", messages=[], tools=tools
    )

    # cache_control should be preserved
    assert transformed_tools[0].get("cache_control") == {"type": "ephemeral"}


def test_minimax_supports_thinking_param():
    """MiniMax reasoning models should support thinking parameter."""
    config = MinimaxChatConfig()

    supported_params = config.get_supported_openai_params(
        model="minimax/MiniMax-M2.1"
    )

    # thinking should be in supported params for reasoning models
    assert "thinking" in supported_params
    # reasoning_split should also be supported
    assert "reasoning_split" in supported_params


def test_zai_preserves_cache_control_in_messages():
    """ZAI should NOT strip cache_control from messages."""
    config = ZAIChatConfig()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
            "cache_control": {"type": "ephemeral"},
        },
        {
            "role": "user",
            "content": "Hello, world!",
        },
    ]

    transformed_messages, _ = config.remove_cache_control_flag_from_messages_and_tools(
        model="zai/glm-4.7", messages=messages
    )

    # cache_control should be preserved
    assert transformed_messages[0].get("cache_control") == {"type": "ephemeral"}


def test_zai_preserves_cache_control_in_tools():
    """ZAI should NOT strip cache_control from tools."""
    config = ZAIChatConfig()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {"type": "object", "properties": {}},
            },
            "cache_control": {"type": "ephemeral"},
        }
    ]

    _, transformed_tools = config.remove_cache_control_flag_from_messages_and_tools(
        model="zai/glm-4.7", messages=[], tools=tools
    )

    # cache_control should be preserved
    assert transformed_tools[0].get("cache_control") == {"type": "ephemeral"}


def test_zai_supports_thinking_param_for_reasoning_models():
    """ZAI reasoning models (glm-4.7, glm-4.6) should support thinking parameter."""
    config = ZAIChatConfig()

    # glm-4.7 supports reasoning
    supported_params_47 = config.get_supported_openai_params(model="zai/glm-4.7")
    assert "thinking" in supported_params_47

    # glm-4.6 supports reasoning
    supported_params_46 = config.get_supported_openai_params(model="zai/glm-4.6")
    assert "thinking" in supported_params_46


def test_openrouter_minimax_supports_cache_control():
    """OpenRouter should preserve cache_control for MiniMax models."""
    config = OpenrouterConfig()

    messages = [
        {
            "role": "user",
            "content": "Hello, world!",
            "cache_control": {"type": "ephemeral"},
        }
    ]

    # Test that cache_control is not removed
    transformed_messages, _ = config.remove_cache_control_flag_from_messages_and_tools(
        model="openrouter/minimax/minimax-m2", messages=messages
    )

    # The method should preserve cache_control for minimax models
    assert transformed_messages[0].get("cache_control") == {"type": "ephemeral"}


def test_openrouter_glm_supports_cache_control():
    """OpenRouter should preserve cache_control for GLM models."""
    config = OpenrouterConfig()

    messages = [
        {
            "role": "user",
            "content": "Hello, world!",
            "cache_control": {"type": "ephemeral"},
        }
    ]

    # Test that cache_control is not removed for GLM models
    transformed_messages, _ = config.remove_cache_control_flag_from_messages_and_tools(
        model="openrouter/z-ai/glm-4.6", messages=messages
    )

    # The method should preserve cache_control for GLM models
    assert transformed_messages[0].get("cache_control") == {"type": "ephemeral"}


def test_openrouter_deepseek_strips_cache_control():
    """OpenRouter should still strip cache_control for non-supported models."""
    config = OpenrouterConfig()

    messages = [
        {
            "role": "user",
            "content": "Hello, world!",
            "cache_control": {"type": "ephemeral"},
        }
    ]

    # DeepSeek doesn't support cache_control, so it should be stripped
    transformed_messages, _ = config.remove_cache_control_flag_from_messages_and_tools(
        model="openrouter/deepseek/deepseek-chat", messages=messages
    )

    # cache_control should be removed for non-supported models
    assert transformed_messages[0].get("cache_control") is None


def test_openrouter_minimax_transform_moves_cache_control_to_content():
    """OpenRouter should move cache_control to content blocks for MiniMax."""
    config = OpenrouterConfig()

    messages = [
        {
            "role": "user",
            "content": "Analyze this data",
            "cache_control": {"type": "ephemeral"},
        }
    ]

    transformed_request = config.transform_request(
        model="openrouter/minimax/minimax-m2",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    # cache_control should be moved to content blocks
    assert "messages" in transformed_request
    user_message = transformed_request["messages"][0]
    assert isinstance(user_message["content"], list)
    assert user_message["content"][0]["cache_control"] == {"type": "ephemeral"}
    # Message-level cache_control should be removed
    assert "cache_control" not in user_message


def test_openrouter_glm_transform_moves_cache_control_to_content():
    """OpenRouter should move cache_control to content blocks for GLM."""
    config = OpenrouterConfig()

    messages = [
        {
            "role": "user",
            "content": "Analyze this data",
            "cache_control": {"type": "ephemeral"},
        }
    ]

    transformed_request = config.transform_request(
        model="openrouter/z-ai/glm-4.6",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    # cache_control should be moved to content blocks
    assert "messages" in transformed_request
    user_message = transformed_request["messages"][0]
    assert isinstance(user_message["content"], list)
    assert user_message["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_openrouter_supports_thinking_param_for_reasoning_models():
    """OpenRouter should support thinking parameter for reasoning-capable models."""
    config = OpenrouterConfig()

    # Test MiniMax (supports reasoning)
    supported_params_minimax = config.get_supported_openai_params(
        model="openrouter/minimax/minimax-m2"
    )
    assert "thinking" in supported_params_minimax
    assert "reasoning_effort" in supported_params_minimax

    # Test GLM (supports reasoning)
    supported_params_glm = config.get_supported_openai_params(
        model="openrouter/z-ai/glm-4.6"
    )
    assert "thinking" in supported_params_glm
    assert "reasoning_effort" in supported_params_glm
