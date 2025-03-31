import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.anthropic.chat.transformation import AnthropicConfig


def test_anthropic_prompt_caching_headers_for_vertex():
    """
    Test that the prompt caching beta header is correctly set for Vertex AI requests
    with Anthropic models when cache control is present in the messages.
    """
    # Create an instance of AnthropicConfig
    config = AnthropicConfig()

    # Test case 1: Vertex request with prompt caching
    # Create a message with cache control
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
            "cache_control": {"type": "ephemeral"}
        },
        {
            "role": "user",
            "content": "Tell me about the solar system."
        }
    ]

    # Check if cache control is detected
    is_cache_control_set = config.is_cache_control_set(messages=messages)
    assert is_cache_control_set is True, "Cache control should be detected in messages"

    # Generate headers for a Vertex AI request with prompt caching
    headers = config.get_anthropic_headers(
        api_key="test-api-key",
        prompt_caching_set=is_cache_control_set,
        is_vertex_request=True
    )

    # Verify that the anthropic-beta header is set with prompt-caching-2024-07-31
    assert "anthropic-beta" in headers, "anthropic-beta header should be present"
    assert "prompt-caching-2024-07-31" in headers["anthropic-beta"], "prompt-caching-2024-07-31 should be in the beta header"

    # Test case 2: Vertex request without prompt caching
    messages_without_cache = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Tell me about the solar system."
        }
    ]

    # Check if cache control is detected
    is_cache_control_set = config.is_cache_control_set(messages=messages_without_cache)
    assert is_cache_control_set is False, "Cache control should not be detected in messages"

    # Generate headers for a Vertex AI request without prompt caching
    headers = config.get_anthropic_headers(
        api_key="test-api-key",
        prompt_caching_set=is_cache_control_set,
        is_vertex_request=True
    )

    # Verify that the anthropic-beta header is not set
    assert "anthropic-beta" not in headers, "anthropic-beta header should not be present"


def test_anthropic_prompt_caching_with_content_blocks():
    """
    Test that prompt caching is correctly detected when cache control is in content blocks.
    """
    config = AnthropicConfig()

    # Message with cache control in content blocks
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant.",
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        },
        {
            "role": "user",
            "content": "Tell me about the solar system."
        }
    ]

    # Check if cache control is detected
    is_cache_control_set = config.is_cache_control_set(messages=messages)
    assert is_cache_control_set is True, "Cache control should be detected in content blocks"

    # Generate headers for a Vertex AI request with prompt caching
    headers = config.get_anthropic_headers(
        api_key="test-api-key",
        prompt_caching_set=is_cache_control_set,
        is_vertex_request=True
    )

    # Verify that the anthropic-beta header is set with prompt-caching-2024-07-31
    assert "anthropic-beta" in headers, "anthropic-beta header should be present"
    assert "prompt-caching-2024-07-31" in headers["anthropic-beta"], "prompt-caching-2024-07-31 should be in the beta header"


def test_anthropic_vertex_other_beta_headers():
    """
    Test that other beta headers are not included for Vertex AI requests.
    """
    config = AnthropicConfig()

    # Generate headers with multiple beta features
    headers = config.get_anthropic_headers(
        api_key="test-api-key",
        prompt_caching_set=True,
        computer_tool_used=True,  # This should be excluded for Vertex
        pdf_used=True,  # This should be excluded for Vertex
        is_vertex_request=True
    )

    # Verify that only prompt-caching is included in the beta header
    assert "anthropic-beta" in headers, "anthropic-beta header should be present"
    assert headers["anthropic-beta"] == "prompt-caching-2024-07-31", "Only prompt-caching should be in the beta header"
    assert "computer-use-2024-10-22" not in headers["anthropic-beta"], "computer-use beta should not be included"
    assert "pdfs-2024-09-25" not in headers["anthropic-beta"], "pdfs beta should not be included"
