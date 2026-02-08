from unittest.mock import patch

import pytest

from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.experimental_pass_through.transformation import (
    VertexAIPartnerModelsAnthropicMessagesConfig,
)


def test_validate_environment_uses_vertex_ai_location():
    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    headers = {}
    litellm_params = {
        "vertex_ai_project": "test-project",
        "vertex_ai_location": "europe-west1",
        "vertex_credentials": "{}",
    }
    optional_params = {}

    with patch.object(
        config, "_ensure_access_token", return_value=("token", "test-project")
    ), patch.object(
        config, "get_complete_vertex_url", return_value="https://mock-url"
    ) as mock_get_url:
        config.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-3-sonnet",
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_base=None,
        )
        assert mock_get_url.call_args.kwargs["vertex_location"] == "europe-west1"


def test_web_search_header_added_for_messages_endpoint():
    """Test that web search tool adds the required beta header for Vertex AI /v1/messages requests"""
    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    headers = {}
    litellm_params = {
        "vertex_ai_project": "test-project",
        "vertex_ai_location": "us-central1",
        "vertex_credentials": "{}",
    }
    # Include web search tool in optional_params
    optional_params = {
        "tools": [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
        ]
    }

    with patch.object(
        config, "_ensure_access_token", return_value=("token", "test-project")
    ), patch.object(
        config, "get_complete_vertex_url", return_value="https://mock-url"
    ):
        updated_headers, api_base = config.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-sonnet-4",
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_base=None,
        )
        
        # Assert that the anthropic-beta header with web-search is present
        assert "anthropic-beta" in updated_headers, "anthropic-beta header should be present"
        assert updated_headers["anthropic-beta"] == "web-search-2025-03-05", \
            f"anthropic-beta should be 'web-search-2025-03-05', got: {updated_headers['anthropic-beta']}"


def test_web_search_header_not_added_without_tool():
    """Test that beta header is NOT added when web search tool is not present"""
    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    headers = {}
    litellm_params = {
        "vertex_ai_project": "test-project",
        "vertex_ai_location": "us-central1",
        "vertex_credentials": "{}",
    }
    # No web search tool
    optional_params = {}

    with patch.object(
        config, "_ensure_access_token", return_value=("token", "test-project")
    ), patch.object(
        config, "get_complete_vertex_url", return_value="https://mock-url"
    ):
        updated_headers, api_base = config.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-sonnet-4",
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_base=None,
        )

        # Assert that the anthropic-beta header is NOT present when no web search tool
        assert "anthropic-beta" not in updated_headers, \
            "anthropic-beta header should not be present without web search tool"


class TestVertexAICacheControlInjection:
    """
    Tests for automatic cache_control injection in Vertex AI Anthropic Messages API.
    Related issue: https://github.com/BerriAI/litellm/issues/20418
    """

    def test_cache_control_injected_to_last_message_list_content(self):
        """Test that cache_control is injected into the last content block of the last message."""
        config = VertexAIPartnerModelsAnthropicMessagesConfig()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "System prompt text"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I understand."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First block"},
                    {"type": "text", "text": "Second block - should get cache_control"},
                ],
            },
        ]

        result = config.transform_anthropic_messages_request(
            model="claude-sonnet-4",
            messages=messages,
            anthropic_messages_optional_request_params={"max_tokens": 1024},
            litellm_params={},
            headers={},
        )

        result_messages = result["messages"]
        # Last message, last content block should have cache_control
        last_message = result_messages[-1]
        last_block = last_message["content"][-1]
        assert "cache_control" in last_block, "cache_control should be injected into last content block"
        assert last_block["cache_control"] == {"type": "ephemeral"}

        # First content block of last message should NOT have cache_control
        first_block = last_message["content"][0]
        assert "cache_control" not in first_block, "cache_control should NOT be on first content block"

        # Earlier messages should NOT have cache_control
        for msg in result_messages[:-1]:
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        assert "cache_control" not in block, \
                            f"cache_control should NOT be on earlier messages: {block}"

    def test_cache_control_injected_to_string_content(self):
        """Test that string content is converted to list format with cache_control."""
        config = VertexAIPartnerModelsAnthropicMessagesConfig()
        messages = [
            {
                "role": "user",
                "content": "Hello, please analyze this document.",
            },
        ]

        result = config.transform_anthropic_messages_request(
            model="claude-sonnet-4",
            messages=messages,
            anthropic_messages_optional_request_params={"max_tokens": 1024},
            litellm_params={},
            headers={},
        )

        result_messages = result["messages"]
        last_message = result_messages[-1]
        # String content should be converted to list format
        assert isinstance(last_message["content"], list), "String content should be converted to list"
        assert len(last_message["content"]) == 1
        block = last_message["content"][0]
        assert block["type"] == "text"
        assert block["text"] == "Hello, please analyze this document."
        assert block["cache_control"] == {"type": "ephemeral"}

    def test_cache_control_not_double_injected(self):
        """Test that cache_control is NOT injected when already present in messages."""
        config = VertexAIPartnerModelsAnthropicMessagesConfig()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First block"},
                    {
                        "type": "text",
                        "text": "Second block",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Third block - no cache_control here"},
                ],
            },
        ]

        result = config.transform_anthropic_messages_request(
            model="claude-sonnet-4",
            messages=messages,
            anthropic_messages_optional_request_params={"max_tokens": 1024},
            litellm_params={},
            headers={},
        )

        result_messages = result["messages"]
        # The last message's last block should NOT get cache_control
        # because cache_control already exists in the first message
        last_message = result_messages[-1]
        last_block = last_message["content"][-1]
        assert "cache_control" not in last_block, \
            "cache_control should NOT be injected when already present in messages"

    def test_cache_control_with_claude_code_style_messages(self):
        """
        Test cache_control injection with typical Claude Code message structure.
        Claude Code sends many text blocks across messages.
        """
        config = VertexAIPartnerModelsAnthropicMessagesConfig()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<system-reminder>SessionStart</system-reminder>"},
                    {"type": "text", "text": "long code block 1"},
                    {"type": "text", "text": "long code block 2"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll help with that."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "tool result 1"},
                    {"type": "text", "text": "tool result 2"},
                    {"type": "text", "text": "Please continue analyzing."},
                ],
            },
        ]

        result = config.transform_anthropic_messages_request(
            model="claude-haiku-4-5-20251001",
            messages=messages,
            anthropic_messages_optional_request_params={"max_tokens": 8192},
            litellm_params={},
            headers={},
        )

        result_messages = result["messages"]

        # Only the very last content block of the very last message should have cache_control
        cache_control_count = 0
        for msg in result_messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        cache_control_count += 1

        assert cache_control_count == 1, \
            f"Expected exactly 1 cache_control injection, found {cache_control_count}"

        # Verify it's on the last block of the last message
        last_block = result_messages[-1]["content"][-1]
        assert last_block["cache_control"] == {"type": "ephemeral"}

    def test_cache_control_injected_to_non_dict_list_content(self):
        """Test that non-dict last content blocks (strings in a list) are normalized."""
        config = VertexAIPartnerModelsAnthropicMessagesConfig()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First block"},
                    "Plain string as last block",
                ],
            },
        ]

        config._inject_cache_control_to_last_message(messages)

        last_block = messages[-1]["content"][-1]
        # Should be normalized to a dict with cache_control
        assert isinstance(last_block, dict), "Non-dict block should be normalized to dict"
        assert last_block["type"] == "text"
        assert last_block["text"] == "Plain string as last block"
        assert last_block["cache_control"] == {"type": "ephemeral"}

    def test_cache_control_with_empty_messages(self):
        """Test that empty messages list doesn't cause errors."""
        config = VertexAIPartnerModelsAnthropicMessagesConfig()

        # _inject_cache_control_to_last_message should handle empty list gracefully
        messages = []
        config._inject_cache_control_to_last_message(messages)
        assert messages == []

    def test_has_cache_control_detection(self):
        """Test the _has_cache_control_in_messages helper."""
        config = VertexAIPartnerModelsAnthropicMessagesConfig()

        # No cache_control
        messages_no_cc = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        ]
        assert config._has_cache_control_in_messages(messages_no_cc) is False

        # cache_control in content block
        messages_with_cc = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}},
                ],
            },
        ]
        assert config._has_cache_control_in_messages(messages_with_cc) is True

        # cache_control at message level (string content)
        messages_msg_level_cc = [
            {
                "role": "user",
                "content": "hello",
                "cache_control": {"type": "ephemeral"},
            },
        ]
        assert config._has_cache_control_in_messages(messages_msg_level_cc) is True
