"""
Regression test for https://github.com/BerriAI/litellm/issues/23047

When Claude performs multiple web searches with extended thinking, the
response content array interleaves thinking blocks with server_tool_use /
web_search_tool_result blocks.  On round-trip, anthropic_messages_pt()
must preserve this interleaving, otherwise Anthropic rejects the request
with "thinking blocks cannot be modified".
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.abspath("."))

from litellm.litellm_core_utils.prompt_templates.factory import anthropic_messages_pt


def _make_interleaved_assistant_message():
    """
    Simulate an OpenAI-format assistant message that originated from an
    Anthropic response with interleaved thinking + server_tool_use +
    web_search_tool_result blocks.

    The original Anthropic content order was:
      [thinking_1, server_tool_use_1, web_search_result_1,
       thinking_2, text_1, server_tool_use_2, web_search_result_2]
    """
    original_content = [
        {
            "type": "thinking",
            "thinking": "Let me search for fast.ai news first.",
            "signature": "sig_1_abc",
        },
        {
            "type": "server_tool_use",
            "id": "srvtoolu_001",
            "name": "web_search",
            "input": {"query": "fast.ai latest news"},
        },
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_001",
            "content": [{"type": "web_search_result", "url": "https://fast.ai", "title": "fast.ai"}],
        },
        {
            "type": "thinking",
            "thinking": "Now let me search for answer.ai news.",
            "signature": "sig_2_def",
        },
        {
            "type": "text",
            "text": "Here are the results so far.",
        },
        {
            "type": "server_tool_use",
            "id": "srvtoolu_002",
            "name": "web_search",
            "input": {"query": "answer.ai latest news"},
        },
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_002",
            "content": [{"type": "web_search_result", "url": "https://answer.ai", "title": "answer.ai"}],
        },
    ]

    return {
        "role": "assistant",
        "content": "Here are the results so far.",
        "thinking_blocks": [
            {
                "type": "thinking",
                "thinking": "Let me search for fast.ai news first.",
                "signature": "sig_1_abc",
            },
            {
                "type": "thinking",
                "thinking": "Now let me search for answer.ai news.",
                "signature": "sig_2_def",
            },
        ],
        "tool_calls": [
            {
                "id": "srvtoolu_001",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": '{"query": "fast.ai latest news"}',
                },
            },
            {
                "id": "srvtoolu_002",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": '{"query": "answer.ai latest news"}',
                },
            },
        ],
        "provider_specific_fields": {
            "citations": None,
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "Let me search for fast.ai news first.",
                    "signature": "sig_1_abc",
                },
                {
                    "type": "thinking",
                    "thinking": "Now let me search for answer.ai news.",
                    "signature": "sig_2_def",
                },
            ],
            "web_search_results": [
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_001",
                    "content": [{"type": "web_search_result", "url": "https://fast.ai", "title": "fast.ai"}],
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_002",
                    "content": [{"type": "web_search_result", "url": "https://answer.ai", "title": "answer.ai"}],
                },
            ],
            "anthropic_content": original_content,
        },
    }


def test_preserved_content_ordering_with_anthropic_content():
    """
    When provider_specific_fields contains 'anthropic_content' (preserved
    from the original Anthropic response), anthropic_messages_pt() must
    use it directly to maintain the exact interleaving order.
    """
    messages = [
        {"role": "user", "content": "Search the web for news about fast.ai and answer.ai"},
        _make_interleaved_assistant_message(),
        {"role": "user", "content": "Now search for news about solveit"},
    ]

    result = anthropic_messages_pt(messages, model="claude-sonnet-4-6", llm_provider="anthropic")

    # Find the assistant message
    assistant_msgs = [m for m in result if m["role"] == "assistant"]
    assert len(assistant_msgs) == 1

    content = assistant_msgs[0]["content"]

    # Extract the type sequence
    type_sequence = [block["type"] for block in content]

    # The order must match the original Anthropic response:
    # thinking, server_tool_use, web_search_tool_result,
    # thinking, text, server_tool_use, web_search_tool_result
    expected = [
        "thinking",
        "server_tool_use",
        "web_search_tool_result",
        "thinking",
        "text",
        "server_tool_use",
        "web_search_tool_result",
    ]
    assert type_sequence == expected, (
        f"Content block ordering is wrong.\n"
        f"Expected: {expected}\n"
        f"Got:      {type_sequence}"
    )


def test_thinking_blocks_not_duplicated():
    """
    When anthropic_content is used, thinking blocks must appear exactly
    as they do in the original content — not duplicated from the top-level
    thinking_blocks field.
    """
    messages = [
        {"role": "user", "content": "Search for news"},
        _make_interleaved_assistant_message(),
        {"role": "user", "content": "Follow up"},
    ]

    result = anthropic_messages_pt(messages, model="claude-sonnet-4-6", llm_provider="anthropic")
    assistant_msgs = [m for m in result if m["role"] == "assistant"]
    content = assistant_msgs[0]["content"]

    thinking_blocks = [b for b in content if b.get("type") == "thinking"]
    # Should have exactly 2 thinking blocks, not 4 (which would happen
    # if both top-level thinking_blocks AND anthropic_content thinking blocks
    # were included)
    assert len(thinking_blocks) == 2


def test_fallback_without_anthropic_content():
    """
    When provider_specific_fields does NOT contain 'anthropic_content'
    (e.g. message was manually constructed), the existing reconstruction
    logic must still work.
    """
    messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "I will help you.",
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "The user wants help.",
                    "signature": "sig_abc",
                },
            ],
            "tool_calls": None,
            "provider_specific_fields": {
                "citations": None,
                "thinking_blocks": [
                    {
                        "type": "thinking",
                        "thinking": "The user wants help.",
                        "signature": "sig_abc",
                    },
                ],
            },
        },
        {"role": "user", "content": "Thanks"},
    ]

    result = anthropic_messages_pt(messages, model="claude-sonnet-4-6", llm_provider="anthropic")
    assistant_msgs = [m for m in result if m["role"] == "assistant"]
    assert len(assistant_msgs) == 1

    content = assistant_msgs[0]["content"]
    type_sequence = [block["type"] for block in content]

    # Without anthropic_content, existing logic prepends thinking blocks,
    # then adds text
    assert "thinking" in type_sequence
    assert "text" in type_sequence


def test_text_only_response_no_anthropic_content():
    """Text-only responses should not store anthropic_content (memory optimisation)."""
    import litellm
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    config = AnthropicConfig()

    completion_response = {
        "id": "msg_text_only",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello, how can I help?"}],
        "model": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }

    class FakeResponse:
        def json(self):
            return completion_response
        headers = {}
        status_code = 200
        text = "{}"

    class FakeLogger:
        model_call_details = {}
        def post_call(self, **kwargs):
            pass

    result = config.transform_response(
        model="claude-sonnet-4-6",
        raw_response=FakeResponse(),
        model_response=litellm.ModelResponse(),
        logging_obj=FakeLogger(),
        request_data={},
        messages=[{"role": "user", "content": "Hi"}],
        optional_params={},
        litellm_params={},
        encoding=None,
        api_key="test-key",
        json_mode=False,
    )

    provider = result.choices[0].message.provider_specific_fields or {}
    assert "anthropic_content" not in provider, (
        "text-only responses should not carry anthropic_content"
    )
