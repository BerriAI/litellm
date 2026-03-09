"""
Tests for thinking block ordering preservation on round-trip with
multiple web searches.

Fixes: https://github.com/BerriAI/litellm/issues/23047

When an assistant response contains interleaved thinking blocks and
server_tool_use / web_search_tool_result blocks, the round-trip
conversion (Anthropic → OpenAI → Anthropic) must preserve the original
content block order.  Anthropic's API verifies thinking block signatures
and rejects requests where the blocks have been reordered.
"""

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.prompt_templates.factory import anthropic_messages_pt
from litellm.llms.anthropic.chat.transformation import AnthropicConfig


# Simulated Anthropic response with interleaved thinking and 2 web searches.
# This is the content array from the raw Anthropic API response.
INTERLEAVED_CONTENT: List[Dict[str, Any]] = [
    {
        "type": "thinking",
        "thinking": "I should search for fast.ai news first.",
        "signature": "sig_thinking_1",
    },
    {
        "type": "server_tool_use",
        "id": "srvtoolu_001",
        "name": "web_search",
        "input": {"query": "fast.ai news"},
    },
    {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_001",
        "content": [
            {"url": "https://fast.ai", "title": "fast.ai", "snippet": "Fast AI news"}
        ],
    },
    {
        "type": "thinking",
        "thinking": "Now let me search for answer.ai as well.",
        "signature": "sig_thinking_2",
    },
    {
        "type": "text",
        "text": "Here are the results from my searches.",
    },
    {
        "type": "server_tool_use",
        "id": "srvtoolu_002",
        "name": "web_search",
        "input": {"query": "answer.ai news"},
    },
    {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_002",
        "content": [
            {
                "url": "https://answer.ai",
                "title": "Answer.AI",
                "snippet": "Answer AI news",
            }
        ],
    },
]


def _make_anthropic_response(content: List[Dict]) -> dict:
    """Build a minimal Anthropic Messages API response dict."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 200,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def _parse_response_to_openai(content: List[Dict]) -> litellm.ModelResponse:
    """Run the Anthropic → OpenAI response transformation."""
    config = AnthropicConfig()
    completion_response = _make_anthropic_response(content)
    model_response = litellm.ModelResponse()

    raw_response = MagicMock(spec=httpx.Response)
    raw_response.headers = {}
    raw_response.status_code = 200

    return config.transform_parsed_response(
        completion_response=completion_response,
        raw_response=raw_response,
        model_response=model_response,
    )


class TestThinkingBlocksRoundTripOrdering:
    """Round-trip: Anthropic response → OpenAI message → Anthropic request."""

    def test_original_content_stored_when_interleaved(self):
        """
        transform_parsed_response should store _original_content in
        provider_specific_fields when both thinking blocks and server_tool_use
        are present.
        """
        resp = _parse_response_to_openai(INTERLEAVED_CONTENT)
        msg = resp.choices[0].message
        psf = msg.provider_specific_fields or {}

        assert "_original_content" in psf, (
            "_original_content must be stored for interleaved thinking + server tools"
        )
        assert psf["_original_content"] == INTERLEAVED_CONTENT

    def test_original_content_not_stored_without_server_tools(self):
        """No _original_content when there are no server_tool_use blocks."""
        content = [
            {"type": "thinking", "thinking": "hmm", "signature": "sig1"},
            {"type": "text", "text": "Hello!"},
        ]
        resp = _parse_response_to_openai(content)
        msg = resp.choices[0].message
        psf = msg.provider_specific_fields or {}

        assert "_original_content" not in psf

    def test_original_content_not_stored_without_thinking(self):
        """No _original_content when there are no thinking blocks."""
        content = [
            {
                "type": "server_tool_use",
                "id": "srvtoolu_001",
                "name": "web_search",
                "input": {"query": "test"},
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_001",
                "content": [],
            },
            {"type": "text", "text": "Result"},
        ]
        resp = _parse_response_to_openai(content)
        msg = resp.choices[0].message
        psf = msg.provider_specific_fields or {}

        assert "_original_content" not in psf

    def test_round_trip_preserves_block_order(self):
        """
        Core test: the Anthropic → OpenAI → Anthropic round-trip must
        produce the same content block type sequence as the original.
        """
        resp = _parse_response_to_openai(INTERLEAVED_CONTENT)
        msg = resp.choices[0].message

        # Build the messages list as a user would for a follow-up call
        openai_messages = [
            {"role": "user", "content": "Search for fast.ai and answer.ai news"},
            msg.model_dump(),
            {"role": "user", "content": "Now search for solveit"},
        ]

        # Run the OpenAI → Anthropic conversion
        anthropic_messages = anthropic_messages_pt(
            openai_messages,
            model="claude-sonnet-4-6",
            llm_provider="anthropic",
        )

        # Find the assistant message
        assistant_msg = next(
            m for m in anthropic_messages if m["role"] == "assistant"
        )
        reconstructed = assistant_msg["content"]

        # Extract type sequences
        original_types = [b["type"] for b in INTERLEAVED_CONTENT]
        reconstructed_types = [
            b.get("type") or ("thinking" if "thinking" in b else "unknown")
            for b in reconstructed
        ]

        assert reconstructed_types == original_types, (
            f"Block ordering must be preserved on round-trip.\n"
            f"  Original:      {original_types}\n"
            f"  Reconstructed: {reconstructed_types}"
        )

    def test_round_trip_preserves_thinking_block_content(self):
        """Thinking block text and signatures must survive round-trip."""
        resp = _parse_response_to_openai(INTERLEAVED_CONTENT)
        msg = resp.choices[0].message
        openai_messages = [
            {"role": "user", "content": "test"},
            msg.model_dump(),
            {"role": "user", "content": "follow up"},
        ]

        anthropic_messages = anthropic_messages_pt(
            openai_messages,
            model="claude-sonnet-4-6",
            llm_provider="anthropic",
        )

        assistant_msg = next(
            m for m in anthropic_messages if m["role"] == "assistant"
        )
        thinking_blocks = [
            b for b in assistant_msg["content"] if b.get("type") == "thinking"
        ]

        assert len(thinking_blocks) == 2
        assert thinking_blocks[0]["thinking"] == "I should search for fast.ai news first."
        assert thinking_blocks[0]["signature"] == "sig_thinking_1"
        assert thinking_blocks[1]["thinking"] == "Now let me search for answer.ai as well."
        assert thinking_blocks[1]["signature"] == "sig_thinking_2"

    def test_round_trip_preserves_server_tool_use_blocks(self):
        """server_tool_use and web_search_tool_result must survive round-trip."""
        resp = _parse_response_to_openai(INTERLEAVED_CONTENT)
        msg = resp.choices[0].message
        openai_messages = [
            {"role": "user", "content": "test"},
            msg.model_dump(),
            {"role": "user", "content": "follow up"},
        ]

        anthropic_messages = anthropic_messages_pt(
            openai_messages,
            model="claude-sonnet-4-6",
            llm_provider="anthropic",
        )

        assistant_msg = next(
            m for m in anthropic_messages if m["role"] == "assistant"
        )
        content = assistant_msg["content"]

        # Verify server_tool_use blocks
        server_tools = [b for b in content if b.get("type") == "server_tool_use"]
        assert len(server_tools) == 2
        assert server_tools[0]["id"] == "srvtoolu_001"
        assert server_tools[1]["id"] == "srvtoolu_002"

        # Verify web_search_tool_result blocks
        ws_results = [
            b for b in content if b.get("type") == "web_search_tool_result"
        ]
        assert len(ws_results) == 2
        assert ws_results[0]["tool_use_id"] == "srvtoolu_001"
        assert ws_results[1]["tool_use_id"] == "srvtoolu_002"

    def test_round_trip_preserves_text_block(self):
        """Text blocks must survive round-trip."""
        resp = _parse_response_to_openai(INTERLEAVED_CONTENT)
        msg = resp.choices[0].message
        openai_messages = [
            {"role": "user", "content": "test"},
            msg.model_dump(),
            {"role": "user", "content": "follow up"},
        ]

        anthropic_messages = anthropic_messages_pt(
            openai_messages,
            model="claude-sonnet-4-6",
            llm_provider="anthropic",
        )

        assistant_msg = next(
            m for m in anthropic_messages if m["role"] == "assistant"
        )
        text_blocks = [
            b for b in assistant_msg["content"] if b.get("type") == "text"
        ]

        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "Here are the results from my searches."

    def test_fallback_to_normal_path_without_original_content(self):
        """
        Messages without _original_content (e.g. single web search or no web
        search) should use the existing reconstruction logic.
        """
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "Hi there!",
                "thinking_blocks": [
                    {
                        "type": "thinking",
                        "thinking": "greeting",
                        "signature": "sig1",
                    }
                ],
            },
            {"role": "user", "content": "Follow up"},
        ]

        result = anthropic_messages_pt(
            messages, model="claude-sonnet-4-6", llm_provider="anthropic"
        )

        assistant_msg = next(m for m in result if m["role"] == "assistant")
        types = [b.get("type") for b in assistant_msg["content"]]

        # Normal path: thinking first, then text
        assert types == ["thinking", "text"]

    def test_single_web_search_with_thinking_preserves_order(self):
        """Even a single web search with thinking should preserve order."""
        single_search_content = [
            {
                "type": "thinking",
                "thinking": "Let me search.",
                "signature": "sig1",
            },
            {
                "type": "server_tool_use",
                "id": "srvtoolu_001",
                "name": "web_search",
                "input": {"query": "test"},
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_001",
                "content": [{"url": "https://example.com", "title": "Test", "snippet": "ok"}],
            },
            {"type": "text", "text": "Here you go."},
        ]

        resp = _parse_response_to_openai(single_search_content)
        msg = resp.choices[0].message
        openai_messages = [
            {"role": "user", "content": "test"},
            msg.model_dump(),
            {"role": "user", "content": "follow up"},
        ]

        anthropic_messages = anthropic_messages_pt(
            openai_messages,
            model="claude-sonnet-4-6",
            llm_provider="anthropic",
        )

        assistant_msg = next(
            m for m in anthropic_messages if m["role"] == "assistant"
        )
        original_types = [b["type"] for b in single_search_content]
        reconstructed_types = [b.get("type") for b in assistant_msg["content"]]

        assert reconstructed_types == original_types
