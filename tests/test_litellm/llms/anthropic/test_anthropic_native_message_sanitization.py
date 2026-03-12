"""
Tests for sanitize_anthropic_native_messages_for_tool_calling().

Covers the /v1/messages pass-through path where messages arrive in
Anthropic-native format (content is a list with tool_use/tool_result blocks)
rather than OpenAI format (tool_calls field).

Bug scenario:
  An Anthropic API client (e.g. an agentic coding tool) sends /v1/messages
  requests through LiteLLM proxy.  After context compaction, two separate
  assistant turns may be merged into a single message whose content list
  looks like:

      [text_A, tool_use_A, text_B, tool_use_B]

  Anthropic rejects this with:
      "messages.N: `tool_use` ids were found without `tool_result` blocks
       immediately after: <id>"

  because text blocks must not appear *after* tool_use blocks in the same
  assistant message.

  The fix reorders content so all text blocks precede all tool_use blocks.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import litellm
from litellm.litellm_core_utils.prompt_templates.factory import (
    sanitize_anthropic_native_messages_for_tool_calling,
)


class TestSanitizeAnthropicNativeMessages:
    def setup_method(self):
        self._orig = litellm.modify_params
        litellm.modify_params = True

    def teardown_method(self):
        litellm.modify_params = self._orig

    # ------------------------------------------------------------------
    # Case C – text block appears after tool_use block (the primary bug)
    # ------------------------------------------------------------------

    def test_text_after_tool_use_reordered(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "do work"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Thinking from first turn."},
                    {"type": "tool_use", "id": "toolu_aaa", "name": "tool_a", "input": {}},
                    {"type": "text", "text": "Thinking from second turn."},
                    {"type": "tool_use", "id": "toolu_bbb", "name": "tool_b", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_aaa", "content": "result_a"},
                    {"type": "tool_result", "tool_use_id": "toolu_bbb", "content": "result_b"},
                ],
            },
        ]

        result = sanitize_anthropic_native_messages_for_tool_calling(messages)

        assert len(result) == 3
        content = result[1]["content"]
        types = [b["type"] for b in content]
        assert types == ["text", "text", "tool_use", "tool_use"], (
            f"Expected all text blocks before tool_use blocks, got: {types}"
        )
        assert content[2]["id"] == "toolu_aaa"
        assert content[3]["id"] == "toolu_bbb"

    def test_valid_text_before_tool_use_unchanged(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will call two tools."},
                    {"type": "tool_use", "id": "toolu_aaa", "name": "tool_a", "input": {}},
                    {"type": "tool_use", "id": "toolu_bbb", "name": "tool_b", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_aaa", "content": "r1"},
                    {"type": "tool_result", "tool_use_id": "toolu_bbb", "content": "r2"},
                ],
            },
        ]

        result = sanitize_anthropic_native_messages_for_tool_calling(messages)

        assert result == messages

    def test_no_text_blocks_unchanged(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_aaa", "name": "tool_a", "input": {}},
                    {"type": "tool_use", "id": "toolu_bbb", "name": "tool_b", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_aaa", "content": "r1"},
                    {"type": "tool_result", "tool_use_id": "toolu_bbb", "content": "r2"},
                ],
            },
        ]

        result = sanitize_anthropic_native_messages_for_tool_calling(messages)

        assert result == messages

    # ------------------------------------------------------------------
    # Case A – orphaned tool_use (no matching tool_result)
    # ------------------------------------------------------------------

    def test_orphaned_tool_use_injects_dummy_user_message(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "do something"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Sure."},
                    {"type": "tool_use", "id": "toolu_orphan", "name": "my_tool", "input": {}},
                ],
            },
        ]

        result = sanitize_anthropic_native_messages_for_tool_calling(messages)

        assert len(result) == 3
        injected = result[2]
        assert injected["role"] == "user"
        assert isinstance(injected["content"], list)
        tool_results = [b for b in injected["content"] if b.get("type") == "tool_result"]
        assert len(tool_results) == 1
        assert tool_results[0]["tool_use_id"] == "toolu_orphan"
        assert "my_tool" in tool_results[0]["content"]

    def test_orphaned_tool_use_prepends_to_existing_user_message(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "start"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_x", "name": "tool_x", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "never mind"}],
            },
        ]

        result = sanitize_anthropic_native_messages_for_tool_calling(messages)

        assert len(result) == 3
        content = result[2]["content"]
        assert isinstance(content, list)
        tool_results = [b for b in content if b.get("type") == "tool_result"]
        text_blocks = [b for b in content if b.get("type") == "text"]
        assert len(tool_results) == 1
        assert tool_results[0]["tool_use_id"] == "toolu_x"
        assert len(text_blocks) == 1

    def test_orphaned_tool_use_string_user_message_wrapped(self):
        messages = [
            {"role": "user", "content": "start"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_y", "name": "tool_y", "input": {}}
                ],
            },
            {"role": "user", "content": "follow-up as plain string"},
        ]

        result = sanitize_anthropic_native_messages_for_tool_calling(messages)

        assert len(result) == 3
        content = result[2]["content"]
        assert isinstance(content, list)
        types = {b.get("type") for b in content}
        assert "tool_result" in types
        assert "text" in types

    def test_partial_tool_results_get_dummies_for_missing_ids(self):
        messages = [
            {"role": "user", "content": "run two tools"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tid_1", "name": "tool_a", "input": {}},
                    {"type": "tool_use", "id": "tid_2", "name": "tool_b", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tid_1", "content": "result_a"}
                ],
            },
        ]

        result = sanitize_anthropic_native_messages_for_tool_calling(messages)

        assert len(result) == 3
        user_content = result[2]["content"]
        result_ids = {b["tool_use_id"] for b in user_content if b.get("type") == "tool_result"}
        assert "tid_1" in result_ids
        assert "tid_2" in result_ids

    def test_complete_tool_use_result_pair_passes_through_unchanged(self):
        messages = [
            {"role": "user", "content": "ping"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_z", "name": "ping_tool", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_z", "content": "pong"}
                ],
            },
        ]

        result = sanitize_anthropic_native_messages_for_tool_calling(messages)

        assert result == messages

    # ------------------------------------------------------------------
    # modify_params gate
    # ------------------------------------------------------------------

    def test_modify_params_false_skips_all_sanitization(self):
        litellm.modify_params = False

        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "A"},
                    {"type": "tool_use", "id": "toolu_a", "name": "t", "input": {}},
                    {"type": "text", "text": "B"},
                ],
            },
        ]

        result = sanitize_anthropic_native_messages_for_tool_calling(messages)

        assert result == messages

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_messages_returns_empty(self):
        assert sanitize_anthropic_native_messages_for_tool_calling([]) == []

    def test_no_tool_use_messages_unchanged(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ]

        result = sanitize_anthropic_native_messages_for_tool_calling(messages)

        assert result == messages

    def test_string_content_messages_unchanged(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

        result = sanitize_anthropic_native_messages_for_tool_calling(messages)

        assert result == messages
