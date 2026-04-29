"""
Tests for splitting assistant messages that mix signed and unsigned parts.

Fixes: https://github.com/BerriAI/litellm/issues/17949

When Gemini thinking mode is enabled and the assistant message contains both:
- text parts (no thoughtSignature)
- functionCall parts (with thoughtSignature)

Gemini rejects the request because signed and unsigned parts cannot coexist
in the same message. The fix splits them into separate ContentType messages.

Reference: https://ai.google.dev/gemini-api/docs/thought-signatures
"Don't merge one part with a signature with another part without a signature"
"""

import json
import pytest

from litellm.llms.vertex_ai.gemini.transformation import (
    _gemini_convert_messages_with_history,
)


def _run_conversion(messages, model="gemini-2.5-flash"):
    """Helper to run message conversion."""
    return _gemini_convert_messages_with_history(messages=messages, model=model)


class TestSplitSignedUnsignedParts:
    """Test that assistant messages with mixed signed/unsigned parts are split."""

    def test_text_and_tool_call_with_signature_are_split(self):
        """
        When an assistant message has text (no signature) and a tool call
        (with signature embedded in ID), they should be split into separate
        ContentType messages to avoid Gemini's 400 error.
        """
        from litellm.litellm_core_utils.prompt_templates.factory import (
            _encode_tool_call_id_with_signature,
        )

        sig = "Co4CAdHtim_test_signature_abc123"
        tool_id = _encode_tool_call_id_with_signature("call_123", sig)

        messages = [
            {"role": "user", "content": "Read the file and explain it."},
            {
                "role": "assistant",
                "content": "Let me read that file for you.",
                "tool_calls": [
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "arguments": json.dumps({"file_path": "/tmp/test.py"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": "def hello():\n    print('hello')",
            },
        ]

        contents = _run_conversion(messages)

        # Find model messages
        model_messages = [c for c in contents if c["role"] == "model"]

        # There should be 2 model messages (split: text without sig, functionCall with sig)
        assert len(model_messages) == 2, (
            f"Expected 2 model messages (split), got {len(model_messages)}. "
            f"Parts: {[[list(p.keys()) for p in m['parts']] for m in model_messages]}"
        )

        # First model message: text only (no thoughtSignature)
        first_parts = model_messages[0]["parts"]
        assert all("thoughtSignature" not in p for p in first_parts), (
            "First model message should have no thoughtSignature"
        )
        assert any("text" in p for p in first_parts), (
            "First model message should have text"
        )

        # Second model message: functionCall with thoughtSignature
        second_parts = model_messages[1]["parts"]
        assert any("function_call" in p for p in second_parts), (
            "Second model message should have function_call"
        )
        assert any("thoughtSignature" in p for p in second_parts), (
            "Second model message should have thoughtSignature"
        )

    def test_no_split_when_all_parts_have_signature(self):
        """When all parts have thoughtSignature, no split needed."""
        from litellm.litellm_core_utils.prompt_templates.factory import (
            _encode_tool_call_id_with_signature,
        )

        sig = "test_sig_all_signed"
        tool_id = _encode_tool_call_id_with_signature("call_456", sig)

        messages = [
            {"role": "user", "content": "Do something."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": "Bash",
                            "arguments": json.dumps({"command": "ls"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": "file1.py\nfile2.py",
            },
        ]

        contents = _run_conversion(messages)
        model_messages = [c for c in contents if c["role"] == "model"]

        # Only 1 model message (no split needed since content is None)
        assert len(model_messages) == 1

    def test_no_split_when_no_signatures(self):
        """When no parts have thoughtSignature, no split needed."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        contents = _run_conversion(messages)
        model_messages = [c for c in contents if c["role"] == "model"]

        # Exactly 1 model message, no split
        assert len(model_messages) == 1

    def test_multi_turn_with_signature_split(self):
        """
        Multi-turn conversation: 2 rounds of tool calling.
        Each assistant message with text + tool_call should be split.
        """
        from litellm.litellm_core_utils.prompt_templates.factory import (
            _encode_tool_call_id_with_signature,
        )

        sig1 = "sig_round_1"
        sig2 = "sig_round_2"
        tid1 = _encode_tool_call_id_with_signature("call_r1", sig1)
        tid2 = _encode_tool_call_id_with_signature("call_r2", sig2)

        messages = [
            {"role": "user", "content": "Read file A and B."},
            # Round 1
            {
                "role": "assistant",
                "content": "Reading file A...",
                "tool_calls": [
                    {
                        "id": tid1,
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "arguments": json.dumps({"file_path": "/a.py"}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": tid1, "content": "content of a.py"},
            # Round 2
            {
                "role": "assistant",
                "content": "Now reading file B...",
                "tool_calls": [
                    {
                        "id": tid2,
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "arguments": json.dumps({"file_path": "/b.py"}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": tid2, "content": "content of b.py"},
        ]

        contents = _run_conversion(messages)
        model_messages = [c for c in contents if c["role"] == "model"]

        # Each round should produce 2 model messages (text + functionCall split)
        # Total: 4 model messages
        assert len(model_messages) == 4, (
            f"Expected 4 model messages (2 rounds × 2 split), got {len(model_messages)}"
        )

        # Verify user messages with functionResponse still have correct count
        user_messages = [c for c in contents if c["role"] == "user"]
        for um in user_messages:
            fr_count = sum(1 for p in um["parts"] if "function_response" in p)
            # Each user message with tool results should have exactly 1 functionResponse
            if fr_count > 0:
                assert fr_count == 1
