"""
Regression test for issue #28978:

Responses API: function_call not converted to tool_calls in mixed-content
assistant messages.

Before the fix, an assistant input item whose `content` list mixed text and
`function_call` parts had the function_call silently dropped, leaving the
matching `function_call_output` orphaned and breaking the downstream Chat
Completions conversation with:

    Missing corresponding tool call for tool response message.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


def _make_input():
    """Repro payload from issue #28978."""
    return [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "ok"},
                {
                    "type": "function_call",
                    "id": "call_good",
                    "name": "test",
                    "arguments": "{}",
                },
            ],
        },
        {
            "type": "function_call_output",
            "call_id": "call_good",
            "output": "result",
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "continue"}],
        },
    ]


def _get(msg, key, default=None):
    return msg.get(key, default) if isinstance(msg, dict) else getattr(msg, key, default)


def test_mixed_content_function_call_emits_tool_calls():
    """The function_call part inside an assistant.content list must be lifted
    into the assistant message's `tool_calls`, with text parts preserved as
    content. The follow-up function_call_output must still be paired by
    matching call_id."""
    messages = LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
        input=_make_input()
    )

    roles = [_get(m, "role") for m in messages]
    assert "assistant" in roles, f"missing assistant message: roles={roles}"

    assistant_msg = next(m for m in messages if _get(m, "role") == "assistant")
    tool_calls = _get(assistant_msg, "tool_calls") or []
    assert len(tool_calls) == 1, (
        f"expected exactly 1 tool_call lifted from mixed content, got {len(tool_calls)} (assistant={assistant_msg})"
    )
    tc = tool_calls[0]
    tc_id = _get(tc, "id")
    tc_fn = _get(tc, "function") or {}
    assert tc_id == "call_good", f"tool_call id mismatch: {tc_id!r}"
    assert _get(tc_fn, "name") == "test"
    assert _get(tc_fn, "arguments") == "{}"
    # index must be the position among tool_calls (0), not the position in the
    # raw content list (would be 1 here, after the leading text part). #29328
    assert _get(tc, "index") == 0, f"tool_call index should be 0, got {_get(tc, 'index')!r}"

    # Text part should survive as content (either as a list with text block
    # or as the normalized string "ok"). What we care about is that "ok"
    # is still reachable.
    content = _get(assistant_msg, "content")
    if isinstance(content, list):
        text_blob = "".join((p.get("text") if isinstance(p, dict) else "") or "" for p in content)
    else:
        text_blob = content or ""
    assert "ok" in text_blob, f"text part lost: content={content!r}"


def test_mixed_content_pairs_with_function_call_output():
    """The matching `function_call_output` (a `role=tool` message after the
    transformation) must be present and reference the same call_id."""
    messages = LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
        input=_make_input()
    )

    tool_msgs = [m for m in messages if _get(m, "role") == "tool"]
    assert len(tool_msgs) == 1, (
        f"expected exactly 1 tool message paired with the lifted tool_call, "
        f"got {len(tool_msgs)} (roles={[_get(m, 'role') for m in messages]})"
    )
    assert _get(tool_msgs[0], "tool_call_id") == "call_good"


def test_text_only_assistant_unchanged():
    """Regression guard — a text-only assistant.content list must not gain a
    tool_calls entry from the new code path."""
    messages = LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
        input=[
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
            }
        ]
    )
    assert len(messages) == 1
    assert _get(messages[0], "role") == "assistant"
    assert not (_get(messages[0], "tool_calls") or []), (
        f"text-only assistant gained spurious tool_calls: {messages[0]!r}"
    )
