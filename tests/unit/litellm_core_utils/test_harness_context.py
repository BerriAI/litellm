import pytest

from litellm.litellm_core_utils.harness_context import (
    CODEX_MARKERS,
    SYSTEM_REMINDER_MARKERS,
    strip_marked_blocks,
    without_harness_context,
    without_marked_blocks,
)

_REMINDER = "<system-reminder>\nContents of AGENTS.md\n</system-reminder>"


def test_text_after_a_character_that_lowercases_longer_keeps_its_own_characters() -> None:
    text = f"İstanbul {_REMINDER} weather today"

    assert strip_marked_blocks(text) == "İstanbul weather today"
    assert without_marked_blocks(text, SYSTEM_REMINDER_MARKERS) == "İstanbul\n\nweather today"


@pytest.mark.parametrize(
    "text",
    ["  plain text with trailing space \n", "an unclosed <system-reminder> tag stays", "</system-reminder> alone"],
)
def test_text_with_no_complete_block_is_left_byte_for_byte(text: str) -> None:
    assert without_marked_blocks(text, SYSTEM_REMINDER_MARKERS) == text


def test_a_conversation_without_harness_blocks_is_unchanged() -> None:
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": [{"type": "text", "text": "  keep my spacing  "}]},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "function": {"name": "ls"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "a.txt"},
    ]

    assert without_harness_context(messages, CODEX_MARKERS) == messages


def test_reminders_leave_every_message_and_tool_result_but_the_conversation_stays_whole() -> None:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": _REMINDER}, {"type": "text", "text": "fix the bug"}]},
        {"role": "assistant", "content": _REMINDER, "tool_calls": [{"id": "call-1", "function": {"name": "ls"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": _REMINDER},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call-2", "content": f"a.txt\n\n{_REMINDER}"}],
        },
        {"type": "function_call_output", "call_id": "call-3", "output": f"b.txt {_REMINDER}"},
        {"role": "user", "content": _REMINDER},
        {"role": "user", "content": [{"type": "text", "text": _REMINDER}]},
    ]

    assert without_harness_context(messages, SYSTEM_REMINDER_MARKERS) == [
        {"role": "user", "content": [{"type": "text", "text": "fix the bug"}]},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "function": {"name": "ls"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": ""},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-2", "content": "a.txt"}]},
        {"type": "function_call_output", "call_id": "call-3", "output": "b.txt"},
    ]


def test_a_codex_instruction_message_is_dropped_while_its_markers_are_only_used_when_asked_for() -> None:
    agents_md = {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": "# AGENTS.md instructions for /work\n\n<INSTRUCTIONS>\nUse tabs.\n</INSTRUCTIONS>",
            }
        ],
    }
    typed = {"role": "user", "content": [{"type": "input_text", "text": "count the words"}]}

    assert without_harness_context([agents_md, typed], CODEX_MARKERS) == [typed]
    assert without_harness_context([agents_md, typed], SYSTEM_REMINDER_MARKERS) == [agents_md, typed]


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        (f"hello {_REMINDER}", "hello"),
        (None, None),
        ({"not": "a list"}, {"not": "a list"}),
        (["raw string item", 3], ["raw string item", 3]),
    ],
)
def test_messages_that_are_not_a_list_of_objects_pass_through(messages: object, expected: object) -> None:
    assert without_harness_context(messages, SYSTEM_REMINDER_MARKERS) == expected
