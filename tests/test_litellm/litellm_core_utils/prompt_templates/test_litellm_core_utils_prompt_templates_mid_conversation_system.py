"""Placement policy for mid-conversation ``role: "system"`` messages on the chat path.

The provider-facing behaviour is covered through ``transform_request`` in the
Anthropic, Vertex, Azure AI and Bedrock Invoke transformation tests; these pin
the pure placement rules on the OpenAI-format message list.
"""

import pytest

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import encrypted_reasoning_signature
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import (
    CONVERTED_SYSTEM_NOTE,
    place_mid_conversation_system,
    split_leading_system_run,
)


def _roles(messages: object) -> list[str]:
    return [m["role"] if isinstance(m, dict) else m.role for m in messages]


def _texts(message: dict) -> list[str]:
    return [block["text"] for block in message["content"]]


SENDS_NOTHING = pytest.mark.parametrize(
    "empty_content",
    [[], None, [{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}]],
    ids=["empty-list", "none", "unsupported-part-only"],
)


def test_split_leading_system_run_keeps_later_system_messages_in_the_conversation():
    messages = [
        {"role": "system", "content": "one"},
        {"role": "system", "content": "two"},
        {"role": "user", "content": "q"},
        {"role": "system", "content": "reminder"},
    ]

    leading, later = split_leading_system_run(messages)

    assert [m["content"] for m in leading] == ["one", "two"]
    assert _roles(later) == ["user", "system"]


def test_flagged_placement_moves_a_system_run_after_the_user_turn_that_follows_it():
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "system", "content": "reminder"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]

    placed = place_mid_conversation_system(messages, supports_mid_conversation_system=True)

    assert _roles(placed) == ["user", "assistant", "user", "system", "assistant"]


def test_flagged_placement_pushes_a_system_between_two_user_turns_after_both():
    """Two user turns collapse into one on the wire, and a system message must
    be followed by an assistant turn or nothing."""
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "system", "content": "reminder"},
        {"role": "user", "content": "q2"},
    ]

    placed = place_mid_conversation_system(messages, supports_mid_conversation_system=True)

    assert _roles(placed) == ["user", "user", "system"]


def test_flagged_placement_keeps_a_system_after_tool_results():
    messages = [
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "r"},
        {"role": "system", "content": "reminder"},
        {"role": "assistant", "content": "a2"},
    ]

    placed = place_mid_conversation_system(messages, supports_mid_conversation_system=True)

    assert _roles(placed) == ["user", "assistant", "tool", "system", "assistant"]


def test_flagged_placement_drops_a_system_message_with_no_text():
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "system", "content": ""},
        {"role": "assistant", "content": "a1"},
    ]

    placed = place_mid_conversation_system(messages, supports_mid_conversation_system=True)

    assert _roles(placed) == ["user", "assistant"]


def test_placement_reads_roles_off_pydantic_messages_in_the_history():
    """Callers routinely append the previous ``litellm.Message`` object straight
    into the history; placement must read its role without assuming a dict and
    hand the object through untouched."""
    assistant = litellm.Message(role="assistant", content="a1")
    messages = [
        {"role": "user", "content": "q1"},
        assistant,
        {"role": "system", "content": "reminder"},
        {"role": "user", "content": "q2"},
    ]

    placed = place_mid_conversation_system(messages, supports_mid_conversation_system=True)

    assert _roles(placed) == ["user", "assistant", "user", "system"]
    assert placed[1] is assistant


def test_unflagged_conversion_keeps_the_client_order_when_no_tool_result_follows():
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "system", "content": "reminder"},
        {"role": "user", "content": "q2"},
    ]

    placed = place_mid_conversation_system(messages, supports_mid_conversation_system=False)

    assert _roles(placed) == ["user", "assistant", "user", "user"]
    assert _texts(placed[2]) == [CONVERTED_SYSTEM_NOTE, "reminder"]


@pytest.mark.parametrize(
    "cache_control, expected",
    [
        ({"type": "ephemeral", "ttl": "1h"}, {"type": "ephemeral", "ttl": "1h"}),
        ({"type": "ephemeral", "ttl": "5m"}, {"type": "ephemeral", "ttl": "5m"}),
        ({"type": "ephemeral", "ttl": "2h"}, {"type": "ephemeral"}),
    ],
)
def test_unflagged_conversion_rebuilds_cache_control_on_the_converted_block(cache_control, expected):
    """Only the shapes Anthropic accepts survive: ephemeral with a 5m or 1h ttl, or no ttl."""
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "system", "content": "reminder", "cache_control": cache_control},
    ]

    placed = place_mid_conversation_system(messages, supports_mid_conversation_system=False)

    assert placed[1]["content"][1] == {"type": "text", "text": "reminder", "cache_control": expected}


def test_unflagged_conversion_drops_a_cache_control_that_is_not_ephemeral():
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "system", "content": "reminder", "cache_control": {"type": "persistent"}},
    ]

    placed = place_mid_conversation_system(messages, supports_mid_conversation_system=False)

    assert placed[1]["content"][1] == {"type": "text", "text": "reminder"}


def test_placement_is_a_no_op_without_later_system_messages():
    messages = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]

    assert place_mid_conversation_system(messages, supports_mid_conversation_system=False) == tuple(messages)
    assert place_mid_conversation_system(messages, supports_mid_conversation_system=True) == tuple(messages)


def test_flagged_placement_converts_a_run_followed_by_an_assistant_turn_in_place():
    placed = place_mid_conversation_system(
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "system", "content": "reminder"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q2"},
        ],
        supports_mid_conversation_system=True,
    )

    assert _roles(placed) == ["user", "assistant", "user", "assistant", "user"]
    assert _texts(placed[2]) == [CONVERTED_SYSTEM_NOTE, "reminder"]


def test_flagged_placement_of_an_earlier_run_does_not_move_when_later_turns_are_appended():
    turn_n = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "system", "content": "reminder"},
    ]
    turn_n_plus_one = [
        *turn_n,
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q2"},
    ]

    placed_n = place_mid_conversation_system(turn_n, supports_mid_conversation_system=True)
    placed_n_plus_one = place_mid_conversation_system(turn_n_plus_one, supports_mid_conversation_system=True)

    assert placed_n_plus_one[: len(placed_n)] == placed_n
    assert _roles(placed_n_plus_one) == ["user", "assistant", "user", "assistant", "user"]


@SENDS_NOTHING
def test_flagged_placement_converts_a_run_whose_preceding_user_turn_sends_nothing(empty_content):
    placed = place_mid_conversation_system(
        [
            {"role": "user", "content": empty_content},
            {"role": "system", "content": "reminder"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ],
        supports_mid_conversation_system=True,
    )

    assert _roles(placed) == ["user", "user", "assistant", "user"]
    assert _texts(placed[1]) == [CONVERTED_SYSTEM_NOTE, "reminder"]


@SENDS_NOTHING
def test_flagged_placement_converts_a_run_whose_following_user_turn_sends_nothing(empty_content):
    placed = place_mid_conversation_system(
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "system", "content": "reminder"},
            {"role": "user", "content": empty_content},
        ],
        supports_mid_conversation_system=True,
    )

    assert _roles(placed) == ["user", "assistant", "user", "user"]
    assert _texts(placed[2]) == [CONVERTED_SYSTEM_NOTE, "reminder"]


def test_flagged_placement_keeps_a_system_behind_a_user_turn_merged_with_an_empty_one():
    placed = place_mid_conversation_system(
        [
            {"role": "user", "content": "q1"},
            {"role": "user", "content": []},
            {"role": "system", "content": "reminder"},
            {"role": "assistant", "content": "a1"},
        ],
        supports_mid_conversation_system=True,
    )

    assert _roles(placed) == ["user", "user", "system", "assistant"]


EMPTY_ASSISTANT = pytest.mark.parametrize(
    "empty_assistant",
    [
        {"role": "assistant", "content": None},
        {"role": "assistant", "content": []},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "unsigned"}]},
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "bridged", "signature": encrypted_reasoning_signature("abc")}],
        },
        {
            "role": "assistant",
            "content": None,
            "thinking_blocks": [{"type": "redacted_thinking", "data": encrypted_reasoning_signature("abc")}],
        },
        litellm.Message(role="assistant", content=None),
    ],
    ids=[
        "none",
        "empty-list",
        "unsigned-thinking-part",
        "encrypted-thinking-part",
        "encrypted-redacted-thinking-block",
        "pydantic-none",
    ],
)


@EMPTY_ASSISTANT
def test_flagged_placement_converts_a_run_when_the_assistant_turn_after_its_anchor_sends_nothing(empty_assistant):
    placed = place_mid_conversation_system(
        [
            {"role": "user", "content": "q1"},
            {"role": "system", "content": "reminder"},
            empty_assistant,
            {"role": "user", "content": "q2"},
        ],
        supports_mid_conversation_system=True,
    )

    assert _roles(placed) == ["user", "user", "assistant", "user"]
    assert _texts(placed[1]) == [CONVERTED_SYSTEM_NOTE, "reminder"]
    assert placed[2] is empty_assistant


@EMPTY_ASSISTANT
def test_flagged_placement_keeps_a_system_whose_empty_assistant_follower_ends_the_array(empty_assistant):
    placed = place_mid_conversation_system(
        [{"role": "user", "content": "q1"}, {"role": "system", "content": "reminder"}, empty_assistant],
        supports_mid_conversation_system=True,
    )

    assert _roles(placed) == ["user", "system", "assistant"]


@pytest.mark.parametrize(
    "assistant_turn",
    [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "toolu_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        },
        {
            "role": "assistant",
            "content": None,
            "thinking_blocks": [{"type": "thinking", "thinking": "hm", "signature": "s"}],
        },
        {"role": "assistant", "content": None, "thinking_blocks": [{"type": "redacted_thinking", "data": "x"}]},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "hm", "signature": "s"}]},
        {"role": "assistant", "content": None, "function_call": {"name": "f", "arguments": "{}"}},
        litellm.Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "toolu_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        ),
    ],
    ids=[
        "tool-calls",
        "signed-thinking-block",
        "redacted-thinking-block",
        "signed-thinking-part",
        "function-call",
        "pydantic-tool-calls",
    ],
)
def test_flagged_placement_keeps_a_system_before_an_assistant_turn_that_renders_without_text(assistant_turn):
    placed = place_mid_conversation_system(
        [
            {"role": "user", "content": "q1"},
            {"role": "system", "content": "reminder"},
            assistant_turn,
            {"role": "user", "content": "q2"},
        ],
        supports_mid_conversation_system=True,
    )

    assert _roles(placed) == ["user", "system", "assistant", "user"]


@pytest.mark.parametrize(
    "padded_assistant",
    [
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": "   "},
        {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        litellm.Message(role="assistant", content=""),
    ],
    ids=["empty-string", "whitespace-string", "empty-text-part", "pydantic-empty-string"],
)
def test_flagged_placement_keeps_a_system_before_an_assistant_turn_whose_empty_text_the_converter_pads(
    padded_assistant,
):
    placed = place_mid_conversation_system(
        [
            {"role": "user", "content": "q1"},
            {"role": "system", "content": "reminder"},
            padded_assistant,
            {"role": "user", "content": "q2"},
        ],
        supports_mid_conversation_system=True,
    )

    assert _roles(placed) == ["user", "system", "assistant", "user"]
