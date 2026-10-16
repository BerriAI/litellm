from typing import Final

import pytest

from litellm.proxy.telemetry.blocks import count_blocks
from litellm.telemetry.records import BlockCounts, BlockType


def test_counts_string_content_part_lists_tool_calls_and_tool_messages() -> None:
    messages: Final = [
        {"role": "system", "content": "be brief"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is in this picture"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "something_new"},
            ],
        },
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a"}, {"id": "b"}]},
        {"role": "tool", "content": "42", "tool_call_id": "a"},
    ]
    assert count_blocks(messages) == BlockCounts(
        total=7,
        by_type=(
            (BlockType.IMAGE, 1),
            (BlockType.OTHER, 1),
            (BlockType.TEXT, 2),
            (BlockType.TOOL_RESULT, 1),
            (BlockType.TOOL_USE, 2),
        ),
    )


@pytest.mark.parametrize("messages", [None, "a prompt string", [{"content": "no role"}]])
def test_anything_that_is_not_a_message_list_has_no_block_counts(messages: object) -> None:
    assert count_blocks(messages) is None
