"""
Regression tests for https://github.com/BerriAI/litellm/issues/23047.

Anthropic requires the latest assistant turn's thinking/tool-use blocks to
be passed back unmodified. These tests verify that `anthropic_messages_pt()`
uses the provider-specific ordered block history when it is available.
"""

import copy
import json
from typing import Optional

from litellm.litellm_core_utils.prompt_templates.factory import anthropic_messages_pt


def _make_ordered_content_blocks(
    *,
    num_thinking: int = 2,
    num_tool_use: int = 2,
    include_compaction: bool = False,
    text: str = "Here are the results so far.",
    cache_control: Optional[dict] = None,
) -> list[dict]:
    ordered_content_blocks: list[dict] = []

    if include_compaction:
        ordered_content_blocks.append(
            {"type": "compaction", "content": "compressed history"}
        )

    thinking_blocks = [
        {
            "type": "thinking",
            "thinking": f"Thinking round {i + 1}.",
            "signature": f"sig_{i + 1}",
        }
        for i in range(num_thinking)
    ]

    if thinking_blocks:
        ordered_content_blocks.append(copy.deepcopy(thinking_blocks[0]))

    text_block = {"type": "text", "text": text}
    if cache_control is not None:
        text_block["cache_control"] = cache_control
    ordered_content_blocks.append(text_block)

    for i in range(num_tool_use):
        tool_use_id = f"srvtoolu_{i + 1:03d}"
        ordered_content_blocks.append(
            {
                "type": "server_tool_use",
                "id": tool_use_id,
                "name": "web_search",
                "input": {"query": f"search {i + 1}"},
            }
        )
        ordered_content_blocks.append(
            {
                "type": "web_search_tool_result",
                "tool_use_id": tool_use_id,
                "content": [
                    {
                        "type": "web_search_result",
                        "url": f"https://example.com/{i + 1}",
                        "title": f"Result {i + 1}",
                    }
                ],
            }
        )

        next_thinking_index = i + 1
        if next_thinking_index < len(thinking_blocks):
            ordered_content_blocks.append(
                copy.deepcopy(thinking_blocks[next_thinking_index])
            )

    for trailing_thinking in thinking_blocks[num_tool_use + 1 :]:
        ordered_content_blocks.append(copy.deepcopy(trailing_thinking))

    return ordered_content_blocks


def _make_assistant_message(
    ordered_content_blocks: list[dict],
    *,
    include_ordered_content_blocks: bool = True,
) -> dict:
    tool_calls = [
        {
            "id": block["id"],
            "type": "function",
            "function": {
                "name": block["name"],
                "arguments": json.dumps(block.get("input", {})),
            },
        }
        for block in ordered_content_blocks
        if block.get("type") in ("tool_use", "server_tool_use")
    ]
    thinking_blocks = [
        copy.deepcopy(block)
        for block in ordered_content_blocks
        if block.get("type") in ("thinking", "redacted_thinking")
    ]
    provider_specific_fields = {
        "web_search_results": [
            copy.deepcopy(block)
            for block in ordered_content_blocks
            if block.get("type") == "web_search_tool_result"
        ]
    }

    compaction_blocks = [
        copy.deepcopy(block)
        for block in ordered_content_blocks
        if block.get("type") == "compaction"
    ]
    if compaction_blocks:
        provider_specific_fields["compaction_blocks"] = compaction_blocks

    if include_ordered_content_blocks:
        provider_specific_fields["ordered_content_blocks"] = copy.deepcopy(
            ordered_content_blocks
        )

    return {
        "role": "assistant",
        "content": "".join(
            block["text"]
            for block in ordered_content_blocks
            if block.get("type") == "text"
        ),
        "thinking_blocks": thinking_blocks,
        "tool_calls": tool_calls,
        "provider_specific_fields": provider_specific_fields,
    }


def _assistant_content(messages: list[dict]) -> list[dict]:
    result = anthropic_messages_pt(
        messages=messages,
        model="claude-sonnet-4-6",
        llm_provider="anthropic",
    )
    assistant_messages = [message for message in result if message["role"] == "assistant"]
    assert len(assistant_messages) == 1
    return assistant_messages[0]["content"]


def test_ordered_content_blocks_preserve_interleaved_search_sequence() -> None:
    ordered_content_blocks = _make_ordered_content_blocks(num_thinking=2, num_tool_use=2)

    content = _assistant_content(
        [
            {"role": "user", "content": "Search for fast.ai and answer.ai"},
            _make_assistant_message(ordered_content_blocks),
            {"role": "user", "content": "Follow up question"},
        ]
    )

    assert [block["type"] for block in content] == [
        block["type"] for block in ordered_content_blocks
    ]
    assert [block["signature"] for block in content if block.get("type") == "thinking"] == [
        "sig_1",
        "sig_2",
    ]


def test_ordered_content_blocks_handle_more_tool_uses_than_thinking_blocks() -> None:
    ordered_content_blocks = _make_ordered_content_blocks(num_thinking=1, num_tool_use=3)

    content = _assistant_content(
        [
            {"role": "user", "content": "Search three things"},
            _make_assistant_message(ordered_content_blocks),
            {"role": "user", "content": "Thanks"},
        ]
    )

    assert [block["type"] for block in content] == [
        block["type"] for block in ordered_content_blocks
    ]


def test_ordered_content_blocks_preserve_compaction_and_cache_control() -> None:
    ordered_content_blocks = _make_ordered_content_blocks(
        num_thinking=2,
        num_tool_use=2,
        include_compaction=True,
        cache_control={"type": "ephemeral"},
    )

    content = _assistant_content(
        [
            {"role": "user", "content": "Search"},
            _make_assistant_message(ordered_content_blocks),
            {"role": "user", "content": "Follow up"},
        ]
    )

    assert [block["type"] for block in content] == [
        block["type"] for block in ordered_content_blocks
    ]
    text_blocks = [block for block in content if block.get("type") == "text"]
    assert text_blocks[0]["cache_control"] == {"type": "ephemeral"}


def test_merged_consecutive_assistant_messages_preserve_each_sequence() -> None:
    first_ordered_content_blocks = _make_ordered_content_blocks(
        num_thinking=1,
        num_tool_use=1,
        text="First search result.",
    )
    second_ordered_content_blocks = _make_ordered_content_blocks(
        num_thinking=1,
        num_tool_use=1,
        text="Second search result.",
    )
    second_ordered_content_blocks[0]["signature"] = "sig_2b"
    second_ordered_content_blocks[1]["text"] = "Second search result."
    second_ordered_content_blocks[2]["id"] = "srvtoolu_999"
    second_ordered_content_blocks[2]["input"] = {"query": "search 999"}
    second_ordered_content_blocks[3]["tool_use_id"] = "srvtoolu_999"

    content = _assistant_content(
        [
            {"role": "user", "content": "Search twice"},
            _make_assistant_message(first_ordered_content_blocks),
            _make_assistant_message(second_ordered_content_blocks),
            {"role": "user", "content": "What happened?"},
        ]
    )

    expected_order = first_ordered_content_blocks + second_ordered_content_blocks
    assert [block["type"] for block in content] == [
        block["type"] for block in expected_order
    ]
    assert [block["id"] for block in content if block.get("type") == "server_tool_use"] == [
        "srvtoolu_001",
        "srvtoolu_999",
    ]


def test_legacy_fallback_without_ordered_content_blocks_still_builds_assistant_content() -> None:
    ordered_content_blocks = _make_ordered_content_blocks(num_thinking=2, num_tool_use=2)

    content = _assistant_content(
        [
            {"role": "user", "content": "Search"},
            _make_assistant_message(
                ordered_content_blocks,
                include_ordered_content_blocks=False,
            ),
            {"role": "user", "content": "Follow up"},
        ]
    )

    assert [block["type"] for block in content] == [
        "thinking",
        "thinking",
        "text",
        "server_tool_use",
        "web_search_tool_result",
        "server_tool_use",
        "web_search_tool_result",
    ]
