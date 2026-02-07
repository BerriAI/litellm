
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath("."))

from litellm.litellm_core_utils.prompt_templates.factory import (
    _bedrock_converse_messages_pt,
    _deduplicate_bedrock_content_blocks,
    _deduplicate_bedrock_tool_content,
    BedrockConverseMessagesProcessor,
)


MODEL = "anthropic.claude-v2"
PROVIDER = "bedrock_converse"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_duplicate_tool_result_messages():
    """Return messages where two consecutive tool-role messages reference the
    same tool_call_id, simulating the duplication scenario."""
    return [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tooluse_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tooluse_abc123",
            "content": '{"temp": 22}',
        },
        {
            "role": "tool",
            "tool_call_id": "tooluse_abc123",  # DUPLICATE
            "content": '{"temp": 22}',
        },
    ]


def _make_duplicate_tool_use_messages():
    """Return messages where two consecutive assistant messages carry tool_calls
    with the same id, simulating assistant-side duplication."""
    return [
        {"role": "user", "content": "Do something"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_1",
                    "type": "function",
                    "function": {"name": "fn_a", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_1",  # DUPLICATE
                    "type": "function",
                    "function": {"name": "fn_a", "arguments": "{}"},
                },
            ],
        },
        # Need a tool result so the conversation is valid
        {
            "role": "tool",
            "tool_call_id": "tool_1",
            "content": '{"ok": true}',
        },
    ]


def _extract_blocks(result, role, key):
    """Extract all content blocks containing ``key`` from messages with ``role``."""
    return [
        block
        for msg in result
        if msg["role"] == role
        for block in msg["content"]
        if key in block
    ]


# ---------------------------------------------------------------------------
# toolResult dedup tests
# ---------------------------------------------------------------------------


def test_bedrock_converse_deduplicates_tool_results():
    """Verify _bedrock_converse_messages_pt deduplicates toolResult blocks
    with the same toolUseId when merging consecutive tool messages."""
    messages = _make_duplicate_tool_result_messages()
    result = _bedrock_converse_messages_pt(messages, MODEL, PROVIDER)

    tool_results = _extract_blocks(result, "user", "toolResult")
    ids = [tr["toolResult"]["toolUseId"] for tr in tool_results]
    assert ids.count("tooluse_abc123") == 1


@pytest.mark.asyncio
async def test_bedrock_converse_deduplicates_tool_results_async():
    """Verify the async path also deduplicates toolResult blocks with the
    same toolUseId when merging consecutive tool messages."""
    messages = _make_duplicate_tool_result_messages()
    result = await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
        messages, MODEL, PROVIDER
    )

    tool_results = _extract_blocks(result, "user", "toolResult")
    ids = [tr["toolResult"]["toolUseId"] for tr in tool_results]
    assert ids.count("tooluse_abc123") == 1


def test_bedrock_converse_preserves_unique_tool_results():
    """Different toolUseIds should all be preserved."""
    messages = [
        {"role": "user", "content": "Weather and time?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                },
                {
                    "id": "tool_2",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "tool_1", "content": '{"temp": 22}'},
        {"role": "tool", "tool_call_id": "tool_2", "content": '{"time": "14:00"}'},
    ]

    result = _bedrock_converse_messages_pt(messages, MODEL, PROVIDER)

    tool_results = _extract_blocks(result, "user", "toolResult")
    assert len(tool_results) == 2
    ids = {tr["toolResult"]["toolUseId"] for tr in tool_results}
    assert ids == {"tool_1", "tool_2"}


def test_bedrock_converse_dedup_preserves_cache_points():
    """cachePoint blocks should not be removed during dedup."""
    messages = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tool_1",
            "content": [
                {
                    "type": "text",
                    "text": "sunny",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tool_1",  # DUPLICATE
            "content": '{"temp": 22}',
        },
    ]

    result = _bedrock_converse_messages_pt(messages, MODEL, PROVIDER)

    tool_results = _extract_blocks(result, "user", "toolResult")
    cache_points = _extract_blocks(result, "user", "cachePoint")

    assert len(tool_results) == 1
    assert len(cache_points) == 1


# ---------------------------------------------------------------------------
# toolUse dedup tests
# ---------------------------------------------------------------------------


def test_bedrock_converse_deduplicates_tool_use_sync():
    """Verify the sync path deduplicates toolUse blocks with the same
    toolUseId when merging consecutive assistant messages."""
    messages = _make_duplicate_tool_use_messages()
    result = _bedrock_converse_messages_pt(messages, MODEL, PROVIDER)

    tool_uses = _extract_blocks(result, "assistant", "toolUse")
    ids = [tu["toolUse"]["toolUseId"] for tu in tool_uses]
    assert ids.count("tool_1") == 1


@pytest.mark.asyncio
async def test_bedrock_converse_deduplicates_tool_use_async():
    """Verify the async path deduplicates toolUse blocks with the same
    toolUseId when merging consecutive assistant messages."""
    messages = _make_duplicate_tool_use_messages()
    result = await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
        messages, MODEL, PROVIDER
    )

    tool_uses = _extract_blocks(result, "assistant", "toolUse")
    ids = [tu["toolUse"]["toolUseId"] for tu in tool_uses]
    assert ids.count("tool_1") == 1


@pytest.mark.asyncio
async def test_bedrock_converse_tool_use_sync_async_parity():
    """Sync and async paths should produce identical results for duplicate
    toolUse blocks."""
    messages = _make_duplicate_tool_use_messages()
    sync_result = _bedrock_converse_messages_pt(messages, MODEL, PROVIDER)
    async_result = await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
        messages, MODEL, PROVIDER
    )
    assert sync_result == async_result


# ---------------------------------------------------------------------------
# Generalized helper unit tests
# ---------------------------------------------------------------------------


def test_deduplicate_bedrock_content_blocks_tool_result():
    """Direct unit test: first occurrence wins, duplicates dropped, non-tool
    blocks preserved."""
    blocks = [
        {"toolResult": {"toolUseId": "id_1", "content": [{"text": "a"}]}},
        {"cachePoint": {"type": "default"}},
        {"toolResult": {"toolUseId": "id_1", "content": [{"text": "b"}]}},  # duplicate
        {"toolResult": {"toolUseId": "id_2", "content": [{"text": "c"}]}},
    ]

    result = _deduplicate_bedrock_content_blocks(blocks, "toolResult")

    assert len(result) == 3  # id_1, cachePoint, id_2
    tool_ids = [b["toolResult"]["toolUseId"] for b in result if "toolResult" in b]
    assert tool_ids == ["id_1", "id_2"]
    # First-wins: content "a" is kept, "b" is dropped
    assert result[0]["toolResult"]["content"] == [{"text": "a"}]


def test_deduplicate_bedrock_content_blocks_tool_use():
    """Direct unit test of toolUse dedup via the generalized helper."""
    blocks = [
        {"toolUse": {"toolUseId": "id_1", "name": "fn_a", "input": {}}},
        {"text": "thinking..."},
        {"toolUse": {"toolUseId": "id_1", "name": "fn_a", "input": {}}},  # duplicate
        {"toolUse": {"toolUseId": "id_2", "name": "fn_b", "input": {}}},
    ]

    result = _deduplicate_bedrock_content_blocks(blocks, "toolUse")

    assert len(result) == 3  # id_1, text, id_2
    tool_ids = [b["toolUse"]["toolUseId"] for b in result if "toolUse" in b]
    assert tool_ids == ["id_1", "id_2"]


def test_deduplicate_preserves_blocks_with_missing_id():
    """Blocks where toolUseId is None or empty should pass through without
    dedup tracking (they cannot be compared)."""
    blocks = [
        {"toolResult": {"toolUseId": None, "content": [{"text": "a"}]}},
        {"toolResult": {"toolUseId": "", "content": [{"text": "b"}]}},
        {"toolResult": {"toolUseId": "id_1", "content": [{"text": "c"}]}},
    ]

    result = _deduplicate_bedrock_content_blocks(blocks, "toolResult")

    # All three should be preserved — None and "" are not tracked
    assert len(result) == 3


def test_deduplicate_bedrock_tool_content_convenience_wrapper():
    """The convenience wrapper should behave identically to calling the
    generalized helper with block_key='toolResult'."""
    blocks = [
        {"toolResult": {"toolUseId": "id_1", "content": [{"text": "a"}]}},
        {"toolResult": {"toolUseId": "id_1", "content": [{"text": "b"}]}},
    ]

    assert _deduplicate_bedrock_tool_content(blocks) == _deduplicate_bedrock_content_blocks(blocks, "toolResult")


# ---------------------------------------------------------------------------
# Sync/async parity for toolResult
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bedrock_converse_sync_async_parity_with_duplicates():
    """Sync and async paths should produce identical results with duplicate
    tool results."""
    messages = _make_duplicate_tool_result_messages()

    sync_result = _bedrock_converse_messages_pt(messages, MODEL, PROVIDER)
    async_result = await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
        messages, MODEL, PROVIDER
    )

    assert sync_result == async_result


# ---------------------------------------------------------------------------
# Empty content filtering tests
# ---------------------------------------------------------------------------


def test_bedrock_converse_filters_empty_assistant_content():
    """Verify that empty assistant content blocks are filtered out to avoid
    Bedrock API errors about blank text fields."""
    messages = [
        {"role": "user", "content": "Say hello"},
        {"role": "assistant", "content": "Hello"},
        {"role": "assistant", "content": " there"},
        {"role": "assistant", "content": "!"},
        {"role": "assistant", "content": ""},  # Empty content
        {"role": "assistant", "content": ""},  # Empty content
        {"role": "user", "content": "How are you?"},
    ]

    result = _bedrock_converse_messages_pt(messages, MODEL, PROVIDER)

    # Should have 3 messages: user, assistant (with merged non-empty content), user
    assert len(result) == 3
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"
    assert result[2]["role"] == "user"

    # Assistant message should only contain non-empty text blocks
    assistant_content = result[1]["content"]
    text_blocks = [block for block in assistant_content if "text" in block]
    assert len(text_blocks) == 3  # "Hello", " there", "!"
    assert text_blocks[0]["text"] == "Hello"
    assert text_blocks[1]["text"] == " there"
    assert text_blocks[2]["text"] == "!"


@pytest.mark.asyncio
async def test_bedrock_converse_filters_empty_assistant_content_async():
    """Verify that the async path also filters empty assistant content blocks."""
    messages = [
        {"role": "user", "content": "Say hello"},
        {"role": "assistant", "content": "Hello"},
        {"role": "assistant", "content": " there"},
        {"role": "assistant", "content": "!"},
        {"role": "assistant", "content": ""},  # Empty content
        {"role": "assistant", "content": ""},  # Empty content
        {"role": "user", "content": "How are you?"},
    ]

    result = await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
        messages, MODEL, PROVIDER
    )

    # Should have 3 messages: user, assistant (with merged non-empty content), user
    assert len(result) == 3
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"
    assert result[2]["role"] == "user"

    # Assistant message should only contain non-empty text blocks
    assistant_content = result[1]["content"]
    text_blocks = [block for block in assistant_content if "text" in block]
    assert len(text_blocks) == 3  # "Hello", " there", "!"
    assert text_blocks[0]["text"] == "Hello"
    assert text_blocks[1]["text"] == " there"
    assert text_blocks[2]["text"] == "!"


def test_bedrock_converse_filters_whitespace_only_content():
    """Verify that whitespace-only content is also filtered out."""
    messages = [
        {"role": "user", "content": "Test"},
        {"role": "assistant", "content": "Response"},
        {"role": "assistant", "content": "   "},  # Whitespace only
        {"role": "assistant", "content": "\n\t"},  # Whitespace only
        {"role": "assistant", "content": ""},  # Empty
    ]

    result = _bedrock_converse_messages_pt(messages, MODEL, PROVIDER)

    # Should have 2 messages: user and assistant
    assert len(result) == 2
    assistant_content = result[1]["content"]
    text_blocks = [block for block in assistant_content if "text" in block]
    # Only "Response" should be present
    assert len(text_blocks) == 1
    assert text_blocks[0]["text"] == "Response"


def test_bedrock_converse_filters_empty_list_content():
    """Verify that empty text elements in list content are filtered out."""
    messages = [
        {"role": "user", "content": "Test"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": ""},  # Empty
                {"type": "text", "text": "World"},
                {"type": "text", "text": "   "},  # Whitespace only
            ],
        },
    ]

    result = _bedrock_converse_messages_pt(messages, MODEL, PROVIDER)

    # Should have 2 messages: user and assistant
    assert len(result) == 2
    assistant_content = result[1]["content"]
    text_blocks = [block for block in assistant_content if "text" in block]
    # Only "Hello" and "World" should be present
    assert len(text_blocks) == 2
    assert text_blocks[0]["text"] == "Hello"
    assert text_blocks[1]["text"] == "World"
