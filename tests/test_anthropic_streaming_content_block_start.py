import pytest
from litellm.llms.anthropic.chat.handler import ModelResponseIterator


def test_chunk_parser_content_block_start_omits_text():
    """
    Test that ModelResponseIterator.chunk_parser handles content_block_start
    when the 'text' key is omitted by third-party Anthropic-compatible upstreams.
    Fixes Issue #40689.
    """
    iterator = ModelResponseIterator(streaming_response=[], sync_stream=True)

    # 1. Chunk omitting "text" entirely
    chunk_without_text = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text"},
    }
    res = iterator.chunk_parser(chunk_without_text)
    assert res is not None
    assert res.choices[0].delta.content == ""

    # 2. Chunk with standard official empty string text
    chunk_with_empty_text = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    res = iterator.chunk_parser(chunk_with_empty_text)
    assert res is not None
    assert res.choices[0].delta.content == ""

    # 3. Chunk with text as None
    chunk_with_none_text = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": None},
    }
    res = iterator.chunk_parser(chunk_with_none_text)
    assert res is not None
    assert res.choices[0].delta.content == ""

    # 4. Chunk with actual initial text
    chunk_with_text = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": "hello"},
    }
    res = iterator.chunk_parser(chunk_with_text)
    assert res is not None
    assert res.choices[0].delta.content == "hello"
