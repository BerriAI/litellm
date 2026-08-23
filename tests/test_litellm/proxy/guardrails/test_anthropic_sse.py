"""
Unit tests for litellm/proxy/guardrails/anthropic_sse.py
"""

import litellm
from litellm.proxy.guardrails.anthropic_sse import is_raw_sse_stream


def _chunk() -> litellm.ModelResponseStream:
    return litellm.ModelResponseStream(
        id="tid",
        choices=[
            litellm.types.utils.StreamingChoices(
                delta=litellm.types.utils.Delta(content="Hi", role="assistant"),
                finish_reason=None,
                index=0,
            )
        ],
        created=1,
        model="gpt-4o-mini",
        object="chat.completion.chunk",
    )


def test_is_raw_sse_stream_all_str_or_bytes():
    assert is_raw_sse_stream([b"event: message_start\ndata: {}"]) is True
    assert is_raw_sse_stream(["event: message_start", b"data: {}"]) is True


def test_is_raw_sse_stream_parsed_chunks():
    assert is_raw_sse_stream([_chunk(), _chunk()]) is False
    assert is_raw_sse_stream([]) is False


def test_is_raw_sse_stream_mixed_stream_is_not_sse():
    # #37873: a provider yielding one stray bytes chunk among parsed chunks
    # must classify as a normal (non-SSE) stream so it reaches
    # stream_chunk_builder, not the fail-closed Anthropic-SSE path.
    assert is_raw_sse_stream([_chunk(), b"stray bytes"]) is False
