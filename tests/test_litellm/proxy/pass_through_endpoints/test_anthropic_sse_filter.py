"""Unit tests for AnthropicSSENoiseFilter."""

from litellm.proxy.pass_through_endpoints.anthropic_sse_filter import (
    AnthropicSSENoiseFilter,
)


def _feed_all(chunks):
    f = AnthropicSSENoiseFilter()
    out = b"".join(f.feed(c) for c in chunks)
    return out + f.flush()


def test_passes_clean_anthropic_stream_unchanged():
    stream = (
        b'event: message_start\ndata: {"type":"message_start","message":{"id":"m_1"}}\n\n'
        b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"hi"}}\n\n'
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )
    assert _feed_all([stream]) == stream


def test_drops_openrouter_processing_comment():
    valid = b'event: message_start\ndata: {"type":"message_start"}\n\n'
    stream = b": OPENROUTER PROCESSING\n\n" + valid
    assert _feed_all([stream]) == valid


def test_drops_done_terminator():
    stop = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    done = b"event: data\ndata: [DONE]\n\n"
    assert _feed_all([stop + done]) == stop


def test_drops_empty_data_payload():
    valid = b'event: message_start\ndata: {"type":"message_start"}\n\n'
    stream = b"data: \n\n" + valid
    assert _feed_all([stream]) == valid


def test_drops_data_comment_payload():
    valid = b'event: message_start\ndata: {"type":"message_start"}\n\n'
    stream = b"data: : OPENROUTER PROCESSING\n\n" + valid
    assert _feed_all([stream]) == valid


def test_preserves_ping_events():
    ping = b'event: ping\ndata: {"type":"ping"}\n\n'
    assert _feed_all([ping]) == ping


def test_buffers_across_chunk_boundary_mid_event():
    event = b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"hello world"}}\n\n'
    mid = len(event) // 2
    assert _feed_all([event[:mid], event[mid:]]) == event


def test_buffers_across_chunk_boundary_mid_separator():
    e1 = b'event: message_start\ndata: {"type":"message_start"}\n\n'
    e2 = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    stream = e1 + e2
    # split exactly between the two \n of e1's terminator
    split = len(e1) - 1
    assert _feed_all([stream[:split], stream[split:]]) == stream


def test_drops_noise_between_real_events():
    start = b'event: message_start\ndata: {"type":"message_start"}\n\n'
    delta = b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"hi"}}\n\n'
    stop = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    stream = (
        start
        + b": OPENROUTER PROCESSING\n\n"
        + delta
        + b"data: : keep-alive\n\n"
        + b"event: data\ndata: [DONE]\n\n"
        + stop
    )
    assert _feed_all([stream]) == start + delta + stop


def test_handles_crlf_separators():
    stream = (
        b'event: message_start\r\ndata: {"type":"message_start"}\r\n\r\n'
        b'event: message_stop\r\ndata: {"type":"message_stop"}\r\n\r\n'
    )
    assert _feed_all([stream]) == stream


def test_flush_emits_trailing_event_without_separator():
    event = b'event: message_stop\ndata: {"type":"message_stop"}'
    assert _feed_all([event]) == event


def test_flush_drops_trailing_noise():
    assert _feed_all([b": OPENROUTER PROCESSING"]) == b""


def test_partial_utf8_at_chunk_boundary():
    # 'é' is two bytes in UTF-8: \xc3\xa9
    event = 'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"héllo"}}\n\n'.encode(
        "utf-8"
    )
    e_idx = event.index(b"\xc3\xa9")
    # Split between the two bytes of 'é'
    a, b = event[: e_idx + 1], event[e_idx + 1 :]
    assert _feed_all([a, b]) == event
