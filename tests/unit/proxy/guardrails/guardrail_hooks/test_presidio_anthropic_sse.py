"""Tests for stateful Presidio unmasking of Anthropic /v1/messages SSE streams."""

import json

import pytest

from litellm.proxy.guardrails.guardrail_hooks.presidio import _OPTIONAL_PresidioPIIMasking
from litellm.proxy.guardrails.guardrail_hooks.presidio_anthropic_sse import AnthropicSSEUnmasker

TOKENS = {"<PERSON_1>": "Maria Gonzalez", "<EMAIL_ADDRESS_2>": "maria@example.com"}


def _event(payload: dict) -> bytes:
    return f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n".encode()


def _text_delta(text: str, index: int = 0) -> bytes:
    return _event({"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}})


def _json_delta(partial: str, index: int = 1) -> bytes:
    return _event(
        {"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": partial}}
    )


def _stop(index: int = 0) -> bytes:
    return _event({"type": "content_block_stop", "index": index})


def _run(chunks: list[bytes], tokens: dict[str, str] = TOKENS) -> bytes:
    unmasker = AnthropicSSEUnmasker(tokens)
    return b"".join(unmasker.feed(c) for c in chunks) + unmasker.flush()


def _deltas(raw: bytes, field: str = "text", index: int | None = None) -> str:
    """Concatenate the given delta field across every event in an SSE stream."""
    out = []
    for line in raw.decode().splitlines():
        if not line.startswith("data: "):
            continue
        event = json.loads(line[6:])
        if event.get("type") == "content_block_delta" and (index is None or event["index"] == index):
            out.append(event["delta"].get(field, ""))
    return "".join(out)


def test_whole_token_in_one_delta():
    raw = _run([_text_delta("Hello <PERSON_1>!"), _stop()])
    assert _deltas(raw) == "Hello Maria Gonzalez!"


def test_token_split_across_deltas():
    # Observed live from claude-sonnet-4-6: " and `<EMAIL_ADDRESS_" + "2>` but"
    raw = _run([_text_delta("Hi <PER"), _text_delta("SON_1>, mail <EMAIL_ADDRESS_"), _text_delta("2> now"), _stop()])
    assert _deltas(raw) == "Hi Maria Gonzalez, mail maria@example.com now"
    assert "<" not in _deltas(raw)


def test_token_split_one_char_per_delta():
    token = "<EMAIL_ADDRESS_2>"
    raw = _run([_text_delta("x ")] + [_text_delta(c) for c in token] + [_text_delta(" y"), _stop()])
    assert _deltas(raw) == "x maria@example.com y"


def test_held_text_is_never_emitted_early():
    unmasker = AnthropicSSEUnmasker(TOKENS)
    first = unmasker.feed(_text_delta("Dear <PERSON_"))
    assert _deltas(first) == "Dear "
    assert "PERSON" not in first.decode()


def test_lone_angle_bracket_that_is_not_a_token_is_released():
    raw = _run([_text_delta("if a <"), _text_delta(" b then"), _stop()])
    assert _deltas(raw) == "if a < b then"


def test_unknown_tag_is_not_held():
    unmasker = AnthropicSSEUnmasker(TOKENS)
    assert _deltas(unmasker.feed(_text_delta("use <div"))) == "use <div"


def test_held_text_flushed_before_block_stop():
    raw = _run([_text_delta("cut <PE"), _stop()])
    lines = [line for line in raw.decode().splitlines() if line.startswith("data: ")]
    types = [json.loads(line[6:])["type"] for line in lines]
    assert types == ["content_block_delta", "content_block_delta", "content_block_stop"]
    # Too short to be unambiguous, so the prefix is released as-is.
    assert _deltas(raw) == "cut <PE"


def test_truncated_token_at_block_end_restores_value():
    # max_tokens cut the token off; long enough prefix -> restore, matching
    # _OPTIONAL_PresidioPIIMasking._unmask_pii_text's truncation fallback.
    raw = _run([_text_delta("mail <EMAIL_ADDRESS"), _stop()])
    assert _deltas(raw) == "mail maria@example.com"


def test_held_text_flushed_at_end_of_stream_without_stop():
    raw = _run([_text_delta("x <PE")])
    assert _deltas(raw) == "x <PE"


def test_message_delta_flushes_all_blocks():
    unmasker = AnthropicSSEUnmasker(TOKENS)
    unmasker.feed(_text_delta("a <PER", index=0))
    out = unmasker.feed(_event({"type": "message_delta", "delta": {"stop_reason": "end_turn"}}))
    assert _deltas(out) == "<PER"


def test_blocks_are_tracked_independently():
    raw = _run(
        [
            _text_delta("A <PER", index=0),
            _text_delta("B <EMAIL_", index=2),
            _text_delta("SON_1>", index=0),
            _text_delta("ADDRESS_2>", index=2),
            _stop(0),
            _stop(2),
        ]
    )
    assert _deltas(raw, index=0) == "A Maria Gonzalez"
    assert _deltas(raw, index=2) == "B maria@example.com"


def test_input_json_delta_split_token_is_json_escaped():
    tokens = {"<PERSON_1>": 'Bob "B" \\ Smith'}
    raw = _run([_json_delta('{"to": "<PER'), _json_delta('SON_1>"}'), _stop(1)], tokens)
    assert json.loads(_deltas(raw, field="partial_json")) == {"to": 'Bob "B" \\ Smith'}


def test_thinking_delta_untouched():
    chunk = _event(
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "<PERSON_1>"}}
    )
    assert _run([chunk]) == chunk


def test_unchanged_events_pass_through_byte_for_byte():
    chunks = [
        _event({"type": "message_start", "message": {"id": "msg_1"}}),
        b'event: ping\ndata: {"type": "ping"}\n\n',
        _text_delta("no pii here"),
        _stop(),
        _event({"type": "message_stop"}),
    ]
    assert _run(chunks) == b"".join(chunks)


def test_event_split_across_byte_chunks():
    whole = _text_delta("Hello <PERSON_1>") + _stop()
    pieces = [whole[i : i + 7] for i in range(0, len(whole), 7)]
    assert _deltas(_run(pieces)) == "Hello Maria Gonzalez"


def test_multibyte_character_split_across_byte_chunks():
    tokens = {"<PERSON_1>": "José"}
    event = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "¡Hola <PERSON_1>! ñ"}}
    whole = f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode() + _stop()
    cut = whole.index("ñ".encode()) + 1  # split inside the 2-byte ñ
    raw = _run([whole[:cut], whole[cut:]], tokens)
    assert _deltas(raw) == "¡Hola José! ñ"
    assert "Jos\\u" not in raw.decode()


def test_crlf_framing():
    event = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi <PERSON_1>"}}
    chunk = ("event: content_block_delta\r\ndata: " + json.dumps(event) + "\r\n\r\n").encode()
    raw = _run([chunk])
    assert raw.endswith(b"\r\n\r\n")
    assert json.loads(raw.decode().split("data: ", 1)[1].strip())["delta"]["text"] == "Hi Maria Gonzalez"


def test_malformed_and_non_sse_bytes_pass_through():
    chunks = [b"data: {not json}\n\n", b"\xff\xfe raw tail"]
    assert _run(chunks) == b"".join(chunks)


@pytest.mark.asyncio
async def test_stream_pii_unmasking_handles_split_tokens():
    guardrail = _OPTIONAL_PresidioPIIMasking(mock_testing=True, output_parse_pii=True)
    request_data = {"metadata": {"pii_tokens": dict(TOKENS)}}

    async def stream():
        yield _text_delta("Dear <PERSON")
        yield _text_delta("_1>, see <EMAIL_ADDRESS_2")
        yield _text_delta(">.")
        yield _stop()

    out = b"".join([c async for c in guardrail._stream_pii_unmasking(stream(), request_data)])
    assert _deltas(out) == "Dear Maria Gonzalez, see maria@example.com."


@pytest.mark.asyncio
async def test_stream_pii_unmasking_flushes_when_stream_ends_mid_token():
    guardrail = _OPTIONAL_PresidioPIIMasking(mock_testing=True, output_parse_pii=True)
    request_data = {"metadata": {"pii_tokens": dict(TOKENS)}}

    async def stream():
        yield _text_delta("x <PE")

    out = b"".join([c async for c in guardrail._stream_pii_unmasking(stream(), request_data)])
    assert _deltas(out) == "x <PE"


def test_unmask_output_callback_runs_native_streaming_hook(monkeypatch):
    """ProxyLogging only runs a guardrail's own streaming hook on /v1/messages when
    mask_response_content is set; otherwise unified_guardrail drops the unmask."""
    import litellm
    from litellm.proxy.guardrails.guardrail_initializers import initialize_presidio
    from litellm.types.guardrails import LitellmParams

    monkeypatch.setattr(litellm, "callbacks", [])
    params = LitellmParams(
        guardrail="presidio",
        mode="pre_call",
        presidio_filter_scope="input",
        output_parse_pii=True,
        presidio_analyzer_api_base="http://analyzer",
        presidio_anonymizer_api_base="http://anonymizer",
    )
    input_cb, unmask_cb = initialize_presidio(params, {"guardrail_name": "g"})
    assert unmask_cb.mask_response_content is True
    assert input_cb.mask_response_content is False
