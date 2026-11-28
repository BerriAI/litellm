"""Very large events, very many tiny events, and multibyte text cut at transfer boundaries, per provider.

The proxy reads provider bytes in whatever transfer chunks the network delivers and must reassemble whole
SSE events (or AWS event stream frames) before parsing them, and it must not split a UTF-8 sequence when it
re-emits text. The Rust work coalesces SSE frames and relays native `/v1/messages` bytes directly, which is
exactly where a 64 KiB event, a one byte transfer chunk or a four byte emoji cut in half can break.
OpenAI chat already has a 1 MB delta, 2000 tiny deltas and a 3 byte cut test (`test_dec1_stream_chat_chunk_shape.py`);
this file covers the other providers and the other two routes.

The contract: every byte of text the provider sent reaches the caller exactly once, in order, with no
U+FFFD replacement character, whatever the event size or transfer cut. Whether the proxy forwards one
frame per provider event or coalesces adjacent events is not pinned for the translated chat path (only the
text is); for the native byte relay the event count is pinned, since coalescing there would change what
Anthropic SDKs see.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration.streaming.dec1_stream2_support import (
    ANTHROPIC,
    AZURE,
    BEDROCK_CONVERSE,
    BEDROCK_INVOKE,
    CHAT_PROVIDERS,
    GEMINI_AI_STUDIO,
    KIB,
    NATIVE_MESSAGES_PROVIDERS,
    OPENAI_RESPONSES,
    Provider,
    cut_every,
    deployed,
    provider_ids,
    streaming,
)
from integration.streaming.dec1_stream_support import chunks_of, content_of, events_of, marker, payloads

TRICKY_TEXT: Final = ("Zoë ", "雪だるま ", "naïve ", "𝄞𠜎 क्‍ष ", "é ", "Ωmega")
"""Two, three and four byte UTF-8 (a musical symbol and a CJK extension B ideograph), a Devanagari conjunct
with a zero width joiner, and a combining accent."""


def _messages_text(text: str) -> str:
    return "".join(
        str(object_value(body.get("delta") or {}).get("text") or "")
        for name, body in events_of(text)
        if name == "content_block_delta"
    )


def _responses_text(text: str) -> str:
    return "".join(
        str(item.get("delta") or "") for item in chunks_of(text) if item["type"] == "response.output_text.delta"
    )


def _big_pieces(size: int, count: int) -> tuple[str, ...]:
    return tuple(f"{index:08d}" + "y" * (size - 8) for index in range(count))


@pytest.mark.parametrize("provider", [ANTHROPIC, AZURE, GEMINI_AI_STUDIO, BEDROCK_CONVERSE], ids=provider_ids)
def test_chat_delta_of_96_kib_from_each_provider_arrives_whole(gateway: Gateway, provider: Provider) -> None:
    """Anthropic, Azure, Gemini or Bedrock Converse streams three deltas of 96 KiB each (larger than a 64 KiB
    read buffer). The chat caller's concatenated content equals the three pieces joined, byte for byte, and
    the stream ends with `[DONE]`."""
    pieces: Final = _big_pieces(96 * KIB, 3)
    with deployed(
        gateway, provider, streaming(provider.frames(f"dec1s2-{marker()}", pieces), provider.content_type)
    ) as rig:
        response: Final = rig.chat()
        assert content_of(chunks_of(response.text)) == "".join(pieces)
        assert payloads(response.text)[-1] == "[DONE]"


@pytest.mark.parametrize("provider", NATIVE_MESSAGES_PROVIDERS, ids=provider_ids)
def test_messages_delta_of_256_kib_is_relayed_whole(gateway: Gateway, provider: Provider) -> None:
    """A native `/v1/messages` provider sends one text delta of 256 KiB. The caller receives exactly one
    `content_block_delta` whose text is that whole string, and the stream ends with `message_stop`. A
    relay that splits a long event across two SSE frames would make the Anthropic SDK drop half."""
    piece: Final = _big_pieces(256 * KIB, 1)
    with deployed(
        gateway, provider, streaming(provider.frames(f"msg_{marker()}", piece), provider.content_type)
    ) as rig:
        events: Final = events_of(rig.messages().text)
        deltas: Final = [body for name, body in events if name == "content_block_delta"]
        assert len(deltas) == 1
        assert object_value(deltas[0]["delta"])["text"] == piece[0]
        assert events[-1][0] == "message_stop"


def test_responses_delta_of_256_kib_is_relayed_whole(gateway: Gateway) -> None:
    """A native OpenAI Responses provider sends one output_text delta of 256 KiB. The `/v1/responses` caller
    receives exactly one `response.output_text.delta` with that whole string, then `response.completed`."""
    piece: Final = _big_pieces(256 * KIB, 1)
    with deployed(gateway, OPENAI_RESPONSES, streaming(OPENAI_RESPONSES.frames(f"resp_{marker()}", piece))) as rig:
        events: Final = chunks_of(rig.responses().text)
        deltas: Final = [item for item in events if item["type"] == "response.output_text.delta"]
        assert [item["delta"] for item in deltas] == [piece[0]]
        assert events[-1]["type"] == "response.completed"


@pytest.mark.parametrize("provider", NATIVE_MESSAGES_PROVIDERS, ids=provider_ids)
def test_messages_two_thousand_tiny_deltas_are_relayed_one_event_each(gateway: Gateway, provider: Provider) -> None:
    """A native `/v1/messages` provider streams 2000 one character deltas. The caller receives exactly 2000
    `content_block_delta` events in order (text `0123456789` repeated), one `message_stop`, and nothing
    merged: the relay forwards events, it does not coalesce them. UNVERIFIED for the Rust SSE coalescing work
    if it lands on this route; then this test documents the change."""
    pieces: Final = tuple(str(index % 10) for index in range(2000))
    with deployed(
        gateway, provider, streaming(provider.frames(f"msg_{marker()}", pieces), provider.content_type)
    ) as rig:
        text: Final = rig.messages().text
        deltas: Final = [body for name, body in events_of(text) if name == "content_block_delta"]
        assert len(deltas) == 2000
        assert _messages_text(text) == "".join(pieces)
        assert [name for name, _ in events_of(text)].count("message_stop") == 1


def test_responses_two_thousand_tiny_deltas_all_arrive_in_order(gateway: Gateway) -> None:
    """A native OpenAI Responses provider streams 2000 one character deltas. The caller receives the same
    text in order, and the `sequence_number` of every event the caller sees strictly increases."""
    pieces: Final = tuple(str(index % 10) for index in range(2000))
    with deployed(gateway, OPENAI_RESPONSES, streaming(OPENAI_RESPONSES.frames(f"resp_{marker()}", pieces))) as rig:
        text: Final = rig.responses().text
        assert _responses_text(text) == "".join(pieces)
        numbers: Final = [int(str(item["sequence_number"])) for item in chunks_of(text)]
        assert numbers == sorted(set(numbers))


@pytest.mark.parametrize("provider", [ANTHROPIC, GEMINI_AI_STUDIO, BEDROCK_CONVERSE], ids=provider_ids)
def test_chat_many_tiny_provider_events_lose_no_text(gateway: Gateway, provider: Provider) -> None:
    """Anthropic, Gemini or Bedrock Converse streams 1000 one character deltas. The chat caller's
    concatenated content is exactly the 1000 characters in order, with one finish and one `[DONE]`; whether
    the proxy coalesces them is not pinned here."""
    pieces: Final = tuple(chr(ord("a") + index % 26) for index in range(1000))
    with deployed(
        gateway, provider, streaming(provider.frames(f"dec1s2-{marker()}", pieces), provider.content_type)
    ) as rig:
        response: Final = rig.chat()
        assert content_of(chunks_of(response.text)) == "".join(pieces)
        assert payloads(response.text).count("[DONE]") == 1


@pytest.mark.parametrize("provider", CHAT_PROVIDERS, ids=provider_ids)
@pytest.mark.parametrize("cut", [1, 5], ids=["one_byte", "five_bytes"])
def test_chat_multibyte_text_cut_at_any_transfer_boundary_arrives_intact(
    gateway: Gateway, provider: Provider, cut: int
) -> None:
    """Every big provider streams accented, CJK, four byte, ZWJ and combining text, and the peer writes its body
    in transfer chunks of 1 or 5 bytes, so cuts land inside UTF-8 sequences, inside JSON escapes and (for
    Bedrock) inside the binary frame prelude. The chat caller's content is exactly
    `TRICKY_TEXT` joined with no U+FFFD anywhere in the body."""
    frames: Final = cut_every(provider.frames(f"dec1s2-{marker()}", TRICKY_TEXT), cut)
    with deployed(gateway, provider, streaming(frames, provider.content_type)) as rig:
        response: Final = rig.chat()
        assert content_of(chunks_of(response.text)) == "".join(TRICKY_TEXT)
        assert "�" not in response.text


@pytest.mark.parametrize("provider", NATIVE_MESSAGES_PROVIDERS, ids=provider_ids)
def test_messages_multibyte_text_cut_every_byte_arrives_intact(gateway: Gateway, provider: Provider) -> None:
    """A native `/v1/messages` provider's body is written one byte per transfer chunk. The caller's
    concatenated `text_delta` text is exactly the tricky multibyte string with no U+FFFD. A byte relay that
    decodes each read as UTF-8 on its own (Python's `decode("utf-8", errors="replace")` on a partial read)
    would corrupt exactly these characters."""
    frames: Final = cut_every(provider.frames(f"msg_{marker()}", TRICKY_TEXT), 1)
    with deployed(gateway, provider, streaming(frames, provider.content_type)) as rig:
        text: Final = rig.messages().text
        assert _messages_text(text) == "".join(TRICKY_TEXT)
        assert "�" not in text


def test_responses_multibyte_text_cut_every_byte_arrives_intact(gateway: Gateway) -> None:
    """A native OpenAI Responses body is written one byte per transfer chunk. The `/v1/responses` caller's
    concatenated deltas are exactly the tricky multibyte string with no U+FFFD."""
    frames: Final = cut_every(OPENAI_RESPONSES.frames(f"resp_{marker()}", TRICKY_TEXT), 1)
    with deployed(gateway, OPENAI_RESPONSES, streaming(frames)) as rig:
        text: Final = rig.responses().text
        assert _responses_text(text) == "".join(TRICKY_TEXT)
        assert "�" not in text


def test_bedrock_invoke_frame_split_inside_its_prelude_is_reassembled(gateway: Gateway) -> None:
    """A Bedrock Invoke `/v1/messages` stream is cut every 3 bytes, so the 12 byte prelude (total length,
    header length, CRC) of every frame is split. The caller still receives every event, the text intact and
    `message_stop` last: the event stream decoder must buffer until a whole frame is present."""
    frames: Final = cut_every(BEDROCK_INVOKE.frames(f"msg_{marker()}", ("split ", "prelude")), 3)
    with deployed(gateway, BEDROCK_INVOKE, streaming(frames, BEDROCK_INVOKE.content_type)) as rig:
        text: Final = rig.messages().text
        assert _messages_text(text) == "split prelude"
        assert events_of(text)[-1][0] == "message_stop"


def test_whole_anthropic_stream_in_one_transfer_chunk_is_split_into_events(gateway: Gateway) -> None:
    """The Anthropic provider writes its entire `/v1/messages` stream (eight events) in one TCP write. The
    caller receives the same eight events in order, each parseable on its own."""
    frames: Final = ANTHROPIC.frames(f"msg_{marker()}", ("all ", "at ", "once"))
    with deployed(gateway, ANTHROPIC, streaming((b"".join(frames),))) as rig:
        names: Final = [name for name, _ in events_of(rig.messages().text)]
        assert names == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]


def test_chat_delta_of_escaped_json_text_survives_byte_cuts(gateway: Gateway) -> None:
    """An Anthropic deployment streams text full of characters JSON must escape (quotes, backslashes,
    newlines, tabs, a NUL written as \\u0000) and the body is cut every 2 bytes. The chat caller's content
    equals the original string exactly."""
    awkward: Final = ('say "hi"\\n', "tab\there\n", "nul\u0000end")
    frames: Final = cut_every(ANTHROPIC.frames(f"msg_{marker()}", awkward), 2)
    with deployed(gateway, ANTHROPIC, streaming(frames)) as rig:
        assert content_of(chunks_of(rig.chat().text)) == "".join(awkward)
