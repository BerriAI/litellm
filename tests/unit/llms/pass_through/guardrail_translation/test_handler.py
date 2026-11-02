import json
from typing import Final, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest
from typing_extensions import ReadOnly, TypedDict

from litellm.constants import SERVER_STREAMING_CLASSIFICATION_KEY, SERVER_STREAMING_CLASSIFICATION_MARKER
from litellm.llms.pass_through.guardrail_translation.handler import PassThroughEndpointHandler


def test_full_payload_guardrail_text_excludes_the_server_streaming_marker():
    body: Final = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}

    text: Final = PassThroughEndpointHandler()._extract_text_for_guardrail(
        {**body, SERVER_STREAMING_CLASSIFICATION_KEY: SERVER_STREAMING_CLASSIFICATION_MARKER},
        None,
    )

    assert json.loads(text) == body, text


def test_full_payload_guardrail_text_excludes_the_injected_litellm_metadata():
    body: Final = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "guardrails": ["g"]}
    litellm_metadata: Final = {
        "user_api_key_token": "hashed-key",
        "user_api_key_user_email": "someone@example.com",
        "user_api_key_user_role": "proxy_admin",
    }

    text: Final = PassThroughEndpointHandler()._extract_text_for_guardrail(
        {**body, "litellm_metadata": litellm_metadata},
        None,
    )

    assert json.loads(text) == body, text


@pytest.mark.parametrize(
    "marker",
    [
        SERVER_STREAMING_CLASSIFICATION_MARKER,
        json.loads(json.dumps(SERVER_STREAMING_CLASSIFICATION_MARKER)),
    ],
    ids=["enum", "json-string"],
)
def test_full_payload_guardrail_text_scans_caller_value_but_not_the_marker(marker: str):
    text: Final = PassThroughEndpointHandler()._extract_text_for_guardrail(
        {
            "model": "m",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
            SERVER_STREAMING_CLASSIFICATION_KEY: "BLOCKME caller content",
        },
        None,
    )

    assert "BLOCKME caller content" in text, text

    marker_text: Final = PassThroughEndpointHandler()._extract_text_for_guardrail(
        {
            "model": "m",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
            SERVER_STREAMING_CLASSIFICATION_KEY: marker,
        },
        None,
    )

    assert SERVER_STREAMING_CLASSIFICATION_KEY not in json.loads(marker_text), marker_text


def _converse_frame(event_type: str, payload: dict) -> bytes:
    import struct
    from binascii import crc32

    headers = b"".join(
        bytes((len(name),)) + name + b"\x07" + struct.pack("!H", len(value)) + value
        for name, value in (
            (b":event-type", event_type.encode()),
            (b":content-type", b"application/json"),
            (b":message-type", b"event"),
        )
    )
    body = json.dumps(payload).encode()
    prelude = struct.pack("!II", 16 + len(headers) + len(body), len(headers))
    message = prelude + struct.pack("!I", crc32(prelude)) + headers + body
    return message + struct.pack("!I", crc32(message))


async def _frames(*raw: bytes) -> list[object]:
    from litellm.llms.bedrock.passthrough.guardrail_translation.handler import BedrockPassthroughGuardrailHandler

    async def chunks():
        for chunk in raw:
            yield chunk

    return [frame async for frame in BedrockPassthroughGuardrailHandler.event_stream_items(chunks(), "converse-stream")]


@pytest.mark.asyncio
async def test_passthrough_route_scans_converse_frames_with_the_bedrock_translation():
    from litellm.llms.pass_through.guardrail_translation.handler import LlmPassthroughRouteHandler

    frames: Final = await _frames(
        _converse_frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "Hello "}}),
        _converse_frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "world"}}),
        _converse_frame("messageStop", {"stopReason": "end_turn"}),
    )

    scan_key: Final = LlmPassthroughRouteHandler().get_streaming_scan_key(frames)

    assert scan_key is not None and (scan_key.texts, scan_key.stream_ended) == (("Hello world",), True)
    assert LlmPassthroughRouteHandler()._check_streaming_has_ended(frames) is True
    assert LlmPassthroughRouteHandler()._check_streaming_has_ended(frames[:2]) is False


def test_passthrough_route_has_no_stream_translation_for_items_no_provider_owns():
    from litellm.llms.pass_through.guardrail_translation.handler import LlmPassthroughRouteHandler

    assert LlmPassthroughRouteHandler().get_streaming_scan_key([b"raw", {"chunk": 1}]) is None
    assert LlmPassthroughRouteHandler()._check_streaming_has_ended([b"raw"]) is False


@pytest.mark.parametrize(("provider", "ends_message"), [("bedrock", True), ("vertex_ai", False)])
def test_passthrough_route_block_before_any_frame_uses_the_request_provider(provider: str, ends_message: bool):
    from litellm.exceptions import ModifyResponseException
    from litellm.llms.pass_through.guardrail_translation.handler import LlmPassthroughRouteHandler

    chunks: Final = LlmPassthroughRouteHandler().build_block_sse_chunks(
        ModifyResponseException(message="blocked", model="m", request_data={"custom_llm_provider": provider}),
        stream_started=False,
        responses_so_far=[],
    )

    assert (chunks is not None) is ends_message


_INVOKE_STREAM: Final = "model/anthropic.claude-sonnet-5-v1:0/invoke-with-response-stream"


class _TextDelta(TypedDict):
    type: ReadOnly[Literal["text_delta"]]
    text: ReadOnly[str]


class _TextDeltaEvent(TypedDict):
    type: ReadOnly[Literal["content_block_delta"]]
    index: ReadOnly[int]
    delta: ReadOnly[_TextDelta]


def _invoke_frame(event: _TextDeltaEvent) -> bytes:
    import base64

    return _converse_frame("chunk", {"bytes": base64.b64encode(json.dumps(event).encode()).decode()})


@pytest.mark.parametrize(
    ("provider", "endpoint", "model", "body", "relayable"),
    [
        ("bedrock", "model/x/converse-stream", "meta.llama4-maverick-17b-instruct-v1:0", None, True),
        ("bedrock", _INVOKE_STREAM, "anthropic.claude-sonnet-5-v1:0", None, True),
        ("bedrock", _INVOKE_STREAM, "meta.llama4-maverick-17b-instruct-v1:0", {"prompt": "hi"}, False),
        ("bedrock", _INVOKE_STREAM, None, None, False),
        ("bedrock", _INVOKE_STREAM, None, {"anthropic_version": "bedrock-2023-05-31"}, True),
        ("bedrock", None, "anthropic.claude-sonnet-5-v1:0", None, False),
        ("vertex_ai", "model/x/converse-stream", "anthropic.claude-sonnet-5-v1:0", None, False),
    ],
)
def test_passthrough_route_relays_and_buffers_the_streams_its_provider_translation_scans(
    provider: str, endpoint: str | None, model: str | None, body: object, relayable: bool
):
    from litellm.llms.pass_through.guardrail_translation.handler import LlmPassthroughRouteHandler

    assert LlmPassthroughRouteHandler.supports_event_stream_relay(provider, endpoint, model, body) is relayable
    assert (
        LlmPassthroughRouteHandler.supports_event_stream_de_anonymization(provider, endpoint, model, body) is relayable
    )


@pytest.mark.asyncio
async def test_passthrough_route_frames_an_invoke_stream_for_the_bedrock_translation():
    from litellm.llms.pass_through.guardrail_translation.handler import LlmPassthroughRouteHandler

    async def chunks():
        yield _invoke_frame({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi "}})
        yield _invoke_frame({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "you"}})

    items: Final = LlmPassthroughRouteHandler.event_stream_items("bedrock", _INVOKE_STREAM, chunks())
    assert items is not None
    frames: Final = [frame async for frame in items]

    scan_key: Final = LlmPassthroughRouteHandler().get_streaming_scan_key(frames)

    assert scan_key is not None and scan_key.texts == ("Hi you",)
    assert LlmPassthroughRouteHandler()._check_streaming_has_ended(frames) is True
    assert LlmPassthroughRouteHandler()._check_streaming_has_ended(frames[:2]) is False


def test_passthrough_route_block_before_any_invoke_frame_is_a_whole_message_in_chunk_frames():
    import base64

    from botocore.eventstream import EventStreamBuffer

    from litellm.exceptions import ModifyResponseException
    from litellm.llms.pass_through.guardrail_translation.handler import LlmPassthroughRouteHandler

    chunks: Final = LlmPassthroughRouteHandler().build_block_sse_chunks(
        ModifyResponseException(
            message="blocked", model="m", request_data={"custom_llm_provider": "bedrock", "endpoint": _INVOKE_STREAM}
        ),
        stream_started=False,
        responses_so_far=[],
    )

    assert chunks is not None
    buffer: Final = EventStreamBuffer()
    buffer.add_data(b"".join(chunks))
    events: Final = [json.loads(base64.b64decode(json.loads(message.payload)["bytes"])) for message in buffer]
    assert [event["type"] for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[2]["delta"]["text"] == "blocked"


async def _scanned_response_texts(response: dict[str, object]) -> tuple[str, ...]:
    guardrail: Final = MagicMock(guardrail_name="post-call", apply_guardrail=AsyncMock())
    await PassThroughEndpointHandler().process_output_response(response, guardrail, request_data={})
    return tuple(text for call in guardrail.apply_guardrail.await_args_list for text in call.kwargs["inputs"]["texts"])


@pytest.mark.asyncio
async def test_post_call_scan_reads_the_reply_text_and_not_the_numbers_around_it():
    response: Final = {
        "id": "chatcmpl-1793600123",
        "object": "chat.completion",
        "created": 1793600123,
        "model": "gpt-5.6",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Paris."}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4111111111111111, "total_tokens": 5551234567},
    }

    assert await _scanned_response_texts(response) == ("Paris.",)


@pytest.mark.asyncio
async def test_post_call_scan_still_reads_every_string_a_reply_carries():
    response: Final = {
        "choices": [
            {"message": {"content": "call 555-123-4567", "tool_calls": [{"function": {"arguments": '{"a": "b"}'}}]}}
        ],
        "candidates": [{"content": {"parts": [{"text": "card 4111 1111 1111 1111"}]}}],
    }

    assert await _scanned_response_texts(response) == ('call 555-123-4567\n{"a": "b"}\ncard 4111 1111 1111 1111',)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"candidates": [{"content": {"parts": [{"functionCall": {"args": {"id": "alice@example.com"}}}]}}]},
        {"content": [{"input": {"id": "alice@example.com", "message": {"role": "alice@example.com"}}}]},
        {"choices": [{"message": {"tool_calls": [{"function": {"nested": {"type": "alice@example.com"}}}]}}]},
        {"id": "alice@example.com", "result": {"status": "alice@example.com"}},
    ],
)
async def test_post_call_scan_reads_a_field_named_like_the_envelope_in_what_the_model_or_api_wrote(response):
    assert all("alice@example.com" in text for text in await _scanned_response_texts(response))
    assert await _scanned_response_texts(response) != ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, expected",
    [
        ({"result": {"credit_card": 4111111111111111, "score": 0.4111111111111111, "ok": True}}, "4111111111111111"),
        (
            {"candidates": [{"content": {"parts": [{"functionCall": {"args": {"card": 4111111111111111}}}]}}]},
            "4111111111111111",
        ),
        ({"choices": [{"message": {"tool_calls": [{"function": {"arguments": {"n": 5551234567}}}]}}]}, "5551234567"),
    ],
)
async def test_post_call_scan_reads_an_integer_in_what_the_model_or_api_wrote(response, expected):
    assert any(expected in text for text in await _scanned_response_texts(response))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"results": {"alice@example.com": "active"}},
        {"candidates": [{"content": {"parts": [{"functionCall": {"args": {"alice@example.com": "active"}}}]}}]},
    ],
)
async def test_post_call_scan_reads_a_field_name_in_what_the_model_or_api_wrote(response):
    assert any("alice@example.com" in text for text in await _scanned_response_texts(response))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"created": 1793600123, "data": [{"url": "https://x.example/a.png"}]},
        {"id": "batch_1", "object": "batch", "created_at": 1793600123, "request_counts": {"total": 5551234567}},
        {"predictions": [{"embeddings": {"values": [0.0123456789]}, "total": 5551234567}]},
    ],
)
async def test_post_call_scan_leaves_out_the_numbers_of_a_provider_reply(response):
    assert all(
        "1793600123" not in text and "5551234567" not in text for text in await _scanned_response_texts(response)
    )


@pytest.mark.asyncio
async def test_post_call_scan_reads_only_the_text_of_an_anthropic_message():
    response: Final = {
        "id": "msg_5551234567",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": [
            {"type": "text", "text": "Paris."},
            {"type": "tool_use", "id": "toolu_5551234567", "name": "lookup", "input": {"card": 4111111111111111}},
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 5551234567, "output_tokens": 1},
    }

    assert await _scanned_response_texts(response) == ("Paris.\nlookup\ncard\n4111111111111111",)


@pytest.mark.asyncio
async def test_post_call_scan_skips_a_reply_with_no_text():
    response: Final = {"object": "list", "data": [{"object": "embedding", "embedding": [0.4111111111111111]}]}

    assert await _scanned_response_texts(response) == ()
