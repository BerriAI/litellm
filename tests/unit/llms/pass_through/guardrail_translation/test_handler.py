import json
from typing import Final

import pytest

from litellm.constants import SERVER_STREAMING_CLASSIFICATION_KEY, SERVER_STREAMING_CLASSIFICATION_MARKER
from litellm.llms.pass_through.guardrail_translation.handler import PassThroughEndpointHandler


def test_full_payload_guardrail_text_excludes_the_server_streaming_marker():
    body: Final = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}

    text: Final = PassThroughEndpointHandler()._extract_text_for_guardrail(
        {**body, SERVER_STREAMING_CLASSIFICATION_KEY: SERVER_STREAMING_CLASSIFICATION_MARKER},
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


def _invoke_frame(event: dict) -> bytes:
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
def test_passthrough_route_relays_the_streams_its_provider_translation_scans(
    provider: str, endpoint: str | None, model: str | None, body: object, relayable: bool
):
    from litellm.llms.pass_through.guardrail_translation.handler import LlmPassthroughRouteHandler

    assert LlmPassthroughRouteHandler.supports_event_stream_relay(provider, endpoint, model, body) is relayable


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
