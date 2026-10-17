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

    return [frame async for frame in BedrockPassthroughGuardrailHandler.event_stream_items(chunks())]


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
