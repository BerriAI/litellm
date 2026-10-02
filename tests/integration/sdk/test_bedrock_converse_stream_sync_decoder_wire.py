import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final
from urllib.parse import unquote

import pytest
from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

import litellm
from litellm.exceptions import BadGatewayError, BadRequestError, MidStreamFallbackError
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

_MODEL_ID: Final = "global.moonshotai.kimi-k3"
_CONVERSE_MODEL: Final = f"bedrock/converse/{_MODEL_ID}"
_STREAM_TARGET: Final = f"/model/{_MODEL_ID}/converse-stream"
_EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
_ANSWER: Final = "bedrock sync decoder control"
_REJECTION: Final = "structured output schema uses unsupported regex negative look-ahead"
_UNKNOWN_TYPE: Final = "somethingBedrockAddedLater"
_USAGE: Final[dict[str, JsonValue]] = {"usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15}}


def _frame(event_type: str, payload: Mapping[str, JsonValue]) -> bytes:
    return _aws_event_frame(event_type, payload, "sc", "u")


_NORMAL: Final = b"".join(
    (
        _frame("messageStart", {"role": "assistant"}),
        _frame("contentBlockDelta", {"delta": {"text": _ANSWER}, "contentBlockIndex": 0}),
        _frame("contentBlockStop", {"contentBlockIndex": 0}),
        _frame("messageStop", {"stopReason": "end_turn"}),
        _frame("metadata", _USAGE),
    )
)
_VALIDATION_FRAME: Final = _frame("validationException", {"message": _REJECTION})
_UNKNOWN_FRAME: Final = _frame(_UNKNOWN_TYPE, {"future": True})


@dataclass(frozen=True, slots=True)
class _Consumed:
    text: str
    finish_reasons: tuple[str | None, ...]


def _peer(frames: bytes) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert unquote(request.target) == _STREAM_TARGET, request.target
        return Reply(body=frames, content_type=_EVENT_STREAM)

    return respond


def _consume_sync_stream(wire: Wire) -> _Consumed:
    response: Final = litellm.completion(
        model=_CONVERSE_MODEL,
        messages=[{"role": "user", "content": "What does the sync decoder do with this stream?"}],
        max_tokens=16,
        stream=True,
        api_base=wire.url,
        aws_access_key_id="AKIASCRIPTEDPROVIDER",
        aws_secret_access_key="scripted-secret",
        aws_region_name="us-east-1",
        num_retries=0,
    )
    assert isinstance(response, CustomStreamWrapper), type(response)
    chunks: Final = tuple(response)
    return _Consumed(
        text="".join(str(chunk.choices[0].delta.content or "") for chunk in chunks),
        finish_reasons=tuple(chunk.choices[0].finish_reason for chunk in chunks),
    )


def test_k01_sync_stream_with_a_validation_exception_frame_first_raises_a_400() -> None:
    with wire_server(_peer(_VALIDATION_FRAME)) as wire:
        with pytest.raises(BadRequestError, match=re.escape(_REJECTION)) as raised:
            _consume_sync_stream(wire)
        assert raised.value.status_code == 400, raised.value
        assert len(wire.drain()) == 1


def test_k02_sync_stream_whose_only_frame_has_an_unknown_event_type_raises_a_502_naming_it() -> None:
    with wire_server(_peer(_UNKNOWN_FRAME)) as wire:
        with pytest.raises(MidStreamFallbackError, match=re.escape(_UNKNOWN_TYPE)) as raised:
            _consume_sync_stream(wire)
        assert raised.value.status_code == 502, raised.value
        assert isinstance(raised.value.original_exception, BadGatewayError), raised.value.original_exception
        assert "none of its 1 events carried a known event type" in str(raised.value), raised.value
        assert len(wire.drain()) == 1


def test_k03_sync_stream_with_normal_frames_delivers_the_text_and_a_stop() -> None:
    with wire_server(_peer(_NORMAL)) as wire:
        consumed: Final = _consume_sync_stream(wire)
        assert consumed.text == _ANSWER, consumed
        assert consumed.finish_reasons[-1] == "stop", consumed
        assert len(wire.drain()) == 1
