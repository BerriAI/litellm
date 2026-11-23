"""
Azure AI Speech synthesis over the websocket protocol the Speech SDK speaks, served as OpenAI speech SSE events

The REST endpoint returns audio only. Visemes arrive only here, in `audio.metadata` messages, see
https://github.com/microsoft/cognitive-services-speech-sdk-js/blob/master/src/common.speech/SynthesisAdapterBase.ts
"""

import base64
import json
import ssl
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final, Protocol
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import realtime_ssl_for_url
from litellm.types.llms.openai import HttpxBinaryResponseContent

SSE_STREAM_FORMAT: Final = "sse"
_SSE_MEDIA_TYPE: Final = "text/event-stream"
_HEADER_LENGTH_BYTES: Final = 2
_TICKS_PER_MILLISECOND: Final = 10_000
_INVALID_PAYLOAD_CLOSE_CODE: Final = 1007


class _MetadataData(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    offset: int = Field(default=0, alias="Offset")
    viseme_id: int | None = Field(default=None, alias="VisemeId")


class _MetadataEntry(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    type: str = Field(alias="Type")
    data: _MetadataData = Field(alias="Data")


class _AudioMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    metadata: tuple[_MetadataEntry, ...] = Field(alias="Metadata")


@dataclass(frozen=True, slots=True)
class _Frame:
    path: str
    body: bytes


class _AsyncConnection(Protocol):
    def __aiter__(self) -> AsyncIterator[str | bytes]: ...

    async def send(self, message: str) -> None: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> None: ...


class _SyncConnection(Protocol):
    def __iter__(self) -> Iterator[str | bytes]: ...

    def send(self, message: str) -> None: ...

    def recv(self) -> str | bytes: ...

    def close(self) -> None: ...


def websocket_url(rest_url: str) -> str:
    parsed: Final = urlparse(rest_url)
    return parsed._replace(
        scheme="ws" if parsed.scheme == "http" else "wss",
        path=f"{parsed.path.removesuffix('/v1')}/websocket/v1",
    ).geturl()


def _text_message(path: str, request_id: str, content_type: str, body: str) -> str:
    timestamp: Final = datetime.now(timezone.utc).isoformat()
    return (
        f"Path: {path}\r\nX-RequestId: {request_id}\r\nX-Timestamp: {timestamp}\r\n"
        f"Content-Type: {content_type}\r\n\r\n{body}"
    )


def synthesis_messages(ssml: str, output_format: str, viseme_enabled: bool) -> tuple[str, str]:
    request_id: Final = uuid4().hex
    context: Final = {
        "synthesis": {"audio": {"metadataOptions": {"visemeEnabled": viseme_enabled}, "outputFormat": output_format}}
    }
    return (
        _text_message("synthesis.context", request_id, "application/json", json.dumps(context)),
        _text_message("ssml", request_id, "application/ssml+xml", ssml),
    )


def _path(headers: str) -> str:
    return next(
        (value.strip().lower() for name, _, value in _header_lines(headers) if name.strip().lower() == "path"), ""
    )


def _header_lines(headers: str) -> Iterator[tuple[str, str, str]]:
    return (line.partition(":") for line in headers.split("\r\n"))


def _frame(message: str | bytes) -> _Frame:
    if isinstance(message, str):
        headers, _, body = message.partition("\r\n\r\n")
        return _Frame(path=_path(headers), body=body.encode())
    body_at: Final = _HEADER_LENGTH_BYTES + int.from_bytes(message[:_HEADER_LENGTH_BYTES], "big")
    return _Frame(path=_path(message[_HEADER_LENGTH_BYTES:body_at].decode()), body=message[body_at:])


def _viseme_events(body: bytes) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {
            "type": "speech.audio.viseme",
            "viseme_id": entry.data.viseme_id,
            "audio_offset_ms": entry.data.offset / _TICKS_PER_MILLISECOND,
        }
        for entry in _AudioMetadata.model_validate_json(body).metadata
        if entry.type == "Viseme" and entry.data.viseme_id is not None
    )


def _events(frame: _Frame) -> tuple[Mapping[str, object], ...]:
    match frame.path:
        case "audio":
            return (
                ({"type": "speech.audio.delta", "audio": base64.b64encode(frame.body).decode()},) if frame.body else ()
            )
        case "audio.metadata":
            return _viseme_events(frame.body)
        case "turn.end":
            return ({"type": "speech.audio.done"},)
        case _:
            return ()


def _sse(frame: _Frame) -> bytes:
    return b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in _events(frame))


def _refusal(error: Exception) -> BaseLLMException:
    from websockets.exceptions import ConnectionClosed, InvalidStatus

    if isinstance(error, InvalidStatus):
        return BaseLLMException(
            status_code=error.response.status_code,
            message=(error.response.body or b"").decode(errors="replace") or str(error),
            headers=dict(error.response.headers.raw_items()),
        )
    if isinstance(error, ConnectionClosed) and error.rcvd is not None:
        return BaseLLMException(
            status_code=400 if error.rcvd.code == _INVALID_PAYLOAD_CLOSE_CODE else 500,
            message=error.rcvd.reason or str(error),
        )
    return BaseLLMException(status_code=500, message=str(error))


def _open_timeout(timeout: float | httpx.Timeout) -> float | None:
    return timeout.connect if isinstance(timeout, httpx.Timeout) else timeout


def _sync_ssl(url: str) -> ssl.SSLContext | None:
    configured: Final = realtime_ssl_for_url(url)
    if isinstance(configured, str):
        return ssl.create_default_context(cafile=configured)
    return configured if isinstance(configured, ssl.SSLContext) else None


def _speech_response(url: str, stream: httpx.AsyncByteStream | httpx.SyncByteStream) -> HttpxBinaryResponseContent:
    return HttpxBinaryResponseContent(
        httpx.Response(
            status_code=200,
            headers={"content-type": _SSE_MEDIA_TYPE},
            stream=stream,
            request=httpx.Request("GET", url),
        )
    )


class _SseAsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, connection: _AsyncConnection, first: _Frame) -> None:
        self._connection: Final = connection
        self._first: Final = first

    async def _frames(self) -> AsyncIterator[_Frame]:
        yield self._first
        async for message in self._connection:
            yield _frame(message)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for frame in self._frames():
            if events := _sse(frame):
                yield events
            if frame.path == "turn.end":
                return

    async def aclose(self) -> None:
        await self._connection.close()


class _SseSyncByteStream(httpx.SyncByteStream):
    def __init__(self, connection: _SyncConnection, first: _Frame) -> None:
        self._connection: Final = connection
        self._first: Final = first

    def _frames(self) -> Iterator[_Frame]:
        yield self._first
        yield from map(_frame, self._connection)

    def __iter__(self) -> Iterator[bytes]:
        for frame in self._frames():
            if events := _sse(frame):
                yield events
            if frame.path == "turn.end":
                return

    def close(self) -> None:
        self._connection.close()


async def _async_first_frame(connection: _AsyncConnection, messages: tuple[str, str]) -> _Frame:
    """Azure refuses a bad voice, format or SSML by closing right after `turn.start`, so wait for what follows it"""
    for message in messages:
        await connection.send(message)
    while (frame := _frame(await connection.recv())).path == "turn.start":
        pass
    return frame


def _sync_first_frame(connection: _SyncConnection, messages: tuple[str, str]) -> _Frame:
    for message in messages:
        connection.send(message)
    while (frame := _frame(connection.recv())).path == "turn.start":
        pass
    return frame


async def async_synthesize_sse(
    url: str,
    headers: Mapping[str, str],
    messages: tuple[str, str],
    timeout: float | httpx.Timeout,
    stream: bool,
) -> HttpxBinaryResponseContent:
    import websockets
    from websockets.exceptions import WebSocketException

    try:
        connection: Final = await websockets.connect(
            url, additional_headers=headers, open_timeout=_open_timeout(timeout), ssl=realtime_ssl_for_url(url)
        )
    except WebSocketException as error:
        raise _refusal(error) from error
    try:
        first: Final = await _async_first_frame(connection, messages)
    except WebSocketException as error:
        await connection.close()
        raise _refusal(error) from error
    response: Final = _speech_response(url, _SseAsyncByteStream(connection, first))
    if not stream:
        await response.aread()
    return response


def synthesize_sse(
    url: str,
    headers: Mapping[str, str],
    messages: tuple[str, str],
    timeout: float | httpx.Timeout,
    stream: bool,
) -> HttpxBinaryResponseContent:
    from websockets.exceptions import WebSocketException
    from websockets.sync.client import connect

    try:
        connection: Final = connect(
            url, additional_headers=headers, open_timeout=_open_timeout(timeout), ssl=_sync_ssl(url)
        )
    except WebSocketException as error:
        raise _refusal(error) from error
    try:
        first: Final = _sync_first_frame(connection, messages)
    except WebSocketException as error:
        connection.close()
        raise _refusal(error) from error
    response: Final = _speech_response(url, _SseSyncByteStream(connection, first))
    if not stream:
        response.read()
    return response
