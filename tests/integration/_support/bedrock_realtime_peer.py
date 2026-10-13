"""A scripted Bedrock Nova Sonic realtime peer.

The handler dials ``api_base`` with ``invoke_model_with_bidirectional_stream`` through
``AWSCRTHTTPClient``, which only speaks h2 over TLS ALPN, so this peer is a loopback TLS
server offering ``h2`` on a thread-owned asyncio loop (same shape as ``async_server.py``).
It answers ``POST /model/{id}/invoke-with-bidirectional-stream``, decodes the client's
signed event-stream chunks into the ``{"event": {...}}`` documents Bedrock would read,
and answers each consumed ``Turn`` with event-stream ``chunk`` frames carrying
``{"bytes": <base64 event json>}`` payloads, the reverse of what the SDK produces.

Turn triggers are inbound events. ``user_input`` fires on the ``contentEnd`` closing the
first inbound USER-role TEXT content block, the end of a forwarded conversation.item.create
text message (the injected trigger-audio block is USER/AUDIO, so it never trips it).
``tool_result`` fires on the Nth inbound ``toolResult`` event, which the proxy sends inside
a TOOL contentStart/toolResult/contentEnd triple whose ``toolResultInputConfiguration``
``toolUseId`` names the call being answered.
"""

from __future__ import annotations

import asyncio
import base64
import json
import ssl
import struct
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from queue import SimpleQueue
from threading import Thread
from typing import Final, Literal

import h2.config
import h2.connection
import h2.events
import h2.exceptions
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.tls import server_context
from tests.integration._support.upstream import aws_event_stream_frame

BedrockEvent = Mapping[str, JsonValue]

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _ids() -> tuple[str, str, str, str]:
    return (uuid.uuid4().hex, uuid.uuid4().hex, uuid.uuid4().hex, uuid.uuid4().hex)


def _block(prompt: str, content: str, session: str, completion: str) -> dict[str, JsonValue]:
    return {"promptName": prompt, "contentName": content, "sessionId": session, "completionId": completion}


def user_transcript(text: str) -> tuple[BedrockEvent, ...]:
    """The USER ASR transcript block Bedrock emits once it has consumed the user turn."""
    prompt, content, session, completion = _ids()
    return (
        {
            "event": {
                "contentStart": {
                    **_block(prompt, content, session, completion),
                    "role": "USER",
                    "type": "TEXT",
                    "interactive": False,
                    "textOutputConfiguration": {"mediaType": "text/plain"},
                    "additionalModelFields": json.dumps({"generationStage": "FINAL"}),
                }
            }
        },
        {"event": {"textOutput": {**_block(prompt, content, session, completion), "role": "USER", "content": text}}},
        {
            "event": {
                "contentEnd": {
                    **_block(prompt, content, session, completion),
                    "stopReason": "PARTIAL_TURN",
                    "type": "TEXT",
                }
            }
        },
    )


def tool_use(tool_use_id: str, name: str, content: str | None) -> tuple[BedrockEvent, ...]:
    """The role-TOOL contentStart/toolUse/contentEnd triple Bedrock emits per tool call."""
    prompt, content_name, session, completion = _ids()
    tool_use_body: Final[dict[str, JsonValue]] = {
        **_block(prompt, content_name, session, completion),
        "role": "TOOL",
        "toolName": name,
        "toolUseId": tool_use_id,
    }
    if content is not None:
        tool_use_body["content"] = content
    return (
        {
            "event": {
                "contentStart": {
                    **_block(prompt, content_name, session, completion),
                    "role": "TOOL",
                    "type": "TOOL",
                    "toolUseOutputConfiguration": {"mediaType": "application/json"},
                }
            }
        },
        {"event": {"toolUse": tool_use_body}},
        {
            "event": {
                "contentEnd": {
                    **_block(prompt, content_name, session, completion),
                    "stopReason": "TOOL_USE",
                    "type": "TOOL",
                }
            }
        },
    )


def assistant_text(text: str, stop_reason: str = "END_TURN") -> tuple[BedrockEvent, ...]:
    """An ASSISTANT text block (SPECULATIVE stage) closed with `stop_reason`."""
    prompt, content, session, completion = _ids()
    return (
        {
            "event": {
                "contentStart": {
                    **_block(prompt, content, session, completion),
                    "role": "ASSISTANT",
                    "type": "TEXT",
                    "interactive": False,
                    "textOutputConfiguration": {"mediaType": "text/plain"},
                    "additionalModelFields": json.dumps({"generationStage": "SPECULATIVE"}),
                }
            }
        },
        {
            "event": {
                "textOutput": {**_block(prompt, content, session, completion), "role": "ASSISTANT", "content": text}
            }
        },
        {
            "event": {
                "contentEnd": {
                    **_block(prompt, content, session, completion),
                    "stopReason": stop_reason,
                    "type": "TEXT",
                }
            }
        },
    )


def usage(input_tokens: int, output_tokens: int) -> BedrockEvent:
    total: Final = {
        "input": {"speechTokens": 0, "textTokens": input_tokens},
        "output": {"speechTokens": 0, "textTokens": output_tokens},
    }
    return {
        "event": {
            "usageEvent": {
                "promptName": uuid.uuid4().hex,
                "sessionId": uuid.uuid4().hex,
                "completionId": uuid.uuid4().hex,
                "details": {"delta": total["output"], "total": total},
                "totalInputTokens": input_tokens,
                "totalOutputTokens": output_tokens,
                "totalTokens": input_tokens + output_tokens,
            }
        }
    }


@dataclass(frozen=True, slots=True)
class Turn:
    after: Literal["user_input", "tool_result"]
    events: tuple[BedrockEvent, ...]


def _event_body(event: BedrockEvent) -> tuple[str, Mapping[str, JsonValue]]:
    body: Final = event["event"]
    assert isinstance(body, Mapping) and len(body) == 1, event
    name: Final = next(iter(body))
    fields: Final = body[name]
    assert isinstance(fields, Mapping), event
    return name, fields


def _outbound_chunk(event: BedrockEvent) -> bytes:
    wrapped: Final = json.dumps({"bytes": base64.b64encode(json.dumps(event).encode()).decode()}).encode()
    return aws_event_stream_frame(
        {":event-type": "chunk", ":content-type": "application/json", ":message-type": "event"}, wrapped
    )


def _parse_frames(buffer: bytes) -> tuple[tuple[bytes, ...], bytes]:
    """Slice complete event-stream frames out of `buffer`; return (payloads, remainder)."""
    payloads: Final[list[bytes]] = []
    view = memoryview(buffer)
    while len(view) >= 12:
        total, header_length = struct.unpack_from("!II", view, 0)
        if total > len(view):
            break
        payloads.append(view[12 + header_length : total - 4].tobytes())
        view = view[total:]
    return tuple(payloads), view.tobytes()


def _frame_payload(raw: bytes) -> bytes:
    """The payload slice of one complete framed message (prelude + headers + payload + CRC)."""
    if len(raw) < 16:
        return b""
    _total, header_length = struct.unpack_from("!II", raw, 0)
    return raw[12 + header_length : -4]


def _decode_inbound(chunk_payload: bytes) -> BedrockEvent | None:
    """One signed chunk frame's payload -> the inner chunk's {"bytes": ...} event."""
    inner_payload: Final = _frame_payload(chunk_payload)
    try:
        wrapped: Final = _JSON_OBJECT.validate_json(inner_payload)
    except ValueError:
        return None
    encoded: Final = wrapped.get("bytes")
    if not isinstance(encoded, str):
        return None
    try:
        return _JSON_OBJECT.validate_json(base64.b64decode(encoded))
    except ValueError:
        return None


class _Inbound:
    """Inbound event bookkeeping for turn triggers and accessors."""

    def __init__(self) -> None:
        self.events: Final = SimpleQueue[BedrockEvent]()
        self.open_blocks: Final[dict[str, tuple[str, str]]] = {}
        self.tool_use_ids: Final[dict[str, str]] = {}
        self.tool_results_seen = 0
        self.user_text_closed = False

    def record(self, event: BedrockEvent) -> None:
        self.events.put(event)
        name, fields = _event_body(event)
        content_name = fields.get("contentName")
        if name == "contentStart" and isinstance(content_name, str):
            self.open_blocks[content_name] = (str(fields.get("role")), str(fields.get("type")))
            configuration = fields.get("toolResultInputConfiguration")
            if isinstance(configuration, Mapping) and isinstance(configuration.get("toolUseId"), str):
                self.tool_use_ids[content_name] = configuration["toolUseId"]
        elif name == "contentEnd" and isinstance(content_name, str):
            role_type: Final = self.open_blocks.pop(content_name, ("", ""))
            if role_type == ("USER", "TEXT"):
                self.user_text_closed = True
        elif name == "toolResult":
            self.tool_results_seen += 1


class _Script:
    """Consumes turns in order as their inbound triggers arrive."""

    def __init__(self, turns: tuple[Turn, ...]) -> None:
        self._turns: Final = turns
        self._next = 0

    @property
    def exhausted(self) -> bool:
        return self._next >= len(self._turns)

    def ready_events(self, state: _Inbound, event: BedrockEvent) -> tuple[BedrockEvent, ...]:
        if self._next >= len(self._turns):
            return ()
        turn: Final = self._turns[self._next]
        if turn.after == "user_input":
            if not state.user_text_closed:
                return ()
        else:
            name, _fields = _event_body(event)
            if name != "toolResult":
                return ()
        self._next += 1
        return turn.events


class _PeerState:
    def __init__(self, turns: tuple[Turn, ...]) -> None:
        self.inbound: Final = _Inbound()
        self.script: Final = _Script(turns)


@dataclass(frozen=True, slots=True)
class BedrockRealtimePeer:
    api_base: str
    _state: _PeerState
    _drained: list[BedrockEvent]

    def received(self) -> tuple[BedrockEvent, ...]:
        while not self._state.inbound.events.empty():
            self._drained.append(self._state.inbound.events.get())  # drained log, not a built value
        return tuple(self._drained)

    def tool_specs(self) -> tuple[Mapping[str, JsonValue], ...]:
        return tuple(chain.from_iterable(_tool_specs(event) for event in self.received()))

    def tool_results(self) -> tuple[tuple[str, str], ...]:
        """(toolUseId, content) per inbound toolResult in arrival order."""
        return tuple(result for event in self.received() if (result := self._tool_result(event)) is not None)

    def _tool_result(self, event: BedrockEvent) -> tuple[str, str] | None:
        name, fields = _event_body(event)
        if name != "toolResult":
            return None
        content_name: Final = fields.get("contentName")
        content: Final = fields.get("content")
        if not isinstance(content_name, str) or not isinstance(content, str):
            return None
        return (self._state.inbound.tool_use_ids.get(content_name, ""), content)


def _tool_specs(event: BedrockEvent) -> tuple[Mapping[str, JsonValue], ...]:
    name, fields = _event_body(event)
    if name != "promptStart":
        return ()
    configuration: Final = fields.get("toolConfiguration")
    if not isinstance(configuration, Mapping):
        return ()
    tools: Final = configuration.get("tools")
    if not isinstance(tools, list):
        return ()
    return tuple(
        entry["toolSpec"]
        for entry in tools
        if isinstance(entry, Mapping) and isinstance(entry.get("toolSpec"), Mapping)
    )


class _H2Session:
    """One bidirectional-stream connection: decode inbound, answer scripted turns."""

    def __init__(self, state: _PeerState, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._state: Final = state
        self._reader: Final = reader
        self._writer: Final = writer
        self._h2: Final = h2.connection.H2Connection(config=h2.config.H2Configuration(client_side=False))
        self._buffers: Final[dict[int, bytes]] = {}
        self._ended: set[int] = set()
        self._closed = False

    async def run(self) -> None:
        self._h2.initiate_connection()
        self._writer.write(self._h2.data_to_send())
        await self._writer.drain()
        while True:
            incoming: Final = await self._reader.read(65536)
            if not incoming:
                return
            for event in self._h2.receive_data(incoming):
                self._apply(event)
            pending: Final = self._h2.data_to_send()
            if pending:
                self._writer.write(pending)
                await self._writer.drain()
            if self._closed:
                self._writer.close()
                return

    def _apply(self, event: h2.events.Event) -> None:
        if isinstance(event, h2.events.RequestReceived):
            self._h2.send_headers(
                event.stream_id,
                [(":status", "200"), ("content-type", "application/vnd.amazon.eventstream")],
            )
        elif isinstance(event, h2.events.DataReceived):
            self._h2.acknowledge_received_data(event.flow_controlled_length, event.stream_id)
            self._buffers[event.stream_id] = self._buffers.get(event.stream_id, b"") + event.data
            payloads, remainder = _parse_frames(self._buffers[event.stream_id])
            self._buffers[event.stream_id] = remainder
            for payload in payloads:
                decoded: Final = _decode_inbound(payload)
                if decoded is None:
                    continue
                self._state.inbound.record(decoded)
                if event.stream_id not in self._ended:
                    for outbound in self._state.script.ready_events(self._state.inbound, decoded):
                        self._h2.send_data(event.stream_id, _outbound_chunk(outbound))
                    if self._state.script.exhausted:
                        self._end_stream(event.stream_id)
        elif isinstance(event, (h2.events.StreamEnded, h2.events.StreamReset)):
            self._close_stream(event.stream_id)
        elif isinstance(event, h2.events.ConnectionTerminated):
            self._writer.close()

    def _close_stream(self, stream_id: int) -> None:
        self._end_stream(stream_id)
        self._closed = True
        self._h2.close_connection()

    def _end_stream(self, stream_id: int) -> None:
        if stream_id in self._ended:
            return
        self._ended.add(stream_id)
        with suppress(h2.exceptions.ProtocolError):
            self._h2.end_stream(stream_id)


class _LoopbackTls:
    """TLS h2 server on a thread-owned loop; one instance per peer."""

    def __init__(self, port: int, context: ssl.SSLContext, state: _PeerState) -> None:
        self.loop: Final = asyncio.new_event_loop()
        self._port: Final = port
        self._context: Final = context
        self._state: Final = state
        self._server: asyncio.Server | None = None
        self._thread: Final = Thread(target=self._run, name=f"bedrock-realtime-peer-{port}", daemon=True)
        self._connections: tuple[asyncio.Task[None], ...] = ()

    def start(self) -> None:
        try:
            self._thread.start()
            asyncio.run_coroutine_threadsafe(self._bind(), self.loop).result(timeout=10)
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        if not self._thread.is_alive():
            if not self.loop.is_closed():
                self.loop.close()
            return
        try:
            asyncio.run_coroutine_threadsafe(self._close(), self.loop).result(timeout=10)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self._thread.join(10)
            assert not self._thread.is_alive(), "Bedrock realtime peer thread did not stop"

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()

    async def _bind(self) -> None:
        self._server = await asyncio.start_server(self._serve, "127.0.0.1", self._port, ssl=self._context)

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task: Final = asyncio.current_task()
        assert task is not None
        self._connections = (*self._connections, task)
        try:
            await _H2Session(self._state, reader, writer).run()
        except (ConnectionError, OSError, asyncio.IncompleteReadError, h2.exceptions.StreamClosedError):
            pass
        finally:
            writer.close()
            with suppress(ConnectionError, OSError):
                await writer.wait_closed()
            self._connections = tuple(connection for connection in self._connections if connection is not task)

    async def _close(self) -> None:
        server: Final = self._server
        if server is not None:
            server.close()
        connections: Final = self._connections
        for task in connections:
            task.cancel()
        await asyncio.gather(*connections, return_exceptions=True)
        current: Final = asyncio.current_task()
        remaining: Final = tuple(task for task in asyncio.all_tasks(self.loop) if task is not current)
        for task in remaining:
            task.cancel()
        await asyncio.gather(*remaining, return_exceptions=True)
        if server is not None:
            await server.wait_closed()


@contextmanager
def bedrock_realtime_peer(turns: tuple[Turn, ...], cert: tuple[Path, Path]) -> Iterator[BedrockRealtimePeer]:
    """Serve scripted `turns` on a loopback TLS h2 endpoint; yield the peer.

    `cert` is the self-signed pair the owned proxy already trusts via SSL_CERT_FILE.
    SigV4 headers are ignored; the stream closes cleanly when the client ends it.
    """
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: Final = probe.getsockname()[1]
    context: Final = server_context(cert[0], cert[1])
    context.set_alpn_protocols(["h2"])
    state: Final = _PeerState(turns)
    server: Final = _LoopbackTls(port, context, state)
    server.start()
    try:
        yield BedrockRealtimePeer(api_base=f"https://localhost:{port}", _state=state, _drained=[])
    finally:
        server.stop()
