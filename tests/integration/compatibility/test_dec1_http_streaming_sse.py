"""Server-sent event framing and client disconnects, as seen on the client side of the proxy.

A streamed chat completion is a `text/event-stream` body made of `data: <json>` lines, each followed by one blank
line, and it ends with the literal frame `data: [DONE]` (`proxy_server.py:10625`). The first frame must reach the
client while the provider is still producing the rest, so anything in front of the proxy has to pass bytes through
as they arrive and never buffer the body. Responses API streams use `event:` lines and have no `[DONE]` frame
(`proxy_server.py:11080`). A client that goes away must not corrupt accounting: with default settings the provider
call of a non-streaming request runs to completion and is billed (`cancel_on_disconnect` is off,
`common_request_processing.py:3347`), while a streaming client that hangs up closes the provider stream. Deeper
stream semantics (mid-stream fallback, usage chunks) are pinned in `tests/integration/streaming/`.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import BinaryIO, Final

from integration._support.client import JSON_OBJECT, Gateway, eventually, list_value, object_value
from integration._support.wire import Reply, Request
from integration.compatibility.dec1_http_support import (
    CHAT,
    chat_body,
    chat_completion,
    chat_stream_chunks,
    connection,
    front_door,
    json_post,
    provider_error,
    read_response,
    spend_rows,
)
from pydantic import JsonValue

RESPONSE_EVENTS: Final = (
    (
        b'event: response.created\ndata: {"type":"response.created","sequence_number":0,"response":{"id":"resp_dec1",'
        b'"object":"response","created_at":1,"status":"in_progress","model":"gpt-4o-mini","output":[]}}\n\n'
    ),
    (
        b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","sequence_number":1,'
        b'"item_id":"msg_dec1","output_index":0,"content_index":0,"delta":"streamed"}\n\n'
    ),
    (
        b'event: response.completed\ndata: {"type":"response.completed","sequence_number":2,"response":{"id":"resp_dec1",'
        b'"object":"response","created_at":1,"status":"completed","model":"gpt-4o-mini","output":[],'
        b'"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}\n\n'
    ),
)


def _stream(gateway: Gateway, path: str, body: dict[str, JsonValue], key: str | None = None) -> tuple[int, str, bytes]:
    headers: Final = {"Authorization": f"Bearer {gateway.key if key is None else key}"}
    with gateway.client.stream("POST", path, json=body, headers=headers) as response:
        return response.status_code, response.headers.get("content-type", ""), response.read()


def _frames(body: bytes) -> list[str]:
    return [frame for frame in body.decode().split("\n\n") if frame]


def _text(frames: list[str]) -> str:
    chunks: Final = [
        JSON_OBJECT.validate_json(frame.removeprefix("data: ")) for frame in frames if "[DONE]" not in frame
    ]
    choices: Final = [object_value(choice) for chunk in chunks for choice in list_value(chunk["choices"])]
    return "".join(str(delta["content"]) for delta in map(_delta, choices) if delta.get("content"))


def _delta(choice: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return object_value(choice["delta"])


def test_a_chat_stream_is_data_frames_separated_by_blank_lines_and_ends_with_done(gateway: Gateway) -> None:
    """The provider streams the two text pieces `streamed ` and `answer`. The client must receive a 200
    `text/event-stream` body whose every frame starts with `data: `, whose last frame is exactly `data: [DONE]`,
    whose bytes end with `\\n\\n`, and whose deltas join to `streamed answer`. SDK stream parsers depend on this
    framing byte for byte."""
    with front_door(gateway) as door:
        status, content_type, body = _stream(gateway, CHAT, chat_body(door.chat_model(), stream=True))
    frames: Final = _frames(body)
    assert status == 200, body
    assert content_type.startswith("text/event-stream")
    assert body.endswith(b"\n\n")
    assert [frame for frame in frames if not frame.startswith("data: ")] == []
    assert frames[-1] == "data: [DONE]"
    assert _text(frames) == "streamed answer"


def test_a_chat_stream_contains_no_event_or_comment_lines(gateway: Gateway) -> None:
    """A chat completion is streamed through the proxy. No line of the body may start with `event:`, `id:`,
    `retry:` or `:` (an SSE comment used as a keep-alive). OpenAI chat streams are data-only, and a forwarder that
    injects heartbeat comments would break strict clients."""
    with front_door(gateway) as door:
        _, _, body = _stream(gateway, CHAT, chat_body(door.chat_model(), stream=True))
    lines: Final = [line for line in body.decode().split("\n") if line]
    assert [line for line in lines if line.startswith(("event:", "id:", "retry:", ":"))] == []


def test_every_chat_stream_chunk_carries_the_same_completion_id(gateway: Gateway) -> None:
    """A chat completion is streamed through the proxy. Every JSON chunk must carry the same `id` and the object
    type `chat.completion.chunk`, so a client can stitch the chunks into one message."""
    with front_door(gateway) as door:
        _, _, body = _stream(gateway, CHAT, chat_body(door.chat_model(), stream=True))
    chunks: Final = [json.loads(frame.removeprefix("data: ")) for frame in _frames(body) if "[DONE]" not in frame]
    assert len({chunk["id"] for chunk in chunks}) == 1
    assert {chunk["object"] for chunk in chunks} == {"chat.completion.chunk"}


def test_the_first_frame_reaches_the_client_before_the_provider_sends_the_rest(gateway: Gateway) -> None:
    """The provider sends the first stream chunk and then waits on a gate. The client must be able to read a
    complete first `data:` frame while the gate is still closed, and after the gate opens the stream must finish
    with `data: [DONE]`. A proxy or forwarder that buffers the response body would deliver nothing until the end
    and the first read would time out."""
    gate: Final = threading.Event()

    def respond(request: Request) -> Reply:
        return Reply(
            content_type="text/event-stream", chunks=chat_stream_chunks(("first ", "second")), gate_after_first=gate
        )

    with front_door(gateway, respond) as door:
        body: Final = json.dumps(chat_body(door.chat_model(), stream=True)).encode()
        with connection(gateway, timeout=30) as (client, stream):
            client.sendall(json_post(gateway, CHAT, body, extra=(("Connection", "close"),)))
            status_line: Final = stream.readline()
            early: Final = _read_until_first_frame(stream)
            gate_was_closed: Final = not gate.is_set()
            gate.set()
            rest: Final = stream.read()
    assert status_line.startswith(b"HTTP/1.1 200")
    assert gate_was_closed
    assert b"data: {" in early
    assert b"data: [DONE]" in rest


def _read_until_first_frame(stream: BinaryIO) -> bytes:
    seen: Final[list[bytes]] = []
    while b"data: {" not in b"".join(seen):
        line = stream.readline()
        assert line, "The connection closed before the first frame arrived"
        seen.append(line)
    return b"".join(seen)


def test_a_slow_reader_still_receives_the_whole_stream(gateway: Gateway) -> None:
    """The provider streams 200 text pieces as fast as it can. The client reads the response in small steps with a
    pause between reads. The client must end up with all 200 pieces in order and the final `data: [DONE]` frame:
    back-pressure may slow the stream down but may never drop or reorder frames."""
    pieces: Final = tuple(f"p{index:03d} " for index in range(200))

    def respond(request: Request) -> Reply:
        return Reply(content_type="text/event-stream", chunks=chat_stream_chunks(pieces))

    with front_door(gateway, respond) as door:
        headers: Final = {"Authorization": f"Bearer {gateway.key}"}
        with gateway.client.stream(
            "POST", CHAT, json=chat_body(door.chat_model(), stream=True), headers=headers
        ) as response:
            received: Final[list[bytes]] = []
            for part in response.iter_raw(256):
                received.append(part)
                time.sleep(0.005)
    frames: Final = _frames(b"".join(received))
    assert frames[-1] == "data: [DONE]"
    assert _text(frames) == "".join(pieces)


def test_a_provider_error_before_the_first_chunk_is_a_plain_json_error_not_a_stream(gateway: Gateway) -> None:
    """The client asks for a stream and the provider answers 429 with a JSON error before sending any chunk. The
    proxy must answer HTTP 429 with an `application/json` body holding an `error` object, and the body must not
    contain `data:`. Clients decide whether to retry from the status line, which a 200 stream with an error frame
    would hide."""
    with front_door(gateway, lambda request: provider_error(429)) as door:
        model: Final = door.chat_model()
        status, content_type, body = _stream(gateway, CHAT, chat_body(model, stream=True, num_retries=0))
    assert status == 429, body
    assert content_type.startswith("application/json")
    assert b"data:" not in body
    assert "error" in json.loads(body)


def test_a_provider_stream_cut_mid_way_ends_the_client_stream_with_an_error_frame(gateway: Gateway) -> None:
    """The provider sends the first two stream chunks and then closes the connection without finishing. The client
    already has HTTP 200, so the proxy must end the body cleanly with a `data:` frame whose JSON has an `error`
    key, followed by `data: [DONE]` (`async_data_generator`, `proxy_server.py:11112`). UNVERIFIED: assumes no
    mid-stream fallback is configured for the model, so the truncation surfaces to the client."""

    def respond(request: Request) -> Reply:
        return Reply(
            content_type="text/event-stream", chunks=chat_stream_chunks(("one ", "two ", "three")), abort_after=2
        )

    with front_door(gateway, respond) as door:
        status, _, body = _stream(gateway, CHAT, chat_body(door.chat_model(), stream=True, num_retries=0))
    frames: Final = _frames(body)
    assert status == 200
    assert frames[-1] == "data: [DONE]"
    assert [frame for frame in frames[:-1] if "error" in json.loads(frame.removeprefix("data: "))] != []


def test_a_responses_stream_uses_event_lines_and_has_no_done_frame(gateway: Gateway) -> None:
    """The provider streams three Responses API events. The client must receive a `text/event-stream` body that
    contains the `response.completed` event and does not contain `[DONE]`: only chat completion streams end with
    that sentinel (`proxy_server.py:11080`). UNVERIFIED: whether the proxy re-emits the `event:` line or only the
    `data:` line, so the test only requires each event type to appear in a `data:` payload."""

    def respond(request: Request) -> Reply:
        return Reply(content_type="text/event-stream", chunks=RESPONSE_EVENTS)

    with front_door(gateway, respond) as door:
        request_body: Final = {"model": door.chat_model(), "input": f"dec1 {uuid.uuid4().hex}", "stream": True}
        status, content_type, body = _stream(gateway, "/v1/responses", request_body)
    assert status == 200, body
    assert content_type.startswith("text/event-stream")
    assert b"[DONE]" not in body
    assert b'"type":"response.completed"' in body.replace(b'": "', b'":"')


def test_a_client_that_hangs_up_on_a_non_streaming_call_is_still_billed_for_it(gateway: Gateway) -> None:
    """A fresh key sends a non-streaming chat call and closes its socket while the provider is still holding the
    answer. The provider then answers. With default settings (`cancel_on_disconnect` off) the provider call must
    run to completion, receive exactly one request, and one spend row must be written for the key. A forwarder
    that resets the upstream connection when the client leaves would turn a billed call into an unbilled one."""
    gate: Final = threading.Event()

    def respond(request: Request) -> Reply:
        return Reply(chunks=(chat_completion(),), gate_before_first=gate)

    with front_door(gateway, respond) as door:
        key: Final = door.scenario.key()
        body: Final = json.dumps(chat_body(door.chat_model())).encode()
        with connection(gateway) as (client, _):
            client.sendall(json_post(gateway, CHAT, body, key=key))
            eventually(door.upstream.received.qsize, lambda waiting: waiting == 1, seconds=30)
        time.sleep(1)
        gate.set()
        rows: Final = spend_rows(gateway, key)
        assert len(rows) == 1
        assert len(door.calls()) == 1


def test_a_client_that_hangs_up_mid_stream_closes_the_provider_stream(gateway: Gateway) -> None:
    """The provider sends the first stream chunk and then keeps the stream open. The client reads the first frame
    and closes its socket. The proxy must close its connection to the provider within 30 seconds, so the provider
    stops generating tokens nobody will read. A forwarder that keeps the upstream leg open after the client leaves
    would leak provider connections."""
    gate: Final = threading.Event()

    def respond(request: Request) -> Reply:
        return Reply(
            content_type="text/event-stream",
            chunks=chat_stream_chunks(("first ", "second")),
            gate_after_first=gate,
            heartbeat=b": keep-alive\n\n",
            pause_between_chunks=0.2,
        )

    with front_door(gateway, respond) as door:
        body: Final = json.dumps(chat_body(door.chat_model(), stream=True)).encode()
        with connection(gateway) as (client, stream):
            client.sendall(json_post(gateway, CHAT, body))
            _read_until_first_frame(stream)
        try:
            assert eventually(door.upstream.disconnected.qsize, lambda count: count >= 1, seconds=30) >= 1
        finally:
            gate.set()


def test_a_non_streaming_request_is_not_answered_as_an_event_stream(gateway: Gateway) -> None:
    """A chat call without `stream` is sent with `Accept: text/event-stream`. The proxy must still answer one
    `application/json` document with a `Content-Length`: streaming is chosen by the body's `stream` field, never by
    the Accept header."""
    with front_door(gateway) as door:
        body: Final = json.dumps(chat_body(door.chat_model())).encode()
        with connection(gateway) as (client, stream):
            client.sendall(json_post(gateway, CHAT, body, extra=(("Accept", "text/event-stream"),)))
            response: Final = read_response(stream)
    assert response.status == 200, response.body
    assert (response.header("content-type") or "").startswith("application/json")
    assert response.header("content-length") == str(len(response.body))
    assert response.json()["object"] == "chat.completion"
