"""How the front door reads a request body: framing, content type, encoding and malformed JSON.

The proxy parses the body of an inference call itself (`read_request_body`, `http_parsing_utils.py:266`), with
`orjson` first and a more forgiving fallback second, and it does so before authentication finishes because the
model name lives in the body. That gives a set of behaviours no schema describes: any content type other than a
form or binary type is parsed as JSON, a duplicated key resolves to its last value, malformed JSON is a 400 in
the OpenAI envelope with `param: "request_body"`, and there is no size cap unless an operator sets one. The
framing cases (`Transfer-Encoding: chunked`, `Expect: 100-continue`) are written as raw bytes on a socket so
the test controls exactly what the server, or a forwarder in front of it, has to decode.
"""

from __future__ import annotations

import gzip
import json
import uuid
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway, list_value, object_value
from integration.compatibility.dec1_http_support import (
    CHAT,
    chat_body,
    connection,
    exchange,
    front_door,
    host_header,
    json_post,
    read_response,
    request_bytes,
)
from pydantic import JsonValue

INVALID_JSON_ERROR: Final = {"type": "invalid_request_error", "param": "request_body", "code": "400"}
NOT_AN_OBJECT: Final = {
    "message": "Invalid JSON payload: expected an object",
    "type": "invalid_request_error",
    "param": "request_body",
    "code": "400",
}
MIB: Final = 1024 * 1024


def _post(gateway: Gateway, data: bytes, content_type: str | None = "application/json") -> tuple[int, bytes]:
    headers: Final = {
        "Authorization": f"Bearer {gateway.key}",
        **({"content-type": content_type} if content_type is not None else {}),
    }
    response: Final = gateway.client.post(CHAT, content=data, headers=headers)
    return response.status_code, response.content


def _error(content: bytes) -> dict[str, JsonValue]:
    return object_value(JSON_OBJECT.validate_json(content)["error"])


def _without_message(error: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: value for key, value in error.items() if key != "message"}


def _received_messages(body: bytes) -> JsonValue:
    return JSON_OBJECT.validate_json(body)["messages"]


def test_malformed_json_is_a_400_naming_the_request_body(gateway: Gateway) -> None:
    """The caller posts `{"model": "x", "messages": [` (cut off mid-array) to the chat path with a valid key. The
    proxy must answer 400 with the OpenAI envelope: a `message` starting "Invalid JSON payload: ", `type:
    "invalid_request_error"`, `param: "request_body"` and `code: "400"` as a string. No provider is called."""
    with front_door(gateway) as door:
        status, content = _post(gateway, b'{"model": "x", "messages": [')
        assert status == 400, content
        error: Final = _error(content)
        assert _without_message(error) == INVALID_JSON_ERROR
        assert str(error["message"]).startswith("Invalid JSON payload: ")
        assert door.calls() == ()


@pytest.mark.parametrize(
    "data",
    (
        pytest.param(b'[{"model": "x"}]', id="array"),
        pytest.param(b'"just a string"', id="string"),
        pytest.param(b"42", id="number"),
        pytest.param(b"null", id="null"),
        pytest.param(b"true", id="boolean"),
    ),
)
def test_valid_json_that_is_not_an_object_is_a_400(gateway: Gateway, data: bytes) -> None:
    """The caller posts a body that is valid JSON but not an object (an array, a string, a number, `null`,
    `true`) to the chat path. The proxy must answer 400 with exactly the error `Invalid JSON payload: expected an
    object`, `type: invalid_request_error`, `param: request_body`, `code: "400"` (`http_parsing_utils.py:377`)."""
    status, content = _post(gateway, data)
    assert status == 400, content
    assert _error(content) == NOT_AN_OBJECT


def test_an_empty_body_is_a_400_and_reaches_no_provider(gateway: Gateway) -> None:
    """The caller posts to the chat path with `Content-Length: 0`. An empty body is read as an empty object, so
    the request has no model and no messages and the proxy must answer 400 with the OpenAI envelope (`type` and
    `code: "400"` present) and call no provider. UNVERIFIED: the exact message, which names the missing field."""
    with front_door(gateway) as door:
        status, content = _post(gateway, b"")
        assert status == 400, content
        error: Final = _error(content)
        assert error["code"] == "400"
        assert isinstance(error["type"], str)
        assert door.calls() == ()


def test_a_byte_order_mark_before_the_json_is_a_400(gateway: Gateway) -> None:
    """The caller posts a valid chat object prefixed with the UTF-8 byte order mark `EF BB BF`. Neither parser
    accepts a BOM, so the proxy must answer 400 with `param: "request_body"` and a message starting "Invalid JSON
    payload: ". Windows tooling produces such bodies, and a lenient front-door parser would silently accept them."""
    with front_door(gateway) as door:
        body: Final = b"\xef\xbb\xbf" + json.dumps(chat_body(door.chat_model())).encode()
        status, content = _post(gateway, body)
        assert status == 400, content
        assert _without_message(_error(content)) == INVALID_JSON_ERROR
        assert door.calls() == ()


def test_a_lone_surrogate_escape_is_a_400(gateway: Gateway) -> None:
    r"""The caller posts a chat object whose message content contains the escape `\ud83d` with no low surrogate.
    The fast parser rejects it and the fallback refuses text no provider can encode, so the proxy must answer 400
    with `param: "request_body"` and call no provider (`http_parsing_utils.py:338`)."""
    with front_door(gateway) as door:
        body: Final = json.dumps(chat_body(door.chat_model(), "PLACEHOLDER")).replace("PLACEHOLDER", "bad \\ud83d")
        status, content = _post(gateway, body.encode())
        assert status == 400, content
        assert _without_message(_error(content)) == INVALID_JSON_ERROR
        assert door.calls() == ()


def test_a_body_that_is_not_utf8_is_refused_and_reaches_no_provider(gateway: Gateway) -> None:
    """The caller posts a chat object whose message content holds the Latin-1 byte `E9`, which is not valid
    UTF-8. The proxy must answer 400 and must not call the provider with a guessed decoding."""
    with front_door(gateway) as door:
        body: Final = (
            json.dumps(chat_body(door.chat_model(), "PLACEHOLDER")).encode().replace(b"PLACEHOLDER", b"caf\xe9")
        )
        status, content = _post(gateway, body)
        assert status == 400, content
        assert door.calls() == ()


def test_a_body_that_is_not_utf8_is_reported_as_invalid_json(gateway: Gateway) -> None:
    """The caller posts a chat object containing the non-UTF-8 byte `E9`. The 400 should say the payload is
    invalid JSON with `param: "request_body"`, like every other unparseable body. Today the decode error is
    swallowed and the body is treated as empty (`http_parsing_utils.py:399`), so the caller is told a required
    field is missing instead."""
    pytest.skip("BUG: a non-UTF-8 body is read as an empty object, so the 400 names a missing field, not bad JSON")
    with front_door(gateway) as door:
        body: Final = (
            json.dumps(chat_body(door.chat_model(), "PLACEHOLDER")).encode().replace(b"PLACEHOLDER", b"caf\xe9")
        )
        status, content = _post(gateway, body)
        assert status == 400, content
        assert _without_message(_error(content)) == INVALID_JSON_ERROR


def test_malformed_json_over_the_repair_limit_is_still_a_400(gateway: Gateway) -> None:
    """The caller posts 2 MiB of malformed JSON. Above `MAX_REQUEST_BODY_SIZE_TO_REPAIR_MB` (1 by default) the
    proxy skips its slow repair pass and must answer 400 at once with `param: "request_body"`, and the liveness
    probe must answer 200 straight after. A rewrite must keep large garbage from stalling the server."""
    status, content = _post(gateway, b'{"model": "x", "messages": [' + b'"a",' * (MIB // 2))
    assert status == 400, content[:300]
    assert _without_message(_error(content)) == INVALID_JSON_ERROR
    assert gateway.client.get("/health/liveliness").status_code == 200


@pytest.mark.parametrize(
    "content_type",
    (
        pytest.param("application/json; charset=utf-8", id="json-with-charset"),
        pytest.param("text/plain", id="text-plain"),
        pytest.param("application/octet-stream", id="octet-stream"),
        pytest.param(None, id="no-content-type"),
    ),
)
def test_a_json_body_is_served_whatever_content_type_labels_it(gateway: Gateway, content_type: str | None) -> None:
    """A chat model is registered. The caller posts a valid chat JSON body labelled with a charset parameter, as
    `text/plain`, as `application/octet-stream`, or with no content type at all. The proxy parses anything that
    is not a form or binary media type as JSON, so each must answer 200 and reach the provider once with the
    caller's messages. Only the two protobuf media types count as binary (`http_parsing_utils.py:35`)."""
    with front_door(gateway) as door:
        sent: Final = chat_body(door.chat_model())
        status, content = _post(gateway, json.dumps(sent).encode(), content_type)
        assert status == 200, content
        calls: Final = door.calls()
        assert [_received_messages(call.body) for call in calls] == [sent["messages"]]


def test_a_duplicated_key_resolves_to_its_last_value(gateway: Gateway) -> None:
    """A chat model is registered. The caller posts a body whose `model` key appears twice: first naming a model
    that does not exist, then the registered one. The proxy's JSON parser keeps the last value, so the request
    must be served with 200 by the registered model. A parser that keeps the first value, or rejects duplicates
    as strict deserialisers do, would turn this into a 400."""
    with front_door(gateway) as door:
        model: Final = door.chat_model()
        text: Final = f"duplicate key {uuid.uuid4().hex}"
        body: Final = (
            f'{{"model": "dec1-missing-{uuid.uuid4().hex}", "model": "{model}", '
            f'"messages": [{{"role": "user", "content": "{text}"}}], "cache": {{"no-cache": true}}}}'
        )
        status, content = _post(gateway, body.encode())
        assert status == 200, content
        assert [_received_messages(call.body) for call in door.calls()] == [[{"role": "user", "content": text}]]


def test_unicode_message_text_reaches_the_provider_unchanged(gateway: Gateway) -> None:
    """A chat model is registered. The caller posts a message containing accented Latin, CJK, an emoji outside the
    basic plane and an escaped quote, sent as raw UTF-8 (not `\\u` escapes). The provider must receive a message
    list equal to the caller's after JSON decoding, with no character replaced or dropped."""
    with front_door(gateway) as door:
        text: Final = f'café 東京 \U0001f680 "quoted" {uuid.uuid4().hex}'
        sent: Final = chat_body(door.chat_model(), text)
        status, content = _post(gateway, json.dumps(sent, ensure_ascii=False).encode())
        assert status == 200, content
        assert [_received_messages(call.body) for call in door.calls()] == [sent["messages"]]


def test_a_chunked_request_body_is_reassembled(gateway: Gateway) -> None:
    """A chat model is registered. The caller sends the chat request over a raw socket with `Transfer-Encoding:
    chunked` and no `Content-Length`, splitting the JSON into 16-byte chunks. The proxy must answer 200 and the
    provider must receive the caller's messages intact. Streaming uploaders and some SDKs send bodies this way."""
    with front_door(gateway) as door:
        sent: Final = chat_body(door.chat_model())
        raw: Final = json.dumps(sent).encode()
        pieces: Final = tuple(raw[start : start + 16] for start in range(0, len(raw), 16))
        framed: Final = b"".join(b"%x\r\n%s\r\n" % (len(piece), piece) for piece in pieces) + b"0\r\n\r\n"
        head: Final = request_bytes(
            "POST",
            CHAT,
            (
                ("Host", host_header(gateway)),
                ("Authorization", f"Bearer {gateway.key}"),
                ("Content-Type", "application/json"),
                ("Transfer-Encoding", "chunked"),
                ("Connection", "close"),
            ),
        )
        (response,) = exchange(gateway, head + framed)
        assert response.status == 200, response.body
        assert [_received_messages(call.body) for call in door.calls()] == [sent["messages"]]


def test_a_chunked_request_with_a_chunk_extension_and_trailer_is_reassembled(gateway: Gateway) -> None:
    """The caller sends a chunked chat body in which one chunk carries an extension (`;name=value`) and the
    terminating chunk is followed by a trailer field. Both are legal HTTP/1.1. The proxy must ignore them, answer
    200, and hand the provider the caller's messages. UNVERIFIED: h11 accepts both; a stricter parser might not."""
    with front_door(gateway) as door:
        sent: Final = chat_body(door.chat_model())
        raw: Final = json.dumps(sent).encode()
        framed: Final = (
            b"%x;note=dec1\r\n%s\r\n" % (len(raw[:10]), raw[:10])
            + b"%x\r\n%s\r\n" % (len(raw[10:]), raw[10:])
            + b"0\r\nx-dec1-trailer: 1\r\n\r\n"
        )
        head: Final = request_bytes(
            "POST",
            CHAT,
            (
                ("Host", host_header(gateway)),
                ("Authorization", f"Bearer {gateway.key}"),
                ("Content-Type", "application/json"),
                ("Transfer-Encoding", "chunked"),
                ("Connection", "close"),
            ),
        )
        (response,) = exchange(gateway, head + framed)
        assert response.status == 200, response.body
        assert [_received_messages(call.body) for call in door.calls()] == [sent["messages"]]


def test_expect_100_continue_gets_an_interim_100_before_the_body_is_sent(gateway: Gateway) -> None:
    """A chat model is registered. The caller sends only the request head with `Expect: 100-continue` and waits.
    The proxy must write `HTTP/1.1 100 Continue` before any body bytes arrive, and after the caller then sends
    the body it must answer 200 with the provider's text. curl and .NET clients send this for larger bodies and
    stall for a second or more when the interim response never comes."""
    with front_door(gateway) as door:
        raw: Final = json.dumps(chat_body(door.chat_model())).encode()
        head: Final = json_post(gateway, CHAT, b"", extra=(("Expect", "100-continue"), ("Connection", "close")))
        declared: Final = head.replace(b"Content-Length: 0", b"Content-Length: %d" % len(raw))
        with connection(gateway) as (client, stream):
            client.sendall(declared)
            interim: Final = read_response(stream)
            client.sendall(raw)
            final: Final = read_response(stream)
        assert (interim.status, interim.reason) == (100, "Continue")
        assert final.status == 200, final.body
        assert len(door.calls()) == 1


def test_a_gzip_encoded_request_body_reaches_the_provider_decoded(gateway: Gateway) -> None:
    """A chat model is registered. The caller posts the chat JSON gzip-compressed with `Content-Encoding: gzip`.
    The proxy must answer 200, and the provider must receive plain JSON with the caller's messages and no
    `content-encoding` request header. The owned-proxy suite in `test_compressed_request_bodies.py` covers limits;
    this pins the default configuration."""
    with front_door(gateway) as door:
        sent: Final = chat_body(door.chat_model())
        response: Final = gateway.client.post(
            CHAT,
            content=gzip.compress(json.dumps(sent).encode(), mtime=0),
            headers={
                "Authorization": f"Bearer {gateway.key}",
                "content-type": "application/json",
                "content-encoding": "gzip",
            },
        )
        assert response.status_code == 200, response.text
        calls: Final = door.calls()
        assert [_received_messages(call.body) for call in calls] == [sent["messages"]]
        assert "content-encoding" not in calls[0].headers


def test_an_eight_megabyte_body_is_served_when_no_size_limit_is_configured(gateway: Gateway) -> None:
    """A chat model is registered on the shared proxy, which sets no `max_request_size_mb`. The caller posts a
    chat request whose single message is 8 MiB of text. The proxy must answer 200 and the provider must receive
    a message of exactly that length. The front door has no built-in cap, so a forwarder must not add one."""
    with front_door(gateway) as door:
        text: Final = "x" * (8 * MIB)
        response: Final = gateway.client.post(
            CHAT,
            json=chat_body(door.chat_model(), text),
            headers={"Authorization": f"Bearer {gateway.key}"},
            timeout=120,
        )
        assert response.status_code == 200, response.text[:300]
        calls: Final = door.calls()
        assert len(calls) == 1
        received: Final = object_value(list_value(_received_messages(calls[0].body))[0])
        assert len(str(received["content"])) == 8 * MIB


def test_a_body_shorter_than_its_content_length_never_reaches_a_provider(gateway: Gateway) -> None:
    """A chat model is registered. The caller declares `Content-Length` 64 bytes larger than the JSON it sends,
    then closes its write side. The request is incomplete, so no provider call may be made, and the proxy must
    stay healthy: the liveness probe answers 200 afterwards. The handler sees a client disconnect
    (`http_parsing_utils.py:387`); nothing is written back that the caller could mistake for success."""
    with front_door(gateway) as door:
        raw: Final = json.dumps(chat_body(door.chat_model())).encode()
        payload: Final = json_post(gateway, CHAT, raw).replace(
            b"Content-Length: %d" % len(raw), b"Content-Length: %d" % (len(raw) + 64)
        )
        with connection(gateway, timeout=10) as (client, stream):
            client.sendall(payload)
            client.shutdown(1)
            leftover: Final = stream.read()
        assert not leftover.startswith(b"HTTP/1.1 200"), leftover[:200]
        assert gateway.client.get("/health/liveliness").status_code == 200
        assert door.calls() == ()


def test_extra_bytes_after_the_json_object_are_a_400(gateway: Gateway) -> None:
    """The caller posts a complete chat object followed by a second JSON object in the same body. Concatenated
    documents are not one JSON value, so the proxy must answer 400 with `param: "request_body"` and call no
    provider, rather than serving the first object and ignoring the rest."""
    with front_door(gateway) as door:
        raw: Final = json.dumps(chat_body(door.chat_model())).encode()
        status, content = _post(gateway, raw + b'{"model": "second"}')
        assert status == 400, content
        assert _without_message(_error(content)) == INVALID_JSON_ERROR
        assert door.calls() == ()
