import json
import struct
import uuid
import zlib
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from integration._support.wire import Reply, Request, Wire
from pydantic import JsonValue, TypeAdapter

ENDPOINT: Final = "integration-vllm-endpoint"
SERVED_MODEL: Final = "integration-org/served-chat-model"
ACCESS_KEY: Final = "AKIAINTEGRATION000003"
SECRET_KEY: Final = "synthetic-secret-key-for-testing"
REGION: Final = "us-east-1"
PROMPT: Final = "synthetic inference component request"
CONTENT: Final = ("sagemaker ", "wire control")
NO_CACHE: Final[Mapping[str, JsonValue]] = MappingProxyType({"cache": {"no-cache": True}})
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MARKER_PREFIX: Final = "marker-"


def answered_choices() -> list[JsonValue]:
    return [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {"role": "assistant", "content": "".join(CONTENT)},
            "provider_specific_fields": {},
        }
    ]


def _completion(identity: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "chat.completion",
            "created": 1,
            "model": SERVED_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(CONTENT)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
        }
    ).encode()


def _chunk(identity: str, delta: Mapping[str, JsonValue], finish_reason: str | None, **extra: JsonValue) -> bytes:
    payload: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": SERVED_MODEL,
        "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish_reason}],
        **extra,
    }
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _event_header(name: str, value: str) -> bytes:
    name_bytes: Final = name.encode()
    value_bytes: Final = value.encode()
    return struct.pack("!B", len(name_bytes)) + name_bytes + b"\x07" + struct.pack("!H", len(value_bytes)) + value_bytes


def _payload_part(body: bytes) -> bytes:
    headers: Final = (
        _event_header(":event-type", "PayloadPart")
        + _event_header(":content-type", "application/octet-stream")
        + _event_header(":message-type", "event")
    )
    prelude: Final = struct.pack("!II", 12 + len(headers) + len(body) + 4, len(headers))
    message: Final = prelude + struct.pack("!I", zlib.crc32(prelude) & 0xFFFFFFFF) + headers + body
    return message + struct.pack("!I", zlib.crc32(message) & 0xFFFFFFFF)


def container_reply(identity: str, stream: bool) -> Reply:
    if not stream:
        return Reply(body=_completion(identity))
    chunks: Final = (
        _chunk(identity, {"role": "assistant", "content": CONTENT[0]}, None),
        _chunk(identity, {"content": CONTENT[1]}, None),
        _chunk(identity, {}, "stop", usage={"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}),
    )
    return Reply(
        chunks=tuple(_payload_part(chunk) for chunk in chunks), content_type="application/vnd.amazon.eventstream"
    )


def marked_prompt() -> str:
    return f"{PROMPT} {_MARKER_PREFIX}{uuid.uuid4().hex}"


def marker(text: str) -> str:
    _, separator, rest = text.partition(_MARKER_PREFIX)
    assert separator and text.count(_MARKER_PREFIX) == 1, text
    return rest[:32]


def container_body(url: str, prompt: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "model": ENDPOINT,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        "sagemaker_base_url": url,
        **extra,
    }


def only_request(wire: Wire) -> Request:
    (request,) = wire.drain()
    assert (request.method, request.target) == ("POST", "/"), request
    return request


def by_marker(received: tuple[Request, ...]) -> Mapping[str, Request]:
    forwarded: Final = {marker(request.body.decode()): request for request in received}
    assert len(forwarded) == len(received), [request.body for request in received]
    return forwarded


def assert_signed(request: Request, access_key: str, region: str, session_token: str | None) -> None:
    authorization: Final = request.headers["authorization"]
    assert authorization.startswith(f"AWS4-HMAC-SHA256 Credential={access_key}/"), authorization
    assert f"/{region}/sagemaker/aws4_request" in authorization, authorization
    assert request.headers.get("x-amz-security-token") == session_token, dict(request.headers)
