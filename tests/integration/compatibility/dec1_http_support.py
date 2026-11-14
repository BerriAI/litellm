"""Wire-level helpers shared by the `test_dec1_http_*` front-door contracts.

Two things live here. `front_door` and `owned_front_door` put a recording HTTP peer behind the proxy so a test
can read the exact request the provider received, headers included, which the shared scripted upstream does not
record. `exchange`, `read_response` and friends speak HTTP/1.1 over a plain socket so a test can pin framing,
connection reuse and header bytes that `httpx` would normalise away. A Rust forwarder placed in front of the
Python proxy has to keep every observation made through these helpers identical.
"""

from __future__ import annotations

import json
import socket
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, Final

import yaml
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
)
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, chunked_body, exactly, wire_server
from pydantic import JsonValue

CHAT: Final = "/v1/chat/completions"
OWNED_CHAT_MODEL: Final = "dec1-http-chat"
OWNED_EMBEDDING_MODEL: Final = "dec1-http-embedding"
OWNED_MESSAGES_MODEL: Final = "dec1-http-messages"
ANTHROPIC_MODEL: Final = "claude-haiku-5-5"
NO_HEADERS: Final[Mapping[str, str]] = MappingProxyType({})
NO_SETTINGS: Final[Mapping[str, JsonValue]] = MappingProxyType({})
SPEND_ROWS: Final = (
    "SELECT request_id, call_type, spend, model, end_user, request_tags, requester_ip_address, metadata, status, "
    'litellm_call_id FROM "LiteLLM_SpendLogs" WHERE api_key = %s ORDER BY "startTime"'
)
Respond = Callable[[Request], Reply]


def chat_completion(text: str = "scripted") -> bytes:
    return json.dumps(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
    ).encode()


def stream_frame(identity: str, delta: Mapping[str, JsonValue], finish_reason: str | None) -> bytes:
    chunk: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish_reason}],
    }
    return b"data: " + json.dumps(chunk).encode() + b"\n\n"


def chat_stream_chunks(texts: Sequence[str]) -> tuple[bytes, ...]:
    identity: Final = f"chatcmpl-{uuid.uuid4().hex}"
    return (
        stream_frame(identity, {"role": "assistant", "content": texts[0]}, None),
        *(stream_frame(identity, {"content": text}, None) for text in texts[1:]),
        stream_frame(identity, {}, "stop"),
        b"data: [DONE]\n\n",
    )


def text_completion() -> bytes:
    return json.dumps(
        {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": 1,
            "model": "gpt-3.5-turbo-instruct",
            "choices": [{"text": "scripted", "index": 0, "logprobs": None, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
    ).encode()


def embedding() -> bytes:
    return json.dumps(
        {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.5, 0.75]}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
    ).encode()


def anthropic_message() -> bytes:
    return json.dumps(
        {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": ANTHROPIC_MODEL,
            "content": [{"type": "text", "text": "scripted"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }
    ).encode()


def responses_object() -> bytes:
    identity: Final = uuid.uuid4().hex
    return json.dumps(
        {
            "id": f"resp_{identity}",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-4o-mini",
            "output": [
                {
                    "type": "message",
                    "id": f"msg_{identity}",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        }
    ).encode()


def provider_error(status: int, message: str = "scripted provider failure") -> Reply:
    kind: Final = "rate_limit_error" if status == 429 else "invalid_request_error" if status < 500 else "server_error"
    return Reply(
        status=status,
        body=json.dumps({"error": {"message": message, "type": kind, "param": None, "code": str(status)}}).encode(),
    )


def path_of(request: Request) -> str:
    return request.target.split("?", 1)[0]


def scripted_reply(request: Request) -> Reply:
    """What a healthy provider answers on each surface the front-door contracts drive."""
    path: Final = path_of(request)
    if request.method != "POST":
        return Reply(body=b'{"object":"list","data":[]}')
    if path.endswith("/embeddings"):
        return Reply(body=embedding())
    if path.endswith("/messages"):
        return Reply(body=anthropic_message())
    if path.endswith("/responses"):
        return Reply(body=responses_object())
    if not path.endswith("/chat/completions"):
        return Reply(body=text_completion())
    if JSON_OBJECT.validate_json(request.body).get("stream") is True:
        return Reply(content_type="text/event-stream", chunks=chat_stream_chunks(("streamed ", "answer")))
    return Reply(body=chat_completion())


def chat_body(model: str, text: str | None = None, **fields: JsonValue) -> dict[str, JsonValue]:
    """A chat request no response cache can answer, so every call reaches the recording peer."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": text or f"dec1 http {uuid.uuid4().hex}"}],
        "cache": {"no-cache": True},
        **fields,
    }


def posts(upstream: Wire) -> tuple[Request, ...]:
    return tuple(request for request in upstream.drain() if request.method == "POST")


@dataclass(frozen=True, slots=True)
class FrontDoor:
    gateway: Gateway
    scenario: Scenario
    upstream: Wire

    def chat_model(self, **parameters: JsonValue) -> str:
        """A chat deployment on the recording peer; `parameters` override any litellm_params, `api_base` included."""
        return self.scenario.model(**{"api_base": f"{self.upstream.url}/v1", **parameters})

    def completion_model(self) -> str:
        return self.scenario.model(
            model="text-completion-openai/gpt-3.5-turbo-instruct", api_base=f"{self.upstream.url}/v1"
        )

    def embedding_model(self) -> str:
        return self.scenario.model(model="openai/text-embedding-3-small", api_base=f"{self.upstream.url}/v1")

    def messages_model(self) -> str:
        return self.scenario.model(model=f"anthropic/{ANTHROPIC_MODEL}", api_base=self.upstream.url)

    def calls(self) -> tuple[Request, ...]:
        return posts(self.upstream)


@contextmanager
def front_door(gateway: Gateway, respond: Respond = scripted_reply) -> Iterator[FrontDoor]:
    """The shared proxy with a recording provider behind models this test registers and removes."""
    with wire_server(respond) as upstream, gateway.scenario() as scenario:
        yield FrontDoor(gateway, scenario, upstream)


@dataclass(frozen=True, slots=True)
class OwnedDoor:
    gateway: Gateway
    upstream: Wire

    def calls(self) -> tuple[Request, ...]:
        return posts(self.upstream)


def _deployment(name: str, model: str, api_base: str) -> Mapping[str, JsonValue]:
    return {"model_name": name, "litellm_params": {"model": model, "api_base": api_base, "api_key": "sk-fixture"}}


def owned_config(
    directory: Path,
    upstream: Wire,
    *,
    general_settings: Mapping[str, JsonValue] = NO_SETTINGS,
    litellm_settings: Mapping[str, JsonValue] = NO_SETTINGS,
) -> Path:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    configuration: Final = {
        **base,
        "model_list": [
            _deployment(OWNED_CHAT_MODEL, "openai/gpt-4o-mini", f"{upstream.url}/v1"),
            _deployment(OWNED_EMBEDDING_MODEL, "openai/text-embedding-3-small", f"{upstream.url}/v1"),
            _deployment(OWNED_MESSAGES_MODEL, f"anthropic/{ANTHROPIC_MODEL}", upstream.url),
        ],
        "general_settings": {**object_value(base.get("general_settings", {})), **general_settings},
        "litellm_settings": {**object_value(base.get("litellm_settings", {})), **litellm_settings},
        "router_settings": {**object_value(base.get("router_settings", {})), "num_retries": 0},
    }
    path: Final = directory / f"dec1-http-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@contextmanager
def owned_front_door(
    directory: Path,
    *,
    general_settings: Mapping[str, JsonValue] = NO_SETTINGS,
    litellm_settings: Mapping[str, JsonValue] = NO_SETTINGS,
    environment: Mapping[str, str] = NO_HEADERS,
    respond: Respond = scripted_reply,
) -> Iterator[OwnedDoor]:
    """A proxy this module boots with its own settings, serving three models from a recording provider."""
    with gateway_from_environment() as root, wire_server(respond) as upstream:
        config: Final = owned_config(
            directory, upstream, general_settings=general_settings, litellm_settings=litellm_settings
        )
        with owned_proxy_process(root, directory, dict(environment), config=config, workers=1) as owned:
            yield OwnedDoor(owned.gateway, upstream)


def spend_rows(gateway: Gateway, key: str, *, count: int = 1, seconds: float = 70) -> list[dict[str, JsonValue]]:
    """The spend rows a virtual key has written, once at least `count` of them have landed."""
    hashed: Final = sha256(key.encode()).hexdigest()
    return eventually(lambda: gateway.rows(SPEND_ROWS, (hashed,)), lambda rows: len(rows) >= count, seconds=seconds)


def address(gateway: Gateway) -> tuple[str, int]:
    url: Final = gateway.client.base_url
    return url.host, url.port or 80


def host_header(gateway: Gateway) -> str:
    host, port = address(gateway)
    return f"{host}:{port}"


@dataclass(frozen=True, slots=True)
class RawResponse:
    version: str
    status: int
    reason: str
    headers: tuple[tuple[str, str], ...]
    body: bytes

    def values(self, name: str) -> tuple[str, ...]:
        return tuple(value for header, value in self.headers if header == name.lower())

    def header(self, name: str) -> str | None:
        return next(iter(self.values(name)), None)

    def json(self) -> dict[str, JsonValue]:
        return JSON_OBJECT.validate_json(self.body)


def _header_lines(stream: BinaryIO) -> Iterator[tuple[str, str]]:
    while True:
        line = stream.readline()
        if line in (b"\r\n", b"\n", b""):
            return
        name, _, value = line.decode("latin-1").partition(":")
        yield name.strip().lower(), value.strip()


def read_response(stream: BinaryIO, *, bodiless: bool = False) -> RawResponse:
    """One HTTP/1.x response off the socket, body framed the way the server said it framed it."""
    status_line: Final = stream.readline().decode("latin-1").rstrip("\r\n")
    assert status_line, "The server closed the connection without answering"
    version, _, rest = status_line.partition(" ")
    code, _, reason = rest.partition(" ")
    headers: Final = tuple(_header_lines(stream))
    fields: Final = dict(headers)
    status: Final = int(code)
    if bodiless or status < 200 or status in (204, 304):
        return RawResponse(version, status, reason, headers, b"")
    if "chunked" in fields.get("transfer-encoding", "").lower():
        return RawResponse(version, status, reason, headers, b"".join(chunked_body(stream)))
    if "content-length" in fields:
        return RawResponse(version, status, reason, headers, exactly(stream, int(fields["content-length"])))
    return RawResponse(version, status, reason, headers, stream.read())


def request_bytes(
    method: str,
    target: str,
    headers: Sequence[tuple[str, str]],
    body: bytes = b"",
    *,
    version: str = "HTTP/1.1",
) -> bytes:
    head: Final = "".join(f"{name}: {value}\r\n" for name, value in headers)
    return f"{method} {target} {version}\r\n{head}\r\n".encode("latin-1") + body


def json_post(
    gateway: Gateway,
    target: str,
    body: bytes,
    *,
    key: str | None = None,
    extra: Sequence[tuple[str, str]] = (),
    version: str = "HTTP/1.1",
) -> bytes:
    """A complete POST with Host, bearer key, JSON content type and an exact Content-Length."""
    return request_bytes(
        "POST",
        target,
        (
            ("Host", host_header(gateway)),
            ("Authorization", f"Bearer {gateway.key if key is None else key}"),
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(body))),
            *extra,
        ),
        body,
        version=version,
    )


@contextmanager
def connection(gateway: Gateway, timeout: float = 30) -> Iterator[tuple[socket.socket, BinaryIO]]:
    with socket.create_connection(address(gateway), timeout=timeout) as client, client.makefile("rb") as stream:
        yield client, stream


def exchange(
    gateway: Gateway, payload: bytes, *, responses: int = 1, bodiless: bool = False, timeout: float = 30
) -> tuple[RawResponse, ...]:
    """Write `payload` on one fresh connection and read `responses` answers back in order."""
    with connection(gateway, timeout) as (client, stream):
        client.sendall(payload)
        return tuple(read_response(stream, bodiless=bodiless) for _ in range(responses))


def peer_closed(client: socket.socket, stream: BinaryIO, within: float = 5) -> bool:
    """Whether the server ends the connection within `within` seconds of the last response."""
    client.settimeout(within)
    try:
        return stream.read(1) == b""
    except TimeoutError:
        return False
