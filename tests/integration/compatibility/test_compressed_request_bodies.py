from __future__ import annotations

import gzip
import json
import os
import secrets
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import yaml
from integration._support.budget import case_timeout_seconds
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    gateway_from_environment,
    list_value,
    object_value,
    string_value,
)
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

MIB: Final = 1024 * 1024
CHAT: Final = "/v1/chat/completions"
CHAT_MODEL: Final = "compressed-chat"
OTHER_MODEL: Final = "compressed-chat-other"
ANTHROPIC_MODEL: Final = "claude-haiku-5-5"
GATED_PREFIX: Final = "gated-"
ORIGIN: Final = "http://browser.example"
MIDDLEWARE_HEADERS: Final = (
    "access-control-allow-origin",
    "x-frame-options",
    "content-security-policy",
    "x-content-type-options",
)
ZSTD_MAGIC: Final = b"\x28\xb5\x2f\xfd"
ZSTD_SINGLE_SEGMENT_FOUR_BYTE_SIZE: Final = b"\xa0"
ZSTD_MAX_BLOCK: Final = 128 * 1024
NO_HEADERS: Final[Mapping[str, str]] = MappingProxyType({})

pytestmark: Final = [
    pytest.mark.timeout(2 * graceful_stop_seconds() + 120),
    pytest.mark.xdist_group("compressed-request-bodies"),
]


def zstd_frame(data: bytes) -> bytes:
    blocks: Final = tuple(data[start : start + ZSTD_MAX_BLOCK] for start in range(0, len(data), ZSTD_MAX_BLOCK))
    framed: Final = blocks or (b"",)
    return (
        ZSTD_MAGIC
        + ZSTD_SINGLE_SEGMENT_FOUR_BYTE_SIZE
        + len(data).to_bytes(4, "little")
        + b"".join(
            (len(block) << 3 | (index == len(framed) - 1)).to_bytes(3, "little") + block
            for index, block in enumerate(framed)
        )
    )


def gzip_body(data: bytes) -> bytes:
    return gzip.compress(data, mtime=0)


def as_sent(data: bytes) -> bytes:
    return data


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    upstream: Wire
    gate: threading.Event
    limited_key: str

    def calls(self) -> tuple[Request, ...]:
        return tuple(request for request in self.upstream.drain() if request.method == "POST")


def _chat_completion() -> bytes:
    return json.dumps(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "scripted"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
    ).encode()


def _stream_frame(identity: str, delta: Mapping[str, JsonValue], finish_reason: str | None) -> bytes:
    chunk: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish_reason}],
    }
    return b"data: " + json.dumps(chunk).encode() + b"\n\n"


def _chat_stream(gate: threading.Event | None) -> Reply:
    identity: Final = f"chatcmpl-{uuid.uuid4().hex}"
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _stream_frame(identity, {"role": "assistant", "content": "streamed "}, None),
            _stream_frame(identity, {"content": "answer"}, "stop"),
            b"data: [DONE]\n\n",
        ),
        gate_after_first=gate,
    )


def _anthropic_message() -> bytes:
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


def _respond(gate: threading.Event, request: Request) -> Reply:
    if request.method == "GET":
        return Reply(body=b'{"object":"list","data":[]}')
    if request.target == "/v1/messages":
        return Reply(body=_anthropic_message())
    body: Final = JSON_OBJECT.validate_json(request.body)
    if body.get("stream") is not True:
        return Reply(body=_chat_completion())
    text: Final = string_value(object_value(list_value(body["messages"])[0])["content"])
    return _chat_stream(gate if text.startswith(GATED_PREFIX) else None)


def _deployment(name: str, upstream: Wire) -> Mapping[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {"model": "openai/gpt-4o-mini", "api_base": f"{upstream.url}/v1", "api_key": "sk-fixture"},
    }


def _config(directory: Path, upstream: Wire) -> Path:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    configuration: Final = {
        **base,
        "model_list": [_deployment(CHAT_MODEL, upstream), _deployment(OTHER_MODEL, upstream)],
        "general_settings": {
            **object_value(base.get("general_settings", {})),
            "max_request_size_mb": 1,
            "enable_claude_code_gateway": True,
        },
        "router_settings": {**object_value(base.get("router_settings", {})), "num_retries": 0},
    }
    path: Final = directory / "compressed-request-bodies.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    assert os.environ.get("LITELLM_LICENSE"), (
        "LITELLM_LICENSE must be set: max_request_size_mb is an enterprise feature"
    )
    gate: Final = threading.Event()
    directory: Final = tmp_path_factory.mktemp("compressed-request-bodies")
    with gateway_from_environment() as root, wire_server(partial(_respond, gate)) as upstream:
        overrides: Final = {
            "ANTHROPIC_API_BASE": upstream.url,
            "ANTHROPIC_API_KEY": "sk-ant-synthetic",
            "MAX_DECOMPRESSED_REQUEST_BODIES_IN_FLIGHT_MB": "1",
        }
        with (
            owned_proxy_process(root, directory, overrides, config=_config(directory, upstream), workers=1) as owned,
            owned.gateway.scenario() as scenario,
        ):
            yield Rig(owned.gateway, upstream, gate, scenario.key(models=[CHAT_MODEL]))


@pytest.fixture(autouse=True)
def empty_upstream_log(rig: Rig) -> None:
    rig.upstream.drain()


def _headers(key: str, coding: str | None, extra: Mapping[str, str] = NO_HEADERS) -> Mapping[str, str]:
    return {
        "Authorization": f"Bearer {key}",
        "content-type": "application/json",
        **({"content-encoding": coding} if coding is not None else {}),
        **extra,
    }


def _send(
    rig: Rig,
    data: bytes,
    coding: str | None,
    *,
    key: str | None = None,
    extra: Mapping[str, str] = NO_HEADERS,
) -> httpx.Response:
    return rig.gateway.client.post(
        CHAT,
        content=data,
        headers=_headers(rig.gateway.key if key is None else key, coding, extra),
        timeout=case_timeout_seconds(),
    )


def _chat_body(model: str, text: str, *, stream: bool = False) -> bytes:
    return json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "stream": stream,
            "cache": {"no-cache": True},
        }
    ).encode()


def _error_type(response: httpx.Response) -> JsonValue:
    return object_value(JSON_OBJECT.validate_json(response.content)["error"])["type"]


def _answer(response: httpx.Response) -> JsonValue:
    choice: Final = object_value(list_value(JSON_OBJECT.validate_json(response.content)["choices"])[0])
    return object_value(choice["message"])["content"]


def _stream(gateway: Gateway, data: bytes, first_text: threading.Event) -> tuple[int, str]:
    with (
        httpx.Client(base_url=gateway.client.base_url, timeout=case_timeout_seconds(), trust_env=False) as client,
        client.stream("POST", CHAT, content=data, headers=_headers(gateway.key, "gzip")) as response,
    ):
        pieces: Final = response.iter_text()
        first: Final = next(pieces, "")
        first_text.set()
        return response.status_code, first + "".join(pieces)


CODINGS: Final = (
    pytest.param("zstd", zstd_frame, id="zstd"),
    pytest.param("gzip", gzip_body, id="gzip"),
    pytest.param("gzip", as_sent, id="plain-labeled-gzip"),
)


@pytest.mark.parametrize(("coding", "encode"), CODINGS)
def test_a_compressed_chat_body_reaches_the_provider_as_its_plain_twin(
    rig: Rig, coding: str, encode: Callable[[bytes], bytes]
) -> None:
    body: Final = _chat_body(CHAT_MODEL, f"twin {uuid.uuid4().hex}")
    plain: Final = _send(rig, body, None)
    compressed: Final = _send(rig, encode(body), coding)
    assert plain.status_code == 200, plain.text
    assert compressed.status_code == 200, compressed.text
    assert _answer(compressed) == _answer(plain)
    assert compressed.headers["x-litellm-response-cost"] == plain.headers["x-litellm-response-cost"]
    calls: Final = rig.calls()
    assert len(calls) == 2, [call.target for call in calls]
    assert JSON_OBJECT.validate_json(calls[1].body) == JSON_OBJECT.validate_json(calls[0].body)
    assert "content-encoding" not in calls[1].headers, calls[1].headers


def test_auth_reads_the_model_from_a_compressed_body(rig: Rig) -> None:
    refused: Final = _send(rig, zstd_frame(_chat_body(OTHER_MODEL, "outside the key")), "zstd", key=rig.limited_key)
    assert refused.status_code == 403, refused.text
    assert _error_type(refused) == "key_model_access_denied", refused.text
    assert rig.calls() == ()
    served: Final = _send(rig, zstd_frame(_chat_body(CHAT_MODEL, "inside the key")), "zstd", key=rig.limited_key)
    assert served.status_code == 200, served.text
    assert len(rig.calls()) == 1


@pytest.mark.parametrize(
    ("coding", "encode"),
    (pytest.param("zstd", zstd_frame, id="zstd"), pytest.param("gzip", as_sent, id="plain-labeled-gzip")),
)
def test_a_compressed_pass_through_body_reaches_the_vendor_decoded(
    rig: Rig, coding: str, encode: Callable[[bytes], bytes]
) -> None:
    body: Final = json.dumps(
        {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 8,
            "messages": [{"role": "user", "content": f"pass through {uuid.uuid4().hex}"}],
        }
    ).encode()
    response: Final = rig.gateway.client.post(
        "/anthropic/v1/messages",
        content=encode(body),
        headers={
            "x-litellm-api-key": rig.gateway.key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "content-encoding": coding,
        },
        timeout=case_timeout_seconds(),
    )
    assert response.status_code == 200, response.text
    calls: Final = rig.calls()
    assert [call.target for call in calls] == ["/v1/messages"]
    assert JSON_OBJECT.validate_json(calls[0].body) == JSON_OBJECT.validate_json(body)
    assert "content-encoding" not in calls[0].headers, calls[0].headers


def test_a_gzip_bomb_gets_a_413_with_the_headers_other_errors_carry(rig: Rig) -> None:
    origin: Final = {"origin": ORIGIN}
    rejected: Final = _send(rig, gzip_body(b"\0" * 800_000), "gzip", extra=origin)
    unauthorized: Final = _send(
        rig, _chat_body(CHAT_MODEL, "unknown key"), None, key=f"sk-unknown-{uuid.uuid4().hex}", extra=origin
    )
    assert rejected.status_code == 413, rejected.text
    assert unauthorized.status_code == 401, unauthorized.text
    assert unauthorized.headers.get("access-control-allow-origin") is not None, dict(unauthorized.headers)
    assert {name: rejected.headers.get(name) for name in MIDDLEWARE_HEADERS} == {
        name: unauthorized.headers.get(name) for name in MIDDLEWARE_HEADERS
    }
    assert rig.calls() == ()


def test_the_operator_size_limit_holds_a_compressed_body_to_its_plain_twins_limit(rig: Rig) -> None:
    over: Final = _chat_body(CHAT_MODEL, secrets.token_hex(600_000))
    compressed_over: Final = gzip_body(over)
    assert len(compressed_over) < MIB < len(over)
    plain: Final = _send(rig, over, None)
    rejected: Final = _send(rig, compressed_over, "gzip")
    served: Final = _send(rig, gzip_body(_chat_body(CHAT_MODEL, secrets.token_hex(300_000))), "gzip")
    assert plain.status_code == 413, plain.text
    assert rejected.status_code == plain.status_code, rejected.text
    assert served.status_code == 200, served.text
    assert len(rig.calls()) == 1


def test_the_decoded_body_budget_turns_away_a_second_body_until_the_stream_ends(rig: Rig) -> None:
    held: Final = gzip_body(_chat_body(CHAT_MODEL, GATED_PREFIX + secrets.token_hex(300_000), stream=True))
    second: Final = gzip_body(_chat_body(CHAT_MODEL, secrets.token_hex(300_000)))
    keyless: Final = _send(rig, second, "gzip", key=f"sk-unknown-{uuid.uuid4().hex}")
    assert keyless.status_code == 401, keyless.text
    first_text: Final = threading.Event()
    with ThreadPoolExecutor(1) as pool:
        streaming: Final = pool.submit(_stream, rig.gateway, held, first_text)
        try:
            eventually(lambda: first_text.is_set() or streaming.done(), bool, seconds=case_timeout_seconds())
            reached: Final = rig.calls()
            turned_away: Final = _send(rig, second, "gzip")
        finally:
            rig.gate.set()
        status, streamed = streaming.result(timeout=case_timeout_seconds())
    assert len(reached) == 1, streamed
    assert turned_away.status_code == 503, turned_away.text
    assert _error_type(turned_away) == "overloaded_error", turned_away.text
    assert turned_away.headers.get("retry-after") == "1", dict(turned_away.headers)
    assert status == 200, streamed
    assert "streamed " in streamed and "answer" in streamed and "data: [DONE]" in streamed, streamed
    assert rig.calls() == ()
    after_the_stream: Final = _send(rig, second, "gzip")
    assert after_the_stream.status_code == 200, after_the_stream.text
    assert len(rig.calls()) == 1


def test_a_claude_code_otlp_export_keeps_a_two_member_gzip_body(rig: Rig) -> None:
    response: Final = rig.gateway.client.post(
        "/claude_code_gateway/v1/metrics",
        content=gzip_body(b"first member") + gzip_body(b"second member"),
        headers={
            "Authorization": f"Bearer {rig.gateway.key}",
            "content-type": "application/x-protobuf",
            "content-encoding": "gzip",
        },
        timeout=case_timeout_seconds(),
    )
    assert response.status_code == 200, response.text
