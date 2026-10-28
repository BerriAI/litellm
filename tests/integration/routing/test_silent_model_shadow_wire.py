from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import unquote

import httpx
import pytest
from integration._support.anthropic_sse import parse_sse
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.openai_wire import chat_reply
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

Shape = Literal["single", "pair"]

_MODEL: Final = "gpt-5.6"
_API_KEY: Final = "synthetic-shadow-key"
_ROUTER_ONLY_KEYS: Final = ("silent_model", "include_fallback_errors")
_ANSWER: Final = "answered by"
_NONCE: Final = re.compile(r"nonce=([0-9a-f]{32})")
_JSON: Final = TypeAdapter(dict[str, JsonValue])
_CHOICES: Final = TypeAdapter(list[dict[str, JsonValue]])
_FRESH_CONNECTION: Final = MappingProxyType({"Connection": "close"})
_SPEND_ROW_SECONDS: Final = 70
_TARGETS: Final[Mapping[Shape, tuple[str, ...]]] = MappingProxyType(
    {"single": ("shadow-a",), "pair": ("shadow-a", "shadow-b")}
)
_CASES: Final = (
    pytest.param("single", False, id="non-stream-single"),
    pytest.param("single", True, id="stream-single"),
    pytest.param("pair", True, id="stream-pair"),
)


def _prompt(nonce: str) -> str:
    return f"which deployment answers this? nonce={nonce}"


def _nonce_of(raw: str) -> str:
    found: Final = _NONCE.search(raw)
    assert found is not None, raw
    return found.group(1)


def _identity(segment: str, nonce: str) -> str:
    return f"chatcmpl-{segment}-{nonce}"


def _request(model: str, nonce: str, *, stream: bool) -> dict[str, JsonValue]:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "stream": stream,
        "messages": [{"role": "user", "content": _prompt(nonce)}],
    }
    return {**body, "stream_options": {"include_usage": True}} if stream else body


def _segment_of(request: Request) -> str:
    return unquote(request.target).lstrip("/").partition("/")[0]


def _peer(request: Request) -> Reply:
    if request.method == "GET":
        return Reply(body=json.dumps({"object": "list", "data": []}).encode())
    segment: Final = _segment_of(request)
    body: Final = _JSON.validate_json(request.body)
    return chat_reply(
        _identity(segment, _nonce_of(request.body.decode())),
        _MODEL,
        f"{_ANSWER} {segment}",
        stream=body.get("stream") is True,
    )


@dataclass(frozen=True, slots=True)
class _Posted:
    segment: str
    streamed: bool
    leaked: tuple[str, ...]


def _posted(wire: Wire, nonce: str) -> tuple[_Posted, ...]:
    return tuple(
        _Posted(
            _segment_of(request),
            _JSON.validate_json(request.body).get("stream") is True,
            tuple(key for key in _ROUTER_ONLY_KEYS if key in request.body.decode()),
        )
        for request in wire.drain()
        if request.method == "POST" and nonce in request.body.decode()
    )


def _delta_text(data: Mapping[str, JsonValue]) -> str:
    deltas: Final = (
        _JSON.validate_python(choice.get("delta") or {}) for choice in _CHOICES.validate_python(data["choices"])
    )
    return "".join(str(delta.get("content") or "") for delta in deltas)


def _served(response: httpx.Response, *, stream: bool) -> tuple[frozenset[str], str]:
    if not stream:
        body: Final = _JSON.validate_json(response.content)
        message: Final = _JSON.validate_python(_CHOICES.validate_python(body["choices"])[0]["message"])
        return frozenset({str(body["id"])}), str(message["content"])
    events: Final = tuple(event.data for event in parse_sse(response.text))
    return frozenset(str(data["id"]) for data in events), "".join(_delta_text(data) for data in events)


@dataclass(frozen=True, slots=True)
class _SpendRow:
    request_id: str
    model_group: str
    spend: float


def _spend_row(gateway: Gateway, request_id: str) -> tuple[_SpendRow, ...]:
    rows: Final = gateway.rows(
        'SELECT model_group, spend FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (request_id,)
    )
    return tuple(_SpendRow(request_id, str(row["model_group"]), float(str(row["spend"]))) for row in rows)


def _spend_rows(gateway: Gateway, request_ids: tuple[str, ...]) -> tuple[_SpendRow, ...]:
    return tuple(row for request_id in request_ids for row in _spend_row(gateway, request_id))


@dataclass(frozen=True, slots=True)
class _Mirrors:
    gateway: Gateway
    wire: Wire
    groups: Mapping[str, str]


@pytest.fixture(scope="module")
def mirrors() -> Iterator[_Mirrors]:
    with gateway_from_environment() as gateway, wire_server(_peer) as wire, gateway.scenario() as scenario:

        def register(segment: str, silent_model: JsonValue) -> str:
            return scenario.model(
                model=f"openai/{_MODEL}", api_key=_API_KEY, api_base=f"{wire.url}/{segment}", silent_model=silent_model
            )

        shadow_a: Final = register("shadow-a", None)
        shadow_b: Final = register("shadow-b", None)
        groups: Final = MappingProxyType(
            {
                "shadow-a": shadow_a,
                "shadow-b": shadow_b,
                "primary-single": register("primary-single", shadow_a),
                "primary-pair": register("primary-pair", [shadow_a, shadow_b]),
            }
        )
        wire.drain()
        yield _Mirrors(gateway, wire, groups)


@pytest.mark.parametrize(("shape", "stream"), _CASES)
def test_each_shadow_target_gets_one_request_in_the_callers_stream_mode(
    mirrors: _Mirrors, shape: Shape, stream: bool
) -> None:
    nonce: Final = uuid.uuid4().hex
    primary: Final = f"primary-{shape}"
    targets: Final = _TARGETS[shape]
    response: Final = mirrors.gateway.request(
        "POST",
        "/v1/chat/completions",
        _request(mirrors.groups[primary], nonce, stream=stream),
        headers=_FRESH_CONNECTION,
    )
    assert response.status_code == 200, response.text
    assert _served(response, stream=stream) == (frozenset({_identity(primary, nonce)}), f"{_ANSWER} {primary}")

    expected_groups: Final = MappingProxyType(
        {_identity(segment, nonce): mirrors.groups[segment] for segment in (primary, *targets)}
    )
    rows: Final = eventually(
        lambda: _spend_rows(mirrors.gateway, tuple(expected_groups)),
        lambda found: len(found) == len(expected_groups),
        seconds=_SPEND_ROW_SECONDS,
    )
    assert {row.request_id: row.model_group for row in rows} == expected_groups, rows
    assert all(row.spend > 0 for row in rows), rows

    posted: Final = _posted(mirrors.wire, nonce)
    assert sorted(item.segment for item in posted) == sorted((primary, *targets)), posted
    assert all(item.streamed is stream for item in posted), posted
    assert all(item.leaked == () for item in posted), posted
