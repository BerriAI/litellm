import json
import threading
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment, list_value, object_value
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_MODEL: Final = "background-scan-chat"
_CLAUDE: Final = "background-scan-claude"
_RATE: Final = 5
_TRIGGER: Final = "FORBIDDEN"
_DENIAL: Final = "Output withheld by policy."
_CLEAN: Final = tuple(f"w{index} " for index in range(40))
_LEAKY: Final = ("w0 ", f"{_TRIGGER} ", *_CLEAN[2:])
_OUTAGE: Final = ("w0 ", "OUTAGE ", *_CLEAN[2:])
_HOLD_SECONDS: Final = 8.0
_ALLOWED: Final = json.dumps({"action": "NONE", "outputs": [], "assessments": []}).encode()
_INTERVENED: Final = json.dumps(
    {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": _DENIAL}],
        "assessments": [{"topicPolicy": {"topics": [{"name": "forbidden", "type": "DENY", "action": "BLOCKED"}]}}],
    }
).encode()


@dataclass(slots=True)
class _Scans:
    """The scripted ApplyGuardrail peer holds each output scan until `release` is set or the hold runs out."""

    release: threading.Event = field(default_factory=threading.Event)
    started_at: list[float] = field(default_factory=list)
    replied_at: list[float] = field(default_factory=list)

    def answer(self, request: Request) -> Reply:
        self.started_at.append(time.monotonic())
        body: Final = request.body.decode()
        if '"OUTPUT"' in body:
            self.release.wait(_HOLD_SECONDS)
        self.replied_at.append(time.monotonic())
        if "OUTAGE" in body:
            return Reply(status=500, body=b'{"message": "synthetic ApplyGuardrail outage"}')
        return Reply(body=_INTERVENED if _TRIGGER in body else _ALLOWED)


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    scans: list[_Scans]


@dataclass(frozen=True, slots=True)
class _Streamed:
    pieces: tuple[str, ...]
    arrived_at: tuple[float, ...]
    finished_after: tuple[int, ...]
    errors: tuple[str, ...]


def _chat_sse(pieces: tuple[str, ...]) -> tuple[bytes, ...]:
    def frame(delta: dict[str, JsonValue], finish: str | None) -> bytes:
        chunk: Final = {
            "id": "chatcmpl-background",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": _MODEL,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return b"data: " + json.dumps(chunk).encode() + b"\n\n"

    return (
        *(frame({"role": "assistant", "content": piece}, None) for piece in pieces),
        frame({}, "stop"),
        b"data: [DONE]\n\n",
    )


def _anthropic_sse(pieces: tuple[str, ...]) -> tuple[bytes, ...]:
    def event(payload: dict[str, JsonValue]) -> bytes:
        return f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n".encode()

    message: Final[dict[str, JsonValue]] = {
        "id": "msg_background",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 1},
    }
    return (
        event({"type": "message_start", "message": message}),
        event({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        *(
            event({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": piece}})
            for piece in pieces
        ),
        event({"type": "content_block_stop", "index": 0}),
        event({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": len(pieces)}}),
        event({"type": "message_stop"}),
    )


def _model_reply(request: Request) -> Reply:
    prompt: Final = str(object_value(list_value(object_value(json.loads(request.body))["messages"])[-1])["content"])
    pieces: Final = _LEAKY if "leak" in prompt else _OUTAGE if "outage" in prompt else _CLEAN
    return Reply(
        chunks=_anthropic_sse(pieces) if request.target.endswith("/v1/messages") else _chat_sse(pieces),
        content_type="text/event-stream",
        pause_between_chunks=0.05,
    )


def _bedrock_rail(name: str, endpoint: str, background: bool) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "bedrock",
            "mode": "post_call",
            "default_on": False,
            "disable_exception_on_block": True,
            "streaming_buffer_until_moderated": False,
            "streaming_end_of_stream_only": False,
            "streaming_sampling_rate": _RATE,
            **({"streaming_scan_in_background": True} if background else {}),
            "guardrailIdentifier": "synthetic" + uuid.uuid4().hex[:8],
            "guardrailVersion": "DRAFT",
            "aws_region_name": "us-east-1",
            "aws_access_key_id": "AKIASYNTHETICGUARDRAIL",
            "aws_secret_access_key": "synthetic-secret",
            "aws_bedrock_runtime_endpoint": endpoint,
        },
    }


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("background-stream-scan")
    scans: Final = [_Scans()]
    stock: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final = directory / "background-stream-scan.yaml"
    with wire_server(lambda request: scans[0].answer(request)) as policy, wire_server(_model_reply) as model:
        config.write_text(
            yaml.safe_dump(
                {
                    **stock,
                    "model_list": [
                        *list_value(stock["model_list"]),
                        {
                            "model_name": _MODEL,
                            "litellm_params": {
                                "model": f"openai/{_MODEL}",
                                "api_base": f"{model.url}/v1",
                                "api_key": "synthetic",
                            },
                        },
                        {
                            "model_name": _CLAUDE,
                            "litellm_params": {
                                "model": "anthropic/claude-sonnet-4-6",
                                "api_base": model.url,
                                "api_key": "synthetic",
                            },
                        },
                    ],
                    "guardrails": [
                        _bedrock_rail("background", policy.url, background=True),
                        _bedrock_rail("inline", policy.url, background=False),
                    ],
                }
            )
        )
        with gateway_from_environment() as shared, owned_proxy(shared, directory, {}, config=config) as candidate:
            yield _Rig(candidate, scans)


def _chat_text(event: dict[str, JsonValue]) -> str | None:
    texts: Final = [
        object_value(object_value(choice).get("delta") or {}).get("content")
        for choice in list_value(event.get("choices") or [])
    ]
    return "".join(text for text in texts if isinstance(text, str)) or None


def _messages_text(event: dict[str, JsonValue]) -> str | None:
    delta: Final = object_value(event.get("delta") or {}) if event.get("type") == "content_block_delta" else {}
    text: Final = delta.get("text")
    return text if isinstance(text, str) and text else None


def _responses_text(event: dict[str, JsonValue]) -> str | None:
    delta: Final = event.get("delta") if event.get("type") == "response.output_text.delta" else None
    return delta if isinstance(delta, str) and delta else None


def _chat_finished(event: dict[str, JsonValue]) -> bool:
    return any(object_value(choice).get("finish_reason") for choice in list_value(event.get("choices") or []))


def _messages_finished(event: dict[str, JsonValue]) -> bool:
    return event.get("type") == "message_delta"


def _responses_finished(event: dict[str, JsonValue]) -> bool:
    return event.get("type") in ("response.completed", "response.incomplete")


def _error_message(event: dict[str, JsonValue]) -> str | None:
    error: Final = event.get("error")
    if isinstance(error, dict):
        return str(error.get("message"))
    return str(event.get("message")) if event.get("type") == "error" else None


def _chat_body(prompt: str) -> dict[str, JsonValue]:
    return {"model": _MODEL, "messages": [{"role": "user", "content": prompt}]}


def _messages_body(prompt: str) -> dict[str, JsonValue]:
    return {**_chat_body(prompt), "model": _CLAUDE, "max_tokens": 256}


def _responses_body(prompt: str) -> dict[str, JsonValue]:
    return {"model": _CLAUDE, "max_output_tokens": 256, "input": prompt}


_ROUTES: Final = {
    "/v1/chat/completions": (_chat_body, _chat_text, _chat_finished),
    "/v1/messages": (_messages_body, _messages_text, _messages_finished),
    "/v1/responses": (_responses_body, _responses_text, _responses_finished),
}


def _stream(
    rig: _Rig, guardrail: str, prompt: str, release_after: int, route: str = "/v1/chat/completions"
) -> tuple[_Streamed, _Scans]:
    """Stream `route`, releasing held scans once `release_after` text pieces have arrived."""
    scans: Final = _Scans()
    rig.scans[0] = scans
    return _read(rig, guardrail, prompt, route, scans, release_after), scans


def _read(rig: _Rig, guardrail: str, prompt: str, route: str, scans: _Scans, release_after: int) -> _Streamed:
    body_of, text_of, finished = _ROUTES[route]
    pieces: Final[list[str]] = []
    arrived_at: Final[list[float]] = []
    finished_after: Final[list[int]] = []
    errors: Final[list[str]] = []
    with rig.gateway.client.stream(
        "POST",
        route,
        json={**body_of(f"{prompt} {uuid.uuid4().hex}"), "stream": True, "guardrails": [guardrail]},
        headers={"Authorization": f"Bearer {rig.gateway.key}"},
    ) as response:
        assert response.status_code == 200, response.read()
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = object_value(json.loads(line[len("data: ") :]))
            text = text_of(event)
            if text is not None:
                pieces.append(text)
                arrived_at.append(time.monotonic())
            if finished(event):
                finished_after.append(len(pieces))
            error = _error_message(event)
            if error is not None:
                errors.append(error)
            if len(pieces) >= release_after:
                scans.release.set()
    return _Streamed(tuple(pieces), tuple(arrived_at), tuple(finished_after), tuple(errors))


def test_chunks_after_the_nth_keep_streaming_while_its_scan_is_pending(rig: _Rig) -> None:
    streamed, scans = _stream(rig, "background", "clean", release_after=_RATE + 1)
    assert streamed.pieces == _CLEAN
    assert streamed.arrived_at[_RATE] < scans.replied_at[0], "the chunk after the Nth waited for the Nth chunk's scan"


@pytest.mark.parametrize("route", tuple(_ROUTES))
def test_block_verdict_stops_the_stream_at_the_next_chunk(rig: _Rig, route: str) -> None:
    streamed, scans = _stream(rig, "background", "leak", release_after=_RATE + 1, route=route)
    *before, last = streamed.pieces
    assert last == _DENIAL
    assert _DENIAL not in before
    assert _RATE < len(before) < len(_LEAKY), before
    assert tuple(before) == _LEAKY[: len(before)]
    assert streamed.arrived_at[_RATE] < scans.replied_at[0]


@pytest.mark.parametrize("route", tuple(_ROUTES))
def test_block_verdict_that_lands_after_the_upstream_finished_replaces_the_finish(rig: _Rig, route: str) -> None:
    streamed, scans = _stream(rig, "background", "leak", release_after=len(_LEAKY) + 1, route=route)
    assert streamed.pieces == (*_LEAKY, _DENIAL)
    assert streamed.arrived_at[len(_LEAKY) - 1] < scans.replied_at[0], "the verdict landed before the upstream finished"
    assert streamed.finished_after == (len(streamed.pieces),), "the stream finished before the block verdict"
    overlaps: Final = zip(scans.started_at[1:], scans.replied_at[:-1], strict=True)
    assert all(started >= replied for started, replied in overlaps), "a scan started while another was in flight"


def test_without_the_flag_the_nth_chunk_still_waits_for_its_scan(rig: _Rig) -> None:
    streamed, scans = _stream(rig, "inline", "clean", release_after=_RATE - 1)
    assert streamed.pieces == _CLEAN
    assert scans.replied_at[0] < streamed.arrived_at[_RATE - 1]


@pytest.mark.parametrize("route", tuple(_ROUTES))
def test_a_failed_background_scan_ends_the_stream_with_the_inline_scan_error(rig: _Rig, route: str) -> None:
    background, _ = _stream(rig, "background", "outage", release_after=_RATE + 1, route=route)
    inline, _ = _stream(rig, "inline", "outage", release_after=0, route=route)
    assert len(inline.errors) == 1, inline
    assert background.errors == inline.errors
    assert _RATE < len(background.pieces) < len(_OUTAGE), background.pieces
    assert background.pieces == _OUTAGE[: len(background.pieces)]
    assert background.finished_after == ()


def test_a_mixed_burst_ends_every_stream_on_its_own_verdict_while_the_scan_peer_fails_some(rig: _Rig) -> None:
    scans: Final = _Scans()
    scans.release.set()
    rig.scans[0] = scans
    calls: Final = tuple(product(("clean", "leak", "outage"), (*_ROUTES, *_ROUTES)))
    with ThreadPoolExecutor(len(calls)) as pool:
        served: Final = tuple(pool.map(lambda call: _read(rig, "background", call[0], call[1], scans, 0), calls))
    for (kind, route), streamed in zip(calls, served, strict=True):
        match kind:
            case "clean":
                assert (streamed.pieces, streamed.errors) == (_CLEAN, ()), (route, streamed)
                assert streamed.finished_after == (len(_CLEAN),), (route, streamed)
            case "leak":
                assert streamed.pieces[-1] == _DENIAL and streamed.errors == (), (route, streamed)
                assert streamed.finished_after == (len(streamed.pieces),), (route, streamed)
            case _:
                assert len(streamed.errors) == 1 and streamed.finished_after == (), (route, streamed)
                assert streamed.pieces == _OUTAGE[: len(streamed.pieces)], (route, streamed)
    after: Final = _read(rig, "background", "clean", "/v1/chat/completions", scans, 0)
    assert after.pieces == _CLEAN
