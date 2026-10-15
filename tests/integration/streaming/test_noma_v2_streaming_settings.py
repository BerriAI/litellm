import asyncio
import json
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Final, Literal, TypeAlias, cast

import anthropic
import httpx
import openai
import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, eventually, list_value, object_value, string_value
from integration._support.process import OwnedProxy, WorkerHealthcheck, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

Endpoint: TypeAlias = Literal["chat", "messages", "responses"]
Spelling: TypeAlias = Literal["noma_v2", "noma_use_v2", "stored", "custom"]
Variant: TypeAlias = Literal[
    "default",
    "buffered",
    "end_only",
    "sparse",
    "dense",
    "release",
    "buffered_word",
    "sparse_word",
    "invalid_rate_buffered",
    "invalid_rate_end_only",
    "release_invalid",
    "unset",
    "off",
    "off_words",
    "invalid",
    "release_alone",
    "fail_open",
    "pre_call",
]
Action: TypeAlias = Literal["pass", "block", "mask", "late", "outage", "slow", "reject"]
Client: TypeAlias = Literal["sync", "async"]
SdkEvent: TypeAlias = tuple[Literal["text", "failed", "status", "error"], str]

ENDPOINTS: Final[tuple[Endpoint, ...]] = ("chat", "messages", "responses")
SPELLINGS: Final[tuple[Spelling, ...]] = ("noma_v2", "noma_use_v2", "stored", "custom")
MODELS: Final[Mapping[Endpoint, str]] = {
    "chat": "noma-chat",
    "messages": "noma-messages",
    "responses": "noma-responses",
}
PATHS: Final[Mapping[Endpoint, str]] = {
    "chat": "/v1/chat/completions",
    "messages": "/v1/messages",
    "responses": "/v1/responses",
}
SECRET: Final = "jane@example.com"
MASK: Final = "[EMAIL]"
BLOCKED: Final = "Request blocked by Noma guardrail"
SLOW_SECONDS: Final = 1.5
BUFFERED_BLOCK_STATUS: Final[Mapping[Endpoint, int]] = {"chat": 400, "messages": 400, "responses": 200}
BUFFERED_OUTAGE_STATUS: Final[Mapping[Endpoint, int]] = {"chat": 503, "messages": 503, "responses": 200}
PIECES: Final = (
    " Contact",
    " me",
    " at",
    " work",
    " via",
    f" {SECRET}",
    " for",
    " the",
    " quarterly",
    " report",
    " and",
    " the",
    " budget",
    ".",
)
STREAMING_KEYS: Final = (
    "streaming_buffer_until_moderated",
    "streaming_end_of_stream_only",
    "streaming_sampling_rate",
    "streaming_buffer_release_on_scan",
)
INVALID: Final[Mapping[str, JsonValue]] = {
    "streaming_buffer_until_moderated": "maybe",
    "streaming_end_of_stream_only": "maybe",
    "streaming_sampling_rate": 0,
    "streaming_buffer_release_on_scan": "maybe",
}
SETTINGS: Final[Mapping[Variant, Mapping[str, JsonValue]]] = {
    "default": {},
    "buffered": {"streaming_buffer_until_moderated": True},
    "end_only": {"streaming_end_of_stream_only": True},
    "sparse": {"streaming_sampling_rate": 1000},
    "dense": {"streaming_sampling_rate": 1},
    "release": {"streaming_buffer_until_moderated": True, "streaming_buffer_release_on_scan": True},
    "buffered_word": {"streaming_buffer_until_moderated": "true"},
    "sparse_word": {"streaming_sampling_rate": "1000"},
    "invalid_rate_buffered": {"streaming_buffer_until_moderated": True, "streaming_sampling_rate": -5},
    "invalid_rate_end_only": {"streaming_end_of_stream_only": True, "streaming_sampling_rate": "often"},
    "release_invalid": {"streaming_buffer_until_moderated": True, "streaming_buffer_release_on_scan": "maybe"},
    "unset": dict.fromkeys(STREAMING_KEYS),
    "off": {
        "streaming_buffer_until_moderated": False,
        "streaming_end_of_stream_only": False,
        "streaming_buffer_release_on_scan": False,
    },
    "off_words": {
        "streaming_buffer_until_moderated": "false",
        "streaming_end_of_stream_only": "false",
        "streaming_buffer_release_on_scan": "false",
    },
    "invalid": INVALID,
    "release_alone": {"streaming_buffer_release_on_scan": True},
    "fail_open": {"streaming_buffer_until_moderated": True, "block_failures": False},
    "pre_call": {"mode": "pre_call", "streaming_buffer_until_moderated": True},
}
SUBCLASS_MODULE: Final = "noma_v2_subclass"
SUBCLASS_SOURCE: Final = """from litellm.proxy.guardrails.guardrail_hooks.noma.noma_v2 import NomaV2Guardrail


class NomaV2Subclass(NomaV2Guardrail):
    async def apply_guardrail(self, *args, **kwargs):
        return await super().apply_guardrail(*args, **kwargs)
"""
SPELLING_PARAMS: Final[Mapping[Spelling, Mapping[str, JsonValue]]] = {
    "noma_v2": {"guardrail": "noma_v2"},
    "noma_use_v2": {"guardrail": "noma", "use_v2": True},
    "stored": {"guardrail": "noma_v2"},
    "custom": {"guardrail": f"{SUBCLASS_MODULE}.NomaV2Subclass"},
}


def _texts(value: JsonValue) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(chain.from_iterable(_texts(item) for item in value))
    if isinstance(value, dict):
        return tuple(chain.from_iterable(_texts(item) for item in value.values()))
    return ()


def _marker(body: Mapping[str, JsonValue]) -> str:
    return next(text for text in _texts(dict(body)) if text.startswith("noma-"))


def _full_text(marker: str) -> str:
    return marker + "".join(PIECES)


def _action(marker: str) -> Action:
    return cast(Action, marker.split("-")[1])


def _scan_action(text: str) -> Action:
    marker: Final = next((word for word in text.split(" ") if word.startswith("noma-")), None)
    return "pass" if marker is None else _action(marker)


def _event(name: str | None, payload: Mapping[str, JsonValue]) -> bytes:
    prefix: Final = f"event: {name}\n" if name is not None else ""
    return f"{prefix}data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _chat_stream(marker: str, choices: int) -> tuple[bytes, ...]:
    chunk: Final = {"id": f"chatcmpl-{marker}", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini"}
    deltas: Final = (
        {"role": "assistant", "content": marker},
        *({"content": piece} for piece in PIECES),
    )
    return (
        *(
            _event(None, {**chunk, "choices": [{"index": index, "delta": delta, "finish_reason": None}]})
            for delta in deltas
            for index in range(choices)
        ),
        _event(
            None,
            {**chunk, "choices": [{"index": index, "delta": {}, "finish_reason": "stop"} for index in range(choices)]},
        ),
        b"data: [DONE]\n\n",
    )


def _message(marker: str) -> dict[str, JsonValue]:
    return {
        "id": f"msg-{marker}",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": [{"type": "text", "text": _full_text(marker)}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 2, "output_tokens": 15},
    }


def _messages_stream(marker: str) -> tuple[bytes, ...]:
    events: Final[tuple[dict[str, JsonValue], ...]] = (
        {
            "type": "message_start",
            "message": {
                **_message(marker),
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 2, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        *(
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}
            for text in (marker, *PIECES)
        ),
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 15}},
        {"type": "message_stop"},
    )
    return tuple(_event(string_value(event["type"]), event) for event in events)


def _response_output(marker: str) -> dict[str, JsonValue]:
    return {
        "type": "message",
        "id": f"msg-{marker}",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": _full_text(marker), "annotations": []}],
    }


def _response(marker: str) -> dict[str, JsonValue]:
    return {
        "id": f"resp-{marker}",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [_response_output(marker)],
        "usage": {"input_tokens": 2, "output_tokens": 15, "total_tokens": 17},
    }


def _responses_stream(marker: str) -> tuple[bytes, ...]:
    item_id: Final = f"msg-{marker}"
    text: Final = _full_text(marker)
    part: Final[dict[str, JsonValue]] = {"type": "output_text", "text": text, "annotations": []}
    output: Final = _response_output(marker)
    response: Final = _response(marker)
    position: Final[dict[str, JsonValue]] = {"item_id": item_id, "output_index": 0, "content_index": 0}
    events: Final[tuple[dict[str, JsonValue], ...]] = (
        {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
        {"type": "response.in_progress", "response": {**response, "status": "in_progress", "output": []}},
        {
            "type": "response.output_item.added",
            "item": {**output, "status": "in_progress", "content": []},
            "output_index": 0,
        },
        {"type": "response.content_part.added", **position, "part": {**part, "text": ""}},
        *({"type": "response.output_text.delta", **position, "delta": delta} for delta in (marker, *PIECES)),
        {"type": "response.output_text.done", **position, "text": text},
        {"type": "response.content_part.done", **position, "part": part},
        {"type": "response.output_item.done", "item": output, "output_index": 0},
        {"type": "response.completed", "response": response},
    )
    return tuple(
        _event(string_value(event["type"]), {**event, "sequence_number": index}) for index, event in enumerate(events)
    )


def _completed(target: str, marker: str) -> dict[str, JsonValue]:
    if target == PATHS["messages"]:
        return _message(marker)
    if target == PATHS["responses"]:
        return _response(marker)
    return {
        "id": f"chatcmpl-{marker}",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": _full_text(marker)}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 15, "total_tokens": 17},
    }


def _rejection(target: str) -> dict[str, JsonValue]:
    message: Final = "synthetic upstream rejection"
    if target == PATHS["messages"]:
        return {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    return {"error": {"message": message, "type": "invalid_request_error", "param": None, "code": None}}


def _provider(request: Request) -> Reply:
    body: Final = JSON_OBJECT.validate_json(request.body)
    marker: Final = _marker(body)
    if request.target not in PATHS.values():
        return Reply(status=404, body=json.dumps({"error": f"unexpected target {request.target}"}).encode())
    if _action(marker) == "reject":
        return Reply(status=400, body=json.dumps(_rejection(request.target)).encode())
    if body.get("stream") is not True:
        return Reply(body=json.dumps(_completed(request.target, marker)).encode())
    choices: Final = body.get("n", 1)
    assert isinstance(choices, int), request.body
    streams: Final = {
        PATHS["chat"]: _chat_stream(marker, choices),
        PATHS["messages"]: _messages_stream(marker),
        PATHS["responses"]: _responses_stream(marker),
    }
    return Reply(content_type="text/event-stream", chunks=streams[request.target], pause_between_chunks=0.05)


@dataclass(frozen=True, slots=True)
class Scan:
    guardrail: str
    input_type: str
    text: str


def _verdict(action: Action, text: str, earlier: int, texts: Sequence[str]) -> dict[str, JsonValue]:
    if action == "block" and SECRET in text:
        return {"action": "BLOCKED", "reason": "synthetic email policy"}
    if action == "late" and earlier > 0:
        return {"action": "BLOCKED", "reason": "synthetic late verdict"}
    if action == "mask" and SECRET in text:
        return {"action": "GUARDRAIL_INTERVENED", "texts": [part.replace(SECRET, MASK) for part in texts]}
    return {"action": "NONE"}


@dataclass(slots=True)
class NomaDouble:
    """Scripted Noma v2 `/litellm/guardrail` endpoint; the action word in the stream's marker picks the verdict."""

    scans: list[Scan] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    down: threading.Event = field(default_factory=threading.Event)

    def respond(self, request: Request) -> Reply:
        guardrail: Final = request.target.strip("/").split("/")[0]
        assert request.target == f"/{guardrail}/litellm/guardrail", request.target
        body: Final = JSON_OBJECT.validate_json(request.body)
        texts: Final = tuple(string_value(text) for text in list_value(object_value(body["inputs"]).get("texts", [])))
        text: Final = "".join(texts)
        scan: Final = Scan(guardrail, string_value(body["input_type"]), text)
        with self.lock:
            earlier: Final = sum(1 for seen in self.scans if seen.text.split(" ")[0] == text.split(" ")[0])
            self.scans.append(scan)
        action: Final = _scan_action(text)
        if self.down.is_set() or action == "outage":
            return Reply(status=503, body=json.dumps({"detail": "synthetic Noma outage"}).encode())
        if action == "slow":
            time.sleep(SLOW_SECONDS)
        return Reply(body=json.dumps(_verdict(action, text, earlier, texts)).encode())

    def for_marker(self, marker: str) -> tuple[Scan, ...]:
        with self.lock:
            return tuple(scan for scan in self.scans if scan.text.startswith(marker))


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: OwnedProxy
    noma: NomaDouble
    noma_url: str
    provider: Wire
    names: Mapping[tuple[Spelling, Variant], str]
    stored: Mapping[Variant, str]

    @property
    def gateway(self) -> Gateway:
        return self.proxy.gateway


def _guardrail_params(noma_url: str, name: str, spelling: Spelling, variant: Variant) -> dict[str, JsonValue]:
    return {
        **SPELLING_PARAMS[spelling],
        "mode": "post_call",
        "default_on": False,
        "api_base": f"{noma_url}/{name}",
        "api_key": "synthetic-noma-key",
        **SETTINGS[variant],
    }


def _config(provider_url: str, noma_url: str, names: Mapping[tuple[Spelling, Variant], str]) -> dict[str, JsonValue]:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    return {
        **base,
        "guardrails": [
            {"guardrail_name": name, "litellm_params": _guardrail_params(noma_url, name, spelling, variant)}
            for (spelling, variant), name in names.items()
            if spelling != "stored"
        ],
        "model_list": [
            {
                "model_name": MODELS["chat"],
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": f"{provider_url}/v1",
                    "api_key": "synthetic-provider-key",
                },
            },
            {
                "model_name": MODELS["messages"],
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5-20250929",
                    "api_base": provider_url,
                    "api_key": "synthetic-provider-key",
                },
            },
            {
                "model_name": MODELS["responses"],
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": f"{provider_url}/v1",
                    "api_key": "synthetic-provider-key",
                },
            },
        ],
    }


def _store(gateway: Gateway, noma_url: str, name: str, variant: Variant) -> str:
    response: Final = gateway.request(
        "POST",
        "/guardrails",
        {"guardrail": {"guardrail_name": name, "litellm_params": _guardrail_params(noma_url, name, "stored", variant)}},
    )
    assert response.status_code == 200, response.text
    return string_value(JSON_OBJECT.validate_json(response.content)["guardrail_id"])


@pytest.fixture(scope="module")
def rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    noma: Final = NomaDouble()
    run: Final = uuid.uuid4().hex[:8]
    names: Final = {
        (spelling, variant): f"{spelling}-{variant}-{run}".replace("_", "-")
        for spelling in SPELLINGS
        for variant in SETTINGS
    }
    with wire_server(_provider) as provider, wire_server(noma.respond) as noma_wire:
        stored: Final = {
            variant: _store(session_gateway, noma_wire.url, name, variant)
            for (spelling, variant), name in names.items()
            if spelling == "stored"
        }
        try:
            directory: Final = tmp_path_factory.mktemp("noma-v2-streaming")
            (directory / f"{SUBCLASS_MODULE}.py").write_text(SUBCLASS_SOURCE)
            config_path: Final = directory / "noma-v2-streaming.yaml"
            config_path.write_text(yaml.safe_dump(_config(provider.url, noma_wire.url, names)))
            with owned_proxy_process(
                session_gateway,
                directory,
                {},
                config=config_path,
                workers=2,
                healthcheck=WorkerHealthcheck.LENIENT,
            ) as proxy:
                yield Rig(proxy, noma, noma_wire.url, provider, names, stored)
        finally:
            for identity in stored.values():
                deleted = session_gateway.request("DELETE", f"/guardrails/{identity}")
                assert deleted.status_code == 200, deleted.text


@dataclass(frozen=True, slots=True)
class Streamed:
    marker: str
    response: httpx.Response
    content: str
    scans: tuple[Scan, ...]

    def shape(self) -> tuple[str, ...]:
        return tuple(scan.text.removeprefix(self.marker) for scan in self.scans)


def _event_text(event: Mapping[str, JsonValue], endpoint: Endpoint) -> str:
    if endpoint == "responses":
        return string_value(event.get("delta", "")) if event.get("type") == "response.output_text.delta" else ""
    if endpoint == "messages":
        return string_value(object_value(event.get("delta", {})).get("text", ""))
    return "".join(
        string_value(object_value(object_value(choice).get("delta", {})).get("content", "") or "")
        for choice in list_value(event.get("choices", []))
    )


def _line_text(line: str, endpoint: Endpoint) -> str:
    return (
        _event_text(JSON_OBJECT.validate_json(line.removeprefix("data: ")), endpoint)
        if line.startswith("data: {")
        else ""
    )


def _body(endpoint: Endpoint, marker: str, guardrail: str, *, stream: bool = True) -> dict[str, JsonValue]:
    prompt: Final[dict[str, JsonValue]] = (
        {"input": marker} if endpoint == "responses" else {"messages": [{"role": "user", "content": marker}]}
    )
    return {
        "model": MODELS[endpoint],
        **prompt,
        **({"max_tokens": 64} if endpoint == "messages" else {}),
        "stream": stream,
        "guardrails": [guardrail],
    }


def _new_marker(action: Action) -> str:
    return f"noma-{action}-{uuid.uuid4().hex}"


def _send(
    rig: Rig,
    endpoint: Endpoint,
    guardrail: str,
    action: Action,
    *,
    input_type: str = "response",
    extra: Mapping[str, JsonValue] | None = None,
    via: Gateway | None = None,
) -> Streamed:
    marker: Final = _new_marker(action)
    response: Final = (via or rig.gateway).request(
        "POST", PATHS[endpoint], {**_body(endpoint, marker, guardrail), **(extra or {})}
    )
    scans: Final = rig.noma.for_marker(marker)
    assert all(scan.guardrail == guardrail and scan.input_type == input_type for scan in scans), scans
    content: Final = "".join(_line_text(line, endpoint) for line in response.text.splitlines())
    return Streamed(marker, response, content, scans)


def _stream(
    rig: Rig,
    endpoint: Endpoint,
    spelling: Spelling,
    variant: Variant,
    action: Action,
    status: int = 200,
    *,
    input_type: str = "response",
    extra: Mapping[str, JsonValue] | None = None,
) -> Streamed:
    streamed: Final = _send(rig, endpoint, rig.names[(spelling, variant)], action, input_type=input_type, extra=extra)
    assert streamed.response.status_code == status, streamed.response.text
    return streamed


def _burst(rig: Rig, cases: Sequence[tuple[Endpoint, Action]], guardrail: str) -> tuple[Streamed, ...]:
    with ThreadPoolExecutor(max_workers=len(cases)) as pool:
        return tuple(pool.map(lambda case: _send(rig, case[0], guardrail, case[1]), cases))


def _attempt(rig: Rig, endpoint: Endpoint, guardrail: str, action: Action) -> str:
    try:
        return _send(rig, endpoint, guardrail, action).response.text
    except httpx.TransportError as error:
        return f"transport error: {type(error).__name__}"


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("variant", ["buffered", "buffered_word", "invalid_rate_buffered", "release_invalid"])
def test_buffer_until_moderated_keeps_flagged_text_from_the_client(
    rig: Rig, endpoint: Endpoint, spelling: Spelling, variant: Variant
) -> None:
    unbuffered: Final = _stream(rig, endpoint, spelling, "default", "block")
    assert SECRET in unbuffered.content, unbuffered.response.text
    assert BLOCKED in unbuffered.response.text, unbuffered.response.text

    buffered: Final = _stream(rig, endpoint, spelling, variant, "block", BUFFERED_BLOCK_STATUS[endpoint])
    assert buffered.content == "", buffered.response.text
    assert SECRET not in buffered.response.text, buffered.response.text
    assert BLOCKED in buffered.response.text, buffered.response.text
    assert [scan.text for scan in buffered.scans] == [_full_text(buffered.marker)], buffered.scans


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("variant", ["end_only", "sparse", "sparse_word", "invalid_rate_end_only"])
def test_one_scan_settings_scan_the_finished_stream_once(
    rig: Rig, endpoint: Endpoint, spelling: Spelling, variant: Variant
) -> None:
    sampled: Final = _stream(rig, endpoint, spelling, "default", "pass")
    assert sampled.content == _full_text(sampled.marker), sampled.response.text
    assert len(sampled.scans) > 1, sampled.scans

    configured: Final = _stream(rig, endpoint, spelling, variant, "pass")
    assert configured.content == _full_text(configured.marker), configured.response.text
    assert [scan.text for scan in configured.scans] == [_full_text(configured.marker)], configured.scans


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_sampling_rate_one_scans_every_new_piece(rig: Rig, endpoint: Endpoint, spelling: Spelling) -> None:
    sampled: Final = _stream(rig, endpoint, spelling, "default", "pass")
    assert len(sampled.scans) < len(PIECES), sampled.scans

    dense: Final = _stream(rig, endpoint, spelling, "dense", "pass")
    assert dense.content == _full_text(dense.marker), dense.response.text
    scanned: Final = {scan.text for scan in dense.scans}
    prefixes: Final = {dense.marker + "".join(PIECES[:count]) for count in range(len(PIECES) + 1)}
    assert prefixes <= scanned, sorted(prefixes - scanned)


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_release_on_scan_releases_only_scanned_windows(rig: Rig, endpoint: Endpoint, spelling: Spelling) -> None:
    released: Final = _stream(rig, endpoint, spelling, "release", "late")
    assert len(released.scans) == 2, released.scans
    assert released.content == released.scans[0].text, (released.response.text, released.scans)
    assert released.content and _full_text(released.marker).startswith(released.content), released.response.text
    assert BLOCKED in released.response.text, released.response.text

    held: Final = _stream(rig, endpoint, spelling, "buffered", "late")
    assert held.content == _full_text(held.marker), held.response.text
    assert len(held.scans) == 1, held.scans


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_invalid_release_setting_keeps_the_buffered_stream_held(
    rig: Rig, endpoint: Endpoint, spelling: Spelling
) -> None:
    held: Final = _stream(rig, endpoint, spelling, "release_invalid", "late")
    assert held.content == _full_text(held.marker), held.response.text
    assert len(held.scans) == 1, held.scans


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("variant", ["unset", "off", "off_words", "invalid", "release_alone"])
def test_unset_off_and_invalid_settings_keep_the_sampled_live_stream(
    rig: Rig, endpoint: Endpoint, spelling: Spelling, variant: Variant
) -> None:
    sampled: Final = _stream(rig, endpoint, spelling, "default", "pass")
    configured: Final = _stream(rig, endpoint, spelling, variant, "pass")
    assert configured.content == _full_text(configured.marker), configured.response.text
    assert configured.shape() == sampled.shape(), (configured.scans, sampled.scans)

    leaked: Final = _stream(rig, endpoint, spelling, variant, "block")
    assert SECRET in leaked.content, leaked.response.text
    assert BLOCKED in leaked.response.text, leaked.response.text


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_invalid_settings_are_logged_with_the_guardrail_name(rig: Rig, spelling: Spelling) -> None:
    name: Final = rig.names[(spelling, "invalid")]
    log: Final = rig.proxy.log.read_text()
    expected: Final = tuple(f"Noma v2 guardrail {name}: ignoring {key}={value!r}" for key, value in INVALID.items())
    assert [line for line in expected if line not in log] == [], name


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_noma_outage_fails_closed_before_any_text_when_buffered(
    rig: Rig, endpoint: Endpoint, spelling: Spelling
) -> None:
    closed: Final = _stream(rig, endpoint, spelling, "buffered", "outage", BUFFERED_OUTAGE_STATUS[endpoint])
    assert closed.content == "", closed.response.text
    assert endpoint != "responses" or "response.failed" in closed.response.text, closed.response.text
    assert [scan.text for scan in closed.scans] == [_full_text(closed.marker)], closed.scans

    opened: Final = _stream(rig, endpoint, spelling, "fail_open", "outage")
    assert opened.content == _full_text(opened.marker), opened.response.text
    assert [scan.text for scan in opened.scans] == [_full_text(opened.marker)], opened.scans


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_upstream_rejection_reaches_the_client_without_a_scan(rig: Rig, endpoint: Endpoint) -> None:
    rejected: Final = _stream(rig, endpoint, "noma_v2", "buffered", "reject", 400)
    assert "synthetic upstream rejection" in rejected.response.text, rejected.response.text
    assert rejected.scans == (), rejected.scans


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_buffered_block_withholds_every_choice(rig: Rig, spelling: Spelling) -> None:
    unbuffered: Final = _stream(rig, "chat", spelling, "default", "block", extra={"n": 2})
    assert SECRET in unbuffered.content, unbuffered.response.text

    buffered: Final = _stream(rig, "chat", spelling, "buffered", "block", 400, extra={"n": 2})
    assert buffered.content == "", buffered.response.text
    assert SECRET not in buffered.response.text, buffered.response.text
    assert BLOCKED in buffered.response.text, buffered.response.text


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_pre_call_guardrail_scans_the_request_and_streams_live(
    rig: Rig, endpoint: Endpoint, spelling: Spelling
) -> None:
    streamed: Final = _stream(rig, endpoint, spelling, "pre_call", "pass", input_type="request")
    assert streamed.content == _full_text(streamed.marker), streamed.response.text
    assert [scan.text for scan in streamed.scans] == [streamed.marker], streamed.scans


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("variant", ["default", "buffered", "end_only"])
def test_non_streaming_post_call_ignores_the_streaming_settings(rig: Rig, endpoint: Endpoint, variant: Variant) -> None:
    guardrail: Final = rig.names[("noma_v2", variant)]
    passed: Final = _send(rig, endpoint, guardrail, "pass", extra={"stream": False})
    assert passed.response.status_code == 200, passed.response.text
    assert _full_text(passed.marker) in passed.response.text, passed.response.text
    assert [scan.text for scan in passed.scans] == [_full_text(passed.marker)], passed.scans

    blocked: Final = _send(rig, endpoint, guardrail, "block", extra={"stream": False})
    assert blocked.response.status_code == 400, blocked.response.text
    assert SECRET not in blocked.response.text, blocked.response.text
    assert BLOCKED in blocked.response.text, blocked.response.text


@dataclass(frozen=True, slots=True)
class SdkOutcome:
    text: str
    status: int | None
    failed: bool


def _outcome(events: Sequence[SdkEvent]) -> SdkOutcome:
    return SdkOutcome(
        "".join(text for kind, text in events if kind == "text"),
        next((int(text) for kind, text in events if kind == "status"), None),
        any(kind in ("failed", "error") for kind, _ in events),
    )


def _openai_events(client: openai.OpenAI, endpoint: Endpoint, marker: str, guardrail: str) -> Iterator[SdkEvent]:
    extra: Final[dict[str, object]] = {"guardrails": [guardrail]}
    if endpoint == "chat":
        for chunk in client.chat.completions.create(
            model=MODELS["chat"], messages=[{"role": "user", "content": marker}], stream=True, extra_body=extra
        ):
            yield from (("text", choice.delta.content or "") for choice in chunk.choices)
        return
    for event in client.responses.create(model=MODELS["responses"], input=marker, stream=True, extra_body=extra):
        if event.type == "response.output_text.delta":
            yield "text", event.delta
        if event.type == "response.failed":
            yield "failed", event.type


async def _async_openai_events(
    client: openai.AsyncOpenAI, endpoint: Endpoint, marker: str, guardrail: str
) -> AsyncIterator[SdkEvent]:
    extra: Final[dict[str, object]] = {"guardrails": [guardrail]}
    if endpoint == "chat":
        async for chunk in await client.chat.completions.create(
            model=MODELS["chat"], messages=[{"role": "user", "content": marker}], stream=True, extra_body=extra
        ):
            for choice in chunk.choices:
                yield "text", choice.delta.content or ""
        return
    async for event in await client.responses.create(
        model=MODELS["responses"], input=marker, stream=True, extra_body=extra
    ):
        if event.type == "response.output_text.delta":
            yield "text", event.delta
        if event.type == "response.failed":
            yield "failed", event.type


def _anthropic_events(client: anthropic.Anthropic, marker: str, guardrail: str) -> Iterator[SdkEvent]:
    for event in client.messages.create(
        model=MODELS["messages"],
        max_tokens=64,
        messages=[{"role": "user", "content": marker}],
        stream=True,
        extra_body={"guardrails": [guardrail]},
    ):
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            yield "text", event.delta.text


async def _async_anthropic_events(
    client: anthropic.AsyncAnthropic, marker: str, guardrail: str
) -> AsyncIterator[SdkEvent]:
    async for event in await client.messages.create(
        model=MODELS["messages"],
        max_tokens=64,
        messages=[{"role": "user", "content": marker}],
        stream=True,
        extra_body={"guardrails": [guardrail]},
    ):
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            yield "text", event.delta.text


def _sync_sdk(gateway: Gateway, endpoint: Endpoint, marker: str, guardrail: str) -> SdkOutcome:
    origin: Final = str(gateway.client.base_url).rstrip("/")

    def events() -> Iterator[SdkEvent]:
        try:
            with httpx.Client(timeout=30, trust_env=False) as transport:
                if endpoint == "messages":
                    with anthropic.Anthropic(
                        api_key=gateway.key, base_url=origin, max_retries=0, http_client=transport
                    ) as client:
                        yield from _anthropic_events(client, marker, guardrail)
                    return
                with openai.OpenAI(
                    api_key=gateway.key, base_url=f"{origin}/v1", max_retries=0, http_client=transport
                ) as client:
                    yield from _openai_events(client, endpoint, marker, guardrail)
        except (openai.APIStatusError, anthropic.APIStatusError) as error:
            yield "status", str(error.status_code)
        except (openai.APIError, anthropic.APIError) as error:
            yield "error", error.message

    return _outcome(tuple(events()))


def _async_sdk(gateway: Gateway, endpoint: Endpoint, marker: str, guardrail: str) -> SdkOutcome:
    origin: Final = str(gateway.client.base_url).rstrip("/")

    async def events() -> AsyncIterator[SdkEvent]:
        try:
            async with httpx.AsyncClient(timeout=30, trust_env=False) as transport:
                if endpoint == "messages":
                    async with anthropic.AsyncAnthropic(
                        api_key=gateway.key, base_url=origin, max_retries=0, http_client=transport
                    ) as client:
                        async for event in _async_anthropic_events(client, marker, guardrail):
                            yield event
                    return
                async with openai.AsyncOpenAI(
                    api_key=gateway.key, base_url=f"{origin}/v1", max_retries=0, http_client=transport
                ) as client:
                    async for event in _async_openai_events(client, endpoint, marker, guardrail):
                        yield event
        except (openai.APIStatusError, anthropic.APIStatusError) as error:
            yield "status", str(error.status_code)
        except (openai.APIError, anthropic.APIError) as error:
            yield "error", error.message

    async def collect() -> tuple[SdkEvent, ...]:
        return tuple([event async for event in events()])

    return _outcome(asyncio.run(collect()))


@pytest.mark.parametrize("client", ["sync", "async"])
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_sdk_clients_receive_no_text_before_a_buffered_block(rig: Rig, endpoint: Endpoint, client: Client) -> None:
    marker: Final = _new_marker("block")
    outcome: Final = {"sync": _sync_sdk, "async": _async_sdk}[client](
        rig.gateway, endpoint, marker, rig.names[("noma_v2", "buffered")]
    )
    assert outcome.text == "", outcome
    if endpoint == "responses":
        assert outcome.failed and outcome.status is None, outcome
    else:
        assert outcome.status == 400, outcome
    assert [scan.text for scan in rig.noma.for_marker(marker)] == [_full_text(marker)]


@pytest.mark.parametrize("variant", ["buffered", "invalid", "unset"])
def test_stored_guardrail_info_reports_the_settings_as_stored(rig: Rig, variant: Variant) -> None:
    info: Final = rig.gateway.get(f"/guardrails/{rig.stored[variant]}/info")
    params: Final = object_value(info["litellm_params"])
    assert {key: params.get(key) for key in SETTINGS[variant]} == dict(SETTINGS[variant]), params


def _on_one_worker(rig: Rig, pid: int, guardrail: str) -> tuple[Streamed, ...]:
    """Two blocked-prompt streams answered by worker `pid` alone, every other worker suspended meanwhile."""
    with (
        ExitStack() as paused,
        httpx.Client(
            base_url=str(rig.gateway.client.base_url),
            timeout=15,
            trust_env=False,
            limits=httpx.Limits(max_keepalive_connections=0),
        ) as client,
    ):
        for other in sorted(rig.proxy.worker_pids() - {pid}):
            paused.enter_context(rig.proxy.suspended(other))
        alone: Final = Gateway(client, rig.gateway.key, rig.gateway.upstream_url)
        return tuple(_send(rig, "chat", guardrail, "block", via=alone) for _ in range(2))


def _converged_on(rig: Rig, pid: int, guardrail: str, satisfied: Callable[[Streamed], bool]) -> tuple[Streamed, ...]:
    return eventually(
        lambda: _on_one_worker(rig, pid, guardrail),
        lambda attempts: all(satisfied(attempt) for attempt in attempts),
        seconds=120,
    )


def test_patching_a_stored_guardrail_applies_the_setting_on_every_worker(rig: Rig) -> None:
    name: Final = f"stored-patched-{uuid.uuid4().hex[:8]}"
    workers: Final = sorted(eventually(rig.proxy.worker_pids, lambda found: len(found) == 2, seconds=30))
    identity: Final = _store(rig.gateway, rig.noma_url, name, "default")
    try:
        before: Final = tuple(
            chain.from_iterable(
                _converged_on(rig, pid, name, lambda attempt: len(attempt.scans) > 0) for pid in workers
            )
        )
        assert all(SECRET in attempt.content for attempt in before), [attempt.response.text for attempt in before]

        patched: Final = rig.gateway.request(
            "PATCH", f"/guardrails/{identity}", {"litellm_params": {"streaming_buffer_until_moderated": True}}
        )
        assert patched.status_code == 200, patched.text
        after: Final = tuple(
            chain.from_iterable(
                _converged_on(rig, pid, name, lambda attempt: attempt.response.status_code == 400) for pid in workers
            )
        )
        assert all(attempt.content == "" and SECRET not in attempt.response.text for attempt in after), [
            attempt.response.text for attempt in after
        ]
    finally:
        deleted: Final = rig.gateway.request("DELETE", f"/guardrails/{identity}")
        assert deleted.status_code == 200, deleted.text


def test_burst_fails_closed_while_noma_is_down_and_recovers_after(rig: Rig) -> None:
    guardrail: Final = rig.names[("noma_v2", "buffered")]
    cases: Final = tuple(
        (endpoint, action)
        for endpoint in ENDPOINTS
        for action in cast(tuple[Action, ...], ("pass", "block"))
        for _ in range(5)
    )
    rig.noma.down.set()
    try:
        down: Final = _burst(rig, cases, guardrail)
    finally:
        rig.noma.down.clear()
    assert [streamed.response.text for streamed in down if streamed.content or SECRET in streamed.response.text] == []
    assert [streamed.scans for streamed in down if len(streamed.scans) != 1] == []

    up: Final = _burst(rig, cases, guardrail)
    delivered: Final = tuple(streamed for streamed in up if _action(streamed.marker) == "pass")
    blocked: Final = tuple(streamed for streamed in up if _action(streamed.marker) == "block")
    assert [s.response.text for s in delivered if s.content != _full_text(s.marker)] == []
    assert [
        s.response.text for s in blocked if s.content or SECRET in s.response.text or BLOCKED not in s.response.text
    ] == []
    assert [streamed.scans for streamed in up if len(streamed.scans) != 1] == []


def _first_text_seconds(rig: Rig, endpoint: Endpoint, guardrail: str) -> tuple[str, float | None, str]:
    marker: Final = _new_marker("slow")
    started: Final = time.monotonic()
    with rig.gateway.client.stream(
        "POST",
        PATHS[endpoint],
        json=_body(endpoint, marker, guardrail),
        headers={"Authorization": f"Bearer {rig.gateway.key}"},
        timeout=30,
    ) as response:
        lines: Final = tuple((time.monotonic() - started, line) for line in response.iter_lines())
    texts: Final = tuple((elapsed, _line_text(line, endpoint)) for elapsed, line in lines)
    return marker, next((elapsed for elapsed, text in texts if text), None), "".join(text for _, text in texts)


def test_slow_noma_holds_every_stream_until_its_verdict(rig: Rig) -> None:
    guardrail: Final = rig.names[("noma_v2", "buffered")]
    with ThreadPoolExecutor(max_workers=18) as pool:
        streams: Final = tuple(pool.map(lambda endpoint: _first_text_seconds(rig, endpoint, guardrail), ENDPOINTS * 6))
    assert [stream for stream in streams if stream[2] != _full_text(stream[0])] == []
    assert [stream for stream in streams if stream[1] is None or stream[1] < SLOW_SECONDS] == []
    assert [marker for marker, _, _ in streams if len(rig.noma.for_marker(marker)) != 1] == []


def test_replaced_workers_keep_the_streaming_settings(rig: Rig) -> None:
    guardrail: Final = rig.names[("noma_v2", "buffered")]
    originals: Final = eventually(rig.proxy.worker_pids, lambda found: len(found) == 2, seconds=30)
    with ThreadPoolExecutor(max_workers=8) as pool:
        inflight: Final = tuple(
            pool.submit(_attempt, rig, ENDPOINTS[index % 3], guardrail, "block") for index in range(16)
        )
        rig.proxy.kill_worker(min(originals))
        during: Final = tuple(future.result() for future in inflight)
    rig.proxy.kill_worker(max(originals))
    assert [text for text in during if SECRET in text] == []
    assert rig.proxy.worker_pids().isdisjoint(originals), rig.proxy.worker_pids()

    after: Final = tuple(
        _stream(rig, endpoint, "noma_v2", "buffered", "block", BUFFERED_BLOCK_STATUS[endpoint])
        for endpoint in ENDPOINTS
        for _ in range(4)
    )
    assert [streamed.response.text for streamed in after if streamed.content or SECRET in streamed.response.text] == []
    assert [streamed.scans for streamed in after if len(streamed.scans) != 1] == []


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_buffer_until_moderated_delivers_the_masked_text(rig: Rig, endpoint: Endpoint, spelling: Spelling) -> None:
    pytest.skip("BUG: LIT-9559 a buffered stream releases the original chunks instead of the guardrail's masked text")
    unbuffered: Final = _stream(rig, endpoint, spelling, "default", "mask")
    assert SECRET in unbuffered.content, unbuffered.response.text

    buffered: Final = _stream(rig, endpoint, spelling, "buffered", "mask")
    assert buffered.content == _full_text(buffered.marker).replace(SECRET, MASK), buffered.response.text
    assert SECRET not in buffered.response.text, buffered.response.text
