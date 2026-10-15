import json
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Final, Literal, TypeAlias, cast

import httpx
import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, list_value, object_value, string_value
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

Endpoint: TypeAlias = Literal["chat", "messages", "responses"]
Spelling: TypeAlias = Literal["noma_v2", "noma_use_v2", "stored"]
Variant: TypeAlias = Literal["default", "buffered", "end_only", "sparse", "dense", "release"]
Action: TypeAlias = Literal["pass", "block", "mask", "late"]

ENDPOINTS: Final[tuple[Endpoint, ...]] = ("chat", "messages", "responses")
SPELLINGS: Final[tuple[Spelling, ...]] = ("noma_v2", "noma_use_v2", "stored")
MODELS: Final[Mapping[Endpoint, str]] = {
    "chat": "noma-chat",
    "messages": "noma-messages",
    "responses": "noma-responses",
}
SECRET: Final = "jane@example.com"
MASK: Final = "[EMAIL]"
BUFFERED_BLOCK_STATUS: Final[Mapping[Endpoint, int]] = {"chat": 400, "messages": 400, "responses": 200}
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
SETTINGS: Final[Mapping[Variant, Mapping[str, JsonValue]]] = {
    "default": {},
    "buffered": {"streaming_buffer_until_moderated": True},
    "end_only": {"streaming_end_of_stream_only": True},
    "sparse": {"streaming_sampling_rate": 1000},
    "dense": {"streaming_sampling_rate": 1},
    "release": {"streaming_buffer_until_moderated": True, "streaming_buffer_release_on_scan": True},
}
SPELLING_PARAMS: Final[Mapping[Spelling, Mapping[str, JsonValue]]] = {
    "noma_v2": {"guardrail": "noma_v2"},
    "noma_use_v2": {"guardrail": "noma", "use_v2": True},
    "stored": {"guardrail": "noma_v2"},
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


def _event(name: str | None, payload: Mapping[str, JsonValue]) -> bytes:
    prefix: Final = f"event: {name}\n" if name is not None else ""
    return f"{prefix}data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _chat_stream(marker: str) -> tuple[bytes, ...]:
    chunk: Final = {"id": f"chatcmpl-{marker}", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini"}
    deltas: Final = (
        {"role": "assistant", "content": marker},
        *({"content": piece} for piece in PIECES),
    )
    return (
        *(
            _event(None, {**chunk, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
            for delta in deltas
        ),
        _event(None, {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        b"data: [DONE]\n\n",
    )


def _messages_stream(marker: str) -> tuple[bytes, ...]:
    events: Final[tuple[dict[str, JsonValue], ...]] = (
        {
            "type": "message_start",
            "message": {
                "id": f"msg-{marker}",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5-20250929",
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


def _responses_stream(marker: str) -> tuple[bytes, ...]:
    item_id: Final = f"msg-{marker}"
    text: Final = _full_text(marker)
    part: Final[dict[str, JsonValue]] = {"type": "output_text", "text": text, "annotations": []}
    output: Final[dict[str, JsonValue]] = {
        "type": "message",
        "id": item_id,
        "status": "completed",
        "role": "assistant",
        "content": [part],
    }
    response: Final[dict[str, JsonValue]] = {
        "id": f"resp-{marker}",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [output],
        "usage": {"input_tokens": 2, "output_tokens": 15, "total_tokens": 17},
    }
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


def _provider(request: Request) -> Reply:
    body: Final = JSON_OBJECT.validate_json(request.body)
    marker: Final = _marker(body)
    assert body.get("stream") is True, request.body
    streams: Final = {
        "/v1/chat/completions": _chat_stream,
        "/v1/messages": _messages_stream,
        "/v1/responses": _responses_stream,
    }
    if request.target not in streams:
        return Reply(status=404, body=json.dumps({"error": f"unexpected target {request.target}"}).encode())
    return Reply(content_type="text/event-stream", chunks=streams[request.target](marker), pause_between_chunks=0.05)


@dataclass(frozen=True, slots=True)
class Scan:
    guardrail: str
    input_type: str
    text: str


@dataclass(slots=True)
class NomaDouble:
    """Scripted Noma v2 `/litellm/guardrail` endpoint; the action word in the stream's marker picks the verdict."""

    scans: list[Scan] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

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
        verdict: Final = self._verdict(text, earlier, texts)
        return Reply(body=json.dumps(verdict).encode())

    @staticmethod
    def _verdict(text: str, earlier: int, texts: Sequence[str]) -> dict[str, JsonValue]:
        marker: Final = next((word for word in text.split(" ") if word.startswith("noma-")), None)
        action: Final = "pass" if marker is None else _action(marker)
        if action == "block" and SECRET in text:
            return {"action": "BLOCKED", "reason": "synthetic email policy"}
        if action == "late" and earlier > 0:
            return {"action": "BLOCKED", "reason": "synthetic late verdict"}
        if action == "mask" and SECRET in text:
            return {"action": "GUARDRAIL_INTERVENED", "texts": [part.replace(SECRET, MASK) for part in texts]}
        return {"action": "NONE"}

    def for_marker(self, marker: str) -> tuple[Scan, ...]:
        with self.lock:
            return tuple(scan for scan in self.scans if scan.text.startswith(marker))


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    noma: NomaDouble
    provider: Wire
    names: Mapping[tuple[Spelling, Variant], str]


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
        stored: Final = tuple(
            _store(session_gateway, noma_wire.url, name, variant)
            for (spelling, variant), name in names.items()
            if spelling == "stored"
        )
        try:
            directory: Final = tmp_path_factory.mktemp("noma-v2-streaming")
            config_path: Final = directory / "noma-v2-streaming.yaml"
            config_path.write_text(yaml.safe_dump(_config(provider.url, noma_wire.url, names)))
            with owned_proxy(session_gateway, directory, {}, config=config_path, workers=2) as candidate:
                yield Rig(candidate, noma, provider, names)
        finally:
            for identity in stored:
                deleted = session_gateway.request("DELETE", f"/guardrails/{identity}")
                assert deleted.status_code == 200, deleted.text


@dataclass(frozen=True, slots=True)
class Streamed:
    marker: str
    response: httpx.Response
    content: str
    scans: tuple[Scan, ...]


def _event_text(event: Mapping[str, JsonValue], endpoint: Endpoint) -> str:
    if endpoint == "responses":
        return string_value(event.get("delta", "")) if event.get("type") == "response.output_text.delta" else ""
    if endpoint == "messages":
        return string_value(object_value(event.get("delta", {})).get("text", ""))
    choices: Final = list_value(event.get("choices", [{}])) or [{}]
    return string_value(object_value(object_value(choices[0]).get("delta", {})).get("content", "") or "")


def _stream(
    rig: Rig, endpoint: Endpoint, spelling: Spelling, variant: Variant, action: Action, status: int = 200
) -> Streamed:
    marker: Final = f"noma-{action}-{uuid.uuid4().hex}"
    guardrail: Final = rig.names[(spelling, variant)]
    prompt: Final[dict[str, JsonValue]] = (
        {"input": marker} if endpoint == "responses" else {"messages": [{"role": "user", "content": marker}]}
    )
    response: Final = rig.gateway.request(
        "POST",
        {"chat": "/v1/chat/completions", "messages": "/v1/messages", "responses": "/v1/responses"}[endpoint],
        {
            "model": MODELS[endpoint],
            **prompt,
            **({"max_tokens": 64} if endpoint == "messages" else {}),
            "stream": True,
            "guardrails": [guardrail],
        },
    )
    assert response.status_code == status, response.text
    events: Final = tuple(
        JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    )
    scans: Final = rig.noma.for_marker(marker)
    assert all(scan.guardrail == guardrail and scan.input_type == "response" for scan in scans), scans
    return Streamed(marker, response, "".join(_event_text(event, endpoint) for event in events), scans)


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_buffer_until_moderated_keeps_flagged_text_from_the_client(
    rig: Rig, endpoint: Endpoint, spelling: Spelling
) -> None:
    unbuffered: Final = _stream(rig, endpoint, spelling, "default", "block")
    assert SECRET in unbuffered.content, unbuffered.response.text
    assert "Request blocked by Noma guardrail" in unbuffered.response.text, unbuffered.response.text

    buffered: Final = _stream(rig, endpoint, spelling, "buffered", "block", BUFFERED_BLOCK_STATUS[endpoint])
    assert buffered.content == "", buffered.response.text
    assert SECRET not in buffered.response.text, buffered.response.text
    assert "Request blocked by Noma guardrail" in buffered.response.text, buffered.response.text
    assert [scan.text for scan in buffered.scans] == [_full_text(buffered.marker)], buffered.scans


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("variant", ["end_only", "sparse"])
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
    assert "Request blocked by Noma guardrail" in released.response.text, released.response.text

    held: Final = _stream(rig, endpoint, spelling, "buffered", "late")
    assert held.content == _full_text(held.marker), held.response.text
    assert len(held.scans) == 1, held.scans


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_buffer_until_moderated_delivers_the_masked_text(rig: Rig, endpoint: Endpoint, spelling: Spelling) -> None:
    pytest.skip("BUG: LIT-9559 a buffered stream releases the original chunks instead of the guardrail's masked text")
    unbuffered: Final = _stream(rig, endpoint, spelling, "default", "mask")
    assert SECRET in unbuffered.content, unbuffered.response.text

    buffered: Final = _stream(rig, endpoint, spelling, "buffered", "mask")
    assert buffered.content == _full_text(buffered.marker).replace(SECRET, MASK), buffered.response.text
    assert SECRET not in buffered.response.text, buffered.response.text
