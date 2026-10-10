import asyncio
import json
import re
import signal
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import anthropic
import httpx
import psutil
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment, held, list_value, object_value
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, group_members, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from openai import AsyncOpenAI, OpenAI
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

SECRET_WORD: Final = "pipelinesecretword"
REWRITTEN: Final = "[PIPELINE_REDACTED]"
KEYWORD_MASK: Final = "[KEYWORD_REDACTED]"
TOKEN: Final = re.compile(rb"plive-[0-9a-f]{32}-\d+")
HOLD: Final = "holdme"
DROP: Final = "dropme"
PROVIDER_FAILURE: Final = "providerfails"
GENERIC_PATH: Final = "/beta/litellm_basic_guardrail_api"

Endpoint: TypeAlias = Literal["chat", "responses", "messages"]
Outcome: TypeAlias = Literal["rewritten", "provider", "keyword_masked"]
ModelKind: TypeAlias = Literal["openai", "gemini", "text"]
Sink: TypeAlias = Callable[[str], None]

PATHS: Final[Mapping[Endpoint, str]] = MappingProxyType(
    {"chat": "/v1/chat/completions", "responses": "/v1/responses", "messages": "/v1/messages"}
)


def _token(index: int = 0) -> str:
    return f"plive-{uuid.uuid4().hex}-{index}"


def _provider_text(token: str) -> str:
    return f"{SECRET_WORD} {token}"


def _prompt(token: str, *markers: str) -> str:
    return " ".join(("synthetic prompt", token, *markers))


def _token_of(body: bytes) -> str:
    found: Final = TOKEN.search(body)
    assert found is not None, body
    return found.group().decode()


class _Log:
    def __init__(self, wire: Wire) -> None:
        self.wire: Final = wire
        self._lock: Final = threading.Lock()
        self.seen: tuple[Request, ...] = ()

    def matching(self, needle: str) -> tuple[Request, ...]:
        with self._lock:
            self.seen = (*self.seen, *self.wire.drain())
            return tuple(request for request in self.seen if needle.encode() in request.body)


class _Gates:
    def __init__(self) -> None:
        self._lock: Final = threading.Lock()
        self.held: Mapping[str, threading.Event] = MappingProxyType({})

    def hold(self, token: str) -> threading.Event:
        gate: Final = threading.Event()
        with self._lock:
            self.held = MappingProxyType({**self.held, token: gate})
        return gate

    def release_all(self) -> None:
        for gate in self.held.values():
            gate.set()


class _Received:
    def __init__(self) -> None:
        self._lock: Final = threading.Lock()
        self.pieces: tuple[str, ...] = ()

    def add(self, piece: str) -> None:
        with self._lock:
            self.pieces = (*self.pieces, piece)

    @property
    def text(self) -> str:
        return "".join(self.pieces)


def _sse(payload: Mapping[str, JsonValue]) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def _chat_stream(text: str) -> tuple[bytes, ...]:
    identity: Final = "chatcmpl-" + uuid.uuid4().hex

    def frame(delta: Mapping[str, JsonValue], finish: str | None) -> bytes:
        return _sse(
            {
                "id": identity,
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish}],
            }
        )

    return (
        frame({"role": "assistant", "content": text}, None),
        frame({"content": " tail"}, "stop"),
        b"data: [DONE]\n\n",
    )


def _completion_stream(text: str) -> tuple[bytes, ...]:
    identity: Final = "cmpl-" + uuid.uuid4().hex

    def frame(piece: str, finish: str | None) -> bytes:
        return _sse(
            {
                "id": identity,
                "object": "text_completion",
                "created": 1,
                "model": "gpt-3.5-turbo-instruct",
                "choices": [{"text": piece, "index": 0, "logprobs": None, "finish_reason": finish}],
            }
        )

    return (frame(text, None), frame(" tail", "stop"), b"data: [DONE]\n\n")


def _responses_object(identity: str, text: str, status: str) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": status,
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "id": "msg_" + identity,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 11,
            "output_tokens": 4,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _responses_stream(text: str) -> tuple[bytes, ...]:
    identity: Final = "resp_" + uuid.uuid4().hex
    completed: Final = _responses_object(identity, text + " tail", "completed")
    events: Final = (
        {"type": "response.created", "response": {**completed, "status": "in_progress", "output": [], "usage": None}},
        {
            "type": "response.output_text.delta",
            "item_id": "msg_" + identity,
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_" + identity,
            "output_index": 0,
            "content_index": 0,
            "delta": " tail",
        },
        {"type": "response.completed", "response": completed},
    )
    encoded: Final = tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)
    return (encoded[0] + encoded[1], encoded[2] + encoded[3])


def _gemini_stream(text: str) -> tuple[bytes, ...]:
    def frame(piece: str, finish: str | None) -> bytes:
        candidate: Final = {
            "content": {"parts": [{"text": piece}], "role": "model"},
            "index": 0,
            **({"finishReason": finish} if finish else {}),
        }
        payload: Final = {
            "candidates": [candidate],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
            "modelVersion": "gemini-2.5-flash",
        }
        return b"data: " + json.dumps(payload).encode() + b"\r\n\r\n"

    return (frame(text, None), frame(" tail", "STOP"))


def _chat_body(text: str) -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": text + " tail"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
        }
    ).encode()


def _stream_frames(path: str, text: str) -> tuple[bytes, ...]:
    if "streamGenerateContent" in path:
        return _gemini_stream(text)
    if path.endswith("/responses"):
        return _responses_stream(text)
    if path.endswith("/chat/completions"):
        return _chat_stream(text)
    return _completion_stream(text)


def _provider(gates: _Gates) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        token: Final = _token_of(request.body)
        text: Final = _provider_text(token)
        path: Final = request.target.split("?")[0]
        if PROVIDER_FAILURE.encode() in request.body:
            return Reply(
                status=500,
                body=json.dumps({"error": {"message": "synthetic provider failure", "type": "server_error"}}).encode(),
            )
        streamed: Final = (
            "streamGenerateContent" in path or object_value(json.loads(request.body)).get("stream") is True
        )
        if not streamed:
            return Reply(
                body=json.dumps(_responses_object("resp_" + uuid.uuid4().hex, text + " tail", "completed")).encode()
                if path.endswith("/responses")
                else _chat_body(text)
            )
        return Reply(
            content_type="text/event-stream",
            chunks=_stream_frames(path, text),
            gate_after_first=gates.held.get(token) if HOLD.encode() in request.body else None,
            abort_after=1 if DROP.encode() in request.body else None,
        )

    return respond


def _rewritten(text: str) -> str:
    return text.replace(SECRET_WORD, REWRITTEN)


def _generic_reply(behavior: str, body: Mapping[str, JsonValue]) -> Reply:
    texts: Final = body.get("texts") or []
    assert isinstance(texts, list), body
    if behavior == "rw":
        return Reply(
            body=json.dumps(
                {"action": "GUARDRAIL_INTERVENED", "texts": [_rewritten(str(text)) for text in texts]}
            ).encode()
        )
    if behavior == "fail500":
        return Reply(status=500, body=b'{"error": "synthetic guardrail failure"}')
    if behavior == "malformed":
        return Reply(body=b"<html>synthetic malformed guardrail body</html>", content_type="text/html")
    if behavior == "blocked":
        return Reply(body=json.dumps({"action": "BLOCKED", "blocked_reason": "synthetic block"}).encode())
    if behavior == "outage" and int(_token_of(json.dumps(body).encode()).rsplit("-", 1)[1]) % 2 == 0:
        return Reply(status=503, body=b'{"error": "synthetic guardrail outage"}')
    return Reply(body=b'{"action": "NONE"}')


def _crowdstrike_reply(body: Mapping[str, JsonValue]) -> Reply:
    guard_input: Final = object_value(body["guard_input"])
    messages: Final = guard_input.get("messages") or []
    assert isinstance(messages, list), body
    redacted: Final = [
        {**object_value(message), "content": _rewritten(str(object_value(message).get("content", "")))}
        for message in messages
    ]
    return Reply(
        body=json.dumps(
            {"result": {"blocked": False, "transformed": True, "guard_output": {"messages": redacted}}}
        ).encode()
    )


def _bedrock_reply(body: Mapping[str, JsonValue]) -> Reply:
    content: Final = body.get("content") or []
    assert isinstance(content, list), body
    texts: Final = tuple(str(object_value(object_value(item)["text"])["text"]) for item in content)
    return Reply(
        body=json.dumps(
            {
                "action": "GUARDRAIL_INTERVENED",
                "outputs": [{"text": _rewritten(text)} for text in texts],
                "assessments": [
                    {
                        "sensitiveInformationPolicy": {
                            "piiEntities": [{"type": "NAME", "match": SECRET_WORD, "action": "ANONYMIZED"}]
                        }
                    }
                ],
            }
        ).encode()
    )


def _peer(request: Request) -> Reply:
    behavior: Final = request.target.split("/")[2]
    body: Final = object_value(json.loads(request.body))
    if behavior == "crowdstrike":
        assert request.target.endswith("/v1/guard_chat_completions"), request.target
        return _crowdstrike_reply(body)
    if behavior == "bedrock":
        assert request.target.endswith("/version/DRAFT/apply"), request.target
        return _bedrock_reply(body)
    if behavior == "straiker":
        assert request.target.endswith("/api/v1/detect/webhook"), request.target
        return Reply(body=b'{"action": "NONE"}')
    assert request.target.endswith(GENERIC_PATH), request.target
    return _generic_reply(behavior, body)


def _flag(value: JsonValue) -> Mapping[str, JsonValue]:
    return MappingProxyType({"streaming_buffer_until_moderated": value})


FLAG_FALSE: Final = _flag(False)
FLAG_TRUE: Final = _flag(True)
UNSET: Final[Mapping[str, JsonValue]] = MappingProxyType({})


def _generic(
    name: str, peer: str, behavior: str, params: Mapping[str, JsonValue], mode: str = "post_call"
) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "generic_guardrail_api",
            "mode": mode,
            "default_on": False,
            "api_base": f"{peer}/{name}/{behavior}",
            "api_key": "synthetic-guardrail-key",
            **params,
        },
    }


def _crowdstrike(name: str, peer: str, params: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "crowdstrike_aidr",
            "mode": "post_call",
            "default_on": False,
            "api_base": f"{peer}/{name}/crowdstrike",
            "api_key": "synthetic-crowdstrike-key",
            **params,
        },
    }


def _bedrock(name: str, peer: str, params: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "bedrock",
            "mode": "post_call",
            "default_on": False,
            "guardrailIdentifier": "synthetic-guardrail",
            "guardrailVersion": "DRAFT",
            "aws_region_name": "us-east-1",
            "aws_access_key_id": "AKIASYNTHETICGUARDRAIL",
            "aws_secret_access_key": "synthetic-guardrail-secret",
            "aws_bedrock_runtime_endpoint": f"{peer}/{name}/bedrock",
            **params,
        },
    }


def _straiker(name: str, peer: str) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "straiker",
            "mode": "post_call",
            "default_on": False,
            "api_base": f"{peer}/{name}/straiker",
            "api_key": "synthetic-straiker-key",
            "default_app": "synthetic-app",
            **FLAG_FALSE,
        },
    }


def _content_filter(name: str) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "litellm_content_filter",
            "mode": "post_call",
            "default_on": False,
            "blocked_words": [{"keyword": SECRET_WORD, "action": "MASK"}],
            **FLAG_FALSE,
        },
    }


def _step(guardrail: str, **actions: str) -> dict[str, JsonValue]:
    return {"guardrail": guardrail, "on_pass": "allow", "on_fail": "next", **actions}


def _policy(steps: tuple[dict[str, JsonValue], ...]) -> dict[str, JsonValue]:
    return {
        "guardrails": {"add": list(dict.fromkeys(str(step["guardrail"]) for step in steps))},
        "pipeline": {"mode": "post_call", "steps": list(steps)},
    }


@dataclass(frozen=True, slots=True)
class _RowSpec:
    rails: tuple[dict[str, JsonValue], ...]
    pipelines: tuple[tuple[dict[str, JsonValue], ...], ...]
    model: ModelKind = "openai"


def _single(name: str, rail: dict[str, JsonValue], **actions: str) -> _RowSpec:
    return _RowSpec((rail,), ((_step(name, **actions),),))


def _row_specs(peer: str, suffix: str) -> Mapping[str, _RowSpec]:
    def rail(row: str, index: int = 0) -> str:
        return f"plive-{row.lower()}-{index}-{suffix}"

    def generic(row: str, behavior: str, params: Mapping[str, JsonValue], **actions: str) -> _RowSpec:
        return _single(rail(row), _generic(rail(row), peer, behavior, params), **actions)

    def unattached(row: str, params: Mapping[str, JsonValue], mode: str = "post_call") -> _RowSpec:
        return _RowSpec((_generic(rail(row), peer, "allow", params, mode),), ())

    hold_rows: Final = {row: generic(row, "allow", FLAG_FALSE) for row in ("A1", "A2", "A3", "A4", "A5", "A6")}
    return MappingProxyType(
        {
            **hold_rows,
            "A7": replace(generic("A7", "allow", FLAG_FALSE), model="gemini"),
            "A8": replace(generic("A8", "allow", FLAG_FALSE), model="text"),
            "B1": generic("B1", "rw", FLAG_FALSE),
            "B2": generic("B2", "rw", FLAG_TRUE),
            "B3": generic("B3", "rw", UNSET),
            "B4": generic("B4", "rw", _flag(None)),
            "B5": generic("B5", "rw", FLAG_FALSE, on_fail="block"),
            "B6": generic("B6", "rw", FLAG_FALSE, on_error="block"),
            "B7": _RowSpec(
                (_generic(rail("B7", 0), peer, "rw", FLAG_FALSE), _generic(rail("B7", 1), peer, "rw", FLAG_TRUE)),
                ((_step(rail("B7", 0), on_pass="next"), _step(rail("B7", 1))),),
            ),
            "B8": _RowSpec(
                (_generic(rail("B8", 0), peer, "rw", FLAG_FALSE), _generic(rail("B8", 1), peer, "rw", FLAG_TRUE)),
                ((_step(rail("B8", 0)),), (_step(rail("B8", 1)),)),
            ),
            "B10": generic("B10", "rw", FLAG_FALSE),
            "B11": generic("B11", "rw", FLAG_FALSE),
            "B12": generic("B12", "rw", FLAG_FALSE),
            "B13": generic("B13", "rw", FLAG_FALSE),
            "B14": generic("B14", "rw", FLAG_FALSE),
            "B15": generic("B15", "rw", {"optional_params": dict(FLAG_FALSE)}),
            "B16": generic("B16", "rw", _flag("")),
            "B17": generic("B17", "rw", _flag("false")),
            "B18": generic("B18", "rw", _flag(0)),
            "B19": generic("B19", "rw", FLAG_FALSE, on_fail="modify_response"),
            "C1": _single(rail("C1"), _crowdstrike(rail("C1"), peer, UNSET)),
            "C2": _single(rail("C2"), _crowdstrike(rail("C2"), peer, FLAG_FALSE)),
            "C3": _single(rail("C3"), _crowdstrike(rail("C3"), peer, FLAG_TRUE)),
            "C4": _single(rail("C4"), _bedrock(rail("C4"), peer, UNSET)),
            "C5": _single(rail("C5"), _bedrock(rail("C5"), peer, FLAG_FALSE)),
            "C6": _single(rail("C6"), _bedrock(rail("C6"), peer, {**FLAG_FALSE, "mask_response_content": True})),
            "C7": _single(rail("C7"), _content_filter(rail("C7"))),
            "C8": _single(rail("C8"), _straiker(rail("C8"), peer)),
            "D1": generic("D1", "fail500", FLAG_FALSE),
            "D2": generic("D2", "malformed", FLAG_FALSE),
            "D3": generic("D3", "blocked", FLAG_FALSE),
            "D4-chat": generic("D4-chat", "allow", FLAG_FALSE),
            "D4-responses": generic("D4-responses", "allow", FLAG_FALSE),
            "D4-messages": generic("D4-messages", "allow", FLAG_FALSE),
            "E1": generic("E1", "allow", FLAG_FALSE),
            "F1": generic("F1", "outage", FLAG_FALSE),
            "H2-chat": unattached("H2-chat", UNSET, "pre_call"),
            "H2-responses": unattached("H2-responses", UNSET, "pre_call"),
            "H2-messages": unattached("H2-messages", UNSET, "pre_call"),
            "H3": _RowSpec((), ()),
            "I1": unattached("I1", FLAG_TRUE),
            "I2": unattached("I2", FLAG_FALSE),
            "I3": unattached("I3", UNSET),
        }
    )


def _deployment(kind: ModelKind, provider: str) -> dict[str, JsonValue]:
    if kind == "gemini":
        return {"model": "gemini/gemini-2.5-flash", "api_base": provider, "api_key": "synthetic-gemini-key"}
    if kind == "text":
        return {
            "model": "text-completion-openai/gpt-3.5-turbo-instruct",
            "api_base": f"{provider}/v1",
            "api_key": "synthetic-provider-key",
        }
    return {"model": "openai/gpt-4o-mini", "api_base": f"{provider}/v1", "api_key": "synthetic-provider-key"}


def _row_policies(
    specs: Mapping[str, _RowSpec], models: Mapping[str, str]
) -> Iterator[tuple[str, str, dict[str, JsonValue]]]:
    for row, spec in specs.items():
        for index, steps in enumerate(spec.pipelines):
            yield f"{models[row]}-p{index}", models[row], _policy(steps)


def _row_rails(specs: Mapping[str, _RowSpec]) -> Iterator[dict[str, JsonValue]]:
    for spec in specs.values():
        yield from spec.rails


def _base_config() -> dict[str, JsonValue]:
    return object_value(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))


def _matrix_config(specs: Mapping[str, _RowSpec], models: Mapping[str, str], provider: str) -> dict[str, JsonValue]:
    policies: Final = tuple(_row_policies(specs, models))
    return {
        **_base_config(),
        "guardrails": list(_row_rails(specs)),
        "model_list": [
            {"model_name": models[row], "litellm_params": _deployment(spec.model, provider)}
            for row, spec in specs.items()
        ],
        "policies": {name: policy for name, _model, policy in policies},
        "policy_attachments": [{"policy": name, "models": [model]} for name, model, _policy_body in policies],
    }


@dataclass(frozen=True, slots=True)
class _Stage:
    gateway: Gateway
    gates: _Gates
    provider: _Log

    def upstream(self, token: str) -> tuple[Request, ...]:
        return self.provider.matching(token)


@dataclass(frozen=True, slots=True)
class _Rig:
    stage: _Stage
    peer: _Log
    models: Mapping[str, str]
    rails: Mapping[str, tuple[str, ...]]

    @property
    def gateway(self) -> Gateway:
        return self.stage.gateway

    def scans(self, row: str, token: str, index: int = 0) -> tuple[Request, ...]:
        prefix: Final = f"/{self.rails[row][index]}/"
        return tuple(request for request in self.peer.matching(token) if request.target.startswith(prefix))

    def upstream(self, token: str) -> tuple[Request, ...]:
        return self.stage.upstream(token)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("pipeline-live-streaming")
    gates: Final = _Gates()
    suffix: Final = uuid.uuid4().hex[:8]
    try:
        with (
            gateway_from_environment() as root,
            wire_server(_provider(gates)) as provider,
            wire_server(_peer) as peer,
        ):
            specs: Final = _row_specs(peer.url, suffix)
            models: Final = MappingProxyType({row: f"plive-{row.lower()}-{suffix}" for row in specs})
            config: Final = directory / "pipeline-live-streaming.yaml"
            config.write_text(yaml.safe_dump(_matrix_config(specs, models, provider.url)))
            rails: Final = MappingProxyType(
                {row: tuple(str(rail["guardrail_name"]) for rail in spec.rails) for row, spec in specs.items()}
            )
            with owned_proxy_process(root, directory, {}, config=config, workers=2) as owned:
                yield _Rig(_Stage(owned.gateway, gates, _Log(provider)), _Log(peer), models, rails)
    finally:
        gates.release_all()


def _request_body(endpoint: Endpoint, model: str, prompt: str, stream: bool) -> dict[str, JsonValue]:
    if endpoint == "chat":
        return {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": stream}
    if endpoint == "responses":
        return {"model": model, "input": prompt, "stream": stream}
    return {"model": model, "max_tokens": 64, "messages": [{"role": "user", "content": prompt}], "stream": stream}


def _call(
    gateway: Gateway,
    endpoint: Endpoint,
    model: str,
    prompt: str,
    *,
    stream: bool = True,
    extra: Mapping[str, JsonValue] = UNSET,
) -> httpx.Response:
    return gateway.request("POST", PATHS[endpoint], {**_request_body(endpoint, model, prompt, stream), **extra})


def _parts_text(parts: JsonValue) -> str:
    return "".join(str(object_value(part).get("text") or "") for part in list_value(parts or []))


def _choice_text(choice: JsonValue) -> str:
    fields: Final = object_value(choice)
    holder: Final = fields.get("delta") or fields.get("message")
    if holder is None:
        return str(fields.get("text") or "")
    return str(object_value(holder).get("content") or "")


def _candidate_text(candidate: JsonValue) -> str:
    return _parts_text(object_value(object_value(candidate).get("content") or {}).get("parts"))


def _event_text(event: Mapping[str, JsonValue]) -> str:
    kind: Final = event.get("type")
    if kind == "response.output_text.delta":
        return str(event["delta"])
    if kind == "content_block_delta":
        return str(object_value(event["delta"]).get("text") or "")
    if kind == "message":
        return _parts_text(event.get("content"))
    if event.get("object") == "response":
        return "".join(_parts_text(object_value(item).get("content")) for item in list_value(event.get("output") or []))
    choices: Final = "".join(_choice_text(choice) for choice in list_value(event.get("choices") or []))
    return choices + "".join(_candidate_text(candidate) for candidate in list_value(event.get("candidates") or []))


def _line_text(line: str) -> str:
    payload: Final = line.removeprefix("data:").strip()
    if not line.startswith("data:") or not payload.startswith("{"):
        return ""
    return _event_text(object_value(json.loads(payload)))


def _text(response: httpx.Response) -> str:
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        return "".join(_line_text(line) for line in response.text.splitlines())
    return _event_text(object_value(response.json()))


def _assert_outcome(text: str, token: str, outcome: Outcome) -> None:
    if outcome == "provider":
        assert f"{_provider_text(token)} tail" in text and REWRITTEN not in text, text
        return
    mask: Final = REWRITTEN if outcome == "rewritten" else KEYWORD_MASK
    assert f"{mask} {token}" in text and SECRET_WORD not in text, text


@dataclass(frozen=True, slots=True)
class _SpendRow:
    status: str
    cache_hit: bool
    verdicts: Mapping[str, tuple[str, ...]]


def _verdicts(metadata: JsonValue, rail: str) -> tuple[str, ...]:
    entries: Final = object_value(metadata).get("guardrail_information") or []
    return tuple(
        str(entry["guardrail_status"])
        for entry in (object_value(value) for value in list_value(entries))
        if entry.get("guardrail_name") == rail
    )


def _read_spend_rows(model: str) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(
        read_rows(
            'SELECT status, cache_hit, metadata FROM "LiteLLM_SpendLogs" WHERE model_group=%s ORDER BY "startTime"',
            (model,),
        )
    )


def _spend_rows(model: str, rails: tuple[str, ...], count: int) -> tuple[_SpendRow, ...]:
    found: Final = (
        held(lambda: _read_spend_rows(model), lambda rows: not rows, holding=10)
        if count == 0
        else eventually(
            lambda: _read_spend_rows(model),
            lambda rows: len(rows) == count,
            seconds=70,
            return_last_on_timeout=True,
        )
    )
    return tuple(
        _SpendRow(
            str(row["status"]),
            str(row["cache_hit"]) == "True",
            MappingProxyType({rail: _verdicts(row["metadata"], rail) for rail in rails}),
        )
        for row in found
    )


def _assert_cached_twin_rows(
    model: str, rails: tuple[str, ...], verdict: str, *, cache_hit_verdict_recorded: bool
) -> None:
    rows: Final = _spend_rows(model, rails, 2)
    expected: Final = {rail: (verdict,) for rail in rails}
    assert len(rows) == 2 and rows[0].verdicts == expected, rows
    assert [spend.cache_hit for spend in rows] == [False, True], rows
    assert not cache_hit_verdict_recorded or rows[1].verdicts == expected, rows


def _assert_twin(
    rig: _Rig,
    row: str,
    endpoint: Endpoint,
    outcome: Outcome,
    *,
    stream: bool = True,
    verdict: str = "success",
    scans_per_request: int = 1,
) -> None:
    token: Final = _token()
    model: Final = rig.models[row]
    first: Final = _call(rig.gateway, endpoint, model, _prompt(token), stream=stream)
    second: Final = _call(rig.gateway, endpoint, model, _prompt(token), stream=stream)
    assert first.status_code == second.status_code == 200, (first.text, second.text)
    _assert_outcome(_text(first), token, outcome)
    _assert_outcome(_text(second), token, outcome)
    assert len(rig.upstream(token)) == 1, "the identical second request must be served from the cache"
    expected_scans: Final = (2 * scans_per_request,) * len(rig.rails[row])
    scans: Final = eventually(
        lambda: tuple(len(rig.scans(row, token, index)) for index in range(len(rig.rails[row]))),
        lambda counts: counts == expected_scans,
        seconds=10,
        return_last_on_timeout=True,
    )
    assert scans == expected_scans, scans
    _assert_cached_twin_rows(model, rig.rails[row], verdict, cache_hit_verdict_recorded=stream)


Reader: TypeAlias = Callable[[Gateway, str, str, Sink], None]


def _read_lines(gateway: Gateway, path: str, body: Mapping[str, JsonValue], sink: Sink) -> None:
    with gateway.client.stream(
        "POST", path, json=dict(body), headers={"Authorization": f"Bearer {gateway.key}"}
    ) as response:
        assert response.status_code == 200, response.read()
        for line in response.iter_lines():
            sink(_line_text(line))


def _chat_httpx(gateway: Gateway, model: str, prompt: str, sink: Sink) -> None:
    _read_lines(gateway, PATHS["chat"], _request_body("chat", model, prompt, True), sink)


def _responses_httpx(gateway: Gateway, model: str, prompt: str, sink: Sink) -> None:
    _read_lines(gateway, PATHS["responses"], _request_body("responses", model, prompt, True), sink)


def _messages_httpx(gateway: Gateway, model: str, prompt: str, sink: Sink) -> None:
    _read_lines(gateway, PATHS["messages"], _request_body("messages", model, prompt, True), sink)


def _gemini_httpx(gateway: Gateway, model: str, prompt: str, sink: Sink) -> None:
    body: Final = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    _read_lines(gateway, f"/v1beta/models/{model}:streamGenerateContent?alt=sse", body, sink)


def _completions_httpx(gateway: Gateway, model: str, prompt: str, sink: Sink) -> None:
    _read_lines(gateway, "/v1/completions", {"model": model, "prompt": prompt, "stream": True}, sink)


def _opting_in(guardrails: tuple[str, ...]) -> Reader:
    def read(gateway: Gateway, model: str, prompt: str, sink: Sink) -> None:
        body: Final = {**_request_body("chat", model, prompt, True), "guardrails": list(guardrails)}
        _read_lines(gateway, PATHS["chat"], body, sink)

    return read


def _raw_reader(endpoint: Endpoint, raw: _Received) -> Reader:
    def read(gateway: Gateway, model: str, prompt: str, sink: Sink) -> None:
        with gateway.client.stream(
            "POST",
            PATHS[endpoint],
            json=_request_body(endpoint, model, prompt, True),
            headers={"Authorization": f"Bearer {gateway.key}"},
        ) as response:
            raw.add(f"status {response.status_code}\n")
            for line in response.iter_lines():
                raw.add(line + "\n")
                sink(_line_text(line))

    return read


def _chat_async_openai_sdk(gateway: Gateway, model: str, prompt: str, sink: Sink) -> None:
    async def read() -> None:
        client: Final = AsyncOpenAI(
            base_url=str(gateway.client.base_url) + "/v1",
            api_key=gateway.key,
            max_retries=0,
            http_client=httpx.AsyncClient(trust_env=False, timeout=30),
        )
        async with client:
            stream: Final = await client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], stream=True
            )
            async for chunk in stream:
                sink("".join(choice.delta.content or "" for choice in chunk.choices))

    asyncio.run(read())


def _responses_openai_sdk(gateway: Gateway, model: str, prompt: str, sink: Sink) -> None:
    client: Final = OpenAI(
        base_url=str(gateway.client.base_url) + "/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=30),
    )
    with client:
        for event in client.responses.create(model=model, input=prompt, stream=True):
            if event.type == "response.output_text.delta":
                sink(event.delta)


def _messages_anthropic_sdk(gateway: Gateway, model: str, prompt: str, sink: Sink) -> None:
    client: Final = anthropic.Anthropic(
        base_url=str(gateway.client.base_url),
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=30),
    )
    with client:
        for event in client.messages.create(
            model=model, max_tokens=64, messages=[{"role": "user", "content": prompt}], stream=True
        ):
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                sink(event.delta.text)


@dataclass(frozen=True, slots=True)
class _HeldStream:
    released_while_held: str
    received: str


def _released(stage: _Stage, reader: Reader, model: str, token: str, *markers: str) -> _HeldStream:
    gate: Final = stage.gates.hold(token)
    received: Final = _Received()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future: Final = pool.submit(reader, stage.gateway, model, _prompt(token, HOLD, *markers), received.add)
        try:
            early: Final = eventually(
                lambda: received.text,
                lambda text: _provider_text(token) in text,
                seconds=8,
                return_last_on_timeout=True,
            )
        finally:
            gate.set()
        future.result(timeout=30)
    return _HeldStream(early, received.text)


def _withheld(stage: _Stage, reader: Reader, model: str, token: str) -> str:
    gate: Final = stage.gates.hold(token)
    received: Final = _Received()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future: Final = pool.submit(reader, stage.gateway, model, _prompt(token, HOLD), received.add)
        try:
            eventually(lambda: stage.upstream(token), lambda requests: len(requests) == 1, seconds=15)
            held(lambda: received.text, lambda text: _provider_text(token) not in text, holding=3)
        finally:
            gate.set()
        future.result(timeout=30)
    return received.text


def _replayed(stage: _Stage, reader: Reader, model: str, token: str) -> str:
    received: Final = _Received()
    reader(stage.gateway, model, _prompt(token, HOLD), received.add)
    return received.text


@dataclass(frozen=True, slots=True)
class _LiveRelease:
    released_while_held: bool
    whole_received: bool
    upstream_requests: int
    scans: int
    replay_whole: bool
    upstream_requests_after_replay: int
    scans_after_replay: int
    cache_hits: tuple[bool, ...]


def _scan_count(rig: _Rig, row: str, token: str, expected: int) -> int:
    return eventually(
        lambda: len(rig.scans(row, token)),
        lambda count: count == expected,
        seconds=10,
        return_last_on_timeout=True,
    )


LIVE_RELEASE: Final = _LiveRelease(True, True, 1, 1, True, 1, 2, (False, True))
RECORDED: Final = frozenset({("success",)})
NEVER_SCANNED: Final = frozenset({()})
LOGGED_BEFORE_THE_SCAN_ENDS: Final = frozenset({(), ("success",)})


@pytest.mark.parametrize(
    ("row", "reader", "expected", "verdicts"),
    (
        pytest.param("A1", _chat_httpx, LIVE_RELEASE, RECORDED, id="A1-chat-httpx"),
        pytest.param("A2", _chat_async_openai_sdk, LIVE_RELEASE, RECORDED, id="A2-chat-async-openai-sdk"),
        pytest.param("A3", _responses_httpx, LIVE_RELEASE, RECORDED, id="A3-responses-httpx"),
        pytest.param("A4", _responses_openai_sdk, LIVE_RELEASE, RECORDED, id="A4-responses-openai-sdk"),
        pytest.param("A5", _messages_httpx, LIVE_RELEASE, RECORDED, id="A5-messages-httpx"),
        pytest.param("A6", _messages_anthropic_sdk, LIVE_RELEASE, RECORDED, id="A6-messages-anthropic-sdk"),
        pytest.param(
            "A7",
            _gemini_httpx,
            _LiveRelease(True, True, 1, 1, True, 2, 2, (False, False)),
            LOGGED_BEFORE_THE_SCAN_ENDS,
            id="A7-native-gemini-stream-generate-content",
        ),
        pytest.param(
            "A8",
            _completions_httpx,
            _LiveRelease(True, True, 1, 0, True, 1, 0, (False, True)),
            NEVER_SCANNED,
            id="A8-text-completions",
        ),
    ),
)
def test_a_live_detect_only_pipeline_releases_the_first_chunk_while_the_provider_holds_the_rest(
    rig: _Rig, row: str, reader: Reader, expected: _LiveRelease, verdicts: frozenset[tuple[str, ...]]
) -> None:
    token: Final = _token()
    whole: Final = f"{_provider_text(token)} tail"
    model: Final = rig.models[row]
    rail: Final = rig.rails[row][0]
    stream: Final = _released(rig.stage, reader, model, token)
    upstream_requests: Final = len(rig.upstream(token))
    scans: Final = _scan_count(rig, row, token, expected.scans)
    replay: Final = _replayed(rig.stage, reader, model, token)
    upstream_requests_after_replay: Final = len(rig.upstream(token))
    scans_after_replay: Final = _scan_count(rig, row, token, expected.scans_after_replay)
    rows: Final = _spend_rows(model, rig.rails[row], 2)
    observed: Final = _LiveRelease(
        _provider_text(token) in stream.released_while_held,
        whole in stream.received,
        upstream_requests,
        scans,
        whole in replay,
        upstream_requests_after_replay,
        scans_after_replay,
        tuple(spend.cache_hit for spend in rows),
    )
    assert observed == expected, (stream, replay, rows)
    assert all(spend.verdicts[rail] in verdicts for spend in rows), rows


@pytest.mark.parametrize(
    ("row", "endpoint", "outcome", "stream"),
    (
        pytest.param("B1", "chat", "provider", True, id="B1-flag-false"),
        pytest.param("B2", "chat", "rewritten", True, id="B2-flag-true"),
        pytest.param("B3", "chat", "rewritten", True, id="B3-flag-unset"),
        pytest.param("B4", "chat", "rewritten", True, id="B4-flag-null"),
        pytest.param("B5", "chat", "rewritten", True, id="B5-on-fail-block"),
        pytest.param("B6", "chat", "rewritten", True, id="B6-on-error-block"),
        pytest.param("B7", "chat", "rewritten", True, id="B7-one-buffering-step-gates-the-pipeline"),
        pytest.param("B8", "chat", "rewritten", True, id="B8-one-buffering-policy-gates-the-stream"),
        pytest.param("B10", "chat", "rewritten", False, id="B10-non-streaming-chat"),
        pytest.param("B11", "responses", "rewritten", False, id="B11-non-streaming-responses"),
        pytest.param("B12", "messages", "rewritten", False, id="B12-non-streaming-messages"),
        pytest.param("B13", "responses", "provider", True, id="B13-responses-stream"),
        pytest.param("B14", "messages", "provider", True, id="B14-messages-stream"),
        pytest.param("B15", "chat", "provider", True, id="B15-flag-false-in-optional-params"),
        pytest.param("B16", "chat", "provider", True, id="B16-flag-empty-string"),
        pytest.param("B17", "chat", "rewritten", True, id="B17-flag-false-string"),
        pytest.param("B18", "chat", "provider", True, id="B18-flag-zero"),
        pytest.param("B19", "chat", "rewritten", True, id="B19-on-fail-modify-response"),
        pytest.param("C1", "chat", "provider", True, id="C1-crowdstrike-flag-unset"),
        pytest.param("C2", "chat", "provider", True, id="C2-crowdstrike-flag-false"),
        pytest.param("C3", "chat", "rewritten", True, id="C3-crowdstrike-flag-true"),
        pytest.param("C4", "chat", "rewritten", True, id="C4-bedrock-flag-unset"),
        pytest.param("C5", "chat", "provider", True, id="C5-bedrock-flag-false"),
        pytest.param("C6", "chat", "rewritten", True, id="C6-bedrock-masking-keeps-buffering"),
    ),
)
def test_b_a_pipeline_rewrite_reaches_the_client_only_while_the_stream_buffers(
    rig: _Rig, row: str, endpoint: Endpoint, outcome: Outcome, stream: bool
) -> None:
    _assert_twin(rig, row, endpoint, outcome, stream=stream)


def test_c7_an_in_process_masking_filter_keeps_masking_a_pipeline_stream(rig: _Rig) -> None:
    _assert_twin(rig, "C7", "chat", "keyword_masked", scans_per_request=0)


def test_c8_straiker_keeps_buffering_when_the_flag_is_set_false(rig: _Rig) -> None:
    token: Final = _token()
    received: Final = _withheld(rig.stage, _chat_httpx, rig.models["C8"], token)
    assert f"{_provider_text(token)} tail" in received, received
    eventually(lambda: rig.scans("C8", token), lambda scans: len(scans) == 1, seconds=10)


@pytest.mark.parametrize(
    ("row", "verdict"),
    (
        pytest.param("D1", "guardrail_failed_to_respond", id="D1-guardrail-500"),
        pytest.param("D2", "guardrail_failed_to_respond", id="D2-guardrail-malformed-body"),
        pytest.param("D3", "guardrail_intervened", id="D3-guardrail-blocked-verdict-on-fail-next"),
    ),
)
def test_d_a_failing_scan_on_a_live_pipeline_still_delivers_the_stream_and_records_the_verdict(
    rig: _Rig, row: str, verdict: str
) -> None:
    _assert_twin(rig, row, "chat", "provider", verdict=verdict)


@dataclass(frozen=True, slots=True)
class _CutStream:
    released_while_held: bool
    status_line: str
    error_after_text: bool
    upstream_requests: int
    scans: int
    rows: tuple[tuple[str, tuple[str, ...]], ...]


CUT_STREAM: Final = _CutStream(True, "status 200", True, 1, 1, (("failure", ("success",)),))


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    (
        pytest.param("chat", CUT_STREAM, id="chat"),
        pytest.param("responses", CUT_STREAM, id="responses"),
        pytest.param("messages", _CutStream(True, "status 200", True, 1, 1, ()), id="messages"),
    ),
)
def test_d4_a_provider_drop_after_a_released_chunk_ends_the_stream_with_an_error_and_records_the_scan(
    rig: _Rig, endpoint: Endpoint, expected: _CutStream
) -> None:
    row: Final = f"D4-{endpoint}"
    token: Final = _token()
    raw: Final = _Received()
    stream: Final = _released(rig.stage, _raw_reader(endpoint, raw), rig.models[row], token, DROP)
    rows: Final = _spend_rows(rig.models[row], rig.rails[row], len(expected.rows))
    observed: Final = _CutStream(
        _provider_text(token) in stream.released_while_held,
        raw.text.split("\n", 1)[0],
        "error" in raw.text.split(token, 1)[-1],
        len(rig.upstream(token)),
        _scan_count(rig, row, token, expected.scans),
        tuple((spend.status, spend.verdicts[rig.rails[row][0]]) for spend in rows),
    )
    assert observed == expected, (stream, raw.text, rows)


def test_e1_repeated_identical_live_streams_each_get_one_upstream_call_scan_and_spend_row(rig: _Rig) -> None:
    token: Final = _token()
    responses: Final = tuple(
        _call(rig.gateway, "chat", rig.models["E1"], _prompt(token), extra={"cache": {"no-cache": True}})
        for _ in range(5)
    )
    assert all(response.status_code == 200 for response in responses), [response.text for response in responses]
    assert all(f"{_provider_text(token)} tail" in _text(response) for response in responses)
    assert len(rig.upstream(token)) == 5
    eventually(lambda: rig.scans("E1", token), lambda scans: len(scans) == 5, seconds=10)
    rows: Final = _spend_rows(rig.models["E1"], rig.rails["E1"], 5)
    assert [spend.verdicts for spend in rows] == [{rig.rails["E1"][0]: ("success",)}] * 5, rows


def test_f1_concurrent_live_streams_during_a_guardrail_outage_each_scan_once(rig: _Rig) -> None:
    base: Final = uuid.uuid4().hex
    tokens: Final = tuple(f"plive-{base}-{index}" for index in range(30))
    endpoints: Final[tuple[Endpoint, ...]] = ("chat", "responses", "messages")
    with ThreadPoolExecutor(max_workers=30) as pool:
        responses: Final = tuple(
            pool.map(
                lambda index: _call(rig.gateway, endpoints[index % 3], rig.models["F1"], _prompt(tokens[index])),
                range(30),
            )
        )
    assert all(response.status_code == 200 for response in responses), [response.text for response in responses]
    assert all(
        f"{_provider_text(token)} tail" in _text(response) for token, response in zip(tokens, responses, strict=True)
    )
    scanned: Final = eventually(
        lambda: tuple(len(rig.scans("F1", token + " ")) for token in tokens),
        lambda counts: all(count >= 1 for count in counts),
        seconds=20,
    )
    assert scanned == (1,) * 30, scanned
    rows: Final = _spend_rows(rig.models["F1"], rig.rails["F1"], 30)
    assert sorted(spend.verdicts[rig.rails["F1"][0]] for spend in rows) == sorted(
        ("guardrail_failed_to_respond",) if index % 2 == 0 else ("success",) for index in range(30)
    ), rows


@pytest.mark.parametrize("endpoint", ("chat", "responses", "messages"))
def test_h2_a_provider_failure_row_keeps_the_pre_call_guardrail_information(rig: _Rig, endpoint: Endpoint) -> None:
    row: Final = f"H2-{endpoint}"
    token: Final = _token()
    response: Final = _call(
        rig.gateway,
        endpoint,
        rig.models[row],
        _prompt(token, PROVIDER_FAILURE),
        stream=False,
        extra={"guardrails": list(rig.rails[row])},
    )
    assert response.status_code == 500, response.text
    rows: Final = _spend_rows(rig.models[row], rig.rails[row], 1)
    assert rows[0].status == "failure" and rows[0].verdicts == {rig.rails[row][0]: ("success",)}, rows


def test_h3_a_provider_failure_without_a_guardrail_records_no_guardrail_information(rig: _Rig) -> None:
    token: Final = _token()
    response: Final = _call(rig.gateway, "chat", rig.models["H3"], _prompt(token, PROVIDER_FAILURE), stream=False)
    assert response.status_code == 500, response.text
    found: Final = eventually(
        lambda: read_rows('SELECT status, metadata FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (rig.models["H3"],)),
        lambda rows: len(rows) == 1,
        seconds=70,
    )
    assert found[0]["status"] == "failure", found
    assert not object_value(found[0]["metadata"]).get("guardrail_information"), found


def test_i1_a_request_opted_guardrail_with_the_flag_set_true_buffers_without_a_pipeline(rig: _Rig) -> None:
    token: Final = _token()
    received: Final = _withheld(rig.stage, _opting_in(rig.rails["I1"]), rig.models["I1"], token)
    assert f"{_provider_text(token)} tail" in received, received
    assert _scan_count(rig, "I1", token, 1) == 1


@pytest.mark.parametrize("row", ("I2", "I3"), ids=("I2-flag-false", "I3-flag-unset"))
def test_i_a_request_opted_guardrail_without_the_flag_set_true_streams_live_without_a_pipeline(
    rig: _Rig, row: str
) -> None:
    token: Final = _token()
    stream: Final = _released(rig.stage, _opting_in(rig.rails[row]), rig.models[row], token)
    assert _provider_text(token) in stream.released_while_held, stream
    assert f"{_provider_text(token)} tail" in stream.received, stream
    assert _scan_count(rig, row, token, 1) == 1


def _single_rail_config(
    directory: Path, name: str, provider: str, guardrails: tuple[dict[str, JsonValue], ...]
) -> Path:
    model: Final = f"{name}-model"
    config: Final = {
        **_base_config(),
        "guardrails": list(guardrails),
        "model_list": [{"model_name": model, "litellm_params": _deployment("openai", provider)}],
        "policies": {f"{name}-policy": _policy((_step(name),))},
        "policy_attachments": [{"policy": f"{name}-policy", "models": [model]}],
    }
    path: Final = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_b9_a_pipeline_step_whose_guardrail_is_not_loaded_does_not_hold_the_stream(
    gateway: Gateway, tmp_path: Path
) -> None:
    gates: Final = _Gates()
    name: Final = f"plive-b9-missing-{uuid.uuid4().hex[:8]}"
    with wire_server(_provider(gates)) as provider:
        config: Final = _single_rail_config(tmp_path, name, provider.url, ())
        try:
            with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
                token: Final = _token()
                stage: Final = _Stage(owned.gateway, gates, _Log(provider))
                stream: Final = _released(stage, _chat_httpx, f"{name}-model", token)
                assert _provider_text(token) in stream.released_while_held, stream
                assert f"{_provider_text(token)} tail" in stream.received, stream
        finally:
            gates.release_all()


def test_f2_live_pipeline_streams_keep_scanning_after_a_worker_is_killed(gateway: Gateway, tmp_path: Path) -> None:
    gates: Final = _Gates()
    name: Final = f"plive-f2-{uuid.uuid4().hex[:8]}"
    with wire_server(_provider(gates)) as provider, wire_server(_peer) as peer:
        config: Final = _single_rail_config(
            tmp_path, name, provider.url, (_generic(name, peer.url, "allow", FLAG_FALSE),)
        )
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            members: Final = tuple(
                member for member in group_members(owned.process.pid) if member.pid != owned.process.pid
            )
            workers: Final = tuple(
                member for member in members if any("spawn_main" in part for part in member.cmdline())
            )
            assert len(workers) >= 2, members
            workers[0].send_signal(signal.SIGKILL)
            psutil.wait_procs((workers[0],), timeout=10)
            base: Final = uuid.uuid4().hex
            tokens: Final = tuple(f"plive-{base}-{index}" for index in range(8))
            with ThreadPoolExecutor(max_workers=8) as pool:
                responses: Final = tuple(
                    pool.map(lambda token: _call(owned.gateway, "chat", f"{name}-model", _prompt(token)), tokens)
                )
            assert all(response.status_code == 200 for response in responses), [r.text for r in responses]
            assert all(f"{_provider_text(token)} tail" in _text(r) for token, r in zip(tokens, responses, strict=True))
            peer_log: Final = _Log(peer)
            eventually(
                lambda: tuple(len(peer_log.matching(token + " ")) for token in tokens),
                lambda counts: counts == (1,) * 8,
                seconds=20,
            )
            rows: Final = _spend_rows(f"{name}-model", (name,), 8)
            assert [spend.verdicts for spend in rows] == [{name: ("success",)}] * 8, rows
            assert owned.process.poll() is None
