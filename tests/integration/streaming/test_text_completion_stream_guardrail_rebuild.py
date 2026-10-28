import json
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    gateway_from_environment,
    list_value,
    object_value,
)
from integration._support.process import OwnedProxy, WorkerHealthcheck, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

_CELL_SECONDS: Final = 2 * graceful_stop_seconds() + 120
pytestmark: Final = pytest.mark.timeout(_CELL_SECONDS)

_PROXY_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
_PROVIDER_KEY: Final = "synthetic-provider-key"
_GUARDRAIL: Final = "audit-text-rebuild-tool-permission"
_TEXT_MODEL: Final = f"audit-text-instruct-{uuid.uuid4().hex}"
_PIECES: Final = ("Hello", " streamed", " world")
_PROMPT_TOKENS: Final = 11
_COMPLETION_TOKENS: Final = 3
_USAGE: Final[dict[str, JsonValue]] = {
    "prompt_tokens": _PROMPT_TOKENS,
    "completion_tokens": _COMPLETION_TOKENS,
    "total_tokens": _PROMPT_TOKENS + _COMPLETION_TOKENS,
}
_BRIDGE_TEXT: Final = "bridge reply"
_TOGETHER_MODEL: Final = "together_ai/Qwen/Qwen3.5-9B"


@dataclass(frozen=True, slots=True)
class _Rig:
    wire: Wire
    proxy: OwnedProxy


@dataclass(frozen=True, slots=True)
class _Stream:
    status: int
    body: bytes

    @property
    def frames(self) -> tuple[dict[str, JsonValue], ...]:
        return tuple(
            JSON_OBJECT.validate_json(line[6:])
            for line in self.body.decode().splitlines()
            if line.startswith("data: {")
        )

    @property
    def choices(self) -> tuple[Mapping[str, JsonValue], ...]:
        return tuple(chain.from_iterable(_choices_of(frame) for frame in self.frames))

    @property
    def text(self) -> str:
        return "".join(str(choice.get("text") or "") for choice in self.choices)

    @property
    def finish_reasons(self) -> tuple[JsonValue, ...]:
        return tuple(choice["finish_reason"] for choice in self.choices if choice.get("finish_reason") is not None)

    @property
    def usages(self) -> tuple[Mapping[str, JsonValue], ...]:
        return tuple(object_value(frame["usage"]) for frame in self.frames if frame.get("usage") is not None)

    @property
    def ids(self) -> frozenset[JsonValue]:
        return frozenset(frame["id"] for frame in self.frames)


def _choices_of(frame: Mapping[str, JsonValue]) -> Iterator[Mapping[str, JsonValue]]:
    return (object_value(choice) for choice in list_value(frame.get("choices") or []))


def _frame(body: Mapping[str, JsonValue]) -> bytes:
    return b"data: " + json.dumps(body).encode() + b"\n\n"


def _text_frame(identity: str, choices: list[JsonValue], **extra: JsonValue) -> bytes:
    return _frame(
        {"id": identity, "object": "text_completion", "created": 1, "model": "scripted", "choices": choices, **extra}
    )


def _text_choice(text: str, finish_reason: str | None) -> dict[str, JsonValue]:
    return {"text": text, "index": 0, "logprobs": None, "finish_reason": finish_reason}


def _respond(request: Request) -> Reply:
    body: Final = JSON_OBJECT.validate_json(request.body)
    assert body.get("stream") is True, body
    assert object_value(body["stream_options"]).get("include_usage") is True, body
    identity: Final = body["prompt"]
    assert isinstance(identity, str), body
    return Reply(
        content_type="text/event-stream",
        chunks=(
            *(_text_frame(identity, [_text_choice(piece, None)]) for piece in _PIECES),
            _text_frame(identity, [_text_choice("", "length")]),
            _text_frame(identity, [], usage=_USAGE),
            b"data: [DONE]\n\n",
        ),
    )


def _config(directory: Path, wire: Wire) -> Path:
    base: Final = object_value(yaml.safe_load(_PROXY_CONFIG.read_text()))
    config: Final = {
        **base,
        "model_list": [
            *list_value(base["model_list"]),
            {
                "model_name": _TEXT_MODEL,
                "litellm_params": {
                    "model": "text-completion-openai/gpt-3.5-turbo-instruct",
                    "api_base": f"{wire.url}/v1",
                    "api_key": _PROVIDER_KEY,
                },
            },
        ],
        "guardrails": [
            {
                "guardrail_name": _GUARDRAIL,
                "litellm_params": {
                    "guardrail": "tool_permission",
                    "mode": "post_call",
                    "default_on": True,
                    "default_action": "allow",
                    "rules": [],
                },
            }
        ],
    }
    path: Final = directory / f"text-rebuild-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("text-rebuild-rig")
    with gateway_from_environment() as gateway, wire_server(_respond) as wire:
        with owned_proxy_process(
            gateway,
            directory,
            {},
            config=_config(directory, wire),
            workers=2,
            healthcheck=WorkerHealthcheck.LENIENT,
            client_timeout=_CELL_SECONDS,
        ) as owned:
            yield _Rig(wire, owned)


def _stream(rig: _Rig, identity: str) -> _Stream:
    gateway: Final = rig.proxy.gateway
    with gateway.client.stream(
        "POST",
        "/v1/completions",
        headers={"Authorization": f"Bearer {gateway.key}"},
        json={
            "model": _TEXT_MODEL,
            "prompt": identity,
            "max_tokens": 16,
            "stream": True,
            "stream_options": {"include_usage": True},
            "cache": {"no-cache": True},
        },
    ) as response:
        return _Stream(response.status_code, response.read())


def _posted_targets(wire: Wire) -> tuple[str, ...]:
    return tuple(request.target for request in wire.drain() if request.method == "POST")


def test_guarded_text_completion_stream_reaches_the_caller_whole_with_its_usage(rig: _Rig) -> None:
    rig.wire.drain()
    identity: Final = f"cmpl-guarded-{uuid.uuid4().hex}"

    stream: Final = _stream(rig, identity)

    assert stream.status == 200, stream.body
    assert stream.body.rstrip().endswith(b"data: [DONE]"), stream.body[-400:]
    assert stream.ids == {identity}, stream.ids
    assert stream.text == "".join(_PIECES), stream.frames
    assert stream.finish_reasons == ("length",), stream.frames
    assert [(usage["prompt_tokens"], usage["completion_tokens"]) for usage in stream.usages] == [
        (_PROMPT_TOKENS, _COMPLETION_TOKENS)
    ], stream.frames
    assert _posted_targets(rig.wire) == ("/v1/completions",)
    (row,) = eventually(
        lambda: rig.proxy.gateway.rows(
            'SELECT prompt_tokens, completion_tokens, metadata FROM "LiteLLM_SpendLogs" WHERE request_id = %s',
            (identity,),
        ),
        lambda rows: len(rows) == 1,
        seconds=70,
    )
    assert (row["prompt_tokens"], row["completion_tokens"]) == (_PROMPT_TOKENS, _COMPLETION_TOKENS), row
    assert _GUARDRAIL in list_value(object_value(row["metadata"])["applied_guardrails"]), row


def _together_chunk(choices: list[JsonValue], **extra: JsonValue) -> bytes:
    return _frame(
        {
            "id": f"chatcmpl-together-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "scripted",
            "choices": choices,
            **extra,
        }
    )


def _together_chat_stream(request: Request) -> Reply:
    assert request.method == "POST" and request.target.endswith("/chat/completions"), request
    assert JSON_OBJECT.validate_json(request.body).get("stream") is True, request.body
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _together_chunk(
                [
                    {
                        "index": 0,
                        "text": _BRIDGE_TEXT,
                        "delta": {"role": "assistant", "content": _BRIDGE_TEXT},
                        "finish_reason": None,
                    }
                ]
            ),
            _together_chunk([{"index": 0, "text": "", "delta": {}, "finish_reason": "stop"}]),
            _together_chunk([], usage=_USAGE),
            b"data: [DONE]\n\n",
        ),
    )


def _completed_events(body: bytes) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(
        object_value(event["response"])
        for event in _Stream(200, body).frames
        if event.get("type") == "response.completed"
    )


def _output_text(response: Mapping[str, JsonValue]) -> str:
    parts: Final = chain.from_iterable(
        _content_parts(object_value(item)) for item in list_value(response.get("output") or [])
    )
    return "".join(str(part.get("text") or "") for part in parts)


def _content_parts(item: Mapping[str, JsonValue]) -> Iterator[Mapping[str, JsonValue]]:
    return (object_value(part) for part in list_value(item.get("content") or []))


def test_responses_bridge_rebuilds_a_chat_stream_whose_choices_also_carry_text(gateway: Gateway) -> None:
    with wire_server(_together_chat_stream) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_TOGETHER_MODEL, api_base=f"{wire.url}/v1", api_key=_PROVIDER_KEY)
        key: Final = scenario.key(models=[model])
        response: Final = gateway.client.post(
            "/v1/responses",
            json={"model": model, "input": f"bridge probe {uuid.uuid4().hex}", "stream": True},
            headers={"Authorization": f"Bearer {key}"},
        )

        assert response.status_code == 200, response.text
        (completed,) = _completed_events(response.content)
        assert _output_text(completed) == _BRIDGE_TEXT, completed
        usage: Final = object_value(completed["usage"])
        assert (usage["input_tokens"], usage["output_tokens"]) == (_PROMPT_TOKENS, _COMPLETION_TOKENS), usage
        assert [request.target for request in wire.drain()] == ["/v1/chat/completions"]
