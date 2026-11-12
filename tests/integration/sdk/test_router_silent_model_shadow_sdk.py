from __future__ import annotations

import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from queue import SimpleQueue
from typing import Final

import pytest
from integration._support.client import eventually
from integration._support.openai_wire import chat_reply
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm import CustomStreamWrapper, Router
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import ModelResponseStream

_MODEL: Final = "gpt-5.6"
_API_KEY: Final = "synthetic-shadow-key"
_ANSWER: Final = "answered by"
_NONCE: Final = re.compile(r"nonce=([0-9a-f]{32})")
_JSON: Final = TypeAdapter(dict[str, JsonValue])
_OBJECT: Final = TypeAdapter(dict[str, object])
_SHADOW_SECONDS: Final = 20
_TARGETS: Final = (
    pytest.param(("shadow-a",), id="single"),
    pytest.param(("shadow-a", "shadow-b"), id="pair"),
)


def _prompt(nonce: str) -> str:
    return f"which deployment answers this? nonce={nonce}"


def _peer(request: Request) -> Reply:
    deployment: Final = request.target.lstrip("/").partition("/")[0]
    raw: Final = request.body.decode()
    found: Final = _NONCE.search(raw)
    assert found is not None, raw
    return chat_reply(
        f"chatcmpl-{deployment}-{found.group(1)}",
        _MODEL,
        f"{_ANSWER} {deployment}",
        stream=_JSON.validate_json(request.body).get("stream") is True,
    )


def _deployment(name: str, wire: Wire, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {"model": f"openai/{_MODEL}", "api_base": f"{wire.url}/{name}", "api_key": _API_KEY, **extra},
    }


@dataclass(frozen=True, slots=True)
class _Posted:
    deployment: str
    streamed: bool
    leaked: bool


def _posted(wire: Wire) -> tuple[_Posted, ...]:
    return tuple(
        _Posted(
            request.target.lstrip("/").partition("/")[0],
            _JSON.validate_json(request.body).get("stream") is True,
            "silent_model" in request.body.decode(),
        )
        for request in wire.drain()
        if request.method == "POST"
    )


class _ShadowRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.shadow_groups: SimpleQueue[str] = SimpleQueue()

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        payload: Final = _OBJECT.validate_python(kwargs["standard_logging_object"])
        if _OBJECT.validate_python(payload["metadata"]).get("is_silent_experiment") is True:
            self.shadow_groups.put(str(payload["model_group"]))

    def drained(self) -> tuple[str, ...]:
        return tuple(self.shadow_groups.get_nowait() for _ in range(self.shadow_groups.qsize()))


def _delta(chunk: object) -> str:
    assert isinstance(chunk, ModelResponseStream), chunk
    return "".join(str(choice.delta.content or "") for choice in chunk.choices)


@pytest.mark.parametrize("targets", _TARGETS)
def test_sync_streamed_completion_streams_and_logs_each_shadow_target(
    targets: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder: Final = _ShadowRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    with wire_server(_peer) as wire:
        silent_model: Final[JsonValue] = targets[0] if len(targets) == 1 else [*targets]
        router: Final = Router(
            model_list=[
                _deployment("mirrored", wire, silent_model=silent_model),
                *(_deployment(target, wire) for target in targets),
            ],
            num_retries=0,
            disable_cooldowns=True,
        )
        response: Final = router.completion(
            model="mirrored",
            messages=[{"role": "user", "content": _prompt(uuid.uuid4().hex)}],
            stream=True,
        )
        assert isinstance(response, CustomStreamWrapper), response
        text: Final = "".join(_delta(chunk) for chunk in response)
        eventually(wire.received.qsize, lambda count: count >= 1 + len(targets), seconds=_SHADOW_SECONDS)
        posted: Final = _posted(wire)
        eventually(recorder.shadow_groups.qsize, lambda count: count >= len(targets), seconds=_SHADOW_SECONDS)
        logged: Final = recorder.drained()
    assert text == f"{_ANSWER} mirrored"
    assert sorted(item.deployment for item in posted) == sorted(("mirrored", *targets)), posted
    assert all(item.streamed for item in posted), posted
    assert not any(item.leaked for item in posted), posted
    assert sorted(logged) == sorted(targets), logged
