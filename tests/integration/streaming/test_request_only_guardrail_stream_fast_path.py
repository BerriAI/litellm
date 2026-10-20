import json
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, list_value, object_value, string_value
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

pytestmark: Final = [
    pytest.mark.timeout(2 * graceful_stop_seconds() + 120),
    pytest.mark.xdist_group("request-only-guardrail-stream-fast-path"),
]

_MODEL: Final = "request-only-guardrail-stream"
_GUARDRAIL: Final = "request-only-check"
_CHUNK_GUARDRAIL: Final = "per-chunk-check"
_MODULE: Final = "request_only_guardrail"
_LOG_ENV: Final = "REQUEST_ONLY_GUARDRAIL_LOG"
_PIECES: Final = tuple(f"word{index} " for index in range(24))
_MARKER: Final = re.compile(rb"marker-([0-9a-f]{32})")
_MODULE_SOURCE: Final = f'''import json
import os
import re

from litellm.integrations.custom_guardrail import CustomGuardrail


class RequestOnlyGuardrail(CustomGuardrail):
    def should_run_guardrail(self, data, event_type):
        found = re.search("marker-([0-9a-f]{{32}})", str(data.get("messages") or data.get("input")))
        if found is not None:
            with open(os.environ["{_LOG_ENV}"], "a") as log:
                log.write(json.dumps({{"marker": found.group(1), "event": str(getattr(event_type, "value", event_type))}}) + "\\n")
        return super().should_run_guardrail(data=data, event_type=event_type)

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        return data


class PerChunkGuardrail(CustomGuardrail):
    async def async_post_call_streaming_hook(self, user_api_key_dict, response):
        with open(os.environ["{_LOG_ENV}"], "a") as log:
            log.write(json.dumps({{"marker": None, "event": "chunk"}}) + "\\n")
        return None
'''


def _chat_chunk(identity: str, content: str | None, finish: str | None = None) -> bytes:
    body: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {} if content is None else {"content": content}, "finish_reason": finish}],
    }
    return b"data: " + json.dumps(body).encode() + b"\n\n"


def _respond(request: Request) -> Reply:
    found: Final = _MARKER.search(request.body)
    assert found is not None, request.body
    identity: Final = found.group(1).decode()
    assert request.target.endswith("/chat/completions"), request.target
    return Reply(
        content_type="text/event-stream",
        chunks=(
            *(_chat_chunk(f"chatcmpl-{identity}", piece) for piece in _PIECES),
            _chat_chunk(f"chatcmpl-{identity}", None, finish="stop"),
            b"data: [DONE]\n\n",
        ),
    )


def _guardrail(name: str, guardrail_class: str, mode: str) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {"guardrail": f"{_MODULE}.{guardrail_class}", "mode": mode, "default_on": True},
    }


def _config(upstream_url: str, guardrails: list[JsonValue]) -> dict[str, JsonValue]:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    return {
        **base,
        "guardrails": guardrails,
        "model_list": [
            {
                "model_name": _MODEL,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": f"{upstream_url}/v1",
                    "api_key": "synthetic-stream-key",
                },
            }
        ],
    }


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: OwnedProxy
    log: Path


@contextmanager
def _rig(gateway: Gateway, directory: Path, guardrails: list[JsonValue]) -> Iterator[Rig]:
    with wire_server(_respond) as upstream:
        (directory / f"{_MODULE}.py").write_text(_MODULE_SOURCE)
        config_path: Final = directory / "request-only-guardrail-stream-fast-path.yaml"
        config_path.write_text(yaml.safe_dump(_config(upstream.url, guardrails)))
        log: Final = directory / "guardrail-asks.jsonl"
        log.touch()
        with owned_proxy_process(gateway, directory, {_LOG_ENV: str(log)}, config=config_path, workers=2) as proxy:
            yield Rig(proxy, log)


@pytest.fixture(scope="module")
def rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with _rig(
        session_gateway,
        tmp_path_factory.mktemp("request-only-guardrail-stream-fast-path"),
        [_guardrail(_GUARDRAIL, "RequestOnlyGuardrail", "pre_call")],
    ) as built:
        yield built


@pytest.fixture(scope="module")
def mixed_rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with _rig(
        session_gateway,
        tmp_path_factory.mktemp("request-only-guardrail-beside-per-chunk-guardrail"),
        [
            _guardrail(_GUARDRAIL, "RequestOnlyGuardrail", "pre_call"),
            _guardrail(_CHUNK_GUARDRAIL, "PerChunkGuardrail", "post_call"),
        ],
    ) as built:
        yield built


def _events(rig: Rig, marker: str | None) -> tuple[str, ...]:
    entries: Final = (JSON_OBJECT.validate_json(line) for line in rig.log.read_text().splitlines() if line)
    return tuple(string_value(entry["event"]) for entry in entries if entry["marker"] == marker)


def _choices(text: str) -> Iterator[JsonValue]:
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            yield from list_value(JSON_OBJECT.validate_json(line.removeprefix("data: ")).get("choices") or [])


def _chat_text(text: str) -> str:
    return "".join(
        string_value(object_value(object_value(choice)["delta"]).get("content") or "") for choice in _choices(text)
    )


def _stream(rig: Rig, marker: str) -> str:
    body: Final = {"model": _MODEL, "stream": True, "messages": [{"role": "user", "content": f"marker-{marker}"}]}
    response: Final = rig.proxy.gateway.request("POST", "/v1/chat/completions", body)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream"), response.headers
    assert _chat_text(response.text) == "".join(_PIECES), response.text
    return response.text


def test_a_request_only_guardrail_is_not_asked_about_each_streamed_chunk(rig: Rig) -> None:
    marker: Final = uuid.uuid4().hex
    _stream(rig, marker)
    asks: Final = _events(rig, marker)
    assert "pre_call" in asks, asks
    assert asks.count("post_call") == 0, f"asked about the response {asks.count('post_call')} times: {asks}"


def test_a_per_chunk_guardrail_still_sees_every_chunk_without_asking_the_request_only_one(mixed_rig: Rig) -> None:
    marker: Final = uuid.uuid4().hex
    text: Final = _stream(mixed_rig, marker)
    asks: Final = _events(mixed_rig, marker)
    assert "pre_call" in asks, asks
    assert asks.count("post_call") <= 1, f"asked about the response {asks.count('post_call')} times: {asks}"
    delivered: Final = sum(1 for line in text.splitlines() if line.startswith("data: ") and line != "data: [DONE]")
    assert _events(mixed_rig, None).count("chunk") >= delivered, (delivered, _events(mixed_rig, None))
