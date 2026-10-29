from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import Gateway, gateway_from_environment
from tests.integration._support.process import graceful_stop_seconds, owned_proxy
from tests.integration._support.wire import Reply, Request, Wire, wire_server

_NATIVE_MODEL: Final = "safeguards-timing-deepseek"
_CLASSIFIER_MODEL: Final = "safeguards-timing-classifier"
_NATIVE_PATH: Final = "/deepseek/anthropic/v1/messages"
_FRAME_TEXT_BYTES: Final = 4_000_000
_WIRE_CHUNK_BYTES: Final = 64 * 1024
_LIVELINESS_PACE_SECONDS: Final = 0.05
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _sse(event: dict[str, JsonValue]) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def _native_stream(message_id: str, text: str) -> bytes:
    return b"".join(
        (
            _sse(
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": "deepseek-v4-pro",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 12, "output_tokens": 1},
                    },
                }
            ),
            _sse({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            _sse({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
            _sse({"type": "content_block_stop", "index": 0}),
            _sse(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 7},
                }
            ),
            _sse({"type": "message_stop"}),
        )
    )


def _text_of(prompt: str) -> str:
    return (prompt.split()[-1] * (_FRAME_TEXT_BYTES // 32 + 1))[:_FRAME_TEXT_BYTES]


def _respond(request: Request) -> Reply:
    assert request.target == _NATIVE_PATH, f"only the native backend is reachable here, got {request.target}"
    body: Final = _JSON_OBJECT.validate_json(request.body)
    messages: Final = body["messages"]
    assert isinstance(messages, list) and len(messages) == 1, body
    prompt: Final = _JSON_OBJECT.validate_python(messages[0])["content"]
    assert isinstance(prompt, str), body
    payload: Final = _native_stream(f"msg_{prompt.split()[-1]}", _text_of(prompt))
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(payload[start : start + _WIRE_CHUNK_BYTES] for start in range(0, len(payload), _WIRE_CHUNK_BYTES)),
    )


def _config(directory: Path, wire: Wire) -> Path:
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["model_list"] = [
        *configuration["model_list"],
        {
            "model_name": _NATIVE_MODEL,
            "litellm_params": {
                "model": "deepseek/deepseek-v4-pro",
                "api_base": f"{wire.url}/deepseek",
                "api_key": "synthetic-deepseek-key",
            },
        },
        {
            "model_name": _CLASSIFIER_MODEL,
            "litellm_params": {
                "model": "openai/gpt-5.4-mini",
                "api_base": f"{wire.url}/classifier/v1",
                "api_key": "synthetic-classifier-key",
            },
        },
    ]
    configuration["general_settings"]["safeguards_classifier_model"] = _CLASSIFIER_MODEL
    path: Final = directory / "safeguards-sse-timing.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@dataclass(frozen=True, slots=True)
class _Timed:
    seconds: float
    slowest_liveliness_seconds: float
    text: str
    stop_reason: JsonValue


def _frames_text(raw: bytes) -> tuple[str, JsonValue]:
    events: Final = tuple(
        _JSON_OBJECT.validate_json(line.removeprefix(b"data: "))
        for line in raw.split(b"\n")
        if line.startswith(b"data: ")
    )
    text: Final = "".join(
        str(_JSON_OBJECT.validate_python(event["delta"]).get("text", ""))
        for event in events
        if event["type"] == "content_block_delta"
    )
    deltas: Final = tuple(
        _JSON_OBJECT.validate_python(event["delta"]) for event in events if event["type"] == "message_delta"
    )
    assert len(deltas) == 1, deltas
    return text, deltas[0]["stop_reason"]


def _probe_once(client: httpx.Client) -> float:
    started: Final = time.monotonic()
    assert client.get("/health/liveliness").status_code == 200
    return time.monotonic() - started


def _probes(client: httpx.Client, done: threading.Event) -> Iterator[float]:
    while not done.wait(timeout=_LIVELINESS_PACE_SECONDS):
        yield _probe_once(client)


def _probe_liveliness(base_url: str, done: threading.Event) -> tuple[float, ...]:
    with httpx.Client(base_url=base_url, timeout=30, trust_env=False) as client:
        return tuple(_probes(client, done))


def _timed_stream(candidate: Gateway, *, safeguards: bool) -> _Timed:
    nonce: Final = uuid.uuid4().hex
    body: Final[dict[str, JsonValue]] = {
        "model": _NATIVE_MODEL,
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": f"repeat the token {nonce}"}],
        **({"safeguards": [{"type": "dangerous_tool_use"}]} if safeguards else {}),
    }
    done: Final = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        probes: Final = executor.submit(_probe_liveliness, str(candidate.client.base_url), done)
        started: Final = time.monotonic()
        with candidate.client.stream(
            "POST",
            "/v1/messages",
            json=body,
            headers={"Authorization": f"Bearer {candidate.key}", "anthropic-beta": "dangerous-tool-use-2026-09-03"},
            timeout=120,
        ) as response:
            assert response.status_code == 200, response.read().decode()
            raw: Final = response.read()
        elapsed: Final = time.monotonic() - started
        done.set()
        latencies: Final = probes.result(timeout=60)
    text, stop_reason = _frames_text(raw)
    return _Timed(elapsed, max(latencies, default=0.0), text, stop_reason)


@pytest.mark.timeout(2 * graceful_stop_seconds() + 240)
def test_a_native_stream_with_one_large_sse_frame_costs_the_same_with_or_without_safeguards(
    tmp_path: Path,
) -> None:
    with (
        wire_server(_respond) as wire,
        gateway_from_environment() as shared,
        owned_proxy(shared, tmp_path, {}, config=_config(tmp_path, wire)) as candidate,
    ):
        warm_up: Final = _timed_stream(candidate, safeguards=False)
        untouched: Final = _timed_stream(candidate, safeguards=False)
        safeguarded: Final = _timed_stream(candidate, safeguards=True)
        reached: Final = [request.target for request in wire.drain()]
    timings: Final = {
        "frame_text_bytes": _FRAME_TEXT_BYTES,
        "without_safeguards_seconds": round(untouched.seconds, 3),
        "with_safeguards_seconds": round(safeguarded.seconds, 3),
        "without_safeguards_slowest_liveliness_seconds": round(untouched.slowest_liveliness_seconds, 3),
        "with_safeguards_slowest_liveliness_seconds": round(safeguarded.slowest_liveliness_seconds, 3),
    }
    results: Final = Path(os.environ.get("INTEGRATION_RESULTS_DIR", str(tmp_path)))
    results.mkdir(parents=True, exist_ok=True)
    (results / "safeguards-sse-frame-timing.json").write_text(json.dumps(timings, indent=2) + "\n")
    assert len(warm_up.text) == _FRAME_TEXT_BYTES and warm_up.stop_reason == "end_turn", timings
    assert len(untouched.text) == _FRAME_TEXT_BYTES and untouched.stop_reason == "end_turn", timings
    assert len(safeguarded.text) == _FRAME_TEXT_BYTES and safeguarded.stop_reason == "end_turn", timings
    assert reached == [_NATIVE_PATH] * 3, timings
    assert safeguarded.slowest_liveliness_seconds < 1.0, timings
    assert safeguarded.seconds <= 3 * untouched.seconds + 1.0, timings
