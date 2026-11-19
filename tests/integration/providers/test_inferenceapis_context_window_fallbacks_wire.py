import json
import threading
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final
from uuid import uuid4

import httpx
import pytest
import yaml
from integration._support.bedrock_discovery import open_connections_to
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = [pytest.mark.timeout(int(2 * graceful_stop_seconds() + 120)), pytest.mark.shares_worker_state]

_BACKEND: Final = "deepseek-ai/DeepSeek-V4-Flash"
_VENDOR_KEY: Final = "synthetic-inferenceapis-key"
_OPENAI_KEY: Final = "synthetic-openai-key"
_INFERENCEAPIS_OVERFLOW: Final = "Input too long: 150055 input tokens, limit is 131072 for this model"
_STANDARD_OVERFLOW: Final = "This model's maximum context length is 4096 tokens. However, you requested 9000 tokens."
_PLAIN_TEXT: Final = "Invalid value for 'temperature': expected a number between 0 and 2."
_FALLBACK_TEXT: Final = "fallback answered"
_SLOW_PIECES: Final = 4
_SLOW_PAUSE_SECONDS: Final = 0.5
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_EMPTY_MODEL_LIST: Final = Reply(body=b'{"object":"list","data":[]}')


def _error_peer(message: str, code: str | None) -> Callable[[Request], Reply]:
    body: Final = json.dumps({"error": {"message": message, "type": "invalid_request_error", "code": code}}).encode()
    return lambda request: _EMPTY_MODEL_LIST if request.method == "GET" else Reply(status=400, body=body)


def _streaming(request: Request) -> bool:
    return _JSON_OBJECT.validate_json(request.body).get("stream") is True


def _sse(events: tuple[Mapping[str, JsonValue], ...]) -> tuple[bytes, ...]:
    return tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)


def _completion(identity: str, text: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
        }
    ).encode()


def _chat_reply(identity: str, stream: bool) -> Reply:
    if not stream:
        return Reply(body=_completion(identity, _FALLBACK_TEXT))
    chunk: Final[dict[str, JsonValue]] = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
    }
    frames: Final[tuple[dict[str, JsonValue], ...]] = (
        {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": _FALLBACK_TEXT}}]},
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"data: {json.dumps(frame)}\n\n".encode() for frame in frames) + (b"data: [DONE]\n\n",),
    )


def _responses_reply(identity: str, stream: bool) -> Reply:
    completed: Final[dict[str, JsonValue]] = {
        "id": f"resp_{identity}",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "id": f"msg_{identity}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": _FALLBACK_TEXT, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 9, "output_tokens": 2, "total_tokens": 11},
    }
    if not stream:
        return Reply(body=json.dumps(completed).encode())
    return Reply(
        content_type="text/event-stream",
        chunks=_sse(
            (
                {
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": {**completed, "status": "in_progress", "output": []},
                },
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": f"msg_{identity}",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": _FALLBACK_TEXT,
                },
                {"type": "response.completed", "sequence_number": 2, "response": completed},
            )
        ),
    )


def _fallback_peer(request: Request) -> Reply:
    if request.method == "GET":
        return _EMPTY_MODEL_LIST
    assert request.headers["authorization"] == f"Bearer {_OPENAI_KEY}", dict(request.headers)
    identity: Final = f"chatcmpl-fallback-{uuid4().hex}"
    if request.target == "/v1/responses":
        return _responses_reply(identity, _streaming(request))
    assert request.target == "/v1/chat/completions", request.target
    return _chat_reply(identity, _streaming(request))


def _healthy_peer(request: Request) -> Reply:
    assert (request.method, request.target) == ("POST", "/v1/chat/completions"), request.target
    return Reply(body=_completion(f"chatcmpl-healthy-{uuid4().hex}", "healthy"))


def _slow_peer(request: Request) -> Reply:
    assert (request.method, request.target) == ("POST", "/v1/chat/completions"), request.target
    body: Final = _completion(f"chatcmpl-slow-{uuid4().hex}", "slow")
    size: Final = -(-len(body) // _SLOW_PIECES)
    return Reply(
        chunks=tuple(body[start : start + size] for start in range(0, len(body), size)),
        pause_between_chunks=_SLOW_PAUSE_SECONDS,
    )


@dataclass(frozen=True, slots=True)
class _HeldPeer:
    release: threading.Event
    held: SimpleQueue[str]

    def respond(self, request: Request) -> Reply:
        self.held.put(request.target)
        assert self.release.wait(timeout=120), "The burst was never released"
        return Reply(body=_completion(f"chatcmpl-held-{uuid4().hex}", "held"))

    @contextmanager
    def released_on_exit(self) -> Iterator[None]:
        try:
            yield
        finally:
            self.release.set()


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    owned: OwnedProxy
    peers: Mapping[str, Wire]
    models: Mapping[str, str]
    held: _HeldPeer


def _vendor(provider_model: str, wire: Wire) -> dict[str, JsonValue]:
    return {"model": provider_model, "api_base": f"{wire.url}/v1", "api_key": _VENDOR_KEY}


def _write_config(root: Path, model_list: list[JsonValue], router_settings: dict[str, JsonValue]) -> Path:
    base: Final = _JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final[dict[str, JsonValue]] = {
        **base,
        "model_list": model_list,
        "router_settings": {"disable_cooldowns": True, "num_retries": 0, **router_settings},
    }
    path: Final = root / "inferenceapis-fallbacks.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    root: Final = tmp_path_factory.mktemp("inferenceapis-fallbacks")
    suffix: Final = uuid4().hex[:8]
    names: Final = {
        label: f"{label}-{suffix}"
        for label in ("ia_cwf", "ia_plain", "mimo_fb", "ia_ok", "ia_slow", "ia_held", "fallback")
    }
    held: Final = _HeldPeer(threading.Event(), SimpleQueue())
    with (
        gateway_from_environment() as gateway,
        wire_server(_error_peer(_INFERENCEAPIS_OVERFLOW, "upstream_rejected")) as ia_overflow,
        wire_server(_error_peer(_PLAIN_TEXT, None)) as ia_plain,
        wire_server(_error_peer(_STANDARD_OVERFLOW, None)) as mimo_overflow,
        wire_server(_healthy_peer) as healthy,
        wire_server(_slow_peer) as slow,
        wire_server(held.respond) as held_wire,
        wire_server(_fallback_peer) as fallback,
    ):
        model_list: Final[list[JsonValue]] = [
            {"model_name": names["ia_cwf"], "litellm_params": _vendor(f"inferenceapis/{_BACKEND}", ia_overflow)},
            {"model_name": names["ia_plain"], "litellm_params": _vendor(f"inferenceapis/{_BACKEND}", ia_plain)},
            {"model_name": names["mimo_fb"], "litellm_params": _vendor("xiaomi_mimo/mimo-v2.6-flash", mimo_overflow)},
            {"model_name": names["ia_ok"], "litellm_params": _vendor(f"inferenceapis/{_BACKEND}", healthy)},
            {"model_name": names["ia_slow"], "litellm_params": _vendor(f"inferenceapis/{_BACKEND}", slow)},
            {"model_name": names["ia_held"], "litellm_params": _vendor(f"inferenceapis/{_BACKEND}", held_wire)},
            {
                "model_name": names["fallback"],
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": f"{fallback.url}/v1",
                    "api_key": _OPENAI_KEY,
                },
            },
        ]
        config: Final = _write_config(
            root,
            model_list,
            {
                "context_window_fallbacks": [{names[label]: [names["fallback"]]} for label in ("ia_cwf", "ia_plain")],
                "fallbacks": [{names["mimo_fb"]: [names["fallback"]]}],
            },
        )
        with owned_proxy_process(gateway, root, {}, config=config, workers=2) as owned:
            yield Rig(
                owned.gateway,
                owned,
                {
                    "ia_cwf": ia_overflow,
                    "ia_plain": ia_plain,
                    "mimo_fb": mimo_overflow,
                    "ia_ok": healthy,
                    "ia_slow": slow,
                    "ia_held": held_wire,
                    "fallback": fallback,
                },
                names,
                held,
            )


def _chat(model: str, prompt: str, stream: bool = False) -> dict[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": stream}


def _messages(model: str, prompt: str, stream: bool = False) -> dict[str, JsonValue]:
    return {"model": model, "max_tokens": 32, "messages": [{"role": "user", "content": prompt}], "stream": stream}


def _responses(model: str, prompt: str, stream: bool = False) -> dict[str, JsonValue]:
    return {"model": model, "input": prompt, "stream": stream}


_BUILDERS: Final[Mapping[str, Callable[[str, str, bool], dict[str, JsonValue]]]] = {
    "/v1/chat/completions": _chat,
    "/v1/messages": _messages,
    "/v1/responses": _responses,
}
_SURFACES: Final = tuple((route, stream) for route in _BUILDERS for stream in (False, True))


def _consumed(gateway: Gateway, route: str, body: Mapping[str, JsonValue]) -> tuple[int, str]:
    headers: Final = {"Authorization": f"Bearer {gateway.key}"}
    with gateway.client.stream("POST", route, json=body, headers=headers) as response:
        text: Final = b"".join(response.iter_bytes()).decode()
    return response.status_code, text


def _prompt(label: str) -> str:
    return f"{label} {uuid4().hex}"


def _calls_with(requests: tuple[Request, ...], prompt: str) -> int:
    return sum(1 for request in requests if prompt in request.body.decode())


def _send(rig: Rig, label: str, route: str, stream: bool, prompt: str) -> tuple[int, str]:
    return _consumed(rig.gateway, route, _BUILDERS[route](rig.models[label], prompt, stream))


@pytest.mark.parametrize(("route", "stream"), _SURFACES)
def test_inferenceapis_overflow_is_answered_by_the_context_window_fallback(rig: Rig, route: str, stream: bool) -> None:
    prompt: Final = _prompt("ia cwf")
    status, text = _send(rig, "ia_cwf", route, stream, prompt)
    assert status == 200, text
    assert _FALLBACK_TEXT in text, text
    assert _calls_with(rig.peers["ia_cwf"].drain(), prompt) == 1
    assert _calls_with(rig.peers["fallback"].drain(), prompt) == 1


def test_generic_fallbacks_still_answer_an_overflow_from_a_provider_without_its_own_mapper(rig: Rig) -> None:
    prompt: Final = _prompt("mimo fb")
    status, text = _send(rig, "mimo_fb", "/v1/chat/completions", False, prompt)
    assert status == 200, text
    assert _FALLBACK_TEXT in text, text
    assert _calls_with(rig.peers["mimo_fb"].drain(), prompt) == 1
    assert _calls_with(rig.peers["fallback"].drain(), prompt) == 1


def test_a_plain_400_is_returned_without_the_context_window_fallback(rig: Rig) -> None:
    prompt: Final = _prompt("ia plain")
    status, text = _send(rig, "ia_plain", "/v1/chat/completions", False, prompt)
    assert status == 400, text
    assert _PLAIN_TEXT in text, text
    assert "ContextWindowExceededError" not in text, text
    assert _calls_with(rig.peers["ia_plain"].drain(), prompt) == 1
    assert _calls_with(rig.peers["fallback"].drain(), prompt) == 0


def test_a_concurrent_overflow_burst_is_answered_once_per_request_by_the_fallback(rig: Rig) -> None:
    plan: Final = tuple((*_SURFACES[index % len(_SURFACES)], _prompt(f"burst {index}")) for index in range(30))
    with ThreadPoolExecutor(max_workers=15) as pool:
        results: Final = tuple(pool.map(partial(_send, rig, "ia_cwf"), *zip(*plan, strict=True)))
    assert [status for status, _ in results] == [200] * len(plan), results
    assert all(_FALLBACK_TEXT in text for _, text in results), results
    overflow_calls: Final = rig.peers["ia_cwf"].drain()
    fallback_calls: Final = rig.peers["fallback"].drain()
    assert [_calls_with(overflow_calls, prompt) for _, _, prompt in plan] == [1] * len(plan)
    assert [_calls_with(fallback_calls, prompt) for _, _, prompt in plan] == [1] * len(plan)


def _completion_id(status: int, text: str) -> str | None:
    if status != 200:
        return None
    identity: Final = _JSON_OBJECT.validate_json(text)["id"]
    assert isinstance(identity, str)
    return identity


def _post_chat(rig: Rig, label: str) -> tuple[int, str]:
    try:
        return _send(rig, label, "/v1/chat/completions", False, _prompt(label))
    except httpx.TransportError as error:
        return 0, repr(error)


def _spend_rows(identities: tuple[str, ...]) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = ANY(string_to_array(%s, \',\'))',
        (",".join(identities),),
    )


def _assert_one_spend_row_each(identities: tuple[str, ...]) -> None:
    rows: Final = eventually(
        lambda: _spend_rows(identities),
        lambda values: len(values) >= len(identities),
        seconds=90,
    )
    assert sorted(str(row["request_id"]) for row in rows) == sorted(identities)


def test_a_slow_vendor_under_a_burst_answers_every_request_once(rig: Rig) -> None:
    with ThreadPoolExecutor(max_workers=16) as pool:
        results: Final = tuple(pool.map(partial(_post_chat, rig), ("ia_slow",) * 16))
    assert [status for status, _ in results] == [200] * 16, results
    identities: Final = tuple(_completion_id(status, text) or "" for status, text in results)
    assert len(set(identities)) == 16
    assert len(rig.peers["ia_slow"].drain()) == 16
    _assert_one_spend_row_each(identities)


def test_killing_one_worker_mid_burst_leaves_the_sibling_serving_and_logging_once(rig: Rig) -> None:
    workers: Final = eventually(rig.owned.worker_pids, lambda pids: len(pids) == 2, seconds=30)
    with ThreadPoolExecutor(max_workers=12) as pool, rig.held.released_on_exit():
        in_flight: Final = tuple(pool.submit(_post_chat, rig, "ia_held") for _ in range(12))
        eventually(rig.held.held.qsize, lambda size: size == len(in_flight), seconds=120)
        held_by: Final = MappingProxyType({pid: open_connections_to(pid, rig.peers["ia_held"].url) for pid in workers})
        assert sum(held_by.values()) == len(in_flight), held_by
        victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
        assert held_by[survivor_pid] >= 1, held_by
        with rig.owned.replacing_worker(victim_pid) as kill:
            kill()
            rig.held.release.set()
            during: Final = tuple(future.result() for future in in_flight)
    after: Final = tuple(_post_chat(rig, "ia_ok") for _ in range(6))
    assert [status for status, _ in after] == [200] * 6, after
    assert [status for status, _ in during].count(200) == held_by[survivor_pid], (held_by, during)
    assert all(status in (200, 0, 500, 502, 503) for status, _ in during), during
    identities: Final = tuple(
        identity for identity in (_completion_id(status, text) for status, text in (*during, *after)) if identity
    )
    assert len(set(identities)) == len(identities)
    _assert_one_spend_row_each(identities)
    assert rig.gateway.request("GET", "/health/liveliness").status_code == 200
