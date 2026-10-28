import json
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, held, object_value
from integration._support.database import read_rows
from integration._support.process import WorkerHealthcheck, graceful_stop_seconds, owned_proxy_process
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.shares_worker_state

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_PROJECT: Final = "scripted-project"
_ADAPTER_MODEL: Final = "settings-adapter-openai"
_NATIVE_MODEL: Final = "settings-native-gemini"
_REPLY_TEXT: Final = "settings reply"
_PROMPT_TOKENS: Final = 13
_COMPLETION_TOKENS: Final = 4
_OPENAI_USAGE: Final = {
    "prompt_tokens": _PROMPT_TOKENS,
    "completion_tokens": _COMPLETION_TOKENS,
    "total_tokens": _PROMPT_TOKENS + _COMPLETION_TOKENS,
}
_GEMINI_USAGE: Final = {
    "promptTokenCount": _PROMPT_TOKENS,
    "candidatesTokenCount": _COMPLETION_TOKENS,
    "totalTokenCount": _PROMPT_TOKENS + _COMPLETION_TOKENS,
}
_ZERO_USAGE: Final = {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0}
_BURST_SELECT: Final = (
    'SELECT api_key, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs"'
    " WHERE api_key = ANY(string_to_array(%s, ','))"
)
_CACHE_SELECT: Final = (
    'SELECT cache_hit, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE api_key=%s ORDER BY "startTime"'
)
_CALLER_OPTIONS: Final = {
    "caller_sends_none": None,
    "caller_opts_in": {"include_usage": True},
    "caller_opts_out": {"include_usage": False},
    "caller_sends_empty_object": {},
}
_UPSTREAM_ASKS_FOR_USAGE: Final = {
    "always_off": {
        "caller_sends_none": False,
        "caller_opts_in": True,
        "caller_opts_out": False,
        "caller_sends_empty_object": False,
    },
    "always_on": {
        "caller_sends_none": True,
        "caller_opts_in": True,
        "caller_opts_out": False,
        "caller_sends_empty_object": True,
    },
}
_SETTING_VALUES: Final = {"always_off": False, "always_on": True}
_STREAMS_PER_WORKER: Final = 6


def _openai_frame(chunk: Mapping[str, JsonValue], response_id: str) -> bytes:
    body: Final = {"id": response_id, "object": "chat.completion.chunk", "created": 1700000000, **chunk}
    return f"data: {json.dumps(body)}\n\n".encode()


def _openai_frames(include_usage: bool) -> tuple[bytes, ...]:
    response_id: Final = f"chatcmpl-{uuid.uuid4().hex}"
    usage: Final = (_openai_frame({"choices": [], "usage": _OPENAI_USAGE}, response_id),) if include_usage else ()
    return (
        _openai_frame({"choices": [{"index": 0, "delta": {"role": "assistant", "content": _REPLY_TEXT}}]}, response_id),
        _openai_frame({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}, response_id),
        *usage,
        b"data: [DONE]\n\n",
    )


def _native_frames() -> tuple[bytes, ...]:
    text: Final = {"candidates": [{"content": {"parts": [{"text": _REPLY_TEXT}], "role": "model"}, "index": 0}]}
    stop: Final = {
        "candidates": [{"content": {"parts": [], "role": "model"}, "finishReason": "STOP", "index": 0}],
        "usageMetadata": _GEMINI_USAGE,
    }
    return (f"data: {json.dumps(text)}\n\n".encode(), f"data: {json.dumps(stop)}\n\n".encode())


def _asks_for_usage(body: Mapping[str, JsonValue]) -> bool:
    options: Final = body.get("stream_options")
    return isinstance(options, dict) and options.get("include_usage") is True


def _peer(gate: threading.Event | None = None) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if ":streamGenerateContent" in request.target:
            return Reply(content_type="text/event-stream", chunks=_native_frames())
        body: Final = _JSON_OBJECT.validate_json(request.body)
        return Reply(
            content_type="text/event-stream",
            chunks=_openai_frames(_asks_for_usage(body)),
            gate_after_first=gate,
        )

    return respond


def _owned_config(path: Path, gateway: Gateway, wire: Wire, general: Mapping[str, JsonValue]) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    path.write_text(
        yaml.safe_dump(
            {
                **config,
                "model_list": [
                    {
                        "model_name": _ADAPTER_MODEL,
                        "litellm_params": {
                            "model": "openai/gpt-5.4-mini",
                            "api_base": wire.url,
                            "api_key": "synthetic-provider-key",
                        },
                    },
                    {
                        "model_name": _NATIVE_MODEL,
                        "litellm_params": {
                            "model": "vertex_ai/gemini-2.5-flash",
                            "api_base": wire.url,
                            "vertex_project": _PROJECT,
                            "vertex_location": "us-central1",
                            "vertex_credentials": service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
                        },
                    },
                ],
                "general_settings": {**(config.get("general_settings") or {}), **general},
            }
        )
    )
    return path


def _stream_body(marker: str, stream_options: JsonValue) -> dict[str, JsonValue]:
    base: Final = {"contents": [{"role": "user", "parts": [{"text": f"settings probe {marker}"}]}]}
    return base if stream_options is None else {**base, "stream_options": stream_options}


def _stream(client: httpx.Client, model: str, key: str, body: Mapping[str, JsonValue]) -> httpx.Response:
    return client.post(
        f"/v1beta/models/{model}:streamGenerateContent?alt=sse",
        json=dict(body),
        headers={"Authorization": f"Bearer {key}"},
    )


def _frames(response: httpx.Response) -> list[dict[str, JsonValue]]:
    parts: Final = [part for part in response.text.split("\n\n") if part.strip()]
    assert all(part.startswith("data: ") for part in parts), response.text
    return [_JSON_OBJECT.validate_json(part.removeprefix("data: ")) for part in parts]


def _stop_frames(frames: Sequence[Mapping[str, JsonValue]]) -> list[Mapping[str, JsonValue]]:
    return [
        frame
        for frame in frames
        if any(object_value(candidate).get("finishReason") for candidate in frame.get("candidates") or [])
    ]


def _model_bodies(wire: Wire) -> list[dict[str, JsonValue]]:
    return [_JSON_OBJECT.validate_json(request.body) for request in wire.drain()]


def _assert_one_stop_with(response: httpx.Response, usage: Mapping[str, int]) -> None:
    assert response.status_code == 200, response.text
    stops: Final = _stop_frames(_frames(response))
    assert len(stops) == 1, response.text
    assert stops[0]["usageMetadata"] == usage, response.text


def _assert_caller_option_reaches_upstream(
    client: httpx.Client, key: str, wire: Wire, setting: str, caller: str
) -> None:
    response: Final = _stream(client, _ADAPTER_MODEL, key, _stream_body(uuid.uuid4().hex, _CALLER_OPTIONS[caller]))
    asks: Final = _UPSTREAM_ASKS_FOR_USAGE[setting][caller]
    _assert_one_stop_with(response, _GEMINI_USAGE if asks else _ZERO_USAGE)
    bodies: Final = _model_bodies(wire)
    assert len(bodies) == 1, (caller, bodies)
    assert bodies[0].get("stream_options") == ({"include_usage": True} if asks else None), (caller, bodies[0])


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
@pytest.mark.parametrize("setting", tuple(_SETTING_VALUES))
def test_always_include_stream_usage_decides_what_the_adapter_asks_upstream(
    gateway: Gateway, tmp_path: Path, setting: str
) -> None:
    with ExitStack() as stack:
        wire: Final = stack.enter_context(wire_server(_peer()))
        config: Final = _owned_config(
            tmp_path / f"{setting}.yaml", gateway, wire, {"always_include_stream_usage": _SETTING_VALUES[setting]}
        )
        owned: Final = stack.enter_context(owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2))
        scenario: Final = stack.enter_context(owned.gateway.scenario())
        key: Final = scenario.key(models=[_ADAPTER_MODEL, _NATIVE_MODEL])
        for caller in _CALLER_OPTIONS:
            _assert_caller_option_reaches_upstream(owned.gateway.client, key, wire, setting, caller)
        native: Final = _stream(owned.gateway.client, _NATIVE_MODEL, key, _stream_body(uuid.uuid4().hex, None))
        _assert_one_stop_with(native, _GEMINI_USAGE)
        native_bodies: Final = _model_bodies(wire)
        assert len(native_bodies) == 1, native_bodies
        assert "stream_options" not in native_bodies[0] and "streamOptions" not in native_bodies[0], native_bodies[0]


def _cache_rows(key: str) -> list[dict[str, JsonValue]]:
    digest: Final = sha256(key.encode()).hexdigest()
    eventually(lambda: read_rows(_CACHE_SELECT, (digest,)), lambda rows: len(rows) >= 2, seconds=70)
    return held(lambda: read_rows(_CACHE_SELECT, (digest,)), lambda rows: len(rows) == 2, holding=3)


def _assert_cache_hit_replays_the_stop_usage(
    client: httpx.Client, key: str, wire: Wire, setting: str, caller: str
) -> None:
    body: Final = _stream_body(uuid.uuid4().hex, _CALLER_OPTIONS[caller])
    usage: Final = _GEMINI_USAGE if _UPSTREAM_ASKS_FOR_USAGE[setting][caller] else _ZERO_USAGE
    _assert_one_stop_with(_stream(client, _ADAPTER_MODEL, key, body), usage)
    _assert_one_stop_with(_stream(client, _ADAPTER_MODEL, key, body), usage)
    assert len(_model_bodies(wire)) == 1, caller
    rows: Final = _cache_rows(key)
    assert [row["cache_hit"] == "True" for row in rows] == [False, True], (caller, rows)
    assert (rows[1]["prompt_tokens"], rows[1]["completion_tokens"]) == (
        rows[0]["prompt_tokens"],
        rows[0]["completion_tokens"],
    ), (caller, rows)


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
@pytest.mark.parametrize("setting", tuple(_SETTING_VALUES))
def test_always_include_stream_usage_cache_hit_replays_the_stop_usage_the_miss_sent(
    gateway: Gateway, tmp_path: Path, setting: str
) -> None:
    with ExitStack() as stack:
        wire: Final = stack.enter_context(wire_server(_peer()))
        config: Final = _owned_config(
            tmp_path / f"{setting}-cache.yaml", gateway, wire, {"always_include_stream_usage": _SETTING_VALUES[setting]}
        )
        owned: Final = stack.enter_context(owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2))
        scenario: Final = stack.enter_context(owned.gateway.scenario())
        for caller in _CALLER_OPTIONS:
            _assert_cache_hit_replays_the_stop_usage(
                owned.gateway.client, scenario.key(models=[_ADAPTER_MODEL]), wire, setting, caller
            )


def _burst_rows(digests: Sequence[str]) -> list[dict[str, JsonValue]]:
    return read_rows(_BURST_SELECT, (",".join(digests),))


def _rows_by_key(digests: Sequence[str], expected: int) -> list[dict[str, JsonValue]]:
    eventually(lambda: _burst_rows(digests), lambda rows: len(rows) >= expected, seconds=120)
    return held(lambda: _burst_rows(digests), lambda rows: len(rows) == expected, holding=5)


def _stream_or_dropped(client: httpx.Client, key: str) -> httpx.Response | None:
    try:
        return _stream(client, _ADAPTER_MODEL, key, _stream_body(uuid.uuid4().hex, None))
    except httpx.TransportError:
        return None


def _held_streams(
    stack: ExitStack, pool: ThreadPoolExecutor, url: str, keys: Sequence[str], wire: Wire, held: int
) -> tuple[Future[httpx.Response | None], ...]:
    clients: Final = tuple(stack.enter_context(httpx.Client(base_url=url, timeout=60, trust_env=False)) for _ in keys)
    futures: Final = tuple(
        pool.submit(_stream_or_dropped, client, key) for client, key in zip(clients, keys, strict=True)
    )
    eventually(wire.received.qsize, lambda size: size >= held, seconds=60)
    return futures


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_worker_killed_mid_stream_leaves_the_sibling_billing_each_stream_once(gateway: Gateway, tmp_path: Path) -> None:
    gate: Final = threading.Event()
    with ExitStack() as stack:
        wire: Final = stack.enter_context(wire_server(_peer(gate)))
        stack.callback(gate.set)
        config: Final = _owned_config(tmp_path / "worker-kill.yaml", gateway, wire, {})
        owned: Final = stack.enter_context(
            owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2, healthcheck=WorkerHealthcheck.LENIENT)
        )
        victim, survivor = sorted(eventually(owned.worker_pids, lambda pids: len(pids) == 2, seconds=30))
        scenario: Final = stack.enter_context(owned.gateway.scenario())
        victim_keys: Final = tuple(scenario.key(models=[_ADAPTER_MODEL]) for _ in range(_STREAMS_PER_WORKER))
        survivor_keys: Final = tuple(scenario.key(models=[_ADAPTER_MODEL]) for _ in range(_STREAMS_PER_WORKER))
        proxy_url: Final = str(owned.gateway.client.base_url)
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=2 * _STREAMS_PER_WORKER))
        with owned.suspended(survivor):
            on_victim: Final = _held_streams(stack, pool, proxy_url, victim_keys, wire, _STREAMS_PER_WORKER)
        with owned.suspended(victim):
            on_survivor: Final = _held_streams(stack, pool, proxy_url, survivor_keys, wire, 2 * _STREAMS_PER_WORKER)
        with owned.replacing_worker(victim) as kill:
            kill()
            gate.set()
            dropped: Final = tuple(future.result(timeout=60) for future in on_victim)
            survivors: Final = tuple(
                zip(survivor_keys, (future.result(timeout=60) for future in on_survivor), strict=True)
            )
        assert dropped == (None,) * _STREAMS_PER_WORKER, dropped
        for _, result in survivors:
            assert result is not None
            _assert_one_stop_with(result, _GEMINI_USAGE)
        second_keys: Final = tuple(scenario.key(models=[_ADAPTER_MODEL]) for _ in range(6))
        second: Final = tuple(_stream_or_dropped(owned.gateway.client, key) for key in second_keys)
        for result in second:
            assert result is not None
            _assert_one_stop_with(result, _GEMINI_USAGE)
        billed: Final = tuple(key for key, _ in survivors) + second_keys
        digests: Final = tuple(sha256(key.encode()).hexdigest() for key in billed)
        rows: Final = _rows_by_key(digests, len(billed))
        assert sorted(str(row["api_key"]) for row in rows) == sorted(digests)
        assert {(row["prompt_tokens"], row["completion_tokens"]) for row in rows} == {
            (_PROMPT_TOKENS, _COMPLETION_TOKENS)
        }, rows
        assert owned.process.poll() is None
