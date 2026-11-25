import itertools
import json
import threading
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.anthropic_sse import message_json, message_stream, stream_reply
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.process import owned_proxy
from integration._support.responses_stream import completed, created, delta, frame, healthy_stream
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

_MODEL: Final = "claude-haiku-5-5"
_API_KEY: Final = "synthetic-anthropic-key"
_CALLS: Final = 12
_SLACK: Final = 2
_NEVER: Final = 1_000_000_000


def _log_records(gateway: Gateway) -> int:
    response: Final = gateway.request(
        "GET", "/debug/memory/details", params={"top_n": "100000", "include_process_info": "false"}
    )
    assert response.status_code == 200, response.text
    objects: Final = object_value(JSON_OBJECT.validate_python(response.json())["objects"])
    top_types: Final = objects["top_types"]
    assert isinstance(top_types, list), objects
    counts: Final = {string_value(object_value(entry)["type"]): object_value(entry)["count"] for entry in top_types}
    count: Final = counts.get("Logging", 0)
    assert isinstance(count, int), counts
    return count


_CONFIG_MODEL: Final = "freed-config-model"
_RESPONSES_MODEL: Final = "freed-responses-model"
_DROPPED_MARKER: Final = "dropped-mid-stream"
_HELD_OPEN: Final = threading.Event()


@pytest.fixture(scope="module")
def uncached_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    upstream: Final = wire_server(_answer)
    wire: Final = upstream.__enter__()
    directory: Final = tmp_path_factory.mktemp("finished-requests-freed")
    stock: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    settings: Final = {
        key: value
        for key, value in JSON_OBJECT.validate_python(stock.get("litellm_settings") or {}).items()
        if key not in ("cache", "cache_params")
    }
    config: Final = directory / "finished-requests-freed.yaml"
    model_list: Final = [
        {
            "model_name": _CONFIG_MODEL,
            "litellm_params": {"model": f"anthropic/{_MODEL}", "api_base": wire.url, "api_key": _API_KEY},
        },
        {
            "model_name": _RESPONSES_MODEL,
            "litellm_params": {"model": "openai/gpt-6.1-sol", "api_base": f"{wire.url}/v1", "api_key": _API_KEY},
        },
    ]
    config.write_text(yaml.safe_dump({**stock, "litellm_settings": settings, "model_list": model_list}))
    try:
        with gateway_from_environment() as shared, owned_proxy(shared, directory, {}, config=config) as owned:
            yield owned
    finally:
        upstream.__exit__(None, None, None)


@pytest.fixture
def no_cleanup_pass(uncached_gateway: Gateway) -> Iterator[None]:
    gateway: Final = uncached_gateway
    configured: Final = gateway.request("POST", "/debug/memory/gc/configure", params={"generation_0": str(_NEVER)})
    assert configured.status_code == 200, configured.text
    previous: Final = [
        part.strip()
        for part in string_value(JSON_OBJECT.validate_python(configured.json())["previous_thresholds"]).split(",")
    ]
    try:
        yield
    finally:
        restored: Final = gateway.request(
            "POST",
            "/debug/memory/gc/configure",
            params={"generation_0": previous[0], "generation_1": previous[1], "generation_2": previous[2]},
        )
        assert restored.status_code == 200, restored.text


def _responses_answer(request: Request) -> Reply:
    identity: Final = f"resp_{uuid.uuid4().hex}"
    if _DROPPED_MARKER.encode() not in request.body:
        return Reply(content_type="text/event-stream", chunks=healthy_stream(identity, "Hi!"))
    return Reply(
        content_type="text/event-stream",
        chunks=(frame(created(identity)) + frame(delta(identity, "Hi!")), frame(completed(identity, "Hi!"))),
        gate_after_first=_HELD_OPEN,
    )


def _answer(request: Request) -> Reply:
    if request.method == "POST" and request.target == "/v1/responses":
        return _responses_answer(request)
    assert request.method == "POST" and request.target == "/v1/messages"
    streamed: Final = JSON_OBJECT.validate_python(json.loads(request.body)).get("stream") is True
    return (
        stream_reply(message_stream("msg_freed", _MODEL, "Hi!"))
        if streamed
        else Reply(body=message_json("msg_freed", _MODEL, "Hi!"))
    )


def _body(path: str, stream: bool, prompt: str) -> Mapping[str, JsonValue]:
    messages: Final[list[JsonValue]] = [{"role": "user", "content": prompt}]
    match path:
        case "/v1/messages":
            return {"model": _CONFIG_MODEL, "max_tokens": 16, "messages": messages, "stream": stream}
        case "/v1/responses":
            return {"model": _RESPONSES_MODEL, "input": prompt, "stream": stream}
        case _:
            return {"model": _CONFIG_MODEL, "messages": messages, "stream": stream}


def _call(gateway: Gateway, path: str, stream: bool) -> None:
    response: Final = gateway.request("POST", path, _body(path, stream, f"Say hi {uuid.uuid4().hex}"))
    assert response.status_code == 200, response.text
    assert "Hi!" in response.text, response.text


@pytest.mark.parametrize(
    ("path", "stream"),
    (
        ("/v1/messages", False),
        ("/v1/messages", True),
        ("/v1/chat/completions", False),
        ("/v1/chat/completions", True),
        ("/v1/responses", True),
    ),
    ids=("messages", "streamed-messages", "chat-completions", "streamed-chat-completions", "streamed-responses"),
)
def test_finished_calls_are_freed_without_a_cleanup_pass(
    uncached_gateway: Gateway, no_cleanup_pass: None, path: str, stream: bool
) -> None:
    gateway: Final = uncached_gateway
    _call(gateway, path, stream)
    before: Final = _log_records(gateway)
    for _ in range(_CALLS):
        _call(gateway, path, stream)
    after: Final = eventually(
        lambda: _log_records(gateway), lambda count: count <= before + _SLACK, seconds=15, return_last_on_timeout=True
    )
    assert after <= before + _SLACK, (
        f"{after - before} of {_CALLS} finished {'streamed ' if stream else ''}{path} calls still hold their log "
        "record with Python's cycle collector held off, so each one waits for a full cleanup pass "
        "before its memory comes back"
    )


def _drop_mid_stream(gateway: Gateway) -> None:
    body: Final = _body("/v1/responses", True, f"{_DROPPED_MARKER} {uuid.uuid4().hex}")
    headers: Final = {"Authorization": f"Bearer {gateway.key}"}
    with gateway.client.stream("POST", "/v1/responses", json=body, headers=headers) as response:
        assert response.status_code == 200, response.read()
        assert any("Hi!" in text for text in itertools.accumulate(response.iter_text()))


def test_a_responses_stream_the_client_drops_mid_stream_is_freed_without_a_cleanup_pass(
    uncached_gateway: Gateway, no_cleanup_pass: None
) -> None:
    gateway: Final = uncached_gateway
    _HELD_OPEN.clear()
    try:
        _drop_mid_stream(gateway)
        before: Final = _log_records(gateway)
        for _ in range(_CALLS):
            _drop_mid_stream(gateway)
        after: Final = eventually(
            lambda: _log_records(gateway),
            lambda count: count <= before + _SLACK,
            seconds=15,
            return_last_on_timeout=True,
        )
    finally:
        _HELD_OPEN.set()
    assert after <= before + _SLACK, (
        f"{after - before} of {_CALLS} /v1/responses streams the client dropped mid-stream still hold their log "
        "record with Python's cycle collector held off, so each one waits for a full cleanup pass "
        "before its memory comes back"
    )
