from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import unquote

import httpx
import pytest
import yaml
from integration._support.anthropic_sse import error_body, message_json, message_stream, parse_sse, stream_reply
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.openai_wire import chat_reply, responses_reply
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

from litellm.constants import PROXY_CONFIG_RELOAD_INTERVAL_SECONDS

Endpoint = Literal["chat", "messages", "responses"]

_OPENAI_MODEL: Final = "gpt-5.4"
_ANTHROPIC_MODEL: Final = "claude-haiku-4-5"
_API_KEY: Final = "synthetic-fallback-errors-key"
_ROUTER_ONLY_KEYS: Final = ("include_fallback_errors", "silent_model")
_ANSWER: Final = "answered by"
_UNAUTHORIZED_MESSAGE: Final = "scripted 401: the primary key was revoked"
_NONCE: Final = re.compile(r"nonce=([0-9a-f]{32})")
_JSON: Final = TypeAdapter(dict[str, JsonValue])
_ERRORS: Final = TypeAdapter(list[dict[str, JsonValue]])
_ITEMS: Final = TypeAdapter(list[JsonValue])
_UPSTREAM_URL_PLACEHOLDER: Final = "upstream-url"
_NOT_YET_ON_EVERY_WORKER: Final = (
    "Invalid model name passed in model=",
    "There are no healthy deployments for this model",
)
_PROXY_WORKERS: Final = int(os.environ.get("INTEGRATION_PROXY_WORKERS", "1"))
_WORKER_SYNC_SECONDS: Final = 0.0 if _PROXY_WORKERS == 1 else PROXY_CONFIG_RELOAD_INTERVAL_SECONDS + 5.0
_FRESH_CONNECTION: Final = MappingProxyType({"Connection": "close"})
_SPEND_ROW_SECONDS: Final = 70
_BURST: Final = 24
_ENDPOINTS: Final[tuple[Endpoint, ...]] = ("chat", "messages", "responses")
_PATHS: Final = MappingProxyType(
    {"chat": "/v1/chat/completions", "messages": "/v1/messages", "responses": "/v1/responses"}
)
_PROVIDERS: Final = MappingProxyType(
    {
        "chat": f"openai/{_OPENAI_MODEL}",
        "messages": f"anthropic/{_ANTHROPIC_MODEL}",
        "responses": f"openai/{_OPENAI_MODEL}",
    }
)
_STREAMING: Final = (pytest.param(False, id="non-stream"), pytest.param(True, id="stream"))
_NON_BOOLEAN_FLAGS: Final = (
    pytest.param(1, True, id="int"),
    pytest.param("", False, id="empty-string"),
    pytest.param([], False, id="list"),
    pytest.param("x" * 5120, True, id="five-kilobytes"),
)
_OPENAI_UNAUTHORIZED: Final = Reply(
    status=401,
    body=json.dumps(
        {
            "error": {
                "message": _UNAUTHORIZED_MESSAGE,
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_api_key",
            }
        }
    ).encode(),
)
_ANTHROPIC_UNAUTHORIZED: Final = Reply(status=401, body=error_body(401, _UNAUTHORIZED_MESSAGE))

_EXPOSED_OPENAI_PRIMARY: Final = "exposed-openai-primary"
_EXPOSED_OPENAI_BACKUP: Final = "exposed-openai-backup"
_EXPOSED_OPENAI_SERVING: Final = "exposed-openai-serving"
_EXPOSED_OPENAI_FLAKY: Final = "exposed-openai-flaky"
_EXPOSED_ANTHROPIC_PRIMARY: Final = "exposed-anthropic-primary"
_EXPOSED_ANTHROPIC_BACKUP: Final = "exposed-anthropic-backup"

_CHAT_BRIDGE_INHERITED_LEAKS: Final = frozenset(
    f"{deployment}:include_fallback_errors" for deployment in (_EXPOSED_ANTHROPIC_PRIMARY, _EXPOSED_ANTHROPIC_BACKUP)
)

pytestmark = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)


def _prompt(nonce: str, *, outage: bool = False) -> str:
    return f"which deployment answers this? nonce={nonce} outage={int(outage)}"


def _nonce_of(text: str) -> str:
    found: Final = _NONCE.search(text)
    assert found is not None, text
    return found.group(1)


def _outage(deployment: str, raw: str) -> bool:
    return deployment.endswith("primary") or (deployment.endswith("flaky") and "outage=1" in raw)


def _identity(endpoint: Endpoint, deployment: str, nonce: str) -> str:
    match endpoint:
        case "chat":
            return f"chatcmpl-{deployment}-{nonce}"
        case "messages":
            return f"msg_{deployment}_{nonce}"
        case "responses":
            return f"resp_{deployment}_{nonce}"


def _peer(request: Request) -> Reply:
    if request.method == "GET":
        return Reply(body=json.dumps({"object": "list", "data": []}).encode())
    deployment, _, route = unquote(request.target).lstrip("/").partition("/")
    raw: Final = request.body.decode()
    body: Final = _JSON.validate_json(request.body)
    nonce: Final = _nonce_of(raw)
    stream: Final = body.get("stream") is True
    text: Final = f"{_ANSWER} {deployment}"
    if route == "v1/messages":
        if _outage(deployment, raw):
            return _ANTHROPIC_UNAUTHORIZED
        identity: Final = _identity("messages", deployment, nonce)
        if stream:
            return stream_reply(message_stream(identity, _ANTHROPIC_MODEL, text))
        return Reply(body=message_json(identity, _ANTHROPIC_MODEL, text))
    if route.endswith("responses"):
        if _outage(deployment, raw):
            return _OPENAI_UNAUTHORIZED
        return responses_reply(_identity("responses", deployment, nonce), _OPENAI_MODEL, text, stream=stream)
    assert route == "chat/completions", request.target
    if _outage(deployment, raw):
        return _OPENAI_UNAUTHORIZED
    return chat_reply(_identity("chat", deployment, nonce), _OPENAI_MODEL, text, stream=stream)


@dataclass(frozen=True, slots=True)
class _Posted:
    deployment: str
    route: str
    raw: str

    def leaked(self) -> tuple[str, ...]:
        return tuple(key for key in _ROUTER_ONLY_KEYS if key in self.raw)


def _posted_of(request: Request) -> _Posted:
    deployment, _, route = unquote(request.target).lstrip("/").partition("/")
    return _Posted(deployment, route, request.body.decode())


def _all_posted(wire: Wire) -> tuple[_Posted, ...]:
    return tuple(_posted_of(request) for request in wire.drain() if request.method == "POST")


def _posted(wire: Wire, nonce: str) -> tuple[_Posted, ...]:
    return tuple(item for item in _all_posted(wire) if nonce in item.raw)


def _header(response: httpx.Response, name: str) -> str | None:
    return response.headers[name] if name in response.headers else None


def _leaks(posted: tuple[_Posted, ...]) -> tuple[str, ...]:
    return tuple(f"{item.deployment}:{','.join(item.leaked())}" for item in posted if item.leaked())


def _hit(posted: tuple[_Posted, ...]) -> tuple[str, ...]:
    return tuple(item.deployment for item in posted)


def _body(endpoint: Endpoint, model: str, nonce: str, *, stream: bool, outage: bool = False) -> dict[str, JsonValue]:
    prompt: Final = _prompt(nonce, outage=outage)
    match endpoint:
        case "chat":
            return {"model": model, "stream": stream, "messages": [{"role": "user", "content": prompt}]}
        case "messages":
            return {
                "model": model,
                "stream": stream,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": prompt}],
            }
        case "responses":
            return {"model": model, "stream": stream, "input": prompt}


def _output_item_identity(item: JsonValue) -> str:
    return str(_JSON.validate_python(item)["id"]).removeprefix("msg_")


def _served_id(endpoint: Endpoint, response: httpx.Response, *, stream: bool) -> str:
    if not stream:
        body: Final = _JSON.validate_json(response.content)
        if endpoint == "responses":
            return _output_item_identity(_ITEMS.validate_python(body["output"])[0])
        return str(body["id"])
    events: Final = parse_sse(response.text)
    match endpoint:
        case "chat":
            return str(events[0].data["id"])
        case "messages":
            start: Final = next(event for event in events if event.event == "message_start")
            return str(_JSON.validate_python(start.data["message"])["id"])
        case "responses":
            done: Final = next(event for event in events if event.data.get("type") == "response.output_item.done")
            return _output_item_identity(done.data["item"])


def _settled(text: str) -> bool:
    return not any(phrase in text for phrase in _NOT_YET_ON_EVERY_WORKER)


@dataclass(frozen=True, slots=True)
class _Sent:
    nonce: str
    response: httpx.Response


def _send(
    gateway: Gateway, path: str, body: Callable[[str], Mapping[str, JsonValue]], *, key: str | None = None
) -> _Sent:
    def attempt() -> _Sent:
        nonce: Final = uuid.uuid4().hex
        return _Sent(nonce, gateway.request("POST", path, body(nonce), key=key, headers=_FRESH_CONNECTION))

    return eventually(attempt, lambda sent: _settled(sent.response.text), seconds=_WORKER_SYNC_SECONDS + 10)


@dataclass(frozen=True, slots=True)
class _Observed:
    status: int
    served_id: str
    spend_id: str
    text: str
    attempted: str | None
    errors_header: str | None
    hit: tuple[str, ...]
    leaks: tuple[str, ...]


def _observe(endpoint: Endpoint, wire: Wire, sent: _Sent, *, stream: bool) -> _Observed:
    posted: Final = _posted(wire, sent.nonce)
    response: Final = sent.response
    return _Observed(
        status=response.status_code,
        served_id=_served_id(endpoint, response, stream=stream) if response.status_code == 200 else "",
        spend_id=_spend_id(response) if response.status_code == 200 and not stream else "",
        text=response.text,
        attempted=_header(response, "x-litellm-attempted-fallbacks"),
        errors_header=_header(response, "x-litellm-fallback-errors"),
        hit=_hit(posted),
        leaks=_leaks(posted),
    )


def _spend_id(response: httpx.Response) -> str:
    return str(_JSON.validate_json(response.content)["id"])


def _spend_row_lands(spend_id: str) -> None:
    eventually(
        lambda: read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (spend_id,)),
        lambda rows: len(rows) == 1,
        seconds=_SPEND_ROW_SECONDS,
    )


@dataclass(frozen=True, slots=True)
class _Registered:
    gateway: Gateway
    wire: Wire
    primary: Mapping[Endpoint, str]
    backup: Mapping[Endpoint, str]


@pytest.fixture(scope="module")
def registered() -> Iterator[_Registered]:
    with gateway_from_environment() as gateway, wire_server(_peer) as wire, gateway.scenario() as scenario:
        primary: Final[dict[Endpoint, str]] = {
            endpoint: scenario.model(model=_PROVIDERS[endpoint], api_key=_API_KEY, api_base=f"{wire.url}/primary")
            for endpoint in _ENDPOINTS
        }
        backup: Final[dict[Endpoint, str]] = {
            endpoint: scenario.model(model=_PROVIDERS[endpoint], api_key=_API_KEY, api_base=f"{wire.url}/backup")
            for endpoint in _ENDPOINTS
        }
        yield _Registered(gateway, wire, MappingProxyType(primary), MappingProxyType(backup))


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("stream", _STREAMING)
def test_default_gateway_keeps_the_flag_off_the_wire_on_a_fallback(
    registered: _Registered, endpoint: Endpoint, stream: bool
) -> None:
    sent: Final = _send(
        registered.gateway,
        _PATHS[endpoint],
        lambda nonce: {
            **_body(endpoint, registered.primary[endpoint], nonce, stream=stream),
            "fallbacks": [registered.backup[endpoint]],
            "include_fallback_errors": True,
        },
    )
    observed: Final = _observe(endpoint, registered.wire, sent, stream=stream)
    assert observed.status == 200, observed
    assert f"{_ANSWER} backup" in observed.text, observed
    assert observed.served_id == _identity(endpoint, "backup", sent.nonce), observed
    assert observed.leaks == (), observed
    assert observed.hit == ("primary", "backup"), observed
    assert observed.errors_header is None, observed
    if not stream:
        _spend_row_lands(observed.spend_id)
    if endpoint == "chat" and not stream:
        assert observed.attempted == "1", observed


@pytest.mark.parametrize(("value", "errors_reported"), _NON_BOOLEAN_FLAGS)
def test_default_gateway_keeps_a_non_boolean_flag_off_the_wire(
    registered: _Registered, value: JsonValue, errors_reported: bool
) -> None:
    sent: Final = _send(
        registered.gateway,
        _PATHS["chat"],
        lambda nonce: {
            **_body("chat", registered.primary["chat"], nonce, stream=False),
            "fallbacks": [registered.backup["chat"]],
            "include_fallback_errors": value,
        },
    )
    observed: Final = _observe("chat", registered.wire, sent, stream=False)
    assert observed.status == 200, observed
    assert f"{_ANSWER} backup" in observed.text, observed
    assert observed.leaks == (), observed
    assert observed.hit == ("primary", "backup"), observed
    assert observed.errors_header is None, observed


def test_default_gateway_keeps_a_duplicated_raw_flag_off_the_wire(registered: _Registered) -> None:
    def raw_body(nonce: str) -> str:
        body: Final = {
            **_body("chat", registered.primary["chat"], nonce, stream=False),
            "fallbacks": [registered.backup["chat"]],
        }
        return json.dumps(body)[:-1] + ', "include_fallback_errors": true, "include_fallback_errors": true}'

    def attempt() -> _Sent:
        nonce: Final = uuid.uuid4().hex
        response: Final = registered.gateway.client.post(
            _PATHS["chat"],
            content=raw_body(nonce),
            headers={
                "Authorization": f"Bearer {registered.gateway.key}",
                "Content-Type": "application/json",
                **_FRESH_CONNECTION,
            },
        )
        return _Sent(nonce, response)

    sent: Final = eventually(attempt, lambda s: _settled(s.response.text), seconds=_WORKER_SYNC_SECONDS + 10)
    observed: Final = _observe("chat", registered.wire, sent, stream=False)
    assert observed.status == 200, observed
    assert observed.served_id == _identity("chat", "backup", sent.nonce), observed
    assert observed.leaks == (), observed
    assert observed.hit == ("primary", "backup"), observed
    assert observed.errors_header is None, observed


def test_default_gateway_rejects_an_unauthenticated_flagged_request_before_the_wire(registered: _Registered) -> None:
    nonce: Final = uuid.uuid4().hex
    response: Final = registered.gateway.request(
        "POST",
        _PATHS["chat"],
        {
            **_body("chat", registered.primary["chat"], nonce, stream=False),
            "fallbacks": [registered.backup["chat"]],
            "include_fallback_errors": True,
        },
        key="sk-not-a-key",
        headers=_FRESH_CONNECTION,
    )
    assert response.status_code == 401, response.text
    assert _posted(registered.wire, nonce) == ()


def _exposed_deployment(name: str, provider: str) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {
            "model": provider,
            "api_base": f"{_UPSTREAM_URL_PLACEHOLDER}/{name}",
            "api_key": _API_KEY,
        },
    }


def _exposed_config(wire: Wire, directory: Path) -> Path:
    config: Final = _JSON.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config["model_list"] = [
        _exposed_deployment(_EXPOSED_OPENAI_PRIMARY, _PROVIDERS["chat"]),
        _exposed_deployment(_EXPOSED_OPENAI_BACKUP, _PROVIDERS["chat"]),
        _exposed_deployment(_EXPOSED_OPENAI_SERVING, _PROVIDERS["chat"]),
        _exposed_deployment(_EXPOSED_OPENAI_FLAKY, _PROVIDERS["chat"]),
        _exposed_deployment(_EXPOSED_ANTHROPIC_PRIMARY, _PROVIDERS["messages"]),
        _exposed_deployment(_EXPOSED_ANTHROPIC_BACKUP, _PROVIDERS["messages"]),
    ]
    config["general_settings"] = {
        **_JSON.validate_python(config["general_settings"]),
        "expose_fallback_errors_to_caller": True,
    }
    config["router_settings"] = {
        "num_retries": 0,
        "disable_cooldowns": True,
        "fallbacks": [
            {_EXPOSED_OPENAI_PRIMARY: [_EXPOSED_OPENAI_BACKUP]},
            {_EXPOSED_OPENAI_FLAKY: [_EXPOSED_OPENAI_BACKUP]},
            {_EXPOSED_ANTHROPIC_PRIMARY: [_EXPOSED_ANTHROPIC_BACKUP]},
        ],
    }
    path: Final = directory / "include-fallback-errors-exposed.yaml"
    path.write_text(yaml.safe_dump(config).replace(_UPSTREAM_URL_PLACEHOLDER, wire.url))
    return path


@dataclass(frozen=True, slots=True)
class _Exposed:
    proxy: Gateway
    wire: Wire


@pytest.fixture(scope="module")
def exposed(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Exposed]:
    directory: Final = tmp_path_factory.mktemp("include-fallback-errors-exposed")
    with gateway_from_environment() as gateway, wire_server(_peer) as wire:
        with owned_proxy(gateway, directory, {}, config=_exposed_config(wire, directory), workers=2) as proxy:
            yield _Exposed(proxy, wire)


def _errors_mention_the_outage(errors_header: str | None) -> bool:
    if errors_header is None:
        return False
    errors: Final = _ERRORS.validate_json(errors_header)
    return len(errors) == 1 and _UNAUTHORIZED_MESSAGE in str(errors[0]["message"])


@pytest.mark.parametrize("stream", _STREAMING)
def test_exposed_chat_fallback_reports_the_errors_without_putting_the_flag_on_the_wire(
    exposed: _Exposed, stream: bool
) -> None:
    sent: Final = _send(
        exposed.proxy,
        _PATHS["chat"],
        lambda nonce: {
            **_body("chat", _EXPOSED_OPENAI_PRIMARY, nonce, stream=stream),
            "include_fallback_errors": True,
        },
    )
    observed: Final = _observe("chat", exposed.wire, sent, stream=stream)
    assert observed.status == 200, observed
    assert f"{_ANSWER} {_EXPOSED_OPENAI_BACKUP}" in observed.text, observed
    assert observed.served_id == _identity("chat", _EXPOSED_OPENAI_BACKUP, sent.nonce), observed
    assert observed.leaks == (), observed
    assert observed.hit == (_EXPOSED_OPENAI_PRIMARY, _EXPOSED_OPENAI_BACKUP), observed
    if not stream:
        assert observed.attempted == "1", observed
        assert _errors_mention_the_outage(observed.errors_header), observed


def test_exposed_chat_without_a_fallback_keeps_the_flag_off_the_wire_and_the_errors_header_off(
    exposed: _Exposed,
) -> None:
    sent: Final = _send(
        exposed.proxy,
        _PATHS["chat"],
        lambda nonce: {
            **_body("chat", _EXPOSED_OPENAI_SERVING, nonce, stream=False),
            "include_fallback_errors": True,
        },
    )
    observed: Final = _observe("chat", exposed.wire, sent, stream=False)
    assert observed.status == 200, observed
    assert observed.served_id == _identity("chat", _EXPOSED_OPENAI_SERVING, sent.nonce), observed
    assert observed.leaks == (), observed
    assert observed.hit == (_EXPOSED_OPENAI_SERVING,), observed
    assert observed.attempted == "0", observed
    assert observed.errors_header is None, observed


@pytest.mark.parametrize(("value", "errors_reported"), _NON_BOOLEAN_FLAGS)
def test_exposed_chat_keeps_a_non_boolean_flag_off_the_wire_and_the_errors_header_follows_its_truthiness(
    exposed: _Exposed, value: JsonValue, errors_reported: bool
) -> None:
    sent: Final = _send(
        exposed.proxy,
        _PATHS["chat"],
        lambda nonce: {
            **_body("chat", _EXPOSED_OPENAI_PRIMARY, nonce, stream=False),
            "include_fallback_errors": value,
        },
    )
    observed: Final = _observe("chat", exposed.wire, sent, stream=False)
    assert observed.status == 200, observed
    assert observed.served_id == _identity("chat", _EXPOSED_OPENAI_BACKUP, sent.nonce), observed
    assert observed.leaks == (), observed
    assert observed.hit == (_EXPOSED_OPENAI_PRIMARY, _EXPOSED_OPENAI_BACKUP), observed
    assert observed.attempted == "1", observed
    assert _errors_mention_the_outage(observed.errors_header) is errors_reported, observed


def test_exposed_proxy_rejects_an_unauthenticated_flagged_request_before_the_wire(exposed: _Exposed) -> None:
    nonce: Final = uuid.uuid4().hex
    response: Final = exposed.proxy.request(
        "POST",
        _PATHS["chat"],
        {**_body("chat", _EXPOSED_OPENAI_PRIMARY, nonce, stream=False), "include_fallback_errors": True},
        key="sk-not-a-key",
        headers=_FRESH_CONNECTION,
    )
    assert response.status_code == 401, response.text
    assert _posted(exposed.wire, nonce) == ()


@pytest.mark.parametrize("stream", _STREAMING)
def test_exposed_messages_fallback_keeps_the_flag_off_the_wire(exposed: _Exposed, stream: bool) -> None:
    sent: Final = _send(
        exposed.proxy,
        _PATHS["messages"],
        lambda nonce: {
            **_body("messages", _EXPOSED_ANTHROPIC_PRIMARY, nonce, stream=stream),
            "include_fallback_errors": True,
        },
    )
    observed: Final = _observe("messages", exposed.wire, sent, stream=stream)
    assert observed.status == 200, observed
    assert observed.served_id == _identity("messages", _EXPOSED_ANTHROPIC_BACKUP, sent.nonce), observed
    assert observed.hit == (_EXPOSED_ANTHROPIC_PRIMARY, _EXPOSED_ANTHROPIC_BACKUP), observed
    assert observed.leaks == (), observed


@pytest.mark.parametrize("stream", _STREAMING)
def test_exposed_responses_fallback_keeps_the_flag_off_the_wire(exposed: _Exposed, stream: bool) -> None:
    sent: Final = _send(
        exposed.proxy,
        _PATHS["responses"],
        lambda nonce: {
            **_body("responses", _EXPOSED_OPENAI_PRIMARY, nonce, stream=stream),
            "include_fallback_errors": True,
        },
    )
    observed: Final = _observe("responses", exposed.wire, sent, stream=stream)
    assert observed.status == 200, observed
    assert observed.served_id == _identity("responses", _EXPOSED_OPENAI_BACKUP, sent.nonce), observed
    assert observed.hit == (_EXPOSED_OPENAI_PRIMARY, _EXPOSED_OPENAI_BACKUP), observed
    assert observed.leaks == (), observed


@pytest.mark.parametrize("stream", _STREAMING)
def test_exposed_messages_on_the_openai_deployment_bridges_the_fallback_with_a_clean_wire(
    exposed: _Exposed, stream: bool
) -> None:
    sent: Final = _send(
        exposed.proxy,
        _PATHS["messages"],
        lambda nonce: {
            **_body("messages", _EXPOSED_OPENAI_PRIMARY, nonce, stream=stream),
            "include_fallback_errors": True,
        },
    )
    observed: Final = _observe("messages", exposed.wire, sent, stream=stream)
    assert observed.status == 200, observed
    assert f"{_ANSWER} {_EXPOSED_OPENAI_BACKUP}" in observed.text, observed
    assert observed.hit == (_EXPOSED_OPENAI_PRIMARY, _EXPOSED_OPENAI_BACKUP), observed
    assert observed.leaks == (), observed


@pytest.mark.parametrize("stream", _STREAMING)
def test_exposed_responses_on_the_anthropic_deployment_falls_back_through_the_chat_bridge_with_no_new_key_on_the_wire(
    exposed: _Exposed, stream: bool
) -> None:
    sent: Final = _send(
        exposed.proxy,
        _PATHS["responses"],
        lambda nonce: {
            **_body("responses", _EXPOSED_ANTHROPIC_PRIMARY, nonce, stream=stream),
            "include_fallback_errors": True,
        },
    )
    observed: Final = _observe("responses", exposed.wire, sent, stream=stream)
    assert observed.status == 200, observed
    assert f"{_ANSWER} {_EXPOSED_ANTHROPIC_BACKUP}" in observed.text, observed
    assert observed.hit == (_EXPOSED_ANTHROPIC_PRIMARY, _EXPOSED_ANTHROPIC_BACKUP), observed
    assert set(observed.leaks) <= _CHAT_BRIDGE_INHERITED_LEAKS, observed


@dataclass(frozen=True, slots=True)
class _BurstRequest:
    endpoint: Endpoint
    stream: bool
    outage: bool
    nonce: str


def _burst_plan() -> tuple[_BurstRequest, ...]:
    shapes: Final[tuple[tuple[Endpoint, bool], ...]] = (("chat", False), ("chat", True), ("responses", False))
    return tuple(
        _BurstRequest(endpoint, stream, index % 2 == 1, uuid.uuid4().hex)
        for index, (endpoint, stream) in enumerate(shapes * (_BURST // len(shapes)))
    )


def _fire(proxy: Gateway, request: _BurstRequest) -> httpx.Response:
    return proxy.request(
        "POST",
        _PATHS[request.endpoint],
        {
            **_body(
                request.endpoint, _EXPOSED_OPENAI_FLAKY, request.nonce, stream=request.stream, outage=request.outage
            ),
            "include_fallback_errors": True,
        },
        headers=_FRESH_CONNECTION,
    )


def _expected_hit(request: _BurstRequest) -> tuple[str, ...]:
    if request.outage:
        return (_EXPOSED_OPENAI_FLAKY, _EXPOSED_OPENAI_BACKUP)
    return (_EXPOSED_OPENAI_FLAKY,)


def _expected_served_id(request: _BurstRequest) -> str:
    deployment: Final = _EXPOSED_OPENAI_BACKUP if request.outage else _EXPOSED_OPENAI_FLAKY
    return _identity(request.endpoint, deployment, request.nonce)


def _check_burst_row(request: _BurstRequest, response: httpx.Response, posted: tuple[_Posted, ...]) -> None:
    own: Final = tuple(item for item in posted if request.nonce in item.raw)
    assert response.status_code == 200, (request, response.text)
    assert _served_id(request.endpoint, response, stream=request.stream) == _expected_served_id(request), request
    assert _hit(own) == _expected_hit(request), (request, own)
    assert _leaks(own) == (), (request, own)
    if request.endpoint == "chat" and not request.stream:
        assert _header(response, "x-litellm-attempted-fallbacks") == ("1" if request.outage else "0"), request
        assert _errors_mention_the_outage(_header(response, "x-litellm-fallback-errors")) is request.outage, request


def test_exposed_burst_with_scripted_outages_lands_every_prompt_once_per_hop_with_a_clean_wire(
    exposed: _Exposed,
) -> None:
    plan: Final = _burst_plan()
    with ThreadPoolExecutor(max_workers=_BURST) as pool:
        responses: Final = tuple(pool.map(partial(_fire, exposed.proxy), plan))
    posted: Final = _all_posted(exposed.wire)
    for request, response in zip(plan, responses, strict=True):
        _check_burst_row(request, response, posted)
