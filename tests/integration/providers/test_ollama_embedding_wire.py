from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Mapping
from typing import Final

import httpx
import pytest
from integration._support.client import JSON_OBJECT, Gateway, list_value, object_value
from integration._support.timing import assert_cut_short
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

_MODEL: Final = "nomic-embed-text"
_DEPLOYMENT_KEY: Final = "synthetic-ollama-deployment-key"
_CALLER_TOKEN: Final = "Bearer synthetic-deployment-header-token"
_DELAY_SECONDS: Final = 5.0


def _vector(index: int) -> list[float]:
    return [0.5, float(index)]


def _embeddings(request: Request) -> Reply:
    body: Final = JSON_OBJECT.validate_json(request.body)
    count: Final = len(body["input"]) if isinstance(body["input"], list) else 1
    return Reply(
        body=json.dumps({"embeddings": [_vector(index) for index in range(count)], "prompt_eval_count": 3}).encode()
    )


def _vectors(response: httpx.Response) -> list[JsonValue]:
    data: Final = list_value(JSON_OBJECT.validate_json(response.content)["data"])
    return [object_value(row)["embedding"] for row in data]


def _only(wire: Wire) -> Request:
    received: Final = wire.drain()
    assert [(request.method, request.target) for request in received] == [("POST", "/api/embed")], [
        (request.method, request.target) for request in received
    ]
    return received[0]


def _values_of(request: Request, name: str) -> list[str]:
    return [value for header, value in request.header_lines if header.lower() == name.lower()]


def _texts(count: int) -> list[str]:
    return [f"ollama proxy embedding {uuid.uuid4().hex}" for _ in range(count)]


@pytest.mark.parametrize(
    ("route", "batched"), (("/v1/embeddings", False), ("/embeddings", True)), ids=("v1-string", "alias-batch")
)
@pytest.mark.parametrize(
    ("credentials", "authorization"),
    (
        ({"api_key": _DEPLOYMENT_KEY}, f"Bearer {_DEPLOYMENT_KEY}"),
        (
            {"api_key": _DEPLOYMENT_KEY, "extra_headers": {"authorization": _CALLER_TOKEN, "X-Embed-Tenant": "t-1"}},
            _CALLER_TOKEN,
        ),
    ),
    ids=("api-key", "lowercase-header-overrides-key"),
)
def test_regression_proxy_ollama_embedding_sends_deployment_credentials(
    gateway: Gateway, route: str, batched: bool, credentials: Mapping[str, JsonValue], authorization: str
) -> None:
    texts: Final = _texts(2 if batched else 1)
    with wire_server(_embeddings) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_MODEL}", api_base=wire.url, **credentials)
        response: Final = gateway.request("POST", route, {"model": model, "input": texts if batched else texts[0]})
        request: Final = _only(wire)
    assert response.status_code == 200, response.text
    assert _vectors(response) == [_vector(i) for i in range(len(texts))]
    assert _values_of(request, "authorization") == [authorization], request.header_lines
    assert JSON_OBJECT.validate_json(request.body) == {"model": _MODEL, "input": texts}, request.body
    expected_tenant: Final = ["t-1"] if "extra_headers" in credentials else []
    assert _values_of(request, "x-embed-tenant") == expected_tenant, request.header_lines


@pytest.mark.parametrize("caller_timeout", ({}, {"timeout": 0}), ids=("omitted", "zero"))
def test_regression_proxy_ollama_deployment_timeout_cuts_a_held_read(
    gateway: Gateway, caller_timeout: Mapping[str, JsonValue]
) -> None:
    release: Final = threading.Event()

    def respond(request: Request) -> Reply:
        release.wait(timeout=_DELAY_SECONDS)
        return _embeddings(request)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_MODEL}", api_base=wire.url, timeout=0.5)
        started: Final = time.monotonic()
        try:
            response: Final = gateway.request(
                "POST", "/v1/embeddings", {"model": model, "input": _texts(1)[0], **caller_timeout}
            )
        finally:
            elapsed: Final = time.monotonic() - started
            release.set()
        _only(wire)
    assert response.status_code == 408, response.text
    assert_cut_short(elapsed, injected_delay=_DELAY_SECONDS)


def test_guard_proxy_keyless_ollama_embedding_sends_no_authorization(gateway: Gateway) -> None:
    texts: Final = _texts(2)
    with wire_server(_embeddings) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_MODEL}", api_base=f"{wire.url}/")
        response: Final = gateway.request("POST", "/v1/embeddings", {"model": model, "input": texts})
        request: Final = _only(wire)
    assert response.status_code == 200, response.text
    assert _vectors(response) == [_vector(0), _vector(1)]
    assert _values_of(request, "authorization") == [], request.header_lines
    assert JSON_OBJECT.validate_json(request.body) == {"model": _MODEL, "input": texts}, request.body


def test_guard_proxy_ollama_upstream_error_reaches_the_caller(gateway: Gateway) -> None:
    marker: Final = f"scripted-bad-request-{uuid.uuid4().hex}"

    def respond(_: Request) -> Reply:
        return Reply(status=400, body=json.dumps({"error": marker}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_MODEL}", api_base=wire.url)
        response: Final = gateway.request("POST", "/v1/embeddings", {"model": model, "input": _texts(1)[0]})
        _only(wire)
    assert response.status_code == 400, response.text
    assert marker in response.text, response.text
