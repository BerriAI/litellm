import asyncio
import contextlib
import json
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import openai
import pytest
import websockets
import yaml
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    held,
    list_value,
    object_value,
    string_value,
)
from integration._support import responses_ws_peer
from integration._support.database import read_rows
from integration._support.openai_wire import answering_model_discovery, chat_reply, responses_reply
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.tls import write_self_signed_cert
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter
from websockets.exceptions import ConnectionClosed

_PROVIDER_KEY: Final = "integration-provider-key"
_EMBEDDING_MODEL: Final = "text-embedding-3-small"
_CHAT_MODEL: Final = "gpt-4o-mini"
_CHAT_BACKEND_PREFIX: Final = "gate-chat-backend-"
_SERVED_TEXT: Final = "served by the chat upstream"
_VECTOR: Final = (0.25, 0.5, 0.75)
_COUNTED_TOKENS: Final = 17
_CLOSE: Final = MappingProxyType({"connection": "close"})
_RELOAD_SECONDS: Final = 5
_EMBED_ONLY: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"mode": "embedding", "supported_endpoints": ["/v1/embeddings"], "enforce_supported_endpoints": True}
)


def _embedding_reply() -> Reply:
    return Reply(
        body=json.dumps(
            {
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": list(_VECTOR)}],
                "model": _EMBEDDING_MODEL,
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            }
        ).encode()
    )


def _not_a_chat_model() -> Reply:
    return Reply(
        status=404,
        body=json.dumps(
            {"error": {"message": "scripted: not a chat model", "type": "invalid_request_error", "code": None}}
        ).encode(),
    )


def _text_completion_reply() -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": f"cmpl-{uuid.uuid4().hex}",
                "object": "text_completion",
                "created": 1,
                "model": _CHAT_MODEL,
                "choices": [{"index": 0, "text": _SERVED_TEXT, "finish_reason": "stop", "logprobs": None}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
            }
        ).encode()
    )


def _serves_chat(model: JsonValue) -> bool:
    return model == _CHAT_MODEL or (isinstance(model, str) and model.startswith(_CHAT_BACKEND_PREFIX))


def _respond(request: Request) -> Reply:
    path: Final = request.target.split("?")[0]
    body: Final = JSON_OBJECT.validate_json(request.body or b"{}")
    streamed: Final = body.get("stream") is True
    if path == "/v1/embeddings":
        return _embedding_reply()
    if path == "/v1/responses/input_tokens":
        return Reply(body=json.dumps({"object": "response.input_tokens", "input_tokens": _COUNTED_TOKENS}).encode())
    if not _serves_chat(body.get("model")):
        return _not_a_chat_model()
    if path == "/v1/chat/completions":
        return chat_reply(f"chatcmpl-{uuid.uuid4().hex}", _CHAT_MODEL, _SERVED_TEXT, stream=streamed)
    if path in ("/v1/responses", "/v1/responses/compact"):
        return responses_reply(f"resp_{uuid.uuid4().hex}", _CHAT_MODEL, _SERVED_TEXT, stream=streamed)
    if path == "/v1/completions":
        return _text_completion_reply()
    return _not_a_chat_model()


class _Upstream:
    def __init__(self, wire: Wire) -> None:
        self.wire = wire
        self.seen: tuple[Request, ...] = ()

    def calls(self, marker: str) -> tuple[Request, ...]:
        self.seen = (*self.seen, *self.wire.drain())
        return tuple(request for request in self.seen if marker.encode() in request.body or marker in request.target)

    def targets(self, marker: str) -> tuple[str, ...]:
        return tuple(request.target for request in self.calls(marker))


@dataclass(frozen=True, slots=True)
class _Deployment:
    litellm_model: str
    declarations: tuple[Mapping[str, JsonValue], ...]


def _single(declaration: Mapping[str, JsonValue], litellm_model: str = f"openai/{_CHAT_MODEL}") -> _Deployment:
    return _Deployment(litellm_model, (declaration,))


_SHARED: Final[Mapping[str, _Deployment]] = MappingProxyType(
    {
        "embedding_only": _single(_EMBED_ONLY, f"openai/{_EMBEDDING_MODEL}"),
        "embedding_only_second": _single(_EMBED_ONLY, f"openai/{_EMBEDDING_MODEL}"),
        "image_generation": _single(
            {"enforce_supported_endpoints": True, "supported_endpoints": ["/v1/images/generations"]}
        ),
        "chat_target": _single({"mode": "chat"}),
        "chat_rate_limited": _single({"mode": "chat"}),
        "mode_not_a_string": _single(
            {"mode": 7, "supported_endpoints": ["/v1/embeddings"]}, "openai/gate-mode-not-a-string"
        ),
    }
)


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    upstream: _Upstream
    names: Mapping[str, str]
    api_base: str


def _create(gateway: Gateway, name: str, litellm_params: Mapping[str, JsonValue], info: Mapping[str, JsonValue]) -> str:
    created: Final = gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {"api_key": _PROVIDER_KEY, **litellm_params},
            "model_info": dict(info),
        },
    )
    return string_value(object_value(created["model_info"])["id"])


def _deploy(scenario: Scenario, api_base: str, label: str, deployment: _Deployment) -> tuple[str, tuple[str, ...]]:
    name: Final = f"gate-{label}-{uuid.uuid4().hex[:12]}"
    identities: Final = tuple(
        _create(scenario.gateway, name, {"model": deployment.litellm_model, "api_base": api_base}, declaration)
        for declaration in deployment.declarations
    )
    tuple(scenario.cleanups.callback(scenario.delete_model, identity) for identity in identities)
    return name, identities


def _listed_ids(gateway: Gateway) -> frozenset[str]:
    listed: Final = gateway.request("GET", "/model/info", headers=_CLOSE)
    assert listed.status_code == 200, listed.text
    return frozenset(
        string_value(object_value(object_value(entry)["model_info"])["id"])
        for entry in list_value(JSON_OBJECT.validate_json(listed.content)["data"])
    )


def _served_by_every_worker(gateway: Gateway, identities: frozenset[str]) -> None:
    held(
        lambda: identities <= _listed_ids(gateway),
        lambda served: served,
        holding=_RELOAD_SECONDS * 2,
        seconds=_RELOAD_SECONDS * 12,
    )


@pytest.fixture(scope="module")
def rig() -> Iterator[_Rig]:
    with (
        wire_server(answering_model_discovery(_respond)) as wire,
        gateway_from_environment() as gateway,
        gateway.scenario() as scenario,
    ):
        api_base: Final = wire.url + "/v1"
        deployed: Final = {label: _deploy(scenario, api_base, label, spec) for label, spec in _SHARED.items()}
        _served_by_every_worker(
            gateway, frozenset(identity for _name, identities in deployed.values() for identity in identities)
        )
        yield _Rig(
            gateway,
            _Upstream(wire),
            MappingProxyType({label: name for label, (name, _ids) in deployed.items()}),
            api_base,
        )


def _marker() -> str:
    return f"gate-{uuid.uuid4().hex}"


def _openai_error(endpoint: str, model: str, supported: str) -> dict[str, JsonValue]:
    return {
        "error": {
            "message": f"{endpoint} is not supported by model={model}. Supported endpoints: {supported}",
            "type": "invalid_request_error",
            "param": "model",
            "code": "400",
        }
    }


def _anthropic_error(model: str, supported: str) -> dict[str, JsonValue]:
    return {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": f"/v1/messages is not supported by model={model}. Supported endpoints: {supported}",
        },
    }


def _chat_body(model: str, marker: str, *, stream: bool = False) -> dict[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": marker}], **({"stream": True} if stream else {})}


def _failure_row(call_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows('SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (call_id,)),
        lambda found: len(found) >= 1,
        seconds=70,
    )
    assert len(rows) == 1, rows
    return rows[0]


def _assert_error(response: httpx.Response, expected: dict[str, JsonValue]) -> None:
    assert response.status_code == 400, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    error: Final = object_value(body["error"])
    message: Final = string_value(error["message"]).split("\n\n", 1)[0]
    assert {**body, "error": {**error, "message": message}} == expected, response.text


def _assert_rejected_before(
    upstream: _Upstream, response: httpx.Response, marker: str, expected: dict[str, JsonValue]
) -> None:
    _assert_error(response, expected)
    assert upstream.calls(marker) == (), upstream.targets(marker)
    assert _failure_row(response.headers["x-litellm-call-id"])["status"] == "failure"


def _assert_rejected(rig: _Rig, response: httpx.Response, marker: str, expected: dict[str, JsonValue]) -> None:
    _assert_rejected_before(rig.upstream, response, marker, expected)


@dataclass(frozen=True, slots=True)
class _Route:
    path: str
    endpoint: str
    anthropic_shape: bool
    stream: bool
    model_in_path: bool
    shape: str


_ROUTES: Final[Mapping[str, _Route]] = MappingProxyType(
    {
        "interactions": _Route("/v1beta/interactions", "/v1beta/interactions", False, False, False, "interactions"),
        "chat_v1": _Route("/v1/chat/completions", "/v1/chat/completions", False, False, False, "chat"),
        "chat_v1_stream": _Route("/v1/chat/completions", "/v1/chat/completions", False, True, False, "chat"),
        "chat_bare": _Route("/chat/completions", "/v1/chat/completions", False, False, False, "chat"),
        "chat_engines": _Route("/engines/{model}/chat/completions", "/v1/chat/completions", False, False, True, "chat"),
        "chat_azure_deployments": _Route(
            "/openai/deployments/{model}/chat/completions", "/v1/chat/completions", False, False, True, "chat"
        ),
        "messages": _Route("/v1/messages", "/v1/messages", True, False, False, "messages"),
        "messages_stream": _Route("/v1/messages", "/v1/messages", True, True, False, "messages"),
        "responses_v1": _Route("/v1/responses", "/v1/responses", False, False, False, "responses"),
        "responses_v1_stream": _Route("/v1/responses", "/v1/responses", False, True, False, "responses"),
        "responses_bare": _Route("/responses", "/v1/responses", False, False, False, "responses"),
        "cursor_responses_body": _Route("/cursor/chat/completions", "/v1/responses", False, False, False, "cursor"),
        "cursor_chat_body": _Route("/cursor/chat/completions", "/v1/chat/completions", False, False, False, "chat"),
    }
)


def _route_body(route: _Route, model: str, marker: str) -> dict[str, JsonValue]:
    shaped: Final[Mapping[str, dict[str, JsonValue]]] = {
        "chat": {"messages": [{"role": "user", "content": marker}]},
        "messages": {"max_tokens": 16, "messages": [{"role": "user", "content": marker}]},
        "responses": {"input": marker},
        "interactions": {"input": marker},
        "cursor": {"input": [{"role": "user", "content": marker}]},
    }
    return {
        **({} if route.model_in_path else {"model": model}),
        **shaped[route.shape],
        **({"stream": True} if route.stream else {}),
    }


def _send(gateway: Gateway, route: _Route, model: str, marker: str) -> httpx.Response:
    path: Final = route.path.format(model=model)
    body: Final = _route_body(route, model, marker)
    if not route.stream:
        return gateway.request("POST", path, body)
    with gateway.client.stream("POST", path, json=body, headers={"Authorization": f"Bearer {gateway.key}"}) as streamed:
        streamed.read()
        return streamed


@pytest.mark.parametrize("route", tuple(_ROUTES))
def test_chat_shaped_route_rejects_an_embedding_only_model_before_the_upstream(rig: _Rig, route: str) -> None:
    spec: Final = _ROUTES[route]
    model: Final = rig.names["embedding_only"]
    marker: Final = _marker()
    response: Final = _send(rig.gateway, spec, model, marker)
    expected: Final = (
        _anthropic_error(model, "/v1/embeddings")
        if spec.anthropic_shape
        else _openai_error(spec.endpoint, model, "/v1/embeddings")
    )
    _assert_rejected(rig, response, marker, expected)


def _sdk_embedding(rig: _Rig, model: str, marker: str) -> tuple[tuple[float, ...], str]:
    async def call() -> tuple[tuple[float, ...], str]:
        async with openai.AsyncOpenAI(
            base_url=f"{rig.gateway.client.base_url}/v1", api_key=rig.gateway.key, max_retries=0
        ) as sdk:
            raw: Final = await sdk.embeddings.with_raw_response.create(
                model=model, input=marker, encoding_format="float"
            )
            parsed: Final = raw.parse()
            return tuple(parsed.data[0].embedding), raw.headers["x-litellm-call-id"]

    return asyncio.run(call())


def _httpx_embedding(rig: _Rig, model: str, marker: str) -> tuple[tuple[float, ...], str]:
    response: Final = rig.gateway.request(
        "POST", "/v1/embeddings", {"model": model, "input": marker, "encoding_format": "float"}
    )
    assert response.status_code == 200, response.text
    data: Final = list_value(JSON_OBJECT.validate_json(response.content)["data"])
    vector: Final = list_value(object_value(data[0])["embedding"])
    return tuple(float(str(value)) for value in vector), response.headers["x-litellm-call-id"]


@pytest.mark.parametrize("client", ("httpx_sync", "openai_async_sdk"))
def test_embeddings_still_reach_the_embedding_only_model(rig: _Rig, client: str) -> None:
    model: Final = rig.names["embedding_only"]
    marker: Final = _marker()
    vector, call_id = (
        _httpx_embedding(rig, model, marker) if client == "httpx_sync" else _sdk_embedding(rig, model, marker)
    )
    assert vector == _VECTOR
    assert rig.upstream.targets(marker) == ("/v1/embeddings",)
    row: Final = eventually(
        lambda: read_rows(
            'SELECT status, call_type FROM "LiteLLM_SpendLogs" WHERE request_id = %s OR litellm_call_id = %s',
            (call_id, call_id),
        ),
        lambda found: len(found) >= 1,
        seconds=70,
    )
    assert [(entry["status"], entry["call_type"]) for entry in row] == [("success", "aembedding")], row


def _assert_served(rig: _Rig, response: httpx.Response, marker: str, target: str) -> None:
    assert response.status_code == 200, response.text
    assert _SERVED_TEXT in response.text, response.text
    assert rig.upstream.targets(marker) == (target,)


def test_key_alias_to_an_embedding_only_model_is_rejected_under_the_resolved_name(rig: _Rig) -> None:
    target: Final = rig.names["embedding_only"]
    alias: Final = f"gate-alias-{uuid.uuid4().hex[:12]}"
    marker: Final = _marker()
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(aliases={alias: target})
        response: Final = rig.gateway.request("POST", "/v1/chat/completions", _chat_body(alias, marker), key=key)
        _assert_rejected(rig, response, marker, _openai_error("/v1/chat/completions", target, "/v1/embeddings"))


async def _websocket_first_frame_and_close(
    gateway: Gateway, key: str, model: str, marker: str
) -> tuple[dict[str, JsonValue], int | None, str | None]:
    url: Final = str(gateway.client.base_url).rstrip("/").replace("http://", "ws://", 1) + "/v1/responses"
    async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {key}"}, open_timeout=10) as ws:
        await ws.send(json.dumps({"type": "response.create", "model": model, "input": marker}))
        first: Final = JSON_OBJECT.validate_json(await asyncio.wait_for(ws.recv(), 30))
        with contextlib.suppress(ConnectionClosed):
            await asyncio.wait_for(ws.recv(), 30)
        return first, ws.close_code, ws.close_reason


def _websocket_rejection(model: str) -> tuple[dict[str, JsonValue], int | None, str | None]:
    return (
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": f"/v1/responses is not supported by model={model}. Supported endpoints: /v1/embeddings",
            },
        },
        1008,
        "Pre-call error",
    )


@pytest.mark.parametrize("aliased", [False, True])
def test_responses_websocket_closes_on_an_embedding_only_model_before_the_upstream(rig: _Rig, aliased: bool) -> None:
    target: Final = rig.names["embedding_only"]
    alias: Final = f"gate-ws-alias-{uuid.uuid4().hex[:12]}"
    marker: Final = _marker()
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(aliases={alias: target})
        closed: Final = asyncio.run(
            _websocket_first_frame_and_close(rig.gateway, key, alias if aliased else target, marker)
        )
    assert closed == _websocket_rejection(target)
    assert rig.upstream.calls(marker) == ()


_RESERVING_KEYS: Final[Mapping[str, Mapping[str, JsonValue]]] = MappingProxyType(
    {
        "max_parallel_requests": {"max_parallel_requests": 1},
        "tpm_limit": {"tpm_limit": 100, "default_estimated_output_tokens": 80},
    }
)


@pytest.mark.parametrize("limit", tuple(_RESERVING_KEYS))
def test_responses_websocket_rejections_leave_the_key_reservation_free(rig: _Rig, limit: str) -> None:
    target: Final = rig.names["embedding_only"]
    markers: Final = tuple(_marker() for _ in range(3))
    served_marker: Final = _marker()
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(**_RESERVING_KEYS[limit])
        closed: Final = tuple(
            asyncio.run(_websocket_first_frame_and_close(rig.gateway, key, target, marker)) for marker in markers
        )
        assert closed == (_websocket_rejection(target),) * 3
        served: Final = rig.gateway.request(
            "POST", "/v1/chat/completions", _chat_body(rig.names["chat_target"], served_marker), key=key
        )
        _assert_served(rig, served, served_marker, "/v1/chat/completions")
    assert [rig.upstream.calls(marker) for marker in markers] == [()] * 3


def test_team_scoped_embedding_only_model_is_rejected_under_its_public_name(rig: _Rig) -> None:
    public: Final = f"gate-team-{uuid.uuid4().hex[:12]}"
    marker: Final = _marker()
    with rig.gateway.scenario() as scenario:
        team: Final = scenario.team()
        identity: Final = _create(
            rig.gateway,
            public,
            {"model": f"openai/{_EMBEDDING_MODEL}", "api_base": rig.api_base},
            {**_EMBED_ONLY, "team_id": team},
        )
        scenario.cleanups.callback(scenario.delete_model, identity)
        key: Final = scenario.key(team_id=team)
        _served_by_every_worker(rig.gateway, frozenset({identity}))
        embedded: Final = rig.gateway.request(
            "POST", "/v1/embeddings", {"model": public, "input": marker, "encoding_format": "float"}, key=key
        )
        assert embedded.status_code == 200, embedded.text
        assert rig.upstream.targets(marker) == ("/v1/embeddings",)
        rejected_marker: Final = _marker()
        response: Final = rig.gateway.request(
            "POST", "/v1/chat/completions", _chat_body(public, rejected_marker), key=key
        )
        _assert_rejected(
            rig, response, rejected_marker, _openai_error("/v1/chat/completions", public, "/v1/embeddings")
        )


def test_wildcard_embedding_only_deployment_rejects_chat_for_any_matched_model(rig: _Rig) -> None:
    prefix: Final = f"gate-wild-{uuid.uuid4().hex[:12]}"
    requested: Final = f"{prefix}/{_EMBEDDING_MODEL}"
    marker: Final = _marker()
    with rig.gateway.scenario() as scenario:
        identity: Final = _create(
            rig.gateway, f"{prefix}/*", {"model": "openai/*", "api_base": rig.api_base}, _EMBED_ONLY
        )
        scenario.cleanups.callback(scenario.delete_model, identity)
        _served_by_every_worker(rig.gateway, frozenset({identity}))
        response: Final = rig.gateway.request("POST", "/v1/chat/completions", _chat_body(requested, marker))
        _assert_rejected(rig, response, marker, _openai_error("/v1/chat/completions", requested, "/v1/embeddings"))


def test_key_without_access_gets_the_access_error_and_learns_nothing_about_endpoints(rig: _Rig) -> None:
    marker: Final = _marker()
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(models=[rig.names["chat_target"]])
        response: Final = rig.gateway.request(
            "POST", "/v1/chat/completions", _chat_body(rig.names["embedding_only"], marker), key=key
        )
    assert response.status_code == 403, response.text
    assert object_value(JSON_OBJECT.validate_json(response.content)["error"])["type"] == "key_model_access_denied", (
        response.text
    )
    assert "Supported endpoints" not in response.text, response.text
    assert rig.upstream.calls(marker) == ()


def test_unknown_model_keeps_the_invalid_model_error(rig: _Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.gateway.request(
        "POST", "/v1/chat/completions", _chat_body(f"gate-missing-{uuid.uuid4().hex}", marker)
    )
    assert response.status_code == 400, response.text
    assert "Invalid model name" in response.text and "Supported endpoints" not in response.text, response.text
    assert rig.upstream.calls(marker) == ()


def test_request_fallbacks_answer_for_an_embedding_only_primary(rig: _Rig) -> None:
    model: Final = rig.names["embedding_only"]
    marker: Final = _marker()
    response: Final = rig.gateway.request(
        "POST", "/v1/chat/completions", {**_chat_body(model, marker), "fallbacks": [rig.names["chat_target"]]}
    )
    assert response.status_code == 200, response.text
    assert _SERVED_TEXT in response.text, response.text
    assert rig.upstream.targets(marker) == ("/v1/chat/completions",)


def test_an_empty_request_fallback_list_keeps_the_rejection(rig: _Rig) -> None:
    model: Final = rig.names["embedding_only"]
    marker: Final = _marker()
    response: Final = rig.gateway.request(
        "POST", "/v1/chat/completions", {**_chat_body(model, marker), "fallbacks": []}
    )
    _assert_rejected(rig, response, marker, _openai_error("/v1/chat/completions", model, "/v1/embeddings"))


def test_fallbacks_that_cannot_answer_chat_keep_the_rejection(rig: _Rig) -> None:
    model: Final = rig.names["embedding_only"]
    marker: Final = _marker()
    fallbacks: Final = [rig.names["image_generation"], model]
    response: Final = rig.gateway.request(
        "POST", "/v1/chat/completions", {**_chat_body(model, marker), "fallbacks": fallbacks}
    )
    _assert_rejected(rig, response, marker, _openai_error("/v1/chat/completions", model, "/v1/embeddings"))


def test_request_fallbacks_do_not_rescue_an_unknown_primary(rig: _Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.gateway.request(
        "POST",
        "/v1/chat/completions",
        {**_chat_body(f"gate-missing-{uuid.uuid4().hex}", marker), "fallbacks": [rig.names["chat_target"]]},
    )
    assert response.status_code == 400, response.text
    assert "Invalid model name" in response.text, response.text
    assert rig.upstream.calls(marker) == ()


def _chat(rig: _Rig, model: str, marker: str, *, key: str | None = None, **extra: JsonValue) -> httpx.Response:
    return rig.gateway.request("POST", "/v1/chat/completions", {**_chat_body(model, marker), **extra}, key=key)


def _assert_rejected_chat(rig: _Rig, response: httpx.Response, marker: str, model: str) -> None:
    _assert_rejected(rig, response, marker, _openai_error("/v1/chat/completions", model, "/v1/embeddings"))


def test_a_fallback_chain_that_reaches_chat_two_hops_away_answers(rig: _Rig) -> None:
    first, second, chat = rig.names["embedding_only"], rig.names["embedding_only_second"], rig.names["chat_target"]
    marker: Final = _marker()
    response: Final = _chat(rig, first, marker, fallbacks=[{first: [second]}, {second: [chat]}])
    assert response.status_code == 200, response.text
    assert _SERVED_TEXT in response.text, response.text
    assert rig.upstream.targets(marker) == ("/v1/chat/completions",)


def test_a_fallback_chain_that_loops_between_embedding_only_groups_keeps_the_rejection(rig: _Rig) -> None:
    first, second = rig.names["embedding_only"], rig.names["embedding_only_second"]
    marker: Final = _marker()
    response: Final = _chat(rig, first, marker, fallbacks=[{first: [second]}, {second: [first]}])
    _assert_rejected_chat(rig, response, marker, first)


def test_a_specific_chain_that_loops_beats_a_catch_all_that_could_answer(rig: _Rig) -> None:
    first, second = rig.names["embedding_only"], rig.names["embedding_only_second"]
    marker: Final = _marker()
    response: Final = _chat(
        rig, first, marker, fallbacks=[{"*": [rig.names["chat_target"]]}, {first: [second]}, {second: [first]}]
    )
    _assert_rejected_chat(rig, response, marker, first)


def test_disable_fallbacks_keeps_the_rejection_over_request_fallbacks(rig: _Rig) -> None:
    model: Final = rig.names["embedding_only"]
    marker: Final = _marker()
    response: Final = _chat(rig, model, marker, fallbacks=[rig.names["chat_target"]], disable_fallbacks=True)
    _assert_rejected_chat(rig, response, marker, model)


def _assert_fallback_answered(
    upstream: _Upstream, response: httpx.Response, marker: str, target: str = "/v1/chat/completions"
) -> None:
    assert response.status_code == 200, response.text
    assert _SERVED_TEXT in response.text, response.text
    assert upstream.targets(marker) == (target,), response.text


def _through_a_group_with_no_deployments(rig: _Rig) -> Mapping[str, JsonValue]:
    ghost: Final = f"gate-ghost-{uuid.uuid4().hex}"
    return {"fallbacks": [{rig.names["embedding_only"]: [ghost]}, {ghost: [rig.names["chat_target"]]}]}


def test_request_fallbacks_naming_a_chat_backend_name_answer_for_an_embedding_only_primary(rig: _Rig) -> None:
    model: Final = rig.names["embedding_only"]
    backend: Final = f"openai/{_CHAT_BACKEND_PREFIX}{uuid.uuid4().hex[:12]}"
    marker: Final = _marker()
    with rig.gateway.scenario() as scenario:
        identity: Final = _create(
            rig.gateway, f"gate-backend-{uuid.uuid4().hex[:12]}", {"model": backend, "api_base": rig.api_base}, {}
        )
        scenario.cleanups.callback(scenario.delete_model, identity)
        _served_by_every_worker(rig.gateway, frozenset({identity}))
        response: Final = _chat(rig, model, marker, fallbacks=[{model: [backend]}])
    _assert_fallback_answered(rig.upstream, response, marker)


@pytest.mark.parametrize("level", ("key", "team"))
def test_router_settings_fallbacks_through_a_group_with_no_deployments_answer(rig: _Rig, level: str) -> None:
    marker: Final = _marker()
    with rig.gateway.scenario() as scenario:
        settings: Final = {"router_settings": dict(_through_a_group_with_no_deployments(rig))}
        key: Final = scenario.key(**settings) if level == "key" else scenario.key(team_id=scenario.team(**settings))
        response: Final = _chat(rig, rig.names["embedding_only"], marker, key=key)
    _assert_fallback_answered(rig.upstream, response, marker)


def test_team_router_settings_fallbacks_answer_for_an_embedding_only_primary(rig: _Rig) -> None:
    model, chat = rig.names["embedding_only"], rig.names["chat_target"]
    marker: Final = _marker()
    with rig.gateway.scenario() as scenario:
        team: Final = scenario.team(router_settings={"fallbacks": [{model: [chat]}]})
        response: Final = _chat(rig, model, marker, key=scenario.key(team_id=team))
    assert response.status_code == 200, response.text
    assert _SERVED_TEXT in response.text, response.text
    assert rig.upstream.targets(marker) == ("/v1/chat/completions",)


def test_key_router_settings_fallbacks_answer_for_an_embedding_only_primary(rig: _Rig) -> None:
    model, chat = rig.names["embedding_only"], rig.names["chat_target"]
    marker: Final = _marker()
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(router_settings={"fallbacks": [{model: [chat]}]})
        response: Final = _chat(rig, model, marker, key=key)
    assert response.status_code == 200, response.text
    assert _SERVED_TEXT in response.text, response.text
    assert rig.upstream.targets(marker) == ("/v1/chat/completions",)


def test_key_router_settings_that_cannot_answer_override_team_fallbacks_that_could(rig: _Rig) -> None:
    model: Final = rig.names["embedding_only"]
    marker: Final = _marker()
    with rig.gateway.scenario() as scenario:
        team: Final = scenario.team(router_settings={"fallbacks": [{model: [rig.names["chat_target"]]}]})
        key: Final = scenario.key(
            team_id=team, router_settings={"fallbacks": [{model: [rig.names["embedding_only_second"]]}]}
        )
        response: Final = _chat(rig, model, marker, key=key)
        _assert_rejected_chat(rig, response, marker, model)


def test_an_empty_request_fallback_list_overrides_key_router_settings_fallbacks(rig: _Rig) -> None:
    model: Final = rig.names["embedding_only"]
    marker: Final = _marker()
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(router_settings={"fallbacks": [{model: [rig.names["chat_target"]]}]})
        response: Final = _chat(rig, model, marker, key=key, fallbacks=[])
        _assert_rejected_chat(rig, response, marker, model)


def test_local_rate_limit_fallback_skips_an_embedding_only_hop(rig: _Rig) -> None:
    limited, chat, model = rig.names["chat_rate_limited"], rig.names["chat_target"], rig.names["embedding_only"]
    first, second = _marker(), _marker()
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(
            model_rpm_limit={limited: 1}, router_settings={"fallbacks": [{limited: [model, chat]}]}
        )
        served_first: Final = _chat(rig, limited, first, key=key)
        served_second: Final = _chat(rig, limited, second, key=key)
    assert (served_first.status_code, served_first.headers.get("x-litellm-model-group")) == (200, limited), (
        served_first.text
    )
    assert (served_second.status_code, served_second.headers.get("x-litellm-model-group")) == (200, chat), (
        served_second.text
    )
    assert (rig.upstream.targets(first), rig.upstream.targets(second)) == (
        ("/v1/chat/completions",),
        ("/v1/chat/completions",),
    )


_COMPACT_PATHS: Final = ("/v1/responses/compact", "/responses/compact", "/openai/v1/responses/compact")


def _compact_body(model: str, marker: str) -> dict[str, JsonValue]:
    return {"model": model, "input": [{"role": "user", "content": marker}]}


@pytest.mark.parametrize("path", _COMPACT_PATHS)
def test_responses_compact_rejects_an_embedding_only_model_before_the_upstream(rig: _Rig, path: str) -> None:
    model: Final = rig.names["embedding_only"]
    marker: Final = _marker()
    response: Final = rig.gateway.request("POST", path, _compact_body(model, marker))
    _assert_rejected(rig, response, marker, _openai_error("/v1/responses/compact", model, "/v1/embeddings"))


def test_responses_compact_keeps_reaching_the_upstream_for_a_chat_model(rig: _Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.gateway.request(
        "POST", "/v1/responses/compact", _compact_body(rig.names["chat_target"], marker)
    )
    _assert_served(rig, response, marker, "/v1/responses/compact")


def test_count_tokens_keeps_counting_for_an_embedding_only_model(rig: _Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.gateway.request(
        "POST",
        "/v1/messages/count_tokens",
        {"model": rig.names["embedding_only"], "messages": [{"role": "user", "content": marker}]},
    )
    assert response.status_code == 200, response.text
    assert JSON_OBJECT.validate_json(response.content)["input_tokens"] == _COUNTED_TOKENS, response.text
    assert rig.upstream.targets(marker) == ("/v1/responses/input_tokens",)


def test_text_completions_reject_an_embedding_only_model_before_the_upstream(rig: _Rig) -> None:
    model: Final = rig.names["embedding_only"]
    marker: Final = _marker()
    response: Final = rig.gateway.request("POST", "/v1/completions", {"model": model, "prompt": marker})
    _assert_rejected(rig, response, marker, _openai_error("/v1/completions", model, "/v1/embeddings"))


@pytest.mark.parametrize("method", ("generateContent", "streamGenerateContent"))
def test_gemini_routes_reject_an_embedding_only_model_before_the_upstream(rig: _Rig, method: str) -> None:
    model: Final = rig.names["embedding_only"]
    marker: Final = _marker()
    response: Final = rig.gateway.request(
        "POST", f"/v1beta/models/{model}:{method}", {"contents": [{"role": "user", "parts": [{"text": marker}]}]}
    )
    assert response.status_code == 400, response.text
    assert f"{method} is not supported by model={model}. Supported endpoints: /v1/embeddings" in response.text
    assert rig.upstream.calls(marker) == (), rig.upstream.targets(marker)


def test_repeated_rejections_log_one_failure_row_per_call(rig: _Rig) -> None:
    model: Final = rig.names["embedding_only"]
    marker: Final = _marker()
    responses: Final = tuple(
        rig.gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker)) for _ in range(3)
    )
    expected: Final = _openai_error("/v1/chat/completions", model, "/v1/embeddings")
    for response in responses:
        _assert_error(response, expected)
    call_ids: Final = tuple(item.headers["x-litellm-call-id"] for item in responses)
    assert len(set(call_ids)) == 3, call_ids
    assert [_failure_row(call_id)["status"] for call_id in call_ids] == ["failure"] * 3
    assert rig.upstream.calls(marker) == ()


def _closed_chat(rig: _Rig, model: str, marker: str) -> httpx.Response:
    return rig.gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker), headers=_CLOSE)


def test_live_declaration_update_turns_the_gate_on_and_off_on_every_worker(rig: _Rig) -> None:
    chat_capable: Final[dict[str, JsonValue]] = {
        "mode": "chat",
        "supported_endpoints": ["/v1/chat/completions", "/v1/embeddings"],
    }
    marker: Final = _marker()
    with rig.gateway.scenario() as scenario:
        name, identities = _deploy(scenario, rig.api_base, "flip", _single(chat_capable))
        _served_by_every_worker(rig.gateway, frozenset(identities))
        served: Final = _closed_chat(rig, name, marker)
        _assert_served(rig, served, marker, "/v1/chat/completions")
        patched: Final = rig.gateway.request(
            "PATCH", f"/model/{identities[0]}/update", {"model_info": dict(_EMBED_ONLY)}
        )
        assert patched.status_code == 200, patched.text
        rejected: Final = held(
            lambda: _closed_chat(rig, name, _marker()),
            lambda response: response.status_code == 400,
            holding=_RELOAD_SECONDS * 2,
            seconds=_RELOAD_SECONDS * 12,
        )
        _assert_error(rejected, _openai_error("/v1/chat/completions", name, "/v1/embeddings"))
        restored: Final = rig.gateway.request("PATCH", f"/model/{identities[0]}/update", {"model_info": chat_capable})
        assert restored.status_code == 200, restored.text
        held(
            lambda: _closed_chat(rig, name, _marker()),
            lambda response: response.status_code == 200,
            holding=_RELOAD_SECONDS * 2,
            seconds=_RELOAD_SECONDS * 12,
        )


_TESTS_ROOT: Final = Path(__file__).resolve().parents[2]
_CONFIG: Final = TypeAdapter(dict[str, JsonValue])


def _owned_embedding(owned: Gateway, model: str, marker: str) -> httpx.Response:
    return owned.request("POST", "/v1/embeddings", {"model": model, "input": marker, "encoding_format": "float"})


@dataclass(frozen=True, slots=True)
class _Routed:
    gateway: Gateway
    upstream: _Upstream
    names: Mapping[str, str]


def _routed_entry(name: str, api_base: str, litellm_model: str, info: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {"model": litellm_model, "api_base": api_base, "api_key": _PROVIDER_KEY},
        "model_info": dict(info),
    }


def _routed_config(directory: Path, api_base: str, names: Mapping[str, str]) -> Path:
    shared: Final = _CONFIG.validate_python(yaml.safe_load((_TESTS_ROOT / "integration/proxy_config.yaml").read_text()))
    config: Final = {
        **shared,
        "model_list": [
            _routed_entry(names["embed"], api_base, f"openai/{_EMBEDDING_MODEL}", _EMBED_ONLY),
            _routed_entry(names["chat"], api_base, f"openai/{_CHAT_MODEL}", {"mode": "chat"}),
        ],
        "router_settings": {
            **object_value(shared["router_settings"]),
            "num_retries": 0,
            "default_fallbacks": [names["chat"]],
            "model_group_alias": {names["alias_embed"]: names["embed"], names["alias_chat"]: names["chat"]},
        },
    }
    path: Final = directory / "gate-routed-config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


@pytest.fixture(scope="module")
def routed(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Routed]:
    directory: Final = tmp_path_factory.mktemp("gate-routed")
    suffix: Final = uuid.uuid4().hex[:12]
    names: Final = MappingProxyType(
        {label: f"gate-routed-{label}-{suffix}" for label in ("embed", "chat", "alias_embed", "alias_chat")}
    )
    with (
        wire_server(answering_model_discovery(_respond)) as wire,
        gateway_from_environment() as rig_gateway,
        owned_proxy_process(
            rig_gateway, directory, {}, config=_routed_config(directory, wire.url + "/v1", names), workers=2
        ) as owned,
    ):
        eventually(
            lambda: owned.gateway.request("POST", "/v1/chat/completions", _chat_body(names["chat"], "probe")),
            lambda response: response.status_code == 200,
            seconds=30,
        )
        yield _Routed(owned.gateway, _Upstream(wire), names)


_ROUTED_TIMEOUT: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
_DEFAULT_FALLBACK_ROUTES: Final[Mapping[str, str]] = MappingProxyType(
    {"chat_v1": "/v1/chat/completions", "messages": "/v1/responses", "responses_v1": "/v1/responses"}
)


@_ROUTED_TIMEOUT
@pytest.mark.parametrize("route", tuple(_DEFAULT_FALLBACK_ROUTES))
def test_router_default_fallbacks_answer_for_an_embedding_only_group(routed: _Routed, route: str) -> None:
    marker: Final = _marker()
    response: Final = _send(routed.gateway, _ROUTES[route], routed.names["embed"], marker)
    assert response.status_code == 200, response.text
    assert _SERVED_TEXT in response.text, response.text
    assert routed.upstream.targets(marker) == (_DEFAULT_FALLBACK_ROUTES[route],)


_OPT_OUTS: Final[Mapping[str, Mapping[str, JsonValue]]] = MappingProxyType(
    {
        "empty_fallbacks": {"fallbacks": []},
        "null_fallbacks": {"fallbacks": None},
        "disable_fallbacks": {"disable_fallbacks": True},
    }
)


@_ROUTED_TIMEOUT
@pytest.mark.parametrize("opt_out", tuple(_OPT_OUTS))
def test_request_opt_outs_keep_the_rejection_over_router_default_fallbacks(routed: _Routed, opt_out: str) -> None:
    model: Final = routed.names["embed"]
    marker: Final = _marker()
    response: Final = routed.gateway.request(
        "POST", "/v1/chat/completions", {**_chat_body(model, marker), **_OPT_OUTS[opt_out]}
    )
    _assert_rejected_before(
        routed.upstream, response, marker, _openai_error("/v1/chat/completions", model, "/v1/embeddings")
    )


@_ROUTED_TIMEOUT
def test_model_group_alias_to_an_embedding_only_group_falls_back_to_the_default(routed: _Routed) -> None:
    marker: Final = _marker()
    response: Final = routed.gateway.request(
        "POST", "/v1/chat/completions", _chat_body(routed.names["alias_embed"], marker)
    )
    assert response.status_code == 200, response.text
    assert _SERVED_TEXT in response.text, response.text
    assert routed.upstream.targets(marker) == ("/v1/chat/completions",)


@_ROUTED_TIMEOUT
def test_model_group_alias_to_an_embedding_only_group_is_rejected_without_fallbacks(routed: _Routed) -> None:
    alias: Final = routed.names["alias_embed"]
    marker: Final = _marker()
    response: Final = routed.gateway.request(
        "POST", "/v1/chat/completions", {**_chat_body(alias, marker), "disable_fallbacks": True}
    )
    _assert_rejected_before(
        routed.upstream, response, marker, _openai_error("/v1/chat/completions", alias, "/v1/embeddings")
    )


@_ROUTED_TIMEOUT
def test_model_group_alias_to_a_chat_group_keeps_reaching_the_upstream(routed: _Routed) -> None:
    marker: Final = _marker()
    response: Final = routed.gateway.request(
        "POST", "/v1/chat/completions", _chat_body(routed.names["alias_chat"], marker)
    )
    assert response.status_code == 200, response.text
    assert _SERVED_TEXT in response.text, response.text
    assert routed.upstream.targets(marker) == ("/v1/chat/completions",)


@_ROUTED_TIMEOUT
def test_model_group_alias_to_an_embedding_only_group_keeps_serving_embeddings(routed: _Routed) -> None:
    marker: Final = _marker()
    response: Final = _owned_embedding(routed.gateway, routed.names["alias_embed"], marker)
    assert response.status_code == 200, response.text
    assert list_value(
        object_value(list_value(JSON_OBJECT.validate_json(response.content)["data"])[0])["embedding"]
    ) == list(_VECTOR), response.text
    assert routed.upstream.targets(marker) == ("/v1/embeddings",)


_REWRITER: Final = """from litellm.integrations.custom_guardrail import CustomGuardrail


class WebSocketModelRewriter(CustomGuardrail):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if call_type == "_aresponses_websocket" and data.get("model") == "__SOURCE__":
            data["model"] = "__TARGET__"
        return data
"""


@dataclass(frozen=True, slots=True)
class _Rewritten:
    gateway: Gateway
    upstream: _Upstream
    source: str
    target: str


def _rewritten_config(directory: Path, api_base: str, source: str, target: str, module: str) -> Path:
    shared: Final = _CONFIG.validate_python(yaml.safe_load((_TESTS_ROOT / "integration/proxy_config.yaml").read_text()))
    config: Final = {
        **shared,
        "model_list": [
            _routed_entry(source, api_base, f"openai/{_CHAT_MODEL}", {"mode": "chat"}),
            _routed_entry(target, api_base, f"openai/{_EMBEDDING_MODEL}", _EMBED_ONLY),
        ],
        "router_settings": {**object_value(shared["router_settings"]), "num_retries": 0},
        "guardrails": [
            {
                "guardrail_name": f"{module}-guardrail",
                "litellm_params": {
                    "guardrail": f"{module}.WebSocketModelRewriter",
                    "mode": "pre_call",
                    "default_on": True,
                },
            }
        ],
    }
    path: Final = directory / "gate-rewritten-config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


@pytest.fixture(scope="module")
def rewritten(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rewritten]:
    directory: Final = tmp_path_factory.mktemp("gate-rewritten")
    suffix: Final = uuid.uuid4().hex[:12]
    source: Final = f"gate-rewrite-chat-{suffix}"
    target: Final = f"gate-rewrite-embed-{suffix}"
    module: Final = f"gate_ws_rewriter_{suffix}"
    (directory / f"{module}.py").write_text(_REWRITER.replace("__SOURCE__", source).replace("__TARGET__", target))
    with (
        wire_server(answering_model_discovery(_respond)) as wire,
        gateway_from_environment() as rig_gateway,
        owned_proxy_process(
            rig_gateway,
            directory,
            {},
            config=_rewritten_config(directory, wire.url + "/v1", source, target, module),
            workers=2,
        ) as owned,
    ):
        eventually(
            lambda: owned.gateway.request("POST", "/v1/chat/completions", _chat_body(source, "probe")),
            lambda response: response.status_code == 200,
            seconds=30,
        )
        yield _Rewritten(owned.gateway, _Upstream(wire), source, target)


@_ROUTED_TIMEOUT
@pytest.mark.parametrize("limit", tuple(_RESERVING_KEYS))
def test_a_websocket_model_a_hook_rewrites_to_an_embedding_only_group_releases_the_key_reservation(
    rewritten: _Rewritten, limit: str
) -> None:
    markers: Final = tuple(_marker() for _ in range(3))
    served_marker: Final = _marker()
    with rewritten.gateway.scenario() as scenario:
        key: Final = scenario.key(**_RESERVING_KEYS[limit])
        closed: Final = tuple(
            asyncio.run(_websocket_first_frame_and_close(rewritten.gateway, key, rewritten.source, marker))
            for marker in markers
        )
        assert closed == (_websocket_rejection(rewritten.target),) * 3
        served: Final = rewritten.gateway.request(
            "POST", "/v1/chat/completions", _chat_body(rewritten.source, served_marker), key=key
        )
        assert served.status_code == 200, served.text
        assert _SERVED_TEXT in served.text, served.text
    assert [rewritten.upstream.calls(marker) for marker in markers] == [()] * 3
    assert rewritten.upstream.targets(served_marker) == ("/v1/chat/completions",)


@dataclass(frozen=True, slots=True)
class _Addressed:
    gateway: Gateway
    rig: Gateway
    upstream: _Upstream
    names: Mapping[str, str]
    addresses: Mapping[str, str]
    api_base: str


def _wildcard_backend_entry(name: str, api_base: str, backend: str) -> dict[str, JsonValue]:
    entry: Final = _routed_entry(name, api_base, backend, _EMBED_ONLY)
    return {**entry, "litellm_params": {**object_value(entry["litellm_params"]), "custom_llm_provider": "openai"}}


def _addressed_config(
    directory: Path, api_base: str, names: Mapping[str, str], suffix: str, with_wildcards: bool
) -> Path:
    shared: Final = _CONFIG.validate_python(yaml.safe_load((_TESTS_ROOT / "integration/proxy_config.yaml").read_text()))
    wildcard: Final = f"gate-addr-wild-{suffix}"
    wildcard_entries: Final = (
        (
            _routed_entry(f"{wildcard}/*", api_base, "openai/*", {}),
            _wildcard_backend_entry(names["wildcard_backend"], api_base, f"{wildcard}/{_CHAT_MODEL}"),
        )
        if with_wildcards
        else ()
    )
    aliases: Final = {names["wildcard_alias"]: f"{wildcard}/{_CHAT_BACKEND_PREFIX}{suffix}"} if with_wildcards else {}
    config: Final = {
        **shared,
        "model_list": [
            _routed_entry(
                names["embed"], api_base, f"openai/gate-addr-backend-{suffix}", {**_EMBED_ONLY, "id": names["embed_id"]}
            ),
            _routed_entry(names["chat"], api_base, f"openai/{_CHAT_MODEL}", {"mode": "chat", "id": names["chat_id"]}),
            _routed_entry(names["shared_embed"], api_base, f"openai/{_CHAT_MODEL}", _EMBED_ONLY),
            *wildcard_entries,
            _routed_entry(names["ghost_primary"], api_base, f"openai/{_EMBEDDING_MODEL}", _EMBED_ONLY),
        ],
        "router_settings": {
            **object_value(shared["router_settings"]),
            "num_retries": 0,
            "model_group_alias": aliases,
            "fallbacks": [{names["ghost_primary"]: [names["ghost"]]}, {names["ghost"]: [names["chat"]]}],
        },
        "general_settings": {
            **object_value(shared["general_settings"]),
            "proxy_config_reload_interval_seconds": 3600,
        },
    }
    path: Final = directory / "gate-addressed-config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


@contextlib.contextmanager
def _addressed_proxy(tmp_path_factory: pytest.TempPathFactory, with_wildcards: bool) -> Iterator[_Addressed]:
    directory: Final = tmp_path_factory.mktemp("gate-addressed")
    suffix: Final = uuid.uuid4().hex[:12]
    names: Final = MappingProxyType(
        {
            label: f"gate-addr-{label}-{suffix}"
            for label in (
                "embed",
                "embed_id",
                "chat",
                "chat_id",
                "shared_embed",
                "wildcard_backend",
                "wildcard_alias",
                "ghost_primary",
                "ghost",
            )
        }
    )
    with (
        wire_server(answering_model_discovery(_respond)) as wire,
        gateway_from_environment() as rig_gateway,
        owned_proxy_process(
            rig_gateway,
            directory,
            {},
            config=_addressed_config(directory, wire.url + "/v1", names, suffix, with_wildcards),
            workers=2,
        ) as owned,
    ):
        eventually(
            lambda: owned.gateway.request("POST", "/v1/chat/completions", _chat_body(names["chat"], "probe")),
            lambda response: response.status_code == 200,
            seconds=30,
        )
        yield _Addressed(
            owned.gateway,
            rig_gateway,
            _Upstream(wire),
            names,
            MappingProxyType(
                {
                    "deployment_id": names["embed_id"],
                    "backend_name": f"openai/gate-addr-backend-{suffix}",
                    "wildcard_backend_name": f"gate-addr-wild-{suffix}/{_CHAT_MODEL}",
                }
            ),
            wire.url + "/v1",
        )


@pytest.fixture(scope="module")
def addressed(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Addressed]:
    with _addressed_proxy(tmp_path_factory, with_wildcards=True) as proxy:
        yield proxy


@pytest.fixture(scope="module")
def addressed_without_wildcards(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Addressed]:
    with _addressed_proxy(tmp_path_factory, with_wildcards=False) as proxy:
        yield proxy


_ADDRESSED_ROUTES: Final = ("chat_v1", "messages", "responses_v1")


@_ROUTED_TIMEOUT
@pytest.mark.parametrize("route", _ADDRESSED_ROUTES)
@pytest.mark.parametrize("address", ("deployment_id", "backend_name"))
def test_chat_sent_to_an_embedding_only_deployment_id_or_backend_name_is_rejected_before_the_upstream(
    addressed: _Addressed, address: str, route: str
) -> None:
    spec: Final = _ROUTES[route]
    model: Final = addressed.addresses[address]
    marker: Final = _marker()
    response: Final = _send(addressed.gateway, spec, model, marker)
    expected: Final = (
        _anthropic_error(model, "/v1/embeddings")
        if spec.anthropic_shape
        else _openai_error(spec.endpoint, model, "/v1/embeddings")
    )
    _assert_rejected_before(addressed.upstream, response, marker, expected)


@_ROUTED_TIMEOUT
@pytest.mark.parametrize("address", ("deployment_id", "backend_name"))
def test_embeddings_sent_to_an_embedding_only_deployment_id_or_backend_name_keep_working(
    addressed: _Addressed, address: str
) -> None:
    marker: Final = _marker()
    response: Final = _owned_embedding(addressed.gateway, addressed.addresses[address], marker)
    assert response.status_code == 200, response.text
    assert list_value(
        object_value(list_value(JSON_OBJECT.validate_json(response.content)["data"])[0])["embedding"]
    ) == list(_VECTOR), response.text
    assert addressed.upstream.targets(marker) == ("/v1/embeddings",)


@_ROUTED_TIMEOUT
@pytest.mark.parametrize("address", ("chat_deployment_id", "backend_name_a_chat_deployment_shares"))
def test_chat_sent_to_a_chat_capable_deployment_id_or_backend_name_keeps_reaching_the_upstream(
    addressed: _Addressed, address: str
) -> None:
    model: Final = addressed.names["chat_id"] if address == "chat_deployment_id" else f"openai/{_CHAT_MODEL}"
    marker: Final = _marker()
    response: Final = addressed.gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker))
    assert response.status_code == 200, response.text
    assert _SERVED_TEXT in response.text, response.text
    assert addressed.upstream.targets(marker) == ("/v1/chat/completions",)


@_ROUTED_TIMEOUT
def test_request_fallbacks_naming_a_chat_deployment_id_answer_for_an_embedding_only_primary(
    addressed: _Addressed,
) -> None:
    marker: Final = _marker()
    response: Final = addressed.gateway.request(
        "POST",
        "/v1/chat/completions",
        {**_chat_body(addressed.names["embed"], marker), "fallbacks": [addressed.names["chat_id"]]},
    )
    assert response.status_code == 200, response.text
    assert _SERVED_TEXT in response.text, response.text
    assert addressed.upstream.targets(marker) == ("/v1/chat/completions",)


@_ROUTED_TIMEOUT
def test_chat_sent_to_an_embedding_only_backend_name_a_wildcard_also_matches_is_served_by_the_wildcard(
    addressed: _Addressed,
) -> None:
    marker: Final = _marker()
    response: Final = addressed.gateway.request(
        "POST", "/v1/chat/completions", _chat_body(addressed.addresses["wildcard_backend_name"], marker)
    )
    assert response.status_code == 200, response.text
    assert _SERVED_TEXT in response.text, response.text
    assert [JSON_OBJECT.validate_json(call.body)["model"] for call in addressed.upstream.calls(marker)] == [_CHAT_MODEL]


@_ROUTED_TIMEOUT
def test_request_fallbacks_naming_an_alias_only_a_wildcard_serves_answer_for_an_embedding_only_primary(
    addressed: _Addressed,
) -> None:
    marker: Final = _marker()
    response: Final = addressed.gateway.request(
        "POST",
        "/v1/chat/completions",
        {**_chat_body(addressed.names["embed"], marker), "fallbacks": [addressed.names["wildcard_alias"]]},
    )
    _assert_fallback_answered(addressed.upstream, response, marker)


@_ROUTED_TIMEOUT
def test_yaml_fallbacks_through_a_group_with_no_deployments_answer_for_an_embedding_only_group(
    addressed: _Addressed,
) -> None:
    marker: Final = _marker()
    response: Final = addressed.gateway.request(
        "POST", "/v1/chat/completions", _chat_body(addressed.names["ghost_primary"], marker)
    )
    _assert_fallback_answered(addressed.upstream, response, marker)


@_ROUTED_TIMEOUT
def test_an_embedding_only_model_the_read_through_loads_from_the_database_is_rejected_on_every_call(
    addressed_without_wildcards: _Addressed,
) -> None:
    addressed: Final = addressed_without_wildcards
    name: Final = f"gate-addr-loaded-{uuid.uuid4().hex[:12]}"
    markers: Final = tuple(_marker() for _ in range(6))
    with addressed.rig.scenario() as scenario:
        identity: Final = _create(
            addressed.rig, name, {"model": f"openai/{_EMBEDDING_MODEL}", "api_base": addressed.api_base}, _EMBED_ONLY
        )
        scenario.cleanups.callback(scenario.delete_model, identity)
        responses: Final = tuple(
            addressed.gateway.request("POST", "/v1/chat/completions", _chat_body(name, marker), headers=_CLOSE)
            for marker in markers
        )
    expected: Final = _openai_error("/v1/chat/completions", name, "/v1/embeddings")
    for response in responses:
        _assert_error(response, expected)
    assert [addressed.upstream.calls(marker) for marker in markers] == [()] * len(markers)


@pytest.mark.parametrize("metadata", ({"model_group": 7}, {"model_group": ["wrong-group"]}))
def test_rejection_with_invalid_metadata_is_logged_without_blocking_embeddings(
    rig: _Rig, metadata: dict[str, JsonValue]
) -> None:
    marker: Final = _marker()
    model: Final = rig.names["embedding_only"]
    rejected: Final = _chat(rig, model, marker, metadata=metadata)
    _assert_rejected_chat(rig, rejected, marker, model)
    embedded_marker: Final = _marker()
    embedded: Final = rig.gateway.request(
        "POST", "/v1/embeddings", {"model": model, "input": embedded_marker, "encoding_format": "float"}
    )
    assert embedded.status_code == 200, embedded.text
    assert rig.upstream.targets(embedded_marker) == ("/v1/embeddings",)


def test_existing_non_chat_metadata_on_a_chat_backend_remains_advisory(rig: _Rig) -> None:
    marker: Final = _marker()
    with rig.gateway.scenario() as scenario:
        name, identities = _deploy(
            scenario,
            rig.api_base,
            "legacy-audio",
            _single({"mode": "audio_speech", "supported_endpoints": ["/v1/audio/speech"]}),
        )
        _served_by_every_worker(rig.gateway, frozenset(identities))
        response: Final = _chat(rig, name, marker)
        _assert_served(rig, response, marker, "/v1/chat/completions")


def test_a_mixed_group_skips_its_restricted_embedding_deployment(rig: _Rig) -> None:
    marker: Final = _marker()
    with rig.gateway.scenario() as scenario:
        name: Final = f"gate-mixed-{uuid.uuid4().hex[:12]}"
        embed: Final = _create(
            scenario.gateway,
            name,
            {"model": f"openai/{_EMBEDDING_MODEL}", "api_base": rig.api_base, "order": 0},
            _EMBED_ONLY,
        )
        scenario.cleanups.callback(scenario.delete_model, embed)
        chat: Final = _create(
            scenario.gateway, name, {"model": f"openai/{_CHAT_MODEL}", "api_base": rig.api_base, "order": 1}, {}
        )
        scenario.cleanups.callback(scenario.delete_model, chat)
        _served_by_every_worker(rig.gateway, frozenset({embed, chat}))
        response: Final = _chat(rig, name, marker)
        _assert_served(rig, response, marker, "/v1/chat/completions")
        calls: Final = rig.upstream.calls(marker)
        assert [JSON_OBJECT.validate_json(call.body)["model"] for call in calls] == [_CHAT_MODEL]


def test_local_rate_limit_with_only_unsupported_fallbacks_keeps_the_original_429(rig: _Rig) -> None:
    limited: Final = rig.names["chat_rate_limited"]
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(
            model_rpm_limit={limited: 1}, router_settings={"fallbacks": [{limited: [rig.names["embedding_only"]]}]}
        )
        first: Final = _chat(rig, limited, _marker(), key=key)
        assert first.status_code == 200, first.text
        marker: Final = _marker()
        second: Final = _chat(rig, limited, marker, key=key)
        assert second.status_code == 429, second.text
        assert "Supported endpoints" not in second.text, second.text
        assert rig.upstream.calls(marker) == ()


def test_responses_websocket_uses_a_chat_fallback_without_calling_the_embedding_provider(
    rig: _Rig, tmp_path: Path
) -> None:
    marker: Final = _marker()
    certificate: Final = write_self_signed_cert(tmp_path)
    with (
        responses_ws_peer.responses_peer(
            certificate, lambda identity: responses_ws_peer.events(identity, _SERVED_TEXT)
        ) as peer,
        owned_proxy_process(rig.gateway, tmp_path, {"SSL_CERT_FILE": str(certificate[0])}, workers=2) as owned,
        owned.gateway.scenario() as scenario,
    ):
        fallback, fallback_ids = _deploy(scenario, peer.url, "ws-chat", _single({"mode": "chat"}))
        primary, primary_ids = _deploy(
            scenario, f"{rig.api_base}/{marker}", "ws-embed", _single(_EMBED_ONLY, f"openai/{_EMBEDDING_MODEL}")
        )
        _served_by_every_worker(owned.gateway, frozenset((*primary_ids, *fallback_ids)))
        key: Final = scenario.key(router_settings={"fallbacks": [{primary: [fallback]}]})
        (received,) = asyncio.run(
            responses_ws_peer.session(str(owned.gateway.client.base_url), key, primary, (marker,))
        )
        assert received[-1]["type"] == "response.completed", received
        assert responses_ws_peer.completed(received)["status"] == "completed"
        assert any(event.get("delta") == _SERVED_TEXT for event in received), received
        (forwarded,) = responses_ws_peer.drain(peer.frames)
        assert marker in json.dumps(forwarded), forwarded
    assert rig.upstream.calls(marker) == ()
