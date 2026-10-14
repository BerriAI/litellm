import asyncio
import json
import uuid
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Final

import anthropic
import httpx
import openai
import pytest
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
from integration._support.database import read_rows
from integration._support.openai_wire import answering_model_discovery, chat_reply, responses_reply
from integration._support.process import graceful_stop_seconds, owned_proxy_process, owned_upstream, paused
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

from tests.integration.cost_calculation.cost_tracking_case import JsonResponse, RoutedResponse

_PROVIDER_KEY: Final = "integration-provider-key"
_EMBEDDING_MODEL: Final = "text-embedding-3-small"
_CHAT_MODEL: Final = "gpt-4o-mini"
_SERVED_TEXT: Final = "served by the chat upstream"
_VECTOR: Final = (0.25, 0.5, 0.75)
_CLOSE: Final = MappingProxyType({"connection": "close"})
_RELOAD_SECONDS: Final = 5
_LONG_ENDPOINT: Final = "/v1/embeddings/" + "x" * 5000
_EMBED_ONLY: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"mode": "embedding", "supported_endpoints": ["/v1/embeddings"]}
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


def _respond(request: Request) -> Reply:
    path: Final = request.target.split("?")[0]
    body: Final = JSON_OBJECT.validate_json(request.body or b"{}")
    streamed: Final = body.get("stream") is True
    if path == "/v1/embeddings":
        return _embedding_reply()
    if body.get("model") != _CHAT_MODEL:
        return _not_a_chat_model()
    if path == "/v1/chat/completions":
        return chat_reply(f"chatcmpl-{uuid.uuid4().hex}", _CHAT_MODEL, _SERVED_TEXT, stream=streamed)
    if path == "/v1/responses":
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
        return tuple(request for request in self.seen if marker.encode() in request.body)

    def targets(self, marker: str) -> tuple[str, ...]:
        return tuple(request.target for request in self.calls(marker))


@dataclass(frozen=True, slots=True)
class _Deployment:
    litellm_model: str
    declarations: tuple[Mapping[str, JsonValue], ...]


def _single(declaration: Mapping[str, JsonValue], litellm_model: str = f"openai/{_CHAT_MODEL}") -> _Deployment:
    return _Deployment(litellm_model, (declaration,))


_GATED: Final[Mapping[str, tuple[_Deployment, str]]] = MappingProxyType(
    {
        "image_generation": (
            _single({"mode": "image_generation", "supported_endpoints": ["/v1/images/generations"]}),
            "/v1/images/generations",
        ),
        "rerank_two_spellings": (
            _single({"mode": "rerank", "supported_endpoints": ["/v1/rerank", "/rerank"]}),
            "/rerank, /v1/rerank",
        ),
        "empty_endpoints": (_single({"mode": "embedding", "supported_endpoints": []}), "none"),
        "duplicated_endpoints": (
            _single({"mode": "embedding", "supported_endpoints": ["/v1/embeddings", "/v1/embeddings", "/embeddings"]}),
            "/embeddings, /v1/embeddings",
        ),
        "trailing_slash_chat_spelling": (
            _single({"mode": "embedding", "supported_endpoints": ["/v1/chat/completions/"]}),
            "/v1/chat/completions/",
        ),
        "five_kb_endpoint": (_single({"mode": "embedding", "supported_endpoints": [_LONG_ENDPOINT]}), _LONG_ENDPOINT),
        "two_embedding_deployments": (
            _Deployment(
                f"openai/{_CHAT_MODEL}",
                (_EMBED_ONLY, {"mode": "embedding", "supported_endpoints": ["/embeddings"]}),
            ),
            "/embeddings, /v1/embeddings",
        ),
    }
)

_UNGATED: Final[Mapping[str, _Deployment]] = MappingProxyType(
    {
        "mode_only": _single({"mode": "embedding"}),
        "endpoints_only": _single({"supported_endpoints": ["/v1/embeddings"]}),
        "endpoints_null": _single({"mode": "embedding", "supported_endpoints": None}),
        "mode_null": _single({"mode": None, "supported_endpoints": ["/v1/embeddings"]}),
        "mode_blank": _single({"mode": "", "supported_endpoints": ["/v1/embeddings"]}),
        "endpoints_a_string": _single({"mode": "embedding", "supported_endpoints": "/v1/embeddings"}),
        "endpoints_not_strings": _single({"mode": "embedding", "supported_endpoints": [1, 2]}),
        "chat_mode": _single({"mode": "chat", "supported_endpoints": ["/v1/embeddings"]}),
        "completion_mode": _single({"mode": "completion", "supported_endpoints": ["/v1/embeddings"]}),
        "responses_mode": _single({"mode": "responses", "supported_endpoints": ["/v1/embeddings"]}),
        "lists_v1_chat": _single(
            {"mode": "embedding", "supported_endpoints": ["/v1/embeddings", "/v1/chat/completions"]}
        ),
        "lists_bare_chat": _single(
            {"mode": "embedding", "supported_endpoints": ["/v1/embeddings", "/chat/completions"]}
        ),
        "lists_messages": _single({"mode": "embedding", "supported_endpoints": ["/v1/embeddings", "/v1/messages"]}),
        "lists_bare_responses": _single({"mode": "embedding", "supported_endpoints": ["/v1/embeddings", "/responses"]}),
        "mixed_group": _Deployment(f"openai/{_CHAT_MODEL}", (_EMBED_ONLY, {"mode": "chat"})),
    }
)

_SHARED: Final[Mapping[str, _Deployment]] = MappingProxyType(
    {
        "embedding_only": _single(_EMBED_ONLY, f"openai/{_EMBEDDING_MODEL}"),
        "chat_target": _single({"mode": "chat"}),
        **{name: deployment for name, (deployment, _supported) in _GATED.items()},
        **_UNGATED,
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


def _assert_rejected(rig: _Rig, response: httpx.Response, marker: str, expected: dict[str, JsonValue]) -> None:
    assert response.status_code == 400, response.text
    assert JSON_OBJECT.validate_json(response.content) == expected, response.text
    assert rig.upstream.calls(marker) == (), rig.upstream.targets(marker)
    assert _failure_row(response.headers["x-litellm-call-id"])["status"] == "failure"


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


def _openai_sdk_error(rig: _Rig, client: str, model: str, marker: str) -> openai.APIStatusError:
    base_url: Final = f"{rig.gateway.client.base_url}/v1"

    def sync_call() -> None:
        with openai.OpenAI(base_url=base_url, api_key=rig.gateway.key, max_retries=0) as sdk:
            if client == "openai_sync_chat":
                sdk.chat.completions.create(model=model, messages=[{"role": "user", "content": marker}])
            else:
                sdk.responses.create(model=model, input=marker)

    async def async_call() -> None:
        async with openai.AsyncOpenAI(base_url=base_url, api_key=rig.gateway.key, max_retries=0) as sdk:
            if client == "openai_async_chat_stream":
                await sdk.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": marker}], stream=True
                )
            else:
                await sdk.responses.create(model=model, input=marker, stream=True)

    def call() -> None:
        return sync_call() if client.startswith("openai_sync") else asyncio.run(async_call())

    with pytest.raises(openai.BadRequestError) as raised:
        call()
    return raised.value


def _anthropic_sdk_error(rig: _Rig, client: str, model: str, marker: str) -> anthropic.APIStatusError:
    base_url: Final = str(rig.gateway.client.base_url)

    def sync_call() -> None:
        with anthropic.Anthropic(base_url=base_url, api_key=rig.gateway.key, max_retries=0) as sdk:
            sdk.messages.create(model=model, max_tokens=16, messages=[{"role": "user", "content": marker}])

    async def async_call() -> None:
        async with anthropic.AsyncAnthropic(base_url=base_url, api_key=rig.gateway.key, max_retries=0) as sdk:
            await sdk.messages.create(
                model=model, max_tokens=16, messages=[{"role": "user", "content": marker}], stream=True
            )

    def call() -> None:
        return sync_call() if client == "anthropic_sync" else asyncio.run(async_call())

    with pytest.raises(anthropic.BadRequestError) as raised:
        call()
    return raised.value


_SDK_CLIENTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "openai_sync_chat": "/v1/chat/completions",
        "openai_async_chat_stream": "/v1/chat/completions",
        "openai_sync_responses": "/v1/responses",
        "openai_async_responses_stream": "/v1/responses",
        "anthropic_sync": "/v1/messages",
        "anthropic_async_stream": "/v1/messages",
    }
)


@pytest.mark.parametrize("client", tuple(_SDK_CLIENTS))
def test_sdk_clients_see_the_rejection_as_a_bad_request(rig: _Rig, client: str) -> None:
    model: Final = rig.names["embedding_only"]
    marker: Final = _marker()
    anthropic_client: Final = client.startswith("anthropic")
    error: Final = (
        _anthropic_sdk_error(rig, client, model, marker)
        if anthropic_client
        else _openai_sdk_error(rig, client, model, marker)
    )
    expected: Final = (
        _anthropic_error(model, "/v1/embeddings")
        if anthropic_client
        else _openai_error(_SDK_CLIENTS[client], model, "/v1/embeddings")
    )
    assert error.status_code == 400, error
    _assert_rejected(rig, error.response, marker, expected)


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


@pytest.mark.parametrize("declaration", tuple(_GATED))
def test_non_chat_declarations_reject_chat_completions(rig: _Rig, declaration: str) -> None:
    model: Final = rig.names[declaration]
    marker: Final = _marker()
    response: Final = rig.gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker))
    _assert_rejected(rig, response, marker, _openai_error("/v1/chat/completions", model, _GATED[declaration][1]))


def _assert_served(rig: _Rig, response: httpx.Response, marker: str, target: str) -> None:
    assert response.status_code == 200, response.text
    assert _SERVED_TEXT in response.text, response.text
    assert rig.upstream.targets(marker) == (target,)


@pytest.mark.parametrize("declaration", tuple(_UNGATED))
def test_chat_capable_or_unclear_declarations_keep_reaching_the_upstream(rig: _Rig, declaration: str) -> None:
    marker: Final = _marker()
    response: Final = rig.gateway.request("POST", "/v1/chat/completions", _chat_body(rig.names[declaration], marker))
    _assert_served(rig, response, marker, "/v1/chat/completions")


def test_key_alias_to_an_embedding_only_model_is_rejected_under_the_resolved_name(rig: _Rig) -> None:
    target: Final = rig.names["embedding_only"]
    alias: Final = f"gate-alias-{uuid.uuid4().hex[:12]}"
    marker: Final = _marker()
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(aliases={alias: target})
        response: Final = rig.gateway.request("POST", "/v1/chat/completions", _chat_body(alias, marker), key=key)
        _assert_rejected(rig, response, marker, _openai_error("/v1/chat/completions", target, "/v1/embeddings"))


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
    assert rig.upstream.targets(marker) == ("/v1/chat/completions", "/v1/chat/completions")


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
    fallbacks: Final = [rig.names["image_generation"], model, f"gate-missing-{uuid.uuid4().hex}"]
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


def test_non_string_mode_keeps_reaching_the_upstream(rig: _Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.gateway.request(
        "POST", "/v1/chat/completions", _chat_body(rig.names["mode_not_a_string"], marker)
    )
    assert response.status_code == 404, response.text
    assert "scripted: not a chat model" in response.text, response.text
    assert "Supported endpoints" not in response.text, response.text
    assert rig.upstream.targets(marker) == ("/v1/chat/completions",)


def test_repeated_rejections_log_one_failure_row_per_call(rig: _Rig) -> None:
    model: Final = rig.names["embedding_only"]
    marker: Final = _marker()
    responses: Final = tuple(
        rig.gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker)) for _ in range(3)
    )
    expected: Final = _openai_error("/v1/chat/completions", model, "/v1/embeddings")
    assert [(item.status_code, item.json()) for item in responses] == [(400, expected)] * 3
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
        assert rejected.json() == _openai_error("/v1/chat/completions", name, "/v1/embeddings"), rejected.text
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
_BURST_ROUTES: Final = (
    "chat_v1",
    "chat_v1_stream",
    "messages",
    "messages_stream",
    "responses_v1",
    "responses_v1_stream",
)


def _scripted(url: str, identity: str, routes: Mapping[str, JsonResponse]) -> None:
    registered: Final = httpx.post(
        f"{url}/__scenarios",
        json={
            "scenario_id": identity,
            "response": RoutedResponse(content_type="application/x-routed", routes=dict(routes)).model_dump(
                mode="json"
            ),
        },
        timeout=10,
        trust_env=False,
    )
    registered.raise_for_status()


class _Observed:
    def __init__(self, url: str) -> None:
        self.url = url
        self.items: tuple[dict[str, JsonValue], ...] = ()

    def read(self) -> tuple[dict[str, JsonValue], ...]:
        payload: Final = JSON_OBJECT.validate_json(
            httpx.get(f"{self.url}/__observations?include_method=true", timeout=10, trust_env=False).content
        )
        self.items = (*self.items, *(object_value(item) for item in list_value(payload["requests"])))
        return self.items

    def carrying(self, marker: str) -> tuple[str, ...]:
        return tuple(str(item["path"]) for item in self.read() if marker in json.dumps(item.get("body")))


def _owned_config(directory: Path, url: str, embed: tuple[str, str], chat: tuple[str, str]) -> Path:
    shared: Final = _CONFIG.validate_python(yaml.safe_load((_TESTS_ROOT / "integration/proxy_config.yaml").read_text()))
    litellm_settings: Final = object_value(shared["litellm_settings"])
    config: Final = {
        **shared,
        "model_list": [
            {
                "model_name": embed[0],
                "litellm_params": {
                    "model": f"openai/{_EMBEDDING_MODEL}",
                    "api_base": f"{url}/{embed[1]}/v1",
                    "api_key": _PROVIDER_KEY,
                },
                "model_info": dict(_EMBED_ONLY),
            },
            {
                "model_name": chat[0],
                "litellm_params": {
                    "model": f"openai/{_CHAT_MODEL}",
                    "api_base": f"{url}/{chat[1]}/v1",
                    "api_key": _PROVIDER_KEY,
                },
            },
        ],
        "litellm_settings": {**litellm_settings, "responses": {"background_mode": {"polling_via_cache": "all"}}},
    }
    path: Final = directory / "gate-config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _owned_embedding(owned: Gateway, model: str, marker: str) -> httpx.Response:
    return owned.request("POST", "/v1/embeddings", {"model": model, "input": marker, "encoding_format": "float"})


def _gated_expectation(route: str, model: str) -> dict[str, JsonValue]:
    spec: Final = _ROUTES[route]
    return (
        _anthropic_error(model, "/v1/embeddings")
        if spec.anthropic_shape
        else _openai_error(spec.endpoint, model, "/v1/embeddings")
    )


def _assert_gated_burst(results: tuple[tuple[str, httpx.Response], ...], model: str) -> tuple[str, ...]:
    assert [(response.status_code, JSON_OBJECT.validate_json(response.content)) for _route, response in results] == [
        (400, _gated_expectation(route, model)) for route, _response in results
    ], [response.text for _route, response in results]
    return tuple(response.headers["x-litellm-call-id"] for _route, response in results)


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_yaml_embedding_only_model_stays_gated_through_an_upstream_pause_and_a_worker_kill(
    gateway: Gateway, tmp_path: Path
) -> None:
    suffix: Final = uuid.uuid4().hex[:12]
    embed: Final = (f"gate-yaml-embed-{suffix}", f"gate-embed-{suffix}")
    chat: Final = (f"gate-yaml-chat-{suffix}", f"gate-chat-{suffix}")
    embedding: Final = JsonResponse(
        content_type="application/json", body=JSON_OBJECT.validate_json(_embedding_reply().body)
    )
    chat_completion: Final = JsonResponse(
        content_type="application/json",
        body={
            "id": f"chatcmpl-{suffix}",
            "object": "chat.completion",
            "created": 1,
            "model": _CHAT_MODEL,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": _SERVED_TEXT}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
        },
    )
    with owned_upstream(tmp_path) as slot:
        upstream: Final = slot.process
        assert upstream is not None
        url: Final = slot.url
        _scripted(url, embed[1], {"POST /v1/embeddings": embedding})
        _scripted(url, chat[1], {"POST /v1/chat/completions": chat_completion})
        config: Final = _owned_config(tmp_path, url, embed, chat)
        with owned_proxy_process(
            gateway, tmp_path, {"INTEGRATION_UPSTREAM_URL": url}, config=config, workers=2
        ) as owned:
            proxy: Final = owned.gateway
            eventually(
                lambda: proxy.request("POST", "/v1/chat/completions", _chat_body(chat[0], "probe")).status_code,
                lambda status: status == 200,
                seconds=30,
            )
            workers: Final = eventually(owned.worker_pids, lambda pids: len(pids) >= 2, seconds=30)
            observed: Final = _Observed(url)
            background_marker: Final = _marker()
            background: Final = proxy.request(
                "POST", "/v1/responses", {"model": embed[0], "input": background_marker, "background": True}
            )
            assert background.status_code == 400, background.text
            assert background.json() == _openai_error("/v1/responses", embed[0], "/v1/embeddings"), background.text
            burst: Final = tuple((_BURST_ROUTES[index % len(_BURST_ROUTES)], _marker()) for index in range(18))
            with paused(upstream), ThreadPoolExecutor(max_workers=9) as pool:
                futures: Final = tuple(
                    (route, pool.submit(_send, proxy, _ROUTES[route], embed[0], marker)) for route, marker in burst
                )
                paused_results: Final = tuple((route, future.result(timeout=15)) for route, future in futures)
            gated_ids: Final = _assert_gated_burst(paused_results, embed[0])
            served_markers: Final = tuple(_marker() for _ in range(8))
            with ThreadPoolExecutor(max_workers=8) as pool:
                served: Final = tuple(pool.map(partial(_owned_embedding, proxy, embed[0]), served_markers))
            assert [response.status_code for response in served] == [200] * 8, [item.text for item in served]
            embedded_paths: Final = eventually(
                lambda: tuple(observed.carrying(marker) for marker in served_markers),
                lambda paths: all(len(entry) >= 1 for entry in paths),
                seconds=20,
            )
            assert embedded_paths == ((f"/{embed[1]}/v1/embeddings",),) * 8, embedded_paths
            assert [observed.carrying(marker) for _route, marker in burst] == [()] * 18
            assert observed.carrying(background_marker) == ()
            assert [_failure_row(call_id)["status"] for call_id in gated_ids] == ["failure"] * 18
            with owned.replacing_worker(min(workers)) as kill:
                kill()
                after_kill: Final = tuple(
                    ("chat_v1", _send(proxy, _ROUTES["chat_v1"], embed[0], _marker())) for _ in range(4)
                )
                _assert_gated_burst(after_kill, embed[0])
                assert _owned_embedding(proxy, embed[0], _marker()).status_code == 200
            respawned: Final = tuple(
                (
                    "chat_v1",
                    proxy.request("POST", "/v1/chat/completions", _chat_body(embed[0], _marker()), headers=_CLOSE),
                )
                for _ in range(6)
            )
            _assert_gated_burst(respawned, embed[0])
