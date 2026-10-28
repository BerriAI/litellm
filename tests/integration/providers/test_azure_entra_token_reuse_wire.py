import asyncio
import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from queue import SimpleQueue
from typing import Final
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import websockets
import yaml
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.openai_wire import chat_reply, responses_reply
from integration._support.ports import allocate_port
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.responses_ws_peer import Peer, completed, responses_peer, session
from integration._support.tls import write_self_signed_cert
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue

pytestmark: Final = [pytest.mark.timeout(2 * graceful_stop_seconds() + 120), pytest.mark.shares_worker_state]

_DEPLOYMENT: Final = "gpt-5.4-nano"
_DATED_API_VERSION: Final = "2024-10-21"
_REALTIME_API_VERSION: Final = "2024-10-01-preview"
_IDENTITY_PATH: Final = "/msi/token"
_EXCHANGE_SUFFIX: Final = "/oauth2/v2.0/token"
_FOUNDRY_PROJECT: Final = "/api/projects/audit"
_BING_TOOL: Final = "audit-bing-entra"
_BING_RESOURCE: Final = "https://ai.azure.com"
_BING_CITATION: Final = "https://result.invalid/bing"
_TENANT: Final = "audit-tenant"
_APP_CLIENT: Final = "audit-app-client"
_HOLD_SECONDS: Final = 60
_LIVELINESS_SECONDS: Final = 3
_HERD: Final = 6
_HERD_WINDOW_SECONDS: Final = 3
_REMOVED_ENVIRONMENT: Final = (
    "AZURE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_AD_TOKEN",
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_SECRET",
    "AZURE_FEDERATED_TOKEN_FILE",
    "AZURE_USERNAME",
    "AZURE_PASSWORD",
    "AZURE_CREDENTIAL",
    "AZURE_SCOPE",
    "MSI_ENDPOINT",
    "MSI_SECRET",
    "IMDS_ENDPOINT",
    "IDENTITY_SERVER_THUMBPRINT",
    "BING_GROUNDING_TOKEN",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
)


class _Surface(StrEnum):
    LEGACY_CHAT = "legacy-chat"
    V1_CHAT = "v1-chat"
    RESPONSES = "responses"
    RESPONSES_WEBSOCKET = "responses-websocket"
    REALTIME = "realtime"
    REALTIME_HEALTH_CHECK = "realtime-health-check"


_HTTP_SURFACES: Final = frozenset({_Surface.LEGACY_CHAT, _Surface.V1_CHAT, _Surface.RESPONSES})


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    port: int
    cert: tuple[Path, Path]

    @property
    def edge(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@dataclass(frozen=True, slots=True)
class _Hold:
    resource: str
    arrivals: SimpleQueue[str] = field(default_factory=SimpleQueue)
    release: threading.Event = field(default_factory=threading.Event)


def _config(directory: Path) -> Path:
    stock: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    path: Final = directory / "azure-entra-token-reuse.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **stock,
                "litellm_settings": {**object_value(stock["litellm_settings"]), "enable_azure_ad_token_refresh": True},
                "router_settings": {**object_value(stock["router_settings"]), "num_retries": 0},
                "search_tools": [
                    {"search_tool_name": _BING_TOOL, "litellm_params": {"search_provider": "bing_grounding"}}
                ],
            }
        )
    )
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("azure-entra-token-reuse")
    port: Final = allocate_port()
    cert: Final = write_self_signed_cert(directory)
    edge: Final = f"http://127.0.0.1:{port}"
    environment: Final = {
        "AZURE_CLIENT_ID": "audit-managed-identity",
        "IDENTITY_ENDPOINT": f"{edge}{_IDENTITY_PATH}",
        "IDENTITY_HEADER": "audit-identity-header",
        "AZURE_AUTHORITY_HOST": edge,
        "BING_GROUNDING_PROJECT_ENDPOINT": f"{edge}{_FOUNDRY_PROJECT}",
        "BING_GROUNDING_MODEL": _DEPLOYMENT,
        "BING_GROUNDING_CONNECTION_ID": "audit-bing-connection",
        "SSL_CERT_FILE": str(cert[0]),
    }
    with (
        gateway_from_environment() as shared,
        owned_proxy(
            shared, directory, environment, config=_config(directory), remove_environment=_REMOVED_ENVIRONMENT
        ) as proxy,
    ):
        yield _Rig(proxy, port, cert)


def _json(payload: Mapping[str, JsonValue]) -> Reply:
    return Reply(body=json.dumps(payload).encode())


def _identity_token(resource: str) -> str:
    return f"mi-token:{resource}"


def _identity_resource(request: Request) -> str | None:
    target: Final = urlsplit(request.target)
    if target.path != _IDENTITY_PATH:
        return None
    return parse_qs(target.query)["resource"][-1]


def _is_token_endpoint(request: Request) -> bool:
    return _identity_resource(request) is not None or request.target.endswith(_EXCHANGE_SUFFIX)


def _managed_identity(resource: str, hold: _Hold | None) -> Reply:
    if hold is not None and resource == hold.resource:
        hold.arrivals.put(resource)
        hold.release.wait(_HOLD_SECONDS)
    return _json(
        {
            "access_token": _identity_token(resource),
            "expires_on": str(int(time.time()) + 3600),
            "resource": resource,
            "token_type": "Bearer",
        }
    )


def _entra_exchange(request: Request) -> Reply:
    assertion: Final = parse_qs(request.body.decode())["client_assertion"][-1]
    return _json({"access_token": f"aad:{assertion}", "expires_in": 3600, "token_type": "Bearer"})


def _edge(upstream: Callable[[Request], Reply], hold: _Hold | None = None) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        resource: Final = _identity_resource(request)
        if resource is not None:
            return _managed_identity(resource, hold)
        if request.target.endswith(_EXCHANGE_SUFFIX):
            return _entra_exchange(request)
        return upstream(request)

    return respond


def _azure_openai(request: Request) -> Reply:
    if urlsplit(request.target).path.endswith("/chat/completions"):
        return chat_reply(f"chatcmpl-{uuid.uuid4().hex}", _DEPLOYMENT, "azure answered", stream=False)
    return responses_reply(f"resp_{uuid.uuid4().hex}", _DEPLOYMENT, "azure answered", stream=False)


def _identity_fetches(requests: tuple[Request, ...], resource: str) -> int:
    return sum(1 for request in requests if _identity_resource(request) == resource)


def _provider_calls(requests: tuple[Request, ...]) -> tuple[Request, ...]:
    return tuple(request for request in requests if not _is_token_endpoint(request))


def _marker() -> str:
    return f"entra audit {uuid.uuid4().hex}"


def _model(scenario: Scenario, surface: _Surface, api_base: str, scope: str) -> str:
    match surface:
        case _Surface.LEGACY_CHAT | _Surface.RESPONSES:
            return scenario.model(
                model=f"azure/{_DEPLOYMENT}",
                api_base=api_base,
                api_key=None,
                api_version=_DATED_API_VERSION,
                azure_scope=scope,
            )
        case _Surface.V1_CHAT:
            return scenario.model(
                model=f"azure/{_DEPLOYMENT}", api_base=api_base, api_key=None, api_version="v1", azure_scope=scope
            )
        case _Surface.RESPONSES_WEBSOCKET:
            return scenario.model(model=f"azure/{_DEPLOYMENT}", api_base=api_base, api_key=None, azure_scope=scope)
        case _Surface.REALTIME | _Surface.REALTIME_HEALTH_CHECK:
            return scenario.model(
                model_info={"mode": "realtime"},
                model=f"azure/{_DEPLOYMENT}",
                api_base=api_base,
                api_key=None,
                api_version=_REALTIME_API_VERSION,
                azure_scope=scope,
            )


def _post(surface: _Surface, gateway: Gateway, key: str, model: str) -> httpx.Response:
    if surface is _Surface.RESPONSES:
        return gateway.request("POST", "/v1/responses", {"model": model, "input": _marker()}, key=key)
    return gateway.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": _marker()}]}, key=key
    )


async def _realtime_handshake(gateway: Gateway, key: str, model: str, peer: Peer) -> str | None:
    proxy: Final = str(gateway.client.base_url).rstrip("/").replace("http://", "ws://")
    async with websockets.connect(
        f"{proxy}/v1/realtime?model={model}",
        additional_headers={"Authorization": f"Bearer {key}"},
        open_timeout=_HOLD_SECONDS,
    ):
        credentials: Final = await asyncio.to_thread(peer.credentials.get, True, _HOLD_SECONDS)
    return credentials.authorization


def _upstream_authorization(surface: _Surface, gateway: Gateway, key: str, model: str, wire: Wire, peer: Peer) -> str:
    match surface:
        case _Surface.LEGACY_CHAT | _Surface.V1_CHAT | _Surface.RESPONSES:
            response: Final = _post(surface, gateway, key, model)
            assert response.status_code == 200, response.text
            (call,) = _provider_calls(wire.drain())
            return call.headers.get("authorization", "")
        case _Surface.RESPONSES_WEBSOCKET:
            (events,) = asyncio.run(session(str(gateway.client.base_url), key, model, (_marker(),)))
            assert completed(events)["status"] == "completed", events
            return peer.credentials.get(timeout=_HOLD_SECONDS).authorization or ""
        case _Surface.REALTIME:
            return asyncio.run(_realtime_handshake(gateway, key, model, peer)) or ""
        case _Surface.REALTIME_HEALTH_CHECK:
            gateway.request("GET", "/health", params={"model": model})
            return peer.credentials.get(timeout=_HOLD_SECONDS).authorization or ""


def _liveliness(gateway: Gateway) -> int | None:
    try:
        return httpx.get(
            f"{str(gateway.client.base_url).rstrip('/')}/health/liveliness", timeout=_LIVELINESS_SECONDS
        ).status_code
    except httpx.TimeoutException:
        return None


def _liveliness_while_held(gateway: Gateway, hold: _Hold) -> int | None:
    try:
        assert hold.arrivals.get(timeout=_HOLD_SECONDS) == hold.resource
        return _liveliness(gateway)
    finally:
        hold.release.set()


@pytest.mark.parametrize("surface", tuple(_Surface))
def test_a_slow_managed_identity_fetch_leaves_the_proxy_answering_other_requests(rig: _Rig, surface: _Surface) -> None:
    resource: Final = f"api://audit-hold-{uuid.uuid4().hex}"
    hold: Final = _Hold(resource)
    with (
        wire_server(_edge(_azure_openai, hold), port=rig.port) as wire,
        responses_peer(rig.cert) as peer,
        rig.gateway.scenario() as scenario,
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        api_base: Final = rig.edge if surface in _HTTP_SURFACES else peer.url.removesuffix("/v1")
        model: Final = _model(scenario, surface, api_base, f"{resource}/.default")
        key: Final = scenario.key()
        call: Final = pool.submit(_upstream_authorization, surface, rig.gateway, key, model, wire, peer)
        liveliness: Final = _liveliness_while_held(rig.gateway, hold)
        authorization: Final = call.result(timeout=_HOLD_SECONDS)

    assert liveliness == 200, f"/health/liveliness did not answer within {_LIVELINESS_SECONDS}s during the fetch"
    assert authorization == f"Bearer {_identity_token(resource)}"


@pytest.mark.parametrize("surface", (_Surface.LEGACY_CHAT, _Surface.RESPONSES))
def test_concurrent_cold_requests_share_one_managed_identity_fetch(rig: _Rig, surface: _Surface) -> None:
    resource: Final = f"api://audit-herd-{uuid.uuid4().hex}"
    hold: Final = _Hold(resource)
    with (
        wire_server(_edge(_azure_openai, hold), port=rig.port) as wire,
        rig.gateway.scenario() as scenario,
        ThreadPoolExecutor(max_workers=_HERD) as pool,
    ):
        model: Final = _model(scenario, surface, rig.edge, f"{resource}/.default")
        key: Final = scenario.key()
        calls: Final = tuple(pool.submit(_post, surface, rig.gateway, key, model) for _ in range(_HERD))
        try:
            eventually(
                hold.arrivals.qsize,
                lambda arrived: arrived >= _HERD,
                seconds=_HERD_WINDOW_SECONDS,
                return_last_on_timeout=True,
            )
        finally:
            hold.release.set()
        responses: Final = tuple(call.result(timeout=_HOLD_SECONDS) for call in calls)
        requests: Final = wire.drain()

    assert [response.status_code for response in responses] == [200] * _HERD, [r.text for r in responses]
    assert _identity_fetches(requests, resource) == 1
    assert [call.headers.get("authorization") for call in _provider_calls(requests)] == [
        f"Bearer {_identity_token(resource)}"
    ] * _HERD


def _oidc_model(scenario: Scenario, rig: _Rig, audience: str) -> str:
    return scenario.model(
        model=f"azure/{_DEPLOYMENT}",
        api_base=rig.edge,
        api_key=None,
        api_version=_DATED_API_VERSION,
        azure_ad_token=f"oidc/azure/{audience}",
        tenant_id=_TENANT,
        client_id=_APP_CLIENT,
    )


def _exchanges(requests: tuple[Request, ...]) -> tuple[Request, ...]:
    return tuple(request for request in requests if request.target.endswith(_EXCHANGE_SUFFIX))


def test_an_oidc_federated_deployment_fetches_its_assertion_once_across_responses_calls(rig: _Rig) -> None:
    audience: Final = f"api://audit-oidc-{uuid.uuid4().hex}"
    with wire_server(_edge(_azure_openai), port=rig.port) as wire, rig.gateway.scenario() as scenario:
        model: Final = _oidc_model(scenario, rig, audience)
        key: Final = scenario.key()
        responses: Final = tuple(_post(_Surface.RESPONSES, rig.gateway, key, model) for _ in range(3))
        requests: Final = wire.drain()

    assert [response.status_code for response in responses] == [200] * 3, [r.text for r in responses]
    assert _identity_fetches(requests, audience) == 1
    exchanges: Final = _exchanges(requests)
    assert [request.target for request in exchanges] == [f"/{_TENANT}{_EXCHANGE_SUFFIX}"]
    assert {name: values[-1] for name, values in parse_qs(exchanges[0].body.decode()).items()} == {
        "client_id": _APP_CLIENT,
        "grant_type": "client_credentials",
        "scope": "https://cognitiveservices.azure.com/.default",
        "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
        "client_assertion": _identity_token(audience),
    }
    assert [call.headers.get("authorization") for call in _provider_calls(requests)] == [
        f"Bearer aad:{_identity_token(audience)}"
    ] * 3


def test_concurrent_cold_oidc_requests_share_one_assertion_fetch_and_one_exchange(rig: _Rig) -> None:
    audience: Final = f"api://audit-oidc-herd-{uuid.uuid4().hex}"
    hold: Final = _Hold(audience)
    with (
        wire_server(_edge(_azure_openai, hold), port=rig.port) as wire,
        rig.gateway.scenario() as scenario,
        ThreadPoolExecutor(max_workers=_HERD) as pool,
    ):
        model: Final = _oidc_model(scenario, rig, audience)
        key: Final = scenario.key()
        calls: Final = tuple(pool.submit(_post, _Surface.RESPONSES, rig.gateway, key, model) for _ in range(_HERD))
        try:
            eventually(
                hold.arrivals.qsize,
                lambda arrived: arrived >= _HERD,
                seconds=_HERD_WINDOW_SECONDS,
                return_last_on_timeout=True,
            )
        finally:
            hold.release.set()
        responses: Final = tuple(call.result(timeout=_HOLD_SECONDS) for call in calls)
        requests: Final = wire.drain()

    assert [response.status_code for response in responses] == [200] * _HERD, [r.text for r in responses]
    assert _identity_fetches(requests, audience) == 1
    assert len(_exchanges(requests)) == 1
    assert [call.headers.get("authorization") for call in _provider_calls(requests)] == [
        f"Bearer aad:{_identity_token(audience)}"
    ] * _HERD


def _foundry(request: Request) -> Reply:
    assert urlsplit(request.target).path == f"{_FOUNDRY_PROJECT}/openai/v1/responses", request.target
    text: Final = "Audit result."
    return _json(
        {
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": text,
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": _BING_CITATION,
                                    "title": "Bing result",
                                    "start_index": len(text),
                                    "end_index": len(text),
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )


class _SearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    url: str


class _SearchResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    results: tuple[_SearchResult, ...]


def test_bing_grounding_search_mints_its_entra_token_once_across_searches(rig: _Rig) -> None:
    with wire_server(_edge(_foundry), port=rig.port) as wire:
        responses: Final = tuple(
            rig.gateway.request("POST", f"/v1/search/{_BING_TOOL}", {"query": _marker(), "max_results": 1})
            for _ in range(3)
        )
        requests: Final = wire.drain()

    assert [response.status_code for response in responses] == [200] * 3, [r.text for r in responses]
    assert [
        tuple(result.url for result in _SearchResponse.model_validate_json(response.content).results)
        for response in responses
    ] == [(_BING_CITATION,)] * 3
    assert _identity_fetches(requests, _BING_RESOURCE) == 1
    assert [call.headers.get("authorization") for call in _provider_calls(requests)] == [
        f"Bearer {_identity_token(_BING_RESOURCE)}"
    ] * 3


def _lifecycle(upstream_id: str) -> Callable[[Request], Reply]:
    response: Final[dict[str, JsonValue]] = {
        "id": upstream_id,
        "object": "response",
        "created_at": 1,
        "model": _DEPLOYMENT,
        "output": [],
    }

    def respond(request: Request) -> Reply:
        path: Final = urlsplit(request.target).path
        match request.method:
            case "DELETE":
                return _json({"id": upstream_id, "object": "response", "deleted": True})
            case "GET" if path.endswith("/input_items"):
                return _json({"object": "list", "data": [], "first_id": None, "last_id": None, "has_more": False})
            case "POST" if path.endswith("/cancel"):
                return _json({**response, "status": "cancelled"})
            case _:
                return _json({**response, "status": "completed"})

    return respond


_LIFECYCLE_ROUTES: Final = (
    ("GET", "/v1/responses/{}", ""),
    ("GET", "/v1/responses/{}/input_items", "/input_items"),
    ("POST", "/v1/responses/{}/cancel", "/cancel"),
    ("DELETE", "/v1/responses/{}", ""),
)


def test_every_responses_route_on_an_entra_deployment_reaches_azure_with_the_managed_identity_token(
    rig: _Rig,
) -> None:
    resource: Final = f"api://audit-routes-{uuid.uuid4().hex}"
    upstream_id: Final = f"resp_{uuid.uuid4().hex}"
    with wire_server(_edge(_lifecycle(upstream_id)), port=rig.port) as wire, rig.gateway.scenario() as scenario:
        model: Final = _model(scenario, _Surface.RESPONSES, rig.edge, f"{resource}/.default")
        key: Final = scenario.key()
        created: Final = _post(_Surface.RESPONSES, rig.gateway, key, model)
        assert created.status_code == 200, created.text
        response_id: Final = string_value(JSON_OBJECT.validate_json(created.content)["id"])
        followups: Final = tuple(
            rig.gateway.request(method, route.format(response_id), key=key) for method, route, _ in _LIFECYCLE_ROUTES
        )
        requests: Final = wire.drain()

    assert [response.status_code for response in followups] == [200] * len(_LIFECYCLE_ROUTES), [
        response.text for response in followups
    ]
    calls: Final = _provider_calls(requests)
    assert (calls[0].method, urlsplit(calls[0].target).path.endswith("/responses")) == ("POST", True)
    assert [
        (call.method, urlsplit(call.target).path.endswith(f"/{upstream_id}{suffix}"))
        for call, (_, _, suffix) in zip(calls[1:], _LIFECYCLE_ROUTES, strict=True)
    ] == [(method, True) for method, _, _ in _LIFECYCLE_ROUTES]
    assert {call.headers.get("authorization") for call in calls} == {f"Bearer {_identity_token(resource)}"}
    assert _identity_fetches(requests, resource) == 1
