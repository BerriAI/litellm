from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, replace
from typing import Final

import httpx
import pytest
import websockets
from openai import AsyncOpenAI, OpenAI
from openai.types.realtime import ClientSecretCreateResponse
from pydantic import BaseModel, ConfigDict, JsonValue
from websockets.exceptions import InvalidStatus

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, object_value, string_value
from tests.integration._support.upstream import delete_scenario, register_scenario
from tests.integration.cost_calculation.cost_tracking_case import (
    JsonResponse,
    RealtimeResponse,
    RoutedResponse,
    TextResponse,
)

pytestmark: Final = [pytest.mark.timeout(180)]

PROXY_WORKERS: Final = max(1, int(os.environ.get("INTEGRATION_PROXY_WORKERS", "1")))
REALTIME_MODEL: Final = "gpt-realtime-2"
CHAT_MODEL: Final = "gpt-4o-mini"
DEPLOYMENT_KEY: Final = "synthetic-openai-realtime-key"
REALTIME_PATHS: Final = ("/v1/realtime", "/realtime", "/openai/v1/realtime")
OTHER_WEBSOCKET_PATHS: Final = (
    "/v1/realtime/translations",
    "/realtime/translations",
    "/v1/live/sessions",
    "/live/sessions",
    "/openai/v1/live/sessions",
    "/v1/responses",
    "/responses",
    "/openai_passthrough/v1/realtime",
    "/vertex_ai/live",
)
NON_CANONICAL_SCHEMES: Final = ("bearer", "BEARER", "bEaReR")
OFFER: Final = "v=0\r\no=- 1 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
ANSWER: Final = "v=0\r\no=- 3 4 IN IP4 203.0.113.7\r\ns=-\r\nt=0 0\r\n"
CALLS_STATUS: Final = 201
HANDSHAKE_REFUSED: Final = 403
HTTP_UNAUTHORIZED: Final = 401


class _ListedModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str


class _ModelList(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: tuple[_ListedModel, ...]


@dataclass(frozen=True, slots=True)
class Handshake:
    refused_with: int | None
    first_event: str | None


@dataclass(frozen=True, slots=True)
class RealtimeDeployment:
    model: str
    scenario_id: str


@dataclass(frozen=True, slots=True)
class WebrtcDeployment:
    model: str
    scenario_id: str
    raw_secret: str


@pytest.fixture
def proxy(gateway: Gateway) -> Gateway:
    return replace(gateway, workers=PROXY_WORKERS)


def _ws_base(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/").replace("http://", "ws://", 1)


async def _open(url: str, authorization: str) -> Handshake:
    try:
        async with websockets.connect(url, additional_headers={"Authorization": authorization}) as socket:
            first: Final = JSON_OBJECT.validate_json(await asyncio.wait_for(socket.recv(), 30))
            return Handshake(None, string_value(first["type"]))
    except InvalidStatus as refusal:
        return Handshake(refusal.response.status_code, None)


async def _upgrade(url: str, authorization: str) -> int | None:
    try:
        async with websockets.connect(url, additional_headers={"Authorization": authorization}, open_timeout=30):
            return None
    except InvalidStatus as refusal:
        return refusal.response.status_code


def _upgrade_refusal(gateway: Gateway, path: str, model: str, authorization: str) -> int | None:
    return asyncio.run(_upgrade(f"{_ws_base(gateway)}{path}?model={model}", authorization))


def _handshake(gateway: Gateway, path: str, model: str, authorization: str) -> Handshake:
    return asyncio.run(_open(f"{_ws_base(gateway)}{path}?model={model}", authorization))


def _realtime_deployment(gateway: Gateway, scenario: Scenario) -> RealtimeDeployment:
    scenario_id: Final = f"bearer-case-{uuid.uuid4().hex[:12]}"
    handle: Final = register_scenario(scenario_id, RealtimeResponse(content_type="application/x-realtime", events=()))
    scenario.cleanups.callback(delete_scenario, handle)
    model: Final = scenario.model(
        model=f"openai/{REALTIME_MODEL}", api_key=scenario_id, api_base=gateway.upstream_url.rstrip("/")
    )
    return RealtimeDeployment(model, scenario_id)


def _observed(gateway: Gateway) -> tuple[dict[str, JsonValue], ...]:
    with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream:
        return tuple(map(object_value, upstream.get("/__observations").json()["requests"]))


def _upgrades(gateway: Gateway, deployment: RealtimeDeployment) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        request
        for request in _observed(gateway)
        if request["method"] == "WEBSOCKET" and request["authorization"] == f"Bearer {deployment.scenario_id}"
    )


def _client_secret(raw: str) -> dict[str, JsonValue]:
    expires_at: Final = int(time.time()) + 600
    return {
        "value": raw,
        "expires_at": expires_at,
        "session": {
            "type": "realtime",
            "object": "realtime.session",
            "id": "sess_bearer_case",
            "model": REALTIME_MODEL,
            "client_secret": {"value": raw, "expires_at": expires_at},
        },
    }


def _webrtc_deployment(gateway: Gateway, scenario: Scenario) -> WebrtcDeployment:
    scenario_id: Final = f"bearer-case-{uuid.uuid4().hex[:12]}"
    raw: Final = f"ek_raw_{uuid.uuid4().hex}"
    routed: Final = RoutedResponse(
        content_type="application/x-routed",
        routes={
            "POST /v1/realtime/client_secrets": JsonResponse(content_type="application/json", body=_client_secret(raw)),
            "POST /v1/realtime/calls": TextResponse(content_type="application/jsonl", body=ANSWER, status=CALLS_STATUS),
        },
    )
    handle: Final = register_scenario(scenario_id, routed)
    scenario.cleanups.callback(delete_scenario, handle)
    model: Final = scenario.model(
        model=f"openai/{REALTIME_MODEL}",
        api_key=DEPLOYMENT_KEY,
        api_base=f"{gateway.upstream_url.rstrip('/')}/{scenario_id}/v1",
    )
    return WebrtcDeployment(model, scenario_id, raw)


def _mint(gateway: Gateway, prefix: str, deployment: WebrtcDeployment, key: str) -> str:
    minted: Final = gateway.request("POST", f"{prefix}/realtime/client_secrets", {"model": deployment.model}, key=key)
    assert minted.status_code == 200, minted.text
    return ClientSecretCreateResponse.model_validate_json(minted.content).value


def _redeem(gateway: Gateway, prefix: str, authorization: str) -> httpx.Response:
    return gateway.client.post(
        f"{prefix}/realtime/calls",
        content=OFFER.encode(),
        headers={"Content-Type": "application/sdp", "Authorization": authorization},
    )


def _listed_models(listed: httpx.Response) -> frozenset[str]:
    return frozenset(entry.id for entry in _ModelList.model_validate_json(listed.content).data)


def _calls(gateway: Gateway, deployment: WebrtcDeployment) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        request for request in _observed(gateway) if request["path"] == f"/{deployment.scenario_id}/v1/realtime/calls"
    )


@pytest.mark.parametrize("path", REALTIME_PATHS)
def test_lowercase_bearer_key_that_lists_models_also_opens_a_realtime_session(proxy: Gateway, path: str) -> None:
    with proxy.scenario() as scenario:
        deployment: Final = _realtime_deployment(proxy, scenario)
        authorization: Final = f"bearer {scenario.key()}"

        listed: Final = proxy.client.get("/v1/models", headers={"Authorization": authorization})
        assert listed.status_code == 200, listed.text
        handshake: Final = _handshake(proxy, path, deployment.model, authorization)

        assert handshake == Handshake(None, "session.created"), handshake
        assert [upgrade["path"] for upgrade in _upgrades(proxy, deployment)] == ["/v1/realtime"]


@pytest.mark.parametrize("scheme", ("BEARER", "bEaReR", "Bearer"))
def test_realtime_handshake_accepts_the_bearer_scheme_in_any_case(proxy: Gateway, scheme: str) -> None:
    with proxy.scenario() as scenario:
        deployment: Final = _realtime_deployment(proxy, scenario)
        handshake: Final = _handshake(proxy, "/v1/realtime", deployment.model, f"{scheme} {scenario.key()}")

        assert handshake == Handshake(None, "session.created"), handshake
        assert len(_upgrades(proxy, deployment)) == 1


@pytest.mark.parametrize("path", OTHER_WEBSOCKET_PATHS)
def test_every_websocket_route_upgrades_a_lowercase_bearer_key_like_the_canonical_one(
    proxy: Gateway, path: str
) -> None:
    with proxy.scenario() as scenario:
        deployment: Final = _realtime_deployment(proxy, scenario)
        key: Final = scenario.key()

        canonical: Final = _upgrade_refusal(proxy, path, deployment.model, f"Bearer {key}")
        lowercase: Final = _upgrade_refusal(proxy, path, deployment.model, f"bearer {key}")

        assert canonical is None, canonical
        assert lowercase is None, lowercase


def test_openai_sdk_async_realtime_session_opens_through_a_lowercase_bearer_gateway(proxy: Gateway) -> None:
    with proxy.scenario() as scenario:
        deployment: Final = _realtime_deployment(proxy, scenario)
        key: Final = scenario.key()

        async def connect() -> str:
            client: Final = AsyncOpenAI(
                api_key=key,
                base_url=f"{str(proxy.client.base_url).rstrip('/')}/v1",
                websocket_base_url=f"{_ws_base(proxy)}/v1",
            )
            async with client.realtime.connect(
                model=deployment.model, extra_headers={"Authorization": f"bearer {key}"}
            ) as connection:
                return (await connection.recv()).type

        assert asyncio.run(connect()) == "session.created"
        assert len(_upgrades(proxy, deployment)) == 1


def test_openai_sdk_sync_realtime_session_opens_through_a_lowercase_bearer_gateway(proxy: Gateway) -> None:
    with proxy.scenario() as scenario:
        deployment: Final = _realtime_deployment(proxy, scenario)
        key: Final = scenario.key()
        client: Final = OpenAI(
            api_key=key,
            base_url=f"{str(proxy.client.base_url).rstrip('/')}/v1",
            websocket_base_url=f"{_ws_base(proxy)}/v1",
        )
        with client.realtime.connect(
            model=deployment.model, extra_headers={"Authorization": f"bearer {key}"}
        ) as connection:
            first_event: Final = connection.recv().type

        assert first_event == "session.created"
        assert len(_upgrades(proxy, deployment)) == 1


def test_realtime_handshake_refuses_an_unknown_key_under_a_lowercase_scheme(proxy: Gateway) -> None:
    with proxy.scenario() as scenario:
        deployment: Final = _realtime_deployment(proxy, scenario)

        refused: Final = _handshake(proxy, "/v1/realtime", deployment.model, f"bearer sk-{uuid.uuid4().hex}")
        accepted: Final = _handshake(proxy, "/v1/realtime", deployment.model, f"bearer {scenario.key()}")

        assert refused == Handshake(HANDSHAKE_REFUSED, None), refused
        assert accepted == Handshake(None, "session.created"), accepted
        assert len(_upgrades(proxy, deployment)) == 1


@pytest.mark.parametrize("shape", ("Basic {key}", "Token {key}", "Bearer{key}", "bearer{key}", "bearer"))
def test_realtime_handshake_refuses_an_authorization_header_without_a_bearer_credential(
    proxy: Gateway, shape: str
) -> None:
    with proxy.scenario() as scenario:
        deployment: Final = _realtime_deployment(proxy, scenario)
        key: Final = scenario.key()

        refused: Final = _handshake(proxy, "/v1/realtime", deployment.model, shape.format(key=key))
        accepted: Final = _handshake(proxy, "/v1/realtime", deployment.model, f"Bearer {key}")

        assert refused == Handshake(HANDSHAKE_REFUSED, None), refused
        assert accepted == Handshake(None, "session.created"), accepted
        assert len(_upgrades(proxy, deployment)) == 1


@pytest.mark.parametrize("prefix", ("/v1", ""))
@pytest.mark.parametrize("scheme", NON_CANONICAL_SCHEMES)
def test_webrtc_call_redeems_the_client_secret_under_any_case_of_the_bearer_scheme(
    proxy: Gateway, scheme: str, prefix: str
) -> None:
    with proxy.scenario() as scenario:
        deployment: Final = _webrtc_deployment(proxy, scenario)
        token: Final = _mint(proxy, prefix, deployment, scenario.key(models=[deployment.model]))

        answered: Final = _redeem(proxy, prefix, f"{scheme} {token}")

        assert (answered.status_code, answered.text) == (CALLS_STATUS, ANSWER), answered.text
        calls: Final = _calls(proxy, deployment)
        assert [call["authorization"] for call in calls] == [f"Bearer {deployment.raw_secret}"], calls
        assert object_value(calls[0]["body"])["sdp"] == OFFER, calls


@pytest.mark.parametrize("authorization", ("bearer not-a-client-secret", "Basic {token}", "bearer{token}"))
def test_webrtc_call_refuses_a_header_that_does_not_carry_a_valid_client_secret(
    proxy: Gateway, authorization: str
) -> None:
    with proxy.scenario() as scenario:
        deployment: Final = _webrtc_deployment(proxy, scenario)
        token: Final = _mint(proxy, "/v1", deployment, scenario.key(models=[deployment.model]))

        refused: Final = _redeem(proxy, "/v1", authorization.format(token=token))
        answered: Final = _redeem(proxy, "/v1", f"bearer {token}")

        assert refused.status_code == HTTP_UNAUTHORIZED, refused.text
        assert (answered.status_code, answered.text) == (CALLS_STATUS, ANSWER), answered.text
        assert len(_calls(proxy, deployment)) == 1


@pytest.mark.parametrize("scheme", NON_CANONICAL_SCHEMES)
def test_http_routes_accept_the_bearer_scheme_in_any_case(proxy: Gateway, scheme: str) -> None:
    with proxy.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{CHAT_MODEL}")
        authorization: Final = {"Authorization": f"{scheme} {scenario.key()}"}

        listed: Final = proxy.client.get("/v1/models", headers=authorization)
        chat: Final = proxy.client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "bearer scheme case"}]},
            headers=authorization,
        )

        assert listed.status_code == 200, listed.text
        assert model in _listed_models(listed), listed.text
        assert chat.status_code == 200, chat.text


def test_a_doubled_bearer_scheme_keeps_authenticating_on_http_and_the_realtime_handshake(proxy: Gateway) -> None:
    with proxy.scenario() as scenario:
        deployment: Final = _realtime_deployment(proxy, scenario)
        authorization: Final = f"Bearer Bearer {scenario.key()}"

        listed: Final = proxy.client.get("/v1/models", headers={"Authorization": authorization})
        handshake: Final = _handshake(proxy, "/v1/realtime", deployment.model, authorization)

        assert listed.status_code == 200, listed.text
        assert handshake == Handshake(None, "session.created"), handshake
        assert len(_upgrades(proxy, deployment)) == 1


@pytest.mark.parametrize("scheme", NON_CANONICAL_SCHEMES)
def test_http_routes_refuse_an_unknown_key_under_any_case_of_the_bearer_scheme(proxy: Gateway, scheme: str) -> None:
    refused: Final = proxy.client.get("/v1/models", headers={"Authorization": f"{scheme} sk-{uuid.uuid4().hex}"})

    assert refused.status_code == HTTP_UNAUTHORIZED, refused.text
