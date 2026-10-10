"""Backend websocket dials carry a TLS context only when the resolved URL is ``wss://``.

An Azure or OpenAI realtime session and a Responses websocket session each open one backend socket. A
plaintext ``api_base`` (``http://`` or ``ws://``) resolves to ``ws://``, where the websockets client refuses
any ``ssl`` argument, and ``https://`` resolves to ``wss://``, where ``litellm_settings.ssl_verify: false``
has to become a context that skips verification rather than ``False``, which the client rejects. Every row
runs against the scripted upstream on 127.0.0.1 and asserts the upgrade it saw, so a dial that never left
the proxy fails on the observation, not only on the client's close.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Final

import httpx
import pytest
import websockets
import yaml
from pydantic import JsonValue
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed

from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from tests.integration._support.process import UpstreamSlot, owned_proxy, owned_upstream
from tests.integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from tests.integration.cost_calculation.cost_tracking_case import RealtimeResponse

pytestmark: Final = pytest.mark.timeout(240)

REALTIME_MODEL: Final = "gpt-realtime"
RESPONSES_MODEL: Final = "gpt-5.6"
API_VERSION: Final = "2025-04-01-preview"
PLAINTEXT_SCHEMES: Final = ("http", "ws")
AZURE_REALTIME_PATHS: Final = frozenset({"/openai/realtime", "/openai/v1/realtime"})
OPENAI_REALTIME_PATH: Final = "/v1/realtime"
RESPONSES_DEPLOYMENTS: Final = (("openai", "/v1", "/v1/responses"), ("azure", "", "/openai/v1/responses"))
RESPONSES_TERMINAL: Final = frozenset({"response.completed", "response.failed", "error"})
COMPLETED_REALTIME_TURN: Final = ("session.created", "response.done")
COMPLETED_RESPONSES_TURN: Final = ("response.created", "response.output_text.delta", "response.completed")
CONVERGENCE_SECONDS: Final = 30
REALTIME_USAGE: Final[dict[str, JsonValue]] = {
    "total_tokens": 12,
    "input_tokens": 7,
    "output_tokens": 5,
    "input_token_details": {"text_tokens": 7, "audio_tokens": 0, "cached_tokens": 0},
    "output_token_details": {"text_tokens": 5, "audio_tokens": 0},
}
RESPONSES_USAGE: Final[dict[str, JsonValue]] = {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}


def _realtime_turn() -> RealtimeResponse:
    return RealtimeResponse(
        content_type="application/x-realtime",
        session_model=REALTIME_MODEL,
        events=(
            {
                "type": "response.done",
                "event_id": "evt_$UNIQUE_ID",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "realtime.response",
                    "status": "completed",
                    "output": [],
                    "usage": REALTIME_USAGE,
                },
            },
        ),
    )


def _responses_turn() -> RealtimeResponse:
    response: Final[dict[str, JsonValue]] = {
        "id": "resp_$REQUEST_ID",
        "object": "response",
        "created_at": 1700000000,
        "model": RESPONSES_MODEL,
    }
    message: Final[dict[str, JsonValue]] = {
        "type": "message",
        "id": "msg_$REQUEST_ID",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "scripted $REQUEST_ID", "annotations": []}],
    }
    return RealtimeResponse(
        content_type="application/x-realtime",
        events=(
            {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
            {
                "type": "response.output_text.delta",
                "item_id": "msg_$REQUEST_ID",
                "output_index": 0,
                "content_index": 0,
                "delta": "scripted $REQUEST_ID",
            },
            {
                "type": "response.completed",
                "response": {**response, "status": "completed", "output": [message], "usage": RESPONSES_USAGE},
            },
        ),
    )


def _scripted(scenario: Scenario, response: RealtimeResponse, *, control_url: str) -> ScenarioHandle:
    handle: Final = register_scenario(f"ws-dial-{uuid.uuid4().hex[:12]}", response, control_url=control_url)
    scenario.cleanups.callback(delete_scenario, handle)
    return handle


def _ws_base(http_url: str) -> str:
    return http_url.replace("https://", "wss://").replace("http://", "ws://")


def _with_scheme(http_url: str, scheme: str) -> str:
    return f"{scheme}://{http_url.removeprefix('http://')}"


def _proxy_ws(gateway: Gateway) -> str:
    return _ws_base(str(gateway.client.base_url).rstrip("/"))


def _upstream_url(gateway: Gateway) -> str:
    return gateway.upstream_url.rstrip("/")


def _realtime_deployment(scenario: Scenario, provider: str, scenario_id: str, api_base: str) -> str:
    return scenario.model(
        model=f"{provider}/{REALTIME_MODEL}",
        api_key=scenario_id,
        api_base=api_base,
        api_version=API_VERSION,
        model_info={"mode": "realtime"},
    )


def _responses_deployment(scenario: Scenario, provider: str, scenario_id: str, api_base: str) -> str:
    return scenario.model(
        model=f"{provider}/{RESPONSES_MODEL}",
        api_key=scenario_id,
        api_base=api_base,
        **({"api_version": API_VERSION} if provider == "azure" else {}),
    )


def _close_code(closed: ConnectionClosed) -> int:
    return 1006 if closed.rcvd is None else closed.rcvd.code


async def _realtime_frames(socket: ClientConnection) -> AsyncIterator[dict[str, JsonValue]]:
    try:
        first: Final = JSON_OBJECT.validate_json(await socket.recv())
        yield first
        if first.get("type") != "session.created":
            async for message in socket:
                yield JSON_OBJECT.validate_json(message)
            return
        await socket.send(json.dumps({"type": "response.create"}))
        async for message in socket:
            event: Final = JSON_OBJECT.validate_json(message)
            yield event
            if event.get("type") == "response.done":
                break
    except ConnectionClosed as closed:
        yield {"type": "closed", "code": _close_code(closed)}


def _create_frame(model: str) -> str:
    return json.dumps(
        {
            "type": "response.create",
            "model": model,
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "say it"}]}],
        }
    )


async def _responses_frames(socket: ClientConnection, model: str) -> AsyncIterator[dict[str, JsonValue]]:
    try:
        await socket.send(_create_frame(model))
        async for message in socket:
            event: Final = JSON_OBJECT.validate_json(message)
            yield event
            if event.get("type") in RESPONSES_TERMINAL:
                break
    except ConnectionClosed as closed:
        yield {"type": "closed", "code": _close_code(closed)}


async def _collect(frames: AsyncIterator[dict[str, JsonValue]]) -> tuple[dict[str, JsonValue], ...]:
    return tuple([frame async for frame in frames])


async def _realtime_session(ws_base: str, model: str, key: str) -> tuple[dict[str, JsonValue], ...]:
    headers: Final = {"Authorization": f"Bearer {key}"}
    async with websockets.connect(f"{ws_base}/v1/realtime?model={model}", additional_headers=headers) as socket:
        return await asyncio.wait_for(_collect(_realtime_frames(socket)), 60)


async def _responses_session(ws_base: str, model: str, key: str) -> tuple[dict[str, JsonValue], ...]:
    headers: Final = {"Authorization": f"Bearer {key}"}
    async with websockets.connect(f"{ws_base}/v1/responses?model={model}", additional_headers=headers) as socket:
        return await asyncio.wait_for(_collect(_responses_frames(socket, model)), 60)


def _types(events: tuple[dict[str, JsonValue], ...]) -> tuple[str, ...]:
    return tuple(string_value(event["type"]) for event in events)


def _observations(control_url: str) -> tuple[dict[str, JsonValue], ...]:
    verify: Final = not control_url.startswith("https://")
    with httpx.Client(base_url=control_url, timeout=5, trust_env=False, verify=verify) as upstream:
        return tuple(map(object_value, upstream.get("/__observations").json()["requests"]))


def _dialed_by(request: dict[str, JsonValue], scenario_id: str) -> bool:
    bearer: Final = string_value(request["authorization"]).removeprefix("Bearer ")
    return request["method"] == "WEBSOCKET" and scenario_id in (bearer, request["api_key"])


def _upgrade_paths(control_url: str, scenario_id: str) -> tuple[str, ...]:
    observed: Final = _observations(control_url)
    return tuple(string_value(request["path"]) for request in observed if _dialed_by(request, scenario_id))


def _assert_azure_realtime_upgrade(control_url: str, scenario_id: str) -> None:
    paths: Final = _upgrade_paths(control_url, scenario_id)
    assert len(paths) == 1 and paths[0] in AZURE_REALTIME_PATHS, paths


def _health(gateway: Gateway, model: str) -> dict[str, JsonValue]:
    response: Final = gateway.request("GET", "/health", params={"model": model})
    return JSON_OBJECT.validate_json(response.content)


def _probed(body: dict[str, JsonValue]) -> int:
    counts: Final = (body.get("healthy_count", 0), body.get("unhealthy_count", 0))
    return sum(count for count in counts if isinstance(count, int))


def _converged_health(gateway: Gateway, model: str) -> dict[str, JsonValue]:
    return eventually(lambda: _health(gateway, model), lambda body: _probed(body) == 1, seconds=CONVERGENCE_SECONDS)


@pytest.mark.parametrize("scheme", PLAINTEXT_SCHEMES)
def test_azure_realtime_session_over_a_plaintext_api_base(gateway: Gateway, scheme: str) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _realtime_turn(), control_url=_upstream_url(gateway))
        api_base: Final = _with_scheme(_upstream_url(gateway), scheme)
        model: Final = _realtime_deployment(scenario, "azure", handle.scenario_id, api_base)
        key: Final = scenario.key(models=[model])
        events: Final = asyncio.run(_realtime_session(_proxy_ws(gateway), model, key))
        assert _types(events) == COMPLETED_REALTIME_TURN, events
        _assert_azure_realtime_upgrade(_upstream_url(gateway), handle.scenario_id)


@pytest.mark.parametrize("scheme", PLAINTEXT_SCHEMES)
def test_azure_realtime_health_check_over_a_plaintext_api_base(gateway: Gateway, scheme: str) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _realtime_turn(), control_url=_upstream_url(gateway))
        api_base: Final = _with_scheme(_upstream_url(gateway), scheme)
        model: Final = _realtime_deployment(scenario, "azure", handle.scenario_id, api_base)
        health: Final = _converged_health(gateway, model)
        assert (health["healthy_count"], health["unhealthy_count"]) == (1, 0), health
        _assert_azure_realtime_upgrade(_upstream_url(gateway), handle.scenario_id)


@pytest.mark.parametrize(("provider", "suffix", "path"), RESPONSES_DEPLOYMENTS, ids=["openai", "azure"])
def test_responses_websocket_session_over_a_plaintext_api_base(
    gateway: Gateway, provider: str, suffix: str, path: str
) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _responses_turn(), control_url=_upstream_url(gateway))
        model: Final = _responses_deployment(scenario, provider, handle.scenario_id, _upstream_url(gateway) + suffix)
        key: Final = scenario.key(models=[model])
        events: Final = asyncio.run(_responses_session(_proxy_ws(gateway), model, key))
        assert _types(events) == COMPLETED_RESPONSES_TURN, events
        assert _upgrade_paths(_upstream_url(gateway), handle.scenario_id) == (path,)


def _unverified_config(directory: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    unverified: Final = {**config, "litellm_settings": {**config["litellm_settings"], "ssl_verify": False}}
    path: Final = directory / "ssl-verify-off.yaml"
    path.write_text(yaml.safe_dump(unverified))
    return path


@pytest.fixture(scope="module")
def tls_upstream(tmp_path_factory: pytest.TempPathFactory) -> Iterator[UpstreamSlot]:
    with owned_upstream(tmp_path_factory.mktemp("ws-dial-tls-upstream"), tls=True) as slot:
        yield slot


@pytest.fixture(scope="module")
def unverified_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("ws-dial-unverified-proxy")
    with (
        gateway_from_environment() as base,
        owned_proxy(base, directory, {}, config=_unverified_config(directory)) as proxy,
    ):
        yield proxy


@pytest.mark.parametrize("provider", ["azure", "openai"])
def test_realtime_session_with_verification_off_over_a_self_signed_upstream(
    unverified_proxy: Gateway, tls_upstream: UpstreamSlot, provider: str
) -> None:
    with unverified_proxy.scenario() as scenario:
        handle: Final = _scripted(scenario, _realtime_turn(), control_url=tls_upstream.url)
        model: Final = _realtime_deployment(scenario, provider, handle.scenario_id, tls_upstream.url)
        key: Final = scenario.key(models=[model])
        events: Final = asyncio.run(_realtime_session(_proxy_ws(unverified_proxy), model, key))
        assert _types(events) == COMPLETED_REALTIME_TURN, events
        paths: Final = _upgrade_paths(tls_upstream.url, handle.scenario_id)
        assert len(paths) == 1 and paths[0] in (AZURE_REALTIME_PATHS | {OPENAI_REALTIME_PATH}), paths


def test_responses_websocket_session_with_verification_off_over_a_self_signed_upstream(
    unverified_proxy: Gateway, tls_upstream: UpstreamSlot
) -> None:
    with unverified_proxy.scenario() as scenario:
        handle: Final = _scripted(scenario, _responses_turn(), control_url=tls_upstream.url)
        model: Final = _responses_deployment(scenario, "openai", handle.scenario_id, f"{tls_upstream.url}/v1")
        key: Final = scenario.key(models=[model])
        events: Final = asyncio.run(_responses_session(_proxy_ws(unverified_proxy), model, key))
        assert _types(events) == COMPLETED_RESPONSES_TURN, events
        assert _upgrade_paths(tls_upstream.url, handle.scenario_id) == ("/v1/responses",)
