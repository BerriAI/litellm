from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Final, Literal

import httpx
import pytest
from integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value
from integration._support.process import owned_proxy_process
from integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from openai import APIStatusError, AsyncOpenAI, OpenAI
from pydantic import JsonValue

from litellm.proxy.openai_files_endpoints.common_utils import encode_file_id_with_model
from litellm.types.videos.utils import encode_video_id_with_provider
from tests.integration.cost_calculation.cost_tracking_case import JsonResponse, RoutedResponse


def _provider_error(status: int) -> dict[str, JsonValue]:
    return {
        "error": {
            "message": f"scripted provider status {status}",
            "type": "rate_limit_error"
            if status == 429
            else "server_error"
            if status >= 500
            else "invalid_request_error",
            "code": str(status),
        }
    }


class _ObservationBuffer:
    def __init__(self, upstream_url: str) -> None:
        self._url = upstream_url.rstrip("/")
        self._items: tuple[dict[str, JsonValue], ...] = ()

    def read(self) -> tuple[dict[str, JsonValue], ...]:
        with httpx.Client(timeout=10, trust_env=False) as client:
            payload: Final = JSON_OBJECT.validate_python(client.get(f"{self._url}/__observations").json())
        requests: Final = payload.get("requests")
        assert isinstance(requests, list)
        self._items = (*self._items, *(object_value(item) for item in requests if isinstance(item, dict)))
        return self._items

    def route(self, scenario_id: str, suffix: str) -> tuple[dict[str, JsonValue], ...]:
        return tuple(
            item
            for item in self._items
            if f"/{scenario_id}/" in str(item.get("path")) and str(item.get("path")).endswith(suffix)
        )


def _ready(gateway: Gateway, model: str) -> None:
    eventually(
        lambda: gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "model readiness"}]},
        ),
        lambda response: not (response.status_code == 400 and "Invalid model name" in response.text),
        seconds=30,
    )


def _assert_observation(gateway: Gateway, scenario_id: str, suffix: str) -> tuple[dict[str, JsonValue], ...]:
    buffer: Final = _ObservationBuffer(gateway.upstream_url)
    result: Final = eventually(
        buffer.read,
        lambda _items: len(buffer.route(scenario_id, suffix)) >= 1,
        seconds=20,
    )
    observations: Final = buffer.route(scenario_id, suffix)
    assert observations, result
    return observations


def _add_vector_store(
    gateway: Gateway,
    scenario: Scenario,
    vector_store_id: str,
    alias: str,
    handle: ScenarioHandle,
) -> None:
    created: Final = gateway.request(
        "POST",
        "/vector_store/new",
        {
            "vector_store_id": vector_store_id,
            "custom_llm_provider": "openai",
            "litellm_params": {"model": alias, "api_base": handle.api_base(), "api_key": handle.scenario_id},
        },
    )
    assert created.status_code == 200, created.text
    scenario.cleanups.callback(gateway.post, "/vector_store/delete", {"vector_store_id": vector_store_id})


def _assert_provider_error(response: httpx.Response, status: int) -> None:
    assert response.status_code == status, response.text
    body: Final = JSON_OBJECT.validate_python(response.json())
    error: Final = object_value(body["error"])
    assert str(error.get("code")) == str(status), response.text
    assert "scripted provider status" in str(error.get("message")), response.text


@pytest.mark.parametrize(
    "status", (400, 401, 404, 429, 500), ids=("bad-request", "unauthorized", "not-found", "rate-limit", "server-error")
)
def test_vector_store_lookup_preserves_provider_status(gateway: Gateway, status: int) -> None:
    with gateway.scenario() as scenario:
        vector_store_id: Final = f"vs-{uuid.uuid4().hex}"
        handle: Final = register_scenario(
            f"vector-{uuid.uuid4().hex}",
            RoutedResponse(
                content_type="application/x-routed",
                routes={
                    f"GET /vector_stores/{vector_store_id}": JsonResponse(
                        content_type="application/json",
                        status=status,
                        body=_provider_error(status),
                    )
                },
            ),
        )
        scenario.cleanups.callback(delete_scenario, handle)
        alias: Final = scenario.model(api_base=handle.api_base(), api_key=handle.scenario_id)
        _add_vector_store(gateway, scenario, vector_store_id, alias, handle)
        _ready(gateway, alias)
        response: Final = gateway.request("GET", f"/v1/vector_stores/{vector_store_id}")
        _assert_provider_error(response, status)
        _assert_observation(gateway, handle.scenario_id, f"/vector_stores/{vector_store_id}")


@pytest.mark.parametrize(
    ("provider_route", "path", "model_name", "status"),
    (
        ("GET /videos/video-id", "/v1/videos/video-id", "openai/gpt-4o-mini", 404),
        ("GET /v1/evals/eval-id", "/v1/evals/eval-id", "openai/gpt-4o-mini", 404),
        ("GET /v1/skills/skill-id", "/v1/skills/skill-id?beta=true", "anthropic/claude-3-5-haiku-20241022", 404),
        ("GET /v1/batch/jobs/batch-id", "/v1/batches/batch-id", "mistral/mistral-large-latest", 404),
        ("GET /v1/messages/batches/batch-id", "/v1/batches/batch-id", "anthropic/claude-3-5-haiku-20241022", 500),
    ),
    ids=("video", "eval", "skill", "mistral-batch", "anthropic-batch-gap"),
)
def test_model_scoped_lookup_returns_scripted_provider_404(
    gateway: Gateway,
    provider_route: str,
    path: str,
    model_name: str,
    status: int,
) -> None:
    with gateway.scenario() as scenario:
        route_path: Final = provider_route.partition(" ")[2]
        handle: Final = register_scenario(
            f"scoped-{uuid.uuid4().hex}",
            RoutedResponse(
                content_type="application/x-routed",
                routes={
                    provider_route: JsonResponse(content_type="application/json", status=404, body=_provider_error(404))
                },
            ),
        )
        scenario.cleanups.callback(delete_scenario, handle)
        alias: Final = scenario.model(model=model_name, api_base=handle.api_base(), api_key=handle.scenario_id)
        _ready(gateway, alias)
        request_path: Final = (
            f"/v1/videos/{encode_video_id_with_provider('video-id', 'openai', model_id=alias)}"
            if "/videos/" in route_path
            else f"/v1/batches/{encode_file_id_with_model('batch-id', alias, id_type='batch')}"
            if "/batches/" in route_path or "/batch/jobs/" in route_path
            else path
        )
        headers: Final = {"x-litellm-model": alias} if "/skills/" in request_path else {}
        params: Final = {"model": alias} if "/evals/" in request_path else None
        response: Final = gateway.request("GET", request_path, params=params, headers=headers)
        assert response.status_code == status, response.text
        body: Final = JSON_OBJECT.validate_python(response.json())
        message: Final = str(object_value(body["error"]).get("message"))
        assert (
            "Client error '404 Not Found'" in message and "/v1/messages/batches/batch-id" in message
            if status == 500
            else "scripted provider status 404" in message
        ), response.text
        suffix: Final = "/videos/video-id" if "/videos/" in route_path else route_path
        _assert_observation(gateway, handle.scenario_id, suffix)


def _sdk_file_lookup(gateway: Gateway, file_id: str) -> httpx.Response:
    with OpenAI(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0) as client:
        try:
            client.files.retrieve(file_id)
        except APIStatusError as error:
            return error.response
    pytest.fail("OpenAI SDK file lookup unexpectedly succeeded")


async def _async_sdk_file_lookup(gateway: Gateway, file_id: str) -> httpx.Response:
    async with AsyncOpenAI(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0) as client:
        try:
            await client.files.retrieve(file_id)
        except APIStatusError as error:
            return error.response
    pytest.fail("OpenAI async SDK file lookup unexpectedly succeeded")


@pytest.mark.parametrize("client_kind", ("sync", "async"), ids=("sync", "async"))
def test_openai_sdk_file_lookup_returns_head_provider_error(
    gateway: Gateway,
    client_kind: Literal["sync", "async"],
    tmp_path: Path,
) -> None:
    with gateway.scenario() as scenario:
        scenario_id: Final = f"d12-file-lookup-{client_kind}"
        handle: Final = register_scenario(
            scenario_id,
            RoutedResponse(
                content_type="application/x-routed",
                routes={
                    "GET /files/file-id": JsonResponse(
                        content_type="application/json",
                        status=404,
                        body=_provider_error(404),
                    )
                },
            ),
        )
        scenario.cleanups.callback(delete_scenario, handle)
        model: Final = f"audit-file-lookup-{uuid.uuid4().hex}"
        config: Final = tmp_path / f"d12-{client_kind}.yaml"
        config.write_text(
            json.dumps(
                {
                    "model_list": [
                        {
                            "model_name": model,
                            "litellm_params": {
                                "model": "openai/gpt-4o-mini",
                                "api_base": handle.api_base(),
                                "api_key": scenario_id,
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        with owned_proxy_process(
            gateway,
            tmp_path,
            {},
            config=config,
            workers=2,
        ) as owned:
            candidate: Final = Gateway(owned.gateway.client, owned.gateway.key, gateway.upstream_url)
            file_id: Final = encode_file_id_with_model("file-id", model)
            response: Final = (
                _sdk_file_lookup(candidate, file_id)
                if client_kind == "sync"
                else asyncio.run(_async_sdk_file_lookup(candidate, file_id))
            )
            expected: Final = {
                "error": {
                    "message": f"Error code: 404 - {_provider_error(404)}",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "404",
                }
            }
            assert response.status_code == 404, response.text
            assert response.json() == expected, response.text
            _assert_observation(candidate, scenario_id, "/files/file-id")
