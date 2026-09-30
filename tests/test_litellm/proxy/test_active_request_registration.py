import json
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import Request

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.hooks.active_request_registry import ActiveRequestRegistry
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    azure_proxy_route,
    vllm_proxy_route,
)
from litellm.router import Router
from litellm.types.utils import CallTypesLiteral


class RecordingRegistry(ActiveRequestRegistry):
    """Real registry with Redis replaced by a recorder, so the call contract is exercised."""

    def __init__(self, returned_id: str | None) -> None:
        super().__init__(SimpleNamespace(dual_cache=SimpleNamespace(redis_cache=None)))
        self.returned_id = returned_id
        self.calls: list[dict[str, object]] = []

    async def register(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: Mapping[str, object],
        call_type: CallTypesLiteral,
        registry_id: str | None = None,
        started_at: float | None = None,
    ) -> str | None:
        self.calls.append(
            {"registry_id": registry_id, "started_at": started_at, "call_type": call_type, "data": dict(data)}
        )
        return self.returned_id


def make_request() -> Request:
    return Request({"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []})


def make_proxy_logging(hook: object) -> SimpleNamespace:
    return SimpleNamespace(get_proxy_hook=lambda _: hook)


def make_body_request(path: str, body: dict[str, object], extra_headers: Sequence[tuple[bytes, bytes]] = ()) -> Request:
    """A Request the route handlers can actually read a body off, so the router branch is real."""
    payload = json.dumps(body).encode()

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": payload, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [(b"content-type", b"application/json"), *extra_headers],
            "query_string": b"",
            "state": {},
        },
        receive,
    )


def make_router(model_name: str) -> Router:
    router = Router(
        model_list=[
            {
                "model_name": model_name,
                "litellm_params": {"model": "hosted_vllm/upstream", "api_base": "http://127.0.0.1:9", "api_key": "k"},
            }
        ]
    )
    router.allm_passthrough_route = AsyncMock(return_value=httpx.Response(200, json={"ok": True}))
    return router


@pytest.mark.asyncio
async def test_repeated_pre_call_registration_reuses_registry_identity_and_start_time():
    processor = ProxyBaseLLMRequestProcessing(data={"litellm_call_id": "call-1", "model": "model-a"})
    registry = RecordingRegistry(returned_id="registry-1")
    proxy_logging = make_proxy_logging(registry)
    request = make_request()
    auth = UserAPIKeyAuth(api_key="key-1")

    await processor._register_active_request(request, auth, proxy_logging, "acompletion")
    first_started_at = request.state.active_request_started_at
    await processor._register_active_request(request, auth, proxy_logging, "acompletion")

    assert len(registry.calls) == 2
    assert registry.calls[0]["registry_id"] is None
    assert registry.calls[1]["registry_id"] == "registry-1"
    assert registry.calls[1]["started_at"] == first_started_at


@pytest.mark.asyncio
async def test_failed_refresh_keeps_registry_identity_for_cleanup():
    processor = ProxyBaseLLMRequestProcessing(data={"litellm_call_id": "call-1"})
    proxy_logging = make_proxy_logging(RecordingRegistry(returned_id=None))
    request = make_request()
    request.state.active_request_registry_id = "registry-1"

    await processor._register_active_request(request, UserAPIKeyAuth(api_key="key-1"), proxy_logging, "acompletion")

    assert request.state.active_request_registry_id == "registry-1"


@pytest.mark.asyncio
async def test_unrelated_hook_is_ignored_without_affecting_requests():
    processor = ProxyBaseLLMRequestProcessing(data={"litellm_call_id": "call-1"})
    proxy_logging = make_proxy_logging(SimpleNamespace(register=lambda **_: None))
    request = make_request()

    await processor._register_active_request(request, UserAPIKeyAuth(api_key="key-1"), proxy_logging, "acompletion")

    assert not hasattr(request.state, "active_request_registry_id")
    assert not hasattr(request.state, "active_request_registry")


@pytest.mark.parametrize("route_type", ["_arealtime", "_aresponses_websocket"])
@pytest.mark.asyncio
async def test_websocket_routes_are_not_registered(route_type):
    """Those routes hand in a synthetic http Request, so the middleware never sees them close."""
    processor = ProxyBaseLLMRequestProcessing(data={"litellm_call_id": "call-1"})
    registry = RecordingRegistry(returned_id="registry-1")
    request = make_request()

    await processor._register_active_request(
        request, UserAPIKeyAuth(api_key="key-1"), make_proxy_logging(registry), route_type
    )

    assert registry.calls == []
    assert not hasattr(request.state, "active_request_registry")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_vllm_router_model_passthrough_is_registered(stream):
    """The router branch returns before pass_through_request, so it has to register itself."""
    registry = RecordingRegistry(returned_id="registry-vllm")
    router = make_router("router-model")
    request = make_body_request("/vllm/chat/completions", {"model": "router-model", "stream": stream})

    with (
        patch("litellm.proxy.proxy_server.llm_router", router),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", make_proxy_logging(registry)),
    ):
        await vllm_proxy_route(
            endpoint="/chat/completions",
            request=request,
            fastapi_response=SimpleNamespace(headers={}),
            user_api_key_dict=UserAPIKeyAuth(api_key="key-1"),
        )

    router.allm_passthrough_route.assert_awaited_once()
    assert request.state.active_request_registry_id == "registry-vllm"
    assert len(registry.calls) == 1
    assert registry.calls[0]["call_type"] == "allm_passthrough_route"
    assert registry.calls[0]["data"]["model"] == "router-model"
    assert registry.calls[0]["data"]["stream"] is stream
    # One id for the overview row and the log entry, so the row can be looked up.
    assert (
        router.allm_passthrough_route.await_args.kwargs["litellm_call_id"]
        == registry.calls[0]["data"]["litellm_call_id"]
    )


@pytest.mark.asyncio
async def test_azure_router_model_passthrough_is_registered():
    registry = RecordingRegistry(returned_id="registry-azure")
    router = make_router("router-model")
    endpoint = "openai/deployments/router-model/chat/completions"
    request = make_body_request(f"/azure/{endpoint}", {"messages": []})

    with (
        patch("litellm.proxy.proxy_server.llm_router", router),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", make_proxy_logging(registry)),
    ):
        await azure_proxy_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=SimpleNamespace(headers={}),
            user_api_key_dict=UserAPIKeyAuth(api_key="key-1"),
        )

    router.allm_passthrough_route.assert_awaited_once()
    assert request.state.active_request_registry_id == "registry-azure"
    assert len(registry.calls) == 1
    assert registry.calls[0]["call_type"] == "allm_passthrough_route"
    # Azure carries the model in the url, not in the body.
    assert registry.calls[0]["data"]["model"] == "router-model"
    assert (
        router.allm_passthrough_route.await_args.kwargs["litellm_call_id"]
        == registry.calls[0]["data"]["litellm_call_id"]
    )


@pytest.mark.asyncio
async def test_router_passthrough_keeps_the_client_supplied_call_id():
    registry = RecordingRegistry(returned_id="registry-vllm")
    router = make_router("router-model")
    request = make_body_request(
        "/vllm/chat/completions",
        {"model": "router-model"},
        extra_headers=((b"x-litellm-call-id", b"caller-supplied-id"),),
    )

    with (
        patch("litellm.proxy.proxy_server.llm_router", router),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", make_proxy_logging(registry)),
    ):
        await vllm_proxy_route(
            endpoint="/chat/completions",
            request=request,
            fastapi_response=SimpleNamespace(headers={}),
            user_api_key_dict=UserAPIKeyAuth(api_key="key-1"),
        )

    assert registry.calls[0]["data"]["litellm_call_id"] == "caller-supplied-id"
