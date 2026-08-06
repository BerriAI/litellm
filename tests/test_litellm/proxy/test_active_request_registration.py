from collections.abc import Mapping
from types import SimpleNamespace

import pytest
from fastapi import Request

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.hooks.active_request_registry import ActiveRequestRegistry
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
        self.calls.append({"registry_id": registry_id, "started_at": started_at, "call_type": call_type})
        return self.returned_id


def make_request() -> Request:
    return Request({"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []})


def make_proxy_logging(hook: object) -> SimpleNamespace:
    return SimpleNamespace(get_proxy_hook=lambda _: hook)


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
