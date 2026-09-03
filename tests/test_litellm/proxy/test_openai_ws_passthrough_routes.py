"""OpenAI passthrough must register WebSocket catch-all routes (#36088)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.routing import WebSocketRoute

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    openai_websocket_proxy_route,
    router,
)


def test_openai_websocket_passthrough_routes_registered():
    ws_paths = {route.path for route in router.routes if isinstance(route, WebSocketRoute)}
    assert "/openai/{endpoint:path}" in ws_paths
    assert "/openai_passthrough/{endpoint:path}" in ws_paths


def _mock_websocket(path: str, query: str, headers: dict[str, str] | None = None) -> MagicMock:
    websocket = MagicMock()
    websocket.url.path = path
    websocket.url.query = query
    websocket.headers = headers or {}
    websocket.accept = AsyncMock()
    websocket.close = AsyncMock()
    return websocket


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["openai", "openai_passthrough"])
async def test_openai_websocket_forwards_query_and_keeps_provider_auth(prefix):
    websocket = _mock_websocket(f"/{prefix}/v1/realtime", "model=gpt-4o-realtime-preview")

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
            return_value="sk-provider",
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._join_url_paths",
            return_value="https://api.openai.com/v1/realtime",
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.websocket_passthrough_request",
            new_callable=AsyncMock,
        ) as mock_ws,
    ):
        await openai_websocket_proxy_route(
            websocket=websocket,
            endpoint="v1/realtime",
            user_api_key_dict=UserAPIKeyAuth(),
        )

    kwargs = mock_ws.await_args.kwargs
    assert kwargs["target"] == "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
    assert kwargs["custom_headers"] == {"Authorization": "Bearer sk-provider"}
    assert kwargs["forward_headers"] is False
    assert kwargs["endpoint"] == f"/{prefix}/v1/realtime"
    assert kwargs["accept_websocket"] is False
    websocket.accept.assert_awaited_once_with(subprotocol=None)
    websocket.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_openai_websocket_accepts_first_client_subprotocol():
    websocket = _mock_websocket(
        "/openai/v1/realtime",
        "model=gpt-4o-realtime-preview",
        headers={
            "sec-websocket-protocol": "realtime, openai-insecure-api-key.sk-abc, openai-beta.realtime-v1"
        },
    )

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
            return_value="sk-provider",
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.websocket_passthrough_request",
            new_callable=AsyncMock,
        ) as mock_ws,
    ):
        await openai_websocket_proxy_route(
            websocket=websocket,
            endpoint="v1/realtime",
            user_api_key_dict=UserAPIKeyAuth(),
        )

    websocket.accept.assert_awaited_once_with(subprotocol="realtime")
    assert mock_ws.await_args.kwargs["accept_websocket"] is False
    websocket.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_openai_websocket_closes_cleanly_when_provider_credentials_missing():
    websocket = _mock_websocket("/openai/v1/realtime", "model=gpt-4o-realtime-preview")

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
            return_value=None,
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.websocket_passthrough_request",
            new_callable=AsyncMock,
        ) as mock_ws,
    ):
        await openai_websocket_proxy_route(
            websocket=websocket,
            endpoint="v1/realtime",
            user_api_key_dict=UserAPIKeyAuth(),
        )

    websocket.close.assert_awaited_once()
    assert websocket.close.await_args.kwargs["code"] == 1011
    websocket.accept.assert_not_awaited()
    mock_ws.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_api_key_dict",
    [
        UserAPIKeyAuth(models=["gpt-4o"]),
        UserAPIKeyAuth(team_models=["gpt-4o-realtime-preview"]),
        UserAPIKeyAuth(models=["all-team-models"], team_models=["gpt-4o"]),
    ],
)
async def test_openai_websocket_rejects_model_restricted_keys(user_api_key_dict):
    websocket = _mock_websocket("/openai/v1/realtime", "model=gpt-4o-realtime-preview")

    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.websocket_passthrough_request",
        new_callable=AsyncMock,
    ) as mock_ws:
        await openai_websocket_proxy_route(
            websocket=websocket,
            endpoint="v1/realtime",
            user_api_key_dict=user_api_key_dict,
        )

    websocket.close.assert_awaited_once()
    assert websocket.close.await_args.kwargs["code"] == 1008
    websocket.accept.assert_not_awaited()
    mock_ws.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_api_key_dict",
    [
        UserAPIKeyAuth(),
        UserAPIKeyAuth(models=["all-proxy-models"]),
        UserAPIKeyAuth(models=["*"]),
        UserAPIKeyAuth(models=["all-team-models"], team_models=["all-proxy-models"]),
    ],
)
async def test_openai_websocket_allows_unrestricted_keys(user_api_key_dict):
    websocket = _mock_websocket("/openai/v1/responses", "")

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
            return_value="sk-provider",
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.websocket_passthrough_request",
            new_callable=AsyncMock,
        ) as mock_ws,
    ):
        await openai_websocket_proxy_route(
            websocket=websocket,
            endpoint="v1/responses",
            user_api_key_dict=user_api_key_dict,
        )

    mock_ws.assert_awaited_once()
    websocket.close.assert_not_awaited()
