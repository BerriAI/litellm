"""OpenAI passthrough must register WebSocket catch-all routes (#36088)."""

from starlette.routing import WebSocketRoute
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    openai_websocket_proxy_route,
    router,
)


def test_openai_websocket_passthrough_routes_registered():
    ws_paths = {
        route.path
        for route in router.routes
        if isinstance(route, WebSocketRoute)
    }
    assert "/openai/{endpoint:path}" in ws_paths
    assert "/openai_passthrough/{endpoint:path}" in ws_paths


@pytest.mark.asyncio
async def test_openai_websocket_forwards_query_and_keeps_provider_auth():
    websocket = MagicMock()
    websocket.url.query = "model=gpt-4o-realtime-preview"
    websocket.close = AsyncMock()
    user = MagicMock()

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
            return_value="sk-provider",
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.BaseOpenAIPassThroughHandler._join_url_paths",
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
            user_api_key_dict=user,
        )

    kwargs = mock_ws.await_args.kwargs
    assert kwargs["target"] == "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
    assert kwargs["custom_headers"] == {"Authorization": "Bearer sk-provider"}
    assert kwargs["forward_headers"] is False
