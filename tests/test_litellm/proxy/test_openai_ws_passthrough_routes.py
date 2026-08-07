"""OpenAI passthrough must register WebSocket catch-all routes (#36088)."""

from starlette.routing import WebSocketRoute

from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import router


def test_openai_websocket_passthrough_routes_registered():
    ws_paths = {
        route.path
        for route in router.routes
        if isinstance(route, WebSocketRoute)
    }
    assert "/openai/{endpoint:path}" in ws_paths
    assert "/openai_passthrough/{endpoint:path}" in ws_paths
