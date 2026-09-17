from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["/live", "/v1/live", "/openai/v1/live"])
@pytest.mark.parametrize(
    ("method", "suffix"),
    [
        ("POST", ""),
        ("POST", "/opaque/fork"),
        ("POST", "/opaque/accept"),
        ("POST", "/opaque/reject"),
        ("POST", "/opaque/refer"),
        ("POST", "/opaque/hangup"),
        ("GET", "/opaque/content"),
    ],
)
async def test_public_live_http_routes_reach_live_auth_before_generic_passthrough(monkeypatch, prefix, method, suffix):
    from litellm.proxy import proxy_server
    from litellm.proxy.realtime_endpoints import live

    authenticate = AsyncMock(side_effect=HTTPException(401, "Live authentication required"))
    monkeypatch.setattr(live, "_auth", authenticate)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://proxy"
    ) as client:
        response = await client.request(
            method,
            prefix + "/sessions" + suffix,
            json={"session": {"model": "voice"}, "transport": {"type": "webrtc", "sdp": "offer"}},
        )
    assert response.status_code == 401
    assert "Live authentication required" in response.text
    authenticate.assert_awaited_once()


@pytest.mark.parametrize("prefix", ["/live", "/v1/live", "/openai/v1/live"])
@pytest.mark.parametrize("suffix", ["", "/opaque/attach", "/opaque/fork"])
def test_public_live_websockets_reach_live_auth_before_legacy_sideband(monkeypatch, prefix, suffix):
    from starlette.websockets import WebSocketDisconnect

    from litellm.proxy import proxy_server
    from litellm.proxy.realtime_endpoints import live

    authenticate = AsyncMock(side_effect=HTTPException(403, "Live authentication rejected"))
    monkeypatch.setattr(live, "_auth", authenticate)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    with TestClient(proxy_server.app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(prefix + "/sessions" + suffix, headers={"authorization": "Bearer test"}):
                pass
    authenticate.assert_awaited_once()
