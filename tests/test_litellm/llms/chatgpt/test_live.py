import json

import httpx
import pytest

from litellm.llms.chatgpt.live import LiveDeployment, LiveTransport, live_session_path


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["chatgpt", "openai"])
@pytest.mark.parametrize("status", [201, 403, 429, 503])
async def test_live_request_preserves_payload_status_and_selected_credentials(provider, status, chatgpt_tokens):
    payload = {
        "session": {"model": "deployment-model", "tools": [{"type": "function", "name": "lookup"}]},
        "transport": {"type": "webrtc", "sdp": "v=0\r\n"},
        "future_option": {"nested": [True, None, 3]},
    }

    def respond(request):
        assert request.url.path == "/custom/v1/live/sessions"
        assert request.headers["authorization"] == (
            "Bearer test-token-default" if provider == "chatgpt" else "Bearer deployment-key"
        )
        assert request.headers.get("chatgpt-account-id") == ("test-account-default" if provider == "chatgpt" else None)
        assert request.headers["x-gateway"] == "configured"
        assert request.headers["openai-beta"] == "feature=v1"
        assert "cookie" not in request.headers
        assert json.loads(request.content) == payload
        assert request.url.params.get_list("tag") == ["a +/&", "b"]
        assert request.url.params["gateway"] == "trusted"
        assert request.url.params["cursor"] == "opaque +/&"
        assert not {"model", "call_id", "session_id", "api_key"}.intersection(request.url.params)
        return httpx.Response(status, json={"result": "upstream"}, headers={"x-request-id": "provider-id"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        transport = LiveTransport(
            LiveDeployment(
                model="deployment-model",
                provider=provider,
                api_key="deployment-key",
                api_base="https://gateway.example/custom/v1/?gateway=base",
                extra_headers={"x-gateway": "configured", "Authorization": "bad", "ChatGPT-Account-Id": "bad"},
                extra_query={"gateway": "trusted", "tag": ("a +/&", "b"), "model": "bad", "session_id": "bad"},
            ),
            {"Authorization": "Bearer proxy-key", "Cookie": "private", "OpenAI-Beta": "feature=v1"},
            http_client=client,
        )
        response = await transport.request(
            "POST",
            "live/sessions",
            payload,
            {"gateway": "untrusted", "cursor": "opaque +/&", "call_id": "bad", "api_key": "bad"},
        )
        assert response.status_code == status
        assert response.json() == {"result": "upstream"}
        assert response.headers["x-request-id"] == "provider-id"
        assert not client.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["fork", "accept", "reject", "refer", "hangup", "content"])
async def test_live_all_http_operations(operation):
    def respond(request):
        assert request.url.path == f"/v1/live/sessions/sess_new-ID/{operation}"
        assert request.method == ("GET" if operation == "content" else "POST")
        assert request.url.params["output_format"] == "json"
        return httpx.Response(204)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        transport = LiveTransport(LiveDeployment("model", provider="openai", api_key="key"), {}, http_client=client)
        response = await transport.request(
            "GET" if operation == "content" else "POST",
            live_session_path("sess_new-ID", operation),
            query={"output_format": "json"},
        )
        assert response.status_code == 204


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["live/sessions", "live/sessions/sess_1/attach", "live/sessions/sess_1/fork"])
async def test_live_websocket_paths_bounds_and_auth(path, chatgpt_tokens):
    from websockets.asyncio.server import serve

    async def observe(connection):
        assert connection.request.headers["Authorization"] == "Bearer test-token-default"
        await connection.send(connection.request.path)

    async with serve(observe, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        transport = LiveTransport(
            LiveDeployment("model", api_base=f"http://127.0.0.1:{port}/v1", extra_query={"route": "a+&b"}),
            {"Authorization": "Bearer proxy-key"},
        )
        connection = await transport.connect(path, {"checkpoint": "opaque+value"})
        try:
            received = await connection.recv()
            url = httpx.URL(f"http://127.0.0.1{received}")
            assert url.path == f"/v1/{path}"
            assert url.params["route"] == "a+&b"
            assert url.params["checkpoint"] == "opaque+value"
        finally:
            await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "https://evil.example/live/sessions",
        "//evil.example/live/sessions",
        "live/sessions/../accept",
        "live/sessions/sess%2Fbad/accept",
        "live/sessions/sess%5Cbad/accept",
        "live/sessions/sess%252Fbad/accept",
        "live/sessions/%252e%252e/accept",
        "live/sessions/%2E%2E/accept",
        "live/sessions/sess%00bad/accept",
        "live/sessions/sess%0Abad/accept",
        "live/sessions/sess.foo?query/accept",
        "live/sessions/sess_1/accept?url=https://evil.example",
        "live/sessions/sess_1/accept#fragment",
        "live/sessions/sess_1/accept\n",
    ],
)
async def test_live_rejects_noncanonical_paths_before_network(path):
    transport = LiveTransport(LiveDeployment("model", provider="openai", api_key="key"), {})
    with pytest.raises(ValueError, match=r"(?:Invalid|Noncanonical) Live"):
        await transport.request("POST", path)
    with pytest.raises(ValueError, match=r"(?:Invalid|Noncanonical) Live"):
        await transport.connect(path)


@pytest.mark.parametrize(
    "session_id",
    ["../x", "sess/x", "sess%2Fx", "sess\\x", "sess%255cx", ".", "..", "%252e%252e", "", "sess\n", "sess\x00"],
)
def test_live_session_ids_cannot_inject_path_or_query(session_id):
    with pytest.raises(ValueError, match="Invalid Live session ID"):
        live_session_path(session_id, "content")


@pytest.mark.asyncio
@pytest.mark.parametrize("session_id", ["sess.foo", "sess-\u00f1\u4e2d", "sess?x#y", "sess 50%", "x" * 1024])
async def test_live_preserves_opaque_session_ids(session_id):
    from urllib.parse import quote

    def respond(request):
        assert request.url.raw_path == f"/v1/live/sessions/{quote(session_id, safe='')}/content".encode()
        assert request.url.params == httpx.QueryParams()
        return httpx.Response(200, json={"session_id": session_id})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        transport = LiveTransport(LiveDeployment("model", provider="openai", api_key="key"), {}, http_client=client)
        response = await transport.request("GET", live_session_path(session_id, "content"))
        assert response.json()["session_id"] == session_id


@pytest.mark.asyncio
async def test_live_does_not_redirect_credentials():
    def respond(request):
        assert request.url.host == "api.openai.com"
        return httpx.Response(307, headers={"location": "https://elsewhere.example/collect"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond), follow_redirects=True) as client:
        transport = LiveTransport(LiveDeployment("model", provider="openai", api_key="key"), {}, http_client=client)
        response = await transport.request("POST", "live/sessions", {})
        assert response.status_code == 307


@pytest.mark.asyncio
async def test_live_websocket_does_not_redirect_credentials():
    from websockets.asyncio.server import serve
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response

    async def unused(connection):
        pytest.fail("Redirected WebSocket must never open")

    def redirect(connection, request):
        assert request.headers["authorization"] == "Bearer deployment-key"
        return Response(307, "Temporary Redirect", Headers({"Location": "/elsewhere"}))

    async with serve(unused, "127.0.0.1", 0, process_request=redirect) as server:
        port = server.sockets[0].getsockname()[1]
        transport = LiveTransport(
            LiveDeployment(
                "model", provider="openai", api_key="deployment-key", api_base=f"http://127.0.0.1:{port}/v1"
            ),
            {},
        )
        with pytest.raises(InvalidStatus) as failure:
            await transport.connect("live/sessions")
        assert failure.value.response.status_code == 307


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "api_base",
    [
        "ftp://gateway.example/v1",
        "https://user:secret@gateway.example/v1",
        "https://gateway.example/v1#fragment",
        "not a url",
    ],
)
async def test_live_rejects_invalid_api_base_before_network(api_base):
    requests: list = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        transport = LiveTransport(
            LiveDeployment("deployment-model", provider="openai", api_key="deployment-key", api_base=api_base),
            {},
            http_client=client,
        )
        with pytest.raises(ValueError, match="Invalid Live API base"):
            await transport.request("POST", "live/sessions", {})
    assert requests == []
