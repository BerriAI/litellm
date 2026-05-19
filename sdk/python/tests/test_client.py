"""Smoke tests for sdk.python.xct_litellm — pure Python, no proxy running.

We exercise the request layer with httpx's MockTransport so we don't need a
live server and can validate exact wire shape (URL, headers, body) per call.
"""

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from xct_litellm import (
    AuthError,
    CapabilityNotFoundError,
    PKCESession,
    XctClient,
    XctError,
)


def _make_client(
    handler, *, access_token="sk-x", app_id=None, base="https://proxy.x.test"
):
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, timeout=2.0)
    ahttp = httpx.AsyncClient(transport=transport, timeout=2.0)
    return XctClient(
        base_url=base,
        access_token=access_token,
        app_id=app_id,
        http_client=http,
        async_http_client=ahttp,
    )


def test_capabilities_list_sends_bearer_and_app_id():
    captured = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(200, json={"caller": {"is_admin": False}})

    client = _make_client(handler, access_token="sk-test", app_id="xct-chat")
    out = client.capabilities.list()
    assert out["caller"]["is_admin"] is False
    req = captured[0]
    assert str(req.url) == "https://proxy.x.test/v1/capabilities"
    assert req.headers["authorization"] == "Bearer sk-test"
    assert req.headers["x-xct-app-id"] == "xct-chat"


def test_agents_list_passes_query_params():
    captured = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(200, json=[])

    client = _make_client(handler)
    client.agents.list(q="research", category="writing", limit=10)
    qs = dict(httpx.QueryParams(captured[0].url.query.decode()))
    assert qs == {"q": "research", "category": "writing", "limit": "10"}


def test_chat_completions_create_posts_payload():
    captured = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = _make_client(handler)
    payload = {
        "model": "deepseek-v3.2",
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = client.chat.completions.create(**payload)
    assert out["choices"][0]["message"]["content"] == "ok"
    body = json.loads(captured[0].content.decode())
    assert body == payload


def test_401_raises_auth_error():
    def handler(req):
        return httpx.Response(401, json={"detail": "invalid token"})

    client = _make_client(handler)
    with pytest.raises(AuthError) as exc:
        client.capabilities.list()
    assert exc.value.status == 401
    assert "invalid token" in str(exc.value)


def test_404_raises_capability_not_found():
    def handler(req):
        return httpx.Response(404, json={"detail": "Agent 'x' not found"})

    client = _make_client(handler)
    with pytest.raises(CapabilityNotFoundError):
        client.agents.get("x")


def test_500_raises_generic_xct_error():
    def handler(req):
        return httpx.Response(500, json={"detail": "boom"})

    client = _make_client(handler)
    with pytest.raises(XctError) as exc:
        client.capabilities.list()
    assert exc.value.status == 500
    assert not isinstance(exc.value, AuthError)
    assert not isinstance(exc.value, CapabilityNotFoundError)


@pytest.mark.asyncio
async def test_async_capabilities():
    def handler(req):
        return httpx.Response(200, json={"ok": True})

    client = _make_client(handler)
    out = await client.capabilities.alist()
    assert out == {"ok": True}


# ---- PKCE helper -----------------------------------------------------------


def test_pkce_session_generates_deterministic_challenge():
    sess = PKCESession(
        client_id="xct_abc",
        redirect_uri="https://chat.xct.test/cb",
        code_verifier="dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk",
    )
    # RFC 7636 §A.1 known-good challenge for that verifier
    assert sess.code_challenge == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"


def test_pkce_authorize_url_carries_params():
    sess = PKCESession(
        client_id="xct_abc",
        redirect_uri="https://chat.xct.test/cb",
        state="s-1",
        code_verifier="dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk",
    )
    url = sess.authorize_url("https://proxy.x.test", scope="read")
    assert url.startswith("https://proxy.x.test/oauth/authorize?")
    qs = dict(httpx.QueryParams(url.split("?", 1)[1]))
    assert qs["client_id"] == "xct_abc"
    assert qs["redirect_uri"] == "https://chat.xct.test/cb"
    assert qs["response_type"] == "code"
    assert qs["state"] == "s-1"
    assert qs["code_challenge"] == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    assert qs["code_challenge_method"] == "S256"
    assert qs["scope"] == "read"


# ---- stream helpers (SSE) --------------------------------------------------


def test_stream_parses_sse_events():
    sse_body = (
        b"event: a2a.message\n"
        b'data: {"role":"assistant","content":"hi"}\n'
        b"\n"
        b"event: a2a.message\n"
        b'data: {"role":"assistant","content":" there"}\n'
        b"\n"
    )

    def handler(req):
        return httpx.Response(
            200,
            content=sse_body,
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler)
    events = list(
        client._stream(
            "POST", "/v1/chat/completions", json={"stream": True}, accept_sse=True
        )
    )
    assert events == [
        {"role": "assistant", "content": "hi"},
        {"role": "assistant", "content": " there"},
    ]


def test_stream_parses_ndjson_fallback():
    ndjson = (
        b'{"role":"assistant","content":"a"}\n' b'{"role":"assistant","content":"b"}\n'
    )

    def handler(req):
        return httpx.Response(
            200, content=ndjson, headers={"content-type": "application/x-ndjson"}
        )

    client = _make_client(handler)
    events = list(
        client._stream("POST", "/v1/a2a/x/message/send", json={}, accept_sse=False)
    )
    assert len(events) == 2
    assert events[0]["content"] == "a"
