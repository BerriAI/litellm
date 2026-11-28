"""Unit tests for ``ForwardedHostMiddleware`` (pure ASGI, no server needed)."""

import pytest

from litellm.proxy.middleware.forwarded_host_middleware import (
    ForwardedHostMiddleware,
    _authority,
    use_x_forwarded_host,
)


async def _echo(scope, receive, send):
    headers = dict(scope["headers"])
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"x-host", headers.get(b"host", b"")),
                (b"x-server", repr(scope["server"]).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": b""})


async def _run(middleware, headers, server=("10.0.0.5", 4000)):
    scope = {"type": "http", "headers": headers, "server": server, "path": "/ui"}
    sent = []

    async def send(message):
        sent.append(message)

    await middleware(scope, None, send)
    return scope, dict(sent[0]["headers"])


@pytest.mark.asyncio
async def test_rewrites_host_header_and_server():
    scope, out = await _run(
        ForwardedHostMiddleware(_echo),
        [(b"host", b"gw.example.com/base"), (b"x-forwarded-host", b"gw.example.com")],
    )
    assert out[b"x-host"] == b"gw.example.com"
    assert scope["server"] == ("gw.example.com", 4000)
    assert scope["path"] == "/ui"


@pytest.mark.asyncio
async def test_first_value_of_comma_list_wins():
    _, out = await _run(
        ForwardedHostMiddleware(_echo),
        [(b"host", b"internal"), (b"x-forwarded-host", b"a.example.com, b.example.com")],
    )
    assert out[b"x-host"] == b"a.example.com"


@pytest.mark.asyncio
async def test_path_and_query_stripped_and_port_kept():
    scope, out = await _run(
        ForwardedHostMiddleware(_echo),
        [(b"host", b"internal"), (b"x-forwarded-host", b"gw.example.com:8443/?x=1")],
    )
    assert out[b"x-host"] == b"gw.example.com:8443"
    assert scope["server"] == ("gw.example.com", 8443)


@pytest.mark.asyncio
async def test_header_ignored_when_absent():
    scope, out = await _run(ForwardedHostMiddleware(_echo), [(b"host", b"gw.example.com/base")])
    assert out[b"x-host"] == b"gw.example.com/base"
    assert scope["server"] == ("10.0.0.5", 4000)


def test_authority_helper():
    assert _authority(" a.example.com , b ") == "a.example.com"
    assert _authority("gw.example.com/base?x=1") == "gw.example.com"
    assert _authority("   ") == ""


def test_gate_env(monkeypatch):
    monkeypatch.delenv("USE_X_FORWARDED_HOST", raising=False)
    assert use_x_forwarded_host() is False
    monkeypatch.setenv("USE_X_FORWARDED_HOST", "1")
    assert use_x_forwarded_host() is True
