import asyncio
import socket
from typing import Final

import aiohttp
import httpx
import pytest
from aiohttp import ClientSession

from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


def _closed_local_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


async def test_client_session_helper() -> None:
    transport: Final = AsyncHTTPHandler._create_aiohttp_transport()
    assert isinstance(transport, LiteLLMAiohttpTransport)
    session1: Final = transport._get_valid_client_session()
    assert isinstance(session1, ClientSession)
    assert session1.closed is False
    assert getattr(session1, "_loop") is asyncio.get_running_loop()
    session2: Final = transport._get_valid_client_session()
    assert session2 is session1
    await session1.close()


async def test_event_loop_robustness() -> None:
    transport: Final = AsyncHTTPHandler._create_aiohttp_transport()
    session: Final = transport._get_valid_client_session()
    assert isinstance(session, ClientSession)
    await session.close()
    session_after_close: Final = transport._get_valid_client_session()
    assert isinstance(session_after_close, ClientSession)
    assert session_after_close is not session
    assert session_after_close.closed is False
    transport.client = lambda: ClientSession()
    session_after_factory: Final = transport._get_valid_client_session()
    assert isinstance(session_after_factory, ClientSession)
    assert session_after_factory is not session_after_close
    assert session_after_factory.closed is False
    assert transport.client is session_after_factory
    await session_after_close.close()
    await session_after_factory.close()


@pytest.mark.parametrize(("ssl_verify", "expected_ssl"), [(False, False), (None, True)])
async def test_refused_connection_maps_to_httpx_connect_error(
    ssl_verify: bool | None, expected_ssl: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    transport: Final = AsyncHTTPHandler._create_aiohttp_transport(ssl_verify=ssl_verify)
    port: Final = _closed_local_port()
    request: Final = httpx.Request("GET", f"https://127.0.0.1:{port}/")
    try:
        with pytest.raises(httpx.ConnectError) as raised:
            await transport.handle_async_request(request)
    finally:
        await transport._get_valid_client_session().close()
    cause: Final = raised.value.__cause__
    assert isinstance(cause, aiohttp.ClientConnectorError)
    assert cause.ssl is expected_ssl
    assert (cause.host, cause.port) == ("127.0.0.1", port)
