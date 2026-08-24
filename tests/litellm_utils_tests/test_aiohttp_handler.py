import asyncio
import copy
import time
from datetime import datetime
from unittest import mock

import httpx
from aiohttp import ClientSession
from dotenv import load_dotenv

from litellm.types.utils import StandardCallbackDynamicParams

load_dotenv()

import pytest

import litellm
from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


@pytest.mark.asyncio
async def test_client_session_helper():
    """Test that the client session helper handles event loop changes correctly"""
    transport = AsyncHTTPHandler._create_aiohttp_transport()
    assert isinstance(transport, LiteLLMAiohttpTransport)

    session1 = transport._get_valid_client_session()
    assert isinstance(session1, ClientSession)
    assert session1.closed is False
    assert getattr(session1, "_loop") is asyncio.get_running_loop()

    # Within the same event loop the valid session is reused, not rebuilt
    session2 = transport._get_valid_client_session()
    assert session2 is session1

    await session1.close()


async def test_event_loop_robustness():
    """Test behavior when event loops change (simulating CI/CD scenario)"""
    transport = AsyncHTTPHandler._create_aiohttp_transport()

    session = transport._get_valid_client_session()
    assert isinstance(session, ClientSession)

    # A closed session must be replaced with a live one bound to this loop
    await session.close()
    session_after_close = transport._get_valid_client_session()
    assert isinstance(session_after_close, ClientSession)
    assert session_after_close is not session
    assert session_after_close.closed is False

    # A client that is a factory rather than a session must also be rebuilt
    transport.client = lambda: ClientSession()  # type: ignore[assignment]
    session_after_factory = transport._get_valid_client_session()
    assert isinstance(session_after_factory, ClientSession)
    assert session_after_factory is not session_after_close
    assert session_after_factory.closed is False
    assert transport.client is session_after_factory

    await session_after_close.close()
    await session_after_factory.close()


async def test_httpx_request_simulation():
    """Test that the transport can handle a simulated HTTP request"""
    transport = AsyncHTTPHandler._create_aiohttp_transport(ssl_verify=False)
    request = httpx.Request("GET", "https://httpbin.org/headers")

    # The per-request SSL override the request path reads must reflect ssl_verify
    assert transport._ssl_verify is False

    session = transport._get_valid_client_session()
    assert isinstance(session, ClientSession)
    assert session.closed is False
    assert callable(session.request)
    assert session.connector is not None
    assert session.connector._ssl is False

    with mock.patch.object(
        transport, "_make_aiohttp_request", new=mock.AsyncMock(side_effect=RuntimeError("boom"))
    ) as mocked_request:
        with pytest.raises(RuntimeError):
            await transport.handle_async_request(request)

    assert mocked_request.call_count == 1
    call_kwargs = mocked_request.call_args.kwargs
    assert call_kwargs["request"] is request
    assert call_kwargs["ssl_verify"] is False
    assert call_kwargs["client_session"] is session

    await session.close()
