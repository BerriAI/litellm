import io
import os
import pathlib
import ssl
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest
from aiohttp import ClientSession, TCPConnector

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler


@pytest.mark.asyncio
async def test_ssl_security_level(monkeypatch):
    # Set environment variable for SSL security level
    monkeypatch.setenv("SSL_SECURITY_LEVEL", "DEFAULT@SECLEVEL=1")

    # Create async client with SSL verification disabled to isolate SSL context testing
    client = AsyncHTTPHandler(ssl_verify=False)

    # Get the transport (should be LiteLLMAiohttpTransport)
    transport = client.client._transport

    # Get the aiohttp ClientSession
    client_session = transport._get_valid_client_session()

    # Get the connector from the session
    connector = client_session.connector

    # Get the SSL context from the connector
    ssl_context = connector._ssl
    print("ssl_context", ssl_context)

    # Verify that the SSL context exists and has the correct cipher string
    assert isinstance(ssl_context, ssl.SSLContext)
    # Optionally, check the ciphers string if needed
    # assert "DEFAULT@SECLEVEL=1" in ssl_context.get_ciphers()


@pytest.mark.asyncio
async def test_force_ipv4_transport():
    """Test transport creation with force_ipv4 enabled"""
    litellm.force_ipv4 = True
    litellm.disable_aiohttp_transport = True

    transport = AsyncHTTPHandler._create_async_transport()

    # Should get an AsyncHTTPTransport
    assert isinstance(transport, httpx.AsyncHTTPTransport)
    # Verify IPv4 configuration through a request
    client = httpx.AsyncClient(transport=transport)
    try:
        response = await client.get("http://example.com")
        assert response.status_code == 200
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_ssl_context_transport():
    """Test transport creation with SSL context"""
    # Create a test SSL context
    ssl_context = ssl.create_default_context()

    transport = AsyncHTTPHandler._create_async_transport(ssl_context=ssl_context)
    assert transport is not None

    if isinstance(transport, LiteLLMAiohttpTransport):
        # Get the client session and verify SSL context is passed through
        client_session = transport._get_valid_client_session()
        assert isinstance(client_session, ClientSession)
        assert isinstance(client_session.connector, TCPConnector)
        # Verify the connector has SSL context set by checking if it's using SSL
        assert client_session.connector._ssl is not None


@pytest.mark.asyncio
async def test_aiohttp_disabled_transport():
    """Test transport creation with aiohttp disabled"""
    litellm.disable_aiohttp_transport = True
    litellm.force_ipv4 = False

    transport = AsyncHTTPHandler._create_async_transport()

    # Should get None when both aiohttp is disabled and force_ipv4 is False
    assert transport is None


@pytest.mark.asyncio
async def test_ssl_verification_with_aiohttp_transport():
    """
    Test aiohttp respects ssl_verify=False

    We validate that the ssl settings for a litellm transport match what a ssl verify=False aiohttp client would have.

    """
    import aiohttp

    # Create a test SSL context
    litellm_async_client = AsyncHTTPHandler(ssl_verify=False)

    transport_connector = (
        litellm_async_client.client._transport._get_valid_client_session().connector
    )
    print("transport_connector", transport_connector)
    print("transport_connector._ssl", transport_connector._ssl)

    aiohttp_session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(verify_ssl=False)
    )
    print("aiohttp_session", aiohttp_session)
    print("aiohttp_session._ssl", aiohttp_session.connector._ssl)

    # assert both litellm transport and aiohttp session have ssl_verify=False
    assert transport_connector._ssl == aiohttp_session.connector._ssl
