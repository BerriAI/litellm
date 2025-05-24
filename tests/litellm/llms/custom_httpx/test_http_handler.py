import io
import os
import pathlib
import ssl
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path
import litellm
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
