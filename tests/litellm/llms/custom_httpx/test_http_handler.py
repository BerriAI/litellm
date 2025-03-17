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

    # Get the SSL context from the client
    ssl_context = client.client._transport._pool._ssl_context

    # Verify that the SSL context exists and has the correct cipher string
    assert isinstance(ssl_context, ssl.SSLContext)
