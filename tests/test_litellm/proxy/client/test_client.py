import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.client import ChatClient, Client, ModelsManagementClient
from litellm.proxy.client.http_client import HTTPClient
from litellm.proxy.client.keys import KeysManagementClient


@pytest.fixture
def base_url():
    return "http://localhost:8000"


@pytest.fixture
def api_key():
    return "test-api-key"


def test_client_initialization(base_url, api_key):
    """Test that the Client is properly initialized with all resource clients"""
    client = Client(base_url=base_url, api_key=api_key)

    # Check base properties
    assert client._base_url == base_url
    assert client._api_key == api_key

    # Check resource clients
    assert isinstance(client.models, ModelsManagementClient)
    assert client.models._base_url == base_url
    assert client.models._api_key == api_key

    # Check chat client
    assert isinstance(client.chat, ChatClient)
    assert client.chat._base_url == base_url
    assert client.chat._api_key == api_key

    # Check keys client
    assert isinstance(client.keys, KeysManagementClient)
    assert client.keys._base_url == base_url
    assert client.keys._api_key == api_key

    # Check http client
    assert isinstance(client.http, HTTPClient)
    assert client.http._base_url == base_url
    assert client.http._api_key == api_key


def test_client_initialization_strips_trailing_slash():
    """Test that the client properly strips trailing slashes from base_url during initialization"""
    base_url = "http://localhost:8000/////"
    client = Client(base_url=base_url)

    assert client._base_url == "http://localhost:8000"
    assert client.models._base_url == "http://localhost:8000"
    assert client.chat._base_url == "http://localhost:8000"
    assert client.keys._base_url == "http://localhost:8000"
    assert client.http._base_url == "http://localhost:8000"


def test_client_without_api_key(base_url):
    """Test that the client works without an API key"""
    client = Client(base_url=base_url)

    assert client._api_key is None
    assert client.models._api_key is None
    assert client.chat._api_key is None
    assert client.keys._api_key is None
    assert client.http._api_key is None


def test_client_initialization():
    """Test that the client is initialized correctly."""
    client = Client(
        base_url="http://localhost:4000",
        api_key="test-key",
        timeout=60,
    )

    # Check that http client is initialized correctly
    assert isinstance(client.http, HTTPClient)
    assert client.http._base_url == "http://localhost:4000"
    assert client.http._api_key == "test-key"
    assert client.http._timeout == 60


def test_client_default_timeout():
    """Test that the client uses default timeout."""
    client = Client(
        base_url="http://localhost:4000",
        api_key="test-key",
    )

    assert client.http._timeout == 30


def test_client_without_api_key():
    """Test that the client works without an API key."""
    client = Client(base_url="http://localhost:4000")

    assert client.http._api_key is None
