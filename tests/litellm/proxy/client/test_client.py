import pytest
from litellm.proxy.client import Client, ModelsManagementClient, ChatClient


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


def test_client_initialization_strips_trailing_slash():
    """Test that the client properly strips trailing slashes from base_url during initialization"""
    base_url = "http://localhost:8000/////"
    client = Client(base_url=base_url)

    assert client._base_url == "http://localhost:8000"
    assert client.models._base_url == "http://localhost:8000"
    assert client.chat._base_url == "http://localhost:8000"


def test_client_without_api_key(base_url):
    """Test that the client works without an API key"""
    client = Client(base_url=base_url)

    assert client._api_key is None
    assert client.models._api_key is None
    assert client.chat._api_key is None
