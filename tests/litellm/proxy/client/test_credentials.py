import pytest
import requests
from litellm.proxy.client.credentials import CredentialsManagementClient
from litellm.proxy.client.exceptions import UnauthorizedError


@pytest.fixture
def base_url():
    return "http://localhost:8000"


@pytest.fixture
def api_key():
    return "test-api-key"


@pytest.fixture
def client(base_url, api_key):
    return CredentialsManagementClient(base_url=base_url, api_key=api_key)


def test_client_initialization(base_url, api_key):
    """Test that the CredentialsManagementClient is properly initialized"""
    client = CredentialsManagementClient(base_url=base_url, api_key=api_key)

    assert client._base_url == base_url
    assert client._api_key == api_key


def test_client_initialization_strips_trailing_slash():
    """Test that the client properly strips trailing slashes from base_url during initialization"""
    base_url = "http://localhost:8000/////"
    client = CredentialsManagementClient(base_url=base_url)

    assert client._base_url == "http://localhost:8000"


def test_client_without_api_key(base_url):
    """Test that the client works without an API key"""
    client = CredentialsManagementClient(base_url=base_url)

    assert client._api_key is None


def test_list_request(client, base_url, api_key):
    """Test list request"""
    request = client.list(return_request=True)

    assert request.method == "GET"
    assert request.url == f"{base_url}/credentials"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["Authorization"] == f"Bearer {api_key}"


def test_list_mock_response(client, requests_mock):
    """Test list with a mocked successful response"""
    mock_response = {
        "credentials": [
            {
                "credential_name": "azure1",
                "credential_info": {"api_type": "azure"},
                "model_id": "gpt-4",
            },
            {
                "credential_name": "anthropic1",
                "credential_info": {"api_type": "anthropic"},
                "model_id": "claude-2",
            },
        ]
    }

    requests_mock.get(f"{client._base_url}/credentials", json=mock_response)

    response = client.list()
    assert response == mock_response


def test_list_unauthorized_error(client, requests_mock):
    """Test that list raises UnauthorizedError for 401 responses"""
    requests_mock.get(f"{client._base_url}/credentials", status_code=401, json={"error": "Unauthorized"})

    with pytest.raises(UnauthorizedError):
        client.list()


def test_create_request(client, base_url, api_key):
    """Test create request"""
    request = client.create(
        credential_name="azure1",
        credential_info={"api_type": "azure"},
        credential_values={"api_key": "sk-123", "api_base": "https://example.azure.openai.com"},
        return_request=True,
    )

    assert request.method == "POST"
    assert request.url == f"{base_url}/credentials"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["Authorization"] == f"Bearer {api_key}"
    assert request.json == {
        "credential_name": "azure1",
        "credential_info": {"api_type": "azure"},
        "credential_values": {"api_key": "sk-123", "api_base": "https://example.azure.openai.com"},
    }


def test_create_mock_response(client, requests_mock):
    """Test create with a mocked successful response"""
    mock_response = {"credential_name": "azure1", "credential_info": {"api_type": "azure"}, "status": "success"}

    requests_mock.post(f"{client._base_url}/credentials", json=mock_response)

    response = client.create(
        credential_name="azure1",
        credential_info={"api_type": "azure"},
        credential_values={"api_key": "sk-123", "api_base": "https://example.azure.openai.com"},
    )
    assert response == mock_response


def test_create_unauthorized_error(client, requests_mock):
    """Test that create raises UnauthorizedError for 401 responses"""
    requests_mock.post(f"{client._base_url}/credentials", status_code=401, json={"error": "Unauthorized"})

    with pytest.raises(UnauthorizedError):
        client.create(
            credential_name="azure1", credential_info={"api_type": "azure"}, credential_values={"api_key": "sk-123"}
        )


def test_delete_request(client, base_url, api_key):
    """Test delete request"""
    request = client.delete(credential_name="azure1", return_request=True)

    assert request.method == "DELETE"
    assert request.url == f"{base_url}/credentials/azure1"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["Authorization"] == f"Bearer {api_key}"


def test_delete_mock_response(client, requests_mock):
    """Test delete with a mocked successful response"""
    mock_response = {"credential_name": "azure1", "status": "deleted"}

    requests_mock.delete(f"{client._base_url}/credentials/azure1", json=mock_response)

    response = client.delete(credential_name="azure1")
    assert response == mock_response


def test_delete_unauthorized_error(client, requests_mock):
    """Test that delete raises UnauthorizedError for 401 responses"""
    requests_mock.delete(f"{client._base_url}/credentials/azure1", status_code=401, json={"error": "Unauthorized"})

    with pytest.raises(UnauthorizedError):
        client.delete(credential_name="azure1")


def test_get_request(client, base_url, api_key):
    """Test get request"""
    request = client.get(credential_name="azure1", return_request=True)

    assert request.method == "GET"
    assert request.url == f"{base_url}/credentials/by_name/azure1"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["Authorization"] == f"Bearer {api_key}"


def test_get_mock_response(client, requests_mock):
    """Test get with a mocked successful response"""
    mock_response = {
        "credential_name": "azure1",
        "credential_info": {"api_type": "azure"},
        "credential_values": {"api_key": "sk-123", "api_base": "https://example.azure.openai.com"},
        "status": "active",
    }

    requests_mock.get(f"{client._base_url}/credentials/by_name/azure1", json=mock_response)

    response = client.get(credential_name="azure1")
    assert response == mock_response


def test_get_unauthorized_error(client, requests_mock):
    """Test that get raises UnauthorizedError for 401 responses"""
    requests_mock.get(f"{client._base_url}/credentials/by_name/azure1", status_code=401, json={"error": "Unauthorized"})

    with pytest.raises(UnauthorizedError):
        client.get(credential_name="azure1")
