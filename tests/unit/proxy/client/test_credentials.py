
import time
from typing import Final

import pytest
import requests
import responses
from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.proxy.client.credentials import (
    CredentialsManagementClient,
    UserConnection,
    UserConnectionDelete,
    UserConnectionPoll,
    UserConnections,
    UserConnectionStart,
)
from litellm.proxy.client.exceptions import UnauthorizedError
from litellm.proxy.credential_endpoints.endpoints import CredentialHelperUtils
from litellm.types.utils import CredentialItem


class _CapturedRequest(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    headers: dict[str, str]
    body: bytes | str | None
    method: str | None
    url: str | None


class _CapturedCall(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    request: _CapturedRequest


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


@responses.activate
def test_list_mock_response(client):
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

    responses.add(
        responses.GET,
        f"{client._base_url}/credentials",
        json=mock_response,
        status=200,
    )

    response = client.list()
    assert response == mock_response


@responses.activate
def test_list_unauthorized_error(client):
    """Test that list raises UnauthorizedError for 401 responses"""
    responses.add(
        responses.GET,
        f"{client._base_url}/credentials",
        status=401,
        json={"error": "Unauthorized"},
    )

    with pytest.raises(UnauthorizedError):
        client.list()


def test_create_request(client, base_url, api_key):
    """Test create request"""
    request = client.create(
        credential_name="azure1",
        credential_info={"api_type": "azure"},
        credential_values={
            "api_key": "sk-123",
            "api_base": "https://example.azure.openai.com",
        },
        return_request=True,
    )

    assert request.method == "POST"
    assert request.url == f"{base_url}/credentials"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["Authorization"] == f"Bearer {api_key}"
    assert request.json == {
        "credential_name": "azure1",
        "credential_info": {"api_type": "azure"},
        "credential_values": {
            "api_key": "sk-123",
            "api_base": "https://example.azure.openai.com",
        },
    }


def test_create_request_includes_the_display_name_only_when_given(client):
    labeled = client.create("azure1", {}, {"api_key": "sk-123"}, return_request=True, display_name="Azure EU")
    unlabeled = client.create("azure1", {}, {"api_key": "sk-123"}, return_request=True)

    assert labeled.json["display_name"] == "Azure EU"
    assert "display_name" not in unlabeled.json


@pytest.mark.parametrize("display_name", ["Azure EU", None])
def test_update_display_name_request(client, base_url, api_key, display_name):
    request = client.update_display_name("azure1", display_name, return_request=True)

    assert request.method == "PATCH"
    assert request.url == f"{base_url}/credentials/azure1"
    assert request.headers["Authorization"] == f"Bearer {api_key}"
    assert request.json == {"display_name": display_name, "credential_info": {}}


@pytest.mark.parametrize(("credential_name", "path"), [("team?prod", "team%3Fprod"), ("a/b#c", "a%2Fb%23c")])
def test_update_display_name_targets_the_whole_credential_name(client, base_url, credential_name, path):
    request = client.update_display_name(credential_name, "Label", return_request=True)

    assert request.prepare().url == f"{base_url}/credentials/{path}"


@responses.activate
def test_create_mock_response(client):
    """Test create with a mocked successful response"""
    mock_response = {
        "credential_name": "azure1",
        "credential_info": {"api_type": "azure"},
        "status": "success",
    }

    responses.add(
        responses.POST,
        f"{client._base_url}/credentials",
        json=mock_response,
        status=200,
    )

    response = client.create(
        credential_name="azure1",
        credential_info={"api_type": "azure"},
        credential_values={
            "api_key": "sk-123",
            "api_base": "https://example.azure.openai.com",
        },
    )
    assert response == mock_response


@responses.activate
def test_create_unauthorized_error(client):
    """Test that create raises UnauthorizedError for 401 responses"""
    responses.add(
        responses.POST,
        f"{client._base_url}/credentials",
        status=401,
        json={"error": "Unauthorized"},
    )

    with pytest.raises(UnauthorizedError):
        client.create(
            credential_name="azure1",
            credential_info={"api_type": "azure"},
            credential_values={"api_key": "sk-123"},
        )


def test_delete_request(client, base_url, api_key):
    """Test delete request"""
    request = client.delete(credential_name="azure1", return_request=True)

    assert request.method == "DELETE"
    assert request.url == f"{base_url}/credentials/azure1"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["Authorization"] == f"Bearer {api_key}"


@responses.activate
def test_delete_mock_response(client):
    """Test delete with a mocked successful response"""
    mock_response = {"credential_name": "azure1", "status": "deleted"}

    responses.add(
        responses.DELETE,
        f"{client._base_url}/credentials/azure1",
        json=mock_response,
        status=200,
    )

    response = client.delete(credential_name="azure1")
    assert response == mock_response


@responses.activate
def test_delete_unauthorized_error(client):
    """Test that delete raises UnauthorizedError for 401 responses"""
    responses.add(
        responses.DELETE,
        f"{client._base_url}/credentials/azure1",
        status=401,
        json={"error": "Unauthorized"},
    )

    with pytest.raises(UnauthorizedError):
        client.delete(credential_name="azure1")


def test_get_request(client, base_url, api_key):
    """Test get request"""
    request = client.get(credential_name="azure1", return_request=True)

    assert request.method == "GET"
    assert request.url == f"{base_url}/credentials/by_name/azure1"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["Authorization"] == f"Bearer {api_key}"


@responses.activate
def test_get_mock_response(client):
    """Test get with a mocked successful response"""
    mock_response = {
        "credential_name": "azure1",
        "credential_info": {"api_type": "azure"},
        "credential_values": {
            "api_key": "sk-123",
            "api_base": "https://example.azure.openai.com",
        },
        "status": "active",
    }

    responses.add(
        responses.GET,
        f"{client._base_url}/credentials/by_name/azure1",
        json=mock_response,
        status=200,
    )

    response = client.get(credential_name="azure1")
    assert response == mock_response


@responses.activate
def test_get_unauthorized_error(client):
    """Test that get raises UnauthorizedError for 401 responses"""
    responses.add(
        responses.GET,
        f"{client._base_url}/credentials/by_name/azure1",
        status=401,
        json={"error": "Unauthorized"},
    )

    with pytest.raises(UnauthorizedError):
        client.get(credential_name="azure1")


def test_encrypt_credential_values_does_not_mutate_original(monkeypatch):
    """Ensure encrypt_credential_values returns a new encrypted object"""
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-key")
    credential = CredentialItem(
        credential_name="azure1",
        credential_values={"api_key": "sk-123"},
        credential_info={"api_type": "azure"},
    )

    encrypted = CredentialHelperUtils.encrypt_credential_values(credential)

    assert encrypted.credential_values["api_key"] != "sk-123"
    assert credential.credential_values["api_key"] == "sk-123"
    assert encrypted.credential_name == credential.credential_name


def test_list_gives_up_at_the_timeout_instead_of_hanging(hanging_server):
    """
    A proxy that accepts the connection but never answers used to pin the caller's
    process forever, since the request carried no timeout at all.
    """
    client = CredentialsManagementClient(base_url=hanging_server, api_key="sk-test", timeout=1)

    started = time.monotonic()
    with pytest.raises(requests.exceptions.Timeout):
        client.list()

    assert time.monotonic() - started < 10


@responses.activate
def test_user_connection_methods_encode_names_send_auth_and_validate_responses(
    client: CredentialsManagementClient,
    base_url: str,
    api_key: str,
) -> None:
    credential_name: Final = "my/ credential"
    credential_path: Final = "my%2F%20credential"
    start_payload: Final = {
        "user_code": "ABCD-1234",
        "verification_uri": "https://github.com/login/device",
        "expires_in": 900,
        "interval": 5,
        "flow_handle": "encrypted-handle",
    }
    poll_payload: Final = {"status": "slow_down", "interval": 10, "github_login": None}
    connections_payload: Final = {
        "connections": [
            {
                "credential_name": credential_name,
                "provider": "github_copilot",
                "connected": True,
                "github_login": "octocat",
                "connected_at": "2025-01-19T12:00:00+00:00",
            }
        ]
    }
    delete_payload: Final = {"status": "disconnected"}
    responses.add(
        responses.POST,
        f"{base_url}/credentials/{credential_path}/user_connection/start",
        json=start_payload,
        status=200,
    )
    responses.add(
        responses.POST,
        f"{base_url}/credentials/{credential_path}/user_connection/poll",
        json=poll_payload,
        status=200,
    )
    responses.add(
        responses.GET,
        f"{base_url}/credentials/user_connections",
        json=connections_payload,
        status=200,
    )
    responses.add(
        responses.DELETE,
        f"{base_url}/credentials/{credential_path}/user_connection",
        json=delete_payload,
        status=200,
    )

    start: Final = client.start_user_connection(credential_name)
    poll: Final = client.poll_user_connection(credential_name, "encrypted-handle")
    connections: Final = client.list_user_connections()
    deleted: Final = client.delete_user_connection(credential_name)
    calls: Final = TypeAdapter(tuple[_CapturedCall, ...]).validate_python(responses.calls)

    assert start == UserConnectionStart.model_validate(start_payload)
    assert poll == UserConnectionPoll.model_validate(poll_payload)
    assert connections == UserConnections.model_validate(connections_payload)
    assert connections.connections == (
        UserConnection.model_validate(connections_payload["connections"][0]),
    )
    assert deleted == UserConnectionDelete.model_validate(delete_payload)
    assert tuple(call.request.headers["Authorization"] for call in calls) == (f"Bearer {api_key}",) * 4
    assert calls[1].request.body == b'{"flow_handle": "encrypted-handle"}'
    assert tuple(call.request.method for call in calls) == ("POST", "POST", "GET", "DELETE")
    assert tuple(call.request.url for call in calls) == (
        f"{base_url}/credentials/{credential_path}/user_connection/start",
        f"{base_url}/credentials/{credential_path}/user_connection/poll",
        f"{base_url}/credentials/user_connections",
        f"{base_url}/credentials/{credential_path}/user_connection",
    )


@responses.activate
def test_start_user_connection_raises_unauthorized_error(
    client: CredentialsManagementClient,
    base_url: str,
) -> None:
    responses.add(
        responses.POST,
        f"{base_url}/credentials/copilot/user_connection/start",
        json={"detail": "An authenticated user is required"},
        status=401,
    )

    with pytest.raises(UnauthorizedError):
        client.start_user_connection("copilot")


def test_user_connection_start_repr_omits_flow_handle() -> None:
    start: Final = UserConnectionStart(
        user_code="ABCD-1234",
        verification_uri="https://github.com/login/device",
        expires_in=900,
        interval=5,
        flow_handle="secret-handle",
    )

    assert "secret-handle" not in repr(start)
