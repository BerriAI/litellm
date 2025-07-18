import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


from litellm.proxy.client.users import (
    NotFoundError,
    UnauthorizedError,
    UsersManagementClient,
)


@pytest.fixture
def client():
    return UsersManagementClient(base_url="http://localhost:4000", api_key="sk-test")


@patch("requests.get")
def test_list_users_success(mock_get, client):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"users": [{"user_id": "u1"}]}
    users = client.list_users()
    assert users == [{"user_id": "u1"}]
    mock_get.assert_called_once()


@patch("requests.get")
def test_list_users_unauthorized(mock_get, client):
    mock_get.return_value.status_code = 401
    mock_get.return_value.text = "unauthorized"
    with pytest.raises(UnauthorizedError):
        client.list_users()


@patch("requests.get")
def test_get_user_success(mock_get, client):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"user_id": "u1"}
    user = client.get_user(user_id="u1")
    assert user["user_id"] == "u1"
    mock_get.assert_called_once()


@patch("requests.get")
def test_get_user_404(mock_get, client):
    mock_get.return_value.status_code = 404
    mock_get.return_value.text = "not found"
    with pytest.raises(NotFoundError):
        client.get_user(user_id="u1")


@patch("requests.post")
def test_create_user_success(mock_post, client):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"user_id": "u1"}
    user = client.create_user({"user_email": "a@b.com"})
    assert user["user_id"] == "u1"
    mock_post.assert_called_once()


@patch("requests.post")
def test_create_user_unauthorized(mock_post, client):
    mock_post.return_value.status_code = 401
    mock_post.return_value.text = "unauthorized"
    with pytest.raises(UnauthorizedError):
        client.create_user({"user_email": "a@b.com"})


@patch("requests.post")
def test_delete_user_success(mock_post, client):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"deleted": 1}
    result = client.delete_user(["u1"])
    assert result["deleted"] == 1
    mock_post.assert_called_once()


@patch("requests.post")
def test_delete_user_unauthorized(mock_post, client):
    mock_post.return_value.status_code = 401
    mock_post.return_value.text = "unauthorized"
    with pytest.raises(UnauthorizedError):
        client.delete_user(["u1"])
