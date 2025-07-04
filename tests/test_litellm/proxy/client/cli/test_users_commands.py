from unittest.mock import patch

import pytest
from click.testing import CliRunner

from litellm.proxy.client.cli import cli


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def mock_env():
    with patch.dict(
        "os.environ",
        {
            "LITELLM_PROXY_URL": "http://localhost:4000",
            "LITELLM_PROXY_API_KEY": "sk-test",
        },
    ):
        yield


@pytest.fixture
def mock_users_client():
    with patch(
        "litellm.proxy.client.cli.commands.users.UsersManagementClient"
    ) as MockClient:
        yield MockClient


def test_users_list(cli_runner, mock_users_client):
    mock_users_client.return_value.list_users.return_value = [
        {
            "user_id": "u1",
            "user_email": "a@b.com",
            "user_role": "internal_user",
            "teams": ["t1"],
        },
        {
            "user_id": "u2",
            "user_email": "b@b.com",
            "user_role": "proxy_admin",
            "teams": ["t2", "t3"],
        },
    ]
    result = cli_runner.invoke(cli, ["users", "list"])
    assert result.exit_code == 0
    assert "u1" in result.output
    assert "a@b.com" in result.output
    assert "proxy_admin" in result.output
    assert "t3" in result.output
    mock_users_client.return_value.list_users.assert_called_once()


def test_users_get(cli_runner, mock_users_client):
    mock_users_client.return_value.get_user.return_value = {
        "user_id": "u1",
        "user_email": "a@b.com",
    }
    result = cli_runner.invoke(cli, ["users", "get", "--id", "u1"])
    assert result.exit_code == 0
    assert '"user_id": "u1"' in result.output
    assert '"user_email": "a@b.com"' in result.output
    mock_users_client.return_value.get_user.assert_called_once_with(user_id="u1")


def test_users_create(cli_runner, mock_users_client):
    mock_users_client.return_value.create_user.return_value = {
        "user_id": "u1",
        "user_email": "a@b.com",
    }
    result = cli_runner.invoke(
        cli, ["users", "create", "--email", "a@b.com", "--role", "internal_user"]
    )
    assert result.exit_code == 0
    assert '"user_id": "u1"' in result.output
    assert '"user_email": "a@b.com"' in result.output
    mock_users_client.return_value.create_user.assert_called_once()


def test_users_delete(cli_runner, mock_users_client):
    mock_users_client.return_value.delete_user.return_value = {"deleted": 1}
    result = cli_runner.invoke(cli, ["users", "delete", "u1", "u2"])
    assert result.exit_code == 0
    assert '"deleted": 1' in result.output
    mock_users_client.return_value.delete_user.assert_called_once_with(["u1", "u2"])
