import json
import os
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
        os.environ,
        {
            "LITELLM_PROXY_URL": "http://localhost:4000",
            "LITELLM_PROXY_API_KEY": "sk-test",
        },
    ):
        yield


@pytest.fixture
def mock_keys_client():
    with patch(
        "litellm.proxy.client.cli.commands.keys.KeysManagementClient"
    ) as MockClient:
        yield MockClient


def test_keys_list_json_format(mock_keys_client, cli_runner):
    mock_keys_client.return_value.list.return_value = {
        "keys": [
            {
                "token": "abc123",
                "key_alias": "alias1",
                "user_id": "u1",
                "team_id": "t1",
                "spend": 10.0,
            }
        ]
    }
    result = cli_runner.invoke(cli, ["keys", "list", "--format", "json"])
    assert result.exit_code == 0
    output_data = json.loads(result.output)
    assert output_data == mock_keys_client.return_value.list.return_value
    mock_keys_client.assert_called_once_with("http://localhost:4000", "sk-test")
    mock_keys_client.return_value.list.assert_called_once()


def test_keys_list_table_format(mock_keys_client, cli_runner):
    mock_keys_client.return_value.list.return_value = {
        "keys": [
            {
                "token": "abc123",
                "key_alias": "alias1",
                "user_id": "u1",
                "team_id": "t1",
                "spend": 10.0,
            }
        ]
    }
    result = cli_runner.invoke(cli, ["keys", "list"])
    assert result.exit_code == 0
    assert "abc123" in result.output
    assert "alias1" in result.output
    assert "u1" in result.output
    assert "t1" in result.output
    assert "10.0" in result.output
    mock_keys_client.assert_called_once_with("http://localhost:4000", "sk-test")
    mock_keys_client.return_value.list.assert_called_once()


def test_keys_generate_success(mock_keys_client, cli_runner):
    mock_keys_client.return_value.generate.return_value = {
        "key": "new-key",
        "spend": 100.0,
    }
    result = cli_runner.invoke(
        cli, ["keys", "generate", "--models", "gpt-4", "--spend", "100"]
    )
    assert result.exit_code == 0
    assert "new-key" in result.output
    mock_keys_client.return_value.generate.assert_called_once()


def test_keys_delete_success(mock_keys_client, cli_runner):
    mock_keys_client.return_value.delete.return_value = {
        "status": "success",
        "deleted_keys": ["abc123"],
    }
    result = cli_runner.invoke(cli, ["keys", "delete", "--keys", "abc123"])
    assert result.exit_code == 0
    assert "success" in result.output
    assert "abc123" in result.output
    mock_keys_client.return_value.delete.assert_called_once()


def test_keys_list_error_handling(mock_keys_client, cli_runner):
    mock_keys_client.return_value.list.side_effect = Exception("API Error")
    result = cli_runner.invoke(cli, ["keys", "list"])
    assert result.exit_code != 0
    assert "API Error" in str(result.exception)


def test_keys_generate_error_handling(mock_keys_client, cli_runner):
    mock_keys_client.return_value.generate.side_effect = Exception("API Error")
    result = cli_runner.invoke(cli, ["keys", "generate", "--models", "gpt-4"])
    assert result.exit_code != 0
    assert "API Error" in str(result.exception)


def test_keys_delete_error_handling(mock_keys_client, cli_runner):
    mock_keys_client.return_value.delete.side_effect = Exception("API Error")
    result = cli_runner.invoke(cli, ["keys", "delete", "--keys", "abc123"])
    assert result.exit_code != 0
    assert "API Error" in str(result.exception)
