import json
from unittest.mock import MagicMock, patch

import pytest
import requests
from click.testing import CliRunner

from litellm.proxy.client.cli.main import cli


@pytest.fixture
def mock_credentials_client():
    with patch(
        "litellm.proxy.client.cli.commands.credentials.CredentialsManagementClient"
    ) as mock:
        yield mock


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_list_credentials_table_format(cli_runner, mock_credentials_client):
    # Mock response data
    mock_response = {
        "credentials": [
            {
                "credential_name": "test-cred-1",
                "credential_info": {"custom_llm_provider": "azure"},
            },
            {
                "credential_name": "test-cred-2",
                "credential_info": {"custom_llm_provider": "anthropic"},
            },
        ]
    }
    mock_instance = mock_credentials_client.return_value
    mock_instance.list.return_value = mock_response

    # Run command
    result = cli_runner.invoke(cli, ["credentials", "list"])

    # Verify
    assert result.exit_code == 0
    assert "test-cred-1" in result.output
    assert "azure" in result.output
    assert "test-cred-2" in result.output
    assert "anthropic" in result.output


def test_list_credentials_json_format(cli_runner, mock_credentials_client):
    # Mock response data
    mock_response = {
        "credentials": [
            {
                "credential_name": "test-cred",
                "credential_info": {"custom_llm_provider": "azure"},
            }
        ]
    }
    mock_instance = mock_credentials_client.return_value
    mock_instance.list.return_value = mock_response

    # Run command
    result = cli_runner.invoke(cli, ["credentials", "list", "--format", "json"])

    # Verify
    assert result.exit_code == 0
    output_data = json.loads(result.output)
    assert output_data == mock_response


def test_create_credential_success(cli_runner, mock_credentials_client):
    # Mock response data
    mock_response = {"status": "success", "credential_name": "test-cred"}
    mock_instance = mock_credentials_client.return_value
    mock_instance.create.return_value = mock_response

    # Run command
    result = cli_runner.invoke(
        cli,
        [
            "credentials",
            "create",
            "test-cred",
            "--info",
            '{"custom_llm_provider": "azure"}',
            "--values",
            '{"api_key": "test-key"}',
        ],
    )

    # Verify
    assert result.exit_code == 0
    output_data = json.loads(result.output)
    assert output_data == mock_response
    mock_instance.create.assert_called_once_with(
        "test-cred",
        {"custom_llm_provider": "azure"},
        {"api_key": "test-key"},
    )


def test_create_credential_invalid_json(cli_runner, mock_credentials_client):
    # Run command with invalid JSON
    result = cli_runner.invoke(
        cli,
        [
            "credentials",
            "create",
            "test-cred",
            "--info",
            "invalid-json",
            "--values",
            '{"api_key": "test-key"}',
        ],
    )

    # Verify
    assert result.exit_code == 2
    assert "Invalid JSON" in result.output
    mock_instance = mock_credentials_client.return_value
    mock_instance.create.assert_not_called()


def test_create_credential_http_error(cli_runner, mock_credentials_client):
    # Mock HTTP error
    mock_instance = mock_credentials_client.return_value
    mock_error_response = MagicMock()
    mock_error_response.status_code = 400
    mock_error_response.json.return_value = {"error": "Invalid request"}
    mock_instance.create.side_effect = requests.exceptions.HTTPError(
        response=mock_error_response
    )

    # Run command
    result = cli_runner.invoke(
        cli,
        [
            "credentials",
            "create",
            "test-cred",
            "--info",
            '{"custom_llm_provider": "azure"}',
            "--values",
            '{"api_key": "test-key"}',
        ],
    )

    # Verify
    assert result.exit_code == 1
    assert "Error: HTTP 400" in result.output
    assert "Invalid request" in result.output


def test_delete_credential_success(cli_runner, mock_credentials_client):
    # Mock response data
    mock_response = {"status": "success", "message": "Credential deleted"}
    mock_instance = mock_credentials_client.return_value
    mock_instance.delete.return_value = mock_response

    # Run command
    result = cli_runner.invoke(cli, ["credentials", "delete", "test-cred"])

    # Verify
    assert result.exit_code == 0
    output_data = json.loads(result.output)
    assert output_data == mock_response
    mock_instance.delete.assert_called_once_with("test-cred")


def test_delete_credential_http_error(cli_runner, mock_credentials_client):
    # Mock HTTP error
    mock_instance = mock_credentials_client.return_value
    mock_error_response = MagicMock()
    mock_error_response.status_code = 404
    mock_error_response.json.return_value = {"error": "Credential not found"}
    mock_instance.delete.side_effect = requests.exceptions.HTTPError(
        response=mock_error_response
    )

    # Run command
    result = cli_runner.invoke(cli, ["credentials", "delete", "test-cred"])

    # Verify
    assert result.exit_code == 1
    assert "Error: HTTP 404" in result.output
    assert "Credential not found" in result.output


def test_get_credential_success(cli_runner, mock_credentials_client):
    # Mock response data
    mock_response = {
        "credential_name": "test-cred",
        "credential_info": {"custom_llm_provider": "azure"},
    }
    mock_instance = mock_credentials_client.return_value
    mock_instance.get.return_value = mock_response

    # Run command
    result = cli_runner.invoke(cli, ["credentials", "get", "test-cred"])

    # Verify
    assert result.exit_code == 0
    output_data = json.loads(result.output)
    assert output_data == mock_response
    mock_instance.get.assert_called_once_with("test-cred")
