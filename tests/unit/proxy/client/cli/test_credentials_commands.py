import json
from unittest.mock import MagicMock

import pytest
import requests
from click.testing import CliRunner



from litellm.proxy.client.cli.main import cli


@pytest.fixture
def mock_credentials_client(monkeypatch):
    """Patch the CredentialsManagementClient used by the CLI commands."""
    mock_client = MagicMock()
    monkeypatch.setattr(
        "litellm.proxy.client.credentials.CredentialsManagementClient",
        mock_client,
    )
    monkeypatch.setattr(
        "litellm.proxy.client.cli.commands.credentials.CredentialsManagementClient",
        mock_client,
    )
    return mock_client


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_alist_credentials_table_format(cli_runner, mock_credentials_client):
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


def test_alist_credentials_json_format(cli_runner, mock_credentials_client):
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


def test_acreate_credential_success(cli_runner, mock_credentials_client):
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
        display_name=None,
    )


def test_acreate_credential_invalid_json(cli_runner, mock_credentials_client):
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


def test_acreate_credential_http_error(cli_runner, mock_credentials_client):
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


def test_adelete_credential_success(cli_runner, mock_credentials_client):
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


def test_adelete_credential_http_error(cli_runner, mock_credentials_client):
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


def test_aget_credential_success(cli_runner, mock_credentials_client):
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


def test_list_credentials_table_shows_display_name_and_source(cli_runner, mock_credentials_client):
    mock_credentials_client.return_value.list.return_value = {
        "credentials": [
            {"credential_name": "openai-prod", "display_name": "Prod OpenAI", "source": "db", "credential_info": {}},
            {"credential_name": "from-yaml", "display_name": None, "source": "config", "credential_info": {}},
        ]
    }

    result = cli_runner.invoke(cli, ["credentials", "list"])

    assert result.exit_code == 0
    assert "Prod OpenAI" in result.output
    assert "config" in result.output
    assert "None" not in result.output


def test_create_credential_forwards_the_display_name(cli_runner, mock_credentials_client):
    mock_instance = mock_credentials_client.return_value
    mock_instance.create.return_value = {"success": True}

    result = cli_runner.invoke(
        cli,
        ["credentials", "create", "openai-prod", "--info", "{}", "--values", "{}", "--display-name", "Prod OpenAI"],
    )

    assert result.exit_code == 0
    mock_instance.create.assert_called_once_with("openai-prod", {}, {}, display_name="Prod OpenAI")


@pytest.mark.parametrize(
    ("flags", "expected"),
    [(["--display-name", "Prod OpenAI"], "Prod OpenAI"), (["--clear-display-name"], None)],
)
def test_update_credential_sets_or_clears_the_display_name(cli_runner, mock_credentials_client, flags, expected):
    mock_instance = mock_credentials_client.return_value
    mock_instance.update_display_name.return_value = {"success": True}

    result = cli_runner.invoke(cli, ["credentials", "update", "openai-prod", *flags])

    assert result.exit_code == 0, result.output
    mock_instance.update_display_name.assert_called_once_with("openai-prod", expected)


@pytest.mark.parametrize("flags", [[], ["--display-name", "Prod", "--clear-display-name"]])
def test_update_credential_requires_exactly_one_display_name_flag(cli_runner, mock_credentials_client, flags):
    result = cli_runner.invoke(cli, ["credentials", "update", "openai-prod", *flags])

    assert result.exit_code != 0
    assert "exactly one" in result.output
    mock_credentials_client.return_value.update_display_name.assert_not_called()
