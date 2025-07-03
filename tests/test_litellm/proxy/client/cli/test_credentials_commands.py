import json
from unittest.mock import MagicMock, patch, Mock

import pytest
import requests
from click.testing import CliRunner

from litellm.proxy.client.cli.main import cli


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_list_credentials_table_format(cli_runner):
    # Mock response data
    mock_response = Mock()
    mock_response.json.return_value = {
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
    
    with patch('requests.Session') as mock_session:
        mock_session_instance = Mock()
        mock_session_instance.send.return_value = mock_response
        mock_session.return_value = mock_session_instance
        
        # Run command
        result = cli_runner.invoke(cli, ["credentials", "list"])

        # Verify
        assert result.exit_code == 0
        assert "test-cred-1" in result.output
        assert "azure" in result.output
        assert "test-cred-2" in result.output
        assert "anthropic" in result.output


def test_list_credentials_json_format(cli_runner):
    # Mock response data
    mock_response = Mock()
    mock_response.json.return_value = {
        "credentials": [
            {
                "credential_name": "test-cred",
                "credential_info": {"custom_llm_provider": "azure"},
            }
        ]
    }
    
    with patch('requests.Session') as mock_session:
        mock_session_instance = Mock()
        mock_session_instance.send.return_value = mock_response
        mock_session.return_value = mock_session_instance
        
        # Run command
        result = cli_runner.invoke(cli, ["credentials", "list", "--format", "json"])

        # Verify
        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert output_data == mock_response.json.return_value


def test_create_credential_success(cli_runner):
    # Mock response data
    mock_response = Mock()
    mock_response.json.return_value = {"status": "success", "credential_name": "test-cred"}
    
    with patch('requests.Session') as mock_session:
        mock_session_instance = Mock()
        mock_session_instance.send.return_value = mock_response
        mock_session.return_value = mock_session_instance
        
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
        assert output_data == mock_response.json.return_value


def test_create_credential_invalid_json(cli_runner):
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


def test_create_credential_http_error(cli_runner):
    # Mock HTTP error
    mock_error_response = Mock()
    mock_error_response.status_code = 400
    mock_error_response.json.return_value = {"error": "Invalid request"}
    
    with patch('requests.Session') as mock_session:
        mock_session_instance = Mock()
        mock_session_instance.send.side_effect = requests.exceptions.HTTPError(
            response=mock_error_response
        )
        mock_session.return_value = mock_session_instance
        
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


def test_delete_credential_success(cli_runner):
    # Mock response data
    mock_response = Mock()
    mock_response.json.return_value = {"status": "success", "message": "Credential deleted"}
    
    with patch('requests.Session') as mock_session:
        mock_session_instance = Mock()
        mock_session_instance.send.return_value = mock_response
        mock_session.return_value = mock_session_instance
        
        # Run command
        result = cli_runner.invoke(cli, ["credentials", "delete", "test-cred"])

        # Verify
        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert output_data == mock_response.json.return_value


def test_delete_credential_http_error(cli_runner):
    # Mock HTTP error
    mock_error_response = Mock()
    mock_error_response.status_code = 404
    mock_error_response.json.return_value = {"error": "Credential not found"}
    
    with patch('requests.Session') as mock_session:
        mock_session_instance = Mock()
        mock_session_instance.send.side_effect = requests.exceptions.HTTPError(
            response=mock_error_response
        )
        mock_session.return_value = mock_session_instance
        
        # Run command
        result = cli_runner.invoke(cli, ["credentials", "delete", "test-cred"])

        # Verify
        assert result.exit_code == 1
        assert "Error: HTTP 404" in result.output
        assert "Credential not found" in result.output


def test_get_credential_success(cli_runner):
    # Mock response data
    mock_response = Mock()
    mock_response.json.return_value = {
        "credential_name": "test-cred",
        "credential_info": {"custom_llm_provider": "azure"},
    }
    
    with patch('requests.Session') as mock_session:
        mock_session_instance = Mock()
        mock_session_instance.send.return_value = mock_response
        mock_session.return_value = mock_session_instance
        
        # Run command
        result = cli_runner.invoke(cli, ["credentials", "get", "test-cred"])

        # Verify
        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert output_data == mock_response.json.return_value
