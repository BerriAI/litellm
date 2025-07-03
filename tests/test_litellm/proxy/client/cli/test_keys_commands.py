import json
import os
from unittest.mock import patch, Mock

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
        "litellm.proxy.client.keys.KeysManagementClient"
    ) as MockClient:
        yield MockClient


def test_keys_list_json_format(mock_keys_client, cli_runner):
    from unittest.mock import Mock, patch
    import requests
    
    # Mock the HTTP response
    mock_response = Mock()
    mock_response.json.return_value = {
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
    
    with patch('requests.Session') as mock_session:
        mock_session_instance = Mock()
        mock_session_instance.send.return_value = mock_response
        mock_session.return_value = mock_session_instance
        
        result = cli_runner.invoke(cli, ["keys", "list", "--format", "json"])
        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert output_data == mock_response.json.return_value


def test_keys_list_table_format(mock_keys_client, cli_runner):
    from unittest.mock import Mock, patch
    import requests
    
    # Mock the HTTP response
    mock_response = Mock()
    mock_response.json.return_value = {
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
    
    with patch('requests.Session') as mock_session:
        mock_session_instance = Mock()
        mock_session_instance.send.return_value = mock_response
        mock_session.return_value = mock_session_instance
        
        result = cli_runner.invoke(cli, ["keys", "list"])
        assert result.exit_code == 0
        assert "abc123" in result.output
        assert "alias1" in result.output
        assert "u1" in result.output
        assert "t1" in result.output
        assert "10.0" in result.output


def test_keys_generate_success(mock_keys_client, cli_runner):
    from unittest.mock import Mock, patch
    import requests
    
    # Mock the HTTP response
    mock_response = Mock()
    mock_response.json.return_value = {
        "key": "new-key",
        "spend": 100.0,
    }
    
    with patch('requests.Session') as mock_session:
        mock_session_instance = Mock()
        mock_session_instance.send.return_value = mock_response
        mock_session.return_value = mock_session_instance
        
        result = cli_runner.invoke(
            cli, ["keys", "generate", "--models", "gpt-4", "--spend", "100"]
        )
        assert result.exit_code == 0
        assert "new-key" in result.output


def test_keys_delete_success(mock_keys_client, cli_runner):
    from unittest.mock import Mock, patch
    import requests
    
    # Mock the HTTP response
    mock_response = Mock()
    mock_response.json.return_value = {
        "status": "success",
        "deleted_keys": ["abc123"],
    }
    
    with patch('requests.Session') as mock_session:
        mock_session_instance = Mock()
        mock_session_instance.send.return_value = mock_response
        mock_session.return_value = mock_session_instance
        
        result = cli_runner.invoke(cli, ["keys", "delete", "--keys", "abc123"])
        assert result.exit_code == 0
        assert "success" in result.output
        assert "abc123" in result.output


def test_keys_list_error_handling(mock_keys_client, cli_runner):
    from unittest.mock import Mock, patch
    import requests
    
    # Mock the HTTP response to raise an exception
    with patch('requests.Session') as mock_session:
        mock_session_instance = Mock()
        mock_session_instance.send.side_effect = Exception("API Error")
        mock_session.return_value = mock_session_instance
        
        result = cli_runner.invoke(cli, ["keys", "list"])
        assert result.exit_code != 0
        assert "API Error" in str(result.exception)


def test_keys_generate_error_handling(mock_keys_client, cli_runner):
    from unittest.mock import Mock, patch
    import requests
    
    # Mock the HTTP response to raise an exception
    with patch('requests.Session') as mock_session:
        mock_session_instance = Mock()
        mock_session_instance.send.side_effect = Exception("API Error")
        mock_session.return_value = mock_session_instance
        
        result = cli_runner.invoke(cli, ["keys", "generate", "--models", "gpt-4"])
        assert result.exit_code != 0
        assert "API Error" in str(result.exception)


def test_keys_delete_error_handling(mock_keys_client, cli_runner):
    import requests

    # Mock a connection error that would normally happen in CI
    # Mock the delete method directly to avoid JSON serialization issues
    mock_keys_client.return_value.delete.side_effect = requests.exceptions.ConnectionError(
        "Connection error"
    )
    
    result = cli_runner.invoke(cli, ["keys", "delete", "--keys", "abc123"])
    assert result.exit_code != 0
    # Check that the exception is properly propagated
    assert result.exception is not None
    # The ConnectionError should propagate since it's not caught by HTTPError handler
    assert (
        "Connection error" in str(result.exception)
        or "Failed to establish a new connection" in str(result.exception)
        or "Connection refused" in str(result.exception)
        or isinstance(result.exception, requests.exceptions.ConnectionError)
    )


def test_keys_delete_http_error_handling(mock_keys_client, cli_runner):
    from unittest.mock import Mock, patch

    import requests

    # Create a mock response object for HTTPError
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.json.return_value = {"error": "Bad request"}
    
    # Mock an HTTPError which should be caught by the delete command
    http_error = requests.exceptions.HTTPError("HTTP Error")
    http_error.response = mock_response
    
    # Mock the requests.Session to raise the HTTPError
    with patch('requests.Session') as mock_session:
        mock_session_instance = Mock()
        mock_session_instance.send.side_effect = http_error
        mock_session.return_value = mock_session_instance
        
        result = cli_runner.invoke(cli, ["keys", "delete", "--keys", "abc123"])
        assert result.exit_code != 0
        # HTTPError should be caught and converted to click.Abort
        assert isinstance(result.exception, SystemExit)  # click.Abort raises SystemExit
