import json
import os
import sys
from unittest.mock import patch

import requests

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path



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
    with patch("litellm.proxy.client.cli.commands.keys.KeysManagementClient") as MockClient:
        yield MockClient


def test_async_keys_list_json_format(mock_keys_client, cli_runner):
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


def test_async_keys_list_table_format(mock_keys_client, cli_runner):
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


def test_async_keys_generate_success(mock_keys_client, cli_runner):
    mock_keys_client.return_value.generate.return_value = {
        "key": "new-key",
        "spend": 100.0,
    }
    result = cli_runner.invoke(cli, ["keys", "generate", "--models", "gpt-4", "--spend", "100"])
    assert result.exit_code == 0
    assert "new-key" in result.output
    mock_keys_client.return_value.generate.assert_called_once()


def test_async_keys_delete_success(mock_keys_client, cli_runner):
    mock_keys_client.return_value.delete.return_value = {
        "status": "success",
        "deleted_keys": ["abc123"],
    }
    result = cli_runner.invoke(cli, ["keys", "delete", "--keys", "abc123"])
    assert result.exit_code == 0
    assert "success" in result.output
    assert "abc123" in result.output
    mock_keys_client.return_value.delete.assert_called_once()


def test_async_keys_list_error_handling(mock_keys_client, cli_runner):
    mock_keys_client.return_value.list.side_effect = Exception("API Error")
    result = cli_runner.invoke(cli, ["keys", "list"])
    assert result.exit_code != 0
    assert "API Error" in str(result.exception)


def test_async_keys_generate_error_handling(mock_keys_client, cli_runner):
    mock_keys_client.return_value.generate.side_effect = Exception("API Error")
    result = cli_runner.invoke(cli, ["keys", "generate", "--models", "gpt-4"])
    assert result.exit_code != 0
    assert "API Error" in str(result.exception)


def test_async_keys_delete_error_handling(mock_keys_client, cli_runner):
    import requests

    # Mock a connection error that would normally happen in CI
    mock_keys_client.return_value.delete.side_effect = requests.exceptions.ConnectionError(
        "Connection error"
    )
    result = cli_runner.invoke(cli, ["keys", "delete", "--keys", "abc123"])
    assert result.exit_code != 0
    # Check that the exception is properly propagated
    assert result.exception is not None
    # The ConnectionError should propagate since it's not caught by HTTPError handler
    # Check for connection-related keywords that appear in both mocked and real errors
    error_str = str(result.exception).lower()
    assert any(keyword in error_str for keyword in ["connection", "connect", "refused", "error"])


def test_async_keys_delete_http_error_handling(mock_keys_client, cli_runner):
    from unittest.mock import Mock

    import requests

    # Create a mock response object for HTTPError
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.json.return_value = {"error": "Bad request"}
    
    # Mock an HTTPError which should be caught by the delete command
    http_error = requests.exceptions.HTTPError("HTTP Error")
    http_error.response = mock_response
    mock_keys_client.return_value.delete.side_effect = http_error
    
    result = cli_runner.invoke(cli, ["keys", "delete", "--keys", "abc123"])
    assert result.exit_code != 0
    # HTTPError should be caught and converted to click.Abort
    assert isinstance(result.exception, SystemExit)  # click.Abort raises SystemExit


# Tests for keys import command
def test_keys_import_dry_run_success(mock_keys_client, cli_runner):
    """Test successful dry-run import showing table of keys that would be imported"""
    # Mock source client response (paginated)
    mock_source_instance = mock_keys_client.return_value
    mock_source_instance.list.side_effect = [
        {
            "keys": [
                {
                    "key_alias": "test-key-1",
                    "user_id": "user1@example.com",
                    "created_at": "2024-01-15T10:30:00Z",
                    "models": ["gpt-4"],
                    "spend": 10.0,
                },
                {
                    "key_alias": "test-key-2", 
                    "user_id": "user2@example.com",
                    "created_at": "2024-01-16T11:45:00Z",
                    "models": [],
                    "spend": 5.0,
                }
            ]
        },
        {"keys": []}  # Empty second page
    ]
    
    result = cli_runner.invoke(cli, [
        "keys", "import",
        "--source-base-url", "https://source.example.com",
        "--source-api-key", "sk-source-123",
        "--dry-run"
    ])
    
    assert result.exit_code == 0
    assert "Found 2 keys in source instance" in result.output
    assert "DRY RUN MODE" in result.output
    assert "test-key-1" in result.output
    assert "user1@example.com" in result.output
    assert "test-key-2" in result.output
    assert "user2@example.com" in result.output
    
    # Verify source client was called (pagination stops early when fewer keys than page_size)
    assert mock_source_instance.list.call_count >= 1
    mock_source_instance.list.assert_any_call(return_full_object=True, page=1, size=100)


def test_keys_import_actual_import_success(mock_keys_client, cli_runner):
    """Test successful actual import of keys"""
    # Create separate mock instances for source and destination
    with patch("litellm.proxy.client.cli.commands.keys.KeysManagementClient") as MockClient:
        mock_source_instance = MockClient.return_value
        mock_dest_instance = MockClient.return_value
        
        # Configure source client
        mock_source_instance.list.side_effect = [
            {
                "keys": [
                    {
                        "key_alias": "import-key-1",
                        "user_id": "user1@example.com",
                        "models": ["gpt-4"],
                        "spend": 100.0,
                        "team_id": "team-1"
                    }
                ]
            },
            {"keys": []}  # Empty second page
        ]
        
        # Configure destination client
        mock_dest_instance.generate.return_value = {
            "key": "sk-new-generated-key",
            "status": "success"
        }
        
        result = cli_runner.invoke(cli, [
            "keys", "import",
            "--source-base-url", "https://source.example.com",
            "--source-api-key", "sk-source-123"
        ])
        
        assert result.exit_code == 0
        assert "Found 1 keys in source instance" in result.output
        assert "✓ Imported key: import-key-1" in result.output
        assert "Successfully imported: 1" in result.output
        assert "Failed to import: 0" in result.output
        
        # Verify generate was called with correct parameters
        mock_dest_instance.generate.assert_called_once_with(
            models=["gpt-4"],
            spend=100.0,
            key_alias="import-key-1",
            team_id="team-1",
            user_id="user1@example.com"
        )


def test_keys_import_pagination_handling(mock_keys_client, cli_runner):
    """Test that import correctly handles pagination to get all keys"""
    mock_source_instance = mock_keys_client.return_value
    mock_source_instance.list.side_effect = [
        {"keys": [{"key_alias": f"key-{i}", "user_id": f"user{i}@example.com"} for i in range(100)]},  # Page 1: 100 keys
        {"keys": [{"key_alias": f"key-{i}", "user_id": f"user{i}@example.com"} for i in range(100, 150)]},  # Page 2: 50 keys
        {"keys": []}  # Page 3: Empty
    ]
    
    result = cli_runner.invoke(cli, [
        "keys", "import", 
        "--source-base-url", "https://source.example.com",
        "--dry-run"
    ])
    
    assert result.exit_code == 0
    assert "Fetched page 1: 100 keys" in result.output
    assert "Fetched page 2: 50 keys" in result.output
    assert "Found 150 keys in source instance" in result.output
    
    # Verify pagination calls (stops early when fewer keys than page_size)
    assert mock_source_instance.list.call_count >= 2
    mock_source_instance.list.assert_any_call(return_full_object=True, page=1, size=100)
    mock_source_instance.list.assert_any_call(return_full_object=True, page=2, size=100)


def test_keys_import_created_since_filter(mock_keys_client, cli_runner):
    """Test that --created-since filter works correctly"""
    mock_source_instance = mock_keys_client.return_value
    mock_source_instance.list.side_effect = [
        {
            "keys": [
                {
                    "key_alias": "old-key",
                    "user_id": "user1@example.com", 
                    "created_at": "2024-01-01T10:00:00Z"  # Before filter
                },
                {
                    "key_alias": "new-key",
                    "user_id": "user2@example.com",
                    "created_at": "2024-07-08T10:00:00Z"  # After filter
                }
            ]
        },
        {"keys": []}
    ]
    
    result = cli_runner.invoke(cli, [
        "keys", "import",
        "--source-base-url", "https://source.example.com", 
        "--created-since", "2024-07-07_18:19",
        "--dry-run"
    ])
    
    assert result.exit_code == 0
    assert "Filtered 2 keys to 1 keys created since 2024-07-07_18:19" in result.output
    assert "Found 1 keys in source instance" in result.output
    assert "new-key" in result.output
    assert "old-key" not in result.output


def test_keys_import_created_since_date_only_format(mock_keys_client, cli_runner):
    """Test --created-since with date-only format (YYYY-MM-DD)"""
    mock_source_instance = mock_keys_client.return_value
    mock_source_instance.list.side_effect = [
        {
            "keys": [
                {
                    "key_alias": "test-key",
                    "user_id": "user@example.com",
                    "created_at": "2024-07-08T10:00:00Z"
                }
            ]
        },
        {"keys": []}
    ]
    
    result = cli_runner.invoke(cli, [
        "keys", "import",
        "--source-base-url", "https://source.example.com",
        "--created-since", "2024-07-07",  # Date only format
        "--dry-run"
    ])
    
    assert result.exit_code == 0
    assert "Filtered 1 keys to 1 keys created since 2024-07-07" in result.output


def test_keys_import_no_keys_found(mock_keys_client, cli_runner):
    """Test handling when no keys are found in source instance"""
    mock_source_instance = mock_keys_client.return_value
    mock_source_instance.list.return_value = {"keys": []}
    
    result = cli_runner.invoke(cli, [
        "keys", "import",
        "--source-base-url", "https://source.example.com",
        "--dry-run"
    ])
    
    assert result.exit_code == 0
    assert "No keys found in source instance" in result.output


def test_keys_import_invalid_date_format(cli_runner):
    """Test error handling for invalid --created-since date format"""
    result = cli_runner.invoke(cli, [
        "keys", "import",
        "--source-base-url", "https://source.example.com",
        "--created-since", "invalid-date",
        "--dry-run"
    ])
    
    assert result.exit_code != 0
    assert "Invalid date format" in result.output
    assert "Use YYYY-MM-DD_HH:MM or YYYY-MM-DD" in result.output


def test_keys_import_source_api_error(mock_keys_client, cli_runner):
    """Test error handling when source API returns an error"""
    mock_source_instance = mock_keys_client.return_value
    mock_source_instance.list.side_effect = Exception("Source API Error")
    
    result = cli_runner.invoke(cli, [
        "keys", "import",
        "--source-base-url", "https://source.example.com",
        "--dry-run"
    ])
    
    assert result.exit_code != 0
    assert "Source API Error" in result.output


def test_keys_import_partial_failure(mock_keys_client, cli_runner):
    """Test handling when some keys fail to import"""
    with patch("litellm.proxy.client.cli.commands.keys.KeysManagementClient") as MockClient:
        mock_source_instance = MockClient.return_value
        mock_dest_instance = MockClient.return_value
        
        # Source returns 2 keys
        mock_source_instance.list.side_effect = [
            {
                "keys": [
                    {"key_alias": "success-key", "user_id": "user1@example.com"},
                    {"key_alias": "fail-key", "user_id": "user2@example.com"}
                ]
            },
            {"keys": []}
        ]
        
        # Destination: first succeeds, second fails
        mock_dest_instance.generate.side_effect = [
            {"key": "sk-new-key", "status": "success"},
            Exception("Import failed for this key")
        ]
        
        result = cli_runner.invoke(cli, [
            "keys", "import",
            "--source-base-url", "https://source.example.com"
        ])
        
        assert result.exit_code == 0  # Command completes even with partial failures
        assert "✓ Imported key: success-key" in result.output
        assert "✗ Failed to import key fail-key" in result.output
        assert "Successfully imported: 1" in result.output
        assert "Failed to import: 1" in result.output
        assert "Total keys processed: 2" in result.output


def test_keys_import_missing_required_source_url(cli_runner):
    """Test error when required --source-base-url is missing"""
    result = cli_runner.invoke(cli, [
        "keys", "import",
        "--dry-run"
    ])
    
    assert result.exit_code != 0
    assert "Missing option" in result.output or "required" in result.output.lower()


def test_keys_import_with_all_key_properties(mock_keys_client, cli_runner):
    """Test import preserves all key properties (models, aliases, config, etc.)"""
    with patch("litellm.proxy.client.cli.commands.keys.KeysManagementClient") as MockClient:
        mock_source_instance = MockClient.return_value
        mock_dest_instance = MockClient.return_value
        
        mock_source_instance.list.side_effect = [
            {
                "keys": [
                    {
                        "key_alias": "full-key",
                        "user_id": "user@example.com",
                        "team_id": "team-123",
                        "budget_id": "budget-456", 
                        "models": ["gpt-4", "gpt-3.5-turbo"],
                        "aliases": {"custom-model": "gpt-4"},
                        "spend": 50.0,
                        "config": {"max_tokens": 1000}
                    }
                ]
            },
            {"keys": []}
        ]
        
        mock_dest_instance.generate.return_value = {"key": "sk-imported", "status": "success"}
        
        result = cli_runner.invoke(cli, [
            "keys", "import",
            "--source-base-url", "https://source.example.com"
        ])
        
        assert result.exit_code == 0
        
        # Verify all properties were passed to generate
        mock_dest_instance.generate.assert_called_once_with(
            models=["gpt-4", "gpt-3.5-turbo"],
            aliases={"custom-model": "gpt-4"},
            spend=50.0,
            key_alias="full-key",
            team_id="team-123",
            user_id="user@example.com",
            budget_id="budget-456",
            config={"max_tokens": 1000}
        )
