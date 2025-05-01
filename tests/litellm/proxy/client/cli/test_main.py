import json
import os
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from litellm.proxy.client.cli import cli


@patch.dict(os.environ, {
    "LITELLM_PROXY_URL": "http://localhost:4000",
    "LITELLM_PROXY_API_KEY": "sk-test"
})
def test_models_list_json_format():
    """Test the models list command with JSON output format"""
    runner = CliRunner()
    
    # Mock data that would be returned by the API
    mock_models = [
        {
            "id": "model-123",
            "object": "model",
            "created": 1699848889,
            "owned_by": "organization-123"
        },
        {
            "id": "model-456",
            "object": "model",
            "created": 1699848890,
            "owned_by": "organization-456"
        }
    ]
    
    # Mock the Client class and its models.list() method
    with patch("litellm.proxy.client.cli.groups.models.Client") as MockClient:
        # Configure the mock
        mock_client_instance = MagicMock()
        mock_client_instance.models.list.return_value = mock_models
        MockClient.return_value = mock_client_instance
        
        # Run the command
        result = runner.invoke(cli, ["models", "list", "--format", "json"])
        
        # Check that the command succeeded
        assert result.exit_code == 0
        
        # Parse the output and verify it matches our mock data
        output_data = json.loads(result.output)
        assert output_data == mock_models
        
        # Verify the client was called correctly with env var values
        MockClient.assert_called_once_with(
            base_url="http://localhost:4000",
            api_key="sk-test"
        )
        mock_client_instance.models.list.assert_called_once()


@patch.dict(os.environ, {
    "LITELLM_PROXY_URL": "http://localhost:4000",
    "LITELLM_PROXY_API_KEY": "sk-test"
})
def test_models_list_table_format():
    """Test the models list command with table output format"""
    runner = CliRunner()
    
    # Use ISO format datetime string
    test_datetime = "2025-04-29T21:31:43.843000+00:00"
    
    # Mock data that would be returned by the API
    mock_models = [
        {
            "id": "model-123",
            "object": "model",
            "created": test_datetime,
            "owned_by": "organization-123"
        }
    ]
    
    # Mock the Client class and its models.list() method
    with patch("litellm.proxy.client.cli.groups.models.Client") as MockClient:
        # Configure the mock
        mock_client_instance = MagicMock()
        mock_client_instance.models.list.return_value = mock_models
        MockClient.return_value = mock_client_instance
        
        # Run the command
        result = runner.invoke(cli, ["models", "list"])
        
        # Check that the command succeeded
        assert result.exit_code == 0
        
        # Verify the output contains expected table elements
        assert "ID" in result.output
        assert "Object" in result.output
        assert "Created" in result.output
        assert "Owned By" in result.output
        assert "model-123" in result.output
        assert "organization-123" in result.output
        
        # Verify timestamp format (should show "2025-04-29 21:31" for test_datetime)
        assert "2025-04-29 21:31" in result.output
        # Verify seconds and microseconds are not shown
        assert "21:31:43" not in result.output
        assert "843000" not in result.output
        
        # Verify the client was called correctly with env var values
        MockClient.assert_called_once_with(
            base_url="http://localhost:4000",
            api_key="sk-test"
        )
        mock_client_instance.models.list.assert_called_once()


@patch.dict(os.environ, {
    "LITELLM_PROXY_URL": "http://localhost:4000",
    "LITELLM_PROXY_API_KEY": "sk-test"
})
def test_models_list_with_base_url():
    """Test the models list command with custom base URL overriding env var"""
    runner = CliRunner()
    custom_base_url = "http://custom.server:8000"
    
    with patch("litellm.proxy.client.cli.groups.models.Client") as MockClient:
        # Configure the mock
        mock_client_instance = MagicMock()
        mock_client_instance.models.list.return_value = []
        MockClient.return_value = mock_client_instance
        
        # Run the command with custom base URL
        result = runner.invoke(cli, [
            "--base-url", custom_base_url,
            "models", "list"
        ])
        
        # Check that the command succeeded
        assert result.exit_code == 0
        
        # Verify the client was created with the custom base URL (overriding env var)
        MockClient.assert_called_once_with(
            base_url=custom_base_url,
            api_key="sk-test"  # Should still use env var for API key
        )


@patch.dict(os.environ, {
    "LITELLM_PROXY_URL": "http://localhost:4000",
    "LITELLM_PROXY_API_KEY": "sk-test"
})
def test_models_list_with_api_key():
    """Test the models list command with API key overriding env var"""
    runner = CliRunner()
    custom_api_key = "custom-test-key"
    
    with patch("litellm.proxy.client.cli.groups.models.Client") as MockClient:
        # Configure the mock
        mock_client_instance = MagicMock()
        mock_client_instance.models.list.return_value = []
        MockClient.return_value = mock_client_instance
        
        # Run the command with custom API key
        result = runner.invoke(cli, [
            "--api-key", custom_api_key,
            "models", "list"
        ])
        
        # Check that the command succeeded
        assert result.exit_code == 0
        
        # Verify the client was created with the custom API key (overriding env var)
        MockClient.assert_called_once_with(
            base_url="http://localhost:4000",  # Should still use env var for base URL
            api_key=custom_api_key
        )


@patch.dict(os.environ, {
    "LITELLM_PROXY_URL": "http://localhost:4000",
    "LITELLM_PROXY_API_KEY": "sk-test"
})
def test_models_list_error_handling():
    """Test error handling in the models list command"""
    runner = CliRunner()
    
    with patch("litellm.proxy.client.cli.groups.models.Client") as MockClient:
        # Configure the mock to raise an exception
        mock_client_instance = MagicMock()
        mock_client_instance.models.list.side_effect = Exception("API Error")
        MockClient.return_value = mock_client_instance
        
        # Run the command
        result = runner.invoke(cli, ["models", "list"])
        
        # Check that the command failed
        assert result.exit_code != 0
        assert "API Error" in str(result.exception)
        
        # Verify the client was created with env var values
        MockClient.assert_called_once_with(
            base_url="http://localhost:4000",
            api_key="sk-test"
        )


@patch.dict(os.environ, {
    "LITELLM_PROXY_URL": "http://localhost:4000",
    "LITELLM_PROXY_API_KEY": "sk-test"
})
def test_models_info_json_format():
    """Test the models info command with JSON output format"""
    runner = CliRunner()
    
    # Mock data that would be returned by the API
    mock_models_info = [
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "gpt-4",
                "litellm_credential_name": "openai-1"
            },
            "model_info": {
                "id": "model-123",
                "created_at": "2025-04-29T21:31:43.843000+00:00",
                "updated_at": "2025-04-29T21:31:43.843000+00:00",
                "input_cost_per_token": 0.00001,
                "output_cost_per_token": 0.00002
            }
        }
    ]
    
    # Mock the Client class and its models.info() method
    with patch("litellm.proxy.client.cli.groups.models.Client") as MockClient:
        # Configure the mock
        mock_client_instance = MagicMock()
        mock_client_instance.models.info.return_value = mock_models_info
        MockClient.return_value = mock_client_instance
        
        # Run the command
        result = runner.invoke(cli, ["models", "info", "--format", "json"])
        
        # Check that the command succeeded
        assert result.exit_code == 0
        
        # Parse the output and verify it matches our mock data
        output_data = json.loads(result.output)
        assert output_data == mock_models_info
        
        # Verify the client was called correctly with env var values
        MockClient.assert_called_once_with(
            base_url="http://localhost:4000",
            api_key="sk-test"
        )
        mock_client_instance.models.info.assert_called_once()


@patch.dict(os.environ, {
    "LITELLM_PROXY_URL": "http://localhost:4000",
    "LITELLM_PROXY_API_KEY": "sk-test"
})
def test_models_info_table_format():
    """Test the models info command with table output format"""
    runner = CliRunner()
    
    # Mock data that would be returned by the API
    mock_models_info = [
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "gpt-4",
                "litellm_credential_name": "openai-1"
            },
            "model_info": {
                "id": "model-123",
                "created_at": "2025-04-29T21:31:43.843000+00:00",
                "updated_at": "2025-04-29T21:31:43.843000+00:00",
                "input_cost_per_token": 0.00001,
                "output_cost_per_token": 0.00002
            }
        }
    ]
    
    # Mock the Client class and its models.info() method
    with patch("litellm.proxy.client.cli.groups.models.Client") as MockClient:
        # Configure the mock
        mock_client_instance = MagicMock()
        mock_client_instance.models.info.return_value = mock_models_info
        MockClient.return_value = mock_client_instance
        
        # Run the command with default columns
        result = runner.invoke(cli, ["models", "info"])
        
        # Check that the command succeeded
        assert result.exit_code == 0
        
        # Verify the output contains expected table elements
        assert "Public Model" in result.output
        assert "Upstream Model" in result.output
        assert "Updated At" in result.output
        assert "gpt-4" in result.output
        assert "2025-04-29 21:31" in result.output
        
        # Verify seconds and microseconds are not shown
        assert "21:31:43" not in result.output
        assert "843000" not in result.output
        
        # Verify the client was called correctly with env var values
        MockClient.assert_called_once_with(
            base_url="http://localhost:4000",
            api_key="sk-test"
        )
        mock_client_instance.models.info.assert_called_once() 