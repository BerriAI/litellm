# stdlib imports
import json
import os
from unittest.mock import patch, MagicMock

# third party imports
from click.testing import CliRunner
import pytest

# local imports
from litellm.proxy.client.cli import cli
from litellm.proxy.client.cli.commands.models import format_timestamp


@pytest.fixture
def mock_client():
    """Fixture to create a mock client with common setup"""
    with patch("litellm.proxy.client.cli.commands.models.Client") as MockClient:
        yield MockClient


@pytest.fixture
def cli_runner():
    """Fixture for Click CLI runner"""
    return CliRunner()


@pytest.fixture(autouse=True)
def mock_env():
    """Fixture to set up environment variables for all tests"""
    with patch.dict(os.environ, {"LITELLM_PROXY_URL": "http://localhost:4000", "LITELLM_PROXY_API_KEY": "sk-test"}):
        yield


@pytest.fixture
def mock_models_list(mock_client):
    """Fixture to set up common mocking pattern for models list tests"""
    mock_client.return_value.models.list.return_value = [
        {"id": "model-123", "object": "model", "created": 1699848889, "owned_by": "organization-123"},
        {"id": "model-456", "object": "model", "created": 1699848890, "owned_by": "organization-456"},
    ]

    mock_client.assert_not_called()  # Ensure clean slate
    return mock_client


@pytest.fixture
def mock_models_info(mock_client):
    """Fixture to set up models info mock"""
    mock_client.return_value.models.info.return_value = [
        {
            "model_name": "gpt-4",
            "litellm_params": {"model": "gpt-4", "litellm_credential_name": "openai-1"},
            "model_info": {
                "id": "model-123",
                "created_at": "2025-04-29T21:31:43.843000+00:00",
                "updated_at": "2025-04-29T21:31:43.843000+00:00",
                "input_cost_per_token": 0.00001,
                "output_cost_per_token": 0.00002,
            },
        }
    ]

    mock_client.assert_not_called()
    return mock_client


def test_models_list_json_format(mock_models_list, cli_runner):
    """Test the models list command with JSON output format"""
    # Run the command
    result = cli_runner.invoke(cli, ["models", "list", "--format", "json"])

    # Check that the command succeeded
    assert result.exit_code == 0

    # Parse the output and verify it matches our mock data
    output_data = json.loads(result.output)
    assert output_data == mock_models_list.return_value.models.list.return_value

    # Verify the client was called correctly
    mock_models_list.assert_called_once_with(base_url="http://localhost:4000", api_key="sk-test")
    mock_models_list.return_value.models.list.assert_called_once()


def test_models_list_table_format(mock_models_list, cli_runner):
    """Test the models list command with table output format"""
    # Run the command
    result = cli_runner.invoke(cli, ["models", "list"])

    # Check that the command succeeded
    assert result.exit_code == 0

    # Verify the output contains expected table elements
    assert "ID" in result.output
    assert "Object" in result.output
    assert "Created" in result.output
    assert "Owned By" in result.output
    assert "model-123" in result.output
    assert "organization-123" in result.output
    assert format_timestamp(1699848889) in result.output

    # Verify the client was called correctly
    mock_models_list.assert_called_once_with(base_url="http://localhost:4000", api_key="sk-test")
    mock_models_list.return_value.models.list.assert_called_once()


def test_models_list_with_base_url(mock_models_list, cli_runner):
    """Test the models list command with custom base URL overriding env var"""
    custom_base_url = "http://custom.server:8000"

    # Run the command with custom base URL
    result = cli_runner.invoke(cli, ["--base-url", custom_base_url, "models", "list"])

    # Check that the command succeeded
    assert result.exit_code == 0

    # Verify the client was created with the custom base URL (overriding env var)
    mock_models_list.assert_called_once_with(
        base_url=custom_base_url,
        api_key="sk-test",  # Should still use env var for API key
    )


def test_models_list_with_api_key(mock_models_list, cli_runner):
    """Test the models list command with API key overriding env var"""
    custom_api_key = "custom-test-key"

    # Run the command with custom API key
    result = cli_runner.invoke(cli, ["--api-key", custom_api_key, "models", "list"])

    # Check that the command succeeded
    assert result.exit_code == 0

    # Verify the client was created with the custom API key (overriding env var)
    mock_models_list.assert_called_once_with(
        base_url="http://localhost:4000",  # Should still use env var for base URL
        api_key=custom_api_key,
    )


def test_models_list_error_handling(mock_client, cli_runner):
    """Test error handling in the models list command"""
    # Configure mock to raise an exception
    mock_client.return_value.models.list.side_effect = Exception("API Error")

    # Run the command
    result = cli_runner.invoke(cli, ["models", "list"])

    # Check that the command failed
    assert result.exit_code != 0
    assert "API Error" in str(result.exception)

    # Verify the client was created with env var values
    mock_client.assert_called_once_with(base_url="http://localhost:4000", api_key="sk-test")


def test_models_info_json_format(mock_models_info, cli_runner):
    """Test the models info command with JSON output format"""
    # Run the command
    result = cli_runner.invoke(cli, ["models", "info", "--format", "json"])

    # Check that the command succeeded
    assert result.exit_code == 0

    # Parse the output and verify it matches our mock data
    output_data = json.loads(result.output)
    assert output_data == mock_models_info.return_value.models.info.return_value

    # Verify the client was called correctly with env var values
    mock_models_info.assert_called_once_with(base_url="http://localhost:4000", api_key="sk-test")
    mock_models_info.return_value.models.info.assert_called_once()


def test_models_info_table_format(mock_models_info, cli_runner):
    """Test the models info command with table output format"""
    # Run the command with default columns
    result = cli_runner.invoke(cli, ["models", "info"])

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
    mock_models_info.assert_called_once_with(base_url="http://localhost:4000", api_key="sk-test")
    mock_models_info.return_value.models.info.assert_called_once()


def make_models():
    return [
        {"id": "b-model", "object": "model", "created": 200, "owned_by": "org1"},
        {"id": "a-model", "object": "model", "created": 100, "owned_by": "org2"},
        {"id": "c-model", "object": "model", "created": 300, "owned_by": "org3"},
    ]


def _mock_send_with_models(models):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"data": models}
    return lambda *args, **kwargs: mock_response


@pytest.fixture
def mock_models_http_response():
    with patch("requests.Session.send", new=_mock_send_with_models(make_models())):
        yield


def test_models_list_sort_by_model_name_asc(cli_runner, mock_models_http_response):
    result = cli_runner.invoke(cli, [
        "models", "list",
        "--sort-by=model_name", "--sort-order=asc",
    ])
    assert result.exit_code == 0
    # Should be a-model, b-model, c-model in order
    idx_a = result.output.find("a-model")
    idx_b = result.output.find("b-model")
    idx_c = result.output.find("c-model")
    assert 0 <= idx_a < idx_b < idx_c


def test_models_list_sort_by_model_name_desc(cli_runner, mock_models_http_response):
    result = cli_runner.invoke(cli, ["models", "list", "--sort-by", "model_name", "--sort-order", "desc"])
    assert result.exit_code == 0
    # Should be c-model, b-model, a-model in order
    idx_c = result.output.find("c-model")
    idx_b = result.output.find("b-model")
    idx_a = result.output.find("a-model")
    assert 0 <= idx_c < idx_b < idx_a


def test_models_list_sort_by_created_asc(cli_runner, mock_models_http_response):
    result = cli_runner.invoke(cli, ["models", "list", "--sort-by", "created", "--sort-order", "asc"])
    assert result.exit_code == 0
    # Should be a-model (100), b-model (200), c-model (300)
    idx_a = result.output.find("a-model")
    idx_b = result.output.find("b-model")
    idx_c = result.output.find("c-model")
    assert 0 <= idx_a < idx_b < idx_c


def test_models_list_sort_by_created_desc(cli_runner, mock_models_http_response):
    result = cli_runner.invoke(cli, ["models", "list", "--sort-by", "created", "--sort-order", "desc"])
    assert result.exit_code == 0
    # Should be c-model (300), b-model (200), a-model (100)
    idx_c = result.output.find("c-model")
    idx_b = result.output.find("b-model")
    idx_a = result.output.find("a-model")
    assert 0 <= idx_c < idx_b < idx_a
