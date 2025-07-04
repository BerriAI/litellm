# stdlib imports
import json
import os
import time
from unittest.mock import patch

import pytest

# third party imports
from click.testing import CliRunner

# local imports
from litellm.proxy.client.cli import cli
from litellm.proxy.client.cli.commands.models import (
    format_cost_per_1k_tokens,
    format_iso_datetime_str,
    format_timestamp,
)


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
    with patch.dict(
        os.environ,
        {
            "LITELLM_PROXY_URL": "http://localhost:4000",
            "LITELLM_PROXY_API_KEY": "sk-test",
        },
    ):
        yield


@pytest.fixture
def mock_models_list(mock_client):
    """Fixture to set up common mocking pattern for models list tests"""
    mock_client.return_value.models.list.return_value = [
        {
            "id": "model-123",
            "object": "model",
            "created": 1699848889,
            "owned_by": "organization-123",
        },
        {
            "id": "model-456",
            "object": "model",
            "created": 1699848890,
            "owned_by": "organization-456",
        },
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


@pytest.fixture
def force_utc_tz():
    """Fixture to force UTC timezone for tests that depend on system TZ."""
    old_tz = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    if hasattr(time, "tzset"):
        time.tzset()
    yield
    # Restore previous TZ
    if old_tz is not None:
        os.environ["TZ"] = old_tz
    else:
        if "TZ" in os.environ:
            del os.environ["TZ"]
    if hasattr(time, "tzset"):
        time.tzset()


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
    mock_models_list.assert_called_once_with(
        base_url="http://localhost:4000", api_key="sk-test"
    )
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
    mock_models_list.assert_called_once_with(
        base_url="http://localhost:4000", api_key="sk-test"
    )
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
    mock_client.assert_called_once_with(
        base_url="http://localhost:4000", api_key="sk-test"
    )


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
    mock_models_info.assert_called_once_with(
        base_url="http://localhost:4000", api_key="sk-test"
    )
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
    mock_models_info.assert_called_once_with(
        base_url="http://localhost:4000", api_key="sk-test"
    )
    mock_models_info.return_value.models.info.assert_called_once()


def test_models_import_only_models_matching_regex(tmp_path, mock_client, cli_runner):
    """Test the --only-models-matching-regex option for models import command"""
    # Prepare a YAML file with a mix of models
    yaml_content = {
        "model_list": [
            {
                "model_name": "gpt-4-model",
                "litellm_params": {"model": "gpt-4"},
                "model_info": {"id": "id-1"},
            },
            {
                "model_name": "gpt-3.5-model",
                "litellm_params": {"model": "gpt-3.5-turbo"},
                "model_info": {"id": "id-2"},
            },
            {
                "model_name": "llama2-model",
                "litellm_params": {"model": "llama2"},
                "model_info": {"id": "id-3"},
            },
            {
                "model_name": "other-model",
                "litellm_params": {"model": "other"},
                "model_info": {"id": "id-4"},
            },
        ]
    }
    import yaml as pyyaml

    yaml_file = tmp_path / "models.yaml"
    with open(yaml_file, "w") as f:
        pyyaml.safe_dump(yaml_content, f)

    # Patch client.models.new to track calls
    mock_new = mock_client.return_value.models.new

    # Only match models containing 'gpt' in their litellm_params.model
    result = cli_runner.invoke(
        cli, ["models", "import", str(yaml_file), "--only-models-matching-regex", "gpt"]
    )

    # Should succeed
    assert result.exit_code == 0
    # Only the two gpt models should be imported
    calls = [call.kwargs["model_params"]["model"] for call in mock_new.call_args_list]
    assert set(calls) == {"gpt-4", "gpt-3.5-turbo"}
    # Should not include llama2 or other
    assert "llama2" not in calls
    assert "other" not in calls
    # Output summary should mention the correct providers
    assert "gpt-4".split("-")[0] in result.output or "gpt" in result.output


def test_models_import_only_access_groups_matching_regex(
    tmp_path, mock_client, cli_runner
):
    """Test the --only-access-groups-matching-regex option for models import command"""
    # Prepare a YAML file with a mix of models
    yaml_content = {
        "model_list": [
            {
                "model_name": "gpt-4-model",
                "litellm_params": {"model": "gpt-4"},
                "model_info": {
                    "id": "id-1",
                    "access_groups": ["beta-models", "prod-models"],
                },
            },
            {
                "model_name": "gpt-3.5-model",
                "litellm_params": {"model": "gpt-3.5-turbo"},
                "model_info": {"id": "id-2", "access_groups": ["alpha-models"]},
            },
            {
                "model_name": "llama2-model",
                "litellm_params": {"model": "llama2"},
                "model_info": {"id": "id-3", "access_groups": ["beta-models"]},
            },
            {
                "model_name": "other-model",
                "litellm_params": {"model": "other"},
                "model_info": {"id": "id-4", "access_groups": ["other-group"]},
            },
            {
                "model_name": "no-access-group-model",
                "litellm_params": {"model": "no-access"},
                "model_info": {"id": "id-5"},
            },
        ]
    }
    import yaml as pyyaml

    yaml_file = tmp_path / "models.yaml"
    with open(yaml_file, "w") as f:
        pyyaml.safe_dump(yaml_content, f)

    # Patch client.models.new to track calls
    mock_new = mock_client.return_value.models.new

    # Only match models with access_groups containing 'beta'
    result = cli_runner.invoke(
        cli,
        [
            "models",
            "import",
            str(yaml_file),
            "--only-access-groups-matching-regex",
            "beta",
        ],
    )

    # Should succeed
    assert result.exit_code == 0
    # Only the two models with 'beta-models' in access_groups should be imported
    calls = [call.kwargs["model_params"]["model"] for call in mock_new.call_args_list]
    assert set(calls) == {"gpt-4", "llama2"}
    # Should not include gpt-3.5, other, or no-access
    assert "gpt-3.5-turbo" not in calls
    assert "other" not in calls
    assert "no-access" not in calls
    # Output summary should mention the correct providers
    assert "gpt-4".split("-")[0] in result.output or "gpt" in result.output


@pytest.mark.parametrize(
    "input_str,expected",
    [
        (None, ""),
        ("", ""),
        ("2024-05-01T12:34:56Z", "2024-05-01 12:34"),
        ("2024-05-01T12:34:56+00:00", "2024-05-01 12:34"),
        ("2024-05-01T12:34:56.123456+00:00", "2024-05-01 12:34"),
        ("2024-05-01T12:34:56.123456Z", "2024-05-01 12:34"),
        ("2024-05-01T12:34:56-04:00", "2024-05-01 12:34"),
        ("2024-05-01", "2024-05-01 00:00"),
        ("not-a-date", "not-a-date"),
    ],
)
def test_format_iso_datetime_str(input_str, expected):
    assert format_iso_datetime_str(input_str) == expected


@pytest.mark.parametrize(
    "input_val,expected",
    [
        (None, ""),
        (1699848889, "2023-11-13 04:14"),
        (1699848889.0, "2023-11-13 04:14"),
        ("not-a-timestamp", "not-a-timestamp"),
        ([1, 2, 3], "[1, 2, 3]"),
    ],
)
def test_format_timestamp(input_val, expected, force_utc_tz):
    actual = format_timestamp(input_val)
    if actual != expected:
        print(f"input: {input_val}, expected: {expected}, actual: {actual}")
    assert actual == expected


@pytest.mark.parametrize(
    "input_val,expected",
    [
        (None, ""),
        (0, "$0.0000"),
        (0.0, "$0.0000"),
        (0.00001, "$0.0100"),
        (0.00002, "$0.0200"),
        (1, "$1000.0000"),
        (1.5, "$1500.0000"),
        ("0.00001", "$0.0100"),
        ("1.5", "$1500.0000"),
        ("not-a-number", "not-a-number"),
        (1e-10, "$0.0000"),
    ],
)
def test_format_cost_per_1k_tokens(input_val, expected):
    actual = format_cost_per_1k_tokens(input_val)
    if actual != expected:
        print(f"input: {input_val}, expected: {expected}, actual: {actual}")
    assert actual == expected
