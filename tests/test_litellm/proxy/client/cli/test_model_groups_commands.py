import json
import os
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from litellm.proxy.client.cli import cli

SAMPLE_MODEL_GROUPS: List[Dict[str, Any]] = [
    {
        "model_group": "gpt-4o",
        "mode": "chat",
        "input_cost_per_token": 0.01,
        "output_cost_per_token": 0.02,
    },
    {
        "model_group": "text-embedding-3-small",
        "mode": "embedding",
        "input_cost_per_token": 0.0001,
        "output_cost_per_token": None,
    },
]


@pytest.fixture
def mock_client():
    with patch("litellm.proxy.client.cli.commands.model_groups.Client") as MockClient:
        yield MockClient


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


def test_list_table_format_shows_model_names_and_modes(mock_client, cli_runner):
    mock_client.return_value.model_groups.info.return_value = SAMPLE_MODEL_GROUPS

    result = cli_runner.invoke(cli, ["model-groups", "list"])

    assert result.exit_code == 0, result.output
    assert "gpt-4o" in result.output
    assert "chat" in result.output
    assert "text-embedding-3-small" in result.output
    assert "embedding" in result.output
    assert "0.01" in result.output
    assert "0.02" in result.output

    mock_client.assert_called_once_with(base_url="http://localhost:4000", api_key="sk-test")
    mock_client.return_value.model_groups.info.assert_called_once()


def test_list_table_format_defaults_missing_mode_to_chat(mock_client, cli_runner):
    mock_client.return_value.model_groups.info.return_value = [{"model_group": "some-model"}]

    result = cli_runner.invoke(cli, ["model-groups", "list"])

    assert result.exit_code == 0, result.output
    assert "some-model" in result.output
    assert "chat" in result.output


def test_list_json_format_round_trips_raw_data(mock_client, cli_runner):
    mock_client.return_value.model_groups.info.return_value = SAMPLE_MODEL_GROUPS

    result = cli_runner.invoke(cli, ["model-groups", "list", "--format", "json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == SAMPLE_MODEL_GROUPS


def test_list_with_custom_base_url_and_api_key(mock_client, cli_runner):
    mock_client.return_value.model_groups.info.return_value = []

    result = cli_runner.invoke(
        cli,
        ["--base-url", "http://custom.server:8000", "--api-key", "custom-key", "model-groups", "list"],
    )

    assert result.exit_code == 0, result.output
    mock_client.assert_called_once_with(base_url="http://custom.server:8000", api_key="custom-key")


def test_list_error_handling(mock_client, cli_runner):
    mock_client.return_value.model_groups.info.side_effect = Exception("API Error")

    result = cli_runner.invoke(cli, ["model-groups", "list"])

    assert result.exit_code != 0
    assert "API Error" in str(result.exception)


def test_list_surfaces_clean_error_when_response_is_not_a_list(mock_client, cli_runner):
    mock_client.return_value.model_groups.info.return_value = {"data": SAMPLE_MODEL_GROUPS}

    result = cli_runner.invoke(cli, ["model-groups", "list"])

    assert result.exit_code != 0
    assert result.exception is None or not isinstance(result.exception, AssertionError)
    assert "Unexpected response from /model_group/info" in result.output
