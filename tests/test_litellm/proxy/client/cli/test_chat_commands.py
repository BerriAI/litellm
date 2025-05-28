import json
from unittest.mock import MagicMock, patch

import pytest
import requests
from click.testing import CliRunner

from litellm.proxy.client.cli.main import cli


@pytest.fixture
def mock_chat_client():
    with patch("litellm.proxy.client.cli.commands.chat.ChatClient") as mock:
        yield mock


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_chat_completions_success(cli_runner, mock_chat_client):
    # Mock response data
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }
    mock_instance = mock_chat_client.return_value
    mock_instance.completions.return_value = mock_response

    # Run command
    result = cli_runner.invoke(
        cli,
        [
            "chat",
            "completions",
            "gpt-4",
            "-m",
            "user:Hello!",
            "--temperature",
            "0.7",
            "--max-tokens",
            "100",
        ],
    )

    # Verify
    assert result.exit_code == 0
    output_data = json.loads(result.output)
    assert output_data == mock_response
    mock_instance.completions.assert_called_once_with(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}],
        temperature=0.7,
        max_tokens=100,
        top_p=None,
        n=None,
        presence_penalty=None,
        frequency_penalty=None,
        user=None,
    )


def test_chat_completions_multiple_messages(cli_runner, mock_chat_client):
    # Mock response data
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Paris has a population of about 2.2 million.",
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }
    mock_instance = mock_chat_client.return_value
    mock_instance.completions.return_value = mock_response

    # Run command
    result = cli_runner.invoke(
        cli,
        [
            "chat",
            "completions",
            "gpt-4",
            "-m",
            "system:You are a helpful assistant",
            "-m",
            "user:What's the population of Paris?",
        ],
    )

    # Verify
    assert result.exit_code == 0
    output_data = json.loads(result.output)
    assert output_data == mock_response
    mock_instance.completions.assert_called_once_with(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What's the population of Paris?"},
        ],
        temperature=None,
        max_tokens=None,
        top_p=None,
        n=None,
        presence_penalty=None,
        frequency_penalty=None,
        user=None,
    )


def test_chat_completions_no_messages(cli_runner, mock_chat_client):
    # Run command without any messages
    result = cli_runner.invoke(cli, ["chat", "completions", "gpt-4"])

    # Verify
    assert result.exit_code == 2
    assert "At least one message is required" in result.output
    mock_instance = mock_chat_client.return_value
    mock_instance.completions.assert_not_called()


def test_chat_completions_invalid_message_format(cli_runner, mock_chat_client):
    # Run command with invalid message format
    result = cli_runner.invoke(
        cli, ["chat", "completions", "gpt-4", "-m", "invalid-format"]
    )

    # Verify
    assert result.exit_code == 2
    assert "Invalid message format" in result.output
    mock_instance = mock_chat_client.return_value
    mock_instance.completions.assert_not_called()


def test_chat_completions_http_error(cli_runner, mock_chat_client):
    # Mock HTTP error
    mock_instance = mock_chat_client.return_value
    mock_error_response = MagicMock()
    mock_error_response.status_code = 400
    mock_error_response.json.return_value = {
        "error": "Invalid request",
        "message": "Invalid model specified",
    }
    mock_instance.completions.side_effect = requests.exceptions.HTTPError(
        response=mock_error_response
    )

    # Run command
    result = cli_runner.invoke(
        cli, ["chat", "completions", "invalid-model", "-m", "user:Hello"]
    )

    # Verify
    assert result.exit_code == 1
    assert "Error: HTTP 400" in result.output
    assert "Invalid request" in result.output
    assert "Invalid model specified" in result.output


def test_chat_completions_all_parameters(cli_runner, mock_chat_client):
    # Mock response data
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Response with all parameters set",
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }
    mock_instance = mock_chat_client.return_value
    mock_instance.completions.return_value = mock_response

    # Run command with all available parameters
    result = cli_runner.invoke(
        cli,
        [
            "chat",
            "completions",
            "gpt-4",
            "-m",
            "user:Test message",
            "--temperature",
            "0.7",
            "--top-p",
            "0.9",
            "--n",
            "1",
            "--max-tokens",
            "100",
            "--presence-penalty",
            "0.5",
            "--frequency-penalty",
            "0.5",
            "--user",
            "test-user",
        ],
    )

    # Verify
    assert result.exit_code == 0
    output_data = json.loads(result.output)
    assert output_data == mock_response
    mock_instance.completions.assert_called_once_with(
        model="gpt-4",
        messages=[{"role": "user", "content": "Test message"}],
        temperature=0.7,
        top_p=0.9,
        n=1,
        max_tokens=100,
        presence_penalty=0.5,
        frequency_penalty=0.5,
        user="test-user",
    )
