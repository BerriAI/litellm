import sys
import os

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.common_utils.callback_utils import (
    get_remaining_tokens_and_requests_from_request_data,
    normalize_callback_names,
)

from unittest.mock import patch
from litellm.proxy.common_utils.callback_utils import process_callback


def test_get_remaining_tokens_and_requests_from_request_data():
    model_group = "openrouter/google/gemini-2.0-flash-001"
    casedata = {
        "metadata": {
            "model_group": model_group,
            f"litellm-key-remaining-requests-{model_group}": 100,
            f"litellm-key-remaining-tokens-{model_group}": 200,
        }
    }

    headers = get_remaining_tokens_and_requests_from_request_data(casedata)

    expected_name = "openrouter-google-gemini-2.0-flash-001"
    assert headers == {
        f"x-litellm-key-remaining-requests-{expected_name}": 100,
        f"x-litellm-key-remaining-tokens-{expected_name}": 200,
    }


@patch(
    "litellm.proxy.common_utils.callback_utils.CustomLogger.get_callback_env_vars",
    return_value=["API_KEY", "MISSING_VAR"],
)
def test_process_callback_with_env_vars(mock_get_env_vars):
    environment_variables = {
        "API_KEY": "PLAIN_VALUE",
        "UNUSED": "SHOULD_BE_IGNORED",
    }

    result = process_callback(
        _callback="my_callback",
        callback_type="input",
        environment_variables=environment_variables,
    )

    assert result["name"] == "my_callback"
    assert result["type"] == "input"
    assert result["variables"] == {
        "API_KEY": "PLAIN_VALUE",
        "MISSING_VAR": None,
    }


@patch(
    "litellm.proxy.common_utils.callback_utils.CustomLogger.get_callback_env_vars",
    return_value=[],
)
def test_process_callback_with_no_required_env_vars(mock_get_env_vars):
    result = process_callback(
        _callback="another_callback",
        callback_type="output",
        environment_variables={"SHOULD_NOT_BE_USED": "VALUE"},
    )

    assert result["name"] == "another_callback"
    assert result["type"] == "output"
    assert result["variables"] == {}


@patch(
    "litellm.proxy.common_utils.callback_utils.CustomLogger.get_callback_env_vars",
    return_value=[],
)
def test_process_callback_returns_correct_type_for_each_callback_mode(mock_get_env_vars):
    """
    Verify that process_callback returns the correct 'type' field for success,
    failure, and success_and_failure callback modes. The UI relies on this field
    to display the correct mode badge.

    Regression test: the UI was reading 'mode' instead of 'type', causing all
    callbacks to display as 'Success'. The frontend was fixed to read 'type'.
    This test ensures the backend contract is maintained.
    """
    for callback_type in ["success", "failure", "success_and_failure"]:
        result = process_callback(
            _callback="s3_v2",
            callback_type=callback_type,
            environment_variables={},
        )
        assert result["type"] == callback_type, (
            f"Expected type '{callback_type}', got '{result['type']}'"
        )
        assert result["name"] == "s3_v2"


def test_normalize_callback_names_none_returns_empty_list():
    assert normalize_callback_names(None) == []
    assert normalize_callback_names([]) == []


def test_normalize_callback_names_lowercases_strings():
    assert normalize_callback_names(["SQS", "S3", "CUSTOM_CALLBACK"]) == ["sqs", "s3", "custom_callback"]

