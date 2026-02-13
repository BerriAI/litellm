import sys
import os

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.common_utils.callback_utils import (
    get_model_group_from_request_data,
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
def test_process_callback_with_callback_params_returns_params_in_result(
    mock_get_env_vars,
):
    """When callback_params is provided (e.g. for websearch_interception), it is included in the result for UI round-trip."""
    result = process_callback(
        _callback="websearch_interception",
        callback_type="success_and_failure",
        environment_variables={},
        callback_params={
            "enabled_providers": ["bedrock", "azure"],
            "search_tool_name": "perplexity-search",
        },
    )

    assert result["name"] == "websearch_interception"
    assert result["type"] == "success_and_failure"
    assert result["variables"] == {}
    assert result["params"] == {
        "enabled_providers": ["bedrock", "azure"],
        "search_tool_name": "perplexity-search",
    }


@patch(
    "litellm.proxy.common_utils.callback_utils.CustomLogger.get_callback_env_vars",
    return_value=[],
)
def test_process_callback_without_callback_params_omits_params_key(mock_get_env_vars):
    """When callback_params is not provided, result does not include 'params' key."""
    result = process_callback(
        _callback="langfuse",
        callback_type="success",
        environment_variables={},
    )

    assert result["name"] == "langfuse"
    assert "params" not in result


def test_normalize_callback_names_none_returns_empty_list():
    # EXTRA BUG FIX #2: normalize_callback_names accepts None and returns []
    assert normalize_callback_names(None) == []
    assert normalize_callback_names([]) == []


def test_normalize_callback_names_lowercases_strings():
    assert normalize_callback_names(["SQS", "S3", "CUSTOM_CALLBACK"]) == ["sqs", "s3", "custom_callback"]


def test_remaining_tokens_requests_returns_empty_when_metadata_empty_or_missing():
    """Empty or missing metadata returns {}."""
    assert get_remaining_tokens_and_requests_from_request_data({}) == {}
    assert get_remaining_tokens_and_requests_from_request_data({"metadata": {}}) == {}
    assert get_remaining_tokens_and_requests_from_request_data({"metadata": None}) == {}


def test_remaining_tokens_requests_sanitizes_colon_and_slash_in_header_keys():
    """model_group with ':' in name is sanitized to '-' in header keys (h11-safe)."""
    data = {
        "metadata": {
            "model_group": "openrouter:google/gemini-2.0-flash",
            "litellm-key-remaining-requests-openrouter:google/gemini-2.0-flash": 10,
            "litellm-key-remaining-tokens-openrouter:google/gemini-2.0-flash": 20,
        }
    }
    headers = get_remaining_tokens_and_requests_from_request_data(data)
    # Header keys must use - not : or / for h11 safety
    assert "x-litellm-key-remaining-requests-openrouter-google-gemini-2.0-flash" in headers
    assert headers["x-litellm-key-remaining-requests-openrouter-google-gemini-2.0-flash"] == 10
    assert "x-litellm-key-remaining-tokens-openrouter-google-gemini-2.0-flash" in headers
    assert headers["x-litellm-key-remaining-tokens-openrouter-google-gemini-2.0-flash"] == 20


def test_get_model_group_returns_value_when_present():
    """Data with metadata.model_group returns it."""
    assert get_model_group_from_request_data({
        "metadata": {"model_group": "my-model-group"},
    }) == "my-model-group"


def test_get_model_group_returns_none_when_metadata_missing():
    """Missing metadata returns None."""
    assert get_model_group_from_request_data({}) is None
    assert get_model_group_from_request_data({"other": "key"}) is None


def test_get_model_group_returns_none_when_model_group_key_missing():
    """Metadata without model_group returns None."""
    assert get_model_group_from_request_data({"metadata": {}}) is None
    assert get_model_group_from_request_data({"metadata": {"other": "x"}}) is None


def test_normalize_callback_names_lowercases_strings_preserves_non_strings():
    """Iterable with non-string items leaves them unchanged (e.g. [1, 'SQS'] -> [1, 'sqs'])."""
    assert normalize_callback_names([1, "SQS"]) == [1, "sqs"]
    assert normalize_callback_names(["a", None, "B"]) == ["a", None, "b"]



