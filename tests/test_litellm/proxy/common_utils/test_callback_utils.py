import sys
import os
from types import SimpleNamespace

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.common_utils.callback_utils import (
    initialize_callbacks_on_proxy,
    get_remaining_tokens_and_requests_from_request_data,
    normalize_callback_names,
)
import litellm

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
    return_value=["GENERIC_LOGGER_ENDPOINT", "GENERIC_LOGGER_HEADERS"],
)
def test_process_callback_generic_api_falls_back_to_os_env(
    mock_get_env_vars, monkeypatch
):
    monkeypatch.setenv("GENERIC_LOGGER_ENDPOINT", "https://callback.example.com")
    monkeypatch.setenv("GENERIC_LOGGER_HEADERS", "Authorization=Bearer token")

    result = process_callback(
        _callback="generic_api",
        callback_type="success",
        environment_variables={},
    )

    assert result["variables"] == {
        "GENERIC_LOGGER_ENDPOINT": "https://callback.example.com",
        "GENERIC_LOGGER_HEADERS": "Authorization=Bearer token",
    }


@patch(
    "litellm.proxy.common_utils.callback_utils.CustomLogger.get_callback_env_vars",
    return_value=["GENERIC_LOGGER_ENDPOINT", "GENERIC_LOGGER_HEADERS"],
)
def test_process_callback_custom_callback_api_falls_back_to_os_env(
    mock_get_env_vars, monkeypatch
):
    monkeypatch.setenv("GENERIC_LOGGER_ENDPOINT", "https://callback.example.com")
    monkeypatch.setenv("GENERIC_LOGGER_HEADERS", "Authorization=Bearer token")

    result = process_callback(
        _callback="custom_callback_api",
        callback_type="success",
        environment_variables={},
    )

    assert result["variables"] == {
        "GENERIC_LOGGER_ENDPOINT": "https://callback.example.com",
        "GENERIC_LOGGER_HEADERS": "Authorization=Bearer token",
    }


@patch(
    "litellm.proxy.common_utils.callback_utils.CustomLogger.get_callback_env_vars",
    return_value=["GENERIC_LOGGER_ENDPOINT", "GENERIC_LOGGER_HEADERS"],
)
def test_process_callback_config_value_wins_over_os_env(mock_get_env_vars, monkeypatch):
    monkeypatch.setenv("GENERIC_LOGGER_ENDPOINT", "https://env.example.com")
    monkeypatch.setenv("GENERIC_LOGGER_HEADERS", "Authorization=Bearer env")

    result = process_callback(
        _callback="generic_api",
        callback_type="success",
        environment_variables={
            "GENERIC_LOGGER_ENDPOINT": "https://config.example.com",
            "GENERIC_LOGGER_HEADERS": "Authorization=Bearer config",
        },
    )

    assert result["variables"] == {
        "GENERIC_LOGGER_ENDPOINT": "https://config.example.com",
        "GENERIC_LOGGER_HEADERS": "Authorization=Bearer config",
    }


@patch(
    "litellm.proxy.common_utils.callback_utils.CustomLogger.get_callback_env_vars",
    return_value=["LANGFUSE_PUBLIC_KEY"],
)
def test_process_callback_falls_back_to_os_env_for_registered_callback_vars(
    mock_get_env_vars, monkeypatch
):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-env")

    result = process_callback(
        _callback="langfuse",
        callback_type="success",
        environment_variables={},
    )

    assert result["variables"] == {"LANGFUSE_PUBLIC_KEY": "pk-env"}


def test_normalize_callback_names_none_returns_empty_list():
    assert normalize_callback_names(None) == []
    assert normalize_callback_names([]) == []


def test_normalize_callback_names_lowercases_strings():
    assert normalize_callback_names(["SQS", "S3", "CUSTOM_CALLBACK"]) == [
        "sqs",
        "s3",
        "custom_callback",
    ]


def test_initialize_callbacks_on_proxy_instantiates_compression_interception(
    monkeypatch,
):
    dummy_callback = object()
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.proxy_server",
        SimpleNamespace(prisma_client=None),
    )
    monkeypatch.setattr(
        "litellm.integrations.compression_interception.handler.CompressionInterceptionLogger.initialize_from_proxy_config",
        lambda litellm_settings, callback_specific_params: dummy_callback,
    )

    original_callbacks = (
        list(litellm.callbacks) if isinstance(litellm.callbacks, list) else []
    )
    litellm.callbacks = []
    try:
        initialize_callbacks_on_proxy(
            value=["compression_interception"],
            premium_user=False,
            config_file_path=".",
            litellm_settings={"compression_interception_params": {"enabled": True}},
            callback_specific_params={},
        )
        assert dummy_callback in litellm.callbacks
        assert "compression_interception" not in litellm.callbacks
    finally:
        litellm.callbacks = original_callbacks
