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


def test_normalize_callback_names_none_returns_empty_list():
    assert normalize_callback_names(None) == []
    assert normalize_callback_names([]) == []


def test_normalize_callback_names_lowercases_strings():
    assert normalize_callback_names(["SQS", "S3", "CUSTOM_CALLBACK"]) == ["sqs", "s3", "custom_callback"]


def test_initialize_callbacks_on_proxy_instantiates_otel():
    """
    Test that initialize_callbacks_on_proxy() actually instantiates the
    OpenTelemetry callback class (not just adding the string "otel").

    Regression test for: when store_model_in_db=true, OTEL callback was
    never instantiated because initialize_callbacks_on_proxy() only added
    the string "otel" to litellm.callbacks without creating the instance.
    """
    import litellm
    from litellm.proxy.common_utils.callback_utils import (
        initialize_callbacks_on_proxy,
    )
    from litellm.integrations.opentelemetry import OpenTelemetry

    # Save original state
    original_callbacks = litellm.callbacks[:]
    original_success = litellm.success_callback[:]
    original_async_success = litellm._async_success_callback[:]
    original_failure = litellm.failure_callback[:]
    original_async_failure = litellm._async_failure_callback[:]

    try:
        # Clear callbacks
        litellm.callbacks = []
        litellm.success_callback = []
        litellm._async_success_callback = []
        litellm.failure_callback = []
        litellm._async_failure_callback = []

        initialize_callbacks_on_proxy(
            value=["otel"],
            premium_user=False,
            config_file_path="",
            litellm_settings={},
        )

        # Verify an OpenTelemetry instance exists in success callbacks
        otel_in_success = any(
            isinstance(cb, OpenTelemetry)
            for cb in litellm.success_callback + litellm._async_success_callback
        )
        assert otel_in_success, (
            "OpenTelemetry instance not found in success callbacks. "
            f"success_callback={litellm.success_callback}, "
            f"_async_success_callback={litellm._async_success_callback}"
        )

        # Verify an OpenTelemetry instance exists in failure callbacks
        otel_in_failure = any(
            isinstance(cb, OpenTelemetry)
            for cb in litellm.failure_callback + litellm._async_failure_callback
        )
        assert otel_in_failure, (
            "OpenTelemetry instance not found in failure callbacks."
        )

        # Verify open_telemetry_logger is set on proxy_server
        from litellm.proxy import proxy_server

        assert proxy_server.open_telemetry_logger is not None, (
            "proxy_server.open_telemetry_logger should be set after "
            "initializing otel callback"
        )
        assert isinstance(proxy_server.open_telemetry_logger, OpenTelemetry)

    finally:
        # Restore original state
        litellm.callbacks = original_callbacks
        litellm.success_callback = original_success
        litellm._async_success_callback = original_async_success
        litellm.failure_callback = original_failure
        litellm._async_failure_callback = original_async_failure

