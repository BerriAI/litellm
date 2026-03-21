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
    assert normalize_callback_names(["SQS", "S3", "CUSTOM_CALLBACK"]) == [
        "sqs",
        "s3",
        "custom_callback",
    ]


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
    from litellm.litellm_core_utils import litellm_logging
    from litellm.proxy import proxy_server

    # Save original state
    original_callbacks = litellm.callbacks[:]
    original_success = litellm.success_callback[:]
    original_async_success = litellm._async_success_callback[:]
    original_failure = litellm.failure_callback[:]
    original_async_failure = litellm._async_failure_callback[:]
    original_otel_logger = getattr(proxy_server, "open_telemetry_logger", None)
    original_in_memory_loggers = litellm_logging._in_memory_loggers[:]

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
        assert otel_in_failure, "OpenTelemetry instance not found in failure callbacks."

        # Verify open_telemetry_logger is set on proxy_server
        assert proxy_server.open_telemetry_logger is not None, (
            "proxy_server.open_telemetry_logger should be set after "
            "initializing otel callback"
        )
        assert isinstance(proxy_server.open_telemetry_logger, OpenTelemetry)

    finally:
        # Restore ALL global state to prevent leaking into other tests
        litellm.callbacks = original_callbacks
        litellm.success_callback = original_success
        litellm._async_success_callback = original_async_success
        litellm.failure_callback = original_failure
        litellm._async_failure_callback = original_async_failure
        proxy_server.open_telemetry_logger = original_otel_logger
        litellm_logging._in_memory_loggers = original_in_memory_loggers


@patch("litellm.utils._add_custom_logger_callback_to_specific_event")
def test_initialize_callbacks_on_proxy_calls_instantiation_for_known_callbacks(
    mock_add_callback,
):
    """
    Verify that initialize_callbacks_on_proxy calls
    _add_custom_logger_callback_to_specific_event for each known callback
    in the config, registering it for both success and failure events.

    This is a unit test that mocks the instantiation to avoid requiring
    real env vars or creating real logger instances.
    """
    from litellm.proxy.common_utils.callback_utils import (
        initialize_callbacks_on_proxy,
    )

    initialize_callbacks_on_proxy(
        value=["otel"],
        premium_user=False,
        config_file_path="",
        litellm_settings={},
    )

    # Should be called twice: once for "success", once for "failure"
    assert mock_add_callback.call_count == 2
    mock_add_callback.assert_any_call("otel", "success")
    mock_add_callback.assert_any_call("otel", "failure")


@patch(
    "litellm.utils._add_custom_logger_callback_to_specific_event",
    side_effect=Exception("Missing LOGFIRE_TOKEN"),
)
def test_initialize_callbacks_on_proxy_handles_instantiation_failure(
    mock_add_callback,
):
    """
    Verify that if a callback fails to instantiate (e.g. missing env vars),
    the proxy does not crash — it logs the error and falls back to adding
    the string for deferred instantiation.
    """
    import litellm
    from litellm.proxy.common_utils.callback_utils import (
        initialize_callbacks_on_proxy,
    )

    original_callbacks = litellm.callbacks[:]

    try:
        litellm.callbacks = []

        initialize_callbacks_on_proxy(
            value=["logfire"],
            premium_user=False,
            config_file_path="",
            litellm_settings={},
        )

        # The string should be added as a fallback so it can be retried later
        assert "logfire" in litellm.callbacks, (
            "Failed callback should be added as string fallback. "
            f"callbacks={litellm.callbacks}"
        )
    finally:
        litellm.callbacks = original_callbacks
