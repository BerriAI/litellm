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


# ---------------------------------------------------------------------------
# Regression tests for issue #17310: CustomLogger instances registered via
# ``litellm_settings.callbacks`` must also land in input / success / failure
# callback lists (sync + async). Otherwise pass-through endpoint logging and
# ``log_pre_api_call`` silently skip the user callback.
# ---------------------------------------------------------------------------


from litellm.integrations.custom_logger import CustomLogger


class _PRTestCustomLogger(CustomLogger):
    """Minimal CustomLogger subclass used in the tests below."""

    pass


_ALL_CALLBACK_LIST_NAMES = (
    "input_callback",
    "success_callback",
    "failure_callback",
    "_async_input_callback",
    "_async_success_callback",
    "_async_failure_callback",
)


def _snapshot_callback_lists():
    return {name: list(getattr(litellm, name)) for name in _ALL_CALLBACK_LIST_NAMES} | {
        "callbacks": list(litellm.callbacks)
        if isinstance(litellm.callbacks, list)
        else []
    }


def _restore_callback_lists(snap):
    for name in _ALL_CALLBACK_LIST_NAMES:
        getattr(litellm, name).clear()
        getattr(litellm, name).extend(snap[name])
    litellm.callbacks = snap["callbacks"]


def test_initialize_callbacks_on_proxy_registers_custom_logger_into_all_lists(
    monkeypatch,
):
    """A CustomLogger instance passed via the list form must land in
    every one of the six dedicated callback lists in addition to
    ``litellm.callbacks``. Regression for issue #17310."""
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.proxy_server",
        SimpleNamespace(prisma_client=None),
    )
    snap = _snapshot_callback_lists()
    logger = _PRTestCustomLogger()
    try:
        initialize_callbacks_on_proxy(
            value=[logger],
            premium_user=False,
            config_file_path=".",
            litellm_settings={},
            callback_specific_params={},
        )
        assert logger in litellm.callbacks
        for name in _ALL_CALLBACK_LIST_NAMES:
            assert logger in getattr(
                litellm, name
            ), f"CustomLogger missing from litellm.{name}"
    finally:
        _restore_callback_lists(snap)


def test_initialize_callbacks_on_proxy_is_idempotent_for_custom_logger(
    monkeypatch,
):
    """Re-running ``initialize_callbacks_on_proxy`` with the same
    CustomLogger must not duplicate it in any callback list."""
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.proxy_server",
        SimpleNamespace(prisma_client=None),
    )
    snap = _snapshot_callback_lists()
    logger = _PRTestCustomLogger()
    try:
        initialize_callbacks_on_proxy(
            value=[logger],
            premium_user=False,
            config_file_path=".",
            litellm_settings={},
            callback_specific_params={},
        )
        initialize_callbacks_on_proxy(
            value=[logger],
            premium_user=False,
            config_file_path=".",
            litellm_settings={},
            callback_specific_params={},
        )
        for name in _ALL_CALLBACK_LIST_NAMES:
            assert (
                getattr(litellm, name).count(logger) == 1
            ), f"CustomLogger duplicated in litellm.{name}"
    finally:
        _restore_callback_lists(snap)


def test_initialize_callbacks_on_proxy_scalar_value_registers_into_all_lists(
    monkeypatch,
):
    """The scalar (non-list) branch of ``initialize_callbacks_on_proxy``
    must also push CustomLogger instances into all six dedicated lists.
    Regression for issue #17310 (covers the ``else`` branch in addition
    to the list branch)."""
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.proxy_server",
        SimpleNamespace(prisma_client=None),
    )
    logger = _PRTestCustomLogger()
    # Make get_instance_fn return our logger regardless of input path.
    monkeypatch.setattr(
        "litellm.proxy.common_utils.callback_utils.get_instance_fn",
        lambda value, config_file_path=None: logger,
    )
    snap = _snapshot_callback_lists()
    try:
        initialize_callbacks_on_proxy(
            value="my_module.callback_instance",
            premium_user=False,
            config_file_path=".",
            litellm_settings={},
            callback_specific_params={},
        )
        assert logger in litellm.callbacks
        for name in _ALL_CALLBACK_LIST_NAMES:
            assert logger in getattr(
                litellm, name
            ), f"CustomLogger missing from litellm.{name} (scalar branch)"
    finally:
        _restore_callback_lists(snap)


def test_register_custom_loggers_into_all_callback_lists_ignores_non_custom_logger():
    """The helper must skip entries that are not CustomLogger instances."""
    from litellm.proxy.common_utils.callback_utils import (
        _register_custom_loggers_into_all_callback_lists,
    )

    snap = _snapshot_callback_lists()
    plain_obj = object()
    plain_str = "lago"
    try:
        _register_custom_loggers_into_all_callback_lists([plain_obj, plain_str])
        for name in _ALL_CALLBACK_LIST_NAMES:
            assert plain_obj not in getattr(
                litellm, name
            ), f"object() should not be in litellm.{name}"
            assert plain_str not in getattr(
                litellm, name
            ), f"plain string should not be in litellm.{name}"
    finally:
        _restore_callback_lists(snap)


def test_register_custom_loggers_into_all_callback_lists_is_idempotent():
    """Direct test of the helper: calling twice does not duplicate entries."""
    from litellm.proxy.common_utils.callback_utils import (
        _register_custom_loggers_into_all_callback_lists,
    )

    snap = _snapshot_callback_lists()
    logger = _PRTestCustomLogger()
    try:
        _register_custom_loggers_into_all_callback_lists([logger])
        _register_custom_loggers_into_all_callback_lists([logger])
        for name in _ALL_CALLBACK_LIST_NAMES:
            assert (
                getattr(litellm, name).count(logger) == 1
            ), f"helper not idempotent for litellm.{name}"
    finally:
        _restore_callback_lists(snap)
