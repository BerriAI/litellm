import os
import sys
import threading
from datetime import datetime

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path

import pytest
from litellm.integrations.langfuse.langfuse import (
    LangFuseLogger,
)
from litellm.integrations.langfuse.langfuse_handler import LangFuseHandler
from litellm.types.utils import StandardCallbackDynamicParams
from litellm.litellm_core_utils.litellm_logging import DynamicLoggingCache
from unittest.mock import Mock, patch


@pytest.fixture
def dynamic_logging_cache():
    return DynamicLoggingCache()


global_langfuse_logger = LangFuseLogger(
    langfuse_public_key="global_public_key",
    langfuse_secret="global_secret",
    langfuse_host="https://global.langfuse.com",
)


# IMPORTANT: Test that passing both langfuse_secret_key and langfuse_secret works
standard_params_1 = StandardCallbackDynamicParams(
    langfuse_public_key="test_public_key",
    langfuse_secret="test_secret",
    langfuse_host="https://test.langfuse.com",
)

standard_params_2 = StandardCallbackDynamicParams(
    langfuse_public_key="test_public_key",
    langfuse_secret_key="test_secret",
    langfuse_host="https://test.langfuse.com",
)


@pytest.mark.parametrize("globalLangfuseLogger", [None, global_langfuse_logger])
@pytest.mark.parametrize("standard_params", [standard_params_1, standard_params_2])
def test_get_langfuse_logger_for_request_with_dynamic_params(
    dynamic_logging_cache, globalLangfuseLogger, standard_params
):
    """
    If StandardCallbackDynamicParams contain langfuse credentials the returned Langfuse logger should use the dynamic params

    the new Langfuse logger should be cached

    Even if globalLangfuseLogger is provided, it should use dynamic params if they are passed
    """

    result = LangFuseHandler.get_langfuse_logger_for_request(
        standard_callback_dynamic_params=standard_params,
        in_memory_dynamic_logger_cache=dynamic_logging_cache,
        globalLangfuseLogger=globalLangfuseLogger,
    )

    assert isinstance(result, LangFuseLogger)
    assert result.public_key == "test_public_key"
    assert result.secret_key == "test_secret"
    assert result.langfuse_host == "https://test.langfuse.com"

    print("langfuse logger=", result)
    print("vars in langfuse logger=", vars(result))

    # Check if the logger is cached
    cached_logger = dynamic_logging_cache.get_cache(
        credentials={
            "langfuse_public_key": "test_public_key",
            "langfuse_secret": "test_secret",
            "langfuse_host": "https://test.langfuse.com",
        },
        service_name="langfuse",
    )
    assert cached_logger is result


@pytest.mark.parametrize("globalLangfuseLogger", [None, global_langfuse_logger])
def test_get_langfuse_logger_for_request_with_no_dynamic_params(
    dynamic_logging_cache, globalLangfuseLogger
):
    """
    If StandardCallbackDynamicParams are not provided, the globalLangfuseLogger should be returned
    """
    result = LangFuseHandler.get_langfuse_logger_for_request(
        standard_callback_dynamic_params=StandardCallbackDynamicParams(),
        in_memory_dynamic_logger_cache=dynamic_logging_cache,
        globalLangfuseLogger=globalLangfuseLogger,
    )

    assert result is not None
    assert isinstance(result, LangFuseLogger)

    print("langfuse logger=", result)

    if globalLangfuseLogger is not None:
        assert result.public_key == "global_public_key"
        assert result.secret_key == "global_secret"
        assert result.langfuse_host == "https://global.langfuse.com"


def test_dynamic_langfuse_credentials_are_passed():
    # Test when credentials are passed
    params_with_credentials = StandardCallbackDynamicParams(
        langfuse_public_key="test_key",
        langfuse_secret="test_secret",
        langfuse_host="https://test.langfuse.com",
    )
    assert (
        LangFuseHandler._dynamic_langfuse_credentials_are_passed(
            params_with_credentials
        )
        is True
    )

    # Test when no credentials are passed
    params_without_credentials = StandardCallbackDynamicParams()
    assert (
        LangFuseHandler._dynamic_langfuse_credentials_are_passed(
            params_without_credentials
        )
        is False
    )

    # Test when only some credentials are passed
    params_partial_credentials = StandardCallbackDynamicParams(
        langfuse_public_key="test_key"
    )
    assert (
        LangFuseHandler._dynamic_langfuse_credentials_are_passed(
            params_partial_credentials
        )
        is True
    )


def test_get_dynamic_langfuse_logging_config():
    # Test with dynamic params
    dynamic_params = StandardCallbackDynamicParams(
        langfuse_public_key="dynamic_key",
        langfuse_secret="dynamic_secret",
        langfuse_host="https://dynamic.langfuse.com",
    )
    config = LangFuseHandler.get_dynamic_langfuse_logging_config(dynamic_params)
    assert config["langfuse_public_key"] == "dynamic_key"
    assert config["langfuse_secret"] == "dynamic_secret"
    assert config["langfuse_host"] == "https://dynamic.langfuse.com"

    # Test with no dynamic params
    empty_params = StandardCallbackDynamicParams()
    config = LangFuseHandler.get_dynamic_langfuse_logging_config(empty_params)
    assert config["langfuse_public_key"] is None
    assert config["langfuse_secret"] is None
    assert config["langfuse_host"] is None


def test_return_global_langfuse_logger():
    mock_cache = Mock()
    global_logger = LangFuseLogger(
        langfuse_public_key="global_key", langfuse_secret="global_secret"
    )

    # Test with existing global logger
    result = LangFuseHandler._return_global_langfuse_logger(global_logger, mock_cache)
    assert result == global_logger

    # Test without global logger, but with cached logger, should return cached logger
    mock_cache.get_cache.return_value = global_logger
    result = LangFuseHandler._return_global_langfuse_logger(None, mock_cache)
    assert result == global_logger

    # Test without global logger and without cached logger, should create new logger
    mock_cache.get_cache.return_value = None
    with patch.object(
        LangFuseHandler,
        "_create_langfuse_logger_from_credentials",
        return_value=global_logger,
    ):
        result = LangFuseHandler._return_global_langfuse_logger(None, mock_cache)
        assert result == global_logger


def test_get_langfuse_logger_for_request_with_cached_logger():
    """
    Test that get_langfuse_logger_for_request returns the cached logger if it exists when dynamic params are passed
    """
    mock_cache = Mock()
    cached_logger = LangFuseLogger(
        langfuse_public_key="cached_key", langfuse_secret="cached_secret"
    )
    mock_cache.get_cache.return_value = cached_logger

    dynamic_params = StandardCallbackDynamicParams(
        langfuse_public_key="test_key",
        langfuse_secret="test_secret",
        langfuse_host="https://test.langfuse.com",
    )

    result = LangFuseHandler.get_langfuse_logger_for_request(
        standard_callback_dynamic_params=dynamic_params,
        in_memory_dynamic_logger_cache=mock_cache,
        globalLangfuseLogger=None,
    )

    assert result == cached_logger
    mock_cache.get_cache.assert_called_once()


@pytest.mark.parametrize(
    "metadata, expected_metadata",
    [
        ({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2, "c": 3}),
        (
            {"a": {"nested_a": 1}, "b": {"nested_b": 2}},
            {"a": {"nested_a": 1}, "b": {"nested_b": 2}},
        ),
        ({"a": [1, 2, 3], "b": {4, 5, 6}}, {"a": [1, 2, 3], "b": {4, 5, 6}}),
        (
            {"a": (1, 2), "b": frozenset([3, 4]), "c": {"d": [5, 6]}},
            {"a": (1, 2), "b": frozenset([3, 4]), "c": {"d": [5, 6]}},
        ),
        ({"lock": threading.Lock()}, {}),
        ({"func": lambda x: x + 1}, {}),
        (
            {
                "int": 42,
                "str": "hello",
                "list": [1, 2, 3],
                "set": {4, 5},
                "dict": {"nested": "value"},
                "non_copyable": threading.Lock(),
                "function": print,
            },
            {
                "int": 42,
                "str": "hello",
                "list": [1, 2, 3],
                "set": {4, 5},
                "dict": {"nested": "value"},
            },
        ),
        (
            {"list": ["list", "not", "a", "dict"]},
            {"list": ["list", "not", "a", "dict"]},
        ),
        ({}, {}),
        (None, None),
    ],
)
def test_langfuse_logger_prepare_metadata(metadata, expected_metadata):
    result = global_langfuse_logger._prepare_metadata(metadata)
    assert result == expected_metadata
