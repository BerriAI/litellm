"""
Tests for LoggingCallbackManager._add_custom_callback_generic_api_str — the GenericAPILogger
cache-resolution path.

Covers:
  - Cache hit reuses the same logger instance across repeated resolutions
  - Invalid empty log_format still raises ValueError
  - Genuine config change recreates the logger and cancels the old flush task
"""

import asyncio

import pytest

import litellm
from litellm.litellm_core_utils.logging_callback_manager import (
    LoggingCallbackManager,
    GenericAPILogger,
    _generic_api_logger_cache,
)


@pytest.fixture(autouse=True)
def _reset_generic_api_logger_state():
    _generic_api_logger_cache.clear()
    litellm.callback_settings = {}
    yield
    _generic_api_logger_cache.clear()
    litellm.callback_settings = {}


class TestGenericAPILoggerCaching:
    @pytest.mark.asyncio
    async def test_generic_api_logger_reused_on_repeated_resolution(self):
        litellm.callback_settings = {
            "cb": {
                "callback_type": "generic_api",
                "endpoint": "http://127.0.0.1:9/x",
                "headers": {"Authorization": "Bearer t"},
            }
        }

        resolved = [LoggingCallbackManager._add_custom_callback_generic_api_str("cb") for _ in range(5)]

        try:
            assert all(isinstance(logger, GenericAPILogger) for logger in resolved)
            assert all(logger is resolved[0] for logger in resolved)
        finally:
            resolved[0].shutdown()

    @pytest.mark.asyncio
    async def test_generic_api_logger_empty_log_format_still_raises(self):
        litellm.callback_settings = {
            "cb": {
                "callback_type": "generic_api",
                "endpoint": "http://127.0.0.1:9/x",
                "headers": {"Authorization": "Bearer t"},
                "log_format": "",
            }
        }

        with pytest.raises(ValueError):
            LoggingCallbackManager._add_custom_callback_generic_api_str("cb")

    @pytest.mark.asyncio
    async def test_generic_api_logger_recreated_and_old_task_cancelled_on_config_change(
        self,
    ):
        litellm.callback_settings = {
            "cb": {
                "callback_type": "generic_api",
                "endpoint": "http://127.0.0.1:9/a",
                "headers": {"Authorization": "Bearer t"},
            }
        }
        logger_a = LoggingCallbackManager._add_custom_callback_generic_api_str("cb")
        flush_task_a = logger_a._flush_task

        litellm.callback_settings["cb"]["endpoint"] = "http://127.0.0.1:9/b"
        logger_b = LoggingCallbackManager._add_custom_callback_generic_api_str("cb")

        try:
            assert logger_a is not logger_b

            await asyncio.sleep(0)
            assert flush_task_a.cancelled() is True
        finally:
            logger_b.shutdown()

    @pytest.mark.asyncio
    async def test_bad_replacement_config_does_not_evict_existing_logger(self):
        litellm.callback_settings = {
            "cb": {
                "callback_type": "generic_api",
                "endpoint": "http://127.0.0.1:9/a",
                "headers": {"Authorization": "Bearer t"},
            }
        }
        logger_a = LoggingCallbackManager._add_custom_callback_generic_api_str("cb")
        flush_task_a = logger_a._flush_task

        litellm.callback_settings["cb"]["log_format"] = "bad_format"

        try:
            with pytest.raises(ValueError):
                LoggingCallbackManager._add_custom_callback_generic_api_str("cb")

            assert flush_task_a.cancelled() is False
            assert _generic_api_logger_cache["cb"] is logger_a
        finally:
            logger_a.shutdown()
