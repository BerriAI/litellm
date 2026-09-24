"""
Tests for the GenericAPILogger cache-resolution path in
LoggingCallbackManager._add_custom_callback_generic_api_str.

Covers:
  - Cache hit reuses the same logger instance across repeated resolutions
  - Invalid empty log_format still raises ValueError
  - Genuine config change recreates the logger and cancels the old flush task
  - Header rotation is compared on the effective headers, not the raw config ones
"""

import asyncio
import contextlib

import pytest

import litellm
from litellm.litellm_core_utils.logging_callback_manager import (
    GenericAPILogger,
    LoggingCallbackManager,
    _generic_api_logger_cache,
)


@pytest.fixture(autouse=True)
def callback_settings(monkeypatch):
    settings = {}
    monkeypatch.setattr(litellm, "callback_settings", settings)
    _generic_api_logger_cache.clear()
    yield settings
    _generic_api_logger_cache.clear()


class TestGenericAPILoggerCaching:
    @pytest.mark.asyncio
    async def test_generic_api_logger_reused_on_repeated_resolution(self, callback_settings):
        callback_settings["cb"] = {
            "callback_type": "generic_api",
            "endpoint": "http://127.0.0.1:9/x",
            "headers": {"Authorization": "Bearer t"},
        }

        resolved = [LoggingCallbackManager._add_custom_callback_generic_api_str("cb") for _ in range(5)]

        try:
            assert all(isinstance(logger, GenericAPILogger) for logger in resolved)
            assert all(logger is resolved[0] for logger in resolved)
        finally:
            resolved[0].shutdown()

    @pytest.mark.asyncio
    async def test_generic_api_logger_empty_log_format_still_raises(self, callback_settings):
        callback_settings["cb"] = {
            "callback_type": "generic_api",
            "endpoint": "http://127.0.0.1:9/x",
            "headers": {"Authorization": "Bearer t"},
            "log_format": "",
        }

        with pytest.raises(ValueError, match="Invalid log_format"):
            LoggingCallbackManager._add_custom_callback_generic_api_str("cb")

    @pytest.mark.asyncio
    async def test_generic_api_logger_recreated_and_old_task_cancelled_on_config_change(self, callback_settings):
        callback_settings["cb"] = {
            "callback_type": "generic_api",
            "endpoint": "http://127.0.0.1:9/a",
            "headers": {"Authorization": "Bearer t"},
        }
        logger_a = LoggingCallbackManager._add_custom_callback_generic_api_str("cb")

        callback_settings["cb"]["endpoint"] = "http://127.0.0.1:9/b"
        logger_b = LoggingCallbackManager._add_custom_callback_generic_api_str("cb")

        try:
            assert logger_a is not logger_b

            with contextlib.suppress(asyncio.CancelledError):
                await logger_a._flush_task
            assert logger_a._flush_task.cancelled()
        finally:
            logger_b.shutdown()

    @pytest.mark.asyncio
    async def test_bad_replacement_config_does_not_evict_existing_logger(self, callback_settings):
        callback_settings["cb"] = {
            "callback_type": "generic_api",
            "endpoint": "http://127.0.0.1:9/a",
            "headers": {"Authorization": "Bearer t"},
        }
        logger_a = LoggingCallbackManager._add_custom_callback_generic_api_str("cb")
        flush_task_a = logger_a._flush_task

        callback_settings["cb"]["log_format"] = "bad_format"

        try:
            with pytest.raises(ValueError, match="Invalid log_format"):
                LoggingCallbackManager._add_custom_callback_generic_api_str("cb")

            assert flush_task_a.cancelled() is False
            assert _generic_api_logger_cache["cb"] is logger_a
        finally:
            logger_a.shutdown()

    @pytest.mark.asyncio
    async def test_env_header_rotation_recreates_logger(self, callback_settings, monkeypatch):
        monkeypatch.setenv("GENERIC_LOGGER_HEADERS", "Authorization=Bearer old")
        callback_settings["cb"] = {
            "callback_type": "generic_api",
            "endpoint": "http://127.0.0.1:9/a",
            "headers": {"X-Static": "s"},
        }
        logger_a = LoggingCallbackManager._add_custom_callback_generic_api_str("cb")

        monkeypatch.setenv("GENERIC_LOGGER_HEADERS", "Authorization=Bearer new")
        logger_b = LoggingCallbackManager._add_custom_callback_generic_api_str("cb")

        try:
            assert logger_a is not logger_b
            assert logger_b.headers["Authorization"] == "Bearer new"

            with contextlib.suppress(asyncio.CancelledError):
                await logger_a._flush_task
            assert logger_a._flush_task.cancelled()
        finally:
            logger_b.shutdown()

    @pytest.mark.asyncio
    async def test_env_header_shadowed_by_config_reuses_logger(self, callback_settings, monkeypatch):
        monkeypatch.setenv("GENERIC_LOGGER_HEADERS", "Authorization=Bearer old")
        callback_settings["cb"] = {
            "callback_type": "generic_api",
            "endpoint": "http://127.0.0.1:9/a",
            "headers": {"Authorization": "Bearer from-config"},
        }
        logger_a = LoggingCallbackManager._add_custom_callback_generic_api_str("cb")

        monkeypatch.setenv("GENERIC_LOGGER_HEADERS", "Authorization=Bearer new")
        logger_b = LoggingCallbackManager._add_custom_callback_generic_api_str("cb")

        try:
            assert logger_a is logger_b
        finally:
            logger_a.shutdown()

    @pytest.mark.asyncio
    async def test_final_flush_error_on_cancellation_is_swallowed(self, callback_settings):
        callback_settings["cb"] = {
            "callback_type": "generic_api",
            "endpoint": "http://127.0.0.1:9/a",
            "headers": {"Authorization": "Bearer t"},
        }
        logger = LoggingCallbackManager._add_custom_callback_generic_api_str("cb")

        flushed = asyncio.Event()

        async def _raise_on_flush():
            flushed.set()
            raise Exception("boom")

        logger.flush_queue = _raise_on_flush
        # Let the flush task reach its first await; cancelling a task that never started
        # skips _run_periodic_flush entirely and the final-flush path goes untested.
        await asyncio.sleep(0)

        logger.shutdown()

        with contextlib.suppress(asyncio.CancelledError):
            await logger._flush_task
        assert flushed.is_set()
        assert logger._flush_task.cancelled() is True
