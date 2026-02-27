"""
Tests for the CustomLogger.setup_proxy(app) hook.

Verifies that:
1. CustomLogger has a setup_proxy method (no-op default)
2. _invoke_callback_setup_proxy calls setup_proxy on all CustomLogger callbacks
3. Non-CustomLogger callbacks are skipped
4. Exceptions in setup_proxy are caught and logged (don't crash startup)
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.proxy_server import _invoke_callback_setup_proxy


class TestCustomLoggerSetupProxy:
    """Tests for the setup_proxy method on CustomLogger."""

    def test_custom_logger_has_setup_proxy_method(self):
        """CustomLogger base class should have a setup_proxy method."""
        logger = CustomLogger()
        assert hasattr(logger, "setup_proxy")
        assert callable(logger.setup_proxy)

    def test_setup_proxy_default_is_noop(self):
        """Default setup_proxy should do nothing and not raise."""
        logger = CustomLogger()
        mock_app = MagicMock()
        logger.setup_proxy(mock_app)  # Should not raise


class TestInvokeCallbackSetupProxy:
    """Tests for _invoke_callback_setup_proxy function."""

    def test_calls_setup_proxy_on_custom_logger_callbacks(self):
        """setup_proxy should be called on each CustomLogger in litellm.callbacks."""

        class MyCallback(CustomLogger):
            def __init__(self):
                super().__init__()
                self.setup_called_with = None

            def setup_proxy(self, app):
                self.setup_called_with = app

        callback = MyCallback()
        mock_app = MagicMock()

        original_callbacks = litellm.callbacks
        litellm.callbacks = [callback]
        try:
            _invoke_callback_setup_proxy(mock_app)
            assert callback.setup_called_with is mock_app
        finally:
            litellm.callbacks = original_callbacks

    def test_skips_non_custom_logger_callbacks(self):
        """String callbacks and non-CustomLogger objects should be skipped."""
        mock_app = MagicMock()

        original_callbacks = litellm.callbacks
        litellm.callbacks = ["langfuse", "sentry", 42]
        try:
            # Should not raise
            _invoke_callback_setup_proxy(mock_app)
        finally:
            litellm.callbacks = original_callbacks

    def test_exception_in_setup_proxy_is_caught(self):
        """If setup_proxy raises, it should be caught and logged, not crash."""

        class BadCallback(CustomLogger):
            def setup_proxy(self, app):
                raise RuntimeError("setup failed")

        bad_callback = BadCallback()
        good_callback = CustomLogger()
        mock_app = MagicMock()

        original_callbacks = litellm.callbacks
        litellm.callbacks = [bad_callback, good_callback]
        try:
            # Should not raise even though bad_callback.setup_proxy raises
            _invoke_callback_setup_proxy(mock_app)
        finally:
            litellm.callbacks = original_callbacks

    def test_multiple_callbacks_all_called(self):
        """All CustomLogger callbacks should have setup_proxy called."""
        call_order = []

        class Callback1(CustomLogger):
            def setup_proxy(self, app):
                call_order.append("cb1")

        class Callback2(CustomLogger):
            def setup_proxy(self, app):
                call_order.append("cb2")

        mock_app = MagicMock()
        original_callbacks = litellm.callbacks
        litellm.callbacks = [Callback1(), "langfuse", Callback2()]
        try:
            _invoke_callback_setup_proxy(mock_app)
            assert call_order == ["cb1", "cb2"]
        finally:
            litellm.callbacks = original_callbacks
