"""
Unit tests for ProxyLogging._init_litellm_callbacks.

Validates that string callbacks in litellm.callbacks are replaced in-place
with their initialized instances, preventing duplicate entries (string + instance)
that caused double-counting of metrics like litellm_proxy_total_requests_metric.
"""

from typing import List, Union
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger


class FakeCustomLogger(CustomLogger):
    """A minimal CustomLogger subclass for testing."""

    pass


class TestInitLitellmCallbacks:
    """Tests for ProxyLogging._init_litellm_callbacks."""

    def _make_proxy_logging(self):
        """Create a ProxyLogging instance with mocked dependencies."""
        from litellm.proxy.utils import ProxyLogging

        mock_cache = MagicMock()
        proxy_logging = ProxyLogging(user_api_key_cache=mock_cache)
        return proxy_logging

    @patch(
        "litellm.proxy.utils.ProxyLogging._add_proxy_hooks",
        new_callable=lambda: lambda self, *a, **kw: None,
    )
    def test_should_replace_string_callback_with_instance(self, _mock_hooks):
        """
        When litellm.callbacks contains a string callback (e.g. "lago"),
        _init_litellm_callbacks should replace the string with the initialized
        CustomLogger instance, not leave both the string and instance in the list.
        """
        fake_logger = FakeCustomLogger()

        # Start with a string callback in litellm.callbacks
        litellm.callbacks = ["lago"]  # type: ignore

        proxy_logging = self._make_proxy_logging()

        with patch(
            "litellm.litellm_core_utils.litellm_logging._init_custom_logger_compatible_class",
            return_value=fake_logger,
        ):
            proxy_logging._init_litellm_callbacks(llm_router=None)

        # The string "lago" should be replaced by the instance, not appended
        string_entries = [c for c in litellm.callbacks if isinstance(c, str)]
        instance_entries = [
            c for c in litellm.callbacks if isinstance(c, FakeCustomLogger)
        ]

        assert len(string_entries) == 0, (
            f"String callbacks should have been replaced, but found: {string_entries}"
        )
        assert len(instance_entries) == 1, (
            f"Expected exactly one FakeCustomLogger instance, found {len(instance_entries)}"
        )
        assert instance_entries[0] is fake_logger

        # Clean up
        litellm.callbacks = []  # type: ignore

    @patch(
        "litellm.proxy.utils.ProxyLogging._add_proxy_hooks",
        new_callable=lambda: lambda self, *a, **kw: None,
    )
    def test_should_not_duplicate_existing_instance_callbacks(self, _mock_hooks):
        """
        When litellm.callbacks already contains a CustomLogger instance (not a string),
        _init_litellm_callbacks should not create a duplicate.
        """
        existing_logger = FakeCustomLogger()

        litellm.callbacks = [existing_logger]  # type: ignore

        proxy_logging = self._make_proxy_logging()

        proxy_logging._init_litellm_callbacks(llm_router=None)

        # Count how many FakeCustomLogger instances are in litellm.callbacks
        instance_count = sum(
            1 for c in litellm.callbacks if isinstance(c, FakeCustomLogger)
        )
        assert instance_count == 1, (
            f"Expected exactly 1 FakeCustomLogger instance, found {instance_count}. "
            f"litellm.callbacks = {litellm.callbacks}"
        )

        # Clean up
        litellm.callbacks = []  # type: ignore

    @patch(
        "litellm.proxy.utils.ProxyLogging._add_proxy_hooks",
        new_callable=lambda: lambda self, *a, **kw: None,
    )
    def test_should_handle_unrecognized_string_callback(self, _mock_hooks):
        """
        When _init_custom_logger_compatible_class returns None for a string callback,
        the string should remain in litellm.callbacks (not crash).
        """
        litellm.callbacks = ["unknown_callback"]  # type: ignore

        proxy_logging = self._make_proxy_logging()

        with patch(
            "litellm.litellm_core_utils.litellm_logging._init_custom_logger_compatible_class",
            return_value=None,
        ):
            proxy_logging._init_litellm_callbacks(llm_router=None)

        # The unknown string callback should still be there (not replaced, not crashed)
        assert "unknown_callback" in litellm.callbacks

        # Clean up
        litellm.callbacks = []  # type: ignore

    @patch(
        "litellm.proxy.utils.ProxyLogging._add_proxy_hooks",
        new_callable=lambda: lambda self, *a, **kw: None,
    )
    def test_should_replace_multiple_string_callbacks(self, _mock_hooks):
        """
        When litellm.callbacks contains multiple string callbacks,
        each should be replaced with its corresponding initialized instance.
        """
        fake_logger_a = FakeCustomLogger()
        fake_logger_b = FakeCustomLogger()

        litellm.callbacks = ["callback_a", "callback_b"]  # type: ignore

        proxy_logging = self._make_proxy_logging()

        call_count = 0

        def mock_init_class(callback_name, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return fake_logger_a
            return fake_logger_b

        with patch(
            "litellm.litellm_core_utils.litellm_logging._init_custom_logger_compatible_class",
            side_effect=mock_init_class,
        ):
            proxy_logging._init_litellm_callbacks(llm_router=None)

        string_entries = [c for c in litellm.callbacks if isinstance(c, str)]
        instance_entries = [
            c for c in litellm.callbacks if isinstance(c, FakeCustomLogger)
        ]

        assert len(string_entries) == 0, (
            f"All string callbacks should have been replaced: {string_entries}"
        )
        assert len(instance_entries) == 2, (
            f"Expected 2 FakeCustomLogger instances, found {len(instance_entries)}"
        )
        assert instance_entries[0] is fake_logger_a
        assert instance_entries[1] is fake_logger_b

        # Clean up
        litellm.callbacks = []  # type: ignore
