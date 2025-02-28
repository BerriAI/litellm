import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.dd_tracing import _should_use_dd_tracer, dd_tracer


def test_dd_tracer_when_package_exists():
    with patch("litellm.litellm_core_utils.dd_tracing.has_ddtrace", True):
        # Test the trace context manager
        with dd_tracer.trace("test_operation") as span:
            assert span is not None

        # Test the wrapper decorator
        @dd_tracer.wrap(name="test_function")
        def sample_function():
            return "test"

        result = sample_function()
        assert result == "test"


def test_dd_tracer_when_package_not_exists():
    with patch("litellm.litellm_core_utils.dd_tracing.has_ddtrace", False):
        # Test the trace context manager with null tracer
        with dd_tracer.trace("test_operation") as span:
            assert span is not None
            # Verify null span methods don't raise exceptions
            span.finish()

        # Test the wrapper decorator with null tracer
        @dd_tracer.wrap(name="test_function")
        def sample_function():
            return "test"

        result = sample_function()
        assert result == "test"


def test_null_tracer_context_manager():
    with patch("litellm.litellm_core_utils.dd_tracing.has_ddtrace", False):
        # Test that the context manager works without raising exceptions
        with dd_tracer.trace("test_operation") as span:
            # Test that we can call methods on the null span
            span.finish()
            assert True  # If we get here without exceptions, the test passes


def test_should_use_dd_tracer():
    with patch(
        "litellm.litellm_core_utils.dd_tracing.get_secret_bool"
    ) as mock_get_secret:
        # Test when USE_DDTRACE is True

        mock_get_secret.return_value = True
        assert _should_use_dd_tracer() is True
        mock_get_secret.assert_called_once_with("USE_DDTRACE", False)

        # Reset the mock for the next test
        mock_get_secret.reset_mock()

        # Test when USE_DDTRACE is False
        mock_get_secret.return_value = False
        assert _should_use_dd_tracer() is False
        mock_get_secret.assert_called_once_with("USE_DDTRACE", False)
