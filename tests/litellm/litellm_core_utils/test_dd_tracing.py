import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.dd_tracing import dd_tracer


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
