import json
from dataclasses import dataclass
from typing import Final
from unittest.mock import MagicMock, patch

import pytest


from litellm.litellm_core_utils import dd_tracing
from litellm.litellm_core_utils.dd_tracing import (
    set_active_span_tag,
    should_use_dd_profiler,
    _should_use_dd_tracer,
)
from litellm.litellm_core_utils.dd_tracing import tracer as dd_tracer


def test_dd_tracer_when_package_exists():
    with patch("litellm.litellm_core_utils.dd_tracing.should_use_dd_tracer", True):
        """
        Operations works as expected when ddtrace is installed and USE_DDTRACE is True
        """
        # Create a mock tracer with the necessary methods
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.trace.return_value.__enter__.return_value = mock_span

        with patch("litellm.litellm_core_utils.dd_tracing.tracer", mock_tracer):
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
    """
    Operations works as expected when ddtrace is not installed and USE_DDTRACE is False
    """
    with patch("litellm.litellm_core_utils.dd_tracing.should_use_dd_tracer", False):
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


def test_should_use_dd_tracer():
    """
    Test that the should_use_dd_tracer function works as expected
    """
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


def test_should_use_dd_profiler():
    """
    Test that the should_use_dd_profiler function works as expected
    """
    with patch(
        "litellm.litellm_core_utils.dd_tracing.get_secret_bool"
    ) as mock_get_secret:
        # Test when USE_DDPROFILER is True

        mock_get_secret.return_value = True
        assert should_use_dd_profiler() is True
        mock_get_secret.assert_called_once_with("USE_DDPROFILER", False)

        # Reset the mock for the next test
        mock_get_secret.reset_mock()

        # Test when USE_DDPROFILER is False
        mock_get_secret.return_value = False
        assert should_use_dd_profiler() is False
        mock_get_secret.assert_called_once_with("USE_DDPROFILER", False)


@dataclass(frozen=True)
class _CurrentSpanTracer:
    span: object

    def current_span(self) -> object:
        return self.span


class _SetTagOnlySpan:
    def __init__(self) -> None:
        self.tags: dict[str, str] = {}

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value


def test_set_active_span_tag_uses_set_tag_on_a_span_without_set_tag_str(monkeypatch: pytest.MonkeyPatch) -> None:
    span: Final = _SetTagOnlySpan()
    monkeypatch.setattr(dd_tracing, "tracer", _CurrentSpanTracer(span))

    assert set_active_span_tag("litellm.call_id", "call-1") is True
    assert span.tags == {"litellm.call_id": "call-1"}


def test_set_active_span_tag_reports_false_for_a_span_that_takes_no_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dd_tracing, "tracer", _CurrentSpanTracer(object()))

    assert set_active_span_tag("litellm.call_id", "call-1") is False
