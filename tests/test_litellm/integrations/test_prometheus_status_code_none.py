"""
Regression tests for #24224 — status_code=None in Prometheus metrics
"""
import pytest
from prometheus_client import REGISTRY

from litellm.integrations.prometheus import PrometheusLogger


@pytest.fixture(scope="function")
def prometheus_logger():
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)
    return PrometheusLogger()


def test_extract_status_code_returns_int_for_exception_with_status_code(prometheus_logger):
    """Exception with status_code attribute should return that code."""
    exc = Exception("fail")
    exc.status_code = 429  # type: ignore
    assert prometheus_logger._extract_status_code(exception=exc) == 429


def test_extract_status_code_returns_int_for_proxy_exception_with_code(prometheus_logger):
    """ProxyException-style exception with code attribute should return that code."""
    exc = Exception("fail")
    exc.code = 403  # type: ignore
    assert prometheus_logger._extract_status_code(exception=exc) == 403


def test_extract_status_code_defaults_to_500_for_bare_exception(prometheus_logger):
    """Exception with no status_code or code should default to 500, not None."""
    exc = Exception("something broke")
    result = prometheus_logger._extract_status_code(exception=exc)
    assert result == 500, f"Expected 500 for bare exception, got {result}"


def test_extract_status_code_returns_none_when_no_exception(prometheus_logger):
    """When no exception is provided at all, should return None."""
    assert prometheus_logger._extract_status_code() is None
