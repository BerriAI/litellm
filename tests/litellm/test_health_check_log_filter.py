"""Tests for the health check access log filter."""

import logging

from litellm._logging import (
    HealthCheckAccessFilter,
    apply_health_check_log_filter,
    remove_health_check_log_filter,
)


class TestHealthCheckAccessFilter:
    """Test HealthCheckAccessFilter suppresses health check access log entries."""

    def _make_record(self, msg: str, args=()) -> logging.LogRecord:
        return logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=args,
            exc_info=None,
        )

    def test_filters_readiness(self):
        f = HealthCheckAccessFilter()
        record = self._make_record(
            '10.0.0.1:1234 - "GET /health/readiness HTTP/1.1" 200 OK'
        )
        assert f.filter(record) is False

    def test_filters_liveliness(self):
        f = HealthCheckAccessFilter()
        record = self._make_record(
            '10.0.0.1:1234 - "GET /health/liveliness HTTP/1.1" 200 OK'
        )
        assert f.filter(record) is False

    def test_filters_liveness(self):
        f = HealthCheckAccessFilter()
        record = self._make_record(
            '10.0.0.1:1234 - "GET /health/liveness HTTP/1.1" 200 OK'
        )
        assert f.filter(record) is False

    def test_allows_regular_request(self):
        f = HealthCheckAccessFilter()
        record = self._make_record(
            '10.0.0.1:1234 - "POST /v1/chat/completions HTTP/1.1" 200 OK'
        )
        assert f.filter(record) is True

    def test_allows_health_services(self):
        f = HealthCheckAccessFilter()
        record = self._make_record(
            '10.0.0.1:1234 - "GET /health/services HTTP/1.1" 200 OK'
        )
        assert f.filter(record) is True

    def test_filters_bare_health_endpoint(self):
        f = HealthCheckAccessFilter()
        record = self._make_record(
            '10.0.0.1:1234 - "GET /health HTTP/1.1" 200 OK'
        )
        assert f.filter(record) is False

    def test_filters_bare_health_endpoint_with_query_param(self):
        f = HealthCheckAccessFilter()
        record = self._make_record(
            '10.0.0.1:1234 - "GET /health?full=true HTTP/1.1" 200 OK'
        )
        assert f.filter(record) is False

    def test_filters_using_uvicorn_tuple_args(self):
        f = HealthCheckAccessFilter()
        record = self._make_record(
            '%s - "%s %s HTTP/%s" %s',
            args=("10.0.0.1:1234", "GET", "/health/liveness?full=true", "1.1", 200),
        )
        assert f.filter(record) is False


class TestApplyHealthCheckLogFilter:
    """Test runtime attachment of the filter to uvicorn.access logger."""

    def test_attaches_filter_to_uvicorn_access(self):
        logger = logging.getLogger("uvicorn.access")
        apply_health_check_log_filter()
        count = len(
            [f for f in logger.filters if isinstance(f, HealthCheckAccessFilter)]
        )
        assert count >= 1
        remove_health_check_log_filter()

    def test_idempotent(self):
        logger = logging.getLogger("uvicorn.access")
        remove_health_check_log_filter()
        apply_health_check_log_filter()
        apply_health_check_log_filter()
        count = len(
            [f for f in logger.filters if isinstance(f, HealthCheckAccessFilter)]
        )
        assert count == 1
        remove_health_check_log_filter()

    def test_remove_filter(self):
        logger = logging.getLogger("uvicorn.access")
        apply_health_check_log_filter()
        remove_health_check_log_filter()
        count = len(
            [f for f in logger.filters if isinstance(f, HealthCheckAccessFilter)]
        )
        assert count == 0
