"""Regression tests for #39816: the 429 body labels its reset time UTC."""

import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.batch_rate_limiter import BatchFileUsage, _PROXY_BatchRateLimiter
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3,
)

WINDOW_SIZE = 60
FIXED_EPOCH = datetime(2026, 9, 4, 21, 53, 21, tzinfo=timezone.utc).timestamp()
EXPECTED_RESET = "2026-09-04 21:54:21 UTC"

needs_tzset = pytest.mark.skipif(
    not hasattr(time, "tzset"), reason="switching the process timezone needs time.tzset(), absent on Windows"
)


@contextmanager
def a_proxy_running_in(tz: str):
    previous = os.environ.get("TZ")
    os.environ["TZ"] = tz
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = previous
        time.tzset()


def _reset_time_reported_by(detail: str) -> str:
    return detail.split("Limit resets at: ")[-1]


def _over_limit_response():
    return {
        "overall_code": "OVER_LIMIT",
        "statuses": [
            {
                "code": "OVER_LIMIT",
                "descriptor_key": "api_key",
                "limit_remaining": 0,
                "rate_limit_type": "requests",
                "current_limit": 2,
            }
        ],
    }


def _descriptors():
    return [{"key": "api_key", "value": "sk-test", "rate_limit": None}]


def _parallel_limiter():
    limiter = _PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=MagicMock(),
        time_provider=lambda: datetime.fromtimestamp(FIXED_EPOCH),
    )
    limiter.window_size = WINDOW_SIZE
    return limiter


class TestParallelLimiterReportsUTC:
    @needs_tzset
    @pytest.mark.parametrize("tz", ["Asia/Kolkata", "Europe/Paris", "America/Los_Angeles"])
    def test_a_non_utc_proxy_still_reports_utc(self, tz):
        with a_proxy_running_in(tz):
            limiter = _parallel_limiter()
            with pytest.raises(ProxyRateLimitError) as exc_info:
                limiter._handle_rate_limit_error(
                    response=_over_limit_response(),
                    descriptors=_descriptors(),
                    requested_model="gpt-4o-mini",
                )

        assert _reset_time_reported_by(str(exc_info.value)) == EXPECTED_RESET

    def test_the_reset_time_is_one_window_ahead(self):
        limiter = _parallel_limiter()
        with pytest.raises(ProxyRateLimitError) as exc_info:
            limiter._handle_rate_limit_error(
                response=_over_limit_response(),
                descriptors=_descriptors(),
                requested_model="gpt-4o-mini",
            )

        expected = datetime.fromtimestamp(FIXED_EPOCH + WINDOW_SIZE, tz=timezone.utc)
        assert _reset_time_reported_by(str(exc_info.value)) == expected.strftime("%Y-%m-%d %H:%M:%S UTC")


class TestBatchLimiterReportsUTC:
    @needs_tzset
    @pytest.mark.parametrize("tz", ["Asia/Kolkata", "America/Los_Angeles"])
    def test_a_non_utc_proxy_still_reports_utc(self, tz):
        parallel_request_limiter = MagicMock()
        parallel_request_limiter.window_size = WINDOW_SIZE
        limiter = _PROXY_BatchRateLimiter(
            internal_usage_cache=MagicMock(),
            parallel_request_limiter=parallel_request_limiter,
        )

        with a_proxy_running_in(tz):
            before = datetime.now(timezone.utc)
            with pytest.raises(ProxyRateLimitError) as exc_info:
                limiter._raise_rate_limit_error(
                    status=_over_limit_response()["statuses"][0],
                    descriptors=_descriptors(),
                    batch_usage=BatchFileUsage(total_tokens=10, request_count=1),
                    limit_type="requests",
                    requested_model="gpt-4o-mini",
                )
            after = datetime.now(timezone.utc)

        reported = _reset_time_reported_by(str(exc_info.value))
        window = {
            datetime.fromtimestamp(t.timestamp() + WINDOW_SIZE, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            for t in (before, after)
        }
        assert reported in window
