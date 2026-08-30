"""Harness coverage for the transport's transient-retry policy.

No proxy needed and no ``e2e`` marker: this pins the retry CONTRACT, which is
load-bearing for the whole suite. Only statuses the proxy itself cannot emit
may ever be retried (today exactly 529, Anthropic's overload signal): 429 must
stay unretried because the quota suites assert the proxy's own rate-limit and
budget 429s, and proxy-capable 5xx must stay unretried or an intermittently
failing proxy would slip through green. The fakes satisfy the
RetryableResponse protocol directly, so nothing here imports requests or
monkeypatches anything.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pytest

from e2e_http import RETRY_ATTEMPTS, TRANSIENT_STATUSES, request_with_retry


@dataclass
class FakeResponse:
    status_code: int
    close_calls: int = 0

    def close(self) -> None:
        self.close_calls += 1


@dataclass
class SleepRecorder:
    delays: list[float] = field(default_factory=list)

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def _issue_from(responses: Sequence[FakeResponse]) -> Callable[[], FakeResponse]:
    it = iter(responses)
    return lambda: next(it)


class TestTransientRetryPolicy:
    def test_transient_set_is_only_statuses_the_proxy_cannot_emit(self) -> None:
        assert TRANSIENT_STATUSES == frozenset({529})
        assert 429 not in TRANSIENT_STATUSES

    @pytest.mark.parametrize("status", [200, 201, 400, 401, 404, 422, 500, 502, 503, 504])
    def test_non_transient_status_returns_immediately(self, status: int) -> None:
        responses = (FakeResponse(status), FakeResponse(200))
        sleep = SleepRecorder()
        result = request_with_retry(_issue_from(responses), sleep=sleep)
        assert result is responses[0]
        assert sleep.delays == []
        assert responses[0].close_calls == 0

    def test_429_is_never_retried(self) -> None:
        responses = (FakeResponse(429), FakeResponse(200))
        sleep = SleepRecorder()
        result = request_with_retry(_issue_from(responses), sleep=sleep)
        assert result is responses[0]
        assert sleep.delays == []
        assert responses[0].close_calls == 0

    def test_overloaded_529_retries_with_backoff_then_returns_the_success(self) -> None:
        responses = (FakeResponse(529), FakeResponse(200))
        sleep = SleepRecorder()
        result = request_with_retry(_issue_from(responses), sleep=sleep)
        assert result is responses[1]
        assert sleep.delays == [0.5]
        assert responses[0].close_calls == 1
        assert responses[1].close_calls == 0

    def test_persistent_transient_is_bounded_and_returns_the_last_response(self) -> None:
        responses = tuple(FakeResponse(529) for _ in range(RETRY_ATTEMPTS + 1))
        sleep = SleepRecorder()
        result = request_with_retry(_issue_from(responses), sleep=sleep)
        assert result is responses[RETRY_ATTEMPTS - 1]
        assert sleep.delays == [0.5, 1.0]
        assert [r.close_calls for r in responses] == [1, 1, 0, 0]
