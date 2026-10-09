from __future__ import annotations

from itertools import count, repeat

import pytest

from e2e_http import NetworkError, Success
from session_anomaly import (
    SessionMessagesResponse,
    TurnMetric,
    retried,
    settled_spend,
    summarize,
)


def _ok_turn(turn_index: int) -> TurnMetric:
    return TurnMetric(
        turn_index=turn_index,
        ok=True,
        latency_seconds=1.0,
        uncached_input_tokens=10,
        cache_read_tokens=100,
        cache_creation_tokens=5,
        failure=None,
    )


def _failed_turn(turn_index: int) -> TurnMetric:
    return TurnMetric(
        turn_index=turn_index,
        ok=False,
        latency_seconds=1.0,
        uncached_input_tokens=0,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        failure="NetworkError()",
    )


class TestSummarizePlannedTurns:
    def test_session_aborted_on_first_turn_counts_all_its_planned_turns_as_failed(
        self,
    ) -> None:
        completed_session = tuple(_ok_turn(index) for index in range(1, 7))
        aborted_session = (_failed_turn(1),)

        report = summarize((*completed_session, *aborted_session), planned_turns=12)

        assert report.attempted_turns == 7
        assert report.failed_turns == 6
        assert report.error_ratio == 0.5

    def test_all_planned_turns_completing_reports_zero_failures(self) -> None:
        report = summarize(
            tuple(_ok_turn(index) for index in range(1, 7)), planned_turns=6
        )

        assert report.failed_turns == 0
        assert report.error_ratio == 0.0


class TestRetried:
    def test_transient_failures_then_success_returns_the_success(self) -> None:
        outcome = Success[SessionMessagesResponse](status_code=200, data=SessionMessagesResponse())
        calls = iter(
            (NetworkError(message="overloaded"), NetworkError(message="overloaded"), outcome)
        )

        result = retried(lambda: next(calls), attempts=3, sleep=lambda _: None)

        assert result is outcome

    def test_exhausted_attempts_return_the_last_failure(self) -> None:
        last_attempt = NetworkError(message="still overloaded")
        never_reached = NetworkError(message="a fourth attempt would break the budget")
        calls = iter(
            (NetworkError(message="overloaded"), last_attempt, never_reached)
        )

        result = retried(lambda: next(calls), attempts=2, sleep=lambda _: None)

        assert result is last_attempt
        assert next(calls) is never_reached

    def test_first_try_success_never_sleeps(self) -> None:
        def sleep_means_retry(_: float) -> None:
            raise AssertionError("slept after a successful attempt")

        result = retried(
            lambda: Success[SessionMessagesResponse](status_code=200, data=SessionMessagesResponse()),
            attempts=3,
            sleep=sleep_means_retry,
        )

        assert isinstance(result, Success)


class TestSettledSpend:
    def test_partial_total_between_batch_flushes_is_not_accepted_as_final(self) -> None:
        reads = iter((0.1, 0.1, 0.1, 0.35, 0.35, 0.35, 0.35, 0.35))
        ticks = count(0.0, 2.5)

        spend = settled_spend(
            lambda: next(reads),
            poll_interval=5.0,
            settle_seconds=10.0,
            timeout_seconds=100.0,
            now=lambda: next(ticks),
            sleep=lambda _: None,
        )

        assert spend == 0.35

    def test_spend_that_never_stabilizes_raises(self) -> None:
        reads = (0.1 * step for step in count(1))
        ticks = count(0.0, 2.5)

        with pytest.raises(AssertionError, match="spend anomaly"):
            settled_spend(
                lambda: next(reads),
                poll_interval=5.0,
                settle_seconds=5.0,
                timeout_seconds=10.0,
                now=lambda: next(ticks),
                sleep=lambda _: None,
            )

    def test_spend_that_never_becomes_nonzero_raises(self) -> None:
        reads = repeat(0.0)
        ticks = count(0.0, 2.5)

        with pytest.raises(AssertionError, match="spend anomaly"):
            settled_spend(
                lambda: next(reads),
                poll_interval=5.0,
                settle_seconds=5.0,
                timeout_seconds=10.0,
                now=lambda: next(ticks),
                sleep=lambda _: None,
            )
