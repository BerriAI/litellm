"""
Application-level graceful shutdown coordination for the LiteLLM proxy.

Kubernetes terminates a pod by sending ``SIGTERM`` and, after
``terminationGracePeriodSeconds``, ``SIGKILL``. By default LiteLLM delegates
the signal to uvicorn and tears down immediately, dropping any in-flight
requests (streaming, batch inference, long-lived calls).

A fixed ``preStop`` sleep can not solve this: it has to be sized for the
*worst-case* request, so it either wastes time on every routine shutdown or is
too short for a long-running request. This manager instead drains based on the
*actual* in-flight request counter (already tracked by
``InFlightRequestsMiddleware``), so a pod terminates as soon as its real
in-flight work is done — and never waits longer than ``GRACEFUL_SHUTDOWN_TIMEOUT``.

The state is process-scoped (class-level), matching the per-uvicorn-worker
granularity of ``InFlightRequestsMiddleware``.
"""

import asyncio
import os
import time
from typing import Callable, Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy.middleware.in_flight_requests_middleware import (
    get_in_flight_requests,
)

# Keep below terminationGracePeriodSeconds so the process exits before SIGKILL.
DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT = 30.0
_DRAIN_POLL_INTERVAL = 0.1
_DRAIN_LOG_INTERVAL = 5.0


class GracefulShutdownManager:
    """
    Process-scoped singleton that tracks whether the worker is draining and
    blocks until in-flight requests reach zero (or a timeout elapses).
    """

    _is_shutting_down: bool = False
    _shutdown_started_at: Optional[float] = None
    _drain_performed: bool = False

    @classmethod
    def is_shutting_down(cls) -> bool:
        """Whether this worker has begun graceful shutdown."""
        return cls._is_shutting_down

    @classmethod
    def get_timeout(cls) -> float:
        """
        Read GRACEFUL_SHUTDOWN_TIMEOUT (seconds) from the environment on each
        call so deployments can tune it without code changes. Falls back to the
        default on an unset or malformed value.
        """
        raw = os.getenv("GRACEFUL_SHUTDOWN_TIMEOUT")
        if raw is None:
            return DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT
        try:
            return float(raw)
        except (TypeError, ValueError):
            verbose_proxy_logger.warning(
                "GRACEFUL_SHUTDOWN_TIMEOUT=%r is not a number; using default %ss",
                raw,
                DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT,
            )
            return DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT

    @classmethod
    def start_shutdown(cls) -> None:
        """
        Mark the worker as draining. Idempotent — repeated calls (e.g. SIGTERM
        followed by a preStop hit on /health/drain) do not reset the clock.
        """
        if cls._is_shutting_down:
            return
        cls._is_shutting_down = True
        cls._shutdown_started_at = time.monotonic()
        verbose_proxy_logger.info(
            "graceful_shutdown_started in_flight_requests=%s",
            get_in_flight_requests(),
        )

    @classmethod
    async def wait_for_drain(
        cls,
        timeout: Optional[float] = None,
        exclude_self: bool = False,
        count_fn: Optional[Callable[[], int]] = None,
        poll_interval: float = _DRAIN_POLL_INTERVAL,
        log_interval: float = _DRAIN_LOG_INTERVAL,
    ) -> int:
        """
        Poll the in-flight request counter until it reaches the drain target or
        ``timeout`` seconds elapse.

        Args:
            timeout: Max seconds to wait. Defaults to ``get_timeout()``.
            exclude_self: When the caller is itself an in-flight HTTP request
                (the /health/drain endpoint), set this so the caller's own
                request is not counted as outstanding work.
            count_fn: Source of the current in-flight count. Defaults to the
                live ``InFlightRequestsMiddleware`` counter; injectable for tests.
            poll_interval: Seconds between counter polls.
            log_interval: Minimum seconds between ``drain_waiting`` log lines.

        Returns:
            Number of requests that drained while waiting (>= 0).
        """
        # A preStop /health/drain hook and the lifespan SIGTERM handler both
        # drain; once one has run, the other must not wait again, otherwise the
        # effective window is 2x the timeout and terminationGracePeriodSeconds
        # has to be doubled to avoid a mid-drain SIGKILL.
        if cls._drain_performed:
            return 0
        cls._drain_performed = True

        if timeout is None:
            timeout = cls.get_timeout()
        if count_fn is None:
            count_fn = get_in_flight_requests

        # The /health/drain HTTP request flows through InFlightRequestsMiddleware
        # and so counts itself; treat <=1 as "drained" in that case.
        target = 1 if exclude_self else 0

        start = time.monotonic()
        initial = count_fn()
        last_log = start

        if timeout <= 0:
            return max(0, initial - target)

        while True:
            current = count_fn()
            if current <= target:
                drained = max(0, initial - current)
                verbose_proxy_logger.info(
                    "graceful_shutdown_complete drained_requests=%s elapsed_s=%.2f",
                    drained,
                    time.monotonic() - start,
                )
                return drained

            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                verbose_proxy_logger.warning(
                    "graceful_shutdown_timeout in_flight_requests=%s elapsed_s=%.2f "
                    "timeout_s=%s — proceeding with teardown",
                    current,
                    elapsed,
                    timeout,
                )
                return max(0, initial - current)

            now = time.monotonic()
            if now - last_log >= log_interval:
                verbose_proxy_logger.info(
                    "drain_waiting in_flight_requests=%s elapsed_s=%.2f",
                    current,
                    elapsed,
                )
                last_log = now

            await asyncio.sleep(poll_interval)

    @classmethod
    def reset(cls) -> None:
        """Reset state. Intended for use in tests."""
        cls._is_shutting_down = False
        cls._shutdown_started_at = None
        cls._drain_performed = False
