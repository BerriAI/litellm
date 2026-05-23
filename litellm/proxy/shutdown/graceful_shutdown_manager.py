"""
Graceful shutdown manager for LiteLLM proxy.

Tracks shutdown state and provides drain logic for in-flight requests
during pod termination (SIGTERM handling in Kubernetes deployments).

Configuration via environment variables:
  GRACEFUL_SHUTDOWN_TIMEOUT  - seconds to wait for drain (default 30)
"""

import asyncio
import os
import time
from typing import Optional


class GracefulShutdownManager:
    """
    Process-scoped shutdown state and in-flight request drain logic.

    The flag is set once (start_shutdown is idempotent) and never cleared
    during normal operation. Use reset() only in tests.
    """

    _is_shutting_down: bool = False
    _shutdown_started_at: Optional[float] = None

    @classmethod
    def is_shutting_down(cls) -> bool:
        """Return True if the proxy is in the process of shutting down."""
        return cls._is_shutting_down

    @classmethod
    def start_shutdown(cls) -> None:
        """Mark the process as shutting down and emit a structured log."""
        if cls._is_shutting_down:
            return
        cls._is_shutting_down = True
        cls._shutdown_started_at = time.monotonic()

        try:
            from litellm._logging import verbose_proxy_logger
            from litellm.proxy.middleware.in_flight_requests_middleware import (
                get_in_flight_requests,
            )

            in_flight = get_in_flight_requests()
            verbose_proxy_logger.info(
                '{"event": "graceful_shutdown_started", "in_flight_requests": %d}',
                in_flight,
            )
        except Exception:
            pass

    @classmethod
    async def wait_for_drain(
        cls, timeout: Optional[float] = None, deduct: int = 0
    ) -> int:
        """
        Wait for all in-flight HTTP requests to complete.

        Polls the InFlightRequestsMiddleware counter every 100 ms until it
        reaches zero or ``timeout`` seconds have elapsed.

        Args:
            timeout: Maximum seconds to wait. Defaults to
                     ``GRACEFUL_SHUTDOWN_TIMEOUT`` env var or 30 s.
            deduct: Number to subtract from the raw in-flight count when
                    evaluating whether the queue has drained.  Pass ``1``
                    when this coroutine is itself running inside an HTTP
                    handler (e.g. ``/health/drain``) so the handler does
                    not count itself as an outstanding request and the
                    loop can reach zero without exhausting the timeout.

        Returns:
            Number of requests that drained during the wait.
        """
        from litellm._logging import verbose_proxy_logger
        from litellm.proxy.middleware.in_flight_requests_middleware import (
            get_in_flight_requests,
        )

        if timeout is None:
            timeout = float(os.environ.get("GRACEFUL_SHUTDOWN_TIMEOUT", "30"))

        start = time.monotonic()
        initial_count = get_in_flight_requests()
        last_log_elapsed = 0.0

        while True:
            count = get_in_flight_requests() - deduct
            if count <= 0:
                break

            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                verbose_proxy_logger.warning(
                    '{"event": "drain_timeout", "in_flight_requests": %d, "elapsed_s": %.1f}',
                    count,
                    elapsed,
                )
                break

            if elapsed - last_log_elapsed >= 5.0:
                verbose_proxy_logger.info(
                    '{"event": "drain_waiting", "in_flight_requests": %d, "elapsed_s": %.1f}',
                    count,
                    elapsed,
                )
                last_log_elapsed = elapsed

            await asyncio.sleep(0.1)

        final_count = get_in_flight_requests()
        drained = max(0, initial_count - final_count)
        total_elapsed = (
            time.monotonic() - cls._shutdown_started_at
            if cls._shutdown_started_at is not None
            else 0.0
        )
        verbose_proxy_logger.info(
            '{"event": "graceful_shutdown_complete", "drained_requests": %d, "elapsed_s": %.1f}',
            drained,
            total_elapsed,
        )
        return drained

    @classmethod
    def reset(cls) -> None:
        """Reset state. For testing only."""
        cls._is_shutting_down = False
        cls._shutdown_started_at = None
