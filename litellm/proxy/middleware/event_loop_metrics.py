"""
Samples the asyncio event-loop's pending-callback queue depth and exposes it as
the Prometheus gauge ``litellm_event_loop_queue_depth``.

A rising queue depth means the event loop cannot drain callbacks fast enough —
long-running synchronous work, slow upstream calls, or an overloaded worker.
When the queue is saturated even ``/health/liveliness`` (a trivial 200 response)
cannot be scheduled, causing the liveness probe to time out and Kubernetes to
restart the pod.

Usage
-----
Call ``start_event_loop_metrics_task()`` once at proxy startup (inside an async
context so ``asyncio.get_event_loop()`` returns the running loop).  The returned
``asyncio.Task`` is fire-and-forget; store it only if you need to cancel it on
shutdown.
"""

import asyncio
import os
from typing import Any, Optional

_SAMPLE_INTERVAL_SECONDS: float = float(
    os.environ.get("LITELLM_EVENT_LOOP_SAMPLE_INTERVAL", "1")
)

_gauge: Optional[Any] = None
_gauge_init_attempted: bool = False
_current_depth: int = 0


def _get_gauge() -> Optional[Any]:
    global _gauge, _gauge_init_attempted
    if _gauge_init_attempted:
        return _gauge
    _gauge_init_attempted = True
    try:
        from prometheus_client import Gauge

        if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
            _gauge = Gauge(
                "litellm_event_loop_queue_depth",
                "Number of callbacks pending in the asyncio event-loop ready queue",
                multiprocess_mode="livesum",
            )
        else:
            _gauge = Gauge(
                "litellm_event_loop_queue_depth",
                "Number of callbacks pending in the asyncio event-loop ready queue",
            )
    except Exception:
        _gauge = None
    return _gauge


async def _sample_loop() -> None:
    """Background coroutine: samples len(loop._ready) every SAMPLE_INTERVAL_SECONDS."""
    global _current_depth
    loop = asyncio.get_event_loop()
    gauge = _get_gauge()
    while True:
        try:
            # _ready is an internal CPython deque of scheduled callbacks.
            # It is not part of the public API but is stable across CPython 3.8–3.12.
            depth = len(loop._ready)  # type: ignore[attr-defined]
        except Exception:
            depth = 0
        _current_depth = depth
        if gauge is not None:
            gauge.set(depth)
        await asyncio.sleep(_SAMPLE_INTERVAL_SECONDS)


def get_event_loop_queue_depth() -> int:
    """Return the most recently sampled event-loop queue depth."""
    return _current_depth


def start_event_loop_metrics_task() -> "asyncio.Task[None]":
    """
    Start the background sampling task.  Must be called from within a running
    event loop (e.g. inside a FastAPI startup handler).
    """
    return asyncio.create_task(_sample_loop())
