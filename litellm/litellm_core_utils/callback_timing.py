import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from threading import Lock
from time import monotonic
from typing import TYPE_CHECKING, Final, Literal

from litellm._internal_context import current_request_parent_otel_span
from litellm._logging import verbose_logger

if TYPE_CHECKING:
    from prometheus_client import Histogram

ExecutionPath = Literal["request_path", "background", "unclassified"]
_REGISTRATION_LOCK: Final = Lock()


@cache
def _callback_timing_enabled() -> bool:
    return os.getenv("LITELLM_CALLBACK_TIMING_METRICS", "").lower() == "true"


@cache
def _load_histogram() -> "Histogram | None":
    try:
        from prometheus_client import Histogram

        return Histogram(
            "litellm_callback_execution_seconds",
            "Wall-clock time spent executing a custom callback hook",
            ("callback_name", "hook", "execution_path"),
        )
    except Exception:  # noqa: BLE001  # optional metrics must not change callback behavior
        verbose_logger.debug("Callback timing metrics could not be initialized", exc_info=True)
        return None


@contextmanager
def measure_callback_execution(callback_name: str | None, hook: str, execution_path: ExecutionPath) -> Iterator[None]:
    if callback_name is None or not _callback_timing_enabled():
        yield
        return
    with _REGISTRATION_LOCK:
        histogram: Final = _load_histogram()
    if histogram is None:
        yield
        return

    started: Final = monotonic()
    try:
        yield
    finally:
        duration: Final = monotonic() - started
        try:
            histogram.labels(callback_name, hook, execution_path).observe(duration)
        except Exception:  # noqa: BLE001  # reporting must not suppress the callback result or exception
            verbose_logger.debug("Callback execution timing could not be recorded", exc_info=True)


@cache
def logging_worker_dispatch_timing_enabled() -> bool:
    return os.getenv("LITELLM_LOGGING_WORKER_DISPATCH_TIMING_METRICS", "").lower() == "true"


@cache
def _load_dispatch_wait_histogram() -> "Histogram | None":
    try:
        from prometheus_client import Histogram

        return Histogram(
            "litellm_logging_worker_dispatch_wait_seconds",
            "Time from LoggingWorker submission to first dispatch",
        )
    except Exception:  # noqa: BLE001  # optional metrics must not change logging behavior
        verbose_logger.debug("Logging worker dispatch timing could not be initialized", exc_info=True)
        return None


def observe_logging_worker_dispatch_wait(seconds: float) -> None:
    with _REGISTRATION_LOCK:
        histogram: Final = _load_dispatch_wait_histogram()
    if histogram is not None:
        try:
            histogram.observe(seconds)
        except Exception:  # noqa: BLE001  # reporting must not interrupt logging delivery
            verbose_logger.debug("Logging worker dispatch timing could not be recorded", exc_info=True)


@contextmanager
def callback_stage_span(name: str) -> Iterator[None]:
    """Trace a code-defined stage under a sampled parent. Keep request values out of the name."""
    try:
        from opentelemetry import trace
    except ImportError:
        yield
        return

    active_span: Final = trace.get_current_span()
    parent: Final = active_span if active_span.get_span_context().is_valid else current_request_parent_otel_span()
    parent_context: Final = parent.get_span_context() if parent is not None else None
    if parent_context is None or not parent_context.is_valid or not parent_context.trace_flags.sampled:
        yield
        return

    tracer: Final = trace.get_tracer("litellm.callbacks")
    with tracer.start_as_current_span(
        f"litellm.callback.stage.{name}",
        context=trace.set_span_in_context(parent),
        record_exception=False,
        set_status_on_exception=False,
    ):
        yield
