import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from threading import Lock
from time import monotonic
from typing import TYPE_CHECKING, Final, Literal

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
