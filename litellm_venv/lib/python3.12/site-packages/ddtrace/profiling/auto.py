"""Automatically starts a collector when imported."""
from ddtrace.internal.logger import get_logger
from ddtrace.profiling.bootstrap import sitecustomize  # noqa:F401


log = get_logger(__name__)
log.debug("Enabling the profiler by auto import")

start_profiler = sitecustomize.start_profiler
