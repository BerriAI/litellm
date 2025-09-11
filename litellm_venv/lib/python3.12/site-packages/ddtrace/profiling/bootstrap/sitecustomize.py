# -*- encoding: utf-8 -*-
"""Bootstrapping code that is run when using `ddtrace.profiling.auto`."""
from ddtrace.profiling import bootstrap
from ddtrace.profiling import profiler


def start_profiler():
    if hasattr(bootstrap, "profiler"):
        bootstrap.profiler.stop()
    # Export the profiler so we can introspect it if needed
    bootstrap.profiler = profiler.Profiler()
    bootstrap.profiler.start()


start_profiler()
