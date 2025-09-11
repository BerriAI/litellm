"""
The ``futures`` integration propagates the current active tracing context
to tasks spawned using a :class:`~concurrent.futures.ThreadPoolExecutor`.
The integration ensures that when operations are executed in another thread,
those operations can continue the previously generated trace.


Enabling
~~~~~~~~

The futures integration is enabled automatically when using
:ref:`ddtrace-run<ddtracerun>` or :ref:`import ddtrace.auto<ddtraceauto>`.

Or use :func:`patch()<ddtrace.patch>` to manually enable the integration::

    from ddtrace import patch
    patch(futures=True)
"""
from ddtrace.internal.utils.importlib import require_modules


required_modules = ["concurrent.futures"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.futures.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        # Expose public methods
        from ddtrace.contrib.internal.futures.patch import get_version
        from ddtrace.contrib.internal.futures.patch import patch
        from ddtrace.contrib.internal.futures.patch import unpatch

        __all__ = [
            "get_version",
            "patch",
            "unpatch",
        ]
