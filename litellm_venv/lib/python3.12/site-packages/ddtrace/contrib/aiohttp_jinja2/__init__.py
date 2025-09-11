"""
The ``aiohttp_jinja2`` integration adds tracing of template rendering.


Enabling
~~~~~~~~

The integration is enabled automatically when using
:ref:`ddtrace-run<ddtracerun>` or :ref:`import ddtrace.auto<ddtraceauto>`.

Or use :func:`patch()<ddtrace.patch>` to manually enable the integration::

    from ddtrace import patch
    patch(aiohttp_jinja2=True)
"""
from ddtrace.internal.utils.importlib import require_modules


required_modules = ["aiohttp_jinja2"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.aiohttp.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        from ddtrace.contrib.internal.aiohttp_jinja2.patch import get_version
        from ddtrace.contrib.internal.aiohttp_jinja2.patch import patch
        from ddtrace.contrib.internal.aiohttp_jinja2.patch import unpatch

        __all__ = ["patch", "unpatch", "get_version"]
