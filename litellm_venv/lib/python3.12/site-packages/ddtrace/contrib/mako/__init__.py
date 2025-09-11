"""
The ``mako`` integration traces templates rendering.
Auto instrumentation is available using the ``patch``. The following is an example::

    from ddtrace import patch
    from mako.template import Template

    patch(mako=True)

    t = Template(filename="index.html")

"""
from ddtrace.internal.utils.importlib import require_modules


required_modules = ["mako"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.mako.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        # Expose public methods
        from ddtrace.contrib.internal.mako.patch import get_version
        from ddtrace.contrib.internal.mako.patch import patch
        from ddtrace.contrib.internal.mako.patch import unpatch

        __all__ = ["patch", "unpatch", "get_version"]
