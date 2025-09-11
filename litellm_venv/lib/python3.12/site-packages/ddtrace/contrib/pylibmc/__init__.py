"""Instrument pylibmc to report Memcached queries.

``import ddtrace.auto`` will automatically patch your pylibmc client to make it work.
::

    # Be sure to import pylibmc and not pylibmc.Client directly,
    # otherwise you won't have access to the patched version
    from ddtrace import Pin, patch
    import pylibmc

    # If not patched yet, you can patch pylibmc specifically
    patch(pylibmc=True)

    # One client instrumented with default configuration
    client = pylibmc.Client(["localhost:11211"]
    client.set("key1", "value1")

    # Use a pin to specify metadata related to this client
    Pin.override(client, service="memcached-sessions")
"""

from ddtrace.internal.utils.importlib import require_modules


required_modules = ["pylibmc"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.pylibmc.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001
        # Expose public methods
        from ddtrace.contrib.internal.pylibmc.client import TracedClient
        from ddtrace.contrib.internal.pylibmc.patch import get_version
        from ddtrace.contrib.internal.pylibmc.patch import patch

        __all__ = ["TracedClient", "patch", "get_version"]
