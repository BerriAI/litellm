"""Instrument mongoengine to report MongoDB queries.

``import ddtrace.auto`` will automatically patch your mongoengine connect method to make it work.
::

    from ddtrace import Pin, patch
    import mongoengine

    # If not patched yet, you can patch mongoengine specifically
    patch(mongoengine=True)

    # At that point, mongoengine is instrumented with the default settings
    mongoengine.connect('db', alias='default')

    # Use a pin to specify metadata related to this client
    client = mongoengine.connect('db', alias='master')
    Pin.override(client, service="mongo-master")
"""

from ddtrace.internal.utils.importlib import require_modules


required_modules = ["mongoengine"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.mongoengine.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        # Expose public methods
        from ddtrace.contrib.internal.mongoengine.patch import get_version
        from ddtrace.contrib.internal.mongoengine.patch import patch

        __all__ = ["patch", "get_version"]
