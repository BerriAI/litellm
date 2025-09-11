"""
Instrument aiopg to report a span for each executed Postgres queries::

    from ddtrace import Pin, patch
    import aiopg

    # If not patched yet, you can patch aiopg specifically
    patch(aiopg=True)

    # This will report a span with the default settings
    async with aiopg.connect(DSN) as db:
        with (await db.cursor()) as cursor:
            await cursor.execute("SELECT * FROM users WHERE id = 1")

    # Use a pin to specify metadata related to this connection
    Pin.override(db, service='postgres-users')
"""
from ddtrace.internal.utils.importlib import require_modules


required_modules = ["aiopg"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.aiohttp.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        from ddtrace.contrib.internal.aiopg.patch import get_version
        from ddtrace.contrib.internal.aiopg.patch import patch

        __all__ = ["patch", "get_version"]
