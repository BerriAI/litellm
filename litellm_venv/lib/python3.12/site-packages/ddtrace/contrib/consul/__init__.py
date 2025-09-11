"""Instrument Consul to trace KV queries.

Only supports tracing for the synchronous client.

``import ddtrace.auto`` will automatically patch your Consul client to make it work.
::

    from ddtrace import Pin, patch
    import consul

    # If not patched yet, you can patch consul specifically
    patch(consul=True)

    # This will report a span with the default settings
    client = consul.Consul(host="127.0.0.1", port=8500)
    client.get("my-key")

    # Use a pin to specify metadata related to this client
    Pin.override(client, service='consul-kv')
"""

from ddtrace.internal.utils.importlib import require_modules


required_modules = ["consul"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.consul.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        # Expose public methods
        from ddtrace.contrib.internal.consul.patch import get_version
        from ddtrace.contrib.internal.consul.patch import patch
        from ddtrace.contrib.internal.consul.patch import unpatch

        __all__ = ["patch", "unpatch", "get_version"]
