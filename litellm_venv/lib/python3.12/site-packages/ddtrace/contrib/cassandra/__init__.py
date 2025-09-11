"""Instrument Cassandra to report Cassandra queries.

``import ddtrace.auto`` will automatically patch your Cluster instance to make it work.
::

    from ddtrace import Pin, patch
    from cassandra.cluster import Cluster

    # If not patched yet, you can patch cassandra specifically
    patch(cassandra=True)

    # This will report spans with the default instrumentation
    cluster = Cluster(contact_points=["127.0.0.1"], port=9042)
    session = cluster.connect("my_keyspace")
    # Example of instrumented query
    session.execute("select id from my_table limit 10;")

    # Use a pin to specify metadata related to this cluster
    cluster = Cluster(contact_points=['10.1.1.3', '10.1.1.4', '10.1.1.5'], port=9042)
    Pin.override(cluster, service='cassandra-backend')
    session = cluster.connect("my_keyspace")
    session.execute("select id from my_table limit 10;")
"""
from ddtrace.internal.utils.importlib import require_modules


required_modules = ["cassandra.cluster"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.cassandra.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        # Expose public methods
        from ddtrace.contrib.internal.cassandra.patch import patch
        from ddtrace.contrib.internal.cassandra.session import get_version

        __all__ = ["patch", "get_version"]
