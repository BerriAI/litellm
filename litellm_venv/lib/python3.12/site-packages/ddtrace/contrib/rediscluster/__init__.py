"""Instrument rediscluster to report Redis Cluster queries.

``import ddtrace.auto`` will automatically patch your Redis Cluster client to make it work.
::

    from ddtrace import Pin, patch
    import rediscluster

    # If not patched yet, you can patch redis specifically
    patch(rediscluster=True)

    # This will report a span with the default settings
    client = rediscluster.StrictRedisCluster(startup_nodes=[{'host':'localhost', 'port':'7000'}])
    client.get('my-key')

    # Use a pin to specify metadata related to this client
    Pin.override(client, service='redis-queue')

Global Configuration
~~~~~~~~~~~~~~~~~~~~

.. py:data:: ddtrace.config.rediscluster["service"]
   The service name reported by default for rediscluster spans

   The option can also be set with the ``DD_REDISCLUSTER_SERVICE`` environment variable

   Default: ``'rediscluster'``


.. py:data:: ddtrace.config.rediscluster["cmd_max_length"]

   Max allowable size for the rediscluster command span tag.
   Anything beyond the max length will be replaced with ``"..."``.

   This option can also be set with the ``DD_REDISCLUSTER_CMD_MAX_LENGTH`` environment
   variable.

   Default: ``1000``

.. py:data:: ddtrace.config.aredis["resource_only_command"]

   The span resource will only include the command executed. To include all
   arguments in the span resource, set this value to ``False``.

   This option can also be set with the ``DD_REDIS_RESOURCE_ONLY_COMMAND`` environment
   variable.

   Default: ``True``
"""

from ddtrace.internal.utils.importlib import require_modules


required_modules = ["rediscluster", "rediscluster.client"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.rediscluster.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        # Expose public methods
        from ddtrace.contrib.internal.rediscluster.patch import get_version
        from ddtrace.contrib.internal.rediscluster.patch import patch

        __all__ = ["patch", "get_version"]
