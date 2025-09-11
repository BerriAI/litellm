"""
The redis integration traces redis requests.


Enabling
~~~~~~~~

The redis integration is enabled automatically when using
:ref:`ddtrace-run<ddtracerun>` or :ref:`import ddtrace.auto<ddtraceauto>`.

Or use :func:`patch()<ddtrace.patch>` to manually enable the integration::

    from ddtrace import patch
    patch(redis=True)


Global Configuration
~~~~~~~~~~~~~~~~~~~~

.. py:data:: ddtrace.config.redis["service"]

   The service name reported by default for redis traces.

   This option can also be set with the ``DD_REDIS_SERVICE`` environment
   variable.

   Default: ``"redis"``


.. py:data:: ddtrace.config.redis["cmd_max_length"]

   Max allowable size for the redis command span tag.
   Anything beyond the max length will be replaced with ``"..."``.

   This option can also be set with the ``DD_REDIS_CMD_MAX_LENGTH`` environment
   variable.

   Default: ``1000``


.. py:data:: ddtrace.config.redis["resource_only_command"]

   The span resource will only include the command executed. To include all
   arguments in the span resource, set this value to ``False``.

   This option can also be set with the ``DD_REDIS_RESOURCE_ONLY_COMMAND`` environment
   variable.

   Default: ``True``


Instance Configuration
~~~~~~~~~~~~~~~~~~~~~~

To configure particular redis instances use the :class:`Pin <ddtrace.Pin>` API::

    import redis
    from ddtrace import Pin

    client = redis.StrictRedis(host="localhost", port=6379)

    # Override service name for this instance
    Pin.override(client, service="my-custom-queue")

    # Traces reported for this client will now have "my-custom-queue"
    # as the service name.
    client.get("my-key")
"""

from ddtrace.internal.utils.importlib import require_modules


required_modules = ["redis", "redis.client"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.redis.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        # Expose public methods
        from ddtrace.contrib.internal.redis.patch import get_version
        from ddtrace.contrib.internal.redis.patch import patch

        __all__ = ["patch", "get_version"]
