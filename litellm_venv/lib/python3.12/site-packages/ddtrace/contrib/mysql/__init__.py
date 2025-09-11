"""
The mysql integration instruments the mysql library to trace MySQL queries.


Enabling
~~~~~~~~

The mysql integration is enabled automatically when using
:ref:`ddtrace-run<ddtracerun>` or :ref:`import ddtrace.auto<ddtraceauto>`.

Or use :func:`patch()<ddtrace.patch>` to manually enable the integration::

    from ddtrace import patch
    patch(mysql=True)


Global Configuration
~~~~~~~~~~~~~~~~~~~~

.. py:data:: ddtrace.config.mysql["service"]

   The service name reported by default for mysql spans.

   This option can also be set with the ``DD_MYSQL_SERVICE`` environment
   variable.

   Default: ``"mysql"``

.. py:data:: ddtrace.config.mysql["trace_fetch_methods"]

   Whether or not to trace fetch methods.

   Can also configured via the ``DD_MYSQL_TRACE_FETCH_METHODS`` environment variable.

   Default: ``False``


Instance Configuration
~~~~~~~~~~~~~~~~~~~~~~

To configure the mysql integration on an per-connection basis use the
``Pin`` API::

    from ddtrace import Pin
    # Make sure to import mysql.connector and not the 'connect' function,
    # otherwise you won't have access to the patched version
    import mysql.connector

    # This will report a span with the default settings
    conn = mysql.connector.connect(user="alice", password="b0b", host="localhost", port=3306, database="test")

    # Use a pin to override the service name for this connection.
    Pin.override(conn, service='mysql-users')

    cursor = conn.cursor()
    cursor.execute("SELECT 6*7 AS the_answer;")


Only the default full-Python integration works. The binary C connector,
provided by _mysql_connector, is not supported.

Help on mysql.connector can be found on:
https://dev.mysql.com/doc/connector-python/en/
"""
from ddtrace.internal.utils.importlib import require_modules


# check `mysql-connector` availability
required_modules = ["mysql.connector"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.mysql.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        # Expose public methods
        from ddtrace.contrib.internal.mysql.patch import get_version
        from ddtrace.contrib.internal.mysql.patch import patch

        __all__ = ["patch", "get_version"]
