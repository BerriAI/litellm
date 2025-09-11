"""
The pymysql integration instruments the pymysql library to trace MySQL queries.


Enabling
~~~~~~~~

The integration is enabled automatically when using
:ref:`ddtrace-run<ddtracerun>` or :ref:`import ddtrace.auto<ddtraceauto>`.

Or use :func:`patch()<ddtrace.patch>` to manually enable the integration::

    from ddtrace import patch
    patch(pymysql=True)


Global Configuration
~~~~~~~~~~~~~~~~~~~~

.. py:data:: ddtrace.config.pymysql["service"]

   The service name reported by default for pymysql spans.

   This option can also be set with the ``DD_PYMYSQL_SERVICE`` environment
   variable.

   Default: ``"mysql"``

.. py:data:: ddtrace.config.pymysql["trace_fetch_methods"]

   Whether or not to trace fetch methods.

   Can also configured via the ``DD_PYMYSQL_TRACE_FETCH_METHODS`` environment variable.

   Default: ``False``


Instance Configuration
~~~~~~~~~~~~~~~~~~~~~~

To configure the integration on an per-connection basis use the
``Pin`` API::

    from ddtrace import Pin
    from pymysql import connect

    # This will report a span with the default settings
    conn = connect(user="alice", password="b0b", host="localhost", port=3306, database="test")

    # Use a pin to override the service name for this connection.
    Pin.override(conn, service="pymysql-users")


    cursor = conn.cursor()
    cursor.execute("SELECT 6*7 AS the_answer;")
"""

from ddtrace.internal.utils.importlib import require_modules


required_modules = ["pymysql"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.pymysql.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        # Expose public methods
        from ddtrace.contrib.internal.pymysql.patch import get_version
        from ddtrace.contrib.internal.pymysql.patch import patch

        __all__ = ["patch", "get_version"]
