r"""
Datadog APM traces can be integrated with the logs produced by ```loguru`` by:

1. Having ``ddtrace`` patch the ``loguru`` module. This will configure a
patcher which appends trace related values to the log.

2. Ensuring the logger has a format which emits new values from the log record

3. For log correlation between APM and logs, the easiest format is via JSON
so that no further configuration needs to be done in the Datadog UI assuming
that the Datadog trace values are at the top level of the JSON

Enabling
--------

Patch ``loguru``
~~~~~~~~~~~~~~~~~~~

If using :ref:`ddtrace-run<ddtracerun>` then set the environment variable ``DD_LOGS_INJECTION=true``.

Or use :func:`patch()<ddtrace.patch>` to manually enable the integration::

    from ddtrace import patch
    patch(loguru=True)

Proper Formatting
~~~~~~~~~~~~~~~~~

The trace values are patched to every log at the top level of the record. In order to correlate
logs, it is highly recommended to use JSON logs. Here are two ways to do this:

1. Use the built-in serialize function within the library that emits the entire log record into a JSON log::

    from loguru import logger

    logger.add("app.log", serialize=True)

This will emit the entire log record with the trace values into a file "app.log"

2. Create a custom format that includes the trace values in JSON format::

    def serialize(record):
        subset = {
            "message": record["message"],
            "dd.trace_id": record["dd.trace_id"],
            "dd.span_id": record["dd.span_id"],
            "dd.env": record["dd.env"],
            "dd.version": record["dd.version"],
            "dd.service": record["dd.service"],
        }
    return json.dumps(subset)

    def log_format(record):
        record["extra"]["serialized"] = serialize(record)
        return "{extra[serialized]}\n"
    logger.add("app.log", format=log_format)

This will emit the log in a format where the output contains the trace values of the log at the top level of a JSON
along with the message. The log will not include all the possible information in the record, but rather only the values
included in the subset object within the ``serialize`` method

For more information, please see the attached guide for the Datadog Logging Product:
https://docs.datadoghq.com/logs/log_collection/python/
"""

from ddtrace.internal.utils.importlib import require_modules


required_modules = ["loguru"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.loguru.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        # Expose public methods
        from ddtrace.contrib.internal.loguru.patch import get_version
        from ddtrace.contrib.internal.loguru.patch import patch
        from ddtrace.contrib.internal.loguru.patch import unpatch

        __all__ = ["patch", "unpatch", "get_version"]
