r"""
Datadog APM traces can be integrated with the logs produced by ```logbook`` by:

1. Having ``ddtrace`` patch the ``logbook`` module. This will configure a
patcher which appends trace related values to the log.

2. Ensuring the logger has a format which emits new values from the log record

3. For log correlation between APM and logs, the easiest format is via JSON
so that no further configuration needs to be done in the Datadog UI assuming
that the Datadog trace values are at the top level of the JSON

Enabling
--------

Patch ``logbook``
~~~~~~~~~~~~~~~~~~~

If using :ref:`ddtrace-run<ddtracerun>` then set the environment variable ``DD_LOGS_INJECTION=true``.

Or use :func:`patch()<ddtrace.patch>` to manually enable the integration::

    from ddtrace import patch
    patch(logbook=True)

Proper Formatting
~~~~~~~~~~~~~~~~~

The trace values are patched to every log at the top level of the record. In order to correlate
logs, it is highly recommended to use JSON logs which can be achieved by using a handler with
a proper formatting::

    handler = FileHandler('output.log', format_string='{{\"message\": "{record.message}",'
                                                          '\"dd.trace_id\": "{record.extra[dd.trace_id]}",'
                                                          '\"dd.span_id\": "{record.extra[dd.span_id]}",'
                                                          '\"dd.env\": "{record.extra[dd.env]}",'
                                                          '\"dd.service\": "{record.extra[dd.service]}",'
                                                          '\"dd.version\": "{record.extra[dd.version]}"}}')
    handler.push_application()

Note that the ``extra`` field does not have a ``dd`` object but rather only a ``dd.trace_id``, ``dd.span_id``, etc.
To access the trace values inside extra, please use the ``[]`` operator.

This will create a handler for the application that formats the logs in a way that is JSON with all the
Datadog trace values in a JSON format that can be automatically parsed by the Datadog backend.

For more information, please see the attached guide for the Datadog Logging Product:
https://docs.datadoghq.com/logs/log_collection/python/
"""

from ddtrace.internal.utils.importlib import require_modules


required_modules = ["logbook"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.logbook.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        # Expose public methods
        from ddtrace.contrib.internal.logbook.patch import get_version
        from ddtrace.contrib.internal.logbook.patch import patch
        from ddtrace.contrib.internal.logbook.patch import unpatch

        __all__ = ["patch", "unpatch", "get_version"]
