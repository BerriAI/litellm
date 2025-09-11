"""
Datadog APM traces can be integrated with the logs product by:

1. Having ``ddtrace`` patch the ``logging`` module. This will add trace
attributes to the log record.

2. Updating the log formatter used by the application. In order to inject
tracing information into a log the formatter must be updated to include the
tracing attributes from the log record.


Enabling
--------

Patch ``logging``
~~~~~~~~~~~~~~~~~

There are a few ways to tell ddtrace to patch the ``logging`` module:

1. If using :ref:`ddtrace-run<ddtracerun>`, you can set the environment variable ``DD_LOGS_INJECTION=true``.

2. Use :func:`patch()<ddtrace.patch>` to manually enable the integration::

    from ddtrace import patch
    patch(logging=True)

3. (beta) Set ``log_injection_enabled`` at runtime via the Datadog UI.


Update Log Format
~~~~~~~~~~~~~~~~~

Make sure that your log format exactly matches the following::

    import logging
    from ddtrace import tracer

    FORMAT = ('%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] '
              '[dd.service=%(dd.service)s dd.env=%(dd.env)s '
              'dd.version=%(dd.version)s '
              'dd.trace_id=%(dd.trace_id)s dd.span_id=%(dd.span_id)s] '
              '- %(message)s')
    logging.basicConfig(format=FORMAT)
    log = logging.getLogger()
    log.level = logging.INFO


    @tracer.wrap()
    def hello():
        log.info('Hello, World!')

    hello()

Note that most host based setups log by default to UTC time.
If the log timestamps aren't automatically in UTC, the formatter can be updated to use UTC::

    import time
    logging.Formatter.converter = time.gmtime

For more information, please see the attached guide on common timestamp issues:
https://docs.datadoghq.com/logs/guide/logs-not-showing-expected-timestamp/
"""

from ddtrace.internal.utils.importlib import require_modules


required_modules = ["logging"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.logging.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        # Expose public methods
        from ddtrace.contrib.internal.logging.patch import get_version
        from ddtrace.contrib.internal.logging.patch import patch
        from ddtrace.contrib.internal.logging.patch import unpatch

        __all__ = ["patch", "unpatch", "get_version"]
