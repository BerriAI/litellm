"""
The Vertica integration will trace queries made using the vertica-python
library.

Vertica will be automatically instrumented with ``import ddtrace.auto``, or when using
the ``ddtrace-run`` command.

Vertica is instrumented on import. To instrument Vertica manually use the
``patch`` function. Note the ordering of the following statements::

    from ddtrace import patch
    patch(vertica=True)

    import vertica_python

    # use vertica_python like usual


To configure the Vertica integration globally you can use the ``Config`` API::

    from ddtrace import config, patch
    patch(vertica=True)

    config.vertica['service_name'] = 'my-vertica-database'


To configure the Vertica integration on an instance-per-instance basis use the
``Pin`` API::

    from ddtrace import Pin, patch, Tracer
    patch(vertica=True)

    import vertica_python

    custom_tracer = Tracer()
    conn = vertica_python.connect(**YOUR_VERTICA_CONFIG)

    # override the service and tracer to be used
    Pin.override(conn, service='myverticaservice', tracer=custom_tracer)
"""

from ddtrace.internal.utils.importlib import require_modules


required_modules = ["vertica_python"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.vertica.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        # Expose public methods
        from ddtrace.contrib.internal.vertica.patch import get_version
        from ddtrace.contrib.internal.vertica.patch import patch
        from ddtrace.contrib.internal.vertica.patch import unpatch

        __all__ = ["patch", "unpatch", "get_version"]
