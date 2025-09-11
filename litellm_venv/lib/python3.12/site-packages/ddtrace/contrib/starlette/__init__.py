"""
The Starlette integration will trace requests to and from Starlette.


Enabling
~~~~~~~~

The starlette integration is enabled automatically when using
:ref:`ddtrace-run<ddtracerun>` or :ref:`import ddtrace.auto<ddtraceauto>`.

Or use :func:`patch()<ddtrace.patch>` to manually enable the integration::

    from ddtrace import patch
    from starlette.applications import Starlette

    patch(starlette=True)
    app = Starlette()


Configuration
~~~~~~~~~~~~~

.. py:data:: ddtrace.config.starlette['distributed_tracing']

   Whether to parse distributed tracing headers from requests received by your Starlette app.

   Can also be enabled with the ``DD_STARLETTE_DISTRIBUTED_TRACING`` environment variable.

   Default: ``True``

.. py:data:: ddtrace.config.starlette['service_name']

   The service name reported for your starlette app.

   Can also be configured via the ``DD_SERVICE`` environment variable.

   Default: ``'starlette'``

.. py:data:: ddtrace.config.starlette['request_span_name']

   The span name for a starlette request.

   Default: ``'starlette.request'``


Example::

    from ddtrace import config

    # Enable distributed tracing
    config.starlette['distributed_tracing'] = True

    # Override service name
    config.starlette['service_name'] = 'custom-service-name'

    # Override request span name
    config.starlette['request_span_name'] = 'custom-request-span-name'

"""
from ddtrace.internal.utils.importlib import require_modules


required_modules = ["starlette"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.starlette.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        # Expose public methods
        from ddtrace.contrib.internal.starlette.patch import get_version
        from ddtrace.contrib.internal.starlette.patch import patch
        from ddtrace.contrib.internal.starlette.patch import unpatch

        __all__ = ["patch", "unpatch", "get_version"]
