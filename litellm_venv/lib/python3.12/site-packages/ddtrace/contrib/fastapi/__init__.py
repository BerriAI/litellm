"""
The fastapi integration will trace requests to and from FastAPI.

Enabling
~~~~~~~~

The fastapi integration is enabled automatically when using
:ref:`ddtrace-run<ddtracerun>` or :ref:`import ddtrace.auto<ddtraceauto>`.

Or use :func:`patch()<ddtrace.patch>` to manually enable the integration::

    from ddtrace import patch
    from fastapi import FastAPI

    patch(fastapi=True)
    app = FastAPI()

When registering your own ASGI middleware using FastAPI's ``add_middleware()`` function,
keep in mind that Datadog spans close after your middleware's call to ``await self.app()`` returns.
This means that accesses of span data from within the middleware should be performed
prior to this call.


Configuration
~~~~~~~~~~~~~

.. py:data:: ddtrace.config.fastapi['service_name']

   The service name reported for your fastapi app.

   Can also be configured via the ``DD_SERVICE`` environment variable.

   Default: ``'fastapi'``

.. py:data:: ddtrace.config.fastapi['request_span_name']

   The span name for a fastapi request.

   Default: ``'fastapi.request'``


Example::

    from ddtrace import config

    # Override service name
    config.fastapi['service_name'] = 'custom-service-name'

    # Override request span name
    config.fastapi['request_span_name'] = 'custom-request-span-name'

"""
from ddtrace.internal.utils.importlib import require_modules


required_modules = ["fastapi"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.fastapi.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        # Expose public methods
        from ddtrace.contrib.internal.fastapi.patch import get_version
        from ddtrace.contrib.internal.fastapi.patch import patch
        from ddtrace.contrib.internal.fastapi.patch import unpatch

        __all__ = ["patch", "unpatch", "get_version"]
