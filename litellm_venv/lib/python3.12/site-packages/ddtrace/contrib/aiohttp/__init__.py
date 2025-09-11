"""
The ``aiohttp`` integration traces requests made with the client or to the server.

The client is automatically instrumented while the server must be manually instrumented using middleware.

Client
******

Enabling
~~~~~~~~

The client integration is enabled automatically when using
:ref:`ddtrace-run<ddtracerun>` or :ref:`import ddtrace.auto<ddtraceauto>`.

Or use :func:`patch()<ddtrace.patch>` to manually enable the integration::

    from ddtrace import patch
    patch(aiohttp=True)


Global Configuration
~~~~~~~~~~~~~~~~~~~~

.. py:data:: ddtrace.config.aiohttp_client['distributed_tracing']

   Include distributed tracing headers in requests sent from the aiohttp client.

   This option can also be set with the ``DD_AIOHTTP_CLIENT_DISTRIBUTED_TRACING``
   environment variable.

   Default: ``True``

.. py:data:: ddtrace.config.aiohttp_client['split_by_domain']

   Whether or not to use the domain name of requests as the service name.

   Default: ``False``


Server
******

Enabling
~~~~~~~~

Automatic instrumentation is not available for the server, instead
the provided ``trace_app`` function must be used::

    from aiohttp import web
    from ddtrace import tracer, patch
    from ddtrace.contrib.aiohttp import trace_app

    # create your application
    app = web.Application()
    app.router.add_get('/', home_handler)

    # trace your application handlers
    trace_app(app, tracer, service='async-api')
    web.run_app(app, port=8000)

Integration settings are attached to your application under the ``datadog_trace``
namespace. You can read or update them as follows::

    # disables distributed tracing for all received requests
    app['datadog_trace']['distributed_tracing_enabled'] = False

Available settings are:

* ``tracer`` (default: ``ddtrace.tracer``): set the default tracer instance that is used to
  trace `aiohttp` internals. By default the `ddtrace` tracer is used.
* ``service`` (default: ``aiohttp-web``): set the service name used by the tracer. Usually
  this configuration must be updated with a meaningful name.
* ``distributed_tracing_enabled`` (default: ``True``): enable distributed tracing during
  the middleware execution, so that a new span is created with the given ``trace_id`` and
  ``parent_id`` injected via request headers.

When a request span is created, a new ``Context`` for this logical execution is attached
to the ``request`` object, so that it can be used in the application code::

    async def home_handler(request):
        ctx = request['datadog_context']
        # do something with the tracing Context

:ref:`All HTTP tags <http-tagging>` are supported for this integration.

"""
from ddtrace.internal.utils.importlib import require_modules


required_modules = ["aiohttp"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.aiohttp.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001
        from ddtrace.contrib.internal.aiohttp.middlewares import trace_app
        from ddtrace.contrib.internal.aiohttp.patch import get_version

        # Expose public methods
        from ddtrace.contrib.internal.aiohttp.patch import patch
        from ddtrace.contrib.internal.aiohttp.patch import unpatch

        __all__ = ["patch", "unpatch", "trace_app", "get_version"]
