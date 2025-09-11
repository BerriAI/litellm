"""
The Datadog WSGI middleware traces all WSGI requests.


Usage
~~~~~

The middleware can be used manually via the following command::


    from ddtrace.contrib.wsgi import DDWSGIMiddleware

    # application is a WSGI application
    application = DDWSGIMiddleware(application)


Global Configuration
~~~~~~~~~~~~~~~~~~~~

.. py:data:: ddtrace.config.wsgi["service"]

   The service name reported for the WSGI application.

   This option can also be set with the ``DD_SERVICE`` environment
   variable.

   Default: ``"wsgi"``

.. py:data:: ddtrace.config.wsgi["distributed_tracing"]

   Configuration that allows distributed tracing to be enabled.

   Default: ``True``


:ref:`All HTTP tags <http-tagging>` are supported for this integration.

"""
from ddtrace.contrib.internal.wsgi.wsgi import DDWSGIMiddleware
from ddtrace.contrib.internal.wsgi.wsgi import get_version


__all__ = ["DDWSGIMiddleware", "get_version"]
