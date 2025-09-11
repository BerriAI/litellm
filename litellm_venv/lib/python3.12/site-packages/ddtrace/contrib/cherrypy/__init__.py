"""
The Cherrypy trace middleware will track request timings.
It uses the cherrypy hooks and creates a tool to track requests and errors


Usage
~~~~~
To install the middleware, add::

    from ddtrace import tracer
    from ddtrace.contrib.cherrypy import TraceMiddleware

and create a `TraceMiddleware` object::

    traced_app = TraceMiddleware(cherrypy, tracer, service="my-cherrypy-app")


Configuration
~~~~~~~~~~~~~

.. py:data:: ddtrace.config.cherrypy['distributed_tracing']

   Whether to parse distributed tracing headers from requests received by your CherryPy app.

   Can also be enabled with the ``DD_CHERRYPY_DISTRIBUTED_TRACING`` environment variable.

   Default: ``True``

.. py:data:: ddtrace.config.cherrypy['service']

   The service name reported for your CherryPy app.

   Can also be configured via the ``DD_SERVICE`` environment variable.

   Default: ``'cherrypy'``


Example::
Here is the end result, in a sample app::

    import cherrypy

    from ddtrace import tracer, Pin
    from ddtrace.contrib.cherrypy import TraceMiddleware
    TraceMiddleware(cherrypy, tracer, service="my-cherrypy-app")

    @cherrypy.tools.tracer()
    class HelloWorld(object):
        def index(self):
            return "Hello World"
        index.exposed = True

    cherrypy.quickstart(HelloWorld())
"""

from ddtrace.internal.utils.importlib import require_modules


required_modules = ["cherrypy"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        from ddtrace.contrib.internal.cherrypy.middleware import TraceMiddleware
        from ddtrace.contrib.internal.cherrypy.middleware import get_version

        __all__ = ["TraceMiddleware", "get_version"]
