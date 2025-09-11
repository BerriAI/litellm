"""
The Sanic__ integration will trace requests to and from Sanic.


Enable Sanic tracing automatically via ``ddtrace-run``::

    ddtrace-run python app.py

Sanic tracing can also be enabled explicitly::

    from ddtrace import patch_all
    patch_all(sanic=True)

    from sanic import Sanic
    from sanic.response import text

    app = Sanic(__name__)

    @app.route('/')
    def index(request):
        return text('hello world')

    if __name__ == '__main__':
        app.run()


Configuration
~~~~~~~~~~~~~

.. py:data:: ddtrace.config.sanic['distributed_tracing_enabled']

   Whether to parse distributed tracing headers from requests received by your Sanic app.

   Default: ``True``


.. py:data:: ddtrace.config.sanic['service_name']

   The service name reported for your Sanic app.

   Can also be configured via the ``DD_SERVICE`` environment variable.

   Default: ``'sanic'``


Example::

    from ddtrace import config

    # Enable distributed tracing
    config.sanic['distributed_tracing_enabled'] = True

    # Override service name
    config.sanic['service_name'] = 'custom-service-name'

.. __: https://sanic.readthedocs.io/en/latest/
"""
from ddtrace.internal.utils.importlib import require_modules


required_modules = ["sanic"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.sanic.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        # Expose public methods
        from ddtrace.contrib.internal.sanic.patch import get_version
        from ddtrace.contrib.internal.sanic.patch import patch
        from ddtrace.contrib.internal.sanic.patch import unpatch

        __all__ = ["patch", "unpatch", "get_version"]
