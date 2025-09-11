"""
The molten web framework is automatically traced by ``ddtrace``::

    import ddtrace.auto
    from molten import App, Route

    def hello(name: str, age: int) -> str:
        return f'Hello {age} year old named {name}!'
    app = App(routes=[Route('/hello/{name}/{age}', hello)])


You may also enable molten tracing automatically via ``ddtrace-run``::

    ddtrace-run python app.py


Configuration
~~~~~~~~~~~~~

.. py:data:: ddtrace.config.molten['distributed_tracing']

   Whether to parse distributed tracing headers from requests received by your Molten app.

   Default: ``True``

.. py:data:: ddtrace.config.molten['service_name']

   The service name reported for your Molten app.

   Can also be configured via the ``DD_SERVICE`` or ``DD_MOLTEN_SERVICE`` environment variables.

   Default: ``'molten'``

:ref:`All HTTP tags <http-tagging>` are supported for this integration.

"""
from ddtrace.internal.utils.importlib import require_modules


required_modules = ["molten"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.molten.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        # Expose public methods
        from ddtrace.contrib.internal.molten.patch import get_version
        from ddtrace.contrib.internal.molten.patch import patch
        from ddtrace.contrib.internal.molten.patch import unpatch

        __all__ = ["patch", "unpatch", "get_version"]
