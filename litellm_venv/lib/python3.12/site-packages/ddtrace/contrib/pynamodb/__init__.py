"""
The PynamoDB integration traces all db calls made with the pynamodb
library through the connection API.

Enabling
~~~~~~~~

The PynamoDB integration is enabled automatically when using
:ref:`ddtrace-run<ddtracerun>` or :ref:`import ddtrace.auto<ddtraceauto>`.

Or use :func:`patch()<ddtrace.patch>` to manually enable the integration::

    import pynamodb
    from ddtrace import patch, config
    patch(pynamodb=True)

Global Configuration
~~~~~~~~~~~~~~~~~~~~

.. py:data:: ddtrace.config.pynamodb["service"]

   The service name reported by default for the PynamoDB instance.

   This option can also be set with the ``DD_PYNAMODB_SERVICE`` environment
   variable.

   Default: ``"pynamodb"``

"""


from ddtrace.internal.utils.importlib import require_modules


required_modules = ["pynamodb.connection.base"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.pynamodb.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        # Expose public methods
        from ddtrace.contrib.internal.pynamodb.patch import get_version
        from ddtrace.contrib.internal.pynamodb.patch import patch

        __all__ = ["patch", "get_version"]
