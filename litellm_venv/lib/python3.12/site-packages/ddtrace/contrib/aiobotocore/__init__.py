"""
The aiobotocore integration will trace all AWS calls made with the ``aiobotocore``
library. This integration is not enabled by default.

Enabling
~~~~~~~~

The aiobotocore integration is not enabled by default. Use
:func:`patch()<ddtrace.patch>` to enable the integration::

    from ddtrace import patch
    patch(aiobotocore=True)

Configuration
~~~~~~~~~~~~~

.. py:data:: ddtrace.config.aiobotocore['tag_no_params']

    This opts out of the default behavior of adding span tags for a narrow set of API parameters.

    To not collect any API parameters, ``ddtrace.config.aiobotocore.tag_no_params = True`` or by setting the environment
    variable ``DD_AWS_TAG_NO_PARAMS=true``.


    Default: ``False``

"""
from ddtrace.internal.utils.importlib import require_modules


required_modules = ["aiobotocore.client"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.aiobotocore.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        # Expose public methods
        from ddtrace.contrib.internal.aiobotocore.patch import get_version
        from ddtrace.contrib.internal.aiobotocore.patch import patch

        __all__ = ["patch", "get_version"]
