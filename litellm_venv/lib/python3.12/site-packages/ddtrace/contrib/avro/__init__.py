"""
The Avro integration will trace all Avro read / write calls made with the ``avro``
library. This integration is enabled by default.

Enabling
~~~~~~~~

The avro integration is enabled by default. Use
:func:`patch()<ddtrace.patch>` to enable the integration::

    from ddtrace import patch
    patch(avro=True)

Configuration
~~~~~~~~~~~~~

"""
from ...internal.utils.importlib import require_modules


required_modules = ["avro"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Expose public methods
        from ..internal.avro.patch import get_version
        from ..internal.avro.patch import patch

        __all__ = ["patch", "get_version"]
