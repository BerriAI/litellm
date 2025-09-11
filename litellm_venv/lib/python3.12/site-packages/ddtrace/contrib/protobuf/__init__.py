"""
The Protobuf integration will trace all Protobuf read / write calls made with the ``google.protobuf``
library. This integration is enabled by default.

Enabling
~~~~~~~~

The protobuf integration is enabled by default. Use
:func:`patch()<ddtrace.patch>` to enable the integration::

    from ddtrace import patch
    patch(protobuf=True)

Configuration
~~~~~~~~~~~~~

"""
from ...internal.utils.importlib import require_modules


required_modules = ["protobuf"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Expose public methods
        from ..internal.protobuf.patch import get_version
        from ..internal.protobuf.patch import patch
        from ..internal.protobuf.patch import unpatch

        __all__ = ["patch", "unpatch", "get_version"]
