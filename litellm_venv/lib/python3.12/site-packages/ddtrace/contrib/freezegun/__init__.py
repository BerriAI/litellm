"""
The freezegun integration reconfigures freezegun's default ignore list to ignore ddtrace.

Enabling
~~~~~~~~
The freezegun integration is enabled by default. Use :func:`patch()<ddtrace.patch>` to enable the integration::
    from ddtrace import patch
    patch(freezegun=True)


Configuration
~~~~~~~~~~~~~
The freezegun integration is not configurable, but may be disabled using DD_PATCH_MODULES=freezegun:false .
"""

from ...internal.utils.importlib import require_modules


required_modules = ["freezegun"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Expose public methods
        from ..internal.freezegun.patch import get_version
        from ..internal.freezegun.patch import patch
        from ..internal.freezegun.patch import unpatch

        __all__ = ["get_version", "patch", "unpatch"]
