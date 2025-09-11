"""
The Selenium integration enriches Test Visibility data with extra tags and, if available,
Real User Monitoring session replays.

Enabling
~~~~~~~~

The Selenium integration is enabled by default in test contexts (eg: pytest, or unittest). Use
:func:`patch()<ddtrace.patch>` to enable the integration::

    from ddtrace import patch
    patch(selenium=True)


When using pytest, the `--ddtrace-patch-all` flag is required in order for this integration to
be enabled.

Configuration
~~~~~~~~~~~~~

The Selenium integration can be configured using the following options:

DD_CIVISIBILITY_RUM_FLUSH_WAIT_MILLIS: The time in milliseconds to wait after flushing the RUM session.
"""
from ...internal.utils.importlib import require_modules


required_modules = ["selenium"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Expose public methods
        from ..internal.selenium.patch import get_version
        from ..internal.selenium.patch import patch
        from ..internal.selenium.patch import unpatch

        __all__ = ["get_version", "patch", "unpatch"]
