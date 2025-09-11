"""
The subprocess integration will add tracing to all subprocess executions
started in your application. It will be automatically enabled if Application
Security is enabled with::

    DD_APPSEC_ENABLED=true


Configuration
~~~~~~~~~~~~~

.. py:data:: ddtrace.config.subprocess['sensitive_wildcards']

   Comma separated list of fnmatch-style wildcards Subprocess parameters matching these
   wildcards will be scrubbed and replaced by a "?".

   Default: ``None`` for the config value but note that there are some wildcards always
   enabled in this integration that you can check on
   ```ddtrace.contrib.subprocess.constants.SENSITIVE_WORDS_WILDCARDS```.
"""

from ddtrace.internal.utils.importlib import require_modules


required_modules = ["os", "subprocess"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.subprocess.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        # Expose public methods
        from ddtrace.contrib.internal.subprocess.patch import get_version
        from ddtrace.contrib.internal.subprocess.patch import patch
        from ddtrace.contrib.internal.subprocess.patch import unpatch

        __all__ = ["patch", "unpatch", "get_version"]
