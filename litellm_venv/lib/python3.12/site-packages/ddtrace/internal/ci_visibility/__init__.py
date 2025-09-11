"""
CI Visibility Service.
This is normally started automatically by including ``ddtrace=1`` or ``--ddtrace`` in the pytest run command.
To start the service manually, invoke the ``enable`` method::
    from ddtrace.internal.ci_visibility import CIVisibility
    CIVisibility.enable()
"""

from .recorder import CIVisibility


__all__ = ["CIVisibility"]
