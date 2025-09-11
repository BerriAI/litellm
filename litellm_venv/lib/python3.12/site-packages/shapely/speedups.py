"""Speedups for Shapely geometry operations.

.. deprecated:: 2.0
        Deprecated in Shapely 2.0, and will be removed in a future version.

"""

import warnings

__all__ = ["available", "disable", "enable", "enabled"]


available = True
enabled = True


_MSG = (
    "This function has no longer any effect, and will be removed in a "
    "future release. Starting with Shapely 2.0, equivalent speedups are "
    "always available"
)


def enable():
    """Will be removed in a future release and has no longer any effect.

    Previously, this function enabled cython-based speedups. Starting with
    Shapely 2.0, equivalent speedups are available in every installation.
    """
    warnings.warn(_MSG, FutureWarning, stacklevel=2)


def disable():
    """Will be removed in a future release and has no longer any effect.

    Previously, this function enabled cython-based speedups. Starting with
    Shapely 2.0, equivalent speedups are available in every installation.
    """
    warnings.warn(_MSG, FutureWarning, stacklevel=2)
