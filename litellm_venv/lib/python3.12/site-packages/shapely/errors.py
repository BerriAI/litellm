"""Shapely errors."""

import threading

from shapely.lib import GEOSException, ShapelyError, _setup_signal_checks  # noqa: F401


def setup_signal_checks(interval=10000):
    """Enable Python signal checks in the ufunc inner loops.

    Doing so allows termination (using CTRL+C) of operations on large arrays of
    vectors.

    Parameters
    ----------
    interval : int, default 10000
        Check for interrupts every x iterations. The higher the number, the
        slower shapely will respond to a signal. However, at low values there
        will be a negative effect on performance. The default of 10000 does not
        have any measureable effects on performance.

    Notes
    -----
    For more information on signals consult the Python docs:

    https://docs.python.org/3/library/signal.html

    """
    if interval <= 0:
        raise ValueError("Signal checks interval must be greater than zero.")

    _setup_signal_checks(interval, threading.main_thread().ident)


class UnsupportedGEOSVersionError(ShapelyError):
    """Raised when the GEOS library version does not support a certain operation."""


class DimensionError(ShapelyError):
    """An error in the number of coordinate dimensions."""


class TopologicalError(ShapelyError):
    """A geometry is invalid or topologically incorrect."""


class ShapelyDeprecationWarning(FutureWarning):
    """Warning for features that will be removed or changed in a future release."""


class EmptyPartError(ShapelyError):
    """An error signifying an empty part was encountered when creating a multi-part."""


class GeometryTypeError(ShapelyError):
    """An error raised when the geometry has an unrecognized or inappropriate type."""


def __getattr__(name):
    import warnings

    # Alias Shapely 1.8 error classes to ShapelyError with deprecation warning
    if name in [
        "ReadingError",
        "WKBReadingError",
        "WKTReadingError",
        "PredicateError",
        "InvalidGeometryError",
    ]:
        warnings.warn(
            f"{name} is deprecated and will be removed in a future version. "
            "Use ShapelyError instead (functions previously raising {name} "
            "will now raise a ShapelyError instead).",
            FutureWarning,
            stacklevel=2,
        )
        return ShapelyError

    raise AttributeError(f"module 'shapely.errors' has no attribute '{name}'")
