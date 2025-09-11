"""Provides multi-point element-wise operations such as ``contains``."""

import warnings

import numpy as np

import shapely
from shapely.prepared import PreparedGeometry


def _construct_points(x, y):
    x, y = np.asanyarray(x), np.asanyarray(y)
    if x.shape != y.shape:
        raise ValueError("X and Y shapes must be equivalent.")

    if x.dtype != np.float64:
        x = x.astype(np.float64)
    if y.dtype != np.float64:
        y = y.astype(np.float64)

    return shapely.points(x, y)


def contains(geometry, x, y):
    """Check whether multiple points are contained by a single geometry.

    Vectorized (element-wise) version of `contains`.

    Parameters
    ----------
    geometry : PreparedGeometry or subclass of BaseGeometry
        The geometry which is to be checked to see whether each point is
        contained within. The geometry will be "prepared" if it is not already
        a PreparedGeometry instance.
    x : array
        The x coordinates of the points to check.
    y : array
        The y coordinates of the points to check.

    Returns
    -------
    Mask of points contained by the given `geometry`.

    """
    warnings.warn(
        "The 'shapely.vectorized.contains' function is deprecated and will be "
        "removed a future version. Use 'shapely.contains_xy' instead (available "
        "since shapely 2.0.0).",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(geometry, PreparedGeometry):
        geometry = geometry.context
    shapely.prepare(geometry)
    return shapely.contains_xy(geometry, x, y)


def touches(geometry, x, y):
    """Check whether multiple points touch the exterior of a single geometry.

    Vectorized (element-wise) version of `touches`.

    Parameters
    ----------
    geometry : PreparedGeometry or subclass of BaseGeometry
        The geometry which is to be checked to see whether each point is
        contained within. The geometry will be "prepared" if it is not already
        a PreparedGeometry instance.
    x : array
        The x coordinates of the points to check.
    y : array
        The y coordinates of the points to check.

    Returns
    -------
    Mask of points which touch the exterior of the given `geometry`.

    """
    warnings.warn(
        "The 'shapely.vectorized.touches' function is deprecated and will be "
        "removed a future version. Use 'shapely.intersects_xy(geometry.boundary, x, y)'"
        " instead (available since shapely 2.0.0).",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(geometry, PreparedGeometry):
        geometry = geometry.context
    # Touches(geom, point) == Intersects(Boundary(geom), point)
    boundary = geometry.boundary
    shapely.prepare(boundary)
    return shapely.intersects_xy(boundary, x, y)
