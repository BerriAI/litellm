import warnings
from enum import IntEnum

import numpy as np

from shapely import _geometry_helpers, geos_version, lib
from shapely._enum import ParamEnum
from shapely.decorators import (
    deprecate_positional,
    multithreading_enabled,
    requires_geos,
)

__all__ = [
    "GeometryType",
    "force_2d",
    "force_3d",
    "get_coordinate_dimension",
    "get_dimensions",
    "get_exterior_ring",
    "get_geometry",
    "get_interior_ring",
    "get_m",
    "get_num_coordinates",
    "get_num_geometries",
    "get_num_interior_rings",
    "get_num_points",
    "get_parts",
    "get_point",
    "get_precision",
    "get_rings",
    "get_srid",
    "get_type_id",
    "get_x",
    "get_y",
    "get_z",
    "set_precision",
    "set_srid",
]


class GeometryType(IntEnum):
    """The enumeration of GEOS geometry types."""

    MISSING = -1
    POINT = 0
    LINESTRING = 1
    LINEARRING = 2
    POLYGON = 3
    MULTIPOINT = 4
    MULTILINESTRING = 5
    MULTIPOLYGON = 6
    GEOMETRYCOLLECTION = 7


# generic


@multithreading_enabled
def get_type_id(geometry, **kwargs):
    """Return the type ID of a geometry.

    Possible values are:

    - None (missing) is -1
    - POINT is 0
    - LINESTRING is 1
    - LINEARRING is 2
    - POLYGON is 3
    - MULTIPOINT is 4
    - MULTILINESTRING is 5
    - MULTIPOLYGON is 6
    - GEOMETRYCOLLECTION is 7

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to get the type ID of.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See Also
    --------
    GeometryType

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString, Point
    >>> shapely.get_type_id(LineString([(0, 0), (1, 1), (2, 2), (3, 3)]))
    1
    >>> shapely.get_type_id([Point(1, 2), Point(2, 3)]).tolist()
    [0, 0]

    """
    return lib.get_type_id(geometry, **kwargs)


@multithreading_enabled
def get_dimensions(geometry, **kwargs):
    """Return the inherent dimensionality of a geometry.

    The inherent dimension is 0 for points, 1 for linestrings and linearrings,
    and 2 for polygons. For geometrycollections it is the max of the containing
    elements. Empty collections and None values return -1.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to get the dimensionality of.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import GeometryCollection, Point, Polygon
    >>> point = Point(0, 0)
    >>> shapely.get_dimensions(point)
    0
    >>> polygon = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
    >>> shapely.get_dimensions(polygon)
    2
    >>> shapely.get_dimensions(GeometryCollection([point, polygon]))
    2
    >>> shapely.get_dimensions(GeometryCollection([]))
    -1
    >>> shapely.get_dimensions(None)
    -1

    """
    return lib.get_dimensions(geometry, **kwargs)


@multithreading_enabled
def get_coordinate_dimension(geometry, **kwargs):
    """Return the dimensionality of the coordinates in a geometry (2, 3 or 4).

    The return value can be one of the following:

    * Return 2 for geometries with XY coordinate types,
    * Return 3 for XYZ or XYM coordinate types
      (distinguished by :meth:`has_z` or :meth:`has_m`),
    * Return 4 for XYZM coordinate types,
    * Return -1 for missing geometries (``None`` values).

    Note that with GEOS < 3.12, if the first Z coordinate equals ``nan``, this function
    will return ``2``. Geometries with M coordinates are supported with GEOS >= 3.12.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to get the coordinate dimension of.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import Point
    >>> shapely.get_coordinate_dimension(Point(0, 0))
    2
    >>> shapely.get_coordinate_dimension(Point(0, 0, 1))
    3
    >>> shapely.get_coordinate_dimension(None)
    -1

    """
    return lib.get_coordinate_dimension(geometry, **kwargs)


@multithreading_enabled
def get_num_coordinates(geometry, **kwargs):
    """Return the total number of coordinates in a geometry.

    Returns 0 for not-a-geometry values.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to get the number of coordinates of.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import GeometryCollection, LineString, Point
    >>> point = Point(0, 0)
    >>> shapely.get_num_coordinates(point)
    1
    >>> shapely.get_num_coordinates(Point(0, 0, 0))
    1
    >>> line = LineString([(0, 0), (1, 1)])
    >>> shapely.get_num_coordinates(line)
    2
    >>> shapely.get_num_coordinates(GeometryCollection([point, line]))
    3
    >>> shapely.get_num_coordinates(None)
    0

    """
    return lib.get_num_coordinates(geometry, **kwargs)


@multithreading_enabled
def get_srid(geometry, **kwargs):
    """Return the SRID of a geometry.

    Returns -1 for not-a-geometry values.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to get the SRID of.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See Also
    --------
    set_srid

    Examples
    --------
    >>> import shapely
    >>> from shapely import Point
    >>> point = Point(0, 0)
    >>> shapely.get_srid(point)
    0
    >>> with_srid = shapely.set_srid(point, 4326)
    >>> shapely.get_srid(with_srid)
    4326

    """
    return lib.get_srid(geometry, **kwargs)


@multithreading_enabled
def set_srid(geometry, srid, **kwargs):
    """Return a geometry with its SRID set.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to set the SRID of.
    srid : int
        The SRID to set on the geometry.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See Also
    --------
    get_srid

    Examples
    --------
    >>> import shapely
    >>> from shapely import Point
    >>> point = Point(0, 0)
    >>> shapely.get_srid(point)
    0
    >>> with_srid = shapely.set_srid(point, 4326)
    >>> shapely.get_srid(with_srid)
    4326

    """
    return lib.set_srid(geometry, np.intc(srid), **kwargs)


# points


@multithreading_enabled
def get_x(point, **kwargs):
    """Return the x-coordinate of a point.

    Parameters
    ----------
    point : Geometry or array_like
        Non-point geometries will result in NaN being returned.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See Also
    --------
    get_y, get_z, get_m

    Examples
    --------
    >>> import shapely
    >>> from shapely import MultiPoint, Point
    >>> shapely.get_x(Point(1, 2))
    1.0
    >>> shapely.get_x(MultiPoint([(1, 1), (1, 2)]))
    nan

    """
    return lib.get_x(point, **kwargs)


@multithreading_enabled
def get_y(point, **kwargs):
    """Return the y-coordinate of a point.

    Parameters
    ----------
    point : Geometry or array_like
        Non-point geometries will result in NaN being returned.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See Also
    --------
    get_x, get_z, get_m

    Examples
    --------
    >>> import shapely
    >>> from shapely import MultiPoint, Point
    >>> shapely.get_y(Point(1, 2))
    2.0
    >>> shapely.get_y(MultiPoint([(1, 1), (1, 2)]))
    nan

    """
    return lib.get_y(point, **kwargs)


@multithreading_enabled
def get_z(point, **kwargs):
    """Return the z-coordinate of a point.

    Parameters
    ----------
    point : Geometry or array_like
        Non-point geometries or geometries without Z dimension will result
        in NaN being returned.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See Also
    --------
    get_x, get_y, get_m

    Examples
    --------
    >>> import shapely
    >>> from shapely import MultiPoint, Point
    >>> shapely.get_z(Point(1, 2, 3))
    3.0
    >>> shapely.get_z(Point(1, 2))
    nan
    >>> shapely.get_z(MultiPoint([(1, 1, 1), (2, 2, 2)]))
    nan

    """
    return lib.get_z(point, **kwargs)


@multithreading_enabled
@requires_geos("3.12.0")
def get_m(point, **kwargs):
    """Return the m-coordinate of a point.

    .. versionadded:: 2.1.0

    Parameters
    ----------
    point : Geometry or array_like
        Non-point geometries or geometries without M dimension will result
        in NaN being returned.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See Also
    --------
    get_x, get_y, get_z

    Examples
    --------
    >>> import shapely
    >>> from shapely import Point, from_wkt
    >>> shapely.get_m(from_wkt("POINT ZM (1 2 3 4)"))
    4.0
    >>> shapely.get_m(from_wkt("POINT M (1 2 4)"))
    4.0
    >>> shapely.get_m(Point(1, 2, 3))
    nan
    >>> shapely.get_m(from_wkt("MULTIPOINT M ((1 1 1), (2 2 2))"))
    nan

    """
    return lib.get_m(point, **kwargs)


# linestrings


@multithreading_enabled
def get_point(geometry, index, **kwargs):
    """Return the nth point of a linestring or linearring.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to get the point of.
    index : int or array_like
        Negative values count from the end of the linestring backwards.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See Also
    --------
    get_num_points

    Examples
    --------
    >>> import shapely
    >>> from shapely import LinearRing, LineString, MultiPoint, Point
    >>> line = LineString([(0, 0), (1, 1), (2, 2), (3, 3)])
    >>> shapely.get_point(line, 1)
    <POINT (1 1)>
    >>> shapely.get_point(line, -2)
    <POINT (2 2)>
    >>> shapely.get_point(line, [0, 3]).tolist()
    [<POINT (0 0)>, <POINT (3 3)>]

    The function works the same for LinearRing input:

    >>> shapely.get_point(LinearRing([(0, 0), (1, 1), (2, 2), (0, 0)]), 1)
    <POINT (1 1)>

    For non-linear geometries it returns None:

    >>> shapely.get_point(MultiPoint([(0, 0), (1, 1), (2, 2), (3, 3)]), 1) is None
    True
    >>> shapely.get_point(Point(1, 1), 0) is None
    True

    """
    return lib.get_point(geometry, np.intc(index), **kwargs)


@multithreading_enabled
def get_num_points(geometry, **kwargs):
    """Return the number of points in a linestring or linearring.

    Returns 0 for not-a-geometry values. The number of points in geometries
    other than linestring or linearring equals zero.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to get the number of points of.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See Also
    --------
    get_point
    get_num_geometries

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString, MultiPoint
    >>> shapely.get_num_points(LineString([(0, 0), (1, 1), (2, 2), (3, 3)]))
    4
    >>> shapely.get_num_points(MultiPoint([(0, 0), (1, 1), (2, 2), (3, 3)]))
    0
    >>> shapely.get_num_points(None)
    0

    """
    return lib.get_num_points(geometry, **kwargs)


# polygons


@multithreading_enabled
def get_exterior_ring(geometry, **kwargs):
    """Return the exterior ring of a polygon.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to get the exterior ring of.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See Also
    --------
    get_interior_ring

    Examples
    --------
    >>> import shapely
    >>> from shapely import Point, Polygon
    >>> shapely.get_exterior_ring(Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)]))
    <LINEARRING (0 0, 0 10, 10 10, 10 0, 0 0)>
    >>> shapely.get_exterior_ring(Point(1, 1)) is None
    True

    """
    return lib.get_exterior_ring(geometry, **kwargs)


@multithreading_enabled
def get_interior_ring(geometry, index, **kwargs):
    """Return the nth interior ring of a polygon.

    The number of interior rings in non-polygons equals zero.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to get the interior ring of.
    index : int or array_like
        Negative values count from the end of the interior rings backwards.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See Also
    --------
    get_exterior_ring
    get_num_interior_rings

    Examples
    --------
    >>> import shapely
    >>> from shapely import Point, Polygon
    >>> polygon_with_hole = Polygon(
    ...     [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
    ...     holes=[[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]]
    ... )
    >>> shapely.get_interior_ring(polygon_with_hole, 0)
    <LINEARRING (2 2, 2 4, 4 4, 4 2, 2 2)>
    >>> shapely.get_interior_ring(polygon_with_hole, 1) is None
    True
    >>> polygon = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
    >>> shapely.get_interior_ring(polygon, 0) is None
    True
    >>> shapely.get_interior_ring(Point(0, 0), 0) is None
    True

    """
    return lib.get_interior_ring(geometry, np.intc(index), **kwargs)


@multithreading_enabled
def get_num_interior_rings(geometry, **kwargs):
    """Return number of internal rings in a polygon.

    Returns 0 for not-a-geometry values.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to get the number of interior rings of.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See Also
    --------
    get_exterior_ring
    get_interior_ring

    Examples
    --------
    >>> import shapely
    >>> from shapely import Point, Polygon
    >>> polygon = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
    >>> shapely.get_num_interior_rings(polygon)
    0
    >>> polygon_with_hole = Polygon(
    ...     [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
    ...     holes=[[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]]
    ... )
    >>> shapely.get_num_interior_rings(polygon_with_hole)
    1
    >>> shapely.get_num_interior_rings(Point(0, 0))
    0
    >>> shapely.get_num_interior_rings(None)
    0

    """
    return lib.get_num_interior_rings(geometry, **kwargs)


# collections


@multithreading_enabled
def get_geometry(geometry, index, **kwargs):
    """Return the nth geometry from a collection of geometries.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to get the nth geometry of.
    index : int or array_like
        Negative values count from the end of the collection backwards.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Notes
    -----
    - simple geometries act as length-1 collections
    - out-of-range values return None

    See Also
    --------
    get_num_geometries, get_parts

    Examples
    --------
    >>> import shapely
    >>> from shapely import Point, MultiPoint
    >>> multipoint = MultiPoint([(0, 0), (1, 1), (2, 2), (3, 3)])
    >>> shapely.get_geometry(multipoint, 1)
    <POINT (1 1)>
    >>> shapely.get_geometry(multipoint, -1)
    <POINT (3 3)>
    >>> shapely.get_geometry(multipoint, 5) is None
    True
    >>> shapely.get_geometry(Point(1, 1), 0)
    <POINT (1 1)>
    >>> shapely.get_geometry(Point(1, 1), 1) is None
    True

    """
    return lib.get_geometry(geometry, np.intc(index), **kwargs)


# Note: future plan is to change this signature over a few releases:
# shapely 2.0:
#   get_parts(geometry, return_index=False)
# shapely 2.1: shows deprecation warning about positional 'return_index'
#   same signature as 2.0
# shapely 2.2(?): enforce keyword-only arguments after 'geometry'
#   get_parts(geometry, *, return_index=False)


@deprecate_positional(["return_index"])
def get_parts(geometry, return_index=False):
    """Get parts of each GeometryCollection or Multi* geometry object.

    A copy of each geometry in the GeometryCollection or Multi* geometry object
    is returned.

    Note: This does not return the individual parts of Multi* geometry objects
    in a GeometryCollection. You may need to call this function multiple times
    to return individual parts of Multi* geometry objects in a
    GeometryCollection.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to get the parts of.
    return_index : bool, default False
        If True, will return a tuple of ndarrays of (parts, indexes), where
        indexes are the indexes of the original geometries in the source array.

    Notes
    -----

    .. deprecated:: 2.1.0
        A deprecation warning is shown if ``return_index`` is specified as
        a positional argument. This will need to be specified as a keyword
        argument in a future release.

    Returns
    -------
    ndarray of parts or tuple of (parts, indexes)

    See Also
    --------
    get_geometry, get_rings

    Examples
    --------
    >>> import shapely
    >>> from shapely import MultiPoint
    >>> shapely.get_parts(MultiPoint([(0, 1), (2, 3)])).tolist()
    [<POINT (0 1)>, <POINT (2 3)>]
    >>> parts, index = shapely.get_parts([MultiPoint([(0, 1)]), \
MultiPoint([(4, 5), (6, 7)])], return_index=True)
    >>> parts.tolist()
    [<POINT (0 1)>, <POINT (4 5)>, <POINT (6 7)>]
    >>> index.tolist()
    [0, 1, 1]

    """
    geometry = np.asarray(geometry, dtype=np.object_)
    geometry = np.atleast_1d(geometry)

    if geometry.ndim != 1:
        raise ValueError("Array should be one dimensional")

    if return_index:
        return _geometry_helpers.get_parts(geometry)

    return _geometry_helpers.get_parts(geometry)[0]


# Note: future plan is to change this signature over a few releases:
# shapely 2.0:
#   get_rings(geometry, return_index=False)
# shapely 2.1: shows deprecation warning about positional 'return_index'
#   same signature as 2.0
# shapely 2.2(?): enforce keyword-only arguments after 'geometry'
#   get_rings(geometry, *, return_index=False)


@deprecate_positional(["return_index"])
def get_rings(geometry, return_index=False):
    """Get rings of Polygon geometry object.

    For each Polygon, the first returned ring is always the exterior ring
    and potential subsequent rings are interior rings.

    If the geometry is not a Polygon, nothing is returned (empty array for
    scalar geometry input or no element in output array for array input).

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to get the rings of.
    return_index : bool, default False
        If True, will return a tuple of ndarrays of (rings, indexes), where
        indexes are the indexes of the original geometries in the source array.

    Notes
    -----

    .. deprecated:: 2.1.0
        A deprecation warning is shown if ``return_index`` is specified as
        a positional argument. This will need to be specified as a keyword
        argument in a future release.

    Returns
    -------
    ndarray of rings or tuple of (rings, indexes)

    See Also
    --------
    get_exterior_ring, get_interior_ring, get_parts

    Examples
    --------
    >>> import shapely
    >>> from shapely import Polygon
    >>> polygon_with_hole = Polygon(
    ...     [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
    ...     holes=[[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]]
    ... )
    >>> shapely.get_rings(polygon_with_hole).tolist()
    [<LINEARRING (0 0, 0 10, 10 10, 10 0, 0 0)>,
     <LINEARRING (2 2, 2 4, 4 4, 4 2, 2 2)>]

    With ``return_index=True``:

    >>> polygon = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
    >>> rings, index = shapely.get_rings(
    ...     [polygon, polygon_with_hole],
    ...     return_index=True
    ... )
    >>> rings.tolist()
    [<LINEARRING (0 0, 2 0, 2 2, 0 2, 0 0)>,
     <LINEARRING (0 0, 0 10, 10 10, 10 0, 0 0)>,
     <LINEARRING (2 2, 2 4, 4 4, 4 2, 2 2)>]
    >>> index.tolist()
    [0, 1, 1]

    """
    geometry = np.asarray(geometry, dtype=np.object_)
    geometry = np.atleast_1d(geometry)

    if geometry.ndim != 1:
        raise ValueError("Array should be one dimensional")

    if return_index:
        return _geometry_helpers.get_parts(geometry, extract_rings=True)

    return _geometry_helpers.get_parts(geometry, extract_rings=True)[0]


@multithreading_enabled
def get_num_geometries(geometry, **kwargs):
    """Return number of geometries in a collection.

    Returns 0 for not-a-geometry values. The number of geometries in points,
    linestrings, linearrings and polygons equals one.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to get the number of geometries of.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See Also
    --------
    get_num_points
    get_geometry

    Examples
    --------
    >>> import shapely
    >>> from shapely import MultiPoint, Point
    >>> shapely.get_num_geometries(MultiPoint([(0, 0), (1, 1), (2, 2), (3, 3)]))
    4
    >>> shapely.get_num_geometries(Point(1, 1))
    1
    >>> shapely.get_num_geometries(None)
    0

    """
    return lib.get_num_geometries(geometry, **kwargs)


@multithreading_enabled
def get_precision(geometry, **kwargs):
    """Get the precision of a geometry.

    If a precision has not been previously set, it will be 0 (double
    precision). Otherwise, it will return the precision grid size that was
    set on a geometry.

    Returns NaN for not-a-geometry values.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to get the precision of.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See Also
    --------
    set_precision

    Examples
    --------
    >>> import shapely
    >>> from shapely import Point
    >>> point = Point(1, 1)
    >>> shapely.get_precision(point)
    0.0
    >>> geometry = shapely.set_precision(point, 1.0)
    >>> shapely.get_precision(geometry)
    1.0
    >>> shapely.get_precision(None)
    nan

    """
    return lib.get_precision(geometry, **kwargs)


class SetPrecisionMode(ParamEnum):
    valid_output = 0
    pointwise = 1
    keep_collapsed = 2


@multithreading_enabled
def set_precision(geometry, grid_size, mode="valid_output", **kwargs):
    """Return geometry with the precision set to a precision grid size.

    By default, geometries use double precision coordinates (grid_size = 0).

    Coordinates will be rounded if the precision grid specified is less precise
    than the input geometry. Duplicated vertices will be dropped from lines and
    polygons for grid sizes greater than 0. Line and polygon geometries may
    collapse to empty geometries if all vertices are closer together than
    ``grid_size`` or if a polygon becomes significantly narrower than
    ``grid_size``. Spikes or sections in polygons narrower than ``grid_size``
    after rounding the vertices will be removed, which can lead to multipolygons
    or empty geometries. Z values, if present, will not be modified.

    Notes
    -----
    * subsequent operations will always be performed in the precision of the
      geometry with higher precision (smaller "grid_size"). That same precision
      will be attached to the operation outputs.
    * input geometries should be geometrically valid; unexpected results may
      occur if input geometries are not.
    * the geometry returned will be in
      :ref:`mild canonical form <canonical-form>`, and the order of vertices can
      change and should not be relied upon.
    * returns None if geometry is None.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to set the precision of.
    grid_size : float
        Precision grid size. If 0, will use double precision (will not modify
        geometry if precision grid size was not previously set). If this
        value is more precise than input geometry, the input geometry will
        not be modified.
    mode : {'valid_output', 'pointwise', 'keep_collapsed'}, default 'valid_output'
        This parameter determines the way a precision reduction is applied on
        the geometry. There are three modes:

        1. `'valid_output'` (default):  The output is always valid. Collapsed
           geometry elements (including both polygons and lines) are removed.
           Duplicate vertices are removed.
        2. `'pointwise'`: Precision reduction is performed pointwise. Output
           geometry may be invalid due to collapse or self-intersection.
           Duplicate vertices are not removed. In GEOS this option is called
           NO_TOPO.

           .. note::

             'pointwise' mode requires at least GEOS 3.10. It is accepted in
             earlier versions, but the results may be unexpected.
        3. `'keep_collapsed'`: Like the default mode, except that collapsed
           linear geometry elements are preserved. Collapsed polygonal input
           elements are removed. Duplicate vertices are removed.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See Also
    --------
    get_precision

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString, Point
    >>> shapely.set_precision(Point(0.9, 0.9), 1.0)
    <POINT (1 1)>
    >>> shapely.set_precision(Point(0.9, 0.9, 0.9), 1.0)
    <POINT Z (1 1 0.9)>
    >>> shapely.set_precision(LineString([(0, 0), (0, 0.1), (0, 1), (1, 1)]), 1.0)
    <LINESTRING (0 0, 0 1, 1 1)>
    >>> shapely.set_precision(LineString([(0, 0), (0, 0.1), (0.1, 0.1)]), 1.0, mode="valid_output")
    <LINESTRING EMPTY>
    >>> shapely.set_precision(LineString([(0, 0), (0, 0.1), (0.1, 0.1)]), 1.0, mode="pointwise")
    <LINESTRING (0 0, 0 0, 0 0)>
    >>> shapely.set_precision(LineString([(0, 0), (0, 0.1), (0.1, 0.1)]), 1.0, mode="keep_collapsed")
    <LINESTRING (0 0, 0 0)>
    >>> shapely.set_precision(None, 1.0) is None
    True

    """  # noqa: E501
    if isinstance(mode, str):
        mode = SetPrecisionMode.get_value(mode)
    elif not np.isscalar(mode):
        raise TypeError("mode only accepts scalar values")
    if mode == SetPrecisionMode.pointwise and geos_version < (3, 10, 0):
        warnings.warn(
            "'pointwise' is only supported for GEOS 3.10",
            UserWarning,
            stacklevel=2,
        )
    return lib.set_precision(geometry, grid_size, np.intc(mode), **kwargs)


@multithreading_enabled
def force_2d(geometry, **kwargs):
    """Force the dimensionality of a geometry to 2D.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to force to 2D.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString, Point, Polygon, from_wkt
    >>> shapely.force_2d(Point(0, 0, 1))
    <POINT (0 0)>
    >>> shapely.force_2d(Point(0, 0))
    <POINT (0 0)>
    >>> shapely.force_2d(LineString([(0, 0, 0), (0, 1, 1), (1, 1, 2)]))
    <LINESTRING (0 0, 0 1, 1 1)>
    >>> shapely.force_2d(from_wkt("POLYGON Z EMPTY"))
    <POLYGON EMPTY>
    >>> shapely.force_2d(None) is None
    True

    """
    return lib.force_2d(geometry, **kwargs)


@multithreading_enabled
def force_3d(geometry, z=0.0, **kwargs):
    """Force the dimensionality of a geometry to 3D.

    2D geometries will get the provided Z coordinate; Z coordinates of 3D geometries
    are unchanged (unless they are nan).

    Note that for empty geometries, 3D is only supported since GEOS 3.9 and then
    still only for simple geometries (non-collections).

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to force to 3D.
    z : float or array_like, default 0.0
        The Z coordinate value to set on the geometry.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString, Point
    >>> shapely.force_3d(Point(0, 0), z=3)
    <POINT Z (0 0 3)>
    >>> shapely.force_3d(Point(0, 0, 0), z=3)
    <POINT Z (0 0 0)>
    >>> shapely.force_3d(LineString([(0, 0), (0, 1), (1, 1)]))
    <LINESTRING Z (0 0 0, 0 1 0, 1 1 0)>
    >>> shapely.force_3d(None) is None
    True

    """
    if np.isnan(z).any():
        raise ValueError("It is not allowed to set the Z coordinate to NaN.")
    return lib.force_3d(geometry, z, **kwargs)
