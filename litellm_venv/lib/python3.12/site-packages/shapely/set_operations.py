"""Set-theoretic operations on geometry objects."""

import warnings

import numpy as np

from shapely import Geometry, GeometryType, lib
from shapely.decorators import (
    deprecate_positional,
    multithreading_enabled,
    requires_geos,
)

__all__ = [
    "coverage_union",
    "coverage_union_all",
    "difference",
    "disjoint_subset_union",
    "disjoint_subset_union_all",
    "intersection",
    "intersection_all",
    "symmetric_difference",
    "symmetric_difference_all",
    "unary_union",
    "union",
    "union_all",
]


# Note: future plan is to change this signature over a few releases:
# shapely 2.0:
#   difference(a, b, grid_size=None, **kwargs)
# shapely 2.1: shows deprecation warning about positional 'grid_size' arg
#   same signature as 2.0
# shapely 2.2(?): enforce keyword-only arguments after 'b'
#   difference(a, b, *, grid_size=None, **kwargs)


@deprecate_positional(["grid_size"], category=DeprecationWarning)
@multithreading_enabled
def difference(a, b, grid_size=None, **kwargs):
    """Return the part of geometry A that does not intersect with geometry B.

    If grid_size is nonzero, input coordinates will be snapped to a precision
    grid of that size and resulting coordinates will be snapped to that same
    grid.  If 0, this operation will use double precision coordinates.  If None,
    the highest precision of the inputs will be used, which may be previously
    set using set_precision.  Note: returned geometry does not have precision
    set unless specified previously by set_precision.

    Parameters
    ----------
    a : Geometry or array_like
        Geometry or geometries to subtract b from.
    b : Geometry or array_like
        Geometry or geometries to subtract from a.
    grid_size : float, optional
        Precision grid size; will use the highest precision of the inputs by default.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Notes
    -----

    .. deprecated:: 2.1.0
        A deprecation warning is shown if ``grid_size`` is specified as a
        positional argument. This will need to be specified as a keyword
        argument in a future release.

    See Also
    --------
    set_precision

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString
    >>> line = LineString([(0, 0), (2, 2)])
    >>> shapely.difference(line, LineString([(1, 1), (3, 3)]))
    <LINESTRING (0 0, 1 1)>
    >>> shapely.difference(line, LineString())
    <LINESTRING (0 0, 2 2)>
    >>> shapely.difference(line, None) is None
    True
    >>> box1 = shapely.box(0, 0, 2, 2)
    >>> box2 = shapely.box(1, 1, 3, 3)
    >>> shapely.difference(box1, box2).normalize()
    <POLYGON ((0 0, 0 2, 1 2, 1 1, 2 1, 2 0, 0 0))>
    >>> box1 = shapely.box(0.1, 0.2, 2.1, 2.1)
    >>> shapely.difference(box1, box2, grid_size=1)
    <POLYGON ((2 0, 0 0, 0 2, 1 2, 1 1, 2 1, 2 0))>

    """
    if grid_size is not None:
        if not np.isscalar(grid_size):
            raise ValueError("grid_size parameter only accepts scalar values")

        return lib.difference_prec(a, b, grid_size, **kwargs)

    return lib.difference(a, b, **kwargs)


# Note: future plan is to change this signature over a few releases:
# shapely 2.0:
#   intersection(a, b, grid_size=None, **kwargs)
# shapely 2.1: shows deprecation warning about positional 'grid_size' arg
#   same signature as 2.0
# shapely 2.2(?): enforce keyword-only arguments after 'b'
#   intersection(a, b, *, grid_size=None, **kwargs)


@deprecate_positional(["grid_size"], category=DeprecationWarning)
@multithreading_enabled
def intersection(a, b, grid_size=None, **kwargs):
    """Return the geometry that is shared between input geometries.

    If grid_size is nonzero, input coordinates will be snapped to a precision
    grid of that size and resulting coordinates will be snapped to that same
    grid.  If 0, this operation will use double precision coordinates.  If None,
    the highest precision of the inputs will be used, which may be previously
    set using set_precision.  Note: returned geometry does not have precision
    set unless specified previously by set_precision.

    Parameters
    ----------
    a, b : Geometry or array_like
        Geometry or geometries to intersect with.
    grid_size : float, optional
        Precision grid size; will use the highest precision of the inputs by default.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Notes
    -----

    .. deprecated:: 2.1.0
        A deprecation warning is shown if ``grid_size`` is specified as a
        positional argument. This will need to be specified as a keyword
        argument in a future release.

    See Also
    --------
    intersection_all
    set_precision

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString
    >>> line = LineString([(0, 0), (2, 2)])
    >>> shapely.intersection(line, LineString([(1, 1), (3, 3)]))
    <LINESTRING (1 1, 2 2)>
    >>> box1 = shapely.box(0, 0, 2, 2)
    >>> box2 = shapely.box(1, 1, 3, 3)
    >>> shapely.intersection(box1, box2).normalize()
    <POLYGON ((1 1, 1 2, 2 2, 2 1, 1 1))>
    >>> box1 = shapely.box(0.1, 0.2, 2.1, 2.1)
    >>> shapely.intersection(box1, box2, grid_size=1)
    <POLYGON ((2 2, 2 1, 1 1, 1 2, 2 2))>

    """
    if grid_size is not None:
        if not np.isscalar(grid_size):
            raise ValueError("grid_size parameter only accepts scalar values")

        return lib.intersection_prec(a, b, grid_size, **kwargs)

    return lib.intersection(a, b, **kwargs)


# Note: future plan is to change this signature over a few releases:
# shapely 2.0:
#   intersection_all(geometries, axis=None, **kwargs)
# shapely 2.1: shows deprecation warning about positional 'axis' arg
#   same signature as 2.0
# shapely 2.2(?): enforce keyword-only arguments after 'geometries'
#   intersection_all(geometries, *, axis=None, **kwargs)


@deprecate_positional(["axis"], category=DeprecationWarning)
@multithreading_enabled
def intersection_all(geometries, axis=None, **kwargs):
    """Return the intersection of multiple geometries.

    This function ignores None values when other Geometry elements are present.
    If all elements of the given axis are None, an empty GeometryCollection is
    returned.

    Parameters
    ----------
    geometries : array_like
        Geometries to calculate the intersection of.
    axis : int, optional
        Axis along which the operation is performed. The default (None)
        performs the operation over all axes, returning a scalar value.
        Axis may be negative, in which case it counts from the last to the
        first axis.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Notes
    -----

    .. deprecated:: 2.1.0
        A deprecation warning is shown if ``axis`` is specified as a
        positional argument. This will need to be specified as a keyword
        argument in a future release.

    See Also
    --------
    intersection

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString
    >>> line1 = LineString([(0, 0), (2, 2)])
    >>> line2 = LineString([(1, 1), (3, 3)])
    >>> shapely.intersection_all([line1, line2])
    <LINESTRING (1 1, 2 2)>
    >>> shapely.intersection_all([[line1, line2, None]], axis=1).tolist()
    [<LINESTRING (1 1, 2 2)>]
    >>> shapely.intersection_all([line1, None])
    <LINESTRING (0 0, 2 2)>

    """
    geometries = np.asarray(geometries)
    if axis is None:
        geometries = geometries.ravel()
    else:
        geometries = np.rollaxis(geometries, axis=axis, start=geometries.ndim)

    return lib.intersection_all(geometries, **kwargs)


# Note: future plan is to change this signature over a few releases:
# shapely 2.0:
#   symmetric_difference(a, b, grid_size=None, **kwargs)
# shapely 2.1: shows deprecation warning about positional 'grid_size' arg
#   same signature as 2.0
# shapely 2.2(?): enforce keyword-only arguments after 'b'
#   symmetric_difference(a, b, *, grid_size=None, **kwargs)


@deprecate_positional(["grid_size"], category=DeprecationWarning)
@multithreading_enabled
def symmetric_difference(a, b, grid_size=None, **kwargs):
    """Return the geometry with the portions of input geometries that do not intersect.

    If grid_size is nonzero, input coordinates will be snapped to a precision
    grid of that size and resulting coordinates will be snapped to that same
    grid.  If 0, this operation will use double precision coordinates.  If None,
    the highest precision of the inputs will be used, which may be previously
    set using set_precision.  Note: returned geometry does not have precision
    set unless specified previously by set_precision.

    Parameters
    ----------
    a, b : Geometry or array_like
        Geometry or geometries to evaluate symmetric difference with.
    grid_size : float, optional
        Precision grid size; will use the highest precision of the inputs by default.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Notes
    -----

    .. deprecated:: 2.1.0
        A deprecation warning is shown if ``grid_size`` is specified as a
        positional argument. This will need to be specified as a keyword
        argument in a future release.

    See Also
    --------
    symmetric_difference_all
    set_precision

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString
    >>> line = LineString([(0, 0), (2, 2)])
    >>> shapely.symmetric_difference(line, LineString([(1, 1), (3, 3)]))
    <MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))>
    >>> box1 = shapely.box(0, 0, 2, 2)
    >>> box2 = shapely.box(1, 1, 3, 3)
    >>> shapely.symmetric_difference(box1, box2).normalize()
    <MULTIPOLYGON (((1 2, 1 3, 3 3, 3 1, 2 1, 2 2, 1 2)), ((0 0, 0 2, 1 2, 1 1, ...>
    >>> box1 = shapely.box(0.1, 0.2, 2.1, 2.1)
    >>> shapely.symmetric_difference(box1, box2, grid_size=1)
    <MULTIPOLYGON (((2 0, 0 0, 0 2, 1 2, 1 1, 2 1, 2 0)), ((2 2, 1 2, 1 3, 3 3, ...>

    """
    if grid_size is not None:
        if not np.isscalar(grid_size):
            raise ValueError("grid_size parameter only accepts scalar values")

        return lib.symmetric_difference_prec(a, b, grid_size, **kwargs)

    return lib.symmetric_difference(a, b, **kwargs)


# Note: future plan is to change this signature over a few releases:
# shapely 2.0:
#   symmetric_difference_all(geometries, axis=None, **kwargs)
# shapely 2.1: shows deprecation warning about positional 'axis' arg
#   same signature as 2.0
# shapely 2.2(?): enforce keyword-only arguments after 'geometries'
#   symmetric_difference_all(geometries, *, axis=None, **kwargs)


@deprecate_positional(["axis"], category=DeprecationWarning)
@multithreading_enabled
def symmetric_difference_all(geometries, axis=None, **kwargs):
    """Return the symmetric difference of multiple geometries.

    This function ignores None values when other Geometry elements are present.
    If all elements of the given axis are None an empty GeometryCollection is
    returned.

    .. deprecated:: 2.1.0

        This function behaves incorrectly and will be removed in a future
        version. See https://github.com/shapely/shapely/issues/2027 for more
        details.

    Parameters
    ----------
    geometries : array_like
        Geometries to calculate the combined symmetric difference of.
    axis : int, optional
        Axis along which the operation is performed. The default (None)
        performs the operation over all axes, returning a scalar value.
        Axis may be negative, in which case it counts from the last to the
        first axis.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Notes
    -----

    .. deprecated:: 2.1.0
        A deprecation warning is shown if ``axis`` is specified as a
        positional argument. This will need to be specified as a keyword
        argument in a future release.

    See Also
    --------
    symmetric_difference

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString
    >>> line1 = LineString([(0, 0), (2, 2)])
    >>> line2 = LineString([(1, 1), (3, 3)])
    >>> shapely.symmetric_difference_all([line1, line2])
    <MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))>
    >>> shapely.symmetric_difference_all([[line1, line2, None]], axis=1).tolist()
    [<MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))>]
    >>> shapely.symmetric_difference_all([line1, None])
    <LINESTRING (0 0, 2 2)>
    >>> shapely.symmetric_difference_all([None, None])
    <GEOMETRYCOLLECTION EMPTY>

    """
    warnings.warn(
        "The symmetric_difference_all function behaves incorrectly and will be "
        "removed in a future version. "
        "See https://github.com/shapely/shapely/issues/2027 for more details.",
        DeprecationWarning,
        stacklevel=2,
    )
    geometries = np.asarray(geometries)
    if axis is None:
        geometries = geometries.ravel()
    else:
        geometries = np.rollaxis(geometries, axis=axis, start=geometries.ndim)

    return lib.symmetric_difference_all(geometries, **kwargs)


# Note: future plan is to change this signature over a few releases:
# shapely 2.0:
#   union(a, b, grid_size=None, **kwargs)
# shapely 2.1: shows deprecation warning about positional 'grid_size' arg
#   same signature as 2.0
# shapely 2.2(?): enforce keyword-only arguments after 'b'
#   union(a, b, *, grid_size=None, **kwargs)


@deprecate_positional(["grid_size"], category=DeprecationWarning)
@multithreading_enabled
def union(a, b, grid_size=None, **kwargs):
    """Merge geometries into one.

    If grid_size is nonzero, input coordinates will be snapped to a precision
    grid of that size and resulting coordinates will be snapped to that same
    grid.  If 0, this operation will use double precision coordinates.  If None,
    the highest precision of the inputs will be used, which may be previously
    set using set_precision.  Note: returned geometry does not have precision
    set unless specified previously by set_precision.

    Parameters
    ----------
    a, b : Geometry or array_like
        Geometry or geometries to merge (union).
    grid_size : float, optional
        Precision grid size; will use the highest precision of the inputs by default.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Notes
    -----

    .. deprecated:: 2.1.0
        A deprecation warning is shown if ``grid_size`` is specified as a
        positional argument. This will need to be specified as a keyword
        argument in a future release.

    See Also
    --------
    union_all
    set_precision

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString
    >>> line = LineString([(0, 0), (2, 2)])
    >>> shapely.union(line, LineString([(2, 2), (3, 3)]))
    <MULTILINESTRING ((0 0, 2 2), (2 2, 3 3))>
    >>> shapely.union(line, None) is None
    True
    >>> box1 = shapely.box(0, 0, 2, 2)
    >>> box2 = shapely.box(1, 1, 3, 3)
    >>> shapely.union(box1, box2).normalize()
    <POLYGON ((0 0, 0 2, 1 2, 1 3, 3 3, 3 1, 2 1, 2 0, 0 0))>
    >>> box1 = shapely.box(0.1, 0.2, 2.1, 2.1)
    >>> shapely.union(box1, box2, grid_size=1)
    <POLYGON ((2 0, 0 0, 0 2, 1 2, 1 3, 3 3, 3 1, 2 1, 2 0))>

    """
    if grid_size is not None:
        if not np.isscalar(grid_size):
            raise ValueError("grid_size parameter only accepts scalar values")

        return lib.union_prec(a, b, grid_size, **kwargs)

    return lib.union(a, b, **kwargs)


# Note: future plan is to change this signature over a few releases:
# shapely 2.0:
#   union_all(geometries, grid_size=None, axis=None, **kwargs)
# shapely 2.1: shows deprecation warning about positional 'grid_size' arg
#   same signature as 2.0
# shapely 2.2(?): enforce keyword-only arguments after 'geometries'
#   union_all(geometries, *, grid_size=None, axis=None, **kwargs)


@deprecate_positional(["grid_size", "axis"], category=DeprecationWarning)
@multithreading_enabled
def union_all(geometries, grid_size=None, axis=None, **kwargs):
    """Return the union of multiple geometries.

    This function ignores None values when other Geometry elements are present.
    If all elements of the given axis are None an empty GeometryCollection is
    returned.

    If grid_size is nonzero, input coordinates will be snapped to a precision
    grid of that size and resulting coordinates will be snapped to that same
    grid.  If 0, this operation will use double precision coordinates.  If None,
    the highest precision of the inputs will be used, which may be previously
    set using set_precision.  Note: returned geometry does not have precision
    set unless specified previously by set_precision.

    `unary_union` is an alias of `union_all`.

    Parameters
    ----------
    geometries : array_like
        Geometries to merge/union.
    grid_size : float, optional
        Precision grid size; will use the highest precision of the inputs by default.
    axis : int, optional
        Axis along which the operation is performed. The default (None)
        performs the operation over all axes, returning a scalar value.
        Axis may be negative, in which case it counts from the last to the
        first axis.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Notes
    -----

    .. deprecated:: 2.1.0
        A deprecation warning is shown if ``grid_size`` or ``axis`` are
        specified as positional arguments. In a future release, these will
        need to be specified as keyword arguments.

    See Also
    --------
    union
    set_precision

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString, Point
    >>> line1 = LineString([(0, 0), (2, 2)])
    >>> line2 = LineString([(2, 2), (3, 3)])
    >>> shapely.union_all([line1, line2])
    <MULTILINESTRING ((0 0, 2 2), (2 2, 3 3))>
    >>> shapely.union_all([[line1, line2, None]], axis=1).tolist()
    [<MULTILINESTRING ((0 0, 2 2), (2 2, 3 3))>]
    >>> box1 = shapely.box(0, 0, 2, 2)
    >>> box2 = shapely.box(1, 1, 3, 3)
    >>> shapely.union_all([box1, box2]).normalize()
    <POLYGON ((0 0, 0 2, 1 2, 1 3, 3 3, 3 1, 2 1, 2 0, 0 0))>
    >>> box1 = shapely.box(0.1, 0.2, 2.1, 2.1)
    >>> shapely.union_all([box1, box2], grid_size=1)
    <POLYGON ((2 0, 0 0, 0 2, 1 2, 1 3, 3 3, 3 1, 2 1, 2 0))>
    >>> shapely.union_all([None, Point(0, 1)])
    <POINT (0 1)>
    >>> shapely.union_all([None, None])
    <GEOMETRYCOLLECTION EMPTY>
    >>> shapely.union_all([])
    <GEOMETRYCOLLECTION EMPTY>

    """
    # for union_all, GEOS provides an efficient route through first creating
    # GeometryCollections
    # first roll the aggregation axis backwards
    geometries = np.asarray(geometries)
    if axis is None:
        geometries = geometries.ravel()
    else:
        geometries = np.rollaxis(geometries, axis=axis, start=geometries.ndim)

    # create_collection acts on the inner axis
    collections = lib.create_collection(
        geometries, np.intc(GeometryType.GEOMETRYCOLLECTION)
    )

    if grid_size is not None:
        if not np.isscalar(grid_size):
            raise ValueError("grid_size parameter only accepts scalar values")

        return lib.unary_union_prec(collections, grid_size, **kwargs)

    return lib.unary_union(collections, **kwargs)


unary_union = union_all


@multithreading_enabled
def coverage_union(a, b, **kwargs):
    """Merge multiple polygons into one.

    This is an optimized version of union which assumes the polygons to be
    non-overlapping.

    Parameters
    ----------
    a, b : Geometry or array_like
        Geometry or geometries to merge (union).
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See Also
    --------
    coverage_union_all

    Examples
    --------
    >>> import shapely
    >>> from shapely import Polygon
    >>> polygon_1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    >>> polygon_2 = Polygon([(1, 0), (1, 1), (2, 1), (2, 0), (1, 0)])
    >>> shapely.coverage_union(polygon_1, polygon_2).normalize()
    <POLYGON ((0 0, 0 1, 1 1, 2 1, 2 0, 1 0, 0 0))>

    Union with None returns same polygon

    >>> shapely.coverage_union(polygon_1, None).normalize()
    <POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))>

    """
    return coverage_union_all([a, b], **kwargs)


# Note: future plan is to change this signature over a few releases:
# shapely 2.0:
#   coverage_union_all(geometries, axis=None, **kwargs)
# shapely 2.1: shows deprecation warning about positional 'axis' arg
#   same signature as 2.0
# shapely 2.2(?): enforce keyword-only arguments after 'geometries'
#   coverage_union_all(geometries, *, axis=None, **kwargs)


@deprecate_positional(["axis"], category=DeprecationWarning)
@multithreading_enabled
def coverage_union_all(geometries, axis=None, **kwargs):
    """Return the union of multiple polygons of a geometry collection.

    This is an optimized version of union which assumes the polygons
    to be non-overlapping.

    This function ignores None values when other Geometry elements are present.
    If all elements of the given axis are None, an empty GeometryCollection is
    returned (before GEOS 3.12 this was an empty MultiPolygon).

    Parameters
    ----------
    geometries : array_like
        Geometries to merge/union.
    axis : int, optional
        Axis along which the operation is performed. The default (None)
        performs the operation over all axes, returning a scalar value.
        Axis may be negative, in which case it counts from the last to the
        first axis.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Notes
    -----

    .. deprecated:: 2.1.0
        A deprecation warning is shown if ``axis`` is specified as a
        positional argument. This will need to be specified as a keyword
        argument in a future release.

    See Also
    --------
    coverage_union

    Examples
    --------
    >>> import shapely
    >>> from shapely import Polygon
    >>> polygon_1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    >>> polygon_2 = Polygon([(1, 0), (1, 1), (2, 1), (2, 0), (1, 0)])
    >>> shapely.coverage_union_all([polygon_1, polygon_2]).normalize()
    <POLYGON ((0 0, 0 1, 1 1, 2 1, 2 0, 1 0, 0 0))>
    >>> shapely.coverage_union_all([polygon_1, None]).normalize()
    <POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))>
    >>> shapely.coverage_union_all([None, None]).normalize()
    <GEOMETRYCOLLECTION EMPTY>

    """
    # coverage union in GEOS works over GeometryCollections
    # first roll the aggregation axis backwards
    geometries = np.asarray(geometries)
    if axis is None:
        geometries = geometries.ravel()
    else:
        geometries = np.rollaxis(
            np.asarray(geometries), axis=axis, start=geometries.ndim
        )
    # create_collection acts on the inner axis
    collections = lib.create_collection(
        geometries, np.intc(GeometryType.GEOMETRYCOLLECTION)
    )
    return lib.coverage_union(collections, **kwargs)


@requires_geos("3.12.0")
@multithreading_enabled
def disjoint_subset_union(a, b, **kwargs):
    """Merge multiple polygons into one using algorithm optimised for subsets.

    This is an optimized version of union which assumes inputs can be
    divided into subsets that do not intersect.

    If there is only one such subset, performance can be expected to be worse than
    :func:`union`. As such, it is recommeded to use ``disjoint_subset_union`` with
    GeometryCollections rather than individual geometries.

    .. versionadded:: 2.1.0

    Parameters
    ----------
    a, b : Geometry or array_like
        Geometry or geometries to merge (union).
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See Also
    --------
    union
    coverage_union
    disjoint_subset_union_all

    Examples
    --------
    >>> import shapely
    >>> from shapely import Polygon
    >>> polygon_1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    >>> polygon_2 = Polygon([(1, 0), (1, 1), (2, 1), (2, 0), (1, 0)])
    >>> shapely.disjoint_subset_union(polygon_1, polygon_2).normalize()
    <POLYGON ((0 0, 0 1, 1 1, 2 1, 2 0, 1 0, 0 0))>

    Union with None returns same polygon:

    >>> shapely.disjoint_subset_union(polygon_1, None).normalize()
    <POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))>
    """
    if (isinstance(a, Geometry) or a is None) and (
        isinstance(b, Geometry) or b is None
    ):
        pass
    elif isinstance(a, Geometry) or a is None:
        a = np.full_like(b, a)
    elif isinstance(b, Geometry) or b is None:
        b = np.full_like(a, b)
    elif len(a) != len(b):
        raise ValueError("Arrays a and b must have the same length")
    return disjoint_subset_union_all([a, b], axis=0, **kwargs)


@requires_geos("3.12.0")
@multithreading_enabled
def disjoint_subset_union_all(geometries, *, axis=None, **kwargs):
    """Return the union of multiple polygons.

    This is an optimized version of union which assumes inputs can be divided into
    subsets that do not intersect.

    If there is only one such subset, performance can be expected to be worse than
    :func:`union_all`.

    This function ignores None values when other Geometry elements are present.
    If all elements of the given axis are None, an empty GeometryCollection is
    returned.

    .. versionadded:: 2.1.0

    Parameters
    ----------
    geometries : array_like
        Geometries to union.
    axis : int, optional
        Axis along which the operation is performed. The default (None)
        performs the operation over all axes, returning a scalar value.
        Axis may be negative, in which case it counts from the last to the
        first axis.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See Also
    --------
    coverage_union_all
    union_all
    disjoint_subset_union

    Examples
    --------
    >>> import shapely
    >>> from shapely import Polygon
    >>> polygon_1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    >>> polygon_2 = Polygon([(1, 0), (1, 1), (2, 1), (2, 0), (1, 0)])
    >>> shapely.disjoint_subset_union_all([polygon_1, polygon_2]).normalize()
    <POLYGON ((0 0, 0 1, 1 1, 2 1, 2 0, 1 0, 0 0))>
    >>> shapely.disjoint_subset_union_all([polygon_1, None]).normalize()
    <POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))>
    >>> shapely.disjoint_subset_union_all([None, None]).normalize()
    <GEOMETRYCOLLECTION EMPTY>
    """
    geometries = np.asarray(geometries)
    if axis is None:
        geometries = geometries.ravel()
    else:
        geometries = np.rollaxis(
            np.asarray(geometries), axis=axis, start=geometries.ndim
        )
    # create_collection acts on the inner axis
    collections = lib.create_collection(
        geometries, np.intc(GeometryType.GEOMETRYCOLLECTION)
    )

    return lib.disjoint_subset_union(collections, **kwargs)
