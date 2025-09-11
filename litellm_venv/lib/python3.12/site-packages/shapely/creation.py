"""Methods to create geometries."""

import numpy as np

from shapely import Geometry, GeometryType, lib
from shapely._enum import ParamEnum
from shapely._geometry_helpers import collections_1d, simple_geometries_1d
from shapely.decorators import deprecate_positional, multithreading_enabled
from shapely.io import from_wkt

__all__ = [
    "box",
    "destroy_prepared",
    "empty",
    "geometrycollections",
    "linearrings",
    "linestrings",
    "multilinestrings",
    "multipoints",
    "multipolygons",
    "points",
    "polygons",
    "prepare",
]


class HandleNaN(ParamEnum):
    allow = 0
    skip = 1
    error = 2


def _xyz_to_coords(x, y, z):
    if y is None:
        return x
    if z is None:
        coords = np.broadcast_arrays(x, y)
    else:
        coords = np.broadcast_arrays(x, y, z)
    return np.stack(coords, axis=-1)


# Note: future plan is to change this signature over a few releases:
# shapely 2.0:
#   points(coords, y=None, z=None, indices=None, out=None, **kwargs)
# shapely 2.1: shows deprecation warning about positional 'indices' arg
#   points(coords, y=None, z=None, indices=None, *, handle_nan=HandleNaN.allow, out=None, **kwargs)  # noqa: E501
# shapely 2.2(?): enforce keyword-only arguments after 'z'
#   points(coords, y=None, z=None, *, indices=None, handle_nan=HandleNaN.allow, out=None, **kwargs)  # noqa: E501


@deprecate_positional(["indices"], category=DeprecationWarning)
@multithreading_enabled
def points(
    coords,
    y=None,
    z=None,
    indices=None,
    *,
    handle_nan=HandleNaN.allow,
    out=None,
    **kwargs,
):
    """Create an array of points.

    Parameters
    ----------
    coords : array_like
        An array of coordinate tuples (2- or 3-dimensional) or, if ``y`` is
        provided, an array of x coordinates.
    y : array_like, optional
        An array of y coordinates.
    z : array_like, optional
        An array of z coordinates.
    indices : array_like, optional
        Indices into the target array where input coordinates belong. If
        provided, the coords should be 2D with shape (N, 2) or (N, 3) and
        indices should be an array of shape (N,) with integers in increasing
        order. Missing indices result in a ValueError unless ``out`` is
        provided, in which case the original value in ``out`` is kept.
    handle_nan : shapely.HandleNaN or {'allow', 'skip', 'error'}, default 'allow'
        Specifies what to do when a NaN or Inf is encountered in the coordinates:

        - 'allow': the geometries are created with NaN or Inf coordinates.
          Note that this can result in unexpected behaviour in subsequent
          operations, and generally it is discouraged to have non-finite
          coordinate values. One can use this option if you know all
          coordinates are finite and want to avoid the overhead of checking
          for this.
        - 'skip': if any of x, y or z values are NaN or Inf, an empty point
          will be created.
        - 'error': if any NaN or Inf is detected in the coordinates, a ValueError
          is raised. This option ensures that the created geometries have all
          finite coordinate values.

        .. versionadded:: 2.1.0
    out : ndarray, optional
        An array (with dtype object) to output the geometries into.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.
        Ignored if ``indices`` is provided.

    Examples
    --------
    >>> import shapely
    >>> shapely.points([[0, 1], [4, 5]]).tolist()
    [<POINT (0 1)>, <POINT (4 5)>]
    >>> shapely.points([0, 1, 2])
    <POINT Z (0 1 2)>

    Notes
    -----
    - GEOS 3.10, 3.11 and 3.12 automatically converts POINT (nan nan) to POINT EMPTY.
    - GEOS 3.10 and 3.11 will transform a 3D point to 2D if its Z coordinate is NaN.
    - Usage of the ``y`` and ``z`` arguments will prevents lazy evaluation in
      ``dask``. Instead provide the coordinates as an array with shape
      ``(..., 2)`` or ``(..., 3)`` using only the ``coords`` argument.

    """
    coords = _xyz_to_coords(coords, y, z)
    if isinstance(handle_nan, str):
        handle_nan = HandleNaN.get_value(handle_nan)
    if indices is None:
        return lib.points(coords, np.intc(handle_nan), out=out, **kwargs)
    else:
        return simple_geometries_1d(
            coords, indices, GeometryType.POINT, handle_nan=handle_nan, out=out
        )


# Note: future plan is to change this signature over a few releases:
# shapely 2.0:
#   linestrings(coords, y=None, z=None, indices=None, out=None, **kwargs)
# shapely 2.1: shows deprecation warning about positional 'indices' arg
#   linestrings(coords, y=None, z=None, indices=None, *, handle_nan=HandleNaN.allow, out=None, **kwargs)  # noqa: E501
# shapely 2.2(?): enforce keyword-only arguments after 'z'
#   linestrings(coords, y=None, z=None, *, indices=None, handle_nan=HandleNaN.allow, out=None, **kwargs)  # noqa: E501


@deprecate_positional(["indices"], category=DeprecationWarning)
@multithreading_enabled
def linestrings(
    coords,
    y=None,
    z=None,
    indices=None,
    *,
    handle_nan=HandleNaN.allow,
    out=None,
    **kwargs,
):
    """Create an array of linestrings.

    This function will raise an exception if a linestring contains less than
    two points.

    Parameters
    ----------
    coords : array_like
        An array of lists of coordinate tuples (2- or 3-dimensional) or, if ``y``
        is provided, an array of lists of x coordinates.
    y : array_like, optional
        An array of y coordinates.
    z : array_like, optional
        An array of z coordinates.
    indices : array_like, optional
        Indices into the target array where input coordinates belong. If
        provided, the coords should be 2D with shape (N, 2) or (N, 3) and
        indices should be an array of shape (N,) with integers in increasing
        order. Missing indices result in a ValueError unless ``out`` is
        provided, in which case the original value in ``out`` is kept.
    handle_nan : shapely.HandleNaN or {'allow', 'skip', 'error'}, default 'allow'
        Specifies what to do when a NaN or Inf is encountered in the coordinates:

        - 'allow': the geometries are created with NaN or Inf coordinates.
          Note that this can result in unexpected behaviour in subsequent
          operations, and generally it is discouraged to have non-finite
          coordinate values. One can use this option if you know all
          coordinates are finite and want to avoid the overhead of checking
          for this.
        - 'skip': the coordinate pairs where any of x, y or z values are
          NaN or Inf are ignored. If this results in ignoring all coordinates
          for one geometry, an empty geometry is created.
        - 'error': if any NaN or Inf is detected in the coordinates, a ValueError
          is raised. This option ensures that the created geometries have all
          finite coordinate values.

        .. versionadded:: 2.1.0

    out : ndarray, optional
        An array (with dtype object) to output the geometries into.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.
        Ignored if ``indices`` is provided.

    Examples
    --------
    >>> import shapely
    >>> shapely.linestrings([[[0, 1], [4, 5]], [[2, 3], [5, 6]]]).tolist()
    [<LINESTRING (0 1, 4 5)>, <LINESTRING (2 3, 5 6)>]
    >>> shapely.linestrings(
    ...     [[0, 1], [4, 5], [2, 3], [5, 6], [7, 8]],
    ...     indices=[0, 0, 1, 1, 1]
    ... ).tolist()
    [<LINESTRING (0 1, 4 5)>, <LINESTRING (2 3, 5 6, 7 8)>]

    Notes
    -----
    - Usage of the ``y`` and ``z`` arguments will prevents lazy evaluation in
      ``dask``. Instead provide the coordinates as a ``(..., 2)`` or
      ``(..., 3)`` array using only ``coords``.

    """
    coords = _xyz_to_coords(coords, y, z)
    if isinstance(handle_nan, str):
        handle_nan = HandleNaN.get_value(handle_nan)
    if indices is None:
        return lib.linestrings(coords, np.intc(handle_nan), out=out, **kwargs)
    else:
        return simple_geometries_1d(
            coords, indices, GeometryType.LINESTRING, handle_nan=handle_nan, out=out
        )


# Note: future plan is to change this signature over a few releases:
# shapely 2.0:
#   linearrings(coords, y=None, z=None, indices=None, out=None, **kwargs)
# shapely 2.1: shows deprecation warning about positional 'indices' arg
#   linearrings(coords, y=None, z=None, indices=None, *, handle_nan=HandleNaN.allow, out=None, **kwargs)  # noqa: E501
# shapely 2.2(?): enforce keyword-only arguments after 'z'
#   linearrings(coords, y=None, z=None, *, indices=None, handle_nan=HandleNaN.allow, out=None, **kwargs)  # noqa: E501


@deprecate_positional(["indices"], category=DeprecationWarning)
@multithreading_enabled
def linearrings(
    coords,
    y=None,
    z=None,
    indices=None,
    *,
    handle_nan=HandleNaN.allow,
    out=None,
    **kwargs,
):
    """Create an array of linearrings.

    If the provided coords do not constitute a closed linestring, or if there
    are only 3 provided coords, the first
    coordinate is duplicated at the end to close the ring. This function will
    raise an exception if a linearring contains less than three points or if
    the terminal coordinates contain NaN (not-a-number).

    Parameters
    ----------
    coords : array_like
        An array of lists of coordinate tuples (2- or 3-dimensional) or, if ``y``
        is provided, an array of lists of x coordinates
    y : array_like, optional
        An array of y coordinates.
    z : array_like, optional
        An array of z coordinates.
    indices : array_like, optional
        Indices into the target array where input coordinates belong. If
        provided, the coords should be 2D with shape (N, 2) or (N, 3) and
        indices should be an array of shape (N,) with integers in increasing
        order. Missing indices result in a ValueError unless ``out`` is
        provided, in which case the original value in ``out`` is kept.
    handle_nan : shapely.HandleNaN or {'allow', 'skip', 'error'}, default 'allow'
        Specifies what to do when a NaN or Inf is encountered in the coordinates:

        - 'allow': the geometries are created with NaN or Inf coordinates.
          Note that this can result in unexpected behaviour in subsequent
          operations, and generally it is discouraged to have non-finite
          coordinate values. One can use this option if you know all
          coordinates are finite and want to avoid the overhead of checking
          for this.
        - 'skip': the coordinate pairs where any of x, y or z values are
          NaN or Inf are ignored. If this results in ignoring all coordinates
          for one geometry, an empty geometry is created.
        - 'error': if any NaN or Inf is detected in the coordinates, a ValueError
          is raised. This option ensures that the created geometries have all
          finite coordinate values.

        .. versionadded:: 2.1.0

    out : ndarray, optional
        An array (with dtype object) to output the geometries into.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.
        Ignored if ``indices`` is provided.

    See Also
    --------
    linestrings

    Examples
    --------
    >>> import shapely
    >>> shapely.linearrings([[0, 0], [0, 1], [1, 1], [0, 0]])
    <LINEARRING (0 0, 0 1, 1 1, 0 0)>
    >>> shapely.linearrings([[0, 0], [0, 1], [1, 1]])
    <LINEARRING (0 0, 0 1, 1 1, 0 0)>

    Notes
    -----
    - Usage of the ``y`` and ``z`` arguments will prevents lazy evaluation in
      ``dask``. Instead provide the coordinates as a ``(..., 2)`` or
      ``(..., 3)`` array using only ``coords``.

    """
    coords = _xyz_to_coords(coords, y, z)
    if isinstance(handle_nan, str):
        handle_nan = HandleNaN.get_value(handle_nan)
    if indices is None:
        return lib.linearrings(coords, np.intc(handle_nan), out=out, **kwargs)
    else:
        return simple_geometries_1d(
            coords, indices, GeometryType.LINEARRING, handle_nan=handle_nan, out=out
        )


# Note: future plan is to change this signature over a few releases:
# shapely 2.0:
#   polygons(geometries, holes=None, indices=None, out=None, **kwargs)
# shapely 2.1: shows deprecation warning about positional 'indices' arg
#   polygons(geometries, holes=None, indices=None, *, out=None, **kwargs)
# shapely 2.2(?): enforce keyword-only arguments after 'holes'
#   polygons(geometries, holes=None, *, indices=None, out=None, **kwargs)


@deprecate_positional(["indices"], category=DeprecationWarning)
@multithreading_enabled
def polygons(geometries, holes=None, indices=None, *, out=None, **kwargs):
    """Create an array of polygons.

    Parameters
    ----------
    geometries : array_like
        An array of linearrings or coordinates (see linearrings).
        Unless ``indices`` are given (see description below), this
        include the outer shells only. The ``holes`` argument should be used
        to create polygons with holes.
    holes : array_like, optional
        An array of lists of linearrings that constitute holes for each shell.
        Not to be used in combination with ``indices``.
    indices : array_like, optional
        Indices into the target array where input geometries belong. If
        provided, the holes are expected to be present inside ``geometries``;
        the first geometry for each index is the outer shell
        and all subsequent geometries in that index are the holes.
        Both geometries and indices should be 1D and have matching sizes.
        Indices should be in increasing order. Missing indices result in a
        ValueError unless ``out`` is  provided, in which case the original value
        in ``out`` is kept.
    out : ndarray, optional
        An array (with dtype object) to output the geometries into.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.
        Ignored if ``indices`` is provided.

    Notes
    -----

    .. deprecated:: 2.1.0
        A deprecation warning is shown if ``indices`` is specified as a
        positional argument. This will need to be specified as a keyword
        argument in a future release.

    Examples
    --------
    >>> import shapely

    Polygons are constructed from rings:

    >>> ring_1 = shapely.linearrings([[0, 0], [0, 10], [10, 10], [10, 0]])
    >>> ring_2 = shapely.linearrings([[2, 6], [2, 7], [3, 7], [3, 6]])
    >>> shapely.polygons([ring_1, ring_2])[0]
    <POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))>
    >>> shapely.polygons([ring_1, ring_2])[1]
    <POLYGON ((2 6, 2 7, 3 7, 3 6, 2 6))>

    Or from coordinates directly:

    >>> shapely.polygons([[0, 0], [0, 10], [10, 10], [10, 0]])
    <POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))>

    Adding holes can be done using the ``holes`` keyword argument:

    >>> shapely.polygons(ring_1, holes=[ring_2])
    <POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0), (2 6, 2 7, 3 7, 3 6, 2 6))>

    Or using the ``indices`` argument:

    >>> shapely.polygons([ring_1, ring_2], indices=[0, 1])[0]
    <POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))>
    >>> shapely.polygons([ring_1, ring_2], indices=[0, 1])[1]
    <POLYGON ((2 6, 2 7, 3 7, 3 6, 2 6))>
    >>> shapely.polygons([ring_1, ring_2], indices=[0, 0])[0]
    <POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0), (2 6, 2 7, 3 7, 3 6, 2 6))>

    Missing input values (``None``) are skipped and may result in an
    empty polygon:

    >>> shapely.polygons(None)
    <POLYGON EMPTY>
    >>> shapely.polygons(ring_1, holes=[None])
    <POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))>
    >>> shapely.polygons([ring_1, None], indices=[0, 0])[0]
    <POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))>

    """
    geometries = np.asarray(geometries)
    if not isinstance(geometries, Geometry) and np.issubdtype(
        geometries.dtype, np.number
    ):
        geometries = linearrings(geometries)

    if indices is not None:
        if holes is not None:
            raise TypeError("Cannot specify separate holes array when using indices.")
        return collections_1d(geometries, indices, GeometryType.POLYGON, out=out)

    if holes is None:
        # no holes provided: initialize an empty holes array matching shells
        shape = geometries.shape + (0,) if isinstance(geometries, np.ndarray) else (0,)
        holes = np.empty(shape, dtype=object)
    else:
        holes = np.asarray(holes)
        # convert holes coordinates into linearrings
        if np.issubdtype(holes.dtype, np.number):
            holes = linearrings(holes)

    return lib.polygons(geometries, holes, out=out, **kwargs)


# Note: future plan is to change this signature over a few releases:
# shapely 2.0:
#   box(xmin, ymin, xmax, ymax, ccw=True, **kwargs)
# shapely 2.1: shows deprecation warning about positional 'ccw' arg
#   same signature as 2.0
# shapely 2.2(?): enforce keyword-only arguments after 'ymax'
#   box(xmin, ymin, xmax, ymax, *, ccw=True, **kwargs)


@deprecate_positional(["ccw"], category=DeprecationWarning)
@multithreading_enabled
def box(xmin, ymin, xmax, ymax, ccw=True, **kwargs):
    """Create box polygons.

    Parameters
    ----------
    xmin : float or array_like
        Float or array of minimum x coordinates.
    ymin : float or array_like
        Float or array of minimum y coordinates.
    xmax : float or array_like
        Float or array of maximum x coordinates.
    ymax : float or array_like
        Float or array of maximum y coordinates.
    ccw : bool, default True
        If True, box will be created in counterclockwise direction starting
        from bottom right coordinate (xmax, ymin).
        If False, box will be created in clockwise direction starting from
        bottom left coordinate (xmin, ymin).
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Notes
    -----

    .. deprecated:: 2.1.0
        A deprecation warning is shown if ``ccw`` is specified as a
        positional argument. This will need to be specified as a keyword
        argument in a future release.

    Examples
    --------
    >>> import shapely
    >>> shapely.box(0, 0, 1, 1)
    <POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))>
    >>> shapely.box(0, 0, 1, 1, ccw=False)
    <POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))>

    """
    return lib.box(xmin, ymin, xmax, ymax, ccw, **kwargs)


# Note: future plan is to change this signature over a few releases:
# shapely 2.0:
#   multipoints(geometries, indices=None, out=None, **kwargs)
# shapely 2.1: shows deprecation warning about positional 'indices' arg
#   multipoints(geometries, indices=None, *, out=None, **kwargs)
# shapely 2.2(?): enforce keyword-only arguments after 'indices'
#   multipoints(geometries, *, indices=None, out=None, **kwargs)


@deprecate_positional(["indices"], category=DeprecationWarning)
@multithreading_enabled
def multipoints(geometries, indices=None, *, out=None, **kwargs):
    """Create multipoints from arrays of points.

    Parameters
    ----------
    geometries : array_like
        An array of points or coordinates (see points).
    indices : array_like, optional
        Indices into the target array where input geometries belong. If
        provided, both geometries and indices should be 1D and have matching
        sizes. Indices should be in increasing order. Missing indices result
        in a ValueError unless ``out`` is  provided, in which case the original
        value in ``out`` is kept.
    out : ndarray, optional
        An array (with dtype object) to output the geometries into.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.
        Ignored if ``indices`` is provided.

    Notes
    -----

    .. deprecated:: 2.1.0
        A deprecation warning is shown if ``indices`` is specified as a
        positional argument. This will need to be specified as a keyword
        argument in a future release.

    Examples
    --------
    >>> import shapely

    Multipoints are constructed from points:

    >>> point_1 = shapely.points([1, 1])
    >>> point_2 = shapely.points([2, 2])
    >>> shapely.multipoints([point_1, point_2])
    <MULTIPOINT ((1 1), (2 2))>
    >>> shapely.multipoints([[point_1, point_2], [point_2, None]]).tolist()
    [<MULTIPOINT ((1 1), (2 2))>, <MULTIPOINT ((2 2))>]

    Or from coordinates directly:

    >>> shapely.multipoints([[0, 0], [2, 2], [3, 3]])
    <MULTIPOINT ((0 0), (2 2), (3 3))>

    Multiple multipoints of different sizes can be constructed efficiently using the
    ``indices`` keyword argument:

    >>> shapely.multipoints([point_1, point_2, point_2], indices=[0, 0, 1]).tolist()
    [<MULTIPOINT ((1 1), (2 2))>, <MULTIPOINT ((2 2))>]

    Missing input values (``None``) are skipped and may result in an
    empty multipoint:

    >>> shapely.multipoints([None])
    <MULTIPOINT EMPTY>
    >>> shapely.multipoints([point_1, None], indices=[0, 0]).tolist()
    [<MULTIPOINT ((1 1))>]
    >>> shapely.multipoints([point_1, None], indices=[0, 1]).tolist()
    [<MULTIPOINT ((1 1))>, <MULTIPOINT EMPTY>]

    """
    typ = GeometryType.MULTIPOINT
    geometries = np.asarray(geometries)
    if not isinstance(geometries, Geometry) and np.issubdtype(
        geometries.dtype, np.number
    ):
        geometries = points(geometries)
    if indices is None:
        return lib.create_collection(geometries, np.intc(typ), out=out, **kwargs)
    else:
        return collections_1d(geometries, indices, typ, out=out)


# Note: future plan is to change this signature over a few releases:
# shapely 2.0:
#   multilinestrings(geometries, indices=None, out=None, **kwargs)
# shapely 2.1: shows deprecation warning about positional 'indices' arg
#   multilinestrings(geometries, indices=None, *, out=None, **kwargs)
# shapely 2.2(?): enforce keyword-only arguments after 'indices'
#   multilinestrings(geometries, *, indices=None, out=None, **kwargs)


@deprecate_positional(["indices"], category=DeprecationWarning)
@multithreading_enabled
def multilinestrings(geometries, indices=None, *, out=None, **kwargs):
    """Create multilinestrings from arrays of linestrings.

    Parameters
    ----------
    geometries : array_like
        An array of linestrings or coordinates (see linestrings).
    indices : array_like, optional
        Indices into the target array where input geometries belong. If
        provided, both geometries and indices should be 1D and have matching
        sizes. Indices should be in increasing order. Missing indices result
        in a ValueError unless ``out`` is  provided, in which case the original
        value in ``out`` is kept.
    out : ndarray, optional
        An array (with dtype object) to output the geometries into.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.
        Ignored if ``indices`` is provided.

    Notes
    -----

    .. deprecated:: 2.1.0
        A deprecation warning is shown if ``indices`` is specified as a
        positional argument. This will need to be specified as a keyword
        argument in a future release.

    See Also
    --------
    multipoints

    """
    typ = GeometryType.MULTILINESTRING
    geometries = np.asarray(geometries)
    if not isinstance(geometries, Geometry) and np.issubdtype(
        geometries.dtype, np.number
    ):
        geometries = linestrings(geometries)

    if indices is None:
        return lib.create_collection(geometries, np.intc(typ), out=out, **kwargs)
    else:
        return collections_1d(geometries, indices, typ, out=out)


# Note: future plan is to change this signature over a few releases:
# shapely 2.0:
#   multipolygons(geometries, indices=None, out=None, **kwargs)
# shapely 2.1: shows deprecation warning about positional 'indices' arg
#   multipolygons(geometries, indices=None, *, out=None, **kwargs)
# shapely 2.2(?): enforce keyword-only arguments after 'indices'
#   multipolygons(geometries, *, indices=None, out=None, **kwargs)


@deprecate_positional(["indices"], category=DeprecationWarning)
@multithreading_enabled
def multipolygons(geometries, indices=None, *, out=None, **kwargs):
    """Create multipolygons from arrays of polygons.

    Parameters
    ----------
    geometries : array_like
        An array of polygons or coordinates (see polygons).
    indices : array_like, optional
        Indices into the target array where input geometries belong. If
        provided, both geometries and indices should be 1D and have matching
        sizes. Indices should be in increasing order. Missing indices result
        in a ValueError unless ``out`` is  provided, in which case the original
        value in ``out`` is kept.
    out : ndarray, optional
        An array (with dtype object) to output the geometries into.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.
        Ignored if ``indices`` is provided.

    Notes
    -----

    .. deprecated:: 2.1.0
        A deprecation warning is shown if ``indices`` is specified as a
        positional argument. This will need to be specified as a keyword
        argument in a future release.

    See Also
    --------
    multipoints

    """
    typ = GeometryType.MULTIPOLYGON
    geometries = np.asarray(geometries)
    if not isinstance(geometries, Geometry) and np.issubdtype(
        geometries.dtype, np.number
    ):
        geometries = polygons(geometries)
    if indices is None:
        return lib.create_collection(geometries, np.intc(typ), out=out, **kwargs)
    else:
        return collections_1d(geometries, indices, typ, out=out)


# Note: future plan is to change this signature over a few releases:
# shapely 2.0:
#   geometrycollections(geometries, indices=None, out=None, **kwargs)
# shapely 2.1: shows deprecation warning about positional 'indices' arg
#   geometrycollections(geometries, indices=None, *, out=None, **kwargs)
# shapely 2.2(?): enforce keyword-only arguments after 'indices'
#   geometrycollections(geometries, *, indices=None, out=None, **kwargs)


@deprecate_positional(["indices"], category=DeprecationWarning)
@multithreading_enabled
def geometrycollections(geometries, indices=None, out=None, **kwargs):
    """Create geometrycollections from arrays of geometries.

    Parameters
    ----------
    geometries : array_like
        An array of geometries.
    indices : array_like, optional
        Indices into the target array where input geometries belong. If
        provided, both geometries and indices should be 1D and have matching
        sizes. Indices should be in increasing order. Missing indices result
        in a ValueError unless ``out`` is  provided, in which case the original
        value in ``out`` is kept.
    out : ndarray, optional
        An array (with dtype object) to output the geometries into.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.
        Ignored if ``indices`` is provided.

    Notes
    -----

    .. deprecated:: 2.1.0
        A deprecation warning is shown if ``indices`` is specified as a
        positional argument. This will need to be specified as a keyword
        argument in a future release.

    See Also
    --------
    multipoints

    """
    typ = GeometryType.GEOMETRYCOLLECTION
    if indices is None:
        return lib.create_collection(geometries, np.intc(typ), out=out, **kwargs)
    else:
        return collections_1d(geometries, indices, typ, out=out)


def prepare(geometry, **kwargs):
    """Prepare a geometry, improving performance of other operations.

    A prepared geometry is a normal geometry with added information such as an
    index on the line segments. This improves the performance of the following
    operations: contains, contains_properly, covered_by, covers, crosses,
    disjoint, intersects, overlaps, touches, and within.

    Note that if a prepared geometry is modified, the newly created Geometry
    object is not prepared. In that case, ``prepare`` should be called again.

    This function does not recompute previously prepared geometries;
    it is efficient to call this function on an array that partially contains
    prepared geometries.

    This function does not return any values; geometries are modified in place.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometries are changed in place
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See Also
    --------
    is_prepared : Identify whether a geometry is prepared already.
    destroy_prepared : Destroy the prepared part of a geometry.

    Examples
    --------
    >>> import shapely
    >>> from shapely import Point
    >>> poly = shapely.buffer(Point(1.0, 1.0), 1)
    >>> shapely.prepare(poly)
    >>> shapely.contains_properly(poly, [Point(0.0, 0.0), Point(0.5, 0.5)]).tolist()
    [False, True]

    """
    lib.prepare(geometry, **kwargs)


def destroy_prepared(geometry, **kwargs):
    """Destroy the prepared part of a geometry, freeing up memory.

    Note that the prepared geometry will always be cleaned up if the geometry itself
    is dereferenced. This function needs only be called in very specific circumstances,
    such as freeing up memory without losing the geometries, or benchmarking.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometries are changed in-place
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See Also
    --------
    prepare

    """
    lib.destroy_prepared(geometry, **kwargs)


def empty(shape, geom_type=None, order="C"):
    """Create a geometry array prefilled with None or with empty geometries.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty array, e.g., ``(2, 3)`` or ``2``.
    geom_type : shapely.GeometryType, optional
        The desired geometry type in case the array should be prefilled
        with empty geometries. Default ``None``.
    order : {'C', 'F'}, optional, default: 'C'
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.

    Examples
    --------
    >>> import shapely
    >>> shapely.empty((2, 3)).tolist()
    [[None, None, None], [None, None, None]]
    >>> shapely.empty(2, geom_type=shapely.GeometryType.POINT).tolist()
    [<POINT EMPTY>, <POINT EMPTY>]

    """
    if geom_type is None:
        return np.empty(shape, dtype=object, order=order)

    geom_type = GeometryType(geom_type)  # cast int to GeometryType
    if geom_type is GeometryType.MISSING:
        return np.empty(shape, dtype=object, order=order)

    fill_value = from_wkt(geom_type.name + " EMPTY")
    return np.full(shape, fill_value, dtype=object, order=order)
