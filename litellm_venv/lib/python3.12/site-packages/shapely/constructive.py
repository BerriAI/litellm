"""Methods that yield new objects not derived from set-theoretic analysis."""

import numpy as np

from shapely import lib
from shapely._enum import ParamEnum
from shapely.algorithms._oriented_envelope import _oriented_envelope_min_area_vectorized
from shapely.algorithms.cga import _orient_polygons_vectorized
from shapely.decorators import (
    deprecate_positional,
    multithreading_enabled,
    requires_geos,
)
from shapely.errors import UnsupportedGEOSVersionError

__all__ = [
    "BufferCapStyle",
    "BufferJoinStyle",
    "boundary",
    "buffer",
    "build_area",
    "centroid",
    "clip_by_rect",
    "concave_hull",
    "constrained_delaunay_triangles",
    "convex_hull",
    "delaunay_triangles",
    "envelope",
    "extract_unique_points",
    "make_valid",
    "maximum_inscribed_circle",
    "minimum_bounding_circle",
    "minimum_clearance_line",
    "minimum_rotated_rectangle",
    "node",
    "normalize",
    "offset_curve",
    "orient_polygons",
    "oriented_envelope",
    "point_on_surface",
    "polygonize",
    "polygonize_full",
    "remove_repeated_points",
    "reverse",
    "segmentize",
    "simplify",
    "snap",
    "voronoi_polygons",
]


class BufferCapStyle(ParamEnum):
    """Enumeration of buffer cap styles.

    Attributes
    ----------
    round : int
        Represents a round cap style.
    flat : int
        Represents a flat cap style.
    square : int
        Represents a square cap style.

    """

    round = 1
    flat = 2
    square = 3


class BufferJoinStyle(ParamEnum):
    """Enumeration of buffer join styles.

    Attributes
    ----------
    round : int
        Specifies a round join style.
    mitre : int
        Specifies a mitre join style.
    bevel : int
        Specifies a bevel join style.

    """

    round = 1
    mitre = 2
    bevel = 3


@multithreading_enabled
def boundary(geometry, **kwargs):
    """Return the topological boundary of a geometry.

    This function will return None for geometrycollections.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry for which to return the boundary.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import GeometryCollection, LinearRing, LineString, \
MultiLineString, MultiPoint, Point, Polygon
    >>> shapely.boundary(Point(0, 0))
    <GEOMETRYCOLLECTION EMPTY>
    >>> shapely.boundary(LineString([(0, 0), (1, 1), (1, 2)]))
    <MULTIPOINT ((0 0), (1 2))>
    >>> shapely.boundary(LinearRing([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]))
    <MULTIPOINT EMPTY>
    >>> shapely.boundary(Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]))
    <LINESTRING (0 0, 1 0, 1 1, 0 1, 0 0)>
    >>> shapely.boundary(MultiPoint([(0, 0), (1, 2)]))
    <GEOMETRYCOLLECTION EMPTY>
    >>> shapely.boundary(MultiLineString([[(0, 0), (1, 1)], [(0, 1), (1, 0)]]))
    <MULTIPOINT ((0 0), (0 1), (1 0), (1 1))>
    >>> shapely.boundary(GeometryCollection([Point(0, 0)])) is None
    True

    """
    return lib.boundary(geometry, **kwargs)


# Note: future plan is to change this signature over a few releases:
# shapely 2.0:
#   buffer(geometry, distance, quad_segs=8, ...)
# shapely 2.1: shows deprecation warning about positional 'quad_segs', etc.
#   same signature as 2.0
# shapely 2.2(?): enforce keyword-only arguments after 'distance'
#   buffer(geometry, distance, *, quad_segs=8, ...)


@deprecate_positional(
    ["quad_segs", "cap_style", "join_style", "mitre_limit", "single_sided"],
    category=DeprecationWarning,
)
@multithreading_enabled
def buffer(
    geometry,
    distance,
    quad_segs=8,
    cap_style="round",
    join_style="round",
    mitre_limit=5.0,
    single_sided=False,
    **kwargs,
):
    """Compute the buffer of a geometry for positive and negative buffer distance.

    The buffer of a geometry is defined as the Minkowski sum (or difference,
    for negative distance) of the geometry with a circle with radius equal
    to the absolute value of the buffer distance.

    The buffer operation always returns a polygonal result. The negative
    or zero-distance buffer of lines and points is always empty.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries for which to compute the buffer.
    distance : float or array_like
        Specifies the circle radius in the Minkowski sum (or difference).
    quad_segs : int, default 8
        Specifies the number of linear segments in a quarter circle in the
        approximation of circular arcs.
    cap_style : shapely.BufferCapStyle or {'round', 'square', 'flat'}, default 'round'
        Specifies the shape of buffered line endings. BufferCapStyle.round ('round')
        results in circular line endings (see ``quad_segs``). Both BufferCapStyle.square
        ('square') and BufferCapStyle.flat ('flat') result in rectangular line endings,
        only BufferCapStyle.flat ('flat') will end at the original vertex,
        while BufferCapStyle.square ('square') involves adding the buffer width.
    join_style : shapely.BufferJoinStyle or {'round', 'mitre', 'bevel'}, default 'round'
        Specifies the shape of buffered line midpoints. BufferJoinStyle.round ('round')
        results in rounded shapes. BufferJoinStyle.bevel ('bevel') results in a beveled
        edge that touches the original vertex. BufferJoinStyle.mitre ('mitre') results
        in a single vertex that is beveled depending on the ``mitre_limit`` parameter.
    mitre_limit : float, default 5.0
        Crops of 'mitre'-style joins if the point is displaced from the
        buffered vertex by more than this limit.
    single_sided : bool, default False
        Only buffer at one side of the geometry.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Notes
    -----

    .. deprecated:: 2.1.0
        A deprecation warning is shown if ``quad_segs``,  ``cap_style``,
        ``join_style``, ``mitre_limit`` or ``single_sided`` are
        specified as positional arguments. In a future release, these will
        need to be specified as keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString, Point, Polygon, BufferCapStyle, BufferJoinStyle
    >>> shapely.buffer(Point(10, 10), 2, quad_segs=1)
    <POLYGON ((12 10, 10 8, 8 10, 10 12, 12 10))>
    >>> shapely.buffer(Point(10, 10), 2, quad_segs=2)
    <POLYGON ((12 10, 11.414 8.586, 10 8, 8.586 8.586, 8 10, 8.5...>
    >>> shapely.buffer(Point(10, 10), -2, quad_segs=1)
    <POLYGON EMPTY>
    >>> line = LineString([(10, 10), (20, 10)])
    >>> shapely.buffer(line, 2, cap_style="square")
    <POLYGON ((20 12, 22 12, 22 8, 10 8, 8 8, 8 12, 20 12))>
    >>> shapely.buffer(line, 2, cap_style="flat")
    <POLYGON ((20 12, 20 8, 10 8, 10 12, 20 12))>
    >>> shapely.buffer(line, 2, single_sided=True, cap_style="flat")
    <POLYGON ((20 10, 10 10, 10 12, 20 12, 20 10))>
    >>> line2 = LineString([(10, 10), (20, 10), (20, 20)])
    >>> shapely.buffer(line2, 2, cap_style="flat", join_style="bevel")
    <POLYGON ((18 12, 18 20, 22 20, 22 10, 20 8, 10 8, 10 12, 18 12))>
    >>> shapely.buffer(line2, 2, cap_style="flat", join_style="mitre")
    <POLYGON ((18 12, 18 20, 22 20, 22 8, 10 8, 10 12, 18 12))>
    >>> shapely.buffer(line2, 2, cap_style="flat", join_style="mitre", mitre_limit=1)
    <POLYGON ((18 12, 18 20, 22 20, 22 9.172, 20.828 8, 10 8, 10 12, 18 12))>
    >>> square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    >>> shapely.buffer(square, 2, join_style="mitre")
    <POLYGON ((-2 -2, -2 12, 12 12, 12 -2, -2 -2))>
    >>> shapely.buffer(square, -2, join_style="mitre")
    <POLYGON ((2 2, 2 8, 8 8, 8 2, 2 2))>
    >>> shapely.buffer(square, -5, join_style="mitre")
    <POLYGON EMPTY>
    >>> shapely.buffer(line, float("nan")) is None
    True

    """
    if isinstance(cap_style, str):
        cap_style = BufferCapStyle.get_value(cap_style)
    if isinstance(join_style, str):
        join_style = BufferJoinStyle.get_value(join_style)
    if not np.isscalar(quad_segs):
        raise TypeError("quad_segs only accepts scalar values")
    if not np.isscalar(cap_style):
        raise TypeError("cap_style only accepts scalar values")
    if not np.isscalar(join_style):
        raise TypeError("join_style only accepts scalar values")
    if not np.isscalar(mitre_limit):
        raise TypeError("mitre_limit only accepts scalar values")
    if not np.isscalar(single_sided):
        raise TypeError("single_sided only accepts scalar values")
    return lib.buffer(
        geometry,
        distance,
        np.intc(quad_segs),
        np.intc(cap_style),
        np.intc(join_style),
        mitre_limit,
        np.bool_(single_sided),
        **kwargs,
    )


# Note: future plan is to change this signature over a few releases:
# shapely 2.0:
#   offset_curve(geometry, distance, quad_segs=8, ...)
# shapely 2.1: shows deprecation warning about positional 'quad_segs', etc.
#   same signature as 2.0
# shapely 2.2(?): enforce keyword-only arguments after 'distance'
#   offset_curve(geometry, distance, *, quad_segs=8, ...)


@deprecate_positional(
    ["quad_segs", "join_style", "mitre_limit"], category=DeprecationWarning
)
@multithreading_enabled
def offset_curve(
    geometry, distance, quad_segs=8, join_style="round", mitre_limit=5.0, **kwargs
):
    """Return a (Multi)LineString at a distance from the object.

    For positive distance the offset will be at the left side of the input
    line. For a negative distance it will be at the right side. In general,
    this function tries to preserve the direction of the input.

    Note: the behaviour regarding orientation of the resulting line depends
    on the GEOS version. With GEOS < 3.11, the line retains the same
    direction for a left offset (positive distance) or has opposite direction
    for a right offset (negative distance), and this behaviour was documented
    as such in previous Shapely versions. Starting with GEOS 3.11, the
    function tries to preserve the orientation of the original line.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries for which to compute the offset.
    distance : float or array_like
        Specifies the offset distance from the input geometry. Negative
        for right side offset, positive for left side offset.
    quad_segs : int, default 8
        Specifies the number of linear segments in a quarter circle in the
        approximation of circular arcs.
    join_style : {'round', 'bevel', 'mitre'}, default 'round'
        Specifies the shape of outside corners. 'round' results in
        rounded shapes. 'bevel' results in a beveled edge that touches the
        original vertex. 'mitre' results in a single vertex that is beveled
        depending on the ``mitre_limit`` parameter.
    mitre_limit : float, default 5.0
        Crops of 'mitre'-style joins if the point is displaced from the
        buffered vertex by more than this limit.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Notes
    -----

    .. deprecated:: 2.1.0
        A deprecation warning is shown if ``quad_segs``, ``join_style`` or
        ``mitre_limit`` are specified as positional arguments. In a future
        release, these will need to be specified as keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString
    >>> line = LineString([(0, 0), (0, 2)])
    >>> shapely.offset_curve(line, 2)
    <LINESTRING (-2 0, -2 2)>
    >>> shapely.offset_curve(line, -2)
    <LINESTRING (2 0, 2 2)>

    """
    if isinstance(join_style, str):
        join_style = BufferJoinStyle.get_value(join_style)
    if not np.isscalar(quad_segs):
        raise TypeError("quad_segs only accepts scalar values")
    if not np.isscalar(join_style):
        raise TypeError("join_style only accepts scalar values")
    if not np.isscalar(mitre_limit):
        raise TypeError("mitre_limit only accepts scalar values")
    return lib.offset_curve(
        geometry,
        distance,
        np.intc(quad_segs),
        np.intc(join_style),
        np.double(mitre_limit),
        **kwargs,
    )


@multithreading_enabled
def centroid(geometry, **kwargs):
    """Compute the geometric center (center-of-mass) of a geometry.

    For multipoints this is computed as the mean of the input coordinates.
    For multilinestrings the centroid is weighted by the length of each
    line segment. For multipolygons the centroid is weighted by the area of
    each polygon.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries for which to compute the centroid.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString, MultiPoint, Polygon
    >>> shapely.centroid(Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]))
    <POINT (5 5)>
    >>> shapely.centroid(LineString([(0, 0), (2, 2), (10, 10)]))
    <POINT (5 5)>
    >>> shapely.centroid(MultiPoint([(0, 0), (10, 10)]))
    <POINT (5 5)>
    >>> shapely.centroid(Polygon())
    <POINT EMPTY>

    """
    return lib.centroid(geometry, **kwargs)


@multithreading_enabled
def clip_by_rect(geometry, xmin, ymin, xmax, ymax, **kwargs):
    """Return the portion of a geometry within a rectangle.

    The geometry is clipped in a fast but possibly dirty way. The output is
    not guaranteed to be valid. No exceptions will be raised for topological
    errors.

    Note: empty geometries or geometries that do not overlap with the
    specified bounds will result in GEOMETRYCOLLECTION EMPTY.

    Parameters
    ----------
    geometry : Geometry or array_like
        The geometry to be clipped.
    xmin : float
        Minimum x value of the rectangle.
    ymin : float
        Minimum y value of the rectangle.
    xmax : float
        Maximum x value of the rectangle.
    ymax : float
        Maximum y value of the rectangle.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString, Polygon
    >>> line = LineString([(0, 0), (10, 10)])
    >>> shapely.clip_by_rect(line, 0., 0., 1., 1.)
    <LINESTRING (0 0, 1 1)>
    >>> polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    >>> shapely.clip_by_rect(polygon, 0., 0., 1., 1.)
    <POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))>

    """
    if not all(np.isscalar(val) for val in [xmin, ymin, xmax, ymax]):
        raise TypeError("xmin/ymin/xmax/ymax only accepts scalar values")
    return lib.clip_by_rect(
        geometry,
        np.double(xmin),
        np.double(ymin),
        np.double(xmax),
        np.double(ymax),
        **kwargs,
    )


@requires_geos("3.11.0")
@multithreading_enabled
def concave_hull(geometry, ratio=0.0, allow_holes=False, **kwargs):
    """Compute a concave geometry that encloses an input geometry.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries for which to compute the concave hull.
    ratio : float, default 0.0
        Number in the range [0, 1]. Higher numbers will include fewer vertices
        in the hull.
    allow_holes : bool, default False
        If set to True, the concave hull may have holes.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import MultiPoint, Polygon
    >>> multi_point = MultiPoint([(0, 0), (0, 3), (1, 1), (3, 0), (3, 3)])
    >>> shapely.concave_hull(multi_point, ratio=0.1)
    <POLYGON ((0 0, 0 3, 1 1, 3 3, 3 0, 0 0))>
    >>> shapely.concave_hull(multi_point, ratio=1.0)
    <POLYGON ((0 0, 0 3, 3 3, 3 0, 0 0))>
    >>> shapely.concave_hull(Polygon())
    <POLYGON EMPTY>

    """
    if not np.isscalar(ratio):
        raise TypeError("ratio must be scalar")
    if not np.isscalar(allow_holes):
        raise TypeError("allow_holes must be scalar")
    return lib.concave_hull(geometry, np.double(ratio), np.bool_(allow_holes), **kwargs)


@multithreading_enabled
def convex_hull(geometry, **kwargs):
    """Compute the minimum convex geometry that encloses an input geometry.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries for which to compute the convex hull.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import MultiPoint, Polygon
    >>> shapely.convex_hull(MultiPoint([(0, 0), (10, 0), (10, 10)]))
    <POLYGON ((0 0, 10 10, 10 0, 0 0))>
    >>> shapely.convex_hull(Polygon())
    <GEOMETRYCOLLECTION EMPTY>

    """
    return lib.convex_hull(geometry, **kwargs)


@multithreading_enabled
def delaunay_triangles(geometry, tolerance=0.0, only_edges=False, **kwargs):
    """Compute a Delaunay triangulation around the vertices of an input geometry.

    The output is a geometrycollection containing polygons (default)
    or linestrings (see ``only_edges``). Returns an empty geometry for input
    geometries that contain less than 3 vertices.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries for which to compute the Delaunay triangulation.
    tolerance : float or array_like, default 0.0
        Snap input vertices together if their distance is less than this value.
    only_edges : bool or array_like, default False
        If set to True, the triangulation will return a collection of
        linestrings instead of polygons.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Returns
    -------
    GeometryCollection or array of GeometryCollections

    See Also
    --------
    constrained_delaunay_triangles

    Examples
    --------
    >>> import shapely
    >>> from shapely import GeometryCollection, LineString, MultiPoint, Polygon
    >>> points = MultiPoint([(50, 30), (60, 30), (100, 100)])
    >>> shapely.delaunay_triangles(points).normalize()
    <GEOMETRYCOLLECTION (POLYGON ((50 30, 100 100, 60 30, 50 30)))>
    >>> shapely.delaunay_triangles(points, only_edges=True)
    <MULTILINESTRING ((50 30, 100 100), (50 30, 60 30), ...>
    >>> shapely.delaunay_triangles(
    ...     MultiPoint([(50, 30), (51, 30), (60, 30), (100, 100)]),
    ...     tolerance=2
    ... ).normalize()
    <GEOMETRYCOLLECTION (POLYGON ((50 30, 100 100, 60 30, 50 30)))>
    >>> shapely.delaunay_triangles(Polygon([(50, 30), (60, 30), (100, 100), (50, 30)]))\
.normalize()
    <GEOMETRYCOLLECTION (POLYGON ((50 30, 100 100, 60 30, 50 30)))>
    >>> shapely.delaunay_triangles(LineString([(50, 30), (60, 30), (100, 100)]))\
.normalize()
    <GEOMETRYCOLLECTION (POLYGON ((50 30, 100 100, 60 30, 50 30)))>
    >>> shapely.delaunay_triangles(GeometryCollection([]))
    <GEOMETRYCOLLECTION EMPTY>

    """
    return lib.delaunay_triangles(geometry, tolerance, only_edges, **kwargs)


@requires_geos("3.10.0")
@multithreading_enabled
def constrained_delaunay_triangles(geometry, **kwargs):
    """Compute the constrained Delaunay triangulation of polygons.

    A constrained Delaunay triangulation requires the edges of the input
    polygon(s) to be in the set of resulting triangle edges. An unconstrained
    delaunay triangulation only triangulates based on the vertices, hence
    triangle edges could cross polygon boundaries.

    .. versionadded:: 2.1.0

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    GeometryCollection or array of GeometryCollections
        * GeometryCollection of polygons, given polygonal input
        * Empty GeometryCollection, given non-polygonal input

    See Also
    --------
    delaunay_triangles

    Examples
    --------
    >>> import shapely
    >>> from shapely import MultiPoint, MultiPolygon, Polygon
    >>> shapely.constrained_delaunay_triangles(Polygon([(10, 10), (20, 40), (90, 90), (90, 10), (10, 10)]))
    <GEOMETRYCOLLECTION (POLYGON ((90 10, 20 40, 90 90, 90 10)), POLYGON ((20 40...>
    >>> shapely.constrained_delaunay_triangles(Polygon())
    <GEOMETRYCOLLECTION EMPTY>
    >>> shapely.constrained_delaunay_triangles(MultiPolygon([Polygon(((50, 30), (60, 30), (100, 100), (50, 30))), Polygon(((10, 10), (20, 40), (90, 90), (90, 10), (10, 10)))]))
    <GEOMETRYCOLLECTION (POLYGON ((50 30, 100 100, 60 30, 50 30)), POLYGON ((90 ...>
    >>> shapely.constrained_delaunay_triangles(MultiPolygon())
    <GEOMETRYCOLLECTION EMPTY>
    >>> shapely.constrained_delaunay_triangles(MultiPoint([(50, 30), (51, 30), (60, 30), (100, 100)]))
    <GEOMETRYCOLLECTION EMPTY>

    """  # noqa: E501
    return lib.constrained_delaunay_triangles(geometry, **kwargs)


@multithreading_enabled
def envelope(geometry, **kwargs):
    """Compute the minimum bounding box that encloses an input geometry.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries for which to compute the envelope.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import GeometryCollection, LineString, MultiPoint, Point
    >>> shapely.envelope(LineString([(0, 0), (10, 10)]))
    <POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))>
    >>> shapely.envelope(MultiPoint([(0, 0), (10, 10)]))
    <POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))>
    >>> shapely.envelope(Point(0, 0))
    <POINT (0 0)>
    >>> shapely.envelope(GeometryCollection([]))
    <POINT EMPTY>

    """
    return lib.envelope(geometry, **kwargs)


@multithreading_enabled
def extract_unique_points(geometry, **kwargs):
    """Return all distinct vertices of an input geometry as a multipoint.

    Note that only 2 dimensions of the vertices are considered when testing
    for equality.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries for which to extract unique points.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString, MultiPoint, Point, Polygon
    >>> shapely.extract_unique_points(Point(0, 0))
    <MULTIPOINT ((0 0))>
    >>> shapely.extract_unique_points(LineString([(0, 0), (1, 1), (1, 1)]))
    <MULTIPOINT ((0 0), (1 1))>
    >>> shapely.extract_unique_points(Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]))
    <MULTIPOINT ((0 0), (1 0), (1 1), (0 1))>
    >>> shapely.extract_unique_points(MultiPoint([(0, 0), (1, 1), (0, 0)]))
    <MULTIPOINT ((0 0), (1 1))>
    >>> shapely.extract_unique_points(LineString())
    <MULTIPOINT EMPTY>

    """
    return lib.extract_unique_points(geometry, **kwargs)


@multithreading_enabled
def build_area(geometry, **kwargs):
    """Create an areal geometry formed by the constituent linework of given geometry.

    Equivalent of the PostGIS ST_BuildArea() function.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries for which to build an area.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import GeometryCollection, Polygon
    >>> polygon1 = Polygon([(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)])
    >>> polygon2 = Polygon([(1, 1), (1, 2), (2, 2), (1, 1)])
    >>> shapely.build_area(GeometryCollection([polygon1, polygon2]))
    <POLYGON ((0 0, 0 3, 3 3, 3 0, 0 0), (1 1, 2 2, 1 2, 1 1))>

    """
    return lib.build_area(geometry, **kwargs)


@multithreading_enabled
def make_valid(geometry, *, method="linework", keep_collapsed=True, **kwargs):
    """Repair invalid geometries.

    Two ``methods`` are available:

    * the 'linework' algorithm tries to preserve every edge and vertex in the input. It
      combines all rings into a set of noded lines and then extracts valid polygons from
      that linework. An alternating even-odd strategy is used to assign areas as
      interior or exterior. A disadvantage is that for some relatively simple invalid
      geometries this produces rather complex results.
    * the 'structure' algorithm tries to reason from the structure of the input to find
      the 'correct' repair: exterior rings bound area, interior holes exclude area.
      It first makes all rings valid, then shells are merged and holes are subtracted
      from the shells to generate valid result. It assumes that holes and shells are
      correctly categorized in the input geometry.

    Example:

    .. plot:: code/make_valid_methods.py

    When using ``make_valid`` on a Polygon, the result can be a GeometryCollection. For
    this example this is the case when the 'linework' ``method`` is used. LineStrings in
    the result are drawn in red.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to repair.
    method : {'linework', 'structure'}, default 'linework'
        Algorithm to use when repairing geometry. 'structure'
        requires GEOS >= 3.10.

        .. versionadded:: 2.1.0
    keep_collapsed : bool, default True
        For the 'structure' method, True will keep components that have collapsed into a
        lower dimensionality. For example, a ring collapsing to a line, or a line
        collapsing to a point. Must be True for the 'linework' method.

        .. versionadded:: 2.1.0
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import Polygon
    >>> polygon = Polygon([(0, 0), (1, 1), (1, 2), (1, 1), (0, 0)])
    >>> shapely.is_valid(polygon)
    False
    >>> shapely.make_valid(polygon)
    <MULTILINESTRING ((0 0, 1 1), (1 1, 1 2))>
    >>> shapely.make_valid(polygon, method="structure", keep_collapsed=True)
    <LINESTRING (0 0, 1 1, 1 2, 1 1, 0 0)>
    >>> shapely.make_valid(polygon, method="structure", keep_collapsed=False)
    <POLYGON EMPTY>

    """
    if not np.isscalar(method):
        raise TypeError("method only accepts scalar values")
    if not np.isscalar(keep_collapsed):
        raise TypeError("keep_collapsed only accepts scalar values")

    if method == "linework":
        if keep_collapsed is False:
            raise ValueError(
                "The 'linework' method does not support 'keep_collapsed=False'"
            )

        # The make_valid code can be removed once support for GEOS < 3.10 is dropped.
        # In GEOS >= 3.10, make_valid just calls make_valid_with_params with
        # method="linework" and keep_collapsed=True, so there is no advantage to keep
        # both code paths in shapely on long term.
        return lib.make_valid(geometry, **kwargs)

    elif method == "structure":
        if lib.geos_version < (3, 10, 0):
            raise ValueError(
                "The 'structure' method is only available in GEOS >= 3.10.0"
            )

        return lib.make_valid_with_params(
            geometry, np.intc(1), np.bool_(keep_collapsed), **kwargs
        )

    else:
        raise ValueError(f"Unknown method: {method}")


@multithreading_enabled
def minimum_clearance_line(geometry, **kwargs):
    """Return a LineString whose endpoints define the minimum clearance.

    A geometry's "minimum clearance" is the smallest distance by which a vertex
    of the geometry could be moved to produce an invalid geometry.

    If the geometry has no minimum clearance, an empty LineString will be
    returned.

    .. versionadded:: 2.1.0

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to determine the minimum clearance line for.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> import shapely
    >>> from shapely import Polygon
    >>> poly = Polygon([(0, 0), (10, 0), (10, 10), (5, 5), (0, 10), (0, 0)])
    >>> shapely.minimum_clearance_line(poly)
    <LINESTRING (5 5, 5 0)>

    See Also
    --------
    minimum_clearance

    """
    return lib.minimum_clearance_line(geometry, **kwargs)


@multithreading_enabled
def normalize(geometry, **kwargs):
    """Convert Geometry to strict normal form (or canonical form).

    In :ref:`strict canonical form <canonical-form>`, the coordinates, rings of
    a polygon and parts of multi geometries are ordered consistently. Typically
    useful for testing purposes (for example in combination with
    ``equals_exact``).

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to normalize.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import MultiLineString
    >>> line = MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]])
    >>> shapely.normalize(line)
    <MULTILINESTRING ((2 2, 3 3), (0 0, 1 1))>

    """
    return lib.normalize(geometry, **kwargs)


@multithreading_enabled
def point_on_surface(geometry, **kwargs):
    """Return a point that intersects an input geometry.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries for which to compute a point on the surface.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString, MultiPoint, Polygon
    >>> shapely.point_on_surface(Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]))
    <POINT (5 5)>
    >>> shapely.point_on_surface(LineString([(0, 0), (2, 2), (10, 10)]))
    <POINT (2 2)>
    >>> shapely.point_on_surface(MultiPoint([(0, 0), (10, 10)]))
    <POINT (0 0)>
    >>> shapely.point_on_surface(Polygon())
    <POINT EMPTY>

    """
    return lib.point_on_surface(geometry, **kwargs)


@multithreading_enabled
def node(geometry, **kwargs):
    """Return the fully noded version of the linear input as MultiLineString.

    Given a linear input geometry, this function returns a new MultiLineString
    in which no lines cross each other but only touch at and points. To
    obtain this, all intersections between segments are computed and added
    to the segments, and duplicate segments are removed.

    Non-linear input (points) will result in an empty MultiLineString.

    This function can for example be used to create a fully-noded linework
    suitable to passed as input to ``polygonize``.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries for which to compute the noded version.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString, Point
    >>> line = LineString([(0, 0), (1,1), (0, 1), (1, 0)])
    >>> shapely.node(line)
    <MULTILINESTRING ((0 0, 0.5 0.5), (0.5 0.5, 1 1, 0 1, 0.5 0.5), (0.5 0.5, 1 0))>
    >>> shapely.node(Point(1, 1))
    <MULTILINESTRING EMPTY>

    """
    return lib.node(geometry, **kwargs)


def polygonize(geometries, **kwargs):
    """Create polygons formed from the linework of a set of Geometries.

    Polygonizes an array of Geometries that contain linework which
    represents the edges of a planar graph. Any type of Geometry may be
    provided as input; only the constituent lines and rings will be used to
    create the output polygons.

    Lines or rings that when combined do not completely close a polygon
    will result in an empty GeometryCollection.  Duplicate segments are
    ignored.

    This function returns the polygons within a GeometryCollection.
    Individual Polygons can be obtained using ``get_geometry`` to get
    a single polygon or ``get_parts`` to get an array of polygons.
    MultiPolygons can be constructed from the output using
    ``shapely.multipolygons(shapely.get_parts(shapely.polygonize(geometries)))``.

    Parameters
    ----------
    geometries : array_like
        An array of geometries.
    axis : int
        Axis along which the geometries are polygonized.
        The default is to perform a reduction over the last dimension
        of the input array. A 1D array results in a scalar geometry.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Returns
    -------
    GeometryCollection or array of GeometryCollections

    See Also
    --------
    get_parts, get_geometry
    polygonize_full
    node

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString
    >>> lines = [
    ...     LineString([(0, 0), (1, 1)]),
    ...     LineString([(0, 0), (0, 1)]),
    ...     LineString([(0, 1), (1, 1)])
    ... ]
    >>> shapely.polygonize(lines)
    <GEOMETRYCOLLECTION (POLYGON ((1 1, 0 0, 0 1, 1 1)))>

    """
    return lib.polygonize(geometries, **kwargs)


def polygonize_full(geometries, **kwargs):
    """Create polygons formed from the linework of a set of Geometries.

    All extra outputs are returned as well.

    Polygonizes an array of Geometries that contain linework which
    represents the edges of a planar graph. Any type of Geometry may be
    provided as input; only the constituent lines and rings will be used to
    create the output polygons.

    This function performs the same polygonization as ``polygonize`` but does
    not only return the polygonal result but all extra outputs as well. The
    return value consists of 4 elements:

    * The polygonal valid output
    * **Cut edges**: edges connected on both ends but not part of polygonal output
    * **dangles**: edges connected on one end but not part of polygonal output
    * **invalid rings**: polygons formed but which are not valid

    This function returns the geometries within GeometryCollections.
    Individual geometries can be obtained using ``get_geometry`` to get
    a single geometry or ``get_parts`` to get an array of geometries.

    Parameters
    ----------
    geometries : array_like
        An array of geometries.
    axis : int
        Axis along which the geometries are polygonized.
        The default is to perform a reduction over the last dimension
        of the input array. A 1D array results in a scalar geometry.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Returns
    -------
    (polygons, cuts, dangles, invalid)
        tuple of 4 GeometryCollections or arrays of GeometryCollections

    See Also
    --------
    polygonize

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString
    >>> lines = [
    ...     LineString([(0, 0), (1, 1)]),
    ...     LineString([(0, 0), (0, 1), (1, 1)]),
    ...     LineString([(0, 1), (1, 1)])
    ... ]
    >>> shapely.polygonize_full(lines)
    (<GEOMETRYCOLLECTION (POLYGON ((1 1, 0 0, 0 1, 1 1)))>,
     <GEOMETRYCOLLECTION EMPTY>,
     <GEOMETRYCOLLECTION (LINESTRING (0 1, 1 1))>,
     <GEOMETRYCOLLECTION EMPTY>)

    """
    return lib.polygonize_full(geometries, **kwargs)


@requires_geos("3.11.0")
@multithreading_enabled
def remove_repeated_points(geometry, tolerance=0.0, **kwargs):
    """Return a copy of a Geometry with repeated points removed.

    From the start of the coordinate sequence, each next point within the
    tolerance is removed.

    Removing repeated points with a non-zero tolerance may result in an invalid
    geometry being returned.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to remove repeated points from.
    tolerance : float or array_like, default=0.0
        Use 0.0 to remove only exactly repeated points.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString, Polygon
    >>> shapely.remove_repeated_points(LineString([(0,0), (0,0), (1,0)]), tolerance=0)
    <LINESTRING (0 0, 1 0)>
    >>> shapely.remove_repeated_points(Polygon([(0, 0), (0, .5), (0, 1), (.5, 1), (0,0)]), tolerance=.5)
    <POLYGON ((0 0, 0 1, 0 0))>

    """  # noqa: E501
    return lib.remove_repeated_points(geometry, tolerance, **kwargs)


@multithreading_enabled
def reverse(geometry, **kwargs):
    """Return a copy of a Geometry with the order of coordinates reversed.

    If a Geometry is a polygon with interior rings, the interior rings are also
    reversed.

    Points are unchanged. None is returned where Geometry is None.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to reverse the coordinates of.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See Also
    --------
    is_ccw : Checks if a Geometry is clockwise.

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString, Polygon
    >>> shapely.reverse(LineString([(0, 0), (1, 2)]))
    <LINESTRING (1 2, 0 0)>
    >>> shapely.reverse(Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]))
    <POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))>
    >>> shapely.reverse(None) is None
    True

    """
    return lib.reverse(geometry, **kwargs)


@requires_geos("3.10.0")
@multithreading_enabled
def segmentize(geometry, max_segment_length, **kwargs):
    """Add vertices to line segments based on maximum segment length.

    Additional vertices will be added to every line segment in an input geometry
    so that segments are no longer than the provided maximum segment length. New
    vertices will evenly subdivide each segment.

    Only linear components of input geometries are densified; other geometries
    are returned unmodified.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to segmentize.
    max_segment_length : float or array_like
        Additional vertices will be added so that all line segments are no
        longer than this value.  Must be greater than 0.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString, Polygon
    >>> line = LineString([(0, 0), (0, 10)])
    >>> shapely.segmentize(line, max_segment_length=5)
    <LINESTRING (0 0, 0 5, 0 10)>
    >>> polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    >>> shapely.segmentize(polygon, max_segment_length=5)
    <POLYGON ((0 0, 5 0, 10 0, 10 5, 10 10, 5 10, 0 10, 0 5, 0 0))>
    >>> shapely.segmentize(None, max_segment_length=5) is None
    True

    """
    return lib.segmentize(geometry, max_segment_length, **kwargs)


# Note: future plan is to change this signature over a few releases:
# shapely 2.0:
#   simplify(geometry, tolerance, preserve_topology=True, **kwargs)
# shapely 2.1: shows deprecation warning about positional 'preserve_topology'
#   same signature as 2.0
# shapely 2.2(?): enforce keyword-only arguments after 'tolerance'
#   simplify(geometry, tolerance, *, preserve_topology=True, **kwargs)


@deprecate_positional(["preserve_topology"], category=DeprecationWarning)
@multithreading_enabled
def simplify(geometry, tolerance, preserve_topology=True, **kwargs):
    """Return a simplified version of an input geometry.

    The Douglas-Peucker algorithm is used to simplify the geometry.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to simplify.
    tolerance : float or array_like
        The maximum allowed geometry displacement. The higher this value, the
        smaller the number of vertices in the resulting geometry.
    preserve_topology : bool, default True
        By default (True), the operation will avoid creating invalid
        geometries (checking for collapses, ring-intersections, etc), but
        this is computationally more expensive.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Notes
    -----

    .. deprecated:: 2.1.0
        A deprecation warning is shown if ``preserve_topology`` is specified as
        a positional argument. This will need to be specified as a keyword
        argument in a future release.

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString, Polygon
    >>> line = LineString([(0, 0), (1, 10), (0, 20)])
    >>> shapely.simplify(line, tolerance=0.9)
    <LINESTRING (0 0, 1 10, 0 20)>
    >>> shapely.simplify(line, tolerance=1)
    <LINESTRING (0 0, 0 20)>
    >>> polygon_with_hole = Polygon(
    ...     [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
    ...     holes=[[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]]
    ... )
    >>> shapely.simplify(polygon_with_hole, tolerance=4, preserve_topology=True)
    <POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0), (4 2, 2 4, 4 4, 4 2))>
    >>> shapely.simplify(polygon_with_hole, tolerance=4, preserve_topology=False)
    <POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))>

    """
    if preserve_topology:
        return lib.simplify_preserve_topology(geometry, tolerance, **kwargs)
    else:
        return lib.simplify(geometry, tolerance, **kwargs)


@multithreading_enabled
def snap(geometry, reference, tolerance, **kwargs):
    """Snap the vertices and segments of the geometry to vertices of the reference.

    Vertices and segments of the input geometry are snapped to vertices of the
    reference geometry, returning a new geometry; the input geometries are not
    modified. The result geometry is the input geometry with the vertices and
    segments snapped. If no snapping occurs then the input geometry is returned
    unchanged. The tolerance is used to control where snapping is performed.

    Where possible, this operation tries to avoid creating invalid geometries;
    however, it does not guarantee that output geometries will be valid. It is
    the responsibility of the caller to check for and handle invalid geometries.

    Because too much snapping can result in invalid geometries being created,
    heuristics are used to determine the number and location of snapped
    vertices that are likely safe to snap. These heuristics may omit
    some potential snaps that are otherwise within the tolerance.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to snap.
    reference : Geometry or array_like
        Geometry or geometries to snap to.
    tolerance : float or array_like
        The maximum distance between the input and reference geometries for
        snapping to occur. A value of 0 will snap only identical points.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString, Point, Polygon, MultiPoint

    >>> point = Point(0.5, 2.5)
    >>> target_point = Point(0, 2)
    >>> shapely.snap(point, target_point, tolerance=1)
    <POINT (0 2)>
    >>> shapely.snap(point, target_point, tolerance=0.49)
    <POINT (0.5 2.5)>

    >>> polygon = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
    >>> shapely.snap(polygon, Point(8, 10), tolerance=5)
    <POLYGON ((0 0, 0 10, 8 10, 10 0, 0 0))>
    >>> shapely.snap(polygon, LineString([(8, 10), (8, 0)]), tolerance=5)
    <POLYGON ((0 0, 0 10, 8 10, 8 0, 0 0))>

    You can snap one line to another, for example to clean imprecise coordinates:

    >>> line1 = LineString([(0.1, 0.1), (0.49, 0.51), (1.01, 0.89)])
    >>> line2 = LineString([(0, 0), (0.5, 0.5), (1.0, 1.0)])
    >>> shapely.snap(line1, line2, 0.25)
    <LINESTRING (0 0, 0.5 0.5, 1 1)>

    Snapping also supports Z coordinates:

    >>> point1 = Point(0.1, 0.1, 0.5)
    >>> multipoint = MultiPoint([(0, 0, 1), (0, 0, 0)])
    >>> shapely.snap(point1, multipoint, 1)
    <POINT Z (0 0 1)>

    Snapping to an empty geometry has no effect:

    >>> shapely.snap(line1, LineString([]), 0.25)
    <LINESTRING (0.1 0.1, 0.49 0.51, 1.01 0.89)>

    Snapping to a non-geometry (None) will always return None:

    >>> shapely.snap(line1, None, 0.25) is None
    True

    Only one vertex of a polygon is snapped to a target point,
    even if all vertices are equidistant to it,
    in order to prevent collapse of the polygon:

    >>> poly = shapely.box(0, 0, 1, 1)
    >>> poly
    <POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))>
    >>> shapely.snap(poly, Point(0.5, 0.5), 1)
    <POLYGON ((0.5 0.5, 1 1, 0 1, 0 0, 0.5 0.5))>

    """
    return lib.snap(geometry, reference, tolerance, **kwargs)


# Note: future plan is to change this signature over a few releases:
# shapely 2.0:
#   voronoi_polygons(geometry, tolerance=0.0, extend_to=None, ...)
# shapely 2.1: shows deprecation warning about positional 'extend_to'
#   same signature as 2.0
# shapely 2.2(?): enforce keyword-only arguments after 'tolerance'
#   voronoi_polygons(geometry, tolerance=0.0, extend_to=None, ...)


@deprecate_positional(
    ["extend_to", "only_edges", "ordered"], category=DeprecationWarning
)
@multithreading_enabled
def voronoi_polygons(
    geometry, tolerance=0.0, extend_to=None, only_edges=False, ordered=False, **kwargs
):
    """Compute a Voronoi diagram from the vertices of an input geometry.

    The output is a geometrycollection containing polygons (default)
    or linestrings (see only_edges). Returns empty if an input geometry
    contains less than 2 vertices or if the provided extent has zero area.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries for which to compute the Voronoi diagram.
    tolerance : float or array_like, default 0.0
        Snap input vertices together if their distance is less than this value.
    extend_to : Geometry or array_like, optional
        If provided, the diagram will be extended to cover the envelope of this
        geometry (unless this envelope is smaller than the input geometry).
    only_edges : bool or array_like, default False
        If set to True, the triangulation will return a collection of
        linestrings instead of polygons.
    ordered : bool or array_like, default False
        If set to True, polygons within the GeometryCollection will be ordered
        according to the order of the input vertices. Note that this may slow
        down the computation. Requires GEOS >= 3.12.0.

        .. versionadded:: 2.1.0
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Notes
    -----

    .. deprecated:: 2.1.0
        A deprecation warning is shown if ``extend_to``, ``only_edges`` or
        ``ordered`` are specified as positional arguments. In a future
        release, these will need to be specified as keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString, MultiPoint, Point
    >>> points = MultiPoint([(2, 2), (4, 2)])
    >>> shapely.voronoi_polygons(points).normalize()
    <GEOMETRYCOLLECTION (POLYGON ((3 0, 3 4, 6 4, 6 0, 3 0)), POLYGON ((0 0, 0 4...>
    >>> shapely.voronoi_polygons(points, only_edges=True)
    <MULTILINESTRING ((3 4, 3 0))>
    >>> shapely.voronoi_polygons(MultiPoint([(2, 2), (4, 2), (4.2, 2)]), 0.5, only_edges=True)
    <MULTILINESTRING ((3 4.2, 3 -0.2))>
    >>> shapely.voronoi_polygons(points, extend_to=LineString([(0, 0), (10, 10)]), only_edges=True)
    <MULTILINESTRING ((3 10, 3 0))>
    >>> shapely.voronoi_polygons(LineString([(2, 2), (4, 2)]), only_edges=True)
    <MULTILINESTRING ((3 4, 3 0))>
    >>> shapely.voronoi_polygons(Point(2, 2))
    <GEOMETRYCOLLECTION EMPTY>
    >>> shapely.voronoi_polygons(points, ordered=True)
    <GEOMETRYCOLLECTION (POLYGON ((0 0, 0 4, 3 4, 3 0, 0 0)), POLYGON ((6 4, 6 0...>

    """  # noqa: E501
    if ordered is not False and lib.geos_version < (3, 12, 0):
        raise UnsupportedGEOSVersionError(
            "Ordered Voronoi polygons require GEOS >= 3.12.0, "
            f"found {lib.geos_version_string}"
        )
    return lib.voronoi_polygons(
        geometry, tolerance, extend_to, only_edges, ordered, **kwargs
    )


@multithreading_enabled
def _oriented_envelope_geos(geometry, **kwargs):
    return lib.oriented_envelope(geometry, **kwargs)


def oriented_envelope(geometry, **kwargs):
    """Compute the oriented envelope (minimum rotated rectangle) of the input geometry.

    The oriented envelope encloses an input geometry, such that the resulting
    rectangle has minimum area.

    Unlike envelope this rectangle is not constrained to be parallel to the
    coordinate axes. If the convex hull of the object is a degenerate (line
    or point) this degenerate is returned.

    The starting point of the rectangle is not fixed. You can use
    :func:`~shapely.normalize` to reorganize the rectangle to
    :ref:`strict canonical form <canonical-form>` so the starting point is
    always the lower left point.

    ``minimum_rotated_rectangle`` is an alias for ``oriented_envelope``.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries for which to compute the oriented envelope.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import GeometryCollection, LineString, MultiPoint, Point, Polygon
    >>> shapely.oriented_envelope(MultiPoint([(0, 0), (10, 0), (10, 10)])).normalize()
    <POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))>
    >>> shapely.oriented_envelope(LineString([(1, 1), (5, 1), (10, 10)])).normalize()
    <POLYGON ((1 1, 10 10, 12 8, 3 -1, 1 1))>
    >>> shapely.oriented_envelope(Polygon([(1, 1), (15, 1), (5, 10), (1, 1)]))\
.normalize()
    <POLYGON ((1 1, 5 10, 16.691 4.804, 12.691 -4.196, 1 1))>
    >>> shapely.oriented_envelope(LineString([(1, 1), (10, 1)])).normalize()
    <LINESTRING (1 1, 10 1)>
    >>> shapely.oriented_envelope(Point(2, 2))
    <POINT (2 2)>
    >>> shapely.oriented_envelope(GeometryCollection([]))
    <POLYGON EMPTY>

    """
    if lib.geos_version < (3, 12, 0):
        f = _oriented_envelope_min_area_vectorized
    else:
        f = _oriented_envelope_geos
    return f(geometry, **kwargs)


minimum_rotated_rectangle = oriented_envelope


@multithreading_enabled
def minimum_bounding_circle(geometry, **kwargs):
    """Compute the minimum bounding circle that encloses an input geometry.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries for which to compute the minimum bounding circle.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import GeometryCollection, LineString, MultiPoint, Point, Polygon
    >>> shapely.minimum_bounding_circle(
    ...     Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
    ... )
    <POLYGON ((12.071 5, 11.935 3.621, 11.533 2.294, 10.879 1.07...>
    >>> shapely.minimum_bounding_circle(LineString([(1, 1), (10, 10)]))
    <POLYGON ((11.864 5.5, 11.742 4.258, 11.38 3.065, 10.791 1.9...>
    >>> shapely.minimum_bounding_circle(MultiPoint([(2, 2), (4, 2)]))
    <POLYGON ((4 2, 3.981 1.805, 3.924 1.617, 3.831 1.444, 3.707...>
    >>> shapely.minimum_bounding_circle(Point(0, 1))
    <POINT (0 1)>
    >>> shapely.minimum_bounding_circle(GeometryCollection([]))
    <POLYGON EMPTY>

    See Also
    --------
    minimum_bounding_radius, maximum_inscribed_circle

    """
    return lib.minimum_bounding_circle(geometry, **kwargs)


@multithreading_enabled
def maximum_inscribed_circle(geometry, tolerance=None, **kwargs):
    """Find the largest circle that is fully contained within the input geometry.

    Constructs the "maximum inscribed circle" (MIC) for a polygonal geometry,
    up to a specified tolerance. The MIC is determined by a point in the
    interior of the area which has the farthest distance from the area
    boundary, along with a boundary point at that distance. In the context of
    geography the center of the MIC is known as the "pole of inaccessibility".
    A cartographic use case is to determine a suitable point to place a map
    label within a polygon.
    The radius length of the MIC is a  measure of how "narrow" a polygon is.
    It is the distance at which the negative buffer becomes empty.

    The function supports polygons with holes and multipolygons.

    Returns a two-point linestring, with the first point at the center of the
    inscribed circle and the second on the boundary of the inscribed circle.

    .. versionadded:: 2.1.0

    Parameters
    ----------
    geometry : Geometry or array_like
    tolerance : float or array_like, optional
        Stop the algorithm when the search area is smaller than this tolerance.
        When not specified, uses `max(width, height) / 1000` per geometry as
        the default.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> import shapely
    >>> from shapely import Polygon
    >>> poly = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
    >>> shapely.maximum_inscribed_circle(poly)
    <LINESTRING (5 5, 0 5)>

    See Also
    --------
    minimum_bounding_circle
    """
    if tolerance is None:
        tolerance = 0.0
    elif np.isscalar(tolerance) and tolerance < 0:
        raise ValueError("'tolerance' should be positive")
    return lib.maximum_inscribed_circle(geometry, tolerance, **kwargs)


@multithreading_enabled
def _orient_polygons_geos(geometry, exterior_cw=False, **kwargs):
    return lib.orient_polygons(geometry, exterior_cw, **kwargs)


@multithreading_enabled
def orient_polygons(geometry, *, exterior_cw=False, **kwargs):
    """Enforce a ring orientation on all polygonal elements in the input geometry.

    Forces (Multi)Polygons to use a counter-clockwise orientation for their
    exterior ring, and a clockwise orientation for their interior rings (or
    the oppposite if ``exterior_cw=True``).

    Also processes geometries inside a GeometryCollection in the same way.
    Other geometries are returned unchanged.

    .. versionadded:: 2.1.0

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries to orient consistently.
    exterior_cw : bool, default False
        If True, exterior rings will be clockwise and interior rings
        will be counter-clockwise.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    A polygon with both shell and hole having clockwise orientation:

    >>> from shapely import Polygon, orient_polygons
    >>> polygon = Polygon(
    ...     [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
    ...     holes=[[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]],
    ... )
    >>> polygon
    <POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0), (2 2, 2 4, 4 4, 4 2, 2 2))>

    By default, the exterior ring is oriented counter-clockwise and
    the holes clockwise:

    >>> orient_polygons(polygon)
    <POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 2 4, 4 4, 4 2, 2 2))>

    Asking for the opposite orientation:

    >>> orient_polygons(polygon, exterior_cw=True)
    <POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))>

    """
    if lib.geos_version < (3, 12, 0):
        f = _orient_polygons_vectorized
    else:
        f = _orient_polygons_geos
    return f(geometry, exterior_cw, **kwargs)
