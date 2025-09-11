import numpy as np
import pytest

import shapely
from shapely import (
    Geometry,
    GeometryCollection,
    GEOSException,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    geos_version,
)
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
    ArrayLike,
    all_types,
    empty,
    empty_line_string,
    empty_point,
    empty_polygon,
    ignore_invalid,
    line_string,
    multi_point,
    point,
    point_z,
)

geos39 = geos_version[0:2] == (3, 9)
geos310 = geos_version[0:2] == (3, 10)
geos311 = geos_version[0:2] == (3, 11)

CONSTRUCTIVE_NO_ARGS = (
    shapely.boundary,
    shapely.centroid,
    shapely.convex_hull,
    pytest.param(
        shapely.concave_hull,
        marks=pytest.mark.skipif(
            shapely.geos_version < (3, 11, 0), reason="GEOS < 3.11"
        ),
    ),
    shapely.envelope,
    shapely.extract_unique_points,
    shapely.minimum_clearance_line,
    shapely.node,
    shapely.normalize,
    shapely.point_on_surface,
    pytest.param(
        shapely.constrained_delaunay_triangles,
        marks=pytest.mark.skipif(
            shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10"
        ),
    ),
)

CONSTRUCTIVE_FLOAT_ARG = (
    shapely.buffer,
    shapely.offset_curve,
    shapely.delaunay_triangles,
    shapely.simplify,
    shapely.voronoi_polygons,
)


@pytest.mark.parametrize("geometry", all_types)
@pytest.mark.parametrize("func", CONSTRUCTIVE_NO_ARGS)
def test_no_args_array(geometry, func):
    if (
        geometry.is_empty
        and shapely.get_num_geometries(geometry) > 0
        and func is shapely.node
        and (
            (geos39 and geos_version < (3, 9, 3))
            or (geos310 and geos_version < (3, 10, 3))
        )
    ):  # GEOS GH-601
        pytest.xfail("GEOS < 3.9.3 or GEOS < 3.10.3 crashes with empty geometries")
    actual = func([geometry, geometry])
    assert actual.shape == (2,)
    assert actual[0] is None or isinstance(actual[0], Geometry)


@pytest.mark.parametrize("geometry", all_types)
@pytest.mark.parametrize("func", CONSTRUCTIVE_FLOAT_ARG)
def test_float_arg_array(geometry, func):
    if (
        func is shapely.offset_curve
        and shapely.get_type_id(geometry) not in [1, 2]
        and shapely.geos_version < (3, 11, 0)
    ):
        with pytest.raises(GEOSException, match="only accept linestrings"):
            func([geometry, geometry], 0.0)
        return
    # voronoi_polygons emits an "invalid" warning when supplied with an empty
    # point (see https://github.com/libgeos/geos/issues/515)
    with ignore_invalid(
        func is shapely.voronoi_polygons
        and shapely.get_type_id(geometry) == 0
        and shapely.geos_version < (3, 12, 0)
    ):
        actual = func([geometry, geometry], 0.0)
    assert actual.shape == (2,)
    assert isinstance(actual[0], Geometry)


@pytest.mark.parametrize("geometry", all_types)
@pytest.mark.parametrize("reference", all_types)
def test_snap_array(geometry, reference):
    actual = shapely.snap([geometry, geometry], [reference, reference], tolerance=1.0)
    assert actual.shape == (2,)
    assert isinstance(actual[0], Geometry)


@pytest.mark.parametrize("func", CONSTRUCTIVE_NO_ARGS)
def test_no_args_missing(func):
    actual = func(None)
    assert actual is None


@pytest.mark.parametrize("func", CONSTRUCTIVE_FLOAT_ARG)
def test_float_arg_missing(func):
    actual = func(None, 1.0)
    assert actual is None


@pytest.mark.parametrize("geometry", all_types)
@pytest.mark.parametrize("func", CONSTRUCTIVE_FLOAT_ARG)
def test_float_arg_nan(geometry, func):
    actual = func(geometry, float("nan"))
    assert actual is None


def test_buffer_cap_style_invalid():
    with pytest.raises(ValueError, match="'invalid' is not a valid option"):
        shapely.buffer(point, 1, cap_style="invalid")


def test_buffer_join_style_invalid():
    with pytest.raises(ValueError, match="'invalid' is not a valid option"):
        shapely.buffer(point, 1, join_style="invalid")


def test_snap_none():
    actual = shapely.snap(None, point, tolerance=1.0)
    assert actual is None


@pytest.mark.parametrize("geometry", all_types)
def test_snap_nan_float(geometry):
    actual = shapely.snap(geometry, point, tolerance=np.nan)
    assert actual is None


def test_build_area_none():
    actual = shapely.build_area(None)
    assert actual is None


@pytest.mark.parametrize(
    "geom,expected",
    [
        (point, empty),  # a point has no area
        (line_string, empty),  # a line string has no area
        # geometry collection of two polygons are combined into one
        (
            GeometryCollection(
                [
                    Polygon([(0, 0), (0, 3), (3, 3), (3, 0), (0, 0)]),
                    Polygon([(1, 1), (2, 2), (1, 2), (1, 1)]),
                ]
            ),
            Polygon(
                [(0, 0), (0, 3), (3, 3), (3, 0), (0, 0)],
                holes=[[(1, 1), (2, 2), (1, 2), (1, 1)]],
            ),
        ),
        (empty, empty),
        ([empty], [empty]),
    ],
)
def test_build_area(geom, expected):
    actual = shapely.build_area(geom)
    assert actual is not expected
    assert actual == expected


def test_make_valid_none():
    actual = shapely.make_valid(None)
    assert actual is None


@pytest.mark.parametrize(
    "geom,expected",
    [
        (point, point),  # a valid geometry stays the same (but is copied)
        # an L shaped polygon without area is converted to a multilinestring
        (
            Polygon([(0, 0), (1, 1), (1, 2), (1, 1), (0, 0)]),
            MultiLineString([((1, 1), (1, 2)), ((0, 0), (1, 1))]),
        ),
        # a polygon with self-intersection (bowtie) is converted into polygons
        (
            Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)]),
            MultiPolygon(
                [
                    Polygon([(1, 1), (2, 2), (2, 0), (1, 1)]),
                    Polygon([(0, 0), (0, 2), (1, 1), (0, 0)]),
                ]
            ),
        ),
        (empty, empty),
        ([empty], [empty]),
    ],
)
def test_make_valid(geom, expected):
    actual = shapely.make_valid(geom)
    assert actual is not expected
    # normalize needed to handle variation in output across GEOS versions
    assert shapely.normalize(actual) == expected


@pytest.mark.parametrize(
    "geom,expected",
    [
        (all_types, all_types),
        # first polygon is valid, second polygon has self-intersection
        (
            [
                Polygon([(0, 0), (2, 2), (0, 2), (0, 0)]),
                Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)]),
            ],
            [
                Polygon([(0, 0), (2, 2), (0, 2), (0, 0)]),
                MultiPolygon(
                    [
                        Polygon([(1, 1), (0, 0), (0, 2), (1, 1)]),
                        Polygon([(1, 1), (2, 2), (2, 0), (1, 1)]),
                    ]
                ),
            ],
        ),
        ([point, None, empty], [point, None, empty]),
    ],
)
def test_make_valid_1d(geom, expected):
    actual = shapely.make_valid(geom)
    # normalize needed to handle variation in output across GEOS versions
    assert np.all(shapely.normalize(actual) == shapely.normalize(expected))


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize(
    "geom,expected",
    [
        (point, point),  # a valid geometry stays the same (but is copied)
        # an L shaped polygon without area is converted to a linestring
        (
            Polygon([(0, 0), (1, 1), (1, 2), (1, 1), (0, 0)]),
            LineString([(0, 0), (1, 1), (1, 2), (1, 1), (0, 0)]),
        ),
        # a polygon with self-intersection (bowtie) is converted into polygons
        (
            Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)]),
            MultiPolygon(
                [
                    Polygon([(1, 1), (2, 2), (2, 0), (1, 1)]),
                    Polygon([(0, 0), (0, 2), (1, 1), (0, 0)]),
                ]
            ),
        ),
        (empty, empty),
        ([empty], [empty]),
    ],
)
def test_make_valid_structure(geom, expected):
    actual = shapely.make_valid(geom, method="structure")
    assert actual is not expected
    # normalize needed to handle variation in output across GEOS versions
    assert shapely.normalize(actual) == expected


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize(
    "geom,expected",
    [
        (point, point),  # a valid geometry stays the same (but is copied)
        # an L shaped polygon without area is converted to Empty Polygon
        (
            Polygon([(0, 0), (1, 1), (1, 2), (1, 1), (0, 0)]),
            Polygon(),
        ),
        # a polygon with self-intersection (bowtie) is converted into polygons
        (
            Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)]),
            MultiPolygon(
                [
                    Polygon([(1, 1), (2, 2), (2, 0), (1, 1)]),
                    Polygon([(0, 0), (0, 2), (1, 1), (0, 0)]),
                ]
            ),
        ),
        (empty, empty),
        ([empty], [empty]),
    ],
)
def test_make_valid_structure_keep_collapsed_false(geom, expected):
    actual = shapely.make_valid(geom, method="structure", keep_collapsed=False)
    assert actual is not expected
    # normalize needed to handle variation in output across GEOS versions
    assert shapely.normalize(actual) == expected


@pytest.mark.skipif(shapely.geos_version >= (3, 10, 0), reason="GEOS >= 3.10")
def test_make_valid_structure_unsupported_geos():
    with pytest.raises(
        ValueError, match="The 'structure' method is only available in GEOS >= 3.10.0"
    ):
        _ = shapely.make_valid(Point(), method="structure")


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize(
    "method, keep_collapsed, error_type, error",
    [
        (
            np.array(["linework", "structure"]),
            True,
            TypeError,
            "method only accepts scalar values",
        ),
        (
            "linework",
            [True, False],
            TypeError,
            "keep_collapsed only accepts scalar values",
        ),
        ("unknown", True, ValueError, "Unknown method: unknown"),
        (
            "linework",
            False,
            ValueError,
            "The 'linework' method does not support 'keep_collapsed=False'",
        ),
    ],
)
def test_make_valid_invalid_params(method, keep_collapsed, error_type, error):
    with pytest.raises(error_type, match=error):
        _ = shapely.make_valid(Point(), method=method, keep_collapsed=keep_collapsed)


@pytest.mark.parametrize(
    "geom,expected",
    [
        (point, point),  # a point is always in normalized form
        # order coordinates of linestrings and parts of multi-linestring
        (
            MultiLineString([((1, 1), (0, 0)), ((1, 1), (1, 2))]),
            MultiLineString([((1, 1), (1, 2)), ((0, 0), (1, 1))]),
        ),
    ],
)
def test_normalize(geom, expected):
    actual = shapely.normalize(geom)
    assert actual == expected


def test_offset_curve_empty():
    with ignore_invalid(shapely.geos_version < (3, 12, 0)):
        # Empty geometries emit an "invalid" warning
        # (see https://github.com/libgeos/geos/issues/515)
        actual = shapely.offset_curve(empty_line_string, 2.0)
    assert shapely.is_empty(actual)


def test_offset_curve_distance_array():
    # check that kwargs are passed through
    result = shapely.offset_curve([line_string, line_string], [-2.0, -3.0])
    assert result[0] == shapely.offset_curve(line_string, -2.0)
    assert result[1] == shapely.offset_curve(line_string, -3.0)


def test_offset_curve_kwargs():
    # check that kwargs are passed through
    result1 = shapely.offset_curve(
        line_string, -2.0, quad_segs=2, join_style="mitre", mitre_limit=2.0
    )
    result2 = shapely.offset_curve(line_string, -2.0)
    assert result1 != result2


def test_offset_curve_non_scalar_kwargs():
    msg = "only accepts scalar values"
    with pytest.raises(TypeError, match=msg):
        shapely.offset_curve([line_string, line_string], 1, quad_segs=np.array([8, 9]))

    with pytest.raises(TypeError, match=msg):
        shapely.offset_curve(
            [line_string, line_string], 1, join_style=["round", "bevel"]
        )

    with pytest.raises(TypeError, match=msg):
        shapely.offset_curve([line_string, line_string], 1, mitre_limit=[5.0, 6.0])


def test_offset_curve_join_style_invalid():
    with pytest.raises(ValueError, match="'invalid' is not a valid option"):
        shapely.offset_curve(line_string, 1.0, join_style="invalid")


@pytest.mark.skipif(shapely.geos_version < (3, 11, 0), reason="GEOS < 3.11")
@pytest.mark.parametrize(
    "geom,expected",
    [
        (LineString([(0, 0), (0, 0), (1, 0)]), LineString([(0, 0), (1, 0)])),
        (
            LinearRing([(0, 0), (1, 2), (1, 2), (1, 3), (0, 0)]),
            LinearRing([(0, 0), (1, 2), (1, 3), (0, 0)]),
        ),
        (
            Polygon([(0, 0), (0, 0), (1, 0), (1, 1), (1, 0), (0, 0)]),
            Polygon([(0, 0), (1, 0), (1, 1), (1, 0), (0, 0)]),
        ),
        (
            Polygon(
                [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
                holes=[[(2, 2), (2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]],
            ),
            Polygon(
                [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
                holes=[[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]],
            ),
        ),
        (
            MultiPolygon(
                [
                    Polygon([(0, 0), (0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
                    Polygon([(2, 2), (2, 2), (2, 3), (3, 3), (3, 2), (2, 2)]),
                ]
            ),
            MultiPolygon(
                [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
                    Polygon([(2, 2), (2, 3), (3, 3), (3, 2), (2, 2)]),
                ]
            ),
        ),
        # points are unchanged
        (point, point),
        (point_z, point_z),
        (multi_point, multi_point),
        # empty geometries are unchanged
        (empty_point, empty_point),
        (empty_line_string, empty_line_string),
        (empty, empty),
        (empty_polygon, empty_polygon),
    ],
)
def test_remove_repeated_points(geom, expected):
    assert_geometries_equal(shapely.remove_repeated_points(geom, 0), expected)


@pytest.mark.skipif(shapely.geos_version < (3, 12, 0), reason="GEOS < 3.12")
@pytest.mark.parametrize(
    "geom, tolerance", [[Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]), 2]]
)
def test_remove_repeated_points_invalid_result(geom, tolerance):
    # Requiring GEOS 3.12 instead of 3.11
    # (GEOS 3.11 had a bug causing this to intermittently not fail)
    with pytest.raises(shapely.GEOSException, match="Invalid number of points"):
        shapely.remove_repeated_points(geom, tolerance)


@pytest.mark.skipif(shapely.geos_version < (3, 11, 0), reason="GEOS < 3.11")
def test_remove_repeated_points_none():
    assert shapely.remove_repeated_points(None, 1) is None
    assert shapely.remove_repeated_points([None], 1).tolist() == [None]

    geometry = LineString([(0, 0), (0, 0), (1, 1)])
    expected = LineString([(0, 0), (1, 1)])
    result = shapely.remove_repeated_points([None, geometry], 1)
    assert result[0] is None
    assert_geometries_equal(result[1], expected)


@pytest.mark.skipif(shapely.geos_version < (3, 11, 0), reason="GEOS < 3.11")
@pytest.mark.parametrize("geom, tolerance", [("Not a geometry", 1), (1, 1)])
def test_remove_repeated_points_invalid_type(geom, tolerance):
    with pytest.raises(TypeError, match="One of the arguments is of incorrect type"):
        shapely.remove_repeated_points(geom, tolerance)


@pytest.mark.parametrize(
    "geom,expected",
    [
        (LineString([(0, 0), (1, 2)]), LineString([(1, 2), (0, 0)])),
        (
            LinearRing([(0, 0), (1, 2), (1, 3), (0, 0)]),
            LinearRing([(0, 0), (1, 3), (1, 2), (0, 0)]),
        ),
        (
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),
        ),
        (
            Polygon(
                [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
                holes=[[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]],
            ),
            Polygon(
                [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
                holes=[[(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)]],
            ),
        ),
        (
            MultiLineString([[(0, 0), (1, 2)], [(3, 3), (4, 4)]]),
            MultiLineString([[(1, 2), (0, 0)], [(4, 4), (3, 3)]]),
        ),
        (
            MultiPolygon(
                [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
                    Polygon([(2, 2), (2, 3), (3, 3), (3, 2), (2, 2)]),
                ]
            ),
            MultiPolygon(
                [
                    Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),
                    Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)]),
                ]
            ),
        ),
        # points are unchanged
        (point, point),
        (point_z, point_z),
        (multi_point, multi_point),
        # empty geometries are unchanged
        (empty_point, empty_point),
        (empty_line_string, empty_line_string),
        (empty, empty),
        (empty_polygon, empty_polygon),
    ],
)
def test_reverse(geom, expected):
    assert_geometries_equal(shapely.reverse(geom), expected)


def test_reverse_none():
    assert shapely.reverse(None) is None
    assert shapely.reverse([None]).tolist() == [None]

    geometry = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    expected = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    result = shapely.reverse([None, geometry])
    assert result[0] is None
    assert_geometries_equal(result[1], expected)


@pytest.mark.parametrize("geom", ["Not a geometry", 1])
def test_reverse_invalid_type(geom):
    with pytest.raises(TypeError, match="One of the arguments is of incorrect type"):
        shapely.reverse(geom)


@pytest.mark.parametrize(
    "geom,expected",
    [
        # Point outside
        (Point(0, 0), GeometryCollection()),
        # Point inside
        (Point(15, 15), Point(15, 15)),
        # Point on boundary
        (Point(15, 10), GeometryCollection()),
        # Line outside
        (LineString([(0, 0), (-5, 5)]), GeometryCollection()),
        # Line inside
        (LineString([(15, 15), (16, 15)]), LineString([(15, 15), (16, 15)])),
        # Line on boundary
        (LineString([(10, 15), (10, 10), (15, 10)]), GeometryCollection()),
        # Line splitting rectangle
        (LineString([(10, 5), (25, 20)]), LineString([(15, 10), (20, 15)])),
    ],
)
def test_clip_by_rect(geom, expected):
    actual = shapely.clip_by_rect(geom, 10, 10, 20, 20)
    assert_geometries_equal(actual, expected)


@pytest.mark.parametrize(
    "geom, rect, expected",
    [
        # Polygon hole (CCW) fully on rectangle boundary"""
        (
            Polygon(
                ((0, 0), (0, 30), (30, 30), (30, 0), (0, 0)),
                holes=[((10, 10), (20, 10), (20, 20), (10, 20), (10, 10))],
            ),
            (10, 10, 20, 20),
            GeometryCollection(),
        ),
        # Polygon hole (CW) fully on rectangle boundary"""
        (
            Polygon(
                ((0, 0), (0, 30), (30, 30), (30, 0), (0, 0)),
                holes=[((10, 10), (10, 20), (20, 20), (20, 10), (10, 10))],
            ),
            (10, 10, 20, 20),
            GeometryCollection(),
        ),
        # Polygon fully within rectangle"""
        (
            Polygon(
                ((1, 1), (1, 30), (30, 30), (30, 1), (1, 1)),
                holes=[((10, 10), (20, 10), (20, 20), (10, 20), (10, 10))],
            ),
            (0, 0, 40, 40),
            Polygon(
                ((1, 1), (1, 30), (30, 30), (30, 1), (1, 1)),
                holes=[((10, 10), (20, 10), (20, 20), (10, 20), (10, 10))],
            ),
        ),
        # Polygon overlapping rectanglez
        (
            Polygon(
                [(0, 0), (0, 30), (30, 30), (30, 0), (0, 0)],
                holes=[[(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)]],
            ),
            (5, 5, 15, 15),
            Polygon([(5, 5), (5, 15), (10, 15), (10, 10), (15, 10), (15, 5), (5, 5)]),
        ),
    ],
)
def test_clip_by_rect_polygon(geom, rect, expected):
    actual = shapely.clip_by_rect(geom, *rect)
    assert_geometries_equal(actual, expected)


@pytest.mark.parametrize("geometry", all_types)
def test_clip_by_rect_array(geometry):
    if (
        geometry.is_empty
        and shapely.get_type_id(geometry) == shapely.GeometryType.POINT
        and (
            (geos39 and geos_version < (3, 9, 5))
            or (geos310 and geos_version < (3, 10, 6))
            or (geos311 and geos_version < (3, 11, 3))
        )
    ):
        # GEOS GH-913
        with pytest.raises(GEOSException):
            shapely.clip_by_rect([geometry, geometry], 0.0, 0.0, 1.0, 1.0)
        return
    actual = shapely.clip_by_rect([geometry, geometry], 0.0, 0.0, 1.0, 1.0)
    assert actual.shape == (2,)
    assert actual[0] is None or isinstance(actual[0], Geometry)


def test_clip_by_rect_missing():
    actual = shapely.clip_by_rect(None, 0, 0, 1, 1)
    assert actual is None


@pytest.mark.parametrize("geom", [empty, empty_line_string, empty_polygon])
def test_clip_by_rect_empty(geom):
    # TODO empty point
    actual = shapely.clip_by_rect(geom, 0, 0, 1, 1)
    assert actual == GeometryCollection()


def test_clip_by_rect_non_scalar_kwargs():
    msg = "only accepts scalar values"
    with pytest.raises(TypeError, match=msg):
        shapely.clip_by_rect([line_string, line_string], 0, 0, 1, np.array([0, 1]))


def test_polygonize():
    lines = [
        LineString([(0, 0), (1, 1)]),
        LineString([(0, 0), (0, 1)]),
        LineString([(0, 1), (1, 1)]),
        LineString([(1, 1), (1, 0)]),
        LineString([(1, 0), (0, 0)]),
        LineString([(5, 5), (6, 6)]),
        Point(0, 0),
        None,
    ]
    result = shapely.polygonize(lines)
    assert shapely.get_type_id(result) == 7  # GeometryCollection
    expected = GeometryCollection(
        [
            Polygon([(0, 0), (1, 1), (1, 0), (0, 0)]),
            Polygon([(1, 1), (0, 0), (0, 1), (1, 1)]),
        ]
    )
    assert result == expected


def test_polygonize_array():
    lines = [
        LineString([(0, 0), (1, 1)]),
        LineString([(0, 0), (0, 1)]),
        LineString([(0, 1), (1, 1)]),
    ]
    expected = GeometryCollection([Polygon([(1, 1), (0, 0), (0, 1), (1, 1)])])
    result = shapely.polygonize(np.array(lines))
    assert isinstance(result, shapely.Geometry)
    assert result == expected

    result = shapely.polygonize(np.array([lines]))
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    assert result[0] == expected

    arr = np.array([lines, lines])
    assert arr.shape == (2, 3)
    result = shapely.polygonize(arr)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    assert result[0] == expected
    assert result[1] == expected

    arr = np.array([[lines, lines], [lines, lines], [lines, lines]])
    assert arr.shape == (3, 2, 3)
    result = shapely.polygonize(arr)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 2)
    for res in result.flatten():
        assert res == expected


def test_polygonize_array_axis():
    lines = [
        LineString([(0, 0), (1, 1)]),
        LineString([(0, 0), (0, 1)]),
        LineString([(0, 1), (1, 1)]),
    ]
    arr = np.array([lines, lines])  # shape (2, 3)
    result = shapely.polygonize(arr, axis=1)
    assert result.shape == (2,)
    result = shapely.polygonize(arr, axis=0)
    assert result.shape == (3,)


def test_polygonize_missing():
    # set of geometries that is all missing
    result = shapely.polygonize([None, None])
    assert result == GeometryCollection()


def test_polygonize_full():
    lines = [
        None,
        LineString([(0, 0), (1, 1)]),
        LineString([(0, 0), (0, 1)]),
        LineString([(0, 1), (1, 1)]),
        LineString([(1, 1), (1, 0)]),
        None,
        LineString([(1, 0), (0, 0)]),
        LineString([(5, 5), (6, 6)]),
        LineString([(1, 1), (100, 100)]),
        Point(0, 0),
        None,
    ]
    result = shapely.polygonize_full(lines)
    assert len(result) == 4
    assert all(shapely.get_type_id(geom) == 7 for geom in result)  # GeometryCollection
    polygons, cuts, dangles, invalid = result
    expected_polygons = GeometryCollection(
        [
            Polygon([(0, 0), (1, 1), (1, 0), (0, 0)]),
            Polygon([(1, 1), (0, 0), (0, 1), (1, 1)]),
        ]
    )
    assert polygons == expected_polygons
    assert cuts == GeometryCollection()
    expected_dangles = GeometryCollection(
        [LineString([(1, 1), (100, 100)]), LineString([(5, 5), (6, 6)])]
    )
    assert dangles == expected_dangles
    assert invalid == GeometryCollection()


def test_polygonize_full_array():
    lines = [
        LineString([(0, 0), (1, 1)]),
        LineString([(0, 0), (0, 1)]),
        LineString([(0, 1), (1, 1)]),
    ]
    expected = GeometryCollection([Polygon([(1, 1), (0, 0), (0, 1), (1, 1)])])
    result = shapely.polygonize_full(np.array(lines))
    assert len(result) == 4
    assert all(isinstance(geom, shapely.Geometry) for geom in result)
    assert result[0] == expected
    assert all(geom == GeometryCollection() for geom in result[1:])

    result = shapely.polygonize_full(np.array([lines]))
    assert len(result) == 4
    assert all(isinstance(geom, np.ndarray) for geom in result)
    assert all(geom.shape == (1,) for geom in result)
    assert result[0][0] == expected
    assert all(geom[0] == GeometryCollection() for geom in result[1:])

    arr = np.array([lines, lines])
    assert arr.shape == (2, 3)
    result = shapely.polygonize_full(arr)
    assert len(result) == 4
    assert all(isinstance(arr, np.ndarray) for arr in result)
    assert all(arr.shape == (2,) for arr in result)
    assert result[0][0] == expected
    assert result[0][1] == expected
    assert all(g == GeometryCollection() for geom in result[1:] for g in geom)

    arr = np.array([[lines, lines], [lines, lines], [lines, lines]])
    assert arr.shape == (3, 2, 3)
    result = shapely.polygonize_full(arr)
    assert len(result) == 4
    assert all(isinstance(arr, np.ndarray) for arr in result)
    assert all(arr.shape == (3, 2) for arr in result)
    for res in result[0].flatten():
        assert res == expected
    for arr in result[1:]:
        for res in arr.flatten():
            assert res == GeometryCollection()


def test_polygonize_full_array_axis():
    lines = [
        LineString([(0, 0), (1, 1)]),
        LineString([(0, 0), (0, 1)]),
        LineString([(0, 1), (1, 1)]),
    ]
    arr = np.array([lines, lines])  # shape (2, 3)
    result = shapely.polygonize_full(arr, axis=1)
    assert len(result) == 4
    assert all(arr.shape == (2,) for arr in result)
    result = shapely.polygonize_full(arr, axis=0)
    assert len(result) == 4
    assert all(arr.shape == (3,) for arr in result)


def test_polygonize_full_missing():
    # set of geometries that is all missing
    result = shapely.polygonize_full([None, None])
    assert len(result) == 4
    assert all(geom == GeometryCollection() for geom in result)


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize("geometry", all_types)
@pytest.mark.parametrize("max_segment_length", [-1, 0])
def test_segmentize_invalid_max_segment_length(geometry, max_segment_length):
    with pytest.raises(GEOSException, match="IllegalArgumentException"):
        shapely.segmentize(geometry, max_segment_length=max_segment_length)


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize("geometry", all_types)
def test_segmentize_max_segment_length_nan(geometry):
    actual = shapely.segmentize(geometry, max_segment_length=np.nan)
    assert actual is None


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize(
    "geometry", [empty, empty_point, empty_line_string, empty_polygon]
)
def test_segmentize_empty(geometry):
    actual = shapely.segmentize(geometry, max_segment_length=5)
    assert_geometries_equal(actual, geometry)


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize("geometry", [point, point_z, multi_point])
def test_segmentize_no_change(geometry):
    actual = shapely.segmentize(geometry, max_segment_length=5)
    assert_geometries_equal(actual, geometry)


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
def test_segmentize_none():
    assert shapely.segmentize(None, max_segment_length=5) is None


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize(
    "geometry,tolerance, expected",
    [
        # tolerance greater than max edge length, no change
        (
            LineString([(0, 0), (0, 10)]),
            20,
            LineString([(0, 0), (0, 10)]),
        ),
        (
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]),
            20,
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]),
        ),
        # tolerance causes one vertex per segment
        (
            LineString([(0, 0), (0, 10)]),
            5,
            LineString([(0, 0), (0, 5), (0, 10)]),
        ),
        (
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]),
            5,
            Polygon(
                [
                    (0, 0),
                    (5, 0),
                    (10, 0),
                    (10, 5),
                    (10, 10),
                    (5, 10),
                    (0, 10),
                    (0, 5),
                    (0, 0),
                ]
            ),
        ),
        # ensure input arrays are broadcast correctly
        (
            [
                LineString([(0, 0), (0, 10)]),
                LineString([(0, 0), (0, 2)]),
            ],
            5,
            [
                LineString([(0, 0), (0, 5), (0, 10)]),
                LineString([(0, 0), (0, 2)]),
            ],
        ),
        (
            [
                LineString([(0, 0), (0, 10)]),
                LineString([(0, 0), (0, 2)]),
            ],
            [5],
            [
                LineString([(0, 0), (0, 5), (0, 10)]),
                LineString([(0, 0), (0, 2)]),
            ],
        ),
        (
            [
                LineString([(0, 0), (0, 10)]),
                LineString([(0, 0), (0, 2)]),
            ],
            [5, 1.5],
            [
                LineString([(0, 0), (0, 5), (0, 10)]),
                LineString([(0, 0), (0, 1), (0, 2)]),
            ],
        ),
    ],
)
def test_segmentize(geometry, tolerance, expected):
    actual = shapely.segmentize(geometry, tolerance)
    assert_geometries_equal(actual, expected)


@pytest.mark.parametrize("geometry", all_types)
def test_minimum_bounding_circle_all_types(geometry):
    actual = shapely.minimum_bounding_circle([geometry, geometry])
    assert actual.shape == (2,)
    assert actual[0] is None or isinstance(actual[0], Geometry)

    actual = shapely.minimum_bounding_circle(None)
    assert actual is None


@pytest.mark.parametrize(
    "geometry, expected",
    [
        (
            Polygon([(0, 5), (5, 10), (10, 5), (5, 0), (0, 5)]),
            shapely.buffer(Point(5, 5), 5),
        ),
        (
            LineString([(1, 0), (1, 10)]),
            shapely.buffer(Point(1, 5), 5),
        ),
        (
            MultiPoint([(2, 2), (4, 2)]),
            shapely.buffer(Point(3, 2), 1),
        ),
        (
            Point(2, 2),
            Point(2, 2),
        ),
        (
            GeometryCollection(),
            Polygon(),
        ),
    ],
)
def test_minimum_bounding_circle(geometry, expected):
    actual = shapely.minimum_bounding_circle(geometry)
    assert_geometries_equal(actual, expected)


@pytest.mark.parametrize("geometry", all_types)
def test_oriented_envelope_all_types(geometry):
    actual = shapely.oriented_envelope([geometry, geometry])
    assert actual.shape == (2,)
    assert actual[0] is None or isinstance(actual[0], Geometry)

    actual = shapely.oriented_envelope(None)
    assert actual is None


@pytest.mark.parametrize(
    "func", [shapely.oriented_envelope, shapely.minimum_rotated_rectangle]
)
@pytest.mark.parametrize(
    "geometry, expected",
    [
        (
            MultiPoint([(1.0, 1.0), (1.0, 5.0), (3.0, 6.0), (4.0, 2.0), (5.0, 5.0)]),
            Polygon([(1.0, 1.0), (1.0, 6.0), (5.0, 6.0), (5.0, 1.0), (1.0, 1.0)]),
        ),
        (
            LineString([(1, 1), (5, 1), (10, 10)]),
            Polygon([(1, 1), (3, -1), (12, 8), (10, 10), (1, 1)]),
        ),
        (
            Polygon([(1, 1), (15, 1), (5, 9), (1, 1)]),
            Polygon([(1.0, 1.0), (5.0, 9.0), (16.2, 3.4), (12.2, -4.6), (1.0, 1.0)]),
        ),
        (
            LineString([(1, 1), (10, 1)]),
            LineString([(1, 1), (10, 1)]),
        ),
        (
            Point(2, 2),
            Point(2, 2),
        ),
        (
            GeometryCollection(),
            Polygon(),
        ),
    ],
)
def test_oriented_envelope(geometry, expected, func):
    actual = func(geometry)
    assert_geometries_equal(actual, expected, normalize=True, tolerance=1e-3)


@pytest.mark.skipif(shapely.geos_version >= (3, 12, 0), reason="GEOS >= 3.12")
@pytest.mark.parametrize(
    "geometry, expected",
    [
        (
            MultiPoint([(1.0, 1.0), (1.0, 5.0), (3.0, 6.0), (4.0, 2.0), (5.0, 5.0)]),
            Polygon([(-0.2, 1.4), (1.5, 6.5), (5.1, 5.3), (3.4, 0.2), (-0.2, 1.4)]),
        ),
        (
            LineString([(1, 1), (5, 1), (10, 10)]),
            Polygon([(1, 1), (3, -1), (12, 8), (10, 10), (1, 1)]),
        ),
        (
            Polygon([(1, 1), (15, 1), (5, 9), (1, 1)]),
            Polygon([(1.0, 1.0), (1.0, 9.0), (15.0, 9.0), (15.0, 1.0), (1.0, 1.0)]),
        ),
        (
            LineString([(1, 1), (10, 1)]),
            LineString([(1, 1), (10, 1)]),
        ),
        (
            Point(2, 2),
            Point(2, 2),
        ),
        (
            GeometryCollection(),
            Polygon(),
        ),
    ],
)
def test_oriented_envelope_pre_geos_312(geometry, expected):
    # use private method (similar as direct shapely.lib.oriented_envelope)
    # to cover the C code for older GEOS versions
    actual = shapely.constructive._oriented_envelope_geos(geometry)
    assert_geometries_equal(actual, expected, normalize=True, tolerance=1e-3)


def test_oriented_evelope_array_like():
    # https://github.com/shapely/shapely/issues/1929
    # because we have a custom python implementation, need to ensure this has
    # the same capabilities as numpy ufuncs to work with array-likes
    geometries = [Point(1, 1).buffer(1), Point(2, 2).buffer(1)]
    actual = shapely.oriented_envelope(ArrayLike(geometries))
    assert isinstance(actual, ArrayLike)
    expected = shapely.oriented_envelope(geometries)
    assert_geometries_equal(np.asarray(actual), expected)


@pytest.mark.skipif(shapely.geos_version < (3, 11, 0), reason="GEOS < 3.11")
def test_concave_hull_kwargs():
    p = Point(10, 10)
    mp = MultiPoint(p.buffer(5).exterior.coords[:] + p.buffer(4).exterior.coords[:])

    result1 = shapely.concave_hull(mp, ratio=0.5)
    assert len(result1.interiors) == 0
    result2 = shapely.concave_hull(mp, ratio=0.5, allow_holes=True)
    assert len(result2.interiors) == 1

    result3 = shapely.concave_hull(mp, ratio=0)
    result4 = shapely.concave_hull(mp, ratio=1)
    assert shapely.get_num_coordinates(result4) < shapely.get_num_coordinates(result3)


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
class TestConstrainedDelaunayTriangulation:
    """
    Only testing the number of triangles and their type here.
    This doesn't actually test the points in the resulting geometries.

    """

    def test_poly(self):
        polys = shapely.constrained_delaunay_triangles(
            Polygon([(10, 10), (20, 40), (90, 90), (90, 10), (10, 10)])
        )
        assert len(polys.geoms) == 2
        for p in polys.geoms:
            assert isinstance(p, Polygon)

    def test_multi_polygon(self):
        multipoly = MultiPolygon(
            [
                Polygon(((50, 30), (60, 30), (100, 100), (50, 30))),
                Polygon(((10, 10), (20, 40), (90, 90), (90, 10), (10, 10))),
            ]
        )
        polys = shapely.constrained_delaunay_triangles(multipoly)
        assert len(polys.geoms) == 3
        for p in polys.geoms:
            assert isinstance(p, Polygon)

    def test_point(self):
        p = Point(1, 1)
        polys = shapely.constrained_delaunay_triangles(p)
        assert len(polys.geoms) == 0

    def test_empty_poly(self):
        polys = shapely.constrained_delaunay_triangles(Polygon())
        assert len(polys.geoms) == 0


@pytest.mark.skipif(shapely.geos_version < (3, 12, 0), reason="GEOS < 3.12")
def test_voronoi_polygons_ordered():
    mp = MultiPoint([(3.0, 1.0), (3.0, 2.0), (1.0, 2.0), (1.0, 1.0)])
    result = shapely.voronoi_polygons(mp, ordered=False)
    assert result.geoms[0].equals(
        Polygon([(-1, -1), (-1, 1.5), (2, 1.5), (2, -1), (-1, -1)])
    )

    result_ordered = shapely.voronoi_polygons(mp, ordered=True)
    assert result_ordered.geoms[0].equals(
        Polygon([(5, -1), (2, -1), (2, 1.5), (5, 1.5), (5, -1)])
    )


@pytest.mark.skipif(shapely.geos_version >= (3, 12, 0), reason="GEOS >= 3.12")
def test_voronoi_polygons_ordered_raise():
    mp = MultiPoint([(3.0, 1.0), (3.0, 2.0), (1.0, 2.0), (1.0, 1.0)])
    with pytest.raises(
        UnsupportedGEOSVersionError, match="Ordered Voronoi polygons require GEOS"
    ):
        shapely.voronoi_polygons(mp, ordered=True)


@pytest.mark.parametrize("geometry", all_types)
def test_maximum_inscribed_circle_all_types(geometry):
    if shapely.get_type_id(geometry) not in [3, 6]:
        # Maximum Inscribed Circle is only supported for (Multi)Polygon input
        with pytest.raises(
            GEOSException,
            match=(
                "Argument must be Polygonal or LinearRing|"  # GEOS < 3.10.4
                "must be a Polygon or MultiPolygon|"
                "Operation not supported by GeometryCollection"
            ),
        ):
            shapely.maximum_inscribed_circle(geometry)
        return

    if geometry.is_empty:
        with pytest.raises(
            GEOSException, match="Empty input(?: geometry)? is not supported"
        ):
            shapely.maximum_inscribed_circle(geometry)
        return

    actual = shapely.maximum_inscribed_circle([geometry, geometry])
    assert actual.shape == (2,)
    assert actual[0] is None or isinstance(actual[0], Geometry)

    actual = shapely.maximum_inscribed_circle(None)
    assert actual is None


@pytest.mark.parametrize(
    "geometry, expected",
    [
        (
            "POLYGON ((0 5, 5 10, 10 5, 5 0, 0 5))",
            "LINESTRING (5 5, 2.5 7.5)",
        ),
    ],
)
def test_maximum_inscribed_circle(geometry, expected):
    geometry, expected = shapely.from_wkt(geometry), shapely.from_wkt(expected)
    actual = shapely.maximum_inscribed_circle(geometry)
    assert_geometries_equal(actual, expected)


def test_maximum_inscribed_circle_empty():
    geometry = shapely.from_wkt("POINT EMPTY")
    with pytest.raises(
        GEOSException,
        match=(
            "Argument must be Polygonal or LinearRing|"  # GEOS < 3.10.4
            "must be a Polygon or MultiPolygon"
        ),
    ):
        shapely.maximum_inscribed_circle(geometry)

    geometry = shapely.from_wkt("POLYGON EMPTY")
    with pytest.raises(
        GEOSException, match="Empty input(?: geometry)? is not supported"
    ):
        shapely.maximum_inscribed_circle(geometry)


def test_maximum_inscribed_circle_invalid_tolerance():
    geometry = shapely.from_wkt("POLYGON ((0 5, 5 10, 10 5, 5 0, 0 5))")
    with pytest.raises(ValueError, match="'tolerance' should be positive"):
        shapely.maximum_inscribed_circle(geometry, tolerance=-1)


@pytest.mark.parametrize("geometry", all_types)
def test_orient_polygons_all_types(geometry):
    actual = shapely.orient_polygons([geometry, geometry])
    assert actual.shape == (2,)
    assert isinstance(actual[0], Geometry)

    actual = shapely.orient_polygons(None)
    assert actual is None


def test_orient_polygons():
    # polygon with both shell and hole having clockwise orientation
    polygon = Polygon(
        [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
        holes=[[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]],
    )

    result = shapely.orient_polygons(polygon)
    assert result.exterior.is_ccw
    assert not result.interiors[0].is_ccw

    result = shapely.orient_polygons(polygon, exterior_cw=True)
    assert not result.exterior.is_ccw
    assert result.interiors[0].is_ccw

    # in a MultiPolygon
    mp = MultiPolygon([polygon, polygon])
    result = shapely.orient_polygons(mp)
    assert len(result.geoms) == 2
    for geom in result.geoms:
        assert geom.exterior.is_ccw
        assert not geom.interiors[0].is_ccw

    result = shapely.orient_polygons([mp], exterior_cw=True)[0]
    assert len(result.geoms) == 2
    for geom in result.geoms:
        assert not geom.exterior.is_ccw
        assert geom.interiors[0].is_ccw

    # in a GeometryCollection
    gc = GeometryCollection([Point(1, 1), polygon, mp])
    result = shapely.orient_polygons(gc)
    assert len(result.geoms) == 3
    assert result.geoms[0] == Point(1, 1)
    assert result.geoms[1] == shapely.orient_polygons(polygon)
    assert result.geoms[2] == shapely.orient_polygons(mp)


def test_orient_polygons_non_polygonal_input():
    arr = np.array([Point(0, 0), LineString([(0, 0), (1, 1)]), None])
    result = shapely.orient_polygons(arr)
    assert_geometries_equal(result, arr)


def test_orient_polygons_array():
    # because we have a custom python implementation for older GEOS, need to
    # ensure this has the same capabilities as numpy ufuncs to work with array-likes
    polygon = Polygon(
        [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
        holes=[[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]],
    )
    geometries = np.array([[polygon] * 3] * 2)
    actual = shapely.orient_polygons(geometries)
    assert isinstance(actual, np.ndarray)
    assert actual.shape == (2, 3)
    expected = shapely.orient_polygons(polygon)
    assert (actual == expected).all()


def test_orient_polygons_array_like():
    # because we have a custom python implementation for older GEOS, need to
    # ensure this has the same capabilities as numpy ufuncs to work with array-likes
    polygon = Polygon(
        [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
        holes=[[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]],
    )
    geometries = [polygon, Point(2, 2).buffer(1)]
    actual = shapely.orient_polygons(ArrayLike(geometries))
    assert isinstance(actual, ArrayLike)
    expected = shapely.orient_polygons(geometries)
    assert_geometries_equal(np.asarray(actual), expected)


def test_buffer_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `quad_segs` for `buffer` is deprecated"
    ):
        shapely.buffer(point, 1.0, 8)
    with pytest.deprecated_call(
        match="positional arguments `quad_segs` and `cap_style` "
        "for `buffer` are deprecated"
    ):
        shapely.buffer(point, 1.0, 8, "round")
    with pytest.deprecated_call(
        match="positional arguments `quad_segs`, `cap_style`, and `join_style` "
        "for `buffer` are deprecated"
    ):
        shapely.buffer(point, 1.0, 8, "round", "round")
    with pytest.deprecated_call():
        shapely.buffer(point, 1.0, 8, "round", "round", 5.0)
    with pytest.deprecated_call():
        shapely.buffer(point, 1.0, 8, "round", "round", 5.0, False)


def test_offset_curve_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `quad_segs` for `offset_curve` is deprecated"
    ):
        shapely.offset_curve(line_string, 1.0, 8)
    with pytest.deprecated_call(
        match="positional arguments `quad_segs` and `join_style` "
        "for `offset_curve` are deprecated"
    ):
        shapely.offset_curve(line_string, 1.0, 8, "round")
    with pytest.deprecated_call(
        match="positional arguments `quad_segs`, `join_style`, and `mitre_limit` "
        "for `offset_curve` are deprecated"
    ):
        shapely.offset_curve(line_string, 1.0, 8, "round", 5.0)


def test_simplify_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `preserve_topology` for `simplify` is deprecated"
    ):
        shapely.simplify(line_string, 1.0, True)


def test_voronoi_polygons_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `extend_to` for `voronoi_polygons` is deprecated"
    ):
        shapely.voronoi_polygons(multi_point, 0.0, None)
    with pytest.deprecated_call(
        match="positional arguments `extend_to` and `only_edges` "
        "for `voronoi_polygons` are deprecated"
    ):
        shapely.voronoi_polygons(multi_point, 0.0, None, False)
    with pytest.deprecated_call(
        match="positional arguments `extend_to`, `only_edges`, and `ordered` "
        "for `voronoi_polygons` are deprecated"
    ):
        shapely.voronoi_polygons(multi_point, 0.0, None, False, False)
