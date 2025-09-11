"""See test_creation_indices.py for tests with 'indices' parameter."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import shapely

# Note: Point is not imported because it is overridden for testing
from shapely import (
    GeometryCollection,
    GeometryType,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Polygon,
)
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
    empty_polygon,
    geometry_collection,
    ignore_invalid,
    line_string,
    linear_ring,
    multi_line_string,
    multi_point,
    multi_polygon,
    point,
    polygon,
)


def box_tpl(x1, y1, x2, y2):
    return (x2, y1), (x2, y2), (x1, y2), (x1, y1), (x2, y1)


def test_points_from_coords():
    actual = shapely.points([[0, 0], [2, 2]])
    assert_geometries_equal(actual, [shapely.Point(0, 0), shapely.Point(2, 2)])


def test_points_from_xy():
    actual = shapely.points(2, [0, 1])
    assert_geometries_equal(actual, [shapely.Point(2, 0), shapely.Point(2, 1)])


def test_points_from_xyz():
    actual = shapely.points(1, 1, [0, 1])
    assert_geometries_equal(actual, [shapely.Point(1, 1, 0), shapely.Point(1, 1, 1)])


def test_points_invalid_ndim():
    with pytest.raises(ValueError, match="dimension should be 2 or 3, got 4"):
        shapely.points([0, 1, 2, 3])

    with pytest.raises(ValueError, match="dimension should be 2 or 3, got 1"):
        shapely.points([0])


@pytest.mark.skipif(
    shapely.geos_version[:2] not in ((3, 10), (3, 11), (3, 12)),
    reason="GEOS not in 3.10, 3.11, 3.12",
)
def test_points_nan_all_nan_becomes_empty():
    actual = shapely.points(np.nan, np.nan)
    assert actual.wkt == "POINT EMPTY"


@pytest.mark.skipif(
    shapely.geos_version[:2] not in ((3, 10), (3, 11)),
    reason="GEOS not in 3.10, 3.11",
)
def test_points_nan_3D_all_nan_becomes_empty_2D():
    actual = shapely.points(np.nan, np.nan, np.nan)
    assert actual.wkt == "POINT EMPTY"


@pytest.mark.skipif(shapely.geos_version[:2] != (3, 12), reason="GEOS != 3.12")
def test_points_nan_3D_all_nan_becomes_empty():
    actual = shapely.points(np.nan, np.nan, np.nan)
    assert actual.wkt == "POINT Z EMPTY"


@pytest.mark.skipif(shapely.geos_version < (3, 12, 0), reason="GEOS < 3.12")
@pytest.mark.parametrize(
    "coords,expected_wkt",
    [
        pytest.param(
            [np.nan, np.nan],
            "POINT (NaN NaN)",
            marks=pytest.mark.skipif(
                shapely.geos_version < (3, 13, 0), reason="GEOS < 3.13"
            ),
        ),
        pytest.param(
            [np.nan, np.nan, np.nan],
            "POINT Z (NaN NaN NaN)",
            marks=pytest.mark.skipif(
                shapely.geos_version < (3, 13, 0), reason="GEOS < 3.13"
            ),
        ),
        ([1, np.nan], "POINT (1 NaN)"),
        ([np.nan, 1], "POINT (NaN 1)"),
        ([np.nan, 1, np.nan], "POINT Z (NaN 1 NaN)"),
        ([np.nan, np.nan, 1], "POINT Z (NaN NaN 1)"),
    ],
)
def test_points_handle_nan_allow(coords, expected_wkt):
    actual = shapely.points(coords, handle_nan="allow")
    assert actual.wkt == expected_wkt


@pytest.mark.parametrize(
    "coords,handle_nan,expected_wkt",
    [
        ([0, np.nan], "skip", "POINT EMPTY"),
        ([np.nan, 0], "skip", "POINT EMPTY"),
        ([0, np.nan, 1], "skip", "POINT Z EMPTY"),
        ([0, 1, np.nan], "skip", "POINT Z EMPTY"),
        ([0, 1], "error", "POINT (0 1)"),
        ([0, 1, 2], "error", "POINT Z (0 1 2)"),
        ([0, np.inf], "skip", "POINT EMPTY"),
    ],
)
def test_points_handle_nan(coords, handle_nan, expected_wkt):
    actual = shapely.points(coords, handle_nan=handle_nan)
    assert actual.wkt in expected_wkt


@pytest.mark.parametrize(
    "coords",
    [
        [0, np.nan],
        [np.nan, 0],
        [0, 0, np.nan],
        [0, np.inf],
        [0, -np.inf],
    ],
)
def test_points_handle_nan_error(coords):
    with pytest.raises(ValueError, match=".*NaN.*"):
        shapely.points(coords, handle_nan="error")


def test_linestrings_from_coords():
    actual = shapely.linestrings([[[0, 0], [1, 1]], [[0, 0], [2, 2]]])
    assert_geometries_equal(
        actual,
        [
            LineString([(0, 0), (1, 1)]),
            LineString([(0, 0), (2, 2)]),
        ],
    )


def test_linestrings_from_xy():
    actual = shapely.linestrings([0, 1], [2, 3])
    assert_geometries_equal(actual, LineString([(0, 2), (1, 3)]))


def test_linestrings_from_xy_broadcast():
    x = [0, 1]  # the same X coordinates for both linestrings
    y = [2, 3], [4, 5]  # each linestring has a different set of Y coordinates
    actual = shapely.linestrings(x, y)
    assert_geometries_equal(
        actual,
        [
            LineString([(0, 2), (1, 3)]),
            LineString([(0, 4), (1, 5)]),
        ],
    )


def test_linestrings_from_xyz():
    actual = shapely.linestrings([0, 1], [2, 3], 0)
    assert_geometries_equal(actual, LineString([(0, 2, 0), (1, 3, 0)]))


@pytest.mark.parametrize("dim", [2, 3])
def test_linestrings_buffer(dim):
    coords = np.random.randn(10, 3, dim)
    coords1 = np.asarray(coords, order="C")
    result1 = shapely.linestrings(coords1)

    coords2 = np.asarray(coords1, order="F")
    result2 = shapely.linestrings(coords2)
    assert_geometries_equal(result1, result2)

    # creating (.., 8, 8*3) strided array so it uses copyFromArrays
    coords3 = np.asarray(np.swapaxes(np.swapaxes(coords, 0, 2), 1, 0), order="F")
    coords3 = np.swapaxes(np.swapaxes(coords3, 0, 2), 1, 2)
    result3 = shapely.linestrings(coords3)
    assert_geometries_equal(result1, result3)


def test_linestrings_empty():
    actual = shapely.linestrings(np.empty((0, 2)))
    assert actual.is_empty


def test_linestrings_invalid_shape_scalar():
    with pytest.raises(ValueError):
        shapely.linestrings((1, 1))


@pytest.mark.parametrize(
    "shape",
    [
        (2, 1, 2),  # 2 linestrings of 1 2D point
        (1, 1, 2),  # 1 linestring of 1 2D point
        (1, 2),  # 1 linestring of 1 2D point (scalar)
    ],
)
def test_linestrings_invalid_shape(shape):
    with pytest.raises(shapely.GEOSException):
        shapely.linestrings(np.ones(shape))


def test_linestrings_invalid_ndim():
    msg = r"The ordinate \(last\) dimension should be 2 or 3, got {}"

    coords = np.ones((10, 2, 4), order="C")
    with pytest.raises(ValueError, match=msg.format(4)):
        shapely.linestrings(coords)

    coords = np.ones((10, 2, 4), order="F")
    with pytest.raises(ValueError, match=msg.format(4)):
        shapely.linestrings(coords)

    coords = np.swapaxes(np.swapaxes(np.ones((10, 2, 4)), 0, 2), 1, 0)
    coords = np.swapaxes(np.swapaxes(np.asarray(coords, order="F"), 0, 2), 1, 2)
    with pytest.raises(ValueError, match=msg.format(4)):
        shapely.linestrings(coords)

    # too few ordinates
    coords = np.ones((10, 2, 1))
    with pytest.raises(ValueError, match=msg.format(1)):
        shapely.linestrings(coords)


@pytest.mark.parametrize(
    "coords",
    [
        [[0, 1], [2, float("nan")]],
        [[0, 1, float("nan")], [2, 2, float("nan")]],
        [[0, 1, 2], [2, 2, float("nan")]],
        [[0, 1, float("nan")], [2, 2, 3]],
        [[float("nan"), float("nan")], [float("nan"), float("nan")]],
    ],
)
def test_linestrings_handle_nan_allow(coords):
    with ignore_invalid():
        actual = shapely.linestrings(coords, handle_nan="allow")
    actual = shapely.get_coordinates(actual, include_z=len(coords[0]) == 3)

    assert_array_equal(actual, coords)


@pytest.mark.parametrize(
    "coords",
    [
        [[2, float("nan")], [0, 1], [2, 3]],
        [[0, 1], [2, float("nan")], [2, 3]],
        [[0, 1], [2, 3], [2, float("nan")]],
    ],
)
def test_linestrings_handle_nan_skip(coords):
    actual = shapely.linestrings(coords, handle_nan="skip")
    assert_geometries_equal(actual, LineString([(0, 1), (2, 3)]))


def test_linestrings_handle_nan_skip_invalid():
    with pytest.raises(shapely.GEOSException):
        shapely.linestrings([[0, 1], [2, float("nan")]], handle_nan="skip")


def test_linestrings_handle_nan_skip_only_nan():
    # all-nan becomes an empty linestring
    actual = shapely.linestrings(np.full((3, 2), fill_value=np.nan), handle_nan="skip")
    assert actual.is_empty


def test_linestrings_handle_nan_error():
    with pytest.raises(ValueError, match=".*NaN.*"):
        shapely.linestrings([[0, 1], [2, float("nan")], [2, 3]], handle_nan="error")


def test_linearrings():
    actual = shapely.linearrings(box_tpl(0, 0, 1, 1))
    assert_geometries_equal(
        actual, LinearRing([(1, 0), (1, 1), (0, 1), (0, 0), (1, 0)])
    )


def test_linearrings_empty():
    actual = shapely.linearrings(np.empty((0, 2)))
    assert actual.is_empty


def test_linearrings_from_xy():
    actual = shapely.linearrings([0, 1, 2, 0], [3, 4, 5, 3])
    assert_geometries_equal(actual, LinearRing([(0, 3), (1, 4), (2, 5), (0, 3)]))


def test_linearrings_unclosed():
    actual = shapely.linearrings(box_tpl(0, 0, 1, 1)[:-1])
    assert_geometries_equal(
        actual, LinearRing([(1, 0), (1, 1), (0, 1), (0, 0), (1, 0)])
    )


def test_linearrings_unclosed_all_coords_equal():
    actual = shapely.linearrings([(0, 0), (0, 0), (0, 0)])
    assert_geometries_equal(actual, LinearRing([(0, 0), (0, 0), (0, 0), (0, 0)]))


def test_linearrings_invalid_shape_scalar():
    with pytest.raises(ValueError):
        shapely.linearrings((1, 1))


@pytest.mark.parametrize(
    "shape",
    [
        (2, 1, 2),  # 2 linearrings of 1 2D point
        (1, 1, 2),  # 1 linearring of 1 2D point
        (1, 2),  # 1 linearring of 1 2D point (scalar)
        (2, 2, 2),  # 2 linearrings of 2 2D points
        (1, 2, 2),  # 1 linearring of 2 2D points
        (2, 2),  # 1 linearring of 2 2D points (scalar)
    ],
)
def test_linearrings_invalid_shape(shape):
    coords = np.ones(shape)
    with pytest.raises(ValueError):
        shapely.linearrings(coords)

    # make sure the first coordinate != second coordinate
    coords[..., 1] += 1
    with pytest.raises(ValueError):
        shapely.linearrings(coords)


def test_linearrings_invalid_ndim():
    msg = r"The ordinate \(last\) dimension should be 2 or 3, got {}"

    coords1 = np.random.randn(10, 3, 4)
    with pytest.raises(ValueError, match=msg.format(4)):
        shapely.linearrings(coords1)

    coords2 = np.hstack((coords1, coords1[:, [0], :]))
    with pytest.raises(ValueError, match=msg.format(4)):
        shapely.linearrings(coords2)

    # too few ordinates
    coords3 = np.random.randn(10, 3, 1)
    with pytest.raises(ValueError, match=msg.format(1)):
        shapely.linearrings(coords3)


def test_linearrings_all_nan():
    coords = np.full((4, 2), np.nan)
    with pytest.raises(shapely.GEOSException):
        shapely.linearrings(coords)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("order", ["C", "F"])
def test_linearrings_buffer(dim, order):
    coords1 = np.random.randn(10, 4, dim)
    coords1 = np.asarray(coords1, order=order)
    result1 = shapely.linearrings(coords1)

    # with manual closure -> can directly copy from buffer if C order
    coords2 = np.hstack((coords1, coords1[:, [0], :]))
    coords2 = np.asarray(coords2, order=order)
    result2 = shapely.linearrings(coords2)
    assert_geometries_equal(result1, result2)

    # create scalar -> can also directly copy from buffer if F order
    coords3 = np.asarray(coords2[0], order=order)
    result3 = shapely.linearrings(coords3)
    assert_geometries_equal(result3, result1[0])


@pytest.mark.parametrize(
    "coords",
    [
        [[0, 2], [1, float("nan")], [1, 3], [0, 2]],
        [[0, 2], [float("nan"), 0], [1, 3], [0, 2]],
        [[0, 2, 5], [float("nan"), 2, 5], [1, 3, 5], [0, 2, 5]],
        [[0, 2, 5], [1, 2, float("nan")], [1, 3, 5], [0, 2, 5]],
    ],
)
def test_linearrings_handle_nan_allow(coords):
    with ignore_invalid():
        actual = shapely.linearrings(coords, handle_nan="allow")
    actual = shapely.get_coordinates(actual, include_z=len(coords[0]) == 3)

    assert_array_equal(actual, coords)


@pytest.mark.parametrize(
    "x,y",
    [
        ([0, 1, float("nan"), 2, 0], [3, 4, 5, 5, 3]),
        ([0, 1, 1, 2, 0], [3, 4, float("nan"), 5, 3]),
        ([0, 1, 2, 0, float("nan")], [3, 4, 5, 3, 3]),
        ([float("nan"), 0, 1, 2, 0], [3, 3, 4, 5, 3]),
    ],
)
def test_linearrings_handle_nan_skip(x, y):
    actual = shapely.linearrings(x, y, handle_nan="skip")
    assert_geometries_equal(actual, LinearRing([(0, 3), (1, 4), (2, 5), (0, 3)]))


def test_linearrings_handle_nan_skip_invalid():
    with pytest.raises(ValueError):
        shapely.linearrings([0, float("nan"), 0], [3, 4, 3], handle_nan="skip")


def test_linearrings_handle_nan_skip_only_nan():
    actual = shapely.linearrings(np.full((5, 2), fill_value=np.nan), handle_nan="skip")
    assert actual.is_empty


def test_linearrings_handle_nan_error():
    with pytest.raises(ValueError, match=".*NaN.*"):
        shapely.linearrings(
            [0, 1, float("nan"), 2, 0], [3, 4, 5, 5, 3], handle_nan="error"
        )


def test_polygon_from_linearring():
    actual = shapely.polygons(shapely.linearrings(box_tpl(0, 0, 1, 1)))
    assert_geometries_equal(actual, Polygon([(1, 0), (1, 1), (0, 1), (0, 0), (1, 0)]))


def test_polygons_none():
    assert_geometries_equal(shapely.polygons(None), empty_polygon)
    assert_geometries_equal(shapely.polygons(None, holes=[linear_ring]), empty_polygon)


def test_polygons():
    actual = shapely.polygons(box_tpl(0, 0, 1, 1))
    assert_geometries_equal(actual, Polygon([(1, 0), (1, 1), (0, 1), (0, 0), (1, 0)]))


def test_polygon_no_hole_list_raises():
    with pytest.raises(ValueError):
        shapely.polygons(box_tpl(0, 0, 10, 10), box_tpl(1, 1, 2, 2))


def test_polygon_no_hole_wrong_type():
    with pytest.raises((TypeError, shapely.GEOSException)):
        shapely.polygons(point)


def test_polygon_with_hole_wrong_type():
    with pytest.raises((TypeError, shapely.GEOSException)):
        shapely.polygons(point, [linear_ring])


def test_polygon_wrong_hole_type():
    with pytest.raises((TypeError, shapely.GEOSException)):
        shapely.polygons(linear_ring, [point])


def test_polygon_with_1_hole():
    actual = shapely.polygons(box_tpl(0, 0, 10, 10), [box_tpl(1, 1, 2, 2)])
    assert shapely.area(actual) == 99.0


def test_polygon_with_2_holes():
    actual = shapely.polygons(
        box_tpl(0, 0, 10, 10), [box_tpl(1, 1, 2, 2), box_tpl(3, 3, 4, 4)]
    )
    assert shapely.area(actual) == 98.0


def test_polygon_with_none_hole():
    actual = shapely.polygons(
        shapely.linearrings(box_tpl(0, 0, 10, 10)),
        [
            shapely.linearrings(box_tpl(1, 1, 2, 2)),
            None,
            shapely.linearrings(box_tpl(3, 3, 4, 4)),
        ],
    )
    assert shapely.area(actual) == 98.0


def test_2_polygons_with_same_hole():
    actual = shapely.polygons(
        [box_tpl(0, 0, 10, 10), box_tpl(0, 0, 5, 5)], [box_tpl(1, 1, 2, 2)]
    )
    assert shapely.area(actual).tolist() == [99.0, 24.0]


def test_2_polygons_with_2_same_holes():
    actual = shapely.polygons(
        [box_tpl(0, 0, 10, 10), box_tpl(0, 0, 5, 5)],
        [box_tpl(1, 1, 2, 2), box_tpl(3, 3, 4, 4)],
    )
    assert shapely.area(actual).tolist() == [98.0, 23.0]


def test_2_polygons_with_different_holes():
    actual = shapely.polygons(
        [box_tpl(0, 0, 10, 10), box_tpl(0, 0, 5, 5)],
        [[box_tpl(1, 1, 3, 3)], [box_tpl(1, 1, 2, 2)]],
    )
    assert shapely.area(actual).tolist() == [96.0, 24.0]


def test_polygons_not_enough_points_in_shell_scalar():
    with pytest.raises(ValueError):
        shapely.polygons((1, 1))


@pytest.mark.parametrize(
    "shape",
    [
        (2, 1, 2),  # 2 linearrings of 1 2D point
        (1, 1, 2),  # 1 linearring of 1 2D point
        (1, 2),  # 1 linearring of 1 2D point (scalar)
        (2, 2, 2),  # 2 linearrings of 2 2D points
        (1, 2, 2),  # 1 linearring of 2 2D points
        (2, 2),  # 1 linearring of 2 2D points (scalar)
    ],
)
def test_polygons_not_enough_points_in_shell(shape):
    coords = np.ones(shape)
    with pytest.raises(ValueError):
        shapely.polygons(coords)

    # make sure the first coordinate != second coordinate
    coords[..., 1] += 1
    with pytest.raises(ValueError):
        shapely.polygons(coords)


def test_polygons_not_enough_points_in_holes_scalar():
    with pytest.raises(ValueError):
        shapely.polygons(np.ones((1, 4, 2)), (1, 1))


@pytest.mark.parametrize(
    "shape",
    [
        (2, 1, 2),  # 2 linearrings of 1 2D point
        (1, 1, 2),  # 1 linearring of 1 2D point
        (1, 2),  # 1 linearring of 1 2D point (scalar)
        (2, 2, 2),  # 2 linearrings of 2 2D points
        (1, 2, 2),  # 1 linearring of 2 2D points
        (2, 2),  # 1 linearring of 2 2D points (scalar)
    ],
)
def test_polygons_not_enough_points_in_holes(shape):
    coords = np.ones(shape)
    with pytest.raises(ValueError):
        shapely.polygons(np.ones((1, 4, 2)), coords)

    # make sure the first coordinate != second coordinate
    coords[..., 1] += 1
    with pytest.raises(ValueError):
        shapely.polygons(np.ones((1, 4, 2)), coords)


@pytest.mark.parametrize(
    "func,expected",
    [
        (shapely.multipoints, MultiPoint()),
        (shapely.multilinestrings, MultiLineString()),
        (shapely.multipolygons, MultiPolygon()),
        (shapely.geometrycollections, GeometryCollection()),
    ],
)
def test_create_collection_only_none(func, expected):
    actual = func(np.array([None], dtype=object))
    assert_geometries_equal(actual, expected)


@pytest.mark.parametrize(
    "func,sub_geom",
    [
        (shapely.multipoints, point),
        (shapely.multilinestrings, line_string),
        (shapely.multilinestrings, linear_ring),
        (shapely.multipolygons, polygon),
        (shapely.geometrycollections, point),
        (shapely.geometrycollections, line_string),
        (shapely.geometrycollections, linear_ring),
        (shapely.geometrycollections, polygon),
        (shapely.geometrycollections, multi_point),
        (shapely.geometrycollections, multi_line_string),
        (shapely.geometrycollections, multi_polygon),
        (shapely.geometrycollections, geometry_collection),
    ],
)
def test_create_collection(func, sub_geom):
    actual = func([sub_geom, sub_geom])
    assert shapely.get_num_geometries(actual) == 2


@pytest.mark.parametrize(
    "func,sub_geom",
    [
        (shapely.multipoints, point),
        (shapely.multilinestrings, line_string),
        (shapely.multipolygons, polygon),
        (shapely.geometrycollections, polygon),
    ],
)
def test_create_collection_skips_none(func, sub_geom):
    actual = func([sub_geom, None, None, sub_geom])
    assert shapely.get_num_geometries(actual) == 2


@pytest.mark.parametrize(
    "func,sub_geom",
    [
        (shapely.multipoints, line_string),
        (shapely.multipoints, geometry_collection),
        (shapely.multipoints, multi_point),
        (shapely.multilinestrings, point),
        (shapely.multilinestrings, polygon),
        (shapely.multilinestrings, multi_line_string),
        (shapely.multipolygons, linear_ring),
        (shapely.multipolygons, multi_point),
        (shapely.multipolygons, multi_polygon),
    ],
)
def test_create_collection_wrong_geom_type(func, sub_geom):
    with pytest.raises(TypeError):
        func([sub_geom])


@pytest.mark.parametrize(
    "coords,ccw,expected",
    [
        ((0, 0, 1, 1), True, Polygon([(1, 0), (1, 1), (0, 1), (0, 0), (1, 0)])),
        (
            (0, 0, 1, 1),
            False,
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),
        ),
    ],
)
def test_box(coords, ccw, expected):
    actual = shapely.box(*coords, ccw=ccw)
    assert_geometries_equal(actual, expected)


@pytest.mark.parametrize(
    "coords,ccw,expected",
    [
        (
            (0, 0, [1, 2], [1, 2]),
            True,
            [
                Polygon([(1, 0), (1, 1), (0, 1), (0, 0), (1, 0)]),
                Polygon([(2, 0), (2, 2), (0, 2), (0, 0), (2, 0)]),
            ],
        ),
        (
            (0, 0, [1, 2], [1, 2]),
            [True, False],
            [
                Polygon([(1, 0), (1, 1), (0, 1), (0, 0), (1, 0)]),
                Polygon([(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]),
            ],
        ),
    ],
)
def test_box_array(coords, ccw, expected):
    actual = shapely.box(*coords, ccw=ccw)
    assert_geometries_equal(actual, expected)


@pytest.mark.parametrize(
    "coords",
    [
        [np.nan, np.nan, np.nan, np.nan],
        [np.nan, 0, 1, 1],
        [0, np.nan, 1, 1],
        [0, 0, np.nan, 1],
        [0, 0, 1, np.nan],
    ],
)
def test_box_nan(coords):
    assert shapely.box(*coords) is None


def test_box_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `ccw` for `box` is deprecated"
    ):
        shapely.box(0, 0, 1, 1, True)


def test_prepare():
    arr = np.array([shapely.points(1, 1), None, shapely.box(0, 0, 1, 1)])
    assert arr[0]._geom_prepared == 0
    assert arr[2]._geom_prepared == 0
    shapely.prepare(arr)
    assert arr[0]._geom_prepared != 0
    assert arr[1] is None
    assert arr[2]._geom_prepared != 0

    # preparing again actually does nothing
    original = arr[0]._geom_prepared
    shapely.prepare(arr)
    assert arr[0]._geom_prepared == original


def test_destroy_prepared():
    arr = np.array([shapely.points(1, 1), None, shapely.box(0, 0, 1, 1)])
    shapely.prepare(arr)
    assert arr[0]._geom_prepared != 0
    assert arr[2]._geom_prepared != 0
    shapely.destroy_prepared(arr)
    assert arr[0]._geom_prepared == 0
    assert arr[1] is None
    assert arr[2]._geom_prepared == 0
    shapely.destroy_prepared(arr)  # does not error


@pytest.mark.parametrize("geom_type", [None, GeometryType.MISSING, -1])
def test_empty_missing(geom_type):
    actual = shapely.empty((2,), geom_type=geom_type)
    assert shapely.is_missing(actual).all()


@pytest.mark.parametrize("geom_type", range(8))
def test_empty(geom_type):
    actual = shapely.empty((2,), geom_type=geom_type)
    assert (~shapely.is_missing(actual)).all()
    assert shapely.is_empty(actual).all()
    assert (shapely.get_type_id(actual) == geom_type).all()
