import numpy as np
import pytest
from numpy.testing import assert_array_equal

import shapely
from shapely import LinearRing, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import empty_point, line_string, linear_ring, point, polygon

pnts = shapely.points
lstrs = shapely.linestrings
geom_coll = shapely.geometrycollections


@pytest.mark.parametrize(
    "func", [shapely.points, shapely.linestrings, shapely.linearrings]
)
@pytest.mark.parametrize(
    "coordinates",
    [
        np.empty((2,)),  # not enough dimensions
        np.empty((2, 4, 1)),  # too many dimensions
        np.empty((2, 4)),  # wrong inner dimension size
        None,
        np.full((2, 2), "foo", dtype=object),  # wrong type
    ],
)
def test_invalid_coordinates(func, coordinates):
    with pytest.raises((TypeError, ValueError)):
        func(coordinates, indices=[0, 1])


@pytest.mark.parametrize(
    "func",
    [
        shapely.multipoints,
        shapely.multilinestrings,
        shapely.multipolygons,
        shapely.geometrycollections,
    ],
)
@pytest.mark.parametrize(
    "geometries", [np.array([1, 2], dtype=np.intp), None, np.array([[point]]), "hello"]
)
def test_invalid_geometries(func, geometries):
    with pytest.raises((TypeError, ValueError)):
        func(geometries, indices=[0, 1])


@pytest.mark.parametrize(
    "func", [shapely.points, shapely.linestrings, shapely.linearrings]
)
@pytest.mark.parametrize("indices", [[point], " hello", [0, 1], [-1]])
def test_invalid_indices_simple(func, indices):
    with pytest.raises((TypeError, ValueError)):
        func([[0.2, 0.3]], indices=indices)


non_writeable = np.empty(3, dtype=object)
non_writeable.flags.writeable = False


@pytest.mark.parametrize(
    "func", [shapely.points, shapely.linestrings, shapely.geometrycollections]
)
@pytest.mark.parametrize(
    "out",
    [
        [None, None, None],  # not an ndarray
        np.empty(3),  # wrong dtype
        non_writeable,  # not writeable
        np.empty((3, 2), dtype=object),  # too many dimensions
        np.empty((), dtype=object),  # too few dimensions
        np.empty((2,), dtype=object),  # too small
    ],
)
def test_invalid_out(func, out):
    if func is shapely.points:
        x = [[0.2, 0.3], [0.4, 0.5]]
        indices = [0, 2]
    elif func is shapely.linestrings:
        x = [[1, 1], [2, 1], [2, 2], [3, 3], [3, 4], [4, 4]]
        indices = [0, 0, 0, 2, 2, 2]
    else:
        x = [point, line_string]
        indices = [0, 2]
    with pytest.raises((TypeError, ValueError)):
        func(x, indices=indices, out=out)


def test_points_invalid():
    # attempt to construct a point with 2 coordinates
    with pytest.raises(shapely.GEOSException):
        shapely.points([[1, 1], [2, 2]], indices=[0, 0])


def test_points():
    actual = shapely.points(
        np.array([[2, 3], [2, 3]], dtype=float),
        indices=np.array([0, 1], dtype=np.intp),
    )
    assert_geometries_equal(actual, [point, point])


def test_points_no_index_raises():
    with pytest.raises(ValueError):
        shapely.points(
            np.array([[2, 3], [2, 3]], dtype=float),
            indices=np.array([0, 2], dtype=np.intp),
        )


@pytest.mark.parametrize(
    "indices,expected",
    [
        ([0, 1], [point, point, empty_point, None]),
        ([0, 3], [point, None, empty_point, point]),
        ([2, 3], [None, None, point, point]),
    ],
)
def test_points_out(indices, expected):
    out = np.empty(4, dtype=object)
    out[2] = empty_point
    actual = shapely.points(
        [[2, 3], [2, 3]],
        indices=indices,
        out=out,
    )
    assert_geometries_equal(out, expected)
    assert actual is out


@pytest.mark.parametrize(
    "coordinates,indices,expected",
    [
        ([[1, 1], [2, 2]], [0, 0], [lstrs([[1, 1], [2, 2]])]),
        ([[1, 1, 1], [2, 2, 2]], [0, 0], [lstrs([[1, 1, 1], [2, 2, 2]])]),
        (
            [[1, 1], [2, 2], [2, 2], [3, 3]],
            [0, 0, 1, 1],
            [lstrs([[1, 1], [2, 2]]), lstrs([[2, 2], [3, 3]])],
        ),
    ],
)
def test_linestrings(coordinates, indices, expected):
    actual = shapely.linestrings(
        np.array(coordinates, dtype=float), indices=np.array(indices, dtype=np.intp)
    )
    assert_geometries_equal(actual, expected)


def test_linestrings_invalid():
    # attempt to construct linestrings with 1 coordinate
    with pytest.raises(shapely.GEOSException):
        shapely.linestrings([[1, 1], [2, 2]], indices=[0, 1])


@pytest.mark.parametrize(
    "indices,expected",
    [
        ([0, 0, 0, 1, 1, 1], [line_string, line_string, empty_point, None]),
        ([0, 0, 0, 3, 3, 3], [line_string, None, empty_point, line_string]),
        ([2, 2, 2, 3, 3, 3], [None, None, line_string, line_string]),
    ],
)
def test_linestrings_out(indices, expected):
    out = np.empty(4, dtype=object)
    out[2] = empty_point
    actual = shapely.linestrings(
        [(0, 0), (1, 0), (1, 1), (0, 0), (1, 0), (1, 1)],
        indices=indices,
        out=out,
    )
    assert_geometries_equal(out, expected)
    assert actual is out


@pytest.mark.parametrize(
    "coordinates",
    [
        [[1, 1], [1, float("nan")], [2, 2]],
    ],
)
def test_linestrings_allow_nan(coordinates):
    actual = shapely.linestrings(
        np.array(coordinates, dtype=float),
        indices=np.zeros(len(coordinates), dtype=np.intp),
    )
    assert_array_equal(shapely.get_coordinates(actual), coordinates)


@pytest.mark.parametrize(
    "coordinates,indices,expected",
    [
        ([[1, 1], [1, float("nan")], [2, 2]], [0, 0, 0], [lstrs([[1, 1], [2, 2]])]),
    ],
)
def test_linestrings_handle_nan_skip(coordinates, indices, expected):
    actual = shapely.linestrings(
        np.array(coordinates, dtype=float),
        indices=np.array(indices, dtype=np.intp),
        handle_nan="skip",
    )
    assert_geometries_equal(actual, expected)


def test_linestrings_handle_nan_skip_invalid():
    # the NaN makes the linestring too short
    with pytest.raises(shapely.GEOSException):
        shapely.linestrings(
            [[1, 1], [2, float("nan")]], indices=[0, 0], handle_nan="skip"
        )


def test_linestrings_handle_nan_skip_only_nan():
    actual = shapely.linestrings(
        np.full((3, 2), fill_value=np.nan), indices=[0, 0, 0], handle_nan="skip"
    )
    assert actual[0].is_empty


def test_linestrings_handle_nan_error():
    with pytest.raises(ValueError, match=".*NaN.*"):
        shapely.linestrings(
            [[0, 0], [float("nan"), 0], [1, 1]], indices=[0, 0, 0], handle_nan="error"
        )


@pytest.mark.parametrize(
    "coordinates", [([[1, 1], [2, 1], [2, 2], [1, 1]]), ([[1, 1], [2, 1], [2, 2]])]
)
def test_linearrings(coordinates):
    actual = shapely.linearrings(
        np.array(coordinates, dtype=np.float64),
        indices=np.zeros(len(coordinates), dtype=np.intp),
    )
    assert_geometries_equal(actual, shapely.linearrings(coordinates))


@pytest.mark.parametrize(
    "coordinates",
    [
        ([[1, 1], [1, 1]]),  # too short
        ([[1, np.nan], [2, 1], [2, 2], [1, 1]]),  # starting with nan
    ],
)
def test_linearrings_invalid(coordinates):
    with pytest.raises((shapely.GEOSException, ValueError)):
        shapely.linearrings(coordinates, indices=np.zeros(len(coordinates)))


def test_linearrings_unclosed_all_coords_equal():
    actual = shapely.linearrings([(0, 0), (0, 0), (0, 0)], indices=np.zeros(3))
    assert_geometries_equal(actual, LinearRing([(0, 0), (0, 0), (0, 0), (0, 0)]))


@pytest.mark.parametrize(
    "indices,expected",
    [
        ([0, 0, 0, 0, 0], [linear_ring, None, None, empty_point]),
        ([1, 1, 1, 1, 1], [None, linear_ring, None, empty_point]),
        ([3, 3, 3, 3, 3], [None, None, None, linear_ring]),
    ],
)
def test_linearrings_out(indices, expected):
    out = np.empty(4, dtype=object)
    out[3] = empty_point
    actual = shapely.linearrings(
        [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)],
        indices=indices,
        out=out,
    )
    assert_geometries_equal(out, expected)
    assert actual is out


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("order", ["C", "F"])
def test_linearrings_buffer(dim, order):
    coords = np.random.randn(10, 4, dim)
    coords1 = np.asarray(coords.reshape(10 * 4, dim), order=order)
    indices1 = np.repeat(range(10), 4)
    result1 = shapely.linearrings(coords1, indices=indices1)

    # with manual closure -> can directly copy from buffer if C order
    coords2 = np.hstack((coords, coords[:, [0], :]))
    coords2 = np.asarray(coords2.reshape(10 * 5, dim), order=order)
    indices2 = np.repeat(range(10), 5)
    result2 = shapely.linearrings(coords2, indices=indices2)
    assert_geometries_equal(result1, result2)


@pytest.mark.parametrize(
    "coordinates",
    [
        [[1, 1], [2, 1], [2, float("nan")], [1, 1]],
    ],
)
def test_linearrings_allow_nan(coordinates):
    actual = shapely.linearrings(
        np.array(coordinates, dtype=float),
        indices=np.zeros(len(coordinates), dtype=np.intp),
    )
    assert_array_equal(shapely.get_coordinates(actual), coordinates)


@pytest.mark.parametrize(
    "coordinates",
    [
        [[1, 1], [2, 1], [2, 2], [2, float("nan")], [1, 1]],
        [[1, 1], [2, 1], [2, float("nan")], [2, 2]],
        [[1, 1], [2, 1], [2, 2], [1, 1], [2, float("nan")]],
        [[2, float("nan")], [1, 1], [2, 1], [2, 2], [1, 1]],
        [[1, 1], [2, 1], [2, 2], [2, float("nan")]],
        [[2, float("nan")], [1, 1], [2, 1], [2, 2]],
    ],
)
def test_linearrings_handle_nan_skip(coordinates):
    actual = shapely.linearrings(
        np.array(coordinates, dtype=np.float64),
        indices=np.zeros(len(coordinates), dtype=np.intp),
        handle_nan="skip",
    )
    assert_geometries_equal(actual, shapely.linearrings(coordinates, handle_nan="skip"))


def test_linearrings_handle_nan_skip_invalid():
    # the NaN makes the linearring too short
    with pytest.raises(ValueError):
        shapely.linearrings(
            [[1, 1], [float("nan"), 1], [1, 1]], indices=[0, 0, 0], handle_nan="skip"
        )


def test_linearrings_handle_nan_skip_only_nan():
    actual = shapely.linearrings(
        np.full((5, 2), fill_value=np.nan), indices=[0] * 5, handle_nan="skip"
    )

    assert actual[0].is_empty


def test_linearrings_handle_nan_error():
    with pytest.raises(ValueError, match=".*NaN.*"):
        shapely.linearrings(
            [[1, 1], [2, 1], [2, 2], [2, float("nan")], [1, 1]],
            indices=[0, 0, 0, 0, 0],
            handle_nan="error",
        )


hole_1 = shapely.linearrings([(0.2, 0.2), (0.2, 0.4), (0.4, 0.4)])
hole_2 = shapely.linearrings([(0.6, 0.6), (0.6, 0.8), (0.8, 0.8)])
poly = shapely.polygons(linear_ring)
poly_empty = Polygon()
poly_hole_1 = shapely.polygons(linear_ring, holes=[hole_1])
poly_hole_2 = shapely.polygons(linear_ring, holes=[hole_2])
poly_hole_1_2 = shapely.polygons(linear_ring, holes=[hole_1, hole_2])


@pytest.mark.parametrize(
    "rings,indices,expected",
    [
        ([linear_ring, linear_ring], [0, 1], [poly, poly]),
        ([None, linear_ring], [0, 1], [poly_empty, poly]),
        ([None, linear_ring, None, None], [0, 0, 1, 1], [poly, poly_empty]),
        ([linear_ring, hole_1, linear_ring], [0, 0, 1], [poly_hole_1, poly]),
        ([linear_ring, linear_ring, hole_1], [0, 1, 1], [poly, poly_hole_1]),
        ([None, linear_ring, linear_ring, hole_1], [0, 0, 1, 1], [poly, poly_hole_1]),
        ([linear_ring, None, linear_ring, hole_1], [0, 0, 1, 1], [poly, poly_hole_1]),
        ([linear_ring, None, linear_ring, hole_1], [0, 1, 1, 1], [poly, poly_hole_1]),
        ([linear_ring, linear_ring, None, hole_1], [0, 1, 1, 1], [poly, poly_hole_1]),
        ([linear_ring, linear_ring, hole_1, None], [0, 1, 1, 1], [poly, poly_hole_1]),
        (
            [linear_ring, hole_1, hole_2, linear_ring],
            [0, 0, 0, 1],
            [poly_hole_1_2, poly],
        ),
        (
            [linear_ring, hole_1, linear_ring, hole_2],
            [0, 0, 1, 1],
            [poly_hole_1, poly_hole_2],
        ),
        (
            [linear_ring, linear_ring, hole_1, hole_2],
            [0, 1, 1, 1],
            [poly, poly_hole_1_2],
        ),
        (
            [linear_ring, hole_1, None, hole_2, linear_ring],
            [0, 0, 0, 0, 1],
            [poly_hole_1_2, poly],
        ),
        (
            [linear_ring, hole_1, None, linear_ring, hole_2],
            [0, 0, 0, 1, 1],
            [poly_hole_1, poly_hole_2],
        ),
        (
            [linear_ring, hole_1, linear_ring, None, hole_2],
            [0, 0, 1, 1, 1],
            [poly_hole_1, poly_hole_2],
        ),
    ],
)
def test_polygons(rings, indices, expected):
    actual = shapely.polygons(
        np.array(rings, dtype=object), indices=np.array(indices, dtype=np.intp)
    )
    assert_geometries_equal(actual, expected)


@pytest.mark.parametrize(
    "indices,expected",
    [
        ([0, 1], [poly, poly, empty_point, None]),
        ([0, 3], [poly, None, empty_point, poly]),
        ([2, 3], [None, None, poly, poly]),
    ],
)
def test_polygons_out(indices, expected):
    out = np.empty(4, dtype=object)
    out[2] = empty_point
    actual = shapely.polygons([linear_ring, linear_ring], indices=indices, out=out)
    assert_geometries_equal(out, expected)
    assert actual is out


@pytest.mark.parametrize(
    "func",
    [
        shapely.polygons,
        shapely.multipoints,
        shapely.multilinestrings,
        shapely.multipolygons,
        shapely.geometrycollections,
    ],
)
@pytest.mark.parametrize("indices", [np.array([point]), " hello", [0, 1], [-1]])
def test_invalid_indices_collections(func, indices):
    with pytest.raises((TypeError, ValueError)):
        func([point], indices=indices)


@pytest.mark.parametrize(
    "geometries,indices,expected",
    [
        ([point, line_string], [0, 0], [geom_coll([point, line_string])]),
        ([point, line_string], [0, 1], [geom_coll([point]), geom_coll([line_string])]),
        ([point, None], [0, 0], [geom_coll([point])]),
        ([point, None], [0, 1], [geom_coll([point]), geom_coll([])]),
        ([None, point, None, None], [0, 0, 1, 1], [geom_coll([point]), geom_coll([])]),
        ([point, None, line_string], [0, 0, 0], [geom_coll([point, line_string])]),
    ],
)
def test_geometrycollections(geometries, indices, expected):
    actual = shapely.geometrycollections(
        np.array(geometries, dtype=object), indices=indices
    )
    assert_geometries_equal(actual, expected)


def test_geometrycollections_no_index_raises():
    with pytest.raises(ValueError):
        shapely.geometrycollections(
            np.array([point, line_string], dtype=object), indices=[0, 2]
        )


@pytest.mark.parametrize(
    "indices,expected",
    [
        ([0, 0], [geom_coll([point, line_string]), None, None, empty_point]),
        ([3, 3], [None, None, None, geom_coll([point, line_string])]),
    ],
)
def test_geometrycollections_out(indices, expected):
    out = np.empty(4, dtype=object)
    out[3] = empty_point
    actual = shapely.geometrycollections([point, line_string], indices=indices, out=out)
    assert_geometries_equal(out, expected)
    assert actual is out


def test_multipoints():
    actual = shapely.multipoints(
        np.array([point], dtype=object), indices=np.zeros(1, dtype=np.intp)
    )
    assert_geometries_equal(actual, shapely.multipoints([point]))


def test_multilinestrings():
    actual = shapely.multilinestrings(
        np.array([line_string], dtype=object), indices=np.zeros(1, dtype=np.intp)
    )
    assert_geometries_equal(actual, shapely.multilinestrings([line_string]))


def test_multilinearrings():
    actual = shapely.multilinestrings(
        np.array([linear_ring], dtype=object), indices=np.zeros(1, dtype=np.intp)
    )
    assert_geometries_equal(actual, shapely.multilinestrings([linear_ring]))


def test_multipolygons():
    actual = shapely.multipolygons(
        np.array([polygon], dtype=object), indices=np.zeros(1, dtype=np.intp)
    )
    assert_geometries_equal(actual, shapely.multipolygons([polygon]))


@pytest.mark.parametrize(
    "geometries,func",
    [
        ([point], shapely.polygons),
        ([line_string], shapely.polygons),
        ([polygon], shapely.polygons),
        ([line_string], shapely.multipoints),
        ([polygon], shapely.multipoints),
        ([point], shapely.multilinestrings),
        ([polygon], shapely.multilinestrings),
        ([point], shapely.multipolygons),
        ([line_string], shapely.multipolygons),
    ],
)
def test_incompatible_types(geometries, func):
    with pytest.raises(TypeError):
        func(geometries, indices=[0])


def test_points_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `indices` for `points` is deprecated"
    ):
        shapely.points([[0, 1], [2, 3]], None, None, [0, 1])


def test_linestrings_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `indices` for `linestrings` is deprecated"
    ):
        shapely.linestrings([[0, 1], [2, 3], [4, 5], [6, 7]], None, None, [0, 0, 1, 1])


def test_linearrings_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `indices` for `linearrings` is deprecated"
    ):
        shapely.linearrings([[0, 1], [1, 1], [1, 0]], None, None, [0, 0, 0])


def test_polygons_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `indices` for `polygons` is deprecated"
    ):
        shapely.polygons([linear_ring, linear_ring], None, [0, 1])


def test_multipoints_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `indices` for `multipoints` is deprecated"
    ):
        shapely.multipoints([point, point], [0, 1])


def test_multilinestrings_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `indices` for `multilinestrings` is deprecated"
    ):
        shapely.multilinestrings([line_string, line_string], [0, 1])


def test_multipolygons_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `indices` for `multipolygons` is deprecated"
    ):
        shapely.multipolygons([polygon, polygon], [0, 1])


def test_geometrycollections_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `indices` for `geometrycollections` is deprecated"
    ):
        shapely.geometrycollections([point, polygon], [0, 1])
