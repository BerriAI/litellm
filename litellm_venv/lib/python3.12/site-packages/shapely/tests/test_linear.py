import numpy as np
import pytest

import shapely
from shapely import GeometryCollection, LinearRing, LineString, MultiLineString, Point
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
    empty_line_string,
    empty_point,
    line_string,
    linear_ring,
    multi_line_string,
    multi_point,
    multi_polygon,
    point,
    polygon,
)


def test_line_interpolate_point_geom_array():
    actual = shapely.line_interpolate_point(
        [line_string, linear_ring, multi_line_string], -1
    )
    assert_geometries_equal(actual[0], Point(1, 0))
    assert_geometries_equal(actual[1], Point(0, 1))
    assert_geometries_equal(actual[2], Point(0.5528, 1.1056), tolerance=0.001)


def test_line_interpolate_point_geom_array_normalized():
    actual = shapely.line_interpolate_point(
        [line_string, linear_ring, multi_line_string], 1, normalized=True
    )
    assert_geometries_equal(actual[0], Point(1, 1))
    assert_geometries_equal(actual[1], Point(0, 0))
    assert_geometries_equal(actual[2], Point(1, 2))


def test_line_interpolate_point_float_array():
    actual = shapely.line_interpolate_point(line_string, [0.2, 1.5, -0.2])
    assert_geometries_equal(actual[0], Point(0.2, 0))
    assert_geometries_equal(actual[1], Point(1, 0.5))
    assert_geometries_equal(actual[2], Point(1, 0.8))


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize(
    "geom",
    [
        LineString(),
        LinearRing(),
        MultiLineString(),
        shapely.from_wkt("MULTILINESTRING (EMPTY, (0 0, 1 1))"),
        GeometryCollection(),
        GeometryCollection([LineString(), Point(1, 1)]),
    ],
)
def test_line_interpolate_point_empty(geom, normalized):
    assert_geometries_equal(
        shapely.line_interpolate_point(geom, 0.2, normalized=normalized), empty_point
    )


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize(
    "geom",
    [
        empty_point,
        point,
        polygon,
        multi_point,
        multi_polygon,
        shapely.geometrycollections([point]),
        shapely.geometrycollections([polygon]),
        shapely.geometrycollections([multi_line_string]),
        shapely.geometrycollections([multi_point]),
        shapely.geometrycollections([multi_polygon]),
    ],
)
def test_line_interpolate_point_invalid_type(geom, normalized):
    with pytest.raises(TypeError):
        assert shapely.line_interpolate_point(geom, 0.2, normalized=normalized)


def test_line_interpolate_point_none():
    assert shapely.line_interpolate_point(None, 0.2) is None


def test_line_interpolate_point_nan():
    assert shapely.line_interpolate_point(line_string, np.nan) is None


def test_line_interpolate_point_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `normalized` for `line_interpolate_point` "
        "is deprecated"
    ):
        shapely.line_interpolate_point(line_string, 0, False)


def test_line_locate_point_geom_array():
    point = shapely.points(0, 1)
    actual = shapely.line_locate_point([line_string, linear_ring], point)
    np.testing.assert_allclose(actual, [0.0, 3.0])


def test_line_locate_point_geom_array2():
    points = shapely.points([[0, 0], [1, 0]])
    actual = shapely.line_locate_point(line_string, points)
    np.testing.assert_allclose(actual, [0.0, 1.0])


@pytest.mark.parametrize("normalized", [False, True])
def test_line_locate_point_none(normalized):
    assert np.isnan(shapely.line_locate_point(line_string, None, normalized=normalized))
    assert np.isnan(shapely.line_locate_point(None, point, normalized=normalized))


@pytest.mark.parametrize("normalized", [False, True])
def test_line_locate_point_empty(normalized):
    assert np.isnan(
        shapely.line_locate_point(line_string, empty_point, normalized=normalized)
    )
    assert np.isnan(
        shapely.line_locate_point(empty_line_string, point, normalized=normalized)
    )


@pytest.mark.parametrize("normalized", [False, True])
def test_line_locate_point_invalid_geometry(normalized):
    with pytest.raises(shapely.GEOSException):
        shapely.line_locate_point(line_string, line_string, normalized=normalized)

    with pytest.raises(shapely.GEOSException):
        shapely.line_locate_point(polygon, point, normalized=normalized)


def test_line_locate_point_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `normalized` for `line_locate_point` is deprecated"
    ):
        shapely.line_locate_point(line_string, point, False)


def test_line_merge_geom_array():
    actual = shapely.line_merge([line_string, multi_line_string])
    assert_geometries_equal(actual[0], line_string)
    assert_geometries_equal(actual[1], LineString([(0, 0), (1, 2)]))


@pytest.mark.skipif(shapely.geos_version < (3, 11, 0), reason="GEOS < 3.11.0")
def test_line_merge_directed():
    lines = MultiLineString([[(0, 0), (1, 0)], [(0, 0), (3, 0)]])
    # Merge lines without directed, this requires changing the vertex ordering
    result = shapely.line_merge(lines)
    assert_geometries_equal(result, LineString([(1, 0), (0, 0), (3, 0)]))
    # Since the lines can't be merged when directed is specified
    # the original geometry is returned
    result = shapely.line_merge(lines, directed=True)
    assert_geometries_equal(result, lines)


@pytest.mark.skipif(shapely.geos_version >= (3, 11, 0), reason="GEOS >= 3.11.0")
def test_line_merge_error():
    lines = MultiLineString([[(0, 0), (1, 0)], [(0, 0), (3, 0)]])
    with pytest.raises(UnsupportedGEOSVersionError):
        shapely.line_merge(lines, directed=True)


def test_shared_paths_linestring():
    g1 = shapely.linestrings([(0, 0), (1, 0), (1, 1)])
    g2 = shapely.linestrings([(0, 0), (1, 0)])
    actual1 = shapely.shared_paths(g1, g2)
    assert_geometries_equal(
        shapely.get_geometry(actual1, 0), shapely.multilinestrings([g2])
    )


def test_shared_paths_none():
    assert shapely.shared_paths(line_string, None) is None
    assert shapely.shared_paths(None, line_string) is None
    assert shapely.shared_paths(None, None) is None


def test_shared_paths_non_linestring():
    g1 = shapely.linestrings([(0, 0), (1, 0), (1, 1)])
    g2 = shapely.points(0, 1)
    with pytest.raises(shapely.GEOSException):
        shapely.shared_paths(g1, g2)


def _prepare_input(geometry, prepare):
    """Prepare without modifying in-place"""
    if prepare:
        geometry = shapely.transform(geometry, lambda x: x)  # makes a copy
        shapely.prepare(geometry)
        return geometry
    else:
        return geometry


@pytest.mark.parametrize("prepare", [True, False])
def test_shortest_line(prepare):
    g1 = shapely.linestrings([(0, 0), (1, 0), (1, 1)])
    g2 = shapely.linestrings([(0, 3), (3, 0)])
    actual = shapely.shortest_line(_prepare_input(g1, prepare), g2)
    expected = shapely.linestrings([(1, 1), (1.5, 1.5)])
    assert shapely.equals(actual, expected)


@pytest.mark.parametrize("prepare", [True, False])
def test_shortest_line_none(prepare):
    assert shapely.shortest_line(_prepare_input(line_string, prepare), None) is None
    assert shapely.shortest_line(None, line_string) is None
    assert shapely.shortest_line(None, None) is None


@pytest.mark.parametrize("prepare", [True, False])
def test_shortest_line_empty(prepare):
    g1 = _prepare_input(line_string, prepare)
    assert shapely.shortest_line(g1, empty_line_string) is None
    g1_empty = _prepare_input(empty_line_string, prepare)
    assert shapely.shortest_line(g1_empty, line_string) is None
    assert shapely.shortest_line(g1_empty, empty_line_string) is None
