import numpy as np
import pytest
from numpy.testing import assert_allclose

import shapely
from shapely import MultiLineString, MultiPoint, MultiPolygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
    empty_line_string,
    empty_line_string_m,
    empty_line_string_z,
    empty_line_string_zm,
    empty_multi_polygon_m,
    empty_multi_polygon_z,
    empty_multi_polygon_zm,
    geometry_collection,
    line_string,
    line_string_m,
    line_string_z,
    line_string_zm,
    linear_ring,
    multi_line_string,
    multi_line_string_m,
    multi_line_string_z,
    multi_line_string_zm,
    multi_point,
    multi_point_m,
    multi_point_z,
    multi_point_zm,
    multi_polygon,
    multi_polygon_m,
    multi_polygon_z,
    multi_polygon_zm,
    point,
    point_m,
    point_z,
    point_zm,
    polygon,
    polygon_m,
    polygon_z,
    polygon_zm,
)

all_types = (
    point,
    line_string,
    polygon,
    multi_point,
    multi_line_string,
    multi_polygon,
)

all_types_z = (
    point_z,
    line_string_z,
    polygon_z,
    multi_point_z,
    multi_line_string_z,
    multi_polygon_z,
)

all_types_m = (
    point_m,
    line_string_m,
    polygon_m,
    multi_point_m,
    multi_line_string_m,
    multi_polygon_m,
)

all_types_zm = (
    point_zm,
    line_string_zm,
    polygon_zm,
    multi_point_zm,
    multi_line_string_zm,
    multi_polygon_zm,
)

all_types_dims_combos = all_types + all_types_z
if shapely.geos_version >= (3, 12, 0):
    all_types_dims_combos = all_types_dims_combos + all_types_m + all_types_zm

all_types_not_supported = (
    linear_ring,
    geometry_collection,
)


@pytest.mark.parametrize("geom", all_types + all_types_z)
def test_roundtrip(geom):
    actual = shapely.from_ragged_array(*shapely.to_ragged_array([geom, geom]))
    assert_geometries_equal(actual, [geom, geom])


@pytest.mark.parametrize("include_m", [None, True, False])
@pytest.mark.parametrize("include_z", [None, True, False])
@pytest.mark.parametrize("geom", all_types_dims_combos)
def test_to_ragged_array(geom, include_z, include_m):
    _, coords, _ = shapely.to_ragged_array(
        [geom, geom], include_z=include_z, include_m=include_m
    )
    nan_dims = np.all(np.isnan(coords), axis=0).tolist()
    expected = [False, False]  # XY
    has_z = geom.has_z
    if include_z or (include_z is None and has_z):
        expected.append(not has_z)  # XYZ or XYZM
    if shapely.geos_version >= (3, 12, 0):
        has_m = geom.has_m
    else:
        has_m = False
    if include_m or (include_m is None and has_m):
        expected.append(not has_m)  # XYM or XYZM
    assert nan_dims == expected


def test_include_z_default():
    # corner cases for inferring dimensionality

    # mixed XY and XYZ -> XYZ
    _, coords, _ = shapely.to_ragged_array([line_string, line_string_z])
    assert coords.shape[1] == 3

    # only empties -> always 2D
    _, coords, _ = shapely.to_ragged_array([empty_line_string])
    assert coords.shape[1] == 2
    _, coords, _ = shapely.to_ragged_array([empty_line_string_z])
    assert coords.shape[1] == 2
    # empty collection -> GEOS indicates 2D
    _, coords, _ = shapely.to_ragged_array([empty_multi_polygon_z])
    assert coords.shape[1] == 2


@pytest.mark.skipif(shapely.geos_version < (3, 12, 0), reason="GEOS < 3.12")
def test_include_m_default():
    # a few other corner cases for inferring dimensionality

    # mixed XY and XYM -> XYM
    _, coords, _ = shapely.to_ragged_array([line_string, line_string_m])
    assert coords.shape[1] == 3

    # mixed XY, XYM, and XYZM -> XYZM
    _, coords, _ = shapely.to_ragged_array([line_string, line_string_m, line_string_zm])
    assert coords.shape[1] == 4

    # only empties -> always 2D
    _, coords, _ = shapely.to_ragged_array([empty_line_string_m])
    assert coords.shape[1] == 2
    _, coords, _ = shapely.to_ragged_array([empty_line_string_zm])
    assert coords.shape[1] == 2
    # empty collection -> GEOS indicates 2D
    _, coords, _ = shapely.to_ragged_array([empty_multi_polygon_m])
    assert coords.shape[1] == 2
    _, coords, _ = shapely.to_ragged_array([empty_multi_polygon_zm])
    assert coords.shape[1] == 2


@pytest.mark.parametrize("geom", all_types)
def test_read_only_arrays(geom):
    # https://github.com/shapely/shapely/pull/1744
    typ, coords, offsets = shapely.to_ragged_array([geom, geom])
    coords.flags.writeable = False
    for arr in offsets:
        arr.flags.writeable = False
    result = shapely.from_ragged_array(typ, coords, offsets)
    assert_geometries_equal(result, [geom, geom])


@pytest.mark.parametrize("geom", all_types_not_supported)
def test_raise_geometry_type(geom):
    with pytest.raises(ValueError):
        shapely.to_ragged_array([geom, geom])


def test_points():
    arr = shapely.from_wkt(
        [
            "POINT (0 0)",
            "POINT (1 1)",
            "POINT EMPTY",
            "POINT EMPTY",
            "POINT (4 4)",
            None,
            "POINT EMPTY",
        ]
    )
    typ, result, offsets = shapely.to_ragged_array(arr)
    expected = np.array(
        [
            [0, 0],
            [1, 1],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [4, 4],
            [np.nan, np.nan],
            [np.nan, np.nan],
        ]
    )
    assert typ == shapely.GeometryType.POINT
    assert len(result) == len(arr)
    assert_allclose(result, expected)
    assert len(offsets) == 0

    geoms = shapely.from_ragged_array(typ, result)
    # in a roundtrip, missing geometries come back as empty
    arr[-2] = shapely.from_wkt("POINT EMPTY")
    assert_geometries_equal(geoms, arr)


def test_linestrings():
    arr = shapely.from_wkt(
        [
            "LINESTRING (30 10, 10 30, 40 40)",
            "LINESTRING (40 40, 30 30, 40 20, 30 10)",
            "LINESTRING EMPTY",
            "LINESTRING EMPTY",
            "LINESTRING (10 10, 20 20, 10 40)",
            None,
            "LINESTRING EMPTY",
        ]
    )
    typ, coords, offsets = shapely.to_ragged_array(arr)
    expected = np.array(
        [
            [30.0, 10.0],
            [10.0, 30.0],
            [40.0, 40.0],
            [40.0, 40.0],
            [30.0, 30.0],
            [40.0, 20.0],
            [30.0, 10.0],
            [10.0, 10.0],
            [20.0, 20.0],
            [10.0, 40.0],
        ]
    )
    expected_offsets = np.array([0, 3, 7, 7, 7, 10, 10, 10], dtype="int32")
    assert typ == shapely.GeometryType.LINESTRING
    assert_allclose(coords, expected)
    assert len(offsets) == 1
    assert offsets[0].dtype == np.int32
    assert_allclose(offsets[0], expected_offsets)

    result = shapely.from_ragged_array(typ, coords, offsets)
    # in a roundtrip, missing geometries come back as empty
    arr[-2] = shapely.from_wkt("LINESTRING EMPTY")
    assert_geometries_equal(result, arr)

    # sliced
    offsets_sliced = (offsets[0][1:],)
    result = shapely.from_ragged_array(typ, coords, offsets_sliced)
    assert_geometries_equal(result, arr[1:])

    offsets_sliced = (offsets[0][:-1],)
    result = shapely.from_ragged_array(typ, coords, offsets_sliced)
    assert_geometries_equal(result, arr[:-1])


def test_polygons():
    arr = shapely.from_wkt(
        [
            "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))",
            "POLYGON ((35 10, 45 45, 15 40, 10 20, 35 10), (20 30, 35 35, 30 20, 20 30))",  # noqa: E501
            "POLYGON EMPTY",
            "POLYGON EMPTY",
            "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))",
            None,
            "POLYGON EMPTY",
        ]
    )
    typ, coords, offsets = shapely.to_ragged_array(arr)
    expected = np.array(
        [
            [30.0, 10.0],
            [40.0, 40.0],
            [20.0, 40.0],
            [10.0, 20.0],
            [30.0, 10.0],
            [35.0, 10.0],
            [45.0, 45.0],
            [15.0, 40.0],
            [10.0, 20.0],
            [35.0, 10.0],
            [20.0, 30.0],
            [35.0, 35.0],
            [30.0, 20.0],
            [20.0, 30.0],
            [30.0, 10.0],
            [40.0, 40.0],
            [20.0, 40.0],
            [10.0, 20.0],
            [30.0, 10.0],
        ]
    )
    expected_offsets1 = np.array([0, 5, 10, 14, 19])
    expected_offsets2 = np.array([0, 1, 3, 3, 3, 4, 4, 4])

    assert typ == shapely.GeometryType.POLYGON
    assert_allclose(coords, expected)
    assert len(offsets) == 2
    assert offsets[0].dtype == np.int32
    assert offsets[1].dtype == np.int32
    assert_allclose(offsets[0], expected_offsets1)
    assert_allclose(offsets[1], expected_offsets2)

    result = shapely.from_ragged_array(typ, coords, offsets)
    # in a roundtrip, missing geometries come back as empty
    arr[-2] = shapely.from_wkt("POLYGON EMPTY")
    assert_geometries_equal(result, arr)

    # sliced:
    # - indices into the coordinate array for the whole buffer
    # - indices into the ring array for *just* the sliced part
    offsets_sliced = (offsets[0], offsets[1][1:])
    result = shapely.from_ragged_array(typ, coords, offsets_sliced)
    assert_geometries_equal(result, arr[1:])

    offsets_sliced = (offsets[0], offsets[1][:-1])
    result = shapely.from_ragged_array(typ, coords, offsets_sliced)
    assert_geometries_equal(result, arr[:-1])


def test_multipoints():
    arr = shapely.from_wkt(
        [
            "MULTIPOINT (10 40, 40 30, 20 20, 30 10)",
            "MULTIPOINT (30 10)",
            "MULTIPOINT EMPTY",
            "MULTIPOINT EMPTY",
            "MULTIPOINT (30 10, 10 30, 40 40)",
            None,
            "MULTIPOINT EMPTY",
        ]
    )
    typ, coords, offsets = shapely.to_ragged_array(arr)
    expected = np.array(
        [
            [10.0, 40.0],
            [40.0, 30.0],
            [20.0, 20.0],
            [30.0, 10.0],
            [30.0, 10.0],
            [30.0, 10.0],
            [10.0, 30.0],
            [40.0, 40.0],
        ]
    )
    expected_offsets = np.array([0, 4, 5, 5, 5, 8, 8, 8])

    assert typ == shapely.GeometryType.MULTIPOINT
    assert_allclose(coords, expected)
    assert len(offsets) == 1
    assert offsets[0].dtype == np.int32
    assert_allclose(offsets[0], expected_offsets)

    result = shapely.from_ragged_array(typ, coords, offsets)
    # in a roundtrip, missing geometries come back as empty
    arr[-2] = shapely.from_wkt("MULTIPOINT EMPTY")
    assert_geometries_equal(result, arr)

    # sliced:
    offsets_sliced = (offsets[0][1:],)
    result = shapely.from_ragged_array(typ, coords, offsets_sliced)
    assert_geometries_equal(result, arr[1:])

    offsets_sliced = (offsets[0][:-1],)
    result = shapely.from_ragged_array(typ, coords, offsets_sliced)
    assert_geometries_equal(result, arr[:-1])


def test_multilinestrings():
    arr = shapely.from_wkt(
        [
            "MULTILINESTRING ((30 10, 10 30, 40 40))",
            "MULTILINESTRING ((10 10, 20 20, 10 40), (40 40, 30 30, 40 20, 30 10))",
            "MULTILINESTRING EMPTY",
            "MULTILINESTRING EMPTY",
            "MULTILINESTRING ((35 10, 45 45), (15 40, 10 20), (30 10, 10 30, 40 40))",
            None,
            "MULTILINESTRING EMPTY",
        ]
    )
    typ, coords, offsets = shapely.to_ragged_array(arr)
    expected = np.array(
        [
            [30.0, 10.0],
            [10.0, 30.0],
            [40.0, 40.0],
            [10.0, 10.0],
            [20.0, 20.0],
            [10.0, 40.0],
            [40.0, 40.0],
            [30.0, 30.0],
            [40.0, 20.0],
            [30.0, 10.0],
            [35.0, 10.0],
            [45.0, 45.0],
            [15.0, 40.0],
            [10.0, 20.0],
            [30.0, 10.0],
            [10.0, 30.0],
            [40.0, 40.0],
        ]
    )
    expected_offsets1 = np.array([0, 3, 6, 10, 12, 14, 17])
    expected_offsets2 = np.array([0, 1, 3, 3, 3, 6, 6, 6])

    assert typ == shapely.GeometryType.MULTILINESTRING
    assert_allclose(coords, expected)
    assert len(offsets) == 2
    assert offsets[0].dtype == np.int32
    assert offsets[1].dtype == np.int32
    assert_allclose(offsets[0], expected_offsets1)
    assert_allclose(offsets[1], expected_offsets2)

    result = shapely.from_ragged_array(typ, coords, offsets)
    # in a roundtrip, missing geometries come back as empty
    arr[-2] = shapely.from_wkt("MULTILINESTRING EMPTY")
    assert_geometries_equal(result, arr)

    # sliced:
    # - indices into the coordinate array for the whole buffer
    # - indices into the line parts for *just* the sliced part
    offsets_sliced = (offsets[0], offsets[1][1:])
    result = shapely.from_ragged_array(typ, coords, offsets_sliced)
    assert_geometries_equal(result, arr[1:])

    offsets_sliced = (offsets[0], offsets[1][:-1])
    result = shapely.from_ragged_array(typ, coords, offsets_sliced)
    assert_geometries_equal(result, arr[:-1])


def test_multipolygons():
    arr = shapely.from_wkt(
        [
            "MULTIPOLYGON (((35 10, 45 45, 15 40, 10 20, 35 10), (20 30, 35 35, 30 20, 20 30)))",  # noqa: E501
            "MULTIPOLYGON (((40 40, 20 45, 45 30, 40 40)), ((20 35, 10 30, 10 10, 30 5, 45 20, 20 35), (30 20, 20 15, 20 25, 30 20)))",  # noqa: E501
            "MULTIPOLYGON EMPTY",
            "MULTIPOLYGON EMPTY",
            "MULTIPOLYGON (((40 40, 20 45, 45 30, 40 40)))",
            None,
            "MULTIPOLYGON EMPTY",
        ]
    )
    typ, coords, offsets = shapely.to_ragged_array(arr)
    expected = np.array(
        [
            [35.0, 10.0],
            [45.0, 45.0],
            [15.0, 40.0],
            [10.0, 20.0],
            [35.0, 10.0],
            [20.0, 30.0],
            [35.0, 35.0],
            [30.0, 20.0],
            [20.0, 30.0],
            [40.0, 40.0],
            [20.0, 45.0],
            [45.0, 30.0],
            [40.0, 40.0],
            [20.0, 35.0],
            [10.0, 30.0],
            [10.0, 10.0],
            [30.0, 5.0],
            [45.0, 20.0],
            [20.0, 35.0],
            [30.0, 20.0],
            [20.0, 15.0],
            [20.0, 25.0],
            [30.0, 20.0],
            [40.0, 40.0],
            [20.0, 45.0],
            [45.0, 30.0],
            [40.0, 40.0],
        ]
    )
    expected_offsets1 = np.array([0, 5, 9, 13, 19, 23, 27])
    expected_offsets2 = np.array([0, 2, 3, 5, 6])
    expected_offsets3 = np.array([0, 1, 3, 3, 3, 4, 4, 4])

    assert typ == shapely.GeometryType.MULTIPOLYGON
    assert_allclose(coords, expected)
    assert len(offsets) == 3
    assert offsets[0].dtype == np.int32
    assert offsets[1].dtype == np.int32
    assert offsets[2].dtype == np.int32
    assert_allclose(offsets[0], expected_offsets1)
    assert_allclose(offsets[1], expected_offsets2)
    assert_allclose(offsets[2], expected_offsets3)

    result = shapely.from_ragged_array(typ, coords, offsets)
    # in a roundtrip, missing geometries come back as empty
    arr[-2] = shapely.from_wkt("MULTIPOLYGON EMPTY")
    assert_geometries_equal(result, arr)

    # sliced:
    offsets_sliced = (offsets[0], offsets[1], offsets[2][1:])
    result = shapely.from_ragged_array(typ, coords, offsets_sliced)
    assert_geometries_equal(result, arr[1:])

    offsets_sliced = (offsets[0], offsets[1], offsets[2][:-3])
    result = shapely.from_ragged_array(typ, coords, offsets_sliced)
    assert_geometries_equal(result, arr[:-3])
    print(result)


def test_mixture_point_multipoint():
    typ, coords, offsets = shapely.to_ragged_array([point, multi_point])
    assert typ == shapely.GeometryType.MULTIPOINT
    result = shapely.from_ragged_array(typ, coords, offsets)
    expected = np.array([MultiPoint([point]), multi_point])
    assert_geometries_equal(result, expected)


def test_mixture_linestring_multilinestring():
    typ, coords, offsets = shapely.to_ragged_array([line_string, multi_line_string])
    assert typ == shapely.GeometryType.MULTILINESTRING
    result = shapely.from_ragged_array(typ, coords, offsets)
    expected = np.array([MultiLineString([line_string]), multi_line_string])
    assert_geometries_equal(result, expected)


def test_mixture_polygon_multipolygon():
    typ, coords, offsets = shapely.to_ragged_array([polygon, multi_polygon])
    assert typ == shapely.GeometryType.MULTIPOLYGON
    result = shapely.from_ragged_array(typ, coords, offsets)
    expected = np.array([MultiPolygon([polygon]), multi_polygon])
    assert_geometries_equal(result, expected)


def test_from_ragged_incorrect_rings_short():
    # too few coordinates for linearring
    coords = np.array([[0, 0], [1, 1]], dtype="float64")
    offsets1 = np.array([0, 2])
    offsets2 = np.array([0, 1])
    offsets3 = np.array([0, 1])

    with pytest.raises(
        ValueError, match="A linearring requires at least 4 coordinates"
    ):
        shapely.from_ragged_array(
            shapely.GeometryType.MULTIPOLYGON, coords, (offsets1, offsets2, offsets3)
        )

    with pytest.raises(
        ValueError, match="A linearring requires at least 4 coordinates"
    ):
        shapely.from_ragged_array(
            shapely.GeometryType.POLYGON, coords, (offsets1, offsets2)
        )


def test_from_ragged_incorrect_rings_unclosed():
    # NaNs cause the ring to be unclosed
    coords = np.full((4, 2), np.nan)
    offsets1 = np.array([0, 4])
    offsets2 = np.array([0, 1])
    offsets3 = np.array([0, 1])

    with pytest.raises(
        shapely.GEOSException,
        match="Points of LinearRing do not form a closed linestring",
    ):
        shapely.from_ragged_array(
            shapely.GeometryType.MULTIPOLYGON, coords, (offsets1, offsets2, offsets3)
        )

    with pytest.raises(
        shapely.GEOSException,
        match="Points of LinearRing do not form a closed linestring",
    ):
        shapely.from_ragged_array(
            shapely.GeometryType.POLYGON, coords, (offsets1, offsets2)
        )


def test_from_ragged_wrong_offsets():
    with pytest.raises(ValueError, match="'offsets' must be provided"):
        shapely.from_ragged_array(
            shapely.GeometryType.LINESTRING, np.array([[0, 0], [0, 1]])
        )

    with pytest.raises(ValueError, match="'offsets' should not be provided"):
        shapely.from_ragged_array(
            shapely.GeometryType.POINT,
            np.array([[0, 0], [0, 1]]),
            offsets=(np.array([0, 1]),),
        )


def test_from_ragged_crash_2284():
    # caused segfault in shapely 2.1.0
    # https://github.com/shapely/shapely/discussions/2284

    # one of the geometries has more rings than the total number of geometries
    coords = np.random.default_rng().random(120).reshape((60, 2))
    offsets1 = np.array([0, 10, 20, 30, 40, 50, 60])
    offsets2 = np.array([0, 1, 5, 6])

    for _ in range(10):
        polygons = shapely.from_ragged_array(
            shapely.GeometryType.POLYGON, coords, (offsets1, offsets2)
        )
        # just ensure it didn't crash
        assert len(polygons) == 3

    offsets3 = np.array([0, 3])
    for _ in range(10):
        polygons = shapely.from_ragged_array(
            shapely.GeometryType.MULTIPOLYGON, coords, (offsets1, offsets2, offsets3)
        )
        # just ensure it didn't crash
        assert len(polygons) == 1


def test_from_ragged_wrong_offsets_values():
    # caused segfault in shapely 2.1.0

    # outer offsets indicates more rings than the shape of the ring offsets
    coords = np.random.default_rng().random(70).reshape((35, 2))
    offsets1 = np.array([0, 10, 20], dtype=np.uint32)
    offsets2 = np.array([0, 1, 5], dtype=np.uint32)
    offsets3 = np.array([0, 2])

    with pytest.raises(
        ValueError, match="Number of rings indicated by the geometry offsets"
    ):
        shapely.from_ragged_array(
            shapely.GeometryType.POLYGON, coords, (offsets1, offsets2)
        )

    with pytest.raises(
        ValueError, match="Number of rings indicated by the part offsets"
    ):
        shapely.from_ragged_array(
            shapely.GeometryType.MULTIPOLYGON, coords, (offsets1, offsets2, offsets3)
        )

    # inner offsets indicating more coordinats that the shape of the coordinates
    coords = np.random.default_rng().random(70).reshape((35, 2))
    offsets1 = np.array([0, 10, 40], dtype=np.uint32)
    offsets2 = np.array([0, 1, 2], dtype=np.uint32)

    with pytest.raises(
        ValueError, match="Number of coordinates indicated by the linear offsets"
    ):
        shapely.from_ragged_array(
            shapely.GeometryType.POLYGON, coords, (offsets1, offsets2)
        )

    with pytest.raises(
        ValueError, match="Number of coordinates indicated by the linear offsets"
    ):
        shapely.from_ragged_array(
            shapely.GeometryType.MULTIPOLYGON, coords, (offsets1, offsets2, offsets3)
        )

    # outer multipolygon offsets indicating too many parts
    offsets3 = np.array([0, 3])
    with pytest.raises(
        ValueError, match="Number of geometry parts indicated by the geometry offsets"
    ):
        shapely.from_ragged_array(
            shapely.GeometryType.MULTIPOLYGON, coords, (offsets1, offsets2, offsets3)
        )
