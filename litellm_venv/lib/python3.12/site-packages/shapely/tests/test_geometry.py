import warnings

import numpy as np
import pytest

import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
    all_types,
    empty as empty_geometry_collection,
    empty_line_string,
    empty_line_string_z,
    empty_point,
    empty_point_z,
    empty_polygon,
    equal_geometries_abnormally_yield_unequal,
    geometry_collection,
    geometry_collection_z,
    ignore_invalid,
    ignore_warnings,
    line_string,
    line_string_nan,
    line_string_z,
    linear_ring,
    multi_line_string,
    multi_line_string_z,
    multi_point,
    multi_point_z,
    multi_polygon,
    multi_polygon_z,
    point,
    point_m,
    point_z,
    point_zm,
    polygon,
    polygon_with_hole,
    polygon_with_hole_z,
    polygon_z,
)


def test_get_num_points():
    actual = shapely.get_num_points(all_types + (None,)).tolist()
    assert actual == [0, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_get_num_interior_rings():
    actual = shapely.get_num_interior_rings(all_types + (None,)).tolist()
    assert actual == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_get_num_geometries():
    actual = shapely.get_num_geometries(all_types + (None,)).tolist()
    assert actual == [1, 1, 1, 1, 1, 2, 1, 2, 2, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0]


@pytest.mark.parametrize(
    "geom",
    [
        point,
        polygon,
        multi_point,
        multi_line_string,
        multi_polygon,
        geometry_collection,
    ],
)
def test_get_point_non_linestring(geom):
    actual = shapely.get_point(geom, [0, 2, -1])
    assert shapely.is_missing(actual).all()


@pytest.mark.parametrize("geom", [line_string, linear_ring])
def test_get_point(geom):
    n = shapely.get_num_points(geom)
    actual = shapely.get_point(geom, [0, -n, n, -(n + 1)])
    assert_geometries_equal(actual[0], actual[1])
    assert shapely.is_missing(actual[2:4]).all()


@pytest.mark.parametrize(
    "geom",
    [
        point,
        line_string,
        linear_ring,
        multi_point,
        multi_line_string,
        multi_polygon,
        geometry_collection,
    ],
)
def test_get_exterior_ring_non_polygon(geom):
    actual = shapely.get_exterior_ring(geom)
    assert shapely.is_missing(actual).all()


def test_get_exterior_ring():
    actual = shapely.get_exterior_ring([polygon, polygon_with_hole])
    assert (shapely.get_type_id(actual) == shapely.GeometryType.LINEARRING).all()


@pytest.mark.parametrize(
    "geom",
    [
        point,
        line_string,
        linear_ring,
        multi_point,
        multi_line_string,
        multi_polygon,
        geometry_collection,
    ],
)
def test_get_interior_ring_non_polygon(geom):
    actual = shapely.get_interior_ring(geom, [0, 2, -1])
    assert shapely.is_missing(actual).all()


def test_get_interior_ring():
    actual = shapely.get_interior_ring(polygon_with_hole, [0, -1, 1, -2])
    assert_geometries_equal(actual[0], actual[1])
    assert shapely.is_missing(actual[2:4]).all()


@pytest.mark.parametrize("geom", [point, line_string, linear_ring, polygon])
def test_get_geometry_simple(geom):
    actual = shapely.get_geometry(geom, [0, -1, 1, -2])
    assert_geometries_equal(actual[0], actual[1])
    assert shapely.is_missing(actual[2:4]).all()


@pytest.mark.parametrize(
    "geom", [multi_point, multi_line_string, multi_polygon, geometry_collection]
)
def test_get_geometry_collection(geom):
    n = shapely.get_num_geometries(geom)
    actual = shapely.get_geometry(geom, [0, -n, n, -(n + 1)])
    assert_geometries_equal(actual[0], actual[1])
    assert shapely.is_missing(actual[2:4]).all()


def test_get_type_id():
    actual = shapely.get_type_id(all_types + (None,)).tolist()
    assert actual == [0, 1, 2, 3, 3, 4, 5, 6, 7, 7, 0, 1, 3, 4, 5, 6, 4, 5, 6, 7, -1]


def test_get_dimensions():
    actual = shapely.get_dimensions(all_types + (None,)).tolist()
    assert actual == [0, 1, 1, 2, 2, 0, 1, 2, 1, -1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1, -1]


def test_get_coordinate_dimension():
    actual = shapely.get_coordinate_dimension([point, point_z, None]).tolist()
    assert actual == [2, 3, -1]


def test_get_num_coordinates():
    actual = shapely.get_num_coordinates(all_types + (None,)).tolist()
    assert actual == [1, 3, 5, 5, 10, 2, 2, 10, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_get_srid():
    """All geometry types have no SRID by default; None returns -1"""
    actual = shapely.get_srid(all_types + (None,)).tolist()
    assert actual == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]


def test_get_set_srid():
    actual = shapely.set_srid(point, 4326)
    assert shapely.get_srid(point) == 0
    assert shapely.get_srid(actual) == 4326


@pytest.mark.parametrize(
    "func",
    [
        shapely.get_x,
        shapely.get_y,
        shapely.get_z,
        pytest.param(
            shapely.get_m,
            marks=pytest.mark.skipif(
                shapely.geos_version < (3, 12, 0), reason="GEOS < 3.12"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "geom",
    np.array(all_types)[shapely.get_type_id(all_types) != shapely.GeometryType.POINT],
)
def test_get_xyz_no_point(func, geom):
    assert np.isnan(func(geom))


def test_get_x():
    assert shapely.get_x([point, point_z]).tolist() == [2.0, 2.0]


def test_get_y():
    assert shapely.get_y([point, point_z]).tolist() == [3.0, 3.0]


def test_get_z():
    assert shapely.get_z([point_z]).tolist() == [4.0]


def test_get_z_2d():
    assert np.isnan(shapely.get_z(point))


@pytest.mark.skipif(
    shapely.geos_version < (3, 12, 0),
    reason="M coordinates not supported with GEOS < 3.12",
)
def test_get_m():
    assert shapely.get_m([point_m, point_zm]).tolist() == [5.0, 5.0]
    assert np.isnan(shapely.get_m(point))
    assert np.isnan(shapely.get_m(point_z))


@pytest.mark.parametrize("geom", all_types)
def test_new_from_wkt(geom):
    actual = shapely.from_wkt(str(geom))
    if equal_geometries_abnormally_yield_unequal(geom):
        # abnormal test
        with pytest.raises(AssertionError):
            assert_geometries_equal(actual, geom)
    else:
        # normal test
        assert_geometries_equal(actual, geom)


def test_adapt_ptr_raises():
    point = Point(2, 2)
    with pytest.raises(AttributeError):
        point._geom += 1


@pytest.mark.parametrize("geom", all_types)
def test_set_unique(geom):
    a = {geom, shapely.transform(geom, lambda x: x)}
    assert len(a) == 1


def test_set_nan():
    # Although NaN != NaN, you cannot have multiple "NaN" points in a set
    # This is because "NaN" coordinates in a geometry are considered as equal.
    with ignore_invalid():
        a = set(shapely.linestrings([[[np.nan, np.nan], [np.nan, np.nan]]] * 10))
    assert len(a) == 1  # same objects: NaN == NaN (as geometry coordinates)


def test_set_nan_same_objects():
    # You can't put identical objects in a set.
    # x = float("nan"); set([x, x]) also returns a set with 1 element
    a = set([line_string_nan] * 10)
    assert len(a) == 1


@pytest.mark.parametrize(
    "geom",
    [
        point,
        multi_point,
        line_string,
        multi_line_string,
        polygon,
        multi_polygon,
        geometry_collection,
        empty_point,
        empty_line_string,
        empty_polygon,
        empty_geometry_collection,
        np.array([None]),
        np.empty_like(np.array([None])),
    ],
)
def test_get_parts(geom):
    expected_num_parts = shapely.get_num_geometries(geom)
    if expected_num_parts == 0:
        expected_parts = []
    else:
        expected_parts = shapely.get_geometry(geom, range(expected_num_parts))

    parts = shapely.get_parts(geom)
    assert len(parts) == expected_num_parts
    assert_geometries_equal(parts, expected_parts)


def test_get_parts_array():
    # note: this also verifies that None is handled correctly
    # in the mix; internally it returns -1 for count of geometries
    geom = np.array([None, empty_line_string, multi_point, point, multi_polygon])
    expected_parts = []
    for g in geom:
        for i in range(shapely.get_num_geometries(g)):
            expected_parts.append(shapely.get_geometry(g, i))

    parts = shapely.get_parts(geom)
    assert len(parts) == len(expected_parts)
    assert_geometries_equal(parts, expected_parts)


def test_get_parts_geometry_collection_multi():
    """On the first pass, the individual Multi* geometry objects are returned
    from the collection.  On the second pass, the individual singular geometry
    objects within those are returned.
    """
    geom = shapely.geometrycollections([multi_point, multi_line_string, multi_polygon])
    expected_num_parts = shapely.get_num_geometries(geom)
    expected_parts = shapely.get_geometry(geom, range(expected_num_parts))

    parts = shapely.get_parts(geom)
    assert len(parts) == expected_num_parts
    assert_geometries_equal(parts, expected_parts)

    expected_subparts = []
    for g in np.asarray(expected_parts):
        for i in range(shapely.get_num_geometries(g)):
            expected_subparts.append(shapely.get_geometry(g, i))

    subparts = shapely.get_parts(parts)
    assert len(subparts) == len(expected_subparts)
    assert_geometries_equal(subparts, expected_subparts)


def test_get_parts_return_index():
    geom = np.array([multi_point, point, multi_polygon])
    expected_parts = []
    expected_index = []
    for i, g in enumerate(geom):
        for j in range(shapely.get_num_geometries(g)):
            expected_parts.append(shapely.get_geometry(g, j))
            expected_index.append(i)

    parts, index = shapely.get_parts(geom, return_index=True)
    assert len(parts) == len(expected_parts)
    assert_geometries_equal(parts, expected_parts)
    assert np.array_equal(index, expected_index)


@pytest.mark.parametrize(
    "geom",
    ([[None]], [[empty_point]], [[multi_point]], [[multi_point, multi_line_string]]),
)
def test_get_parts_invalid_dimensions(geom):
    """Only 1D inputs are supported"""
    with pytest.raises(ValueError, match="Array should be one dimensional"):
        shapely.get_parts(geom)


@pytest.mark.parametrize("geom", [point, line_string, polygon])
def test_get_parts_non_multi(geom):
    """Non-multipart geometries should be returned identical to inputs"""
    assert_geometries_equal(geom, shapely.get_parts(geom))


@pytest.mark.parametrize("geom", [None, [None], []])
def test_get_parts_None(geom):
    assert len(shapely.get_parts(geom)) == 0


@pytest.mark.parametrize("geom", ["foo", ["foo"], 42])
def test_get_parts_invalid_geometry(geom):
    with pytest.raises(TypeError, match="One of the arguments is of incorrect type."):
        shapely.get_parts(geom)


def test_get_parts_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `return_index` for `get_parts` is deprecated"
    ):
        shapely.get_parts(multi_point, False)


@pytest.mark.parametrize(
    "geom",
    [
        point,
        multi_point,
        line_string,
        multi_line_string,
        polygon,
        multi_polygon,
        geometry_collection,
        empty_point,
        empty_line_string,
        empty_polygon,
        empty_geometry_collection,
        None,
    ],
)
def test_get_rings(geom):
    if (shapely.get_type_id(geom) != shapely.GeometryType.POLYGON) or shapely.is_empty(
        geom
    ):
        rings = shapely.get_rings(geom)
        assert len(rings) == 0
    else:
        rings = shapely.get_rings(geom)
        assert len(rings) == 1
        assert rings[0] == shapely.get_exterior_ring(geom)


def test_get_rings_holes():
    rings = shapely.get_rings(polygon_with_hole)
    assert len(rings) == 2
    assert rings[0] == shapely.get_exterior_ring(polygon_with_hole)
    assert rings[1] == shapely.get_interior_ring(polygon_with_hole, 0)


def test_get_rings_return_index():
    geom = np.array([polygon, None, empty_polygon, polygon_with_hole])
    expected_parts = []
    expected_index = []
    for i, g in enumerate(geom):
        if g is None or shapely.is_empty(g):
            continue
        expected_parts.append(shapely.get_exterior_ring(g))
        expected_index.append(i)
        for j in range(shapely.get_num_interior_rings(g)):
            expected_parts.append(shapely.get_interior_ring(g, j))
            expected_index.append(i)

    parts, index = shapely.get_rings(geom, return_index=True)
    assert len(parts) == len(expected_parts)
    assert_geometries_equal(parts, expected_parts)
    assert np.array_equal(index, expected_index)


@pytest.mark.parametrize("geom", [[[None]], [[polygon]]])
def test_get_rings_invalid_dimensions(geom):
    """Only 1D inputs are supported"""
    with pytest.raises(ValueError, match="Array should be one dimensional"):
        shapely.get_parts(geom)


def test_get_rings_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `return_index` for `get_rings` is deprecated"
    ):
        shapely.get_rings(polygon, False)


def test_get_precision():
    geometries = all_types + (point_z, empty_point, empty_line_string, empty_polygon)
    # default is 0
    actual = shapely.get_precision(geometries).tolist()
    assert actual == [0] * len(geometries)

    geometry = shapely.set_precision(geometries, 1)
    actual = shapely.get_precision(geometry).tolist()
    assert actual == [1] * len(geometries)


def test_get_precision_none():
    assert np.all(np.isnan(shapely.get_precision([None])))


@pytest.mark.parametrize("mode", ("valid_output", "pointwise", "keep_collapsed"))
def test_set_precision(mode):
    initial_geometry = Point(0.9, 0.9)
    assert shapely.get_precision(initial_geometry) == 0

    with ignore_warnings((3, 10, 0), UserWarning):
        # GEOS < 3.10 emits warning for 'pointwise'
        geometry = shapely.set_precision(initial_geometry, 0, mode=mode)
    assert shapely.get_precision(geometry) == 0
    assert_geometries_equal(geometry, initial_geometry)

    with ignore_warnings((3, 10, 0), UserWarning):
        geometry = shapely.set_precision(initial_geometry, 1, mode=mode)
    assert shapely.get_precision(geometry) == 1
    assert_geometries_equal(geometry, Point(1, 1))
    # original should remain unchanged
    assert_geometries_equal(initial_geometry, Point(0.9, 0.9))


def test_set_precision_drop_coords():
    # setting precision of 0 will not drop duplicated points in original
    geometry = shapely.set_precision(LineString([(0, 0), (0, 0), (0, 1), (1, 1)]), 0)
    assert_geometries_equal(geometry, LineString([(0, 0), (0, 0), (0, 1), (1, 1)]))

    # setting precision will remove duplicated points
    geometry = shapely.set_precision(geometry, 1)
    assert_geometries_equal(geometry, LineString([(0, 0), (0, 1), (1, 1)]))


@pytest.mark.parametrize("mode", ("valid_output", "pointwise", "keep_collapsed"))
def test_set_precision_z(mode):
    with ignore_warnings((3, 10, 0), UserWarning):
        # GEOS < 3.10 emits warning for 'pointwise'
        geometry = shapely.set_precision(Point(0.9, 0.9, 0.9), 1, mode=mode)
    assert shapely.get_precision(geometry) == 1
    assert_geometries_equal(geometry, Point(1, 1, 0.9))


@pytest.mark.parametrize("mode", ("valid_output", "pointwise", "keep_collapsed"))
def test_set_precision_nan(mode):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # GEOS emits warnings
        actual = shapely.set_precision(line_string_nan, 1, mode=mode)
        assert_geometries_equal(actual, line_string_nan)


def test_set_precision_none():
    assert shapely.set_precision(None, 0) is None


def test_set_precision_grid_size_nan():
    assert shapely.set_precision(Point(0.9, 0.9), np.nan) is None


@pytest.mark.parametrize(
    "geometry,mode,expected",
    [
        (
            Polygon([(2, 2), (4, 2), (3.2, 3), (4, 4), (2, 4), (2.8, 3), (2, 2)]),
            "valid_output",
            MultiPolygon(
                [
                    Polygon([(4, 2), (2, 2), (3, 3), (4, 2)]),
                    Polygon([(2, 4), (4, 4), (3, 3), (2, 4)]),
                ]
            ),
        ),
        pytest.param(
            Polygon([(2, 2), (4, 2), (3.2, 3), (4, 4), (2, 4), (2.8, 3), (2, 2)]),
            "pointwise",
            Polygon([(2, 2), (4, 2), (3, 3), (4, 4), (2, 4), (3, 3), (2, 2)]),
            marks=pytest.mark.skipif(
                shapely.geos_version < (3, 10, 0),
                reason="pointwise does not work pre-GEOS 3.10",
            ),
        ),
        (
            Polygon([(2, 2), (4, 2), (3.2, 3), (4, 4), (2, 4), (2.8, 3), (2, 2)]),
            "keep_collapsed",
            MultiPolygon(
                [
                    Polygon([(4, 2), (2, 2), (3, 3), (4, 2)]),
                    Polygon([(2, 4), (4, 4), (3, 3), (2, 4)]),
                ]
            ),
        ),
        (LineString([(0, 0), (0.1, 0.1)]), "valid_output", LineString()),
        pytest.param(
            LineString([(0, 0), (0.1, 0.1)]),
            "pointwise",
            LineString([(0, 0), (0, 0)]),
            marks=pytest.mark.skipif(
                shapely.geos_version < (3, 10, 0),
                reason="pointwise does not work pre-GEOS 3.10",
            ),
        ),
        (
            LineString([(0, 0), (0.1, 0.1)]),
            "keep_collapsed",
            LineString([(0, 0), (0, 0)]),
        ),
        pytest.param(
            LinearRing([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1), (0, 0)]),
            "valid_output",
            LinearRing(),
            marks=pytest.mark.skipif(
                shapely.geos_version == (3, 10, 0), reason="Segfaults on GEOS 3.10.0"
            ),
        ),
        pytest.param(
            LinearRing([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1), (0, 0)]),
            "pointwise",
            LinearRing([(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]),
            marks=pytest.mark.skipif(
                shapely.geos_version < (3, 10, 0),
                reason="pointwise does not work pre-GEOS 3.10",
            ),
        ),
        pytest.param(
            LinearRing([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1), (0, 0)]),
            "keep_collapsed",
            # See https://trac.osgeo.org/geos/ticket/1135#comment:5
            LineString([(0, 0), (0, 0), (0, 0)]),
            marks=pytest.mark.skipif(
                shapely.geos_version < (3, 10, 0),
                reason="this collapsed into an invalid linearring pre-GEOS 3.10",
            ),
        ),
        (
            Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1), (0, 0)]),
            "valid_output",
            Polygon(),
        ),
        pytest.param(
            Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1), (0, 0)]),
            "pointwise",
            Polygon([(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]),
            marks=pytest.mark.skipif(
                shapely.geos_version < (3, 10, 0),
                reason="pointwise does not work pre-GEOS 3.10",
            ),
        ),
        (
            Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1), (0, 0)]),
            "keep_collapsed",
            Polygon(),
        ),
    ],
)
def test_set_precision_collapse(geometry, mode, expected):
    """Lines and polygons collapse to empty geometries if vertices are too close"""
    actual = shapely.set_precision(geometry, 1, mode=mode)
    assert_geometries_equal(
        # force to 2D because of various dimension issues; GEOS GH-1152
        shapely.force_2d(actual),
        expected,
        normalize=shapely.geos_version == (3, 9, 0),
    )


def test_set_precision_intersection():
    """Operations should use the most precise precision grid size of the inputs"""

    box1 = shapely.normalize(shapely.box(0, 0, 0.9, 0.9))
    box2 = shapely.normalize(shapely.box(0.75, 0, 1.75, 0.75))

    assert shapely.get_precision(shapely.intersection(box1, box2)) == 0

    # GEOS will use and keep the most precise precision grid size
    box1 = shapely.set_precision(box1, 0.5)
    box2 = shapely.set_precision(box2, 1)
    out = shapely.intersection(box1, box2)
    assert shapely.get_precision(out) == 0.5
    assert_geometries_equal(out, LineString([(1, 1), (1, 0)]))


@pytest.mark.parametrize("preserve_topology", [False, True])
def set_precision_preserve_topology(preserve_topology):
    # the preserve_topology kwarg is deprecated (ignored)
    with pytest.warns(UserWarning):
        actual = shapely.set_precision(
            LineString([(0, 0), (0.1, 0.1)]),
            1.0,
            preserve_topology=preserve_topology,
        )
    assert_geometries_equal(shapely.force_2d(actual), LineString())


@pytest.mark.skipif(shapely.geos_version >= (3, 10, 0), reason="GEOS >= 3.10")
def set_precision_pointwise_pre_310():
    # using 'pointwise' emits a warning
    with pytest.warns(UserWarning):
        actual = shapely.set_precision(
            LineString([(0, 0), (0.1, 0.1)]),
            1.0,
            mode="pointwise",
        )
    assert_geometries_equal(shapely.force_2d(actual), LineString())


@pytest.mark.parametrize("flags", [np.array([0, 1]), 4, "foo"])
def set_precision_illegal_flags(flags):
    # the preserve_topology kwarg is deprecated (ignored)
    with pytest.raises((ValueError, TypeError)):
        shapely.lib.set_precision(line_string, 1.0, flags)


def test_empty():
    """Compatibility with empty_like, see GH373"""
    g = np.empty_like(np.array([None, None]))
    assert shapely.is_missing(g).all()


# corresponding to geometry_collection_z:
geometry_collection_2 = shapely.geometrycollections([point, line_string])


@pytest.mark.parametrize(
    "geom,expected",
    [
        (point, point),
        (point_z, point),
        (empty_point, empty_point),
        (empty_point_z, empty_point),
        (line_string, line_string),
        (line_string_z, line_string),
        (empty_line_string, empty_line_string),
        (empty_line_string_z, empty_line_string),
        (polygon, polygon),
        (polygon_z, polygon),
        (polygon_with_hole, polygon_with_hole),
        (polygon_with_hole_z, polygon_with_hole),
        (multi_point, multi_point),
        (multi_point_z, multi_point),
        (multi_line_string, multi_line_string),
        (multi_line_string_z, multi_line_string),
        (multi_polygon, multi_polygon),
        (multi_polygon_z, multi_polygon),
        (geometry_collection_2, geometry_collection_2),
        (geometry_collection_z, geometry_collection_2),
    ],
)
def test_force_2d(geom, expected):
    actual = shapely.force_2d(geom)
    assert shapely.get_coordinate_dimension(actual) == 2
    assert_geometries_equal(actual, expected)


@pytest.mark.parametrize(
    "geom,expected",
    [
        (point, point_z),
        (point_z, point_z),
        (empty_point, empty_point_z),
        (empty_point_z, empty_point_z),
        (line_string, line_string_z),
        (line_string_z, line_string_z),
        (empty_line_string, empty_line_string_z),
        (empty_line_string_z, empty_line_string_z),
        (polygon, polygon_z),
        (polygon_z, polygon_z),
        (polygon_with_hole, polygon_with_hole_z),
        (polygon_with_hole_z, polygon_with_hole_z),
        (multi_point, multi_point_z),
        (multi_point_z, multi_point_z),
        (multi_line_string, multi_line_string_z),
        (multi_line_string_z, multi_line_string_z),
        (multi_polygon, multi_polygon_z),
        (multi_polygon_z, multi_polygon_z),
        (geometry_collection_2, geometry_collection_z),
        (geometry_collection_z, geometry_collection_z),
    ],
)
def test_force_3d(geom, expected):
    actual = shapely.force_3d(geom, z=4)
    assert shapely.get_coordinate_dimension(actual) == 3
    assert_geometries_equal(actual, expected)
