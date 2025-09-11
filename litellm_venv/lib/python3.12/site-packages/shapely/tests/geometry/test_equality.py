import numpy as np
import pytest

import shapely
from shapely import LinearRing, LineString, MultiLineString, Point, Polygon
from shapely.tests.common import all_types, all_types_z, empty_point, ignore_invalid

all_non_empty_types = np.array(all_types + all_types_z)[
    ~shapely.is_empty(all_types + all_types_z)
]


# TODO add all_types_m and all_types_zm once tranform supports M coordinates
@pytest.mark.parametrize("geom", all_types + all_types_z)
def test_equality(geom):
    assert geom == geom  # noqa: PLR0124
    transformed = shapely.transform(geom, lambda x: x, include_z=True)
    if (
        shapely.geos_version < (3, 9, 0)
        and isinstance(geom, Point)
        and geom.is_empty
        and not geom.has_z
    ):
        # the transformed empty 2D point has become 3D on GEOS 3.8
        transformed = shapely.force_2d(geom)
    assert geom == transformed
    assert not (geom != transformed)


@pytest.mark.parametrize(
    "left, right",
    # automated test cases with transformed coordinate values
    [(geom, shapely.transform(geom, lambda x: x + 1)) for geom in all_non_empty_types]
    + [
        # (slightly) different coordinate values
        (LineString([(0, 0), (1, 1)]), LineString([(0, 0), (1, 2)])),
        (LineString([(0, 0), (1, 1)]), LineString([(0, 0), (1, 1 + 1e-12)])),
        # different coordinate order
        (LineString([(0, 0), (1, 1)]), LineString([(1, 1), (0, 0)])),
        # different number of coordinates (but spatially equal)
        (LineString([(0, 0), (1, 1)]), LineString([(0, 0), (1, 1), (1, 1)])),
        (LineString([(0, 0), (1, 1)]), LineString([(0, 0), (0.5, 0.5), (1, 1)])),
        # different order of sub-geometries
        (
            MultiLineString([[(1, 1), (2, 2)], [(2, 2), (3, 3)]]),
            MultiLineString([[(2, 2), (3, 3)], [(1, 1), (2, 2)]]),
        ),
        # M coordinates (don't work yet with automated cases)
        pytest.param(
            shapely.from_wkt("POINT M (0 0 0)"),
            shapely.from_wkt("POINT M (0 0 1)"),
            marks=pytest.mark.skipif(
                shapely.geos_version < (3, 12, 0), reason="GEOS < 3.12"
            ),
        ),
        pytest.param(
            shapely.from_wkt("POINT ZM (0 0 0 0)"),
            shapely.from_wkt("POINT ZM (0 0 0 1)"),
            marks=pytest.mark.skipif(
                shapely.geos_version < (3, 12, 0), reason="GEOS < 3.12"
            ),
        ),
    ],
)
def test_equality_false(left, right):
    assert left != right
    assert not (left == right)


with ignore_invalid():
    cases1 = [
        (LineString([(0, 1), (2, np.nan)]), LineString([(0, 1), (2, np.nan)])),
        (
            LineString([(0, 1), (np.nan, np.nan)]),
            LineString([(0, 1), (np.nan, np.nan)]),
        ),
        (LineString([(np.nan, 1), (2, 3)]), LineString([(np.nan, 1), (2, 3)])),
        (LineString([(0, np.nan), (2, 3)]), LineString([(0, np.nan), (2, 3)])),
        (
            LineString([(np.nan, np.nan), (np.nan, np.nan)]),
            LineString([(np.nan, np.nan), (np.nan, np.nan)]),
        ),
        # NaN as explicit Z coordinate
        # TODO: if first z is NaN -> considered as 2D -> tested below explicitly
        # (
        #     LineString([(0, 1, np.nan), (2, 3, np.nan)]),
        #     LineString([(0, 1, np.nan), (2, 3, np.nan)]),
        # ),
        (
            LineString([(0, 1, 2), (2, 3, np.nan)]),
            LineString([(0, 1, 2), (2, 3, np.nan)]),
        ),
        # (
        #     LineString([(0, 1, np.nan), (2, 3, 4)]),
        #     LineString([(0, 1, np.nan), (2, 3, 4)]),
        # ),
    ]


@pytest.mark.parametrize("left, right", cases1)
def test_equality_with_nan(left, right):
    assert left == right
    assert not (left != right)


with ignore_invalid():
    cases2 = [
        (
            LineString([(0, 1, np.nan), (2, 3, np.nan)]),
            LineString([(0, 1, np.nan), (2, 3, np.nan)]),
        ),
        (
            LineString([(0, 1, np.nan), (2, 3, 4)]),
            LineString([(0, 1, np.nan), (2, 3, 4)]),
        ),
    ]


@pytest.mark.parametrize("left, right", cases2)
def test_equality_with_nan_z(left, right):
    assert left == right
    assert not (left != right)


with ignore_invalid():
    cases3 = [
        (LineString([(0, np.nan), (2, 3)]), LineString([(0, 1), (2, 3)])),
        (LineString([(0, 1), (2, np.nan)]), LineString([(0, 1), (2, 3)])),
        (LineString([(0, 1, np.nan), (2, 3, 4)]), LineString([(0, 1, 2), (2, 3, 4)])),
        (LineString([(0, 1, 2), (2, 3, np.nan)]), LineString([(0, 1, 2), (2, 3, 4)])),
        pytest.param(
            shapely.from_wkt("POINT M (0 0 0)"),
            shapely.from_wkt("POINT M (0 0 NaN)"),
            marks=pytest.mark.skipif(
                shapely.geos_version < (3, 12, 0), reason="GEOS < 3.12"
            ),
        ),
        pytest.param(
            shapely.from_wkt("POINT ZM (0 0 0 0)"),
            shapely.from_wkt("POINT ZM (0 0 0 NaN)"),
            marks=pytest.mark.skipif(
                shapely.geos_version < (3, 12, 0), reason="GEOS < 3.12"
            ),
        ),
    ]


@pytest.mark.parametrize("left, right", cases3)
def test_equality_with_nan_false(left, right):
    assert left != right


def test_equality_with_nan_z_false():
    with ignore_invalid():
        left = LineString([(0, 1, np.nan), (2, 3, np.nan)])
        right = LineString([(0, 1, np.nan), (2, 3, 4)])

    if shapely.geos_version < (3, 10, 0):
        # GEOS <= 3.9 fill the NaN with 0, so the z dimension is different
        assert left != right
    elif shapely.geos_version < (3, 12, 0):
        # GEOS 3.10-3.11 ignore NaN for Z also when explicitly created with 3D
        # and so the geometries are considered as 2D (and thus z dimension is ignored)
        assert left == right
    else:
        assert left != right


def test_equality_z():
    # different dimensionality
    geom1 = Point(0, 1)
    geom2 = Point(0, 1, 0)
    assert geom1 != geom2

    # different dimensionality with NaN z
    geom2 = Point(0, 1, np.nan)
    if shapely.geos_version < (3, 12, 0):
        # GEOS 3.10-3.11 ignore NaN for Z also when explicitly created with 3D
        # and so the geometries are considered as 2D (and thus z dimension is ignored)
        assert geom1 == geom2
    else:
        assert geom1 != geom2


def test_equality_exact_type():
    # geometries with different type but same coord seq are not equal
    geom1 = LineString([(0, 0), (1, 1), (0, 1), (0, 0)])
    geom2 = LinearRing([(0, 0), (1, 1), (0, 1), (0, 0)])
    geom3 = Polygon([(0, 0), (1, 1), (0, 1), (0, 0)])
    assert geom1 != geom2
    assert geom1 != geom3
    assert geom2 != geom3

    # empty with different type
    geom1 = shapely.from_wkt("POINT EMPTY")
    geom2 = shapely.from_wkt("LINESTRING EMPTY")
    assert geom1 != geom2


def test_equality_polygon():
    # different exterior rings
    geom1 = shapely.from_wkt("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))")
    geom2 = shapely.from_wkt("POLYGON ((0 0, 10 0, 10 10, 0 15, 0 0))")
    assert geom1 != geom2

    # different number of holes
    geom1 = shapely.from_wkt(
        "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (1 1, 2 1, 2 2, 1 1))"
    )
    geom2 = shapely.from_wkt(
        "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (1 1, 2 1, 2 2, 1 1), "
        "(3 3, 4 3, 4 4, 3 3))"
    )
    assert geom1 != geom2

    # different order of holes
    geom1 = shapely.from_wkt(
        "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 4 3, 4 4, 3 3), "
        "(1 1, 2 1, 2 2, 1 1))"
    )
    geom2 = shapely.from_wkt(
        "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (1 1, 2 1, 2 2, 1 1), "
        "(3 3, 4 3, 4 4, 3 3))"
    )
    assert geom1 != geom2


@pytest.mark.parametrize("geom", all_types)
def test_comparison_notimplemented(geom):
    # comparing to a non-geometry class should return NotImplemented in __eq__
    # to ensure proper delegation to other (eg to ensure comparison of scalar
    # with array works)
    # https://github.com/shapely/shapely/issues/1056
    assert geom.__eq__(1) is NotImplemented

    # with array
    arr = np.array([geom, geom], dtype=object)

    result = arr == geom
    assert isinstance(result, np.ndarray)
    assert result.all()

    result = geom == arr
    assert isinstance(result, np.ndarray)
    assert result.all()

    result = arr != geom
    assert isinstance(result, np.ndarray)
    assert not result.any()

    result = geom != arr
    assert isinstance(result, np.ndarray)
    assert not result.any()


def test_comparison_not_supported():
    geom1 = Point(1, 1)
    geom2 = Point(2, 2)

    with pytest.raises(TypeError, match="not supported between instances"):
        assert geom1 > geom2

    with pytest.raises(TypeError, match="not supported between instances"):
        assert geom1 < geom2

    with pytest.raises(TypeError, match="not supported between instances"):
        assert geom1 >= geom2

    with pytest.raises(TypeError, match="not supported between instances"):
        assert geom1 <= geom2


@pytest.mark.parametrize(
    "geom", all_types + (shapely.points(np.nan, np.nan), empty_point)
)
def test_hash_same_equal(geom):
    hash1 = hash(geom)
    hash2 = hash(shapely.transform(geom, lambda x: x))
    assert hash1 == hash2, geom


@pytest.mark.parametrize("geom", all_non_empty_types)
def test_hash_same_not_equal(geom):
    assert hash(geom) != hash(shapely.transform(geom, lambda x: x + 1))
