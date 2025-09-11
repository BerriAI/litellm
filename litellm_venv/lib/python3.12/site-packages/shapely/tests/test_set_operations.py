import numpy as np
import pytest

import shapely
from shapely import Geometry, GeometryCollection, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types, empty, ignore_invalid, point, polygon

pytestmark = pytest.mark.filterwarnings(
    "ignore:The symmetric_difference_all function:DeprecationWarning"
)

# fixed-precision operations raise GEOS exceptions on mixed dimension geometry
# collections
all_single_types = np.array(all_types)[
    ~shapely.is_empty(all_types)
    & (shapely.get_type_id(all_types) != shapely.GeometryType.GEOMETRYCOLLECTION)
]

SET_OPERATIONS = (
    shapely.difference,
    shapely.intersection,
    shapely.symmetric_difference,
    shapely.union,
    # shapely.coverage_union is tested separately
)

REDUCE_SET_OPERATIONS = (
    (shapely.intersection_all, shapely.intersection),
    (shapely.symmetric_difference_all, shapely.symmetric_difference),
    (shapely.union_all, shapely.union),
    #  shapely.coverage_union_all, shapely.coverage_union) is tested separately
)

# operations that support fixed precision
REDUCE_SET_OPERATIONS_PREC = ((shapely.union_all, shapely.union),)

if shapely.geos_version >= (3, 12, 0):
    SET_OPERATIONS += (shapely.disjoint_subset_union,)
    REDUCE_SET_OPERATIONS += (
        (shapely.disjoint_subset_union_all, shapely.disjoint_subset_union),
    )

reduce_test_data = [
    shapely.box(0, 0, 5, 5),
    shapely.box(2, 2, 7, 7),
    shapely.box(4, 4, 9, 9),
    shapely.box(5, 5, 10, 10),
]

non_polygon_types = np.array(all_types)[
    ~shapely.is_empty(all_types) & (shapely.get_dimensions(all_types) != 2)
]


@pytest.mark.parametrize("a", all_types)
@pytest.mark.parametrize("func", SET_OPERATIONS)
def test_set_operation_array(a, func):
    if (
        func is shapely.difference
        and a.geom_type == "GeometryCollection"
        and shapely.get_num_geometries(a) == 2
        and shapely.geos_version == (3, 9, 5)
    ):
        pytest.xfail("GEOS 3.9.5 crashes with mixed collection")
    actual = func(a, point)
    assert isinstance(actual, Geometry)

    actual = func([a, a], point)
    assert actual.shape == (2,)
    assert isinstance(actual[0], Geometry)


@pytest.mark.parametrize("func", SET_OPERATIONS)
def test_set_operation_prec_nonscalar_grid_size(func):
    if func is shapely.disjoint_subset_union:
        pytest.skip("disjoint_subset_union does not support grid_size")
    with pytest.raises(
        ValueError, match="grid_size parameter only accepts scalar values"
    ):
        func(point, point, grid_size=[1])


@pytest.mark.parametrize("a", all_single_types)
@pytest.mark.parametrize("func", SET_OPERATIONS)
@pytest.mark.parametrize("grid_size", [0, 1, 2])
def test_set_operation_prec_array(a, func, grid_size):
    if func is shapely.disjoint_subset_union:
        pytest.skip("disjoint_subset_union does not support grid_size")
    actual = func([a, a], point, grid_size=grid_size)
    assert actual.shape == (2,)
    assert isinstance(actual[0], Geometry)

    # results should match the operation when the precision is previously set
    # to same grid_size
    b = shapely.set_precision(a, grid_size=grid_size)
    point2 = shapely.set_precision(point, grid_size=grid_size)
    expected = func([b, b], point2)

    assert shapely.equals(shapely.normalize(actual), shapely.normalize(expected)).all()


@pytest.mark.parametrize("n", range(1, 5))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_1dim(n, func, related_func):
    actual = func(reduce_test_data[:n])
    # perform the reduction in a python loop and compare
    expected = reduce_test_data[0]
    for i in range(1, n):
        expected = related_func(expected, reduce_test_data[i])
    assert shapely.equals(actual, expected)


@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_single_geom(func, related_func):
    geom = shapely.Point(1, 1)
    actual = func([geom, None, None])
    assert shapely.equals(actual, geom)


@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_axis(func, related_func):
    data = [[point] * 2] * 3  # shape = (3, 2)
    actual = func(data, axis=None)  # default
    assert isinstance(actual, Geometry)  # scalar output
    actual = func(data, axis=0)
    assert actual.shape == (2,)
    actual = func(data, axis=1)
    assert actual.shape == (3,)
    actual = func(data, axis=-1)
    assert actual.shape == (3,)


@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_empty(func, related_func):
    assert func(np.empty((0,), dtype=object)) == empty
    arr_empty_2D = np.empty((0, 2), dtype=object)
    assert func(arr_empty_2D) == empty
    assert func(arr_empty_2D, axis=0).tolist() == [empty] * 2
    assert func(arr_empty_2D, axis=1).tolist() == []


@pytest.mark.parametrize("none_position", range(3))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_one_none(func, related_func, none_position):
    # API change: before, intersection_all and symmetric_difference_all returned
    # None if any input geometry was None.
    # The new behaviour is to ignore None values.
    test_data = reduce_test_data[:2]
    test_data.insert(none_position, None)
    actual = func(test_data)
    expected = related_func(reduce_test_data[0], reduce_test_data[1])
    assert_geometries_equal(actual, expected)


@pytest.mark.parametrize("none_position", range(3))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_two_none(func, related_func, none_position):
    test_data = reduce_test_data[:2]
    test_data.insert(none_position, None)
    test_data.insert(none_position, None)
    actual = func(test_data)
    expected = related_func(reduce_test_data[0], reduce_test_data[1])
    assert_geometries_equal(actual, expected)


@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_some_none_len2(func, related_func):
    # in a previous implementation, this would take a different code path
    # and return wrong result
    assert func([empty, None]) == empty


@pytest.mark.parametrize("n", range(1, 3))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_all_none(n, func, related_func):
    assert_geometries_equal(func([None] * n), GeometryCollection([]))


@pytest.mark.parametrize("n", range(1, 3))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_all_none_arr(n, func, related_func):
    assert func([[None] * n] * 2, axis=1).tolist() == [empty, empty]
    assert func([[None] * 2] * n, axis=0).tolist() == [empty, empty]


@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS_PREC)
def test_set_operation_prec_reduce_nonscalar_grid_size(func, related_func):
    with pytest.raises(
        ValueError, match="grid_size parameter only accepts scalar values"
    ):
        func([point, point], grid_size=[1])


@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS_PREC)
def test_set_operation_prec_reduce_grid_size_nan(func, related_func):
    actual = func([point, point], grid_size=np.nan)
    assert actual is None


@pytest.mark.parametrize("n", range(1, 5))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS_PREC)
@pytest.mark.parametrize("grid_size", [0, 1])
def test_set_operation_prec_reduce_1dim(n, func, related_func, grid_size):
    actual = func(reduce_test_data[:n], grid_size=grid_size)
    # perform the reduction in a python loop and compare
    expected = reduce_test_data[0]
    for i in range(1, n):
        expected = related_func(expected, reduce_test_data[i], grid_size=grid_size)

    assert shapely.equals(actual, expected)


@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS_PREC)
def test_set_operation_prec_reduce_axis(func, related_func):
    data = [[point] * 2] * 3  # shape = (3, 2)
    actual = func(data, grid_size=1, axis=None)  # default
    assert isinstance(actual, Geometry)  # scalar output
    actual = func(data, grid_size=1, axis=0)
    assert actual.shape == (2,)
    actual = func(data, grid_size=1, axis=1)
    assert actual.shape == (3,)
    actual = func(data, grid_size=1, axis=-1)
    assert actual.shape == (3,)


@pytest.mark.parametrize("none_position", range(3))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS_PREC)
def test_set_operation_prec_reduce_one_none(func, related_func, none_position):
    test_data = reduce_test_data[:2]
    test_data.insert(none_position, None)
    actual = func(test_data, grid_size=1)
    expected = related_func(reduce_test_data[0], reduce_test_data[1], grid_size=1)
    assert_geometries_equal(actual, expected)


@pytest.mark.parametrize("none_position", range(3))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS_PREC)
def test_set_operation_prec_reduce_two_none(func, related_func, none_position):
    test_data = reduce_test_data[:2]
    test_data.insert(none_position, None)
    test_data.insert(none_position, None)
    actual = func(test_data, grid_size=1)
    expected = related_func(reduce_test_data[0], reduce_test_data[1], grid_size=1)
    assert_geometries_equal(actual, expected)


@pytest.mark.parametrize("n", range(1, 3))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS_PREC)
def test_set_operation_prec_reduce_all_none(n, func, related_func):
    assert_geometries_equal(func([None] * n, grid_size=1), GeometryCollection([]))


@pytest.mark.parametrize("n", range(1, 4))
def test_coverage_union_reduce_1dim(n):
    """
    This is tested separately from other set operations as it expects only
    non-overlapping polygons
    """
    test_data = [
        shapely.box(0, 0, 1, 1),
        shapely.box(1, 0, 2, 1),
        shapely.box(2, 0, 3, 1),
    ]
    actual = shapely.coverage_union_all(test_data[:n])
    # perform the reduction in a python loop and compare
    expected = test_data[0]
    for i in range(1, n):
        expected = shapely.coverage_union(expected, test_data[i])
    assert_geometries_equal(actual, expected, normalize=True)


def test_coverage_union_reduce_axis():
    # shape = (3, 2), all polygons - none of them overlapping
    data = [[shapely.box(i, j, i + 1, j + 1) for i in range(2)] for j in range(3)]
    actual = shapely.coverage_union_all(data, axis=None)  # default
    assert isinstance(actual, Geometry)
    actual = shapely.coverage_union_all(data, axis=0)
    assert actual.shape == (2,)
    actual = shapely.coverage_union_all(data, axis=1)
    assert actual.shape == (3,)
    actual = shapely.coverage_union_all(data, axis=-1)
    assert actual.shape == (3,)


def test_coverage_union_overlapping_inputs():
    polygon = Polygon([(1, 1), (1, 0), (0, 0), (0, 1), (1, 1)])
    other = Polygon([(1, 0), (0.9, 1), (2, 1), (2, 0), (1, 0)])

    if shapely.geos_version >= (3, 12, 0):
        # Return mostly unchanged output
        result = shapely.coverage_union(polygon, other)
        expected = shapely.multipolygons([polygon, other])
        assert_geometries_equal(result, expected, normalize=True)
    else:
        # Overlapping polygons raise an error
        with pytest.raises(
            shapely.GEOSException,
            match="CoverageUnion cannot process incorrectly noded inputs.",
        ):
            shapely.coverage_union(polygon, other)


@pytest.mark.parametrize(
    "geom_1, geom_2",
    # All possible polygon, non_polygon combinations
    [[polygon, non_polygon] for non_polygon in non_polygon_types]
    # All possible non_polygon, non_polygon combinations
    + [
        [non_polygon_1, non_polygon_2]
        for non_polygon_1 in non_polygon_types
        for non_polygon_2 in non_polygon_types
    ],
)
def test_coverage_union_non_polygon_inputs(geom_1, geom_2):
    if shapely.geos_version >= (3, 12, 0):

        def effective_geom_types(geom):
            if hasattr(geom, "geoms") and not geom.is_empty:
                gts = set()
                for part in geom.geoms:
                    gts |= effective_geom_types(part)
                return gts
            return {geom.geom_type.lstrip("Multi").replace("LinearRing", "LineString")}

        geom_types_1 = effective_geom_types(geom_1)
        geom_types_2 = effective_geom_types(geom_2)
        if len(geom_types_1) == 1 and geom_types_1 == geom_types_2:
            with ignore_invalid():
                # these show "invalid value encountered in coverage_union"
                result = shapely.coverage_union(geom_1, geom_2)
            assert geom_types_1 == effective_geom_types(result)
        else:
            with pytest.raises(
                shapely.GEOSException, match="Overlay input is mixed-dimension"
            ):
                shapely.coverage_union(geom_1, geom_2)
    else:
        # Non polygon geometries raise an error
        with pytest.raises(
            shapely.GEOSException, match="Unhandled geometry type in CoverageUnion."
        ):
            shapely.coverage_union(geom_1, geom_2)


@pytest.mark.parametrize(
    "geom,grid_size,expected",
    [
        # floating point precision, expect no change
        (
            [shapely.box(0.1, 0.1, 5, 5), shapely.box(0, 0.2, 5.1, 10)],
            0,
            Polygon(
                (
                    (0, 0.2),
                    (0, 10),
                    (5.1, 10),
                    (5.1, 0.2),
                    (5, 0.2),
                    (5, 0.1),
                    (0.1, 0.1),
                    (0.1, 0.2),
                    (0, 0.2),
                )
            ),
        ),
        # grid_size is at effective precision, expect no change
        (
            [shapely.box(0.1, 0.1, 5, 5), shapely.box(0, 0.2, 5.1, 10)],
            0.1,
            Polygon(
                (
                    (0, 0.2),
                    (0, 10),
                    (5.1, 10),
                    (5.1, 0.2),
                    (5, 0.2),
                    (5, 0.1),
                    (0.1, 0.1),
                    (0.1, 0.2),
                    (0, 0.2),
                )
            ),
        ),
        # grid_size forces rounding to nearest integer
        (
            [shapely.box(0.1, 0.1, 5, 5), shapely.box(0, 0.2, 5.1, 10)],
            1,
            Polygon([(0, 5), (0, 10), (5, 10), (5, 5), (5, 0), (0, 0), (0, 5)]),
        ),
        # grid_size much larger than effective precision causes rounding to nearest
        # multiple of 10
        (
            [shapely.box(0.1, 0.1, 5, 5), shapely.box(0, 0.2, 5.1, 10)],
            10,
            Polygon([(0, 10), (10, 10), (10, 0), (0, 0), (0, 10)]),
        ),
        # grid_size is so large that polygons collapse to empty
        (
            [shapely.box(0.1, 0.1, 5, 5), shapely.box(0, 0.2, 5.1, 10)],
            100,
            Polygon(),
        ),
    ],
)
def test_union_all_prec(geom, grid_size, expected):
    actual = shapely.union_all(geom, grid_size=grid_size)
    assert shapely.equals(actual, expected)


def test_uary_union_alias():
    geoms = [shapely.box(0.1, 0.1, 5, 5), shapely.box(0, 0.2, 5.1, 10)]
    actual = shapely.unary_union(geoms, grid_size=1)
    expected = shapely.union_all(geoms, grid_size=1)
    assert shapely.equals(actual, expected)


def test_difference_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `grid_size` for `difference` is deprecated"
    ):
        shapely.difference(point, point, None)


def test_intersection_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `grid_size` for `intersection` is deprecated"
    ):
        shapely.intersection(point, point, None)


def test_intersection_all_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `axis` for `intersection_all` is deprecated"
    ):
        shapely.intersection_all([point, point], None)


def test_symmetric_difference_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `grid_size` for `symmetric_difference` is deprecated"
    ):
        shapely.symmetric_difference(point, point, None)


def test_symmetric_difference_all_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `axis` for `symmetric_difference_all` is deprecated"
    ):
        shapely.symmetric_difference_all([point, point], None)


def test_union_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `grid_size` for `union` is deprecated"
    ):
        shapely.union(point, point, None)


def test_union_all_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `grid_size` for `union_all` is deprecated"
    ):
        shapely.union_all([point, point], None)
    with pytest.deprecated_call(
        match="positional arguments `grid_size` and `axis` for `union_all` "
        "are deprecated"
    ):
        shapely.union_all([point, point], None, None)


def test_coverage_union_all_deprecate_positional():
    data = [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)]
    with pytest.deprecated_call(
        match="positional argument `axis` for `coverage_union_all` is deprecated"
    ):
        shapely.coverage_union_all(data, None)
