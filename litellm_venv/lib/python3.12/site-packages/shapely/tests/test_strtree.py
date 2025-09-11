import itertools
import math
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import shapely
from shapely import LineString, MultiPoint, Point, STRtree, box, geos_version
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
    empty,
    empty_line_string,
    empty_point,
    ignore_invalid,
    point,
)

# the distance between 2 points spaced at whole numbers along a diagonal
HALF_UNIT_DIAG = math.sqrt(2) / 2
EPS = 1e-9


@pytest.fixture(scope="session")
def tree():
    geoms = shapely.points(np.arange(10), np.arange(10))
    yield STRtree(geoms)


@pytest.fixture(scope="session")
def line_tree():
    x = np.arange(10)
    y = np.arange(10)
    offset = 1
    geoms = shapely.linestrings(np.array([[x, x + offset], [y, y + offset]]).T)
    yield STRtree(geoms)


@pytest.fixture(scope="session")
def poly_tree():
    # create buffers so that midpoint between two buffers intersects
    # each buffer.  NOTE: add EPS to help mitigate rounding errors at midpoint.
    geoms = shapely.buffer(
        shapely.points(np.arange(10), np.arange(10)), HALF_UNIT_DIAG + EPS, quad_segs=32
    )
    yield STRtree(geoms)


@pytest.mark.parametrize(
    "geometry,count, hits",
    [
        # Empty array produces empty tree
        ([], 0, 0),
        ([point], 1, 1),
        # None geometries are ignored when creating tree
        ([None], 0, 0),
        ([point, None], 1, 1),
        # empty geometries are ignored when creating tree
        ([empty, empty_point, empty_line_string], 0, 0),
        # only the valid geometry should have a hit
        ([empty, point, empty_point, empty_line_string], 1, 1),
    ],
)
def test_init(geometry, count, hits):
    tree = STRtree(geometry)
    assert len(tree) == count
    assert tree.query(box(0, 0, 100, 100)).size == hits


def test_init_with_invalid_geometry():
    with pytest.raises(TypeError):
        STRtree(["Not a geometry"])


def test_references():
    point1 = Point()
    point2 = Point(0, 1)

    geoms = [point1, point2]
    tree = STRtree(geoms)

    point1 = None
    point2 = None

    import gc

    gc.collect()

    # query after freeing geometries does not lead to segfault
    assert tree.query(box(0, 0, 1, 1)).tolist() == [1]


def test_flush_geometries():
    arr = shapely.points(np.arange(10), np.arange(10))
    tree = STRtree(arr)

    # Dereference geometries
    arr[:] = None
    import gc

    gc.collect()
    # Still it does not lead to a segfault
    tree.query(point)


def test_geometries_property():
    arr = np.array([point])
    tree = STRtree(arr)
    assert_geometries_equal(arr, tree.geometries)

    # modifying elements of input should not modify tree.geometries
    arr[0] = shapely.Point(0, 0)
    assert_geometries_equal(point, tree.geometries[0])


def test_pickle_persistence(tmp_path):
    # write the pickeled tree to another process; the process should not crash
    tree = STRtree([Point(i, i).buffer(0.1) for i in range(3)])

    pickled_strtree = pickle.dumps(tree)
    unpickle_script = """
import pickle
import sys

from shapely import Point

pickled_strtree = sys.stdin.buffer.read()
print("received pickled strtree:", repr(pickled_strtree))
tree = pickle.loads(pickled_strtree)

tree.query(Point(0, 0))
tree.nearest(Point(0, 0))
print("done")
"""

    filename = tmp_path / "unpickle-strtree.py"
    with open(filename, "w") as out:
        out.write(unpickle_script)

    proc = subprocess.Popen(
        [sys.executable, str(filename)],
        stdin=subprocess.PIPE,
    )
    proc.communicate(input=pickled_strtree)
    proc.wait()
    assert proc.returncode == 0


@pytest.mark.parametrize(
    "geometry",
    [
        "I am not a geometry",
        ["I am not a geometry"],
        [Point(0, 0), "still not a geometry"],
        [[], "in a mixed array", 1],
    ],
)
@pytest.mark.filterwarnings("ignore:Creating an ndarray from ragged nested sequences:")
def test_query_invalid_geometry(tree, geometry):
    with pytest.raises((TypeError, ValueError)):
        tree.query(geometry)


def test_query_invalid_dimension(tree):
    with pytest.raises(TypeError, match="Array should be one dimensional"):
        tree.query([[Point(0.5, 0.5)]])


@pytest.mark.parametrize(
    "tree_geometry, geometry,expected",
    [
        # Empty tree returns no results
        ([], point, []),
        ([], [point], [[], []]),
        ([], None, []),
        ([], [None], [[], []]),
        # Tree with only None returns no results
        ([None], point, []),
        ([None], [point], [[], []]),
        ([None], None, []),
        ([None], [None], [[], []]),
        # querying with None returns no results
        ([point], None, []),
        ([point], [None], [[], []]),
        # Empty is included in the tree, but ignored when querying the tree
        ([empty], empty, []),
        ([empty], [empty], [[], []]),
        ([empty], point, []),
        ([empty], [point], [[], []]),
        ([point, empty], empty, []),
        ([point, empty], [empty], [[], []]),
        # None and empty are ignored in the tree, but the index of the valid
        # geometry should be retained.
        ([None, point], box(0, 0, 10, 10), [1]),
        ([None, point], [box(0, 0, 10, 10)], [[0], [1]]),
        ([None, empty, point], box(0, 0, 10, 10), [2]),
        ([point, None, point], box(0, 0, 10, 10), [0, 2]),
        ([point, None, point], [box(0, 0, 10, 10)], [[0, 0], [0, 2]]),
        # Only the non-empty query geometry gets hits
        ([empty, point], [empty, point], [[1], [1]]),
        (
            [empty, empty_point, empty_line_string, point],
            [empty, empty_point, empty_line_string, point],
            [[3], [3]],
        ),
    ],
)
def test_query_with_none_and_empty(tree_geometry, geometry, expected):
    tree = STRtree(tree_geometry)
    assert_array_equal(tree.query(geometry), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (Point(0.5, 0.5), []),
        ([Point(0.5, 0.5)], [[], []]),
        # points intersect
        (Point(1, 1), [1]),
        ([Point(1, 1)], [[0], [1]]),
        # first and last points intersect
        (
            [Point(1, 1), Point(-1, -1), Point(2, 2)],
            [[0, 2], [1, 2]],
        ),
        # box contains points
        (box(0, 0, 1, 1), [0, 1]),
        ([box(0, 0, 1, 1)], [[0, 0], [0, 1]]),
        # bigger box contains more points
        (box(5, 5, 15, 15), [5, 6, 7, 8, 9]),
        ([box(5, 5, 15, 15)], [[0, 0, 0, 0, 0], [5, 6, 7, 8, 9]]),
        # first and last boxes contains points
        (
            [box(0, 0, 1, 1), box(100, 100, 110, 110), box(5, 5, 15, 15)],
            [[0, 0, 2, 2, 2, 2, 2], [0, 1, 5, 6, 7, 8, 9]],
        ),
        # envelope of buffer contains points
        (shapely.buffer(Point(3, 3), 1), [2, 3, 4]),
        ([shapely.buffer(Point(3, 3), 1)], [[0, 0, 0], [2, 3, 4]]),
        # envelope of points contains points
        (MultiPoint([[5, 7], [7, 5]]), [5, 6, 7]),
        ([MultiPoint([[5, 7], [7, 5]])], [[0, 0, 0], [5, 6, 7]]),
    ],
)
def test_query_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects first line
        (Point(0, 0), [0]),
        ([Point(0, 0)], [[0], [0]]),
        (Point(0.5, 0.5), [0]),
        ([Point(0.5, 0.5)], [[0], [0]]),
        # point within envelope of first line
        (Point(0, 0.5), [0]),
        ([Point(0, 0.5)], [[0], [0]]),
        # point at shared vertex between 2 lines
        (Point(1, 1), [0, 1]),
        ([Point(1, 1)], [[0, 0], [0, 1]]),
        # box overlaps envelope of first 2 lines (touches edge of 1)
        (box(0, 0, 1, 1), [0, 1]),
        ([box(0, 0, 1, 1)], [[0, 0], [0, 1]]),
        # envelope of buffer overlaps envelope of 2 lines
        (shapely.buffer(Point(3, 3), 0.5), [2, 3]),
        ([shapely.buffer(Point(3, 3), 0.5)], [[0, 0], [2, 3]]),
        # envelope of points overlaps 5 lines (touches edge of 2 envelopes)
        (MultiPoint([[5, 7], [7, 5]]), [4, 5, 6, 7]),
        ([MultiPoint([[5, 7], [7, 5]])], [[0, 0, 0, 0], [4, 5, 6, 7]]),
    ],
)
def test_query_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects edge of envelopes of 2 polygons
        (Point(0.5, 0.5), [0, 1]),
        ([Point(0.5, 0.5)], [[0, 0], [0, 1]]),
        # point intersects single polygon
        (Point(1, 1), [1]),
        ([Point(1, 1)], [[0], [1]]),
        # box overlaps envelope of 2 polygons
        (box(0, 0, 1, 1), [0, 1]),
        ([box(0, 0, 1, 1)], [[0, 0], [0, 1]]),
        # larger box overlaps envelope of 3 polygons
        (box(0, 0, 1.5, 1.5), [0, 1, 2]),
        ([box(0, 0, 1.5, 1.5)], [[0, 0, 0], [0, 1, 2]]),
        # first and last boxes overlap envelope of 2 polyons
        (
            [box(0, 0, 1, 1), box(100, 100, 110, 110), box(2, 2, 3, 3)],
            [[0, 0, 2, 2], [0, 1, 2, 3]],
        ),
        # envelope of buffer overlaps envelope of 3 polygons
        (shapely.buffer(Point(3, 3), HALF_UNIT_DIAG), [2, 3, 4]),
        (
            [shapely.buffer(Point(3, 3), HALF_UNIT_DIAG)],
            [[0, 0, 0], [2, 3, 4]],
        ),
        # envelope of larger buffer overlaps envelope of 6 polygons
        (shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG), [1, 2, 3, 4, 5]),
        (
            [shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG)],
            [[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]],
        ),
        # envelope of points overlaps 3 polygons
        (MultiPoint([[5, 7], [7, 5]]), [5, 6, 7]),
        ([MultiPoint([[5, 7], [7, 5]])], [[0, 0, 0], [5, 6, 7]]),
    ],
)
def test_query_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry), expected)


@pytest.mark.parametrize(
    "predicate",
    [
        "bad_predicate",
        # disjoint is a valid GEOS binary predicate, but not supported for query
        "disjoint",
    ],
)
def test_query_invalid_predicate(tree, predicate):
    with pytest.raises(ValueError, match="is not a valid option"):
        tree.query(Point(1, 1), predicate=predicate)


@pytest.mark.parametrize(
    "predicate,expected",
    [
        ("intersects", [0, 1, 2]),
        ("within", []),
        ("contains", [1]),
        ("overlaps", []),
        ("crosses", []),
        ("covers", [0, 1, 2]),
        ("covered_by", []),
        ("contains_properly", [1]),
    ],
)
def test_query_prepared_inputs(tree, predicate, expected):
    geom = box(0, 0, 2, 2)
    shapely.prepare(geom)
    assert_array_equal(tree.query(geom, predicate=predicate), expected)


def test_query_with_partially_prepared_inputs(tree):
    geom = np.array([box(0, 0, 1, 1), box(3, 3, 5, 5)])
    expected = tree.query(geom, predicate="intersects")

    # test with array of partially prepared geometries
    shapely.prepare(geom[0])
    assert_array_equal(expected, tree.query(geom, predicate="intersects"))


@pytest.mark.parametrize(
    "predicate,expected",
    [
        pytest.param(
            "intersects",
            [1],
            marks=pytest.mark.xfail(geos_version < (3, 13, 0), reason="GEOS < 3.13"),
        ),
        ("within", []),
        ("contains", []),
        ("overlaps", []),
        ("crosses", [1]),
        ("touches", []),
        ("covers", []),
        ("covered_by", []),
        ("contains_properly", []),
    ],
)
def test_query_predicate_errors(tree, predicate, expected):
    with ignore_invalid():
        line_nan = shapely.linestrings([1, 1], [1, float("nan")])
    if geos_version < (3, 13, 0):
        with pytest.raises(shapely.GEOSException):
            tree.query(line_nan, predicate=predicate)
    else:
        assert_array_equal(tree.query(line_nan, predicate=predicate), expected)


### predicate == 'intersects'


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (Point(0.5, 0.5), []),
        ([Point(0.5, 0.5)], [[], []]),
        # points intersect
        (Point(1, 1), [1]),
        ([Point(1, 1)], [[0], [1]]),
        # box contains points
        (box(3, 3, 6, 6), [3, 4, 5, 6]),
        ([box(3, 3, 6, 6)], [[0, 0, 0, 0], [3, 4, 5, 6]]),
        # first and last boxes contain points
        (
            [box(0, 0, 1, 1), box(100, 100, 110, 110), box(3, 3, 6, 6)],
            [[0, 0, 2, 2, 2, 2], [0, 1, 3, 4, 5, 6]],
        ),
        # envelope of buffer contains more points than intersect buffer
        # due to diagonal distance
        (shapely.buffer(Point(3, 3), 1), [3]),
        ([shapely.buffer(Point(3, 3), 1)], [[0], [3]]),
        # envelope of buffer with 1/2 distance between points should intersect
        # same points as envelope
        (shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG), [2, 3, 4]),
        (
            [shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG)],
            [[0, 0, 0], [2, 3, 4]],
        ),
        # multipoints intersect
        (
            MultiPoint([[5, 5], [7, 7]]),
            [5, 7],
        ),
        (
            [MultiPoint([[5, 5], [7, 7]])],
            [[0, 0], [5, 7]],
        ),
        # envelope of points contains points, but points do not intersect
        (MultiPoint([[5, 7], [7, 5]]), []),
        ([MultiPoint([[5, 7], [7, 5]])], [[], []]),
        # only one point of multipoint intersects
        (
            MultiPoint([[5, 7], [7, 7]]),
            [7],
        ),
        (
            [MultiPoint([[5, 7], [7, 7]])],
            [[0], [7]],
        ),
    ],
)
def test_query_intersects_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate="intersects"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects first line
        (Point(0, 0), [0]),
        ([Point(0, 0)], [[0], [0]]),
        (Point(0.5, 0.5), [0]),
        ([Point(0.5, 0.5)], [[0], [0]]),
        # point within envelope of first line but does not intersect
        (Point(0, 0.5), []),
        ([Point(0, 0.5)], [[], []]),
        # point at shared vertex between 2 lines
        (Point(1, 1), [0, 1]),
        ([Point(1, 1)], [[0, 0], [0, 1]]),
        # box overlaps envelope of first 2 lines (touches edge of 1)
        (box(0, 0, 1, 1), [0, 1]),
        ([box(0, 0, 1, 1)], [[0, 0], [0, 1]]),
        # first and last boxes overlap multiple lines each
        (
            [box(0, 0, 1, 1), box(100, 100, 110, 110), box(2, 2, 3, 3)],
            [[0, 0, 2, 2, 2], [0, 1, 1, 2, 3]],
        ),
        # buffer intersects 2 lines
        (shapely.buffer(Point(3, 3), 0.5), [2, 3]),
        ([shapely.buffer(Point(3, 3), 0.5)], [[0, 0], [2, 3]]),
        # buffer intersects midpoint of line at tangent
        (shapely.buffer(Point(2, 1), HALF_UNIT_DIAG), [1]),
        ([shapely.buffer(Point(2, 1), HALF_UNIT_DIAG)], [[0], [1]]),
        # envelope of points overlaps lines but intersects none
        (MultiPoint([[5, 7], [7, 5]]), []),
        ([MultiPoint([[5, 7], [7, 5]])], [[], []]),
        # only one point of multipoint intersects
        (MultiPoint([[5, 7], [7, 7]]), [6, 7]),
        ([MultiPoint([[5, 7], [7, 7]])], [[0, 0], [6, 7]]),
    ],
)
def test_query_intersects_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry, predicate="intersects"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point within first polygon
        (Point(0, 0.5), [0]),
        ([Point(0, 0.5)], [[0], [0]]),
        (Point(0.5, 0), [0]),
        ([Point(0.5, 0)], [[0], [0]]),
        # midpoint between two polygons intersects both
        (Point(0.5, 0.5), [0, 1]),
        ([Point(0.5, 0.5)], [[0, 0], [0, 1]]),
        # point intersects single polygon
        (Point(1, 1), [1]),
        ([Point(1, 1)], [[0], [1]]),
        # box overlaps envelope of 2 polygons
        (box(0, 0, 1, 1), [0, 1]),
        ([box(0, 0, 1, 1)], [[0, 0], [0, 1]]),
        # larger box intersects 3 polygons
        (box(0, 0, 1.5, 1.5), [0, 1, 2]),
        ([box(0, 0, 1.5, 1.5)], [[0, 0, 0], [0, 1, 2]]),
        # first and last boxes overlap
        (
            [box(0, 0, 1, 1), box(100, 100, 110, 110), box(2, 2, 3, 3)],
            [[0, 0, 2, 2], [0, 1, 2, 3]],
        ),
        # buffer overlaps 3 polygons
        (shapely.buffer(Point(3, 3), HALF_UNIT_DIAG), [2, 3, 4]),
        (
            [shapely.buffer(Point(3, 3), HALF_UNIT_DIAG)],
            [[0, 0, 0], [2, 3, 4]],
        ),
        # larger buffer overlaps 6 polygons (touches midpoints)
        (shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG), [1, 2, 3, 4, 5]),
        (
            [shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG)],
            [[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]],
        ),
        # envelope of points overlaps polygons, but points do not intersect
        (MultiPoint([[5, 7], [7, 5]]), []),
        ([MultiPoint([[5, 7], [7, 5]])], [[], []]),
        # only one point of multipoint within polygon
        (MultiPoint([[5, 7], [7, 7]]), [7]),
        ([MultiPoint([[5, 7], [7, 7]])], [[0], [7]]),
    ],
)
def test_query_intersects_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry, predicate="intersects"), expected)


### predicate == 'within'
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (Point(0.5, 0.5), []),
        ([Point(0.5, 0.5)], [[], []]),
        # points intersect
        (Point(1, 1), [1]),
        ([Point(1, 1)], [[0], [1]]),
        # box not within points
        (box(3, 3, 6, 6), []),
        ([box(3, 3, 6, 6)], [[], []]),
        # envelope of buffer not within points
        (shapely.buffer(Point(3, 3), 1), []),
        ([shapely.buffer(Point(3, 3), 1)], [[], []]),
        # multipoints intersect but are not within points in tree
        (MultiPoint([[5, 5], [7, 7]]), []),
        ([MultiPoint([[5, 5], [7, 7]])], [[], []]),
        # only one point of multipoint intersects, but multipoints are not
        # within any points in tree
        (MultiPoint([[5, 7], [7, 7]]), []),
        ([MultiPoint([[5, 7], [7, 7]])], [[], []]),
        # envelope of points contains points, but points do not intersect
        (MultiPoint([[5, 7], [7, 5]]), []),
        ([MultiPoint([[5, 7], [7, 5]])], [[], []]),
    ],
)
def test_query_within_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate="within"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # endpoint not within first line
        (Point(0, 0), []),
        ([Point(0, 0)], [[], []]),
        # point within first line
        (Point(0.5, 0.5), [0]),
        ([Point(0.5, 0.5)], [[0], [0]]),
        # point within envelope of first line but does not intersect
        (Point(0, 0.5), []),
        ([Point(0, 0.5)], [[], []]),
        # point at shared vertex between 2 lines (but within neither)
        (Point(1, 1), []),
        ([Point(1, 1)], [[], []]),
        # box not within line
        (box(0, 0, 1, 1), []),
        ([box(0, 0, 1, 1)], [[], []]),
        # buffer intersects 2 lines but not within either
        (shapely.buffer(Point(3, 3), 0.5), []),
        ([shapely.buffer(Point(3, 3), 0.5)], [[], []]),
        # envelope of points overlaps lines but intersects none
        (MultiPoint([[5, 7], [7, 5]]), []),
        ([MultiPoint([[5, 7], [7, 5]])], [[], []]),
        # only one point of multipoint intersects, but both are not within line
        (MultiPoint([[5, 7], [7, 7]]), []),
        ([MultiPoint([[5, 7], [7, 7]])], [[], []]),
        (MultiPoint([[6.5, 6.5], [7, 7]]), [6]),
        ([MultiPoint([[6.5, 6.5], [7, 7]])], [[0], [6]]),
    ],
)
def test_query_within_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry, predicate="within"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point within first polygon
        (Point(0, 0.5), [0]),
        ([Point(0, 0.5)], [[0], [0]]),
        (Point(0.5, 0), [0]),
        ([Point(0.5, 0)], [[0], [0]]),
        # midpoint between two polygons intersects both
        (Point(0.5, 0.5), [0, 1]),
        ([Point(0.5, 0.5)], [[0, 0], [0, 1]]),
        # point intersects single polygon
        (Point(1, 1), [1]),
        ([Point(1, 1)], [[0], [1]]),
        # box overlaps envelope of 2 polygons but within neither
        (box(0, 0, 1, 1), []),
        ([box(0, 0, 1, 1)], [[], []]),
        # box within polygon
        (box(0, 0, 0.5, 0.5), [0]),
        ([box(0, 0, 0.5, 0.5)], [[0], [0]]),
        # larger box intersects 3 polygons but within none
        (box(0, 0, 1.5, 1.5), []),
        ([box(0, 0, 1.5, 1.5)], [[], []]),
        # buffer intersects 3 polygons but only within one
        (shapely.buffer(Point(3, 3), HALF_UNIT_DIAG), [3]),
        ([shapely.buffer(Point(3, 3), HALF_UNIT_DIAG)], [[0], [3]]),
        # larger buffer overlaps 6 polygons (touches midpoints) but within none
        (shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG), []),
        ([shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG)], [[], []]),
        # envelope of points overlaps polygons, but points do not intersect
        (MultiPoint([[5, 7], [7, 5]]), []),
        ([MultiPoint([[5, 7], [7, 5]])], [[], []]),
        # only one point of multipoint within polygon
        (MultiPoint([[5, 7], [7, 7]]), []),
        ([MultiPoint([[5, 7], [7, 7]])], [[], []]),
        # both points in multipoint within polygon
        (MultiPoint([[5.25, 5.5], [5.25, 5.0]]), [5]),
        ([MultiPoint([[5.25, 5.5], [5.25, 5.0]])], [[0], [5]]),
    ],
)
def test_query_within_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry, predicate="within"), expected)


### predicate == 'contains'
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (Point(0.5, 0.5), []),
        ([Point(0.5, 0.5)], [[], []]),
        # points intersect
        (Point(1, 1), [1]),
        ([Point(1, 1)], [[0], [1]]),
        # box contains points (2 are at edges and not contained)
        (box(3, 3, 6, 6), [4, 5]),
        ([box(3, 3, 6, 6)], [[0, 0], [4, 5]]),
        # envelope of buffer contains more points than within buffer
        # due to diagonal distance
        (shapely.buffer(Point(3, 3), 1), [3]),
        ([shapely.buffer(Point(3, 3), 1)], [[0], [3]]),
        # envelope of buffer with 1/2 distance between points should intersect
        # same points as envelope
        (shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG), [2, 3, 4]),
        ([shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG)], [[0, 0, 0], [2, 3, 4]]),
        # multipoints intersect
        (MultiPoint([[5, 5], [7, 7]]), [5, 7]),
        ([MultiPoint([[5, 5], [7, 7]])], [[0, 0], [5, 7]]),
        # envelope of points contains points, but points do not intersect
        (MultiPoint([[5, 7], [7, 5]]), []),
        ([MultiPoint([[5, 7], [7, 5]])], [[], []]),
        # only one point of multipoint intersects
        (MultiPoint([[5, 7], [7, 7]]), [7]),
        ([MultiPoint([[5, 7], [7, 7]])], [[0], [7]]),
    ],
)
def test_query_contains_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate="contains"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point does not contain any lines (not valid relation)
        (Point(0, 0), []),
        ([Point(0, 0)], [[], []]),
        # box contains first line (touches edge of 1 but does not contain it)
        (box(0, 0, 1, 1), [0]),
        ([box(0, 0, 1, 1)], [[0], [0]]),
        # buffer intersects 2 lines but contains neither
        (shapely.buffer(Point(3, 3), 0.5), []),
        ([shapely.buffer(Point(3, 3), 0.5)], [[], []]),
        # envelope of points overlaps lines but intersects none
        (MultiPoint([[5, 7], [7, 5]]), []),
        ([MultiPoint([[5, 7], [7, 5]])], [[], []]),
        # only one point of multipoint intersects
        (MultiPoint([[5, 7], [7, 7]]), []),
        ([MultiPoint([[5, 7], [7, 7]])], [[], []]),
        # both points intersect but do not contain any lines (not valid relation)
        (MultiPoint([[5, 5], [6, 6]]), []),
        ([MultiPoint([[5, 5], [6, 6]])], [[], []]),
    ],
)
def test_query_contains_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry, predicate="contains"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point does not contain any polygons (not valid relation)
        (Point(0, 0), []),
        ([Point(0, 0)], [[], []]),
        # box overlaps envelope of 2 polygons but contains neither
        (box(0, 0, 1, 1), []),
        ([box(0, 0, 1, 1)], [[], []]),
        # larger box intersects 3 polygons but contains only one
        (box(0, 0, 2, 2), [1]),
        ([box(0, 0, 2, 2)], [[0], [1]]),
        # buffer overlaps 3 polygons but contains none
        (shapely.buffer(Point(3, 3), HALF_UNIT_DIAG), []),
        ([shapely.buffer(Point(3, 3), HALF_UNIT_DIAG)], [[], []]),
        # larger buffer overlaps 6 polygons (touches midpoints) but contains one
        (shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG), [3]),
        ([shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG)], [[0], [3]]),
        # envelope of points overlaps polygons, but points do not intersect
        # (not valid relation)
        (MultiPoint([[5, 7], [7, 5]]), []),
        ([MultiPoint([[5, 7], [7, 5]])], [[], []]),
    ],
)
def test_query_contains_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry, predicate="contains"), expected)


### predicate == 'overlaps'
# Overlaps only returns results where geometries are of same dimensions
# and do not completely contain each other.
# See: https://postgis.net/docs/ST_Overlaps.html
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (Point(0.5, 0.5), []),
        ([Point(0.5, 0.5)], [[], []]),
        # points intersect but do not overlap
        (Point(1, 1), []),
        ([Point(1, 1)], [[], []]),
        # box overlaps points including those at edge but does not overlap
        # (completely contains all points)
        (box(3, 3, 6, 6), []),
        ([box(3, 3, 6, 6)], [[], []]),
        # envelope of buffer contains points, but does not overlap
        (shapely.buffer(Point(3, 3), 1), []),
        ([shapely.buffer(Point(3, 3), 1)], [[], []]),
        # multipoints intersect but do not overlap (both completely contain each other)
        (MultiPoint([[5, 5], [7, 7]]), []),
        ([MultiPoint([[5, 5], [7, 7]])], [[], []]),
        # envelope of points contains points in tree, but points do not intersect
        (MultiPoint([[5, 7], [7, 5]]), []),
        ([MultiPoint([[5, 7], [7, 5]])], [[], []]),
        # only one point of multipoint intersects but does not overlap
        # the intersecting point from multipoint completely contains point in tree
        (MultiPoint([[5, 7], [7, 7]]), []),
        ([MultiPoint([[5, 7], [7, 7]])], [[], []]),
    ],
)
def test_query_overlaps_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate="overlaps"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects line but is completely contained by it
        (Point(0, 0), []),
        ([Point(0, 0)], [[], []]),
        # box overlaps second line (contains first line)
        # but of different dimensions so does not overlap
        (box(0, 0, 1.5, 1.5), []),
        ([box(0, 0, 1.5, 1.5)], [[], []]),
        # buffer intersects 2 lines but of different dimensions so does not overlap
        (shapely.buffer(Point(3, 3), 0.5), []),
        ([shapely.buffer(Point(3, 3), 0.5)], [[], []]),
        # envelope of points overlaps lines but intersects none
        (MultiPoint([[5, 7], [7, 5]]), []),
        ([MultiPoint([[5, 7], [7, 5]])], [[], []]),
        # only one point of multipoint intersects
        (MultiPoint([[5, 7], [7, 7]]), []),
        ([MultiPoint([[5, 7], [7, 7]])], [[], []]),
        # both points intersect but different dimensions
        (MultiPoint([[5, 5], [6, 6]]), []),
        ([MultiPoint([[5, 5], [6, 6]])], [[], []]),
    ],
)
def test_query_overlaps_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry, predicate="overlaps"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point does not overlap any polygons (different dimensions)
        (Point(0, 0), []),
        ([Point(0, 0)], [[], []]),
        # box overlaps 2 polygons
        (box(0, 0, 1, 1), [0, 1]),
        ([box(0, 0, 1, 1)], [[0, 0], [0, 1]]),
        # larger box intersects 3 polygons and contains one
        (box(0, 0, 2, 2), [0, 2]),
        ([box(0, 0, 2, 2)], [[0, 0], [0, 2]]),
        # buffer overlaps 3 polygons and contains 1
        (shapely.buffer(Point(3, 3), HALF_UNIT_DIAG), [2, 4]),
        ([shapely.buffer(Point(3, 3), HALF_UNIT_DIAG)], [[0, 0], [2, 4]]),
        # larger buffer overlaps 6 polygons (touches midpoints) but contains one
        (shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG), [1, 2, 4, 5]),
        (
            [shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG)],
            [[0, 0, 0, 0], [1, 2, 4, 5]],
        ),
        # one of two points intersects but different dimensions
        (MultiPoint([[5, 7], [7, 7]]), []),
        ([MultiPoint([[5, 7], [7, 7]])], [[], []]),
    ],
)
def test_query_overlaps_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry, predicate="overlaps"), expected)


### predicate == 'crosses'
# Only valid for certain geometry combinations
# See: https://postgis.net/docs/ST_Crosses.html
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points intersect but not valid relation
        (Point(1, 1), []),
        # all points of result from tree are in common with box
        (box(3, 3, 6, 6), []),
        # all points of result from tree are in common with buffer
        (shapely.buffer(Point(3, 3), 1), []),
        # only one point of multipoint intersects but not valid relation
        (MultiPoint([[5, 7], [7, 7]]), []),
    ],
)
def test_query_crosses_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate="crosses"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects first line but is completely in common with line
        (Point(0, 0), []),
        ([Point(0, 0)], [[], []]),
        # box overlaps envelope of first 2 lines, contains first and crosses second
        (box(0, 0, 1.5, 1.5), [1]),
        ([box(0, 0, 1.5, 1.5)], [[0], [1]]),
        # buffer intersects 2 lines
        (shapely.buffer(Point(3, 3), 0.5), [2, 3]),
        ([shapely.buffer(Point(3, 3), 0.5)], [[0, 0], [2, 3]]),
        # line crosses line
        (shapely.linestrings([(1, 0), (0, 1)]), [0]),
        ([shapely.linestrings([(1, 0), (0, 1)])], [[0], [0]]),
        # envelope of points overlaps lines but intersects none
        (MultiPoint([[5, 7], [7, 5]]), []),
        ([MultiPoint([[5, 7], [7, 5]])], [[], []]),
        # only one point of multipoint intersects
        (MultiPoint([[5, 7], [7, 7], [7, 8]]), []),
        ([MultiPoint([[5, 7], [7, 7], [7, 8]])], [[], []]),
    ],
)
def test_query_crosses_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry, predicate="crosses"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point within first polygon but not valid relation
        (Point(0, 0.5), []),
        ([Point(0, 0.5)], [[], []]),
        # box overlaps 2 polygons but not valid relation
        (box(0, 0, 1.5, 1.5), []),
        ([box(0, 0, 1.5, 1.5)], [[], []]),
        # buffer overlaps 3 polygons but not valid relation
        (shapely.buffer(Point(3, 3), HALF_UNIT_DIAG), []),
        ([shapely.buffer(Point(3, 3), HALF_UNIT_DIAG)], [[], []]),
        # only one point of multipoint within
        (MultiPoint([[5, 7], [7, 7], [7, 8]]), [7]),
        ([MultiPoint([[5, 7], [7, 7], [7, 8]])], [[0], [7]]),
    ],
)
def test_query_crosses_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry, predicate="crosses"), expected)


### predicate == 'touches'
# See: https://postgis.net/docs/ST_Touches.html
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (Point(0.5, 0.5), []),
        ([Point(0.5, 0.5)], [[], []]),
        # points intersect but not valid relation
        (Point(1, 1), []),
        ([Point(1, 1)], [[], []]),
        # box contains points but touches only those at edges
        (box(3, 3, 6, 6), [3, 6]),
        ([box(3, 3, 6, 6)], [[0, 0], [3, 6]]),
        # polygon completely contains point in tree
        (shapely.buffer(Point(3, 3), 1), []),
        ([shapely.buffer(Point(3, 3), 1)], [[], []]),
        # linestring intersects 2 points but touches only one
        (LineString([(-1, -1), (1, 1)]), [1]),
        ([LineString([(-1, -1), (1, 1)])], [[0], [1]]),
        # multipoints intersect but not valid relation
        (MultiPoint([[5, 5], [7, 7]]), []),
        ([MultiPoint([[5, 5], [7, 7]])], [[], []]),
    ],
)
def test_query_touches_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate="touches"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects first line
        (Point(0, 0), [0]),
        ([Point(0, 0)], [[0], [0]]),
        # point is within line
        (Point(0.5, 0.5), []),
        ([Point(0.5, 0.5)], [[], []]),
        # point at shared vertex between 2 lines
        (Point(1, 1), [0, 1]),
        ([Point(1, 1)], [[0, 0], [0, 1]]),
        # box overlaps envelope of first 2 lines (touches edge of 1)
        (box(0, 0, 1, 1), [1]),
        ([box(0, 0, 1, 1)], [[0], [1]]),
        # buffer intersects 2 lines but does not touch edges of either
        (shapely.buffer(Point(3, 3), 0.5), []),
        ([shapely.buffer(Point(3, 3), 0.5)], [[], []]),
        # buffer intersects midpoint of line at tangent but there is a little overlap
        # due to precision issues
        (shapely.buffer(Point(2, 1), HALF_UNIT_DIAG + 1e-7), []),
        ([shapely.buffer(Point(2, 1), HALF_UNIT_DIAG + 1e-7)], [[], []]),
        # envelope of points overlaps lines but intersects none
        (MultiPoint([[5, 7], [7, 5]]), []),
        ([MultiPoint([[5, 7], [7, 5]])], [[], []]),
        # only one point of multipoint intersects at vertex between lines
        (MultiPoint([[5, 7], [7, 7], [7, 8]]), [6, 7]),
        ([MultiPoint([[5, 7], [7, 7], [7, 8]])], [[0, 0], [6, 7]]),
    ],
)
def test_query_touches_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry, predicate="touches"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point within first polygon
        (Point(0, 0.5), []),
        ([Point(0, 0.5)], [[], []]),
        # point is at edge of first polygon
        (Point(HALF_UNIT_DIAG + EPS, 0), [0]),
        ([Point(HALF_UNIT_DIAG + EPS, 0)], [[0], [0]]),
        # box overlaps envelope of 2 polygons does not touch any at edge
        (box(0, 0, 1, 1), []),
        ([box(0, 0, 1, 1)], [[], []]),
        # box overlaps 2 polygons and touches edge of first
        (box(HALF_UNIT_DIAG + EPS, 0, 2, 2), [0]),
        ([box(HALF_UNIT_DIAG + EPS, 0, 2, 2)], [[0], [0]]),
        # buffer overlaps 3 polygons but does not touch any at edge
        (shapely.buffer(Point(3, 3), HALF_UNIT_DIAG + EPS), []),
        ([shapely.buffer(Point(3, 3), HALF_UNIT_DIAG + EPS)], [[], []]),
        # only one point of multipoint within polygon but does not touch
        (MultiPoint([[0, 0], [7, 7], [7, 8]]), []),
        ([MultiPoint([[0, 0], [7, 7], [7, 8]])], [[], []]),
    ],
)
def test_query_touches_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry, predicate="touches"), expected)


### predicate == 'covers'
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (Point(0.5, 0.5), []),
        ([Point(0.5, 0.5)], [[], []]),
        # points intersect and thus no point is outside the other
        (Point(1, 1), [1]),
        ([Point(1, 1)], [[0], [1]]),
        # box covers any points that intersect or are within
        (box(3, 3, 6, 6), [3, 4, 5, 6]),
        ([box(3, 3, 6, 6)], [[0, 0, 0, 0], [3, 4, 5, 6]]),
        # envelope of buffer covers more points than are covered by buffer
        # due to diagonal distance
        (shapely.buffer(Point(3, 3), 1), [3]),
        ([shapely.buffer(Point(3, 3), 1)], [[0], [3]]),
        # envelope of buffer with 1/2 distance between points should intersect
        # same points as envelope
        (shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG), [2, 3, 4]),
        ([shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG)], [[0, 0, 0], [2, 3, 4]]),
        # multipoints intersect and thus no point is outside the other
        (MultiPoint([[5, 5], [7, 7]]), [5, 7]),
        ([MultiPoint([[5, 5], [7, 7]])], [[0, 0], [5, 7]]),
        # envelope of points contains points, but points do not intersect
        (MultiPoint([[5, 7], [7, 5]]), []),
        ([MultiPoint([[5, 7], [7, 5]])], [[], []]),
        # only one point of multipoint intersects
        (MultiPoint([[5, 7], [7, 7]]), [7]),
        ([MultiPoint([[5, 7], [7, 7]])], [[0], [7]]),
    ],
)
def test_query_covers_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate="covers"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point does not cover any lines (not valid relation)
        (Point(0, 0), []),
        ([Point(0, 0)], [[], []]),
        # box covers first line (intersects another does not contain it)
        (box(0, 0, 1.5, 1.5), [0]),
        ([box(0, 0, 1.5, 1.5)], [[0], [0]]),
        # box completely covers 2 lines (touches edges of 2 others)
        (box(1, 1, 3, 3), [1, 2]),
        ([box(1, 1, 3, 3)], [[0, 0], [1, 2]]),
        # buffer intersects 2 lines but does not completely cover either
        (shapely.buffer(Point(3, 3), 0.5), []),
        ([shapely.buffer(Point(3, 3), 0.5)], [[], []]),
        # envelope of points overlaps lines but intersects none
        (MultiPoint([[5, 7], [7, 5]]), []),
        ([MultiPoint([[5, 7], [7, 5]])], [[], []]),
        # only one point of multipoint intersects a line, but does not completely cover
        # it
        (MultiPoint([[5, 7], [7, 7]]), []),
        ([MultiPoint([[5, 7], [7, 7]])], [[], []]),
        # both points intersect but do not cover any lines (not valid relation)
        (MultiPoint([[5, 5], [6, 6]]), []),
        ([MultiPoint([[5, 5], [6, 6]])], [[], []]),
    ],
)
def test_query_covers_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry, predicate="covers"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point does not cover any polygons (not valid relation)
        (Point(0, 0), []),
        ([Point(0, 0)], [[], []]),
        # box overlaps envelope of 2 polygons but does not completely cover either
        (box(0, 0, 1, 1), []),
        ([box(0, 0, 1, 1)], [[], []]),
        # larger box intersects 3 polygons but covers only one
        (box(0, 0, 2, 2), [1]),
        ([box(0, 0, 2, 2)], [[0], [1]]),
        # buffer overlaps 3 polygons but does not completely cover any
        (shapely.buffer(Point(3, 3), HALF_UNIT_DIAG), []),
        ([shapely.buffer(Point(3, 3), HALF_UNIT_DIAG)], [[], []]),
        # larger buffer overlaps 6 polygons (touches midpoints) but covers only one
        (shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG), [3]),
        ([shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG)], [[0], [3]]),
        # envelope of points overlaps polygons, but points do not intersect
        # (not valid relation)
        (MultiPoint([[5, 7], [7, 5]]), []),
        ([MultiPoint([[5, 7], [7, 5]])], [[], []]),
    ],
)
def test_query_covers_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry, predicate="covers"), expected)


### predicate == 'covered_by'
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (Point(0.5, 0.5), []),
        ([Point(0.5, 0.5)], [[], []]),
        # points intersect
        (Point(1, 1), [1]),
        ([Point(1, 1)], [[0], [1]]),
        # box not covered by points
        (box(3, 3, 6, 6), []),
        ([box(3, 3, 6, 6)], [[], []]),
        # envelope of buffer not covered by points
        (shapely.buffer(Point(3, 3), 1), []),
        ([shapely.buffer(Point(3, 3), 1)], [[], []]),
        # multipoints intersect but are not covered by points in tree
        (MultiPoint([[5, 5], [7, 7]]), []),
        ([MultiPoint([[5, 5], [7, 7]])], [[], []]),
        # only one point of multipoint intersects, but multipoints are not
        # covered by any points in tree
        (MultiPoint([[5, 7], [7, 7]]), []),
        ([MultiPoint([[5, 7], [7, 7]])], [[], []]),
        # envelope of points overlaps points, but points do not intersect
        (MultiPoint([[5, 7], [7, 5]]), []),
        ([MultiPoint([[5, 7], [7, 5]])], [[], []]),
    ],
)
def test_query_covered_by_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate="covered_by"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # endpoint is covered by first line
        (Point(0, 0), [0]),
        ([Point(0, 0)], [[0], [0]]),
        # point covered by first line
        (Point(0.5, 0.5), [0]),
        ([Point(0.5, 0.5)], [[0], [0]]),
        # point within envelope of first line but does not intersect
        (Point(0, 0.5), []),
        ([Point(0, 0.5)], [[], []]),
        # point at shared vertex between 2 lines and is covered by both
        (Point(1, 1), [0, 1]),
        ([Point(1, 1)], [[0, 0], [0, 1]]),
        # line intersects 3 lines, but is covered by only one
        (shapely.linestrings([[1, 1], [2, 2]]), [1]),
        ([shapely.linestrings([[1, 1], [2, 2]])], [[0], [1]]),
        # line intersects 2 lines, but is covered by neither
        (shapely.linestrings([[1.5, 1.5], [2.5, 2.5]]), []),
        ([shapely.linestrings([[1.5, 1.5], [2.5, 2.5]])], [[], []]),
        # box not covered by line (not valid geometric relation)
        (box(0, 0, 1, 1), []),
        ([box(0, 0, 1, 1)], [[], []]),
        # buffer intersects 2 lines but not within either (not valid geometric relation)
        (shapely.buffer(Point(3, 3), 0.5), []),
        ([shapely.buffer(Point(3, 3), 0.5)], [[], []]),
        # envelope of points overlaps lines but intersects none
        (MultiPoint([[5, 7], [7, 5]]), []),
        ([MultiPoint([[5, 7], [7, 5]])], [[], []]),
        # only one point of multipoint intersects, but both are not covered by line
        (MultiPoint([[5, 7], [7, 7]]), []),
        ([MultiPoint([[5, 7], [7, 7]])], [[], []]),
        # both points are covered by a line
        (MultiPoint([[6.5, 6.5], [7, 7]]), [6]),
        ([MultiPoint([[6.5, 6.5], [7, 7]])], [[0], [6]]),
    ],
)
def test_query_covered_by_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry, predicate="covered_by"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point covered by polygon
        (Point(0, 0.5), [0]),
        ([Point(0, 0.5)], [[0], [0]]),
        (Point(0.5, 0), [0]),
        ([Point(0.5, 0)], [[0], [0]]),
        (Point(1, 1), [1]),
        ([Point(1, 1)], [[0], [1]]),
        # midpoint between two polygons is covered by both
        (Point(0.5, 0.5), [0, 1]),
        ([Point(0.5, 0.5)], [[0, 0], [0, 1]]),
        # line intersects multiple polygons but is not covered by any
        (shapely.linestrings([[0, 0], [2, 2]]), []),
        ([shapely.linestrings([[0, 0], [2, 2]])], [[], []]),
        # line intersects multiple polygons but is covered by only one
        (shapely.linestrings([[1.5, 1.5], [2.5, 2.5]]), [2]),
        ([shapely.linestrings([[1.5, 1.5], [2.5, 2.5]])], [[0], [2]]),
        # box overlaps envelope of 2 polygons but not covered by either
        (box(0, 0, 1, 1), []),
        ([box(0, 0, 1, 1)], [[], []]),
        # box covered by polygon
        (box(0, 0, 0.5, 0.5), [0]),
        ([box(0, 0, 0.5, 0.5)], [[0], [0]]),
        # larger box intersects 3 polygons but not covered by any
        (box(0, 0, 1.5, 1.5), []),
        ([box(0, 0, 1.5, 1.5)], [[], []]),
        # buffer intersects 3 polygons but only within one
        (shapely.buffer(Point(3, 3), HALF_UNIT_DIAG), [3]),
        ([shapely.buffer(Point(3, 3), HALF_UNIT_DIAG)], [[0], [3]]),
        # larger buffer overlaps 6 polygons (touches midpoints) but within none
        (shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG), []),
        ([shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG)], [[], []]),
        # envelope of points overlaps polygons, but points do not intersect
        (MultiPoint([[5, 7], [7, 5]]), []),
        ([MultiPoint([[5, 7], [7, 5]])], [[], []]),
        # only one point of multipoint within polygon
        (MultiPoint([[5, 7], [7, 7]]), []),
        ([MultiPoint([[5, 7], [7, 7]])], [[], []]),
        # both points in multipoint within polygon
        (MultiPoint([[5.25, 5.5], [5.25, 5.0]]), [5]),
        ([MultiPoint([[5.25, 5.5], [5.25, 5.0]])], [[0], [5]]),
    ],
)
def test_query_covered_by_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry, predicate="covered_by"), expected)


### predicate == 'contains_properly'
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (Point(0.5, 0.5), []),
        ([Point(0.5, 0.5)], [[], []]),
        # points intersect
        (Point(1, 1), [1]),
        ([Point(1, 1)], [[0], [1]]),
        # line contains every point that is not on its first or last coordinate
        # these are on the "exterior" of the line
        (shapely.linestrings([[0, 0], [2, 2]]), [1]),
        ([shapely.linestrings([[0, 0], [2, 2]])], [[0], [1]]),
        # slightly longer line contains multiple points
        (shapely.linestrings([[0.5, 0.5], [2.5, 2.5]]), [1, 2]),
        ([shapely.linestrings([[0.5, 0.5], [2.5, 2.5]])], [[0, 0], [1, 2]]),
        # line intersects and contains one point
        (shapely.linestrings([[0, 2], [2, 0]]), [1]),
        ([shapely.linestrings([[0, 2], [2, 0]])], [[0], [1]]),
        # box contains points (2 are at edges and not contained)
        (box(3, 3, 6, 6), [4, 5]),
        ([box(3, 3, 6, 6)], [[0, 0], [4, 5]]),
        # envelope of buffer contains more points than within buffer
        # due to diagonal distance
        (shapely.buffer(Point(3, 3), 1), [3]),
        ([shapely.buffer(Point(3, 3), 1)], [[0], [3]]),
        # envelope of buffer with 1/2 distance between points should intersect
        # same points as envelope
        (shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG), [2, 3, 4]),
        ([shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG)], [[0, 0, 0], [2, 3, 4]]),
        # multipoints intersect
        (MultiPoint([[5, 5], [7, 7]]), [5, 7]),
        ([MultiPoint([[5, 5], [7, 7]])], [[0, 0], [5, 7]]),
        # envelope of points contains points, but points do not intersect
        (MultiPoint([[5, 7], [7, 5]]), []),
        ([MultiPoint([[5, 7], [7, 5]])], [[], []]),
        # only one point of multipoint intersects
        (MultiPoint([[5, 7], [7, 7]]), [7]),
        ([MultiPoint([[5, 7], [7, 7]])], [[0], [7]]),
    ],
)
def test_query_contains_properly_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate="contains_properly"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # None of the following conditions satisfy the relation for linestrings
        # because they have no interior:
        # "a contains b if no points of b lie in the exterior of a, and at least one
        # point of the interior of b lies in the interior of a"
        (Point(0, 0), []),
        ([Point(0, 0)], [[], []]),
        (shapely.linestrings([[0, 0], [1, 1]]), []),
        ([shapely.linestrings([[0, 0], [1, 1]])], [[], []]),
        (shapely.linestrings([[0, 0], [2, 2]]), []),
        ([shapely.linestrings([[0, 0], [2, 2]])], [[], []]),
        (shapely.linestrings([[0, 2], [2, 0]]), []),
        ([shapely.linestrings([[0, 2], [2, 0]])], [[], []]),
        (MultiPoint([[5, 7], [7, 5]]), []),
        ([MultiPoint([[5, 7], [7, 5]])], [[], []]),
        (MultiPoint([[5, 7], [7, 7]]), []),
        ([MultiPoint([[5, 7], [7, 7]])], [[], []]),
        (MultiPoint([[5, 5], [6, 6]]), []),
        ([MultiPoint([[5, 5], [6, 6]])], [[], []]),
        (box(0, 0, 1, 1), []),
        ([box(0, 0, 1, 1)], [[], []]),
        (box(0, 0, 2, 2), []),
        ([box(0, 0, 2, 2)], [[], []]),
        (shapely.buffer(Point(3, 3), 0.5), []),
        ([shapely.buffer(Point(3, 3), 0.5)], [[], []]),
    ],
)
def test_query_contains_properly_lines(line_tree, geometry, expected):
    assert_array_equal(
        line_tree.query(geometry, predicate="contains_properly"), expected
    )


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point does not contain any polygons (not valid relation)
        (Point(0, 0), []),
        ([Point(0, 0)], [[], []]),
        # line intersects multiple polygons but does not contain any (not valid
        # relation)
        (shapely.linestrings([[0, 0], [2, 2]]), []),
        ([shapely.linestrings([[0, 0], [2, 2]])], [[], []]),
        # box overlaps envelope of 2 polygons but contains neither
        (box(0, 0, 1, 1), []),
        ([box(0, 0, 1, 1)], [[], []]),
        # larger box intersects 3 polygons but contains only one
        (box(0, 0, 2, 2), [1]),
        ([box(0, 0, 2, 2)], [[0], [1]]),
        # buffer overlaps 3 polygons but contains none
        (shapely.buffer(Point(3, 3), HALF_UNIT_DIAG), []),
        ([shapely.buffer(Point(3, 3), HALF_UNIT_DIAG)], [[], []]),
        # larger buffer overlaps 6 polygons (touches midpoints) but contains one
        (shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG), [3]),
        ([shapely.buffer(Point(3, 3), 3 * HALF_UNIT_DIAG)], [[0], [3]]),
        # envelope of points overlaps polygons, but points do not intersect
        # (not valid relation)
        (MultiPoint([[5, 7], [7, 5]]), []),
        ([MultiPoint([[5, 7], [7, 5]])], [[], []]),
    ],
)
def test_query_contains_properly_polygons(poly_tree, geometry, expected):
    assert_array_equal(
        poly_tree.query(geometry, predicate="contains_properly"), expected
    )


### predicate = 'dwithin'


@pytest.mark.skipif(geos_version >= (3, 10, 0), reason="GEOS >= 3.10")
@pytest.mark.parametrize(
    "geometry", [Point(0, 0), [Point(0, 0)], None, [None], empty, [empty]]
)
def test_query_dwithin_geos_version(tree, geometry):
    with pytest.raises(UnsupportedGEOSVersionError, match="requires GEOS >= 3.10"):
        tree.query(geometry, predicate="dwithin", distance=1)


@pytest.mark.skipif(geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize(
    "geometry,distance,match",
    [
        (Point(0, 0), None, "distance parameter must be provided"),
        ([Point(0, 0)], None, "distance parameter must be provided"),
        (Point(0, 0), "foo", "could not convert string to float"),
        ([Point(0, 0)], "foo", "could not convert string to float"),
        ([Point(0, 0)], ["foo"], "could not convert string to float"),
        (Point(0, 0), [0, 1], "Could not broadcast distance to match geometry"),
        ([Point(0, 0)], [0, 1], "Could not broadcast distance to match geometry"),
        (Point(0, 0), [[1.0]], "should be one dimensional"),
        ([Point(0, 0)], [[1.0]], "should be one dimensional"),
    ],
)
def test_query_dwithin_invalid_distance(tree, geometry, distance, match):
    with pytest.raises(ValueError, match=match):
        tree.query(geometry, predicate="dwithin", distance=distance)


@pytest.mark.skipif(geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize(
    "geometry,distance,expected",
    [
        (None, 1.0, []),
        ([None], 1.0, [[], []]),
        (Point(0.25, 0.25), 0, []),
        ([Point(0.25, 0.25)], 0, [[], []]),
        (Point(0.25, 0.25), -1, []),
        ([Point(0.25, 0.25)], -1, [[], []]),
        (Point(0.25, 0.25), np.nan, []),
        ([Point(0.25, 0.25)], np.nan, [[], []]),
        (Point(), 1, []),
        ([Point()], 1, [[], []]),
        (Point(0.25, 0.25), 0.5, [0]),
        ([Point(0.25, 0.25)], 0.5, [[0], [0]]),
        (Point(0.25, 0.25), 2.5, [0, 1, 2]),
        ([Point(0.25, 0.25)], 2.5, [[0, 0, 0], [0, 1, 2]]),
        (Point(3, 3), 1.5, [2, 3, 4]),
        ([Point(3, 3)], 1.5, [[0, 0, 0], [2, 3, 4]]),
        # 2 equidistant points in tree
        (Point(0.5, 0.5), 0.75, [0, 1]),
        ([Point(0.5, 0.5)], 0.75, [[0, 0], [0, 1]]),
        (
            [None, Point(0.5, 0.5)],
            0.75,
            [
                [
                    1,
                    1,
                ],
                [0, 1],
            ],
        ),
        (
            [Point(0.5, 0.5), Point(0.25, 0.25)],
            0.75,
            [[0, 0, 1], [0, 1, 0]],
        ),
        (
            [Point(0, 0.2), Point(1.75, 1.75)],
            [0.25, 2],
            [[0, 1, 1, 1], [0, 1, 2, 3]],
        ),
        # all points intersect box
        (box(0, 0, 3, 3), 0, [0, 1, 2, 3]),
        ([box(0, 0, 3, 3)], 0, [[0, 0, 0, 0], [0, 1, 2, 3]]),
        (box(0, 0, 3, 3), 0.25, [0, 1, 2, 3]),
        ([box(0, 0, 3, 3)], 0.25, [[0, 0, 0, 0], [0, 1, 2, 3]]),
        # intersecting and nearby points
        (box(1, 1, 2, 2), 1.5, [0, 1, 2, 3]),
        ([box(1, 1, 2, 2)], 1.5, [[0, 0, 0, 0], [0, 1, 2, 3]]),
        # # return nearest point in tree for each point in multipoint
        (MultiPoint([[0.25, 0.25], [1.5, 1.5]]), 0.75, [0, 1, 2]),
        ([MultiPoint([[0.25, 0.25], [1.5, 1.5]])], 0.75, [[0, 0, 0], [0, 1, 2]]),
        # 2 equidistant points per point in multipoint
        (
            MultiPoint([[0.5, 0.5], [3.5, 3.5]]),
            0.75,
            [0, 1, 3, 4],
        ),
        (
            [MultiPoint([[0.5, 0.5], [3.5, 3.5]])],
            0.75,
            [[0, 0, 0, 0], [0, 1, 3, 4]],
        ),
    ],
)
def test_query_dwithin_points(tree, geometry, distance, expected):
    assert_array_equal(
        tree.query(geometry, predicate="dwithin", distance=distance), expected
    )


@pytest.mark.skipif(geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize(
    "geometry,distance,expected",
    [
        (None, 1.0, []),
        ([None], 1.0, [[], []]),
        (Point(0.5, 0.5), 0, [0]),
        ([Point(0.5, 0.5)], 0, [[0], [0]]),
        (Point(0.5, 0.5), 1.0, [0, 1]),
        ([Point(0.5, 0.5)], 1.0, [[0, 0], [0, 1]]),
        (Point(2, 2), 0.5, [1, 2]),
        ([Point(2, 2)], 0.5, [[0, 0], [1, 2]]),
        (box(0, 0, 1, 1), 0.5, [0, 1]),
        ([box(0, 0, 1, 1)], 0.5, [[0, 0], [0, 1]]),
        (box(0.5, 0.5, 1.5, 1.5), 0.5, [0, 1]),
        ([box(0.5, 0.5, 1.5, 1.5)], 0.5, [[0, 0], [0, 1]]),
        # multipoints at endpoints of 2 lines each
        (MultiPoint([[5, 5], [7, 7]]), 0.5, [4, 5, 6, 7]),
        ([MultiPoint([[5, 5], [7, 7]])], 0.5, [[0, 0, 0, 0], [4, 5, 6, 7]]),
        # multipoints are equidistant from 2 lines
        (MultiPoint([[5, 7], [7, 5]]), 1.5, [5, 6]),
        ([MultiPoint([[5, 7], [7, 5]])], 1.5, [[0, 0], [5, 6]]),
    ],
)
def test_query_dwithin_lines(line_tree, geometry, distance, expected):
    assert_array_equal(
        line_tree.query(geometry, predicate="dwithin", distance=distance),
        expected,
    )


@pytest.mark.skipif(geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize(
    "geometry,distance,expected",
    [
        (Point(0, 0), 0, [0]),
        ([Point(0, 0)], 0, [[0], [0]]),
        (Point(0, 0), 0.5, [0]),
        ([Point(0, 0)], 0.5, [[0], [0]]),
        (Point(0, 0), 1.5, [0, 1]),
        ([Point(0, 0)], 1.5, [[0, 0], [0, 1]]),
        (Point(0.5, 0.5), 1, [0, 1]),
        ([Point(0.5, 0.5)], 1, [[0, 0], [0, 1]]),
        (Point(0.5, 0.5), 0.5, [0, 1]),
        ([Point(0.5, 0.5)], 0.5, [[0, 0], [0, 1]]),
        (box(0, 0, 1, 1), 0, [0, 1]),
        ([box(0, 0, 1, 1)], 0, [[0, 0], [0, 1]]),
        (box(0, 0, 1, 1), 2, [0, 1, 2]),
        ([box(0, 0, 1, 1)], 2, [[0, 0, 0], [0, 1, 2]]),
        (MultiPoint([[5, 5], [7, 7]]), 0.5, [5, 7]),
        ([MultiPoint([[5, 5], [7, 7]])], 0.5, [[0, 0], [5, 7]]),
        (
            MultiPoint([[5, 5], [7, 7]]),
            2.5,
            [3, 4, 5, 6, 7, 8, 9],
        ),
        (
            [MultiPoint([[5, 5], [7, 7]])],
            2.5,
            [[0, 0, 0, 0, 0, 0, 0], [3, 4, 5, 6, 7, 8, 9]],
        ),
    ],
)
def test_query_dwithin_polygons(poly_tree, geometry, distance, expected):
    assert_array_equal(
        poly_tree.query(geometry, predicate="dwithin", distance=distance),
        expected,
    )


### STRtree nearest


def test_nearest_empty_tree():
    tree = STRtree([])
    assert tree.nearest(point) is None


@pytest.mark.parametrize("geometry", ["I am not a geometry"])
def test_nearest_invalid_geom(tree, geometry):
    with pytest.raises(TypeError):
        tree.nearest(geometry)


@pytest.mark.parametrize("geometry", [None, [None], [Point(1, 1), None]])
def test_nearest_none(tree, geometry):
    with pytest.raises(ValueError):
        tree.nearest(geometry)


@pytest.mark.parametrize(
    "geometry", [empty_point, [empty_point], [Point(1, 1), empty_point]]
)
def test_nearest_empty(tree, geometry):
    with pytest.raises(ValueError):
        tree.nearest(geometry)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        (Point(0.25, 0.25), 0),
        (Point(0.75, 0.75), 1),
        (Point(1, 1), 1),
        ([Point(1, 1), Point(0, 0)], [1, 0]),
        ([Point(1, 1), Point(0.25, 1)], [1, 1]),
        ([Point(-10, -10), Point(100, 100)], [0, 9]),
        (box(0.5, 0.5, 0.75, 0.75), 1),
        (shapely.buffer(Point(2.5, 2.5), HALF_UNIT_DIAG), 2),
        (shapely.buffer(Point(3, 3), HALF_UNIT_DIAG), 3),
        (MultiPoint([[5.5, 5], [7, 7]]), 7),
        (MultiPoint([[5, 7], [7, 5]]), 6),
    ],
)
def test_nearest_points(tree, geometry, expected):
    assert_array_equal(tree.nearest(geometry), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # 2 equidistant points in tree
        (Point(0.5, 0.5), [0, 1]),
        # multiple points in box
        (box(0, 0, 3, 3), [0, 1, 2, 3]),
        # return nearest point in tree for each point in multipoint
        (MultiPoint([[5, 5], [7, 7]]), [5, 7]),
    ],
)
def test_nearest_points_equidistant(tree, geometry, expected):
    # results are returned in order they are traversed when searching the tree,
    # which can vary between GEOS versions, so we test that one of the valid
    # results is present
    result = tree.nearest(geometry)
    assert result in expected


@pytest.mark.parametrize(
    "geometry,expected",
    [
        (Point(0.5, 0.5), 0),
        (Point(1.5, 0.5), 0),
        (shapely.box(0.5, 1.5, 1, 2), 1),
        (shapely.linestrings([[0, 0.5], [1, 2.5]]), 0),
    ],
)
def test_nearest_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.nearest(geometry), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # at junction between 2 lines
        (Point(2, 2), [1, 2]),
        # contains one line, intersects with another
        (box(0, 0, 1, 1), [0, 1]),
        # overlaps 2 lines
        (box(0.5, 0.5, 1.5, 1.5), [0, 1]),
        # box overlaps 2 lines and intersects endpoints of 2 more
        (box(3, 3, 5, 5), [2, 3, 4, 5]),
        (shapely.buffer(Point(2.5, 2.5), HALF_UNIT_DIAG), [1, 2]),
        (shapely.buffer(Point(3, 3), HALF_UNIT_DIAG), [2, 3]),
        # multipoints at endpoints of 2 lines each
        (MultiPoint([[5, 5], [7, 7]]), [4, 5, 6, 7]),
        # second point in multipoint at endpoints of 2 lines
        (MultiPoint([[5.5, 5], [7, 7]]), [6, 7]),
        # multipoints are equidistant from 2 lines
        (MultiPoint([[5, 7], [7, 5]]), [5, 6]),
    ],
)
def test_nearest_lines_equidistant(line_tree, geometry, expected):
    # results are returned in order they are traversed when searching the tree,
    # which can vary between GEOS versions, so we test that one of the valid
    # results is present
    result = line_tree.nearest(geometry)
    assert result in expected


@pytest.mark.parametrize(
    "geometry,expected",
    [
        (Point(0, 0), 0),
        (Point(2, 2), 2),
        (shapely.box(0, 5, 1, 6), 3),
        (MultiPoint([[5, 7], [7, 5]]), 6),
    ],
)
def test_nearest_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.nearest(geometry), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # 2 polygons in tree overlap point
        (Point(0.5, 0.5), [0, 1]),
        # box overlaps multiple polygons
        (box(0, 0, 1, 1), [0, 1]),
        (box(0.5, 0.5, 1.5, 1.5), [0, 1, 2]),
        (box(3, 3, 5, 5), [3, 4, 5]),
        (shapely.buffer(Point(2.5, 2.5), HALF_UNIT_DIAG), [2, 3]),
        # completely overlaps one polygon, touches 2 others
        (shapely.buffer(Point(3, 3), HALF_UNIT_DIAG), [2, 3, 4]),
        # each point in multi point intersects a polygon in tree
        (MultiPoint([[5, 5], [7, 7]]), [5, 7]),
        (MultiPoint([[5.5, 5], [7, 7]]), [5, 7]),
    ],
)
def test_nearest_polygons_equidistant(poly_tree, geometry, expected):
    # results are returned in order they are traversed when searching the tree,
    # which can vary between GEOS versions, so we test that one of the valid
    # results is present
    result = poly_tree.nearest(geometry)
    assert result in expected


def test_query_nearest_empty_tree():
    tree = STRtree([])
    assert_array_equal(tree.query_nearest(point), [])
    assert_array_equal(tree.query_nearest([point]), [[], []])


@pytest.mark.parametrize("geometry", ["I am not a geometry", ["still not a geometry"]])
def test_query_nearest_invalid_geom(tree, geometry):
    with pytest.raises(TypeError):
        tree.query_nearest(geometry)


@pytest.mark.parametrize(
    "geometry,return_distance,expected",
    [
        (None, False, []),
        ([None], False, [[], []]),
        (None, True, ([], [])),
        ([None], True, ([[], []], [])),
    ],
)
def test_query_nearest_none(tree, geometry, return_distance, expected):
    if return_distance:
        index, distance = tree.query_nearest(geometry, return_distance=True)
        assert_array_equal(index, expected[0])
        assert_array_equal(distance, expected[1])

    else:
        assert_array_equal(tree.query_nearest(geometry), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [(empty, []), ([empty], [[], []]), ([empty, point], [[1, 1], [2, 3]])],
)
def test_query_nearest_empty_geom(tree, geometry, expected):
    assert_array_equal(tree.query_nearest(geometry), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        (Point(0.25, 0.25), [0]),
        ([Point(0.25, 0.25)], [[0], [0]]),
        (Point(0.75, 0.75), [1]),
        ([Point(0.75, 0.75)], [[0], [1]]),
        (Point(1, 1), [1]),
        ([Point(1, 1)], [[0], [1]]),
        # 2 equidistant points in tree
        (Point(0.5, 0.5), [0, 1]),
        ([Point(0.5, 0.5)], [[0, 0], [0, 1]]),
        ([Point(1, 1), Point(0, 0)], [[0, 1], [1, 0]]),
        ([Point(1, 1), Point(0.25, 1)], [[0, 1], [1, 1]]),
        ([Point(-10, -10), Point(100, 100)], [[0, 1], [0, 9]]),
        (box(0.5, 0.5, 0.75, 0.75), [1]),
        ([box(0.5, 0.5, 0.75, 0.75)], [[0], [1]]),
        # multiple points in box
        (box(0, 0, 3, 3), [0, 1, 2, 3]),
        ([box(0, 0, 3, 3)], [[0, 0, 0, 0], [0, 1, 2, 3]]),
        (shapely.buffer(Point(2.5, 2.5), 1), [2, 3]),
        ([shapely.buffer(Point(2.5, 2.5), 1)], [[0, 0], [2, 3]]),
        (shapely.buffer(Point(3, 3), 0.5), [3]),
        ([shapely.buffer(Point(3, 3), 0.5)], [[0], [3]]),
        (MultiPoint([[5.5, 5], [7, 7]]), [7]),
        ([MultiPoint([[5.5, 5], [7, 7]])], [[0], [7]]),
        (MultiPoint([[5, 7], [7, 5]]), [6]),
        ([MultiPoint([[5, 7], [7, 5]])], [[0], [6]]),
        # return nearest point in tree for each point in multipoint
        (MultiPoint([[5, 5], [7, 7]]), [5, 7]),
        ([MultiPoint([[5, 5], [7, 7]])], [[0, 0], [5, 7]]),
        # 2 equidistant points per point in multipoint
        (MultiPoint([[0.5, 0.5], [3.5, 3.5]]), [0, 1, 3, 4]),
        ([MultiPoint([[0.5, 0.5], [3.5, 3.5]])], [[0, 0, 0, 0], [0, 1, 3, 4]]),
    ],
)
def test_query_nearest_points(tree, geometry, expected):
    assert_array_equal(tree.query_nearest(geometry), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        (Point(0.5, 0.5), [0]),
        ([Point(0.5, 0.5)], [[0], [0]]),
        # at junction between 2 lines, will return both
        (Point(2, 2), [1, 2]),
        ([Point(2, 2)], [[0, 0], [1, 2]]),
        # contains one line, intersects with another
        (box(0, 0, 1, 1), [0, 1]),
        ([box(0, 0, 1, 1)], [[0, 0], [0, 1]]),
        # overlaps 2 lines
        (box(0.5, 0.5, 1.5, 1.5), [0, 1]),
        ([box(0.5, 0.5, 1.5, 1.5)], [[0, 0], [0, 1]]),
        # second box overlaps 2 lines and intersects endpoints of 2 more
        ([box(0, 0, 0.5, 0.5), box(3, 3, 5, 5)], [[0, 1, 1, 1, 1], [0, 2, 3, 4, 5]]),
        (shapely.buffer(Point(2.5, 2.5), 1), [1, 2, 3]),
        ([shapely.buffer(Point(2.5, 2.5), 1)], [[0, 0, 0], [1, 2, 3]]),
        (shapely.buffer(Point(3, 3), 0.5), [2, 3]),
        ([shapely.buffer(Point(3, 3), 0.5)], [[0, 0], [2, 3]]),
        # multipoints at endpoints of 2 lines each
        (MultiPoint([[5, 5], [7, 7]]), [4, 5, 6, 7]),
        ([MultiPoint([[5, 5], [7, 7]])], [[0, 0, 0, 0], [4, 5, 6, 7]]),
        # second point in multipoint at endpoints of 2 lines
        (MultiPoint([[5.5, 5], [7, 7]]), [6, 7]),
        ([MultiPoint([[5.5, 5], [7, 7]])], [[0, 0], [6, 7]]),
        # multipoints are equidistant from 2 lines
        (MultiPoint([[5, 7], [7, 5]]), [5, 6]),
        ([MultiPoint([[5, 7], [7, 5]])], [[0, 0], [5, 6]]),
    ],
)
def test_query_nearest_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query_nearest(geometry), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        (Point(0, 0), [0]),
        ([Point(0, 0)], [[0], [0]]),
        (Point(2, 2), [2]),
        ([Point(2, 2)], [[0], [2]]),
        # 2 polygons in tree overlap point
        (Point(0.5, 0.5), [0, 1]),
        ([Point(0.5, 0.5)], [[0, 0], [0, 1]]),
        # box overlaps multiple polygons
        (box(0, 0, 1, 1), [0, 1]),
        ([box(0, 0, 1, 1)], [[0, 0], [0, 1]]),
        (box(0.5, 0.5, 1.5, 1.5), [0, 1, 2]),
        ([box(0.5, 0.5, 1.5, 1.5)], [[0, 0, 0], [0, 1, 2]]),
        ([box(0, 0, 1, 1), box(3, 3, 5, 5)], [[0, 0, 1, 1, 1], [0, 1, 3, 4, 5]]),
        (shapely.buffer(Point(2.5, 2.5), HALF_UNIT_DIAG), [2, 3]),
        ([shapely.buffer(Point(2.5, 2.5), HALF_UNIT_DIAG)], [[0, 0], [2, 3]]),
        # completely overlaps one polygon, touches 2 others
        (shapely.buffer(Point(3, 3), HALF_UNIT_DIAG), [2, 3, 4]),
        ([shapely.buffer(Point(3, 3), HALF_UNIT_DIAG)], [[0, 0, 0], [2, 3, 4]]),
        # each point in multi point intersects a polygon in tree
        (MultiPoint([[5, 5], [7, 7]]), [5, 7]),
        ([MultiPoint([[5, 5], [7, 7]])], [[0, 0], [5, 7]]),
        (MultiPoint([[5.5, 5], [7, 7]]), [5, 7]),
        ([MultiPoint([[5.5, 5], [7, 7]])], [[0, 0], [5, 7]]),
        (MultiPoint([[5, 7], [7, 5]]), [6]),
        ([MultiPoint([[5, 7], [7, 5]])], [[0], [6]]),
    ],
)
def test_query_nearest_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query_nearest(geometry), expected)


@pytest.mark.parametrize(
    "geometry,max_distance,expected",
    [
        # using unset max_distance should return all nearest
        (Point(0.5, 0.5), None, [0, 1]),
        ([Point(0.5, 0.5)], None, [[0, 0], [0, 1]]),
        # using large max_distance should return all nearest
        (Point(0.5, 0.5), 10, [0, 1]),
        ([Point(0.5, 0.5)], 10, [[0, 0], [0, 1]]),
        # using small max_distance should return no results
        (Point(0.5, 0.5), 0.1, []),
        ([Point(0.5, 0.5)], 0.1, [[], []]),
        # using small max_distance should only return results in that distance
        ([Point(0.5, 0.5), Point(0, 0)], 0.1, [[1], [0]]),
    ],
)
def test_query_nearest_max_distance(tree, geometry, max_distance, expected):
    assert_array_equal(
        tree.query_nearest(geometry, max_distance=max_distance), expected
    )


@pytest.mark.parametrize(
    "geometry,max_distance",
    [
        (Point(0.5, 0.5), 0),
        ([Point(0.5, 0.5)], 0),
        (Point(0.5, 0.5), -1),
        ([Point(0.5, 0.5)], -1),
    ],
)
def test_query_nearest_invalid_max_distance(tree, geometry, max_distance):
    with pytest.raises(ValueError, match="max_distance must be greater than 0"):
        tree.query_nearest(geometry, max_distance=max_distance)


def test_query_nearest_nonscalar_max_distance(tree):
    with pytest.raises(ValueError, match="parameter only accepts scalar values"):
        tree.query_nearest(Point(0.5, 0.5), max_distance=[1])


@pytest.mark.parametrize(
    "geometry,expected",
    [
        (Point(0, 0), ([0], [0.0])),
        ([Point(0, 0)], ([[0], [0]], [0.0])),
        (Point(0.5, 0.5), ([0, 1], [0.7071, 0.7071])),
        ([Point(0.5, 0.5)], ([[0, 0], [0, 1]], [0.7071, 0.7071])),
        (box(0, 0, 1, 1), ([0, 1], [0.0, 0.0])),
        ([box(0, 0, 1, 1)], ([[0, 0], [0, 1]], [0.0, 0.0])),
    ],
)
def test_query_nearest_return_distance(tree, geometry, expected):
    expected_indices, expected_dist = expected

    actual_indices, actual_dist = tree.query_nearest(geometry, return_distance=True)

    assert_array_equal(actual_indices, expected_indices)
    assert_array_equal(np.round(actual_dist, 4), expected_dist)


@pytest.mark.parametrize(
    "geometry,exclusive,expected",
    [
        (Point(1, 1), False, [1]),
        ([Point(1, 1)], False, [[0], [1]]),
        (Point(1, 1), True, [0, 2]),
        ([Point(1, 1)], True, [[0, 0], [0, 2]]),
        ([Point(1, 1), Point(2, 2)], True, [[0, 0, 1, 1], [0, 2, 1, 3]]),
    ],
)
def test_query_nearest_exclusive(tree, geometry, exclusive, expected):
    assert_array_equal(tree.query_nearest(geometry, exclusive=exclusive), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        (Point(1, 1), []),
        ([Point(1, 1)], [[], []]),
    ],
)
def test_query_nearest_exclusive_no_results(tree, geometry, expected):
    tree = STRtree([Point(1, 1)])
    assert_array_equal(tree.query_nearest(geometry, exclusive=True), expected)


@pytest.mark.parametrize(
    "geometry,exclusive",
    [
        (Point(1, 1), "invalid"),
        # non-scalar exclusive parameter not allowed
        (Point(1, 1), ["also invalid"]),
        ([Point(1, 1)], []),
        ([Point(1, 1)], [False]),
    ],
)
def test_query_nearest_invalid_exclusive(tree, geometry, exclusive):
    with pytest.raises(ValueError):
        tree.query_nearest(geometry, exclusive=exclusive)


@pytest.mark.parametrize(
    "geometry,all_matches",
    [
        (Point(1, 1), "invalid"),
        # non-scalar all_matches parameter not allowed
        (Point(1, 1), ["also invalid"]),
        ([Point(1, 1)], []),
        ([Point(1, 1)], [False]),
    ],
)
def test_query_nearest_invalid_all_matches(tree, geometry, all_matches):
    with pytest.raises(ValueError):
        tree.query_nearest(geometry, all_matches=all_matches)


def test_query_nearest_all_matches(tree):
    point = Point(0.5, 0.5)
    assert_array_equal(tree.query_nearest(point, all_matches=True), [0, 1])

    indices = tree.query_nearest(point, all_matches=False)
    # result is dependent on tree traversal order; may vary across test runs
    assert np.array_equal(indices, [0]) or np.array_equal(indices, [1])


def test_strtree_threaded_query():
    ## Create data
    polygons = shapely.polygons(np.random.randn(1000, 3, 2))
    # needs to be big enough to trigger the segfault
    N = 100_000
    points = shapely.points(4 * np.random.random(N) - 2, 4 * np.random.random(N) - 2)

    ## Slice parts of the arrays -> 4x4 => 16 combinations
    n = int(len(polygons) / 4)
    polygons_parts = [
        polygons[:n],
        polygons[n : 2 * n],
        polygons[2 * n : 3 * n],
        polygons[3 * n :],
    ]
    n = int(len(points) / 4)
    points_parts = [
        points[:n],
        points[n : 2 * n],
        points[2 * n : 3 * n],
        points[3 * n :],
    ]

    ## Creating the trees in advance
    trees = []
    for i in range(4):
        left = points_parts[i]
        tree = STRtree(left)
        trees.append(tree)

    ## The function querying the trees in parallel

    def thread_func(idxs):
        i, j = idxs
        tree = trees[i]
        right = polygons_parts[j]
        return tree.query(right, predicate="contains")

    with ThreadPoolExecutor() as pool:
        list(pool.map(thread_func, itertools.product(range(4), range(4))))
