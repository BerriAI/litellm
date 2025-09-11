import numpy as np
import pytest

import shapely
from shapely import LinearRing, LineString, Point
from shapely.coords import CoordinateSequence


def test_from_coordinate_sequence():
    # From coordinate tuples
    line = LineString([(1.0, 2.0), (3.0, 4.0)])
    assert len(line.coords) == 2
    assert line.coords[:] == [(1.0, 2.0), (3.0, 4.0)]

    line = LineString([(1.0, 2.0), (3.0, 4.0)])
    assert line.coords[:] == [(1.0, 2.0), (3.0, 4.0)]


def test_from_coordinate_sequence_z():
    line = LineString([(1.0, 2.0, 3.0), (3.0, 4.0, 5.0)])
    assert line.has_z
    assert line.coords[:] == [(1.0, 2.0, 3.0), (3.0, 4.0, 5.0)]


def test_from_points():
    # From Points
    line = LineString([Point(1.0, 2.0), Point(3.0, 4.0)])
    assert line.coords[:] == [(1.0, 2.0), (3.0, 4.0)]

    line = LineString([Point(1.0, 2.0), Point(3.0, 4.0)])
    assert line.coords[:] == [(1.0, 2.0), (3.0, 4.0)]


def test_from_mix():
    # From mix of tuples and Points
    line = LineString([Point(1.0, 2.0), (2.0, 3.0), Point(3.0, 4.0)])
    assert line.coords[:] == [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]


def test_from_linestring():
    # From another linestring
    line = LineString([(1.0, 2.0), (3.0, 4.0)])
    copy = LineString(line)
    assert copy.coords[:] == [(1.0, 2.0), (3.0, 4.0)]
    assert copy.geom_type == "LineString"


def test_from_linearring():
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    ring = LinearRing(coords)
    copy = LineString(ring)
    assert copy.coords[:] == coords
    assert copy.geom_type == "LineString"


def test_from_linestring_z():
    coords = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    line = LineString(coords)
    copy = LineString(line)
    assert copy.coords[:] == coords
    assert copy.geom_type == "LineString"


def test_from_generator():
    gen = (coord for coord in [(1.0, 2.0), (3.0, 4.0)])
    line = LineString(gen)
    assert line.coords[:] == [(1.0, 2.0), (3.0, 4.0)]


def test_from_empty():
    line = LineString()
    assert line.is_empty
    assert isinstance(line.coords, CoordinateSequence)
    assert line.coords[:] == []

    line = LineString([])
    assert line.is_empty
    assert isinstance(line.coords, CoordinateSequence)
    assert line.coords[:] == []


def test_from_numpy():
    # Construct from a numpy array
    line = LineString(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert line.coords[:] == [(1.0, 2.0), (3.0, 4.0)]


def test_numpy_empty_linestring_coords():
    # Check empty
    line = LineString([])
    la = np.asarray(line.coords)

    assert la.shape == (0, 2)


def test_numpy_object_array():
    geom = LineString([(0.0, 0.0), (0.0, 1.0)])
    ar = np.empty(1, object)
    ar[:] = [geom]
    assert ar[0] == geom


@pytest.mark.filterwarnings("ignore:Creating an ndarray from ragged nested sequences:")
def test_from_invalid_dim():
    # TODO(shapely-2.0) better error message?
    # pytest.raises(
    #     ValueError, match="at least 2 coordinate tuples|at least 2 coordinates"
    # ):
    with pytest.raises(shapely.GEOSException):
        LineString([(1, 2)])

    # exact error depends on numpy version
    with pytest.raises((ValueError, TypeError)):
        LineString([(1, 2, 3), (4, 5)])

    with pytest.raises((ValueError, TypeError)):
        LineString([(1, 2), (3, 4, 5)])

    msg = r"The ordinate \(last\) dimension should be 2 or 3, got {}"
    with pytest.raises(ValueError, match=msg.format(4)):
        LineString([(1, 2, 3, 4), (4, 5, 6, 7)])

    with pytest.raises(ValueError, match=msg.format(1)):
        LineString([(1,), (4,)])


def test_from_single_coordinate():
    """Test for issue #486"""
    coords = [[-122.185933073564, 37.3629353839073]]
    with pytest.raises(shapely.GEOSException):
        ls = LineString(coords)
        ls.geom_type  # caused segfault before fix


class TestLineString:
    def test_linestring(self):
        # From coordinate tuples
        line = LineString([(1.0, 2.0), (3.0, 4.0)])
        assert len(line.coords) == 2
        assert line.coords[:] == [(1.0, 2.0), (3.0, 4.0)]

        # Bounds
        assert line.bounds == (1.0, 2.0, 3.0, 4.0)

        # Coordinate access
        assert tuple(line.coords) == ((1.0, 2.0), (3.0, 4.0))
        assert line.coords[0] == (1.0, 2.0)
        assert line.coords[1] == (3.0, 4.0)
        with pytest.raises(IndexError):
            line.coords[2]  # index out of range

        # Geo interface
        assert line.__geo_interface__ == {
            "type": "LineString",
            "coordinates": ((1.0, 2.0), (3.0, 4.0)),
        }

    def test_linestring_empty(self):
        # Test Non-operability of Null geometry
        l_null = LineString()
        assert l_null.wkt == "LINESTRING EMPTY"
        assert l_null.length == 0.0

    def test_equals_argument_order(self):
        """
        Test equals predicate functions correctly regardless of the order
        of the inputs. See issue #317.
        """
        coords = ((0, 0), (1, 0), (1, 1), (0, 0))
        ls = LineString(coords)
        lr = LinearRing(coords)

        assert ls.__eq__(lr) is False  # previously incorrectly returned True
        assert lr.__eq__(ls) is False
        assert (ls == lr) is False
        assert (lr == ls) is False

        ls_clone = LineString(coords)
        lr_clone = LinearRing(coords)

        assert ls.__eq__(ls_clone) is True
        assert lr.__eq__(lr_clone) is True
        assert (ls == ls_clone) is True
        assert (lr == lr_clone) is True

    def test_numpy_linestring_coords(self):
        from numpy.testing import assert_array_equal

        line = LineString([(1.0, 2.0), (3.0, 4.0)])
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])

        # Coordinate sequences can be adapted as well
        la = np.asarray(line.coords)
        assert_array_equal(la, expected)


def test_linestring_immutable():
    line = LineString([(1.0, 2.0), (3.0, 4.0)])

    with pytest.raises(AttributeError):
        line.coords = [(-1.0, -1.0), (1.0, 1.0)]

    with pytest.raises(TypeError):
        line.coords[0] = (-1.0, -1.0)


def test_linestring_array_coercion():
    # don't convert to array of coordinates, keep objects
    line = LineString([(1.0, 2.0), (3.0, 4.0)])
    arr = np.array(line)
    assert arr.ndim == 0
    assert arr.size == 1
    assert arr.dtype == np.dtype("object")
    assert arr.item() == line


def test_offset_curve_deprecate_positional():
    line_string = LineString([(1.0, 2.0), (3.0, 4.0)])
    with pytest.deprecated_call(
        match="positional argument `quad_segs` for `offset_curve` is deprecated"
    ):
        line_string.offset_curve(1.0, 8)
    with pytest.deprecated_call(
        match="positional arguments `quad_segs` and `join_style` "
        "for `offset_curve` are deprecated"
    ):
        line_string.offset_curve(1.0, 8, "round")
    with pytest.deprecated_call(
        match="positional arguments `quad_segs`, `join_style`, and `mitre_limit` "
        "for `offset_curve` are deprecated"
    ):
        line_string.offset_curve(1.0, 8, "round", 5.0)
