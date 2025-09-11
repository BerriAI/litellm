import numpy as np
import pytest

from shapely import Point, geos_version
from shapely.coords import CoordinateSequence
from shapely.errors import DimensionError, UnsupportedGEOSVersionError


def test_from_coordinates():
    # Point
    p = Point(1.0, 2.0)
    assert p.coords[:] == [(1.0, 2.0)]
    assert p.has_z is False

    # PointZ
    p = Point(1.0, 2.0, 3.0)
    assert p.coords[:] == [(1.0, 2.0, 3.0)]
    assert p.has_z

    # empty
    p = Point()
    assert p.is_empty
    assert isinstance(p.coords, CoordinateSequence)
    assert p.coords[:] == []


def test_from_sequence():
    # From single coordinate pair
    p = Point((3.0, 4.0))
    assert p.coords[:] == [(3.0, 4.0)]
    p = Point([3.0, 4.0])
    assert p.coords[:] == [(3.0, 4.0)]

    # From coordinate sequence
    p = Point([(3.0, 4.0)])
    assert p.coords[:] == [(3.0, 4.0)]
    p = Point([[3.0, 4.0]])
    assert p.coords[:] == [(3.0, 4.0)]

    # PointZ
    p = Point((3.0, 4.0, 5.0))
    assert p.coords[:] == [(3.0, 4.0, 5.0)]
    p = Point([3.0, 4.0, 5.0])
    assert p.coords[:] == [(3.0, 4.0, 5.0)]
    p = Point([(3.0, 4.0, 5.0)])
    assert p.coords[:] == [(3.0, 4.0, 5.0)]


def test_from_numpy():
    # Construct from a numpy array
    p = Point(np.array([1.0, 2.0]))
    assert p.coords[:] == [(1.0, 2.0)]

    p = Point(np.array([1.0, 2.0, 3.0]))
    assert p.coords[:] == [(1.0, 2.0, 3.0)]


def test_from_numpy_xy():
    # Construct from separate x, y numpy arrays - if those are length 1,
    # this is allowed for compat with shapely 1.8
    # (https://github.com/shapely/shapely/issues/1587)
    p = Point(np.array([1.0]), np.array([2.0]))
    assert p.coords[:] == [(1.0, 2.0)]

    p = Point(np.array([1.0]), np.array([2.0]), np.array([3.0]))
    assert p.coords[:] == [(1.0, 2.0, 3.0)]


def test_from_point():
    # From another point
    p = Point(3.0, 4.0)
    q = Point(p)
    assert q.coords[:] == [(3.0, 4.0)]

    p = Point(3.0, 4.0, 5.0)
    q = Point(p)
    assert q.coords[:] == [(3.0, 4.0, 5.0)]


def test_from_generator():
    gen = (coord for coord in [(1.0, 2.0)])
    p = Point(gen)
    assert p.coords[:] == [(1.0, 2.0)]


def test_from_invalid():
    with pytest.raises(TypeError, match="takes at most 3 arguments"):
        Point(1, 2, 3, 4)

    # this worked in shapely 1.x, just ignoring the other coords
    with pytest.raises(
        ValueError, match="takes only scalar or 1-size vector arguments"
    ):
        Point([(2, 3), (11, 4)])


class TestPoint:
    def test_point(self):
        # Test XY point
        p = Point(1.0, 2.0)
        assert p.x == 1.0
        assert type(p.x) is float
        assert p.y == 2.0
        assert type(p.y) is float
        assert p.coords[:] == [(1.0, 2.0)]
        assert str(p) == p.wkt
        assert p.has_z is False
        with pytest.raises(DimensionError):
            p.z
        if geos_version >= (3, 12, 0):
            assert p.has_m is False
            with pytest.raises(DimensionError):
                p.m
        else:
            with pytest.raises(UnsupportedGEOSVersionError):
                p.m

        # Check XYZ point
        p = Point(1.0, 2.0, 3.0)
        assert p.coords[:] == [(1.0, 2.0, 3.0)]
        assert str(p) == p.wkt
        assert p.has_z is True
        assert p.z == 3.0
        assert type(p.z) is float
        if geos_version >= (3, 12, 0):
            assert p.has_m is False
            with pytest.raises(DimensionError):
                p.m

            # TODO: Check XYM and XYZM points

        # Coordinate access
        p = Point((3.0, 4.0))
        assert p.x == 3.0
        assert p.y == 4.0
        assert tuple(p.coords) == ((3.0, 4.0),)
        assert p.coords[0] == (3.0, 4.0)
        with pytest.raises(IndexError):  # index out of range
            p.coords[1]

        # Bounds
        assert p.bounds == (3.0, 4.0, 3.0, 4.0)

        # Geo interface
        assert p.__geo_interface__ == {"type": "Point", "coordinates": (3.0, 4.0)}

    def test_point_empty(self):
        # Test Non-operability of Null geometry
        p_null = Point()
        assert p_null.wkt == "POINT EMPTY"
        assert p_null.coords[:] == []
        assert p_null.area == 0.0

        assert p_null.__geo_interface__ == {"type": "Point", "coordinates": ()}

    def test_coords(self):
        # From Array.txt
        p = Point(0.0, 0.0, 1.0)
        coords = p.coords[0]
        assert coords == (0.0, 0.0, 1.0)

        # Convert to Numpy array, passing through Python sequence
        a = np.asarray(coords)
        assert a.ndim == 1
        assert a.size == 3
        assert a.shape == (3,)


def test_point_immutable():
    p = Point(3.0, 4.0)

    with pytest.raises(AttributeError):
        p.coords = (2.0, 1.0)

    with pytest.raises(TypeError):
        p.coords[0] = (2.0, 1.0)


def test_point_array_coercion():
    # don't convert to array of coordinates, keep objects
    p = Point(3.0, 4.0)
    arr = np.array(p)
    assert arr.ndim == 0
    assert arr.size == 1
    assert arr.dtype == np.dtype("object")
    assert arr.item() == p


def test_numpy_empty_point_coords():
    pe = Point()

    # Access the coords
    a = np.asarray(pe.coords)
    assert a.shape == (0, 2)


def test_numpy_object_array():
    geom = Point(3.0, 4.0)
    ar = np.empty(1, object)
    ar[:] = [geom]
    assert ar[0] == geom
