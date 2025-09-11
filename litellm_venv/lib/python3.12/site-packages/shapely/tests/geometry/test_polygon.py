"""Polygons and Linear Rings"""

import numpy as np
import pytest

from shapely import LinearRing, LineString, Point, Polygon
from shapely.coords import CoordinateSequence
from shapely.errors import TopologicalError
from shapely.wkb import loads as load_wkb


def test_empty_linearring_coords():
    assert LinearRing().coords[:] == []


def test_linearring_from_coordinate_sequence():
    expected_coords = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]

    ring = LinearRing([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
    assert ring.coords[:] == expected_coords

    ring = LinearRing([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
    assert ring.coords[:] == expected_coords


def test_linearring_from_points():
    # From Points
    expected_coords = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]

    ring = LinearRing([Point(0.0, 0.0), Point(0.0, 1.0), Point(1.0, 1.0)])
    assert ring.coords[:] == expected_coords


def test_linearring_from_closed_linestring():
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    line = LineString(coords)
    ring = LinearRing(line)
    assert len(ring.coords) == 4
    assert ring.coords[:] == coords
    assert ring.geom_type == "LinearRing"


def test_linearring_from_unclosed_linestring():
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    line = LineString(coords[:-1])  # Pass in unclosed line
    ring = LinearRing(line)
    assert len(ring.coords) == 4
    assert ring.coords[:] == coords
    assert ring.geom_type == "LinearRing"


def test_linearring_from_invalid():
    coords = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    line = LineString(coords)
    assert not line.is_valid
    with pytest.raises(TopologicalError):
        LinearRing(line)


def test_linearring_from_too_short_linestring():
    # Creation of LinearRing request at least 3 coordinates (unclosed) or
    # 4 coordinates (closed)
    coords = [(0.0, 0.0), (1.0, 1.0)]
    line = LineString(coords)
    with pytest.raises(ValueError, match="requires at least 4 coordinates"):
        LinearRing(line)


def test_linearring_from_linearring():
    coords = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]
    ring = LinearRing(coords)
    assert ring.coords[:] == coords


def test_linearring_from_generator():
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    gen = (coord for coord in coords)
    ring = LinearRing(gen)
    assert ring.coords[:] == coords


def test_linearring_from_empty():
    ring = LinearRing()
    assert ring.is_empty
    assert isinstance(ring.coords, CoordinateSequence)
    assert ring.coords[:] == []

    ring = LinearRing([])
    assert ring.is_empty
    assert isinstance(ring.coords, CoordinateSequence)
    assert ring.coords[:] == []


def test_linearring_from_numpy():
    # Construct from a numpy array
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]

    ring = LinearRing(np.array(coords))
    assert ring.coords[:] == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]


def test_numpy_linearring_coords():
    from numpy.testing import assert_array_equal

    ring = LinearRing([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
    ra = np.asarray(ring.coords)
    expected = np.asarray([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)])
    assert_array_equal(ra, expected)


def test_numpy_empty_linearring_coords():
    ring = LinearRing()
    assert np.asarray(ring.coords).shape == (0, 2)


def test_numpy_object_array():
    geom = Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
    ar = np.empty(1, object)
    ar[:] = [geom]
    assert ar[0] == geom


def test_polygon_from_coordinate_sequence():
    coords = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]

    # Construct a polygon, exterior ring only
    polygon = Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
    assert polygon.exterior.coords[:] == coords
    assert len(polygon.interiors) == 0

    polygon = Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
    assert polygon.exterior.coords[:] == coords
    assert len(polygon.interiors) == 0


def test_polygon_from_coordinate_sequence_with_holes():
    coords = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]

    # Interior rings (holes)
    polygon = Polygon(coords, [[(0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25)]])
    assert polygon.exterior.coords[:] == coords
    assert len(polygon.interiors) == 1
    assert len(polygon.interiors[0].coords) == 5

    # Multiple interior rings with different length
    coords = [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)]
    holes = [
        [(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)],
        [(3, 3), (3, 4), (4, 5), (5, 4), (5, 3), (3, 3)],
    ]
    polygon = Polygon(coords, holes)
    assert polygon.exterior.coords[:] == coords
    assert len(polygon.interiors) == 2
    assert len(polygon.interiors[0].coords) == 5
    assert len(polygon.interiors[1].coords) == 6


def test_polygon_from_linearring():
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    ring = LinearRing(coords)

    polygon = Polygon(ring)
    assert polygon.exterior.coords[:] == coords
    assert len(polygon.interiors) == 0

    # from shell and holes linearrings
    shell = LinearRing([(0.0, 0.0), (70.0, 120.0), (140.0, 0.0), (0.0, 0.0)])
    holes = [
        LinearRing([(60.0, 80.0), (80.0, 80.0), (70.0, 60.0), (60.0, 80.0)]),
        LinearRing([(30.0, 10.0), (50.0, 10.0), (40.0, 30.0), (30.0, 10.0)]),
        LinearRing([(90.0, 10), (110.0, 10.0), (100.0, 30.0), (90.0, 10.0)]),
    ]
    polygon = Polygon(shell, holes)
    assert polygon.exterior.coords[:] == shell.coords[:]
    assert len(polygon.interiors) == 3
    for i in range(3):
        assert polygon.interiors[i].coords[:] == holes[i].coords[:]


def test_polygon_from_linestring():
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    line = LineString(coords)
    polygon = Polygon(line)
    assert polygon.exterior.coords[:] == coords

    # from unclosed linestring
    line = LineString(coords[:-1])
    polygon = Polygon(line)
    assert polygon.exterior.coords[:] == coords


def test_polygon_from_points():
    polygon = Polygon([Point(0.0, 0.0), Point(0.0, 1.0), Point(1.0, 1.0)])
    expected_coords = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]
    assert polygon.exterior.coords[:] == expected_coords


def test_polygon_from_polygon():
    coords = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
    polygon = Polygon(coords, [[(0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25)]])

    # Test from another Polygon
    copy = Polygon(polygon)
    assert len(copy.exterior.coords) == 5
    assert len(copy.interiors) == 1
    assert len(copy.interiors[0].coords) == 5


def test_polygon_from_invalid():
    # Error handling
    with pytest.raises(ValueError):
        # A LinearRing must have at least 3 coordinate tuples
        Polygon([[1, 2], [2, 3]])


def test_polygon_from_empty():
    polygon = Polygon()
    assert polygon.is_empty
    assert polygon.exterior.coords[:] == []

    polygon = Polygon([])
    assert polygon.is_empty
    assert polygon.exterior.coords[:] == []


def test_polygon_from_numpy():
    a = np.array(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)))
    polygon = Polygon(a)
    assert len(polygon.exterior.coords) == 5
    assert polygon.exterior.coords[:] == [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (1.0, 0.0),
        (0.0, 0.0),
    ]
    assert len(polygon.interiors) == 0


def test_polygon_from_generator():
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    gen = (coord for coord in coords)
    polygon = Polygon(gen)
    assert polygon.exterior.coords[:] == coords


class TestPolygon:
    def test_linearring(self):
        # Initialization
        # Linear rings won't usually be created by users, but by polygons
        coords = ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))
        ring = LinearRing(coords)
        assert len(ring.coords) == 5
        assert ring.coords[0] == ring.coords[4]
        assert ring.coords[0] == ring.coords[-1]
        assert ring.is_ring is True

    def test_polygon(self):
        coords = ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))

        # Construct a polygon, exterior ring only
        polygon = Polygon(coords)
        assert len(polygon.exterior.coords) == 5

        # Ring Access
        assert isinstance(polygon.exterior, LinearRing)
        ring = polygon.exterior
        assert len(ring.coords) == 5
        assert ring.coords[0] == ring.coords[4]
        assert ring.coords[0] == (0.0, 0.0)
        assert ring.is_ring is True
        assert len(polygon.interiors) == 0

        # Create a new polygon from WKB
        data = polygon.wkb
        polygon = None
        ring = None
        polygon = load_wkb(data)
        ring = polygon.exterior
        assert len(ring.coords) == 5
        assert ring.coords[0] == ring.coords[4]
        assert ring.coords[0] == (0.0, 0.0)
        assert ring.is_ring is True
        polygon = None

        # Interior rings (holes)
        polygon = Polygon(
            coords, [((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25))]
        )
        assert len(polygon.exterior.coords) == 5
        assert len(polygon.interiors[0].coords) == 5
        with pytest.raises(IndexError):  # index out of range
            polygon.interiors[1]

        # Coordinate getter raises exceptions
        with pytest.raises(NotImplementedError):
            polygon.coords

        # Geo interface
        assert polygon.__geo_interface__ == {
            "type": "Polygon",
            "coordinates": (
                ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)),
                ((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25), (0.25, 0.25)),
            ),
        }

    def test_linearring_empty(self):
        # Test Non-operability of Null rings
        r_null = LinearRing()
        assert r_null.wkt == "LINEARRING EMPTY"
        assert r_null.length == 0.0

    def test_dimensions(self):
        # Background: see http://trac.gispython.org/lab/ticket/168
        # http://lists.gispython.org/pipermail/community/2008-August/001859.html

        coords = ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0))
        polygon = Polygon(coords)
        assert polygon._ndim == 3
        gi = polygon.__geo_interface__
        assert gi["coordinates"] == (
            (
                (0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (1.0, 1.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
        )

        e = polygon.exterior
        assert e._ndim == 3
        gi = e.__geo_interface__
        assert gi["coordinates"] == (
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )

    def test_attribute_chains(self):
        # Attribute Chaining
        # See also ticket #151.
        p = Polygon([(0.0, 0.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)])
        assert list(p.boundary.coords) == [
            (0.0, 0.0),
            (0.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 0.0),
            (0.0, 0.0),
        ]

        ec = list(Point(0.0, 0.0).buffer(1.0, quad_segs=1).exterior.coords)
        assert isinstance(ec, list)  # TODO: this is a poor test

        # Test chained access to interiors
        p = Polygon(
            [(0.0, 0.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)],
            [[(-0.25, 0.25), (-0.25, 0.75), (-0.75, 0.75), (-0.75, 0.25)]],
        )
        assert p.area == 0.75

        """Not so much testing the exact values here, which are the
        responsibility of the geometry engine (GEOS), but that we can get
        chain functions and properties using anonymous references.
        """
        assert list(p.interiors[0].coords) == [
            (-0.25, 0.25),
            (-0.25, 0.75),
            (-0.75, 0.75),
            (-0.75, 0.25),
            (-0.25, 0.25),
        ]
        xy = next(iter(p.interiors[0].buffer(1).exterior.coords))
        assert len(xy) == 2

        # Test multiple operators, boundary of a buffer
        ec = list(p.buffer(1).boundary.coords)
        assert isinstance(ec, list)  # TODO: this is a poor test

    def test_empty_equality(self):
        # Test equals operator, including empty geometries
        # see issue #338

        point1 = Point(0, 0)
        polygon1 = Polygon([(0.0, 0.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)])
        polygon2 = Polygon([(0.0, 0.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)])
        polygon_empty1 = Polygon()
        polygon_empty2 = Polygon()

        assert point1 != polygon1
        assert polygon_empty1 == polygon_empty2
        assert polygon1 != polygon_empty1
        assert polygon1 == polygon2
        assert polygon_empty1 is not None

    def test_from_bounds(self):
        xmin, ymin, xmax, ymax = -180, -90, 180, 90
        coords = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
        assert Polygon(coords) == Polygon.from_bounds(xmin, ymin, xmax, ymax)

    def test_empty_polygon_exterior(self):
        p = Polygon()
        assert p.exterior == LinearRing()


def test_linearring_immutable():
    ring = LinearRing([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])

    with pytest.raises(AttributeError):
        ring.coords = [(1.0, 1.0), (2.0, 2.0), (1.0, 2.0)]

    with pytest.raises(TypeError):
        ring.coords[0] = (1.0, 1.0)


class TestLinearRingGetItem:
    def test_index_linearring(self):
        shell = LinearRing([(0.0, 0.0), (70.0, 120.0), (140.0, 0.0), (0.0, 0.0)])
        holes = [
            LinearRing([(60.0, 80.0), (80.0, 80.0), (70.0, 60.0), (60.0, 80.0)]),
            LinearRing([(30.0, 10.0), (50.0, 10.0), (40.0, 30.0), (30.0, 10.0)]),
            LinearRing([(90.0, 10), (110.0, 10.0), (100.0, 30.0), (90.0, 10.0)]),
        ]
        g = Polygon(shell, holes)
        for i in range(-3, 3):
            assert g.interiors[i].equals(holes[i])
        with pytest.raises(IndexError):
            g.interiors[3]
        with pytest.raises(IndexError):
            g.interiors[-4]

    def test_index_linearring_misc(self):
        g = Polygon()  # empty
        with pytest.raises(IndexError):
            g.interiors[0]
        with pytest.raises(TypeError):
            g.interiors[0.0]

    def test_slice_linearring(self):
        shell = LinearRing([(0.0, 0.0), (70.0, 120.0), (140.0, 0.0), (0.0, 0.0)])
        holes = [
            LinearRing([(60.0, 80.0), (80.0, 80.0), (70.0, 60.0), (60.0, 80.0)]),
            LinearRing([(30.0, 10.0), (50.0, 10.0), (40.0, 30.0), (30.0, 10.0)]),
            LinearRing([(90.0, 10), (110.0, 10.0), (100.0, 30.0), (90.0, 10.0)]),
        ]
        g = Polygon(shell, holes)
        t = [a.equals(b) for (a, b) in zip(g.interiors[1:], holes[1:])]
        assert all(t)
        t = [a.equals(b) for (a, b) in zip(g.interiors[:-1], holes[:-1])]
        assert all(t)
        t = [a.equals(b) for (a, b) in zip(g.interiors[::-1], holes[::-1])]
        assert all(t)
        t = [a.equals(b) for (a, b) in zip(g.interiors[::2], holes[::2])]
        assert all(t)
        t = [a.equals(b) for (a, b) in zip(g.interiors[:3], holes[:3])]
        assert all(t)
        assert g.interiors[3:] == holes[3:] == []
