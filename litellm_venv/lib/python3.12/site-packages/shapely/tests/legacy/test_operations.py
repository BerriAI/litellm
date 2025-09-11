import unittest

import pytest

import shapely
from shapely import geos_version
from shapely.errors import TopologicalError
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point, Polygon
from shapely.wkt import loads


class OperationsTestCase(unittest.TestCase):
    def test_operations(self):
        point = Point(0.0, 0.0)

        # General geometry
        assert point.area == 0.0
        assert point.length == 0.0
        assert point.distance(Point(-1.0, -1.0)) == pytest.approx(1.4142135623730951)

        # Topology operations

        # Envelope
        assert isinstance(point.envelope, Point)

        # Intersection
        assert point.intersection(Point(-1, -1)).is_empty

        # Buffer
        assert isinstance(point.buffer(10.0), Polygon)
        assert isinstance(point.buffer(10.0, quad_segs=32), Polygon)

        # Simplify
        p = loads(
            "POLYGON ((120 120, 140 199, 160 200, 180 199, 220 120, 122 122, 121 121, "
            "120 120))"
        )
        expected = loads(
            "POLYGON ((120 120, 140 199, 160 200, 180 199, 220 120, 120 120))"
        )
        s = p.simplify(10.0, preserve_topology=False)
        assert s.equals_exact(expected, 0.001)

        p = loads(
            "POLYGON ((80 200, 240 200, 240 60, 80 60, 80 200),"
            "(120 120, 220 120, 180 199, 160 200, 140 199, 120 120))"
        )
        expected = loads(
            "POLYGON ((80 200, 240 200, 240 60, 80 60, 80 200),"
            "(120 120, 220 120, 180 199, 160 200, 140 199, 120 120))"
        )
        s = p.simplify(10.0, preserve_topology=True)
        assert s.equals_exact(expected, 0.001)

        # Convex Hull
        assert isinstance(point.convex_hull, Point)

        # Differences
        assert isinstance(point.difference(Point(-1, 1)), Point)

        assert isinstance(point.symmetric_difference(Point(-1, 1)), MultiPoint)

        # Boundary
        assert isinstance(point.boundary, GeometryCollection)

        # Union
        assert isinstance(point.union(Point(-1, 1)), MultiPoint)

        assert isinstance(point.representative_point(), Point)
        assert isinstance(point.point_on_surface(), Point)
        assert point.representative_point() == point.point_on_surface()

        assert isinstance(point.centroid, Point)

    def test_relate(self):
        # Relate
        assert Point(0, 0).relate(Point(-1, -1)) == "FF0FFF0F2"

        # issue #294: should raise TopologicalError on exception
        invalid_polygon = loads(
            "POLYGON ((40 100, 80 100, 80 60, 40 60, 40 100), "
            "(60 60, 80 60, 80 40, 60 40, 60 60))"
        )
        assert not invalid_polygon.is_valid
        if geos_version < (3, 13, 0):
            with pytest.raises((TopologicalError, shapely.GEOSException)):
                invalid_polygon.relate(invalid_polygon)
        else:  # resolved with RelateNG
            assert invalid_polygon.relate(invalid_polygon) == "2FFF1FFF2"

    def test_hausdorff_distance(self):
        point = Point(1, 1)
        line = LineString([(2, 0), (2, 4), (3, 4)])

        distance = point.hausdorff_distance(line)
        assert distance == point.distance(Point(3, 4))

    def test_interpolate(self):
        # successful interpolation
        test_line = LineString([(1, 1), (1, 2)])
        known_point = Point(1, 1.5)
        interpolated_point = test_line.interpolate(0.5, normalized=True)
        assert interpolated_point == known_point

        # Issue #653; should nog segfault for empty geometries
        empty_line = loads("LINESTRING EMPTY")
        assert empty_line.is_empty
        interpolated_point = empty_line.interpolate(0.5, normalized=True)
        assert interpolated_point.is_empty

        # invalid geometry should raise TypeError on exception
        polygon = loads("POLYGON EMPTY")
        with pytest.raises(TypeError, match="incorrect geometry type"):
            polygon.interpolate(0.5, normalized=True)

    def test_normalize(self):
        point = Point(1, 1)
        result = point.normalize()
        assert result == point

        line = loads("MULTILINESTRING ((1 1, 0 0), (1 1, 1 2))")
        result = line.normalize()
        expected = loads("MULTILINESTRING ((1 1, 1 2), (0 0, 1 1))")
        assert result == expected
