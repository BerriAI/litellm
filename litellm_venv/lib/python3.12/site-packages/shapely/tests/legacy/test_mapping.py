import unittest

from shapely.geometry import Point, Polygon, mapping


class MappingTestCase(unittest.TestCase):
    def test_point(self):
        m = mapping(Point(0, 0))
        assert m["type"] == "Point"
        assert m["coordinates"] == (0.0, 0.0)

    def test_empty_polygon(self):
        """Empty polygons will round trip without error"""
        assert mapping(Polygon()) is not None
