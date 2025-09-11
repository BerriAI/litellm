import unittest

from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import dump_coords
from shapely.ops import polygonize, polygonize_full


class PolygonizeTestCase(unittest.TestCase):
    def test_polygonize(self):
        lines = [
            LineString([(0, 0), (1, 1)]),
            LineString([(0, 0), (0, 1)]),
            LineString([(0, 1), (1, 1)]),
            LineString([(1, 1), (1, 0)]),
            LineString([(1, 0), (0, 0)]),
            LineString([(5, 5), (6, 6)]),
            Point(0, 0),
        ]
        result = list(polygonize(lines))
        assert all(isinstance(x, Polygon) for x in result)

    def test_polygonize_full(self):
        lines2 = [
            [(0, 0), (1, 1)],
            [(0, 0), (0, 1)],
            [(0, 1), (1, 1)],
            [(1, 1), (1, 0)],
            [(1, 0), (0, 0)],
            [(5, 5), (6, 6)],
            [(1, 1), (100, 100)],
        ]

        result2, cuts, dangles, invalids = polygonize_full(lines2)
        assert len(result2.geoms) == 2
        assert all(isinstance(x, Polygon) for x in result2.geoms)
        assert list(cuts.geoms) == []
        assert all(isinstance(x, LineString) for x in dangles.geoms)

        assert dump_coords(dangles) == [
            [(1.0, 1.0), (100.0, 100.0)],
            [(5.0, 5.0), (6.0, 6.0)],
        ]
        assert list(invalids.geoms) == []
