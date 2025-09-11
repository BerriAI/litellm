import unittest

from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge


class LineMergeTestCase(unittest.TestCase):
    def test_linemerge(self):
        lines = MultiLineString([[(0, 0), (1, 1)], [(2, 0), (2, 1), (1, 1)]])
        result = linemerge(lines)
        assert isinstance(result, LineString)
        assert not result.is_ring
        assert len(result.coords) == 4
        assert result.coords[0] == (0.0, 0.0)
        assert result.coords[3] == (2.0, 0.0)

        lines2 = MultiLineString([((0, 0), (1, 1)), ((0, 0), (2, 0), (2, 1), (1, 1))])
        result = linemerge(lines2)
        assert result.is_ring
        assert len(result.coords) == 5

        lines3 = [
            LineString([(0, 0), (1, 1)]),
            LineString([(0, 0), (0, 1)]),
        ]
        result = linemerge(lines3)
        assert not result.is_ring
        assert len(result.coords) == 3
        assert result.coords[0] == (0.0, 1.0)
        assert result.coords[2] == (1.0, 1.0)

        lines4 = [
            [(0, 0), (1, 1)],
            [(0, 0), (0, 1)],
        ]
        assert result.equals(linemerge(lines4))

        lines5 = [
            ((0, 0), (1, 1)),
            ((1, 0), (0, 1)),
        ]
        result = linemerge(lines5)
        assert result.geom_type == "MultiLineString"
