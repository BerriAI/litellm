import unittest

import pytest

from shapely.errors import GeometryTypeError
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point
from shapely.ops import shared_paths


class SharedPaths(unittest.TestCase):
    def test_shared_paths_forward(self):
        g1 = LineString([(0, 0), (10, 0), (10, 5), (20, 5)])
        g2 = LineString([(5, 0), (15, 0)])
        result = shared_paths(g1, g2)

        assert isinstance(result, GeometryCollection)
        assert len(result.geoms) == 2
        a, b = result.geoms
        assert isinstance(a, MultiLineString)
        assert len(a.geoms) == 1
        assert a.geoms[0].coords[:] == [(5, 0), (10, 0)]
        assert b.is_empty

    def test_shared_paths_forward2(self):
        g1 = LineString([(0, 0), (10, 0), (10, 5), (20, 5)])
        g2 = LineString([(15, 0), (5, 0)])
        result = shared_paths(g1, g2)

        assert isinstance(result, GeometryCollection)
        assert len(result.geoms) == 2
        a, b = result.geoms
        assert isinstance(b, MultiLineString)
        assert len(b.geoms) == 1
        assert b.geoms[0].coords[:] == [(5, 0), (10, 0)]
        assert a.is_empty

    def test_wrong_type(self):
        g1 = Point(0, 0)
        g2 = LineString([(5, 0), (15, 0)])

        with pytest.raises(GeometryTypeError):
            shared_paths(g1, g2)

        with pytest.raises(GeometryTypeError):
            shared_paths(g2, g1)
