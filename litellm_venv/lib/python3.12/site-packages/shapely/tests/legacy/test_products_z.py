import unittest

from shapely.geometry import LineString


class ProductZTestCase(unittest.TestCase):
    def test_line_intersection(self):
        line1 = LineString([(0, 0, 0), (1, 1, 1)])
        line2 = LineString([(0, 1, 1), (1, 0, 0)])
        interxn = line1.intersection(line2)
        assert interxn.has_z
        assert interxn._ndim == 3
        assert 0.0 <= interxn.z <= 1.0
