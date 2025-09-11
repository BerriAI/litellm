import unittest

from shapely import geometry


class BoxTestCase(unittest.TestCase):
    def test_ccw(self):
        b = geometry.box(0, 0, 1, 1, ccw=True)
        assert b.exterior.coords[0] == (1.0, 0.0)
        assert b.exterior.coords[1] == (1.0, 1.0)

    def test_ccw_default(self):
        b = geometry.box(0, 0, 1, 1)
        assert b.exterior.coords[0] == (1.0, 0.0)
        assert b.exterior.coords[1] == (1.0, 1.0)

    def test_cw(self):
        b = geometry.box(0, 0, 1, 1, ccw=False)
        assert b.exterior.coords[0] == (0.0, 0.0)
        assert b.exterior.coords[1] == (0.0, 1.0)
