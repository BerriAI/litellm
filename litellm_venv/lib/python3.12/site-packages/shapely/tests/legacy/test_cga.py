import unittest

import pytest

from shapely.geometry.polygon import LinearRing, Polygon, orient, signed_area


class SignedAreaTestCase(unittest.TestCase):
    def test_triangle(self):
        tri = LinearRing([(0, 0), (2, 5), (7, 0)])
        assert signed_area(tri) == pytest.approx(-7 * 5 / 2)

    def test_square(self):
        xmin, xmax = (-1, 1)
        ymin, ymax = (-2, 3)
        rect = LinearRing(
            [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]
        )
        assert signed_area(rect) == pytest.approx(10.0)


class RingOrientationTestCase(unittest.TestCase):
    def test_ccw(self):
        ring = LinearRing([(1, 0), (0, 1), (0, 0)])
        assert ring.is_ccw

    def test_cw(self):
        ring = LinearRing([(0, 0), (0, 1), (1, 0)])
        assert not ring.is_ccw


class PolygonOrienterTestCase(unittest.TestCase):
    def test_no_holes(self):
        ring = LinearRing([(0, 0), (0, 1), (1, 0)])
        polygon = Polygon(ring)
        assert not polygon.exterior.is_ccw
        polygon = orient(polygon, 1)
        assert polygon.exterior.is_ccw

    def test_holes(self):
        # fmt: off
        polygon = Polygon(
            [(0, 0), (0, 1), (1, 0)],
            [[(0.5, 0.25), (0.25, 0.5), (0.25, 0.25)]]
        )
        # fmt: on
        assert not polygon.exterior.is_ccw
        assert polygon.interiors[0].is_ccw
        polygon = orient(polygon, 1)
        assert polygon.exterior.is_ccw
        assert not polygon.interiors[0].is_ccw
