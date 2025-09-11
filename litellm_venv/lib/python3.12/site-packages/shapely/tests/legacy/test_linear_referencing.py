import unittest

import pytest

import shapely
from shapely.geometry import LineString, MultiLineString, Point


class LinearReferencingTestCase(unittest.TestCase):
    def setUp(self):
        self.point = Point(1, 1)
        self.line1 = LineString([(0, 0), (2, 0)])
        self.line2 = LineString([(3, 0), (3, 6)])
        self.multiline = MultiLineString(
            [list(self.line1.coords), list(self.line2.coords)]
        )

    def test_line1_project(self):
        assert self.line1.project(self.point) == 1.0
        assert self.line1.project(self.point, normalized=True) == 0.5

    def test_alias_project(self):
        assert self.line1.line_locate_point(self.point) == 1.0
        assert self.line1.line_locate_point(self.point, normalized=True) == 0.5

    def test_line2_project(self):
        assert self.line2.project(self.point) == 1.0
        assert self.line2.project(self.point, normalized=True) == pytest.approx(
            0.16666666666, 8
        )

    def test_multiline_project(self):
        assert self.multiline.project(self.point) == 1.0
        assert self.multiline.project(self.point, normalized=True) == 0.125

    def test_not_supported_project(self):
        with pytest.raises(shapely.GEOSException, match="IllegalArgumentException"):
            self.point.buffer(1.0).project(self.point)

    def test_not_on_line_project(self):
        # Points that aren't on the line project to 0.
        assert self.line1.project(Point(-10, -10)) == 0.0

    def test_line1_interpolate(self):
        assert self.line1.interpolate(0.5).equals(Point(0.5, 0.0))
        assert self.line1.interpolate(-0.5).equals(Point(1.5, 0.0))
        assert self.line1.interpolate(0.5, normalized=True).equals(Point(1, 0))
        assert self.line1.interpolate(-0.5, normalized=True).equals(Point(1, 0))

    def test_alias_interpolate(self):
        assert self.line1.line_interpolate_point(0.5).equals(Point(0.5, 0.0))
        assert self.line1.line_interpolate_point(-0.5).equals(Point(1.5, 0.0))
        assert self.line1.line_interpolate_point(0.5, normalized=True).equals(
            Point(1, 0)
        )
        assert self.line1.line_interpolate_point(-0.5, normalized=True).equals(
            Point(1, 0)
        )

    def test_line2_interpolate(self):
        assert self.line2.interpolate(0.5).equals(Point(3.0, 0.5))
        assert self.line2.interpolate(0.5, normalized=True).equals(Point(3, 3))

    def test_multiline_interpolate(self):
        assert self.multiline.interpolate(0.5).equals(Point(0.5, 0))
        assert self.multiline.interpolate(0.5, normalized=True).equals(Point(3.0, 2.0))

    def test_line_ends_interpolate(self):
        # Distances greater than length of the line or less than
        # zero yield the line's ends.
        assert self.line1.interpolate(-1000).equals(Point(0.0, 0.0))
        assert self.line1.interpolate(1000).equals(Point(2.0, 0.0))
