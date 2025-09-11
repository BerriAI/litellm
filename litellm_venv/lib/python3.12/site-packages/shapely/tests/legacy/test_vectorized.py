import unittest

import numpy as np
import pytest

from shapely.geometry import MultiPolygon, Point, box


@pytest.mark.filterwarnings("ignore:The 'shapely.vectorized:")
class VectorizedContainsTestCase(unittest.TestCase):
    def assertContainsResults(self, geom, x, y):
        from shapely.vectorized import contains

        result = contains(geom, x, y)
        x = np.asanyarray(x)
        y = np.asanyarray(y)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, bool)

        result_flat = result.flat
        x_flat, y_flat = x.flat, y.flat

        # Do the equivalent operation, only slowly, comparing the result
        # as we go.
        for idx in range(x.size):
            assert result_flat[idx] == geom.contains(Point(x_flat[idx], y_flat[idx]))
        return result

    def construct_torus(self):
        point = Point(0, 0)
        return point.buffer(5).symmetric_difference(point.buffer(2.5))

    def test_contains_poly(self):
        y, x = np.mgrid[-10:10:5j], np.mgrid[-5:15:5j]
        self.assertContainsResults(self.construct_torus(), x, y)

    def test_contains_point(self):
        y, x = np.mgrid[-10:10:5j], np.mgrid[-5:15:5j]
        self.assertContainsResults(Point(x[0], y[0]), x, y)

    def test_contains_linestring(self):
        y, x = np.mgrid[-10:10:5j], np.mgrid[-5:15:5j]
        self.assertContainsResults(Point(x[0], y[0]), x, y)

    def test_contains_multipoly(self):
        y, x = np.mgrid[-10:10:5j], np.mgrid[-5:15:5j]
        # Construct a geometry of the torus cut in half vertically.
        cut_poly = box(-1, -10, -2.5, 10)
        geom = self.construct_torus().difference(cut_poly)
        assert isinstance(geom, MultiPolygon)
        self.assertContainsResults(geom, x, y)

    def test_y_array_order(self):
        y, x = np.mgrid[-10:10:5j, -5:15:5j]
        y = y.copy("f")
        self.assertContainsResults(self.construct_torus(), x, y)

    def test_x_array_order(self):
        y, x = np.mgrid[-10:10:5j, -5:15:5j]
        x = x.copy("f")
        self.assertContainsResults(self.construct_torus(), x, y)

    def test_xy_array_order(self):
        y, x = np.mgrid[-10:10:5j, -5:15:5j]
        x = x.copy("f")
        y = y.copy("f")
        result = self.assertContainsResults(self.construct_torus(), x, y)
        # Preserve the order
        assert result.flags["F_CONTIGUOUS"]

    def test_array_dtype(self):
        y, x = np.mgrid[-10:10:5j], np.mgrid[-5:15:5j]
        x = x.astype(np.int16)
        self.assertContainsResults(self.construct_torus(), x, y)

    def test_array_2d(self):
        y, x = np.mgrid[-10:10:15j, -5:15:16j]
        result = self.assertContainsResults(self.construct_torus(), x, y)
        assert result.shape == x.shape

    def test_shapely_xy_attr_contains(self):
        g = Point(0, 0).buffer(10.0)
        self.assertContainsResults(self.construct_torus(), *g.exterior.xy)


@pytest.mark.filterwarnings("ignore:The 'shapely.vectorized:")
class VectorizedTouchesTestCase(unittest.TestCase):
    def test_touches(self):
        from shapely.vectorized import touches

        y, x = np.mgrid[-2:3:6j, -1:3:5j]
        geom = box(0, -1, 2, 2)
        result = touches(geom, x, y)
        expected = np.array(
            [
                [False, False, False, False, False],
                [False, True, True, True, False],
                [False, True, False, True, False],
                [False, True, False, True, False],
                [False, True, True, True, False],
                [False, False, False, False, False],
            ],
            dtype=bool,
        )
        from numpy.testing import assert_array_equal

        assert_array_equal(result, expected)
