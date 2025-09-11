import unittest

import pytest

from shapely.geometry import Point
from shapely.ops import nearest_points


class Nearest(unittest.TestCase):
    def test_nearest(self):
        first, second = nearest_points(
            Point(0, 0).buffer(1.0),
            Point(3, 0).buffer(1.0),
        )
        assert first.x == pytest.approx(1.0)
        assert second.x == pytest.approx(2.0)
        assert first.y == pytest.approx(0.0)
        assert second.y == pytest.approx(0.0)
