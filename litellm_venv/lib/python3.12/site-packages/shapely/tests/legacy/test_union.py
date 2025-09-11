import random
import unittest
from functools import partial
from itertools import islice

import pytest

from shapely.geometry import MultiPolygon, Point
from shapely.ops import unary_union


def halton(base):
    """Returns an iterator over an infinite Halton sequence"""

    def value(index):
        result = 0.0
        f = 1.0 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i = i // base
            f = f / base
        return result

    i = 1
    while i > 0:
        yield value(i)
        i += 1


class UnionTestCase(unittest.TestCase):
    def test_unary_union_partial(self):
        # Use a partial function to make 100 points uniformly distributed
        # in a 40x40 box centered on 0,0.

        r = partial(random.uniform, -20.0, 20.0)
        points = [Point(r(), r()) for i in range(100)]

        # Buffer the points, producing 100 polygon spots
        spots = [p.buffer(2.5) for p in points]

        # Perform a cascaded union of the polygon spots, dissolving them
        # into a collection of polygon patches
        u = unary_union(spots)
        assert u.geom_type in ("Polygon", "MultiPolygon")

    def setUp(self):
        # Instead of random points, use deterministic, pseudo-random Halton
        # sequences for repeatability sake.
        self.coords = zip(
            list(islice(halton(5), 20, 120)),
            list(islice(halton(7), 20, 120)),
        )

    def test_unary_union(self):
        patches = [Point(xy).buffer(0.05) for xy in self.coords]
        u = unary_union(patches)
        assert u.geom_type == "MultiPolygon"
        assert u.area == pytest.approx(0.718572540569)

    def test_unary_union_multi(self):
        # Test of multipart input based on comment by @schwehr at
        # https://github.com/shapely/shapely/issues/47#issuecomment-21809308
        patches = MultiPolygon([Point(xy).buffer(0.05) for xy in self.coords])
        assert unary_union(patches).area == pytest.approx(0.71857254056)
        assert unary_union([patches, patches]).area == pytest.approx(0.71857254056)
