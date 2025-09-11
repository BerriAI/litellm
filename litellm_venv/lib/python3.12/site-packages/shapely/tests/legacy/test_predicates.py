"""Test GEOS predicates"""

import unittest

import pytest

import shapely
from shapely import geos_version
from shapely.geometry import Point, Polygon


class PredicatesTestCase(unittest.TestCase):
    def test_binary_predicates(self):
        point = Point(0.0, 0.0)
        point2 = Point(2.0, 2.0)

        assert point.disjoint(Point(-1.0, -1.0))
        assert not point.touches(Point(-1.0, -1.0))
        assert not point.crosses(Point(-1.0, -1.0))
        assert not point.within(Point(-1.0, -1.0))
        assert not point.contains(Point(-1.0, -1.0))
        assert not point.equals(Point(-1.0, -1.0))
        assert not point.touches(Point(-1.0, -1.0))
        assert point.equals(Point(0.0, 0.0))
        assert point.covers(Point(0.0, 0.0))
        assert point.covered_by(Point(0.0, 0.0))
        assert not point.covered_by(point2)
        assert not point2.covered_by(point)
        assert not point.covers(Point(-1.0, -1.0))

    def test_unary_predicates(self):
        point = Point(0.0, 0.0)

        assert not point.is_empty
        assert point.is_valid
        assert point.is_simple
        assert not point.is_ring
        assert not point.has_z

    def test_binary_predicate_exceptions(self):
        p1 = [
            (339, 346),
            (459, 346),
            (399, 311),
            (340, 277),
            (399, 173),
            (280, 242),
            (339, 415),
            (280, 381),
            (460, 207),
            (339, 346),
        ]
        p2 = [
            (339, 207),
            (280, 311),
            (460, 138),
            (399, 242),
            (459, 277),
            (459, 415),
            (399, 381),
            (519, 311),
            (520, 242),
            (519, 173),
            (399, 450),
            (339, 207),
        ]

        g1 = Polygon(p1)
        g2 = Polygon(p2)
        assert not g1.is_valid
        assert not g2.is_valid
        if geos_version < (3, 13, 0):
            with pytest.raises(shapely.GEOSException):
                g1.within(g2)
        else:  # resolved with RelateNG
            assert not g1.within(g2)

    def test_relate_pattern(self):
        # a pair of partially overlapping polygons, and a nearby point
        g1 = Polygon([(0, 0), (0, 1), (3, 1), (3, 0), (0, 0)])
        g2 = Polygon([(1, -1), (1, 2), (2, 2), (2, -1), (1, -1)])
        g3 = Point(5, 5)

        assert g1.relate(g2) == "212101212"
        assert g1.relate_pattern(g2, "212101212")
        assert g1.relate_pattern(g2, "*********")
        assert g1.relate_pattern(g2, "2********")
        assert g1.relate_pattern(g2, "T********")
        assert not g1.relate_pattern(g2, "112101212")
        assert not g1.relate_pattern(g2, "1********")
        assert g1.relate_pattern(g3, "FF2FF10F2")

        # an invalid pattern should raise an exception
        with pytest.raises(shapely.GEOSException, match="IllegalArgumentException"):
            g1.relate_pattern(g2, "fail")
