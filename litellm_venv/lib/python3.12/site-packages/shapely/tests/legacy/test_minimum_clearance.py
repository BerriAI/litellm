"""
Tests for the minimum clearance property.
"""

import math

from shapely.wkt import loads as load_wkt


def test_point():
    point = load_wkt("POINT (0 0)")
    assert point.minimum_clearance == math.inf


def test_linestring():
    line = load_wkt("LINESTRING (0 0, 1 1, 2 2)")
    assert round(line.minimum_clearance, 6) == 1.414214


def test_simple_polygon():
    poly = load_wkt("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))")
    assert poly.minimum_clearance == 1.0


def test_more_complicated_polygon():
    poly = load_wkt(
        "POLYGON ((20 20, 34 124, 70 140, 130 130, 70 100, 110 70, 170 20, 90 10, "
        "20 20))"
    )
    assert round(poly.minimum_clearance, 6) == 35.777088
