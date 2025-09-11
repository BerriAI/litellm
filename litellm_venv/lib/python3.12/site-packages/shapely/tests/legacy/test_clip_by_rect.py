"""
Tests for GEOSClipByRect based on unit tests from libgeos.

There are some expected differences due to Shapely's handling of empty
geometries.
"""

import pytest

from shapely.ops import clip_by_rect
from shapely.wkt import dumps as dump_wkt, loads as load_wkt


def test_point_outside():
    """Point outside"""
    geom1 = load_wkt("POINT (0 0)")
    geom2 = clip_by_rect(geom1, 10, 10, 20, 20)
    assert dump_wkt(geom2, rounding_precision=0) == "GEOMETRYCOLLECTION EMPTY"


def test_point_inside():
    """Point inside"""
    geom1 = load_wkt("POINT (15 15)")
    geom2 = clip_by_rect(geom1, 10, 10, 20, 20)
    assert dump_wkt(geom2, rounding_precision=0) == "POINT (15 15)"


def test_point_on_boundary():
    """Point on boundary"""
    geom1 = load_wkt("POINT (15 10)")
    geom2 = clip_by_rect(geom1, 10, 10, 20, 20)
    assert dump_wkt(geom2, rounding_precision=0) == "GEOMETRYCOLLECTION EMPTY"


def test_line_outside():
    """Line outside"""
    geom1 = load_wkt("LINESTRING (0 0, -5 5)")
    geom2 = clip_by_rect(geom1, 10, 10, 20, 20)
    assert dump_wkt(geom2, rounding_precision=0) == "GEOMETRYCOLLECTION EMPTY"


def test_line_inside():
    """Line inside"""
    geom1 = load_wkt("LINESTRING (15 15, 16 15)")
    geom2 = clip_by_rect(geom1, 10, 10, 20, 20)
    assert dump_wkt(geom2, rounding_precision=0) == "LINESTRING (15 15, 16 15)"


def test_line_on_boundary():
    """Line on boundary"""
    geom1 = load_wkt("LINESTRING (10 15, 10 10, 15 10)")
    geom2 = clip_by_rect(geom1, 10, 10, 20, 20)
    assert dump_wkt(geom2, rounding_precision=0) == "GEOMETRYCOLLECTION EMPTY"


def test_line_splitting_rectangle():
    """Line splitting rectangle"""
    geom1 = load_wkt("LINESTRING (10 5, 25 20)")
    geom2 = clip_by_rect(geom1, 10, 10, 20, 20)
    assert dump_wkt(geom2, rounding_precision=0) == "LINESTRING (15 10, 20 15)"


@pytest.mark.xfail(reason="TODO issue to CCW")
def test_polygon_shell_ccw_fully_on_rectangle_boundary():
    """Polygon shell (CCW) fully on rectangle boundary"""
    geom1 = load_wkt("POLYGON ((10 10, 20 10, 20 20, 10 20, 10 10))")
    geom2 = clip_by_rect(geom1, 10, 10, 20, 20)
    assert (
        dump_wkt(geom2, rounding_precision=0)
        == "POLYGON ((10 10, 20 10, 20 20, 10 20, 10 10))"
    )


@pytest.mark.xfail(reason="TODO issue to CW")
def test_polygon_shell_cc_fully_on_rectangle_boundary():
    """Polygon shell (CW) fully on rectangle boundary"""
    geom1 = load_wkt("POLYGON ((10 10, 10 20, 20 20, 20 10, 10 10))")
    geom2 = clip_by_rect(geom1, 10, 10, 20, 20)
    assert (
        dump_wkt(geom2, rounding_precision=0)
        == "POLYGON ((10 10, 20 10, 20 20, 10 20, 10 10))"
    )


def polygon_hole_ccw_fully_on_rectangle_boundary():
    """Polygon hole (CCW) fully on rectangle boundary"""
    geom1 = load_wkt(
        "POLYGON ((0 0, 0 30, 30 30, 30 0, 0 0), (10 10, 20 10, 20 20, 10 20, 10 10))"
    )
    geom2 = clip_by_rect(geom1, 10, 10, 20, 20)
    assert dump_wkt(geom2, rounding_precision=0) == "GEOMETRYCOLLECTION EMPTY"


def polygon_hole_cw_fully_on_rectangle_boundary():
    """Polygon hole (CW) fully on rectangle boundary"""
    geom1 = load_wkt(
        "POLYGON ((0 0, 0 30, 30 30, 30 0, 0 0), (10 10, 10 20, 20 20, 20 10, 10 10))"
    )
    geom2 = clip_by_rect(geom1, 10, 10, 20, 20)
    assert dump_wkt(geom2, rounding_precision=0) == "GEOMETRYCOLLECTION EMPTY"


def polygon_fully_within_rectangle():
    """Polygon fully within rectangle"""
    wkt = "POLYGON ((1 1, 1 30, 30 30, 30 1, 1 1), (10 10, 20 10, 20 20, 10 20, 10 10))"
    geom1 = load_wkt(wkt)
    geom2 = clip_by_rect(geom1, 0, 0, 40, 40)
    assert dump_wkt(geom2, rounding_precision=0) == wkt


def polygon_overlapping_rectangle():
    """Polygon overlapping rectangle"""
    wkt = "POLYGON ((0 0, 0 30, 30 30, 30 0, 0 0), (10 10, 20 10, 20 20, 10 20, 10 10))"
    geom1 = load_wkt(wkt)
    geom2 = clip_by_rect(geom1, 5, 5, 15, 15)
    assert (
        dump_wkt(geom2, rounding_precision=0)
        == "POLYGON ((5 5, 5 15, 10 15, 10 10, 15 10, 15 5, 5 5))"
    )
