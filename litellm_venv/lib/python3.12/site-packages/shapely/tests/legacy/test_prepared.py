import numpy as np
import pytest

from shapely.geometry import Point, Polygon
from shapely.prepared import PreparedGeometry, prep


def test_prepared_geometry():
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    p = PreparedGeometry(polygon)
    assert p.contains(Point(0.5, 0.5))
    assert not p.contains(Point(0.5, 1.5))


def test_prep():
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    p = prep(polygon)
    assert p.contains(Point(0.5, 0.5))
    assert not p.contains(Point(0.5, 1.5))


def test_op_not_allowed():
    p = PreparedGeometry(Point(0.0, 0.0).buffer(1.0))
    with pytest.raises(TypeError):
        Point(0.0, 0.0).union(p)


def test_predicate_not_allowed():
    p = PreparedGeometry(Point(0.0, 0.0).buffer(1.0))
    with pytest.raises(TypeError):
        Point(0.0, 0.0).contains(p)


def test_prepared_predicates():
    # check prepared predicates give the same result as regular predicates
    polygon1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    polygon2 = Polygon([(0.5, 0.5), (1.5, 0.5), (1.0, 1.0), (0.5, 0.5)])
    point2 = Point(0.5, 0.5)
    polygon_empty = Polygon()
    prepared_polygon1 = PreparedGeometry(polygon1)
    for geom2 in (polygon2, point2, polygon_empty):
        with np.errstate(invalid="ignore"):
            assert polygon1.disjoint(geom2) == prepared_polygon1.disjoint(geom2)
            assert polygon1.touches(geom2) == prepared_polygon1.touches(geom2)
            assert polygon1.intersects(geom2) == prepared_polygon1.intersects(geom2)
            assert polygon1.crosses(geom2) == prepared_polygon1.crosses(geom2)
            assert polygon1.within(geom2) == prepared_polygon1.within(geom2)
            assert polygon1.contains(geom2) == prepared_polygon1.contains(geom2)
            assert polygon1.contains_properly(
                geom2
            ) == prepared_polygon1.contains_properly(geom2)
            assert polygon1.overlaps(geom2) == prepared_polygon1.overlaps(geom2)


def test_prepare_already_prepared():
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    prepared = prep(polygon)
    # attempt to prepare an already prepared geometry with `prep`
    result = prep(prepared)
    assert isinstance(result, PreparedGeometry)
    assert result.context is polygon
    # attempt to prepare an already prepared geometry with `PreparedGeometry`
    result = PreparedGeometry(prepared)
    assert isinstance(result, PreparedGeometry)
    assert result.context is polygon
