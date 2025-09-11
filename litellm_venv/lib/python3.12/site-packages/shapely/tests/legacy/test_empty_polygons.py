from shapely.geometry import MultiPolygon, Point, Polygon


def test_empty_polygon():
    """No constructor arg makes an empty polygon geometry."""
    assert Polygon().is_empty


def test_empty_multipolygon():
    """No constructor arg makes an empty multipolygon geometry."""
    assert MultiPolygon().is_empty


def test_multipolygon_empty_polygon():
    """An empty polygon passed to MultiPolygon() makes an empty
    multipolygon geometry."""
    assert MultiPolygon([Polygon()]).is_empty


def test_multipolygon_empty_among_polygon():
    """An empty polygon passed to MultiPolygon() is ignored."""
    assert len(MultiPolygon([Point(0, 0).buffer(1.0), Polygon()]).geoms) == 1
