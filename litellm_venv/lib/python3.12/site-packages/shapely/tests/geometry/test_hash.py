import pytest

import shapely
from shapely.affinity import translate
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point


@pytest.mark.parametrize(
    "geom",
    [
        Point(1, 2),
        MultiPoint([(1, 2), (3, 4)]),
        LineString([(1, 2), (3, 4)]),
        Point(0, 0).buffer(1.0),
        GeometryCollection([Point(1, 2), LineString([(1, 2), (3, 4)])]),
    ],
    ids=[
        "Point",
        "MultiPoint",
        "LineString",
        "Polygon",
        "GeometryCollection",
    ],
)
def test_hash(geom):
    h1 = hash(geom)
    assert h1 == hash(shapely.from_wkb(geom.wkb))
    assert h1 != hash(translate(geom, 1.0, 2.0))
