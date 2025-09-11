import numpy as np
import pytest

import shapely
from shapely import GeometryCollection, LineString, Point, wkt
from shapely.geometry import shape


@pytest.fixture()
def geometrycollection_geojson():
    return {
        "type": "GeometryCollection",
        "geometries": [
            {"type": "Point", "coordinates": (0, 3, 0)},
            {"type": "LineString", "coordinates": ((2, 0), (1, 0))},
        ],
    }


@pytest.mark.parametrize(
    "geom",
    [
        GeometryCollection(),
        GeometryCollection([]),
        shape({"type": "GeometryCollection", "geometries": []}),
        wkt.loads("GEOMETRYCOLLECTION EMPTY"),
    ],
)
def test_empty(geom):
    assert geom.geom_type == "GeometryCollection"
    assert geom.is_empty
    assert len(geom.geoms) == 0
    assert list(geom.geoms) == []


def test_empty_subgeoms():
    geom = GeometryCollection([Point(), LineString()])
    assert geom.geom_type == "GeometryCollection"
    assert geom.is_empty
    assert len(geom.geoms) == 2
    parts = list(geom.geoms)
    if shapely.geos_version < (3, 9, 0):
        # the accessed empty 2D point has a 3D coordseq on GEOS 3.8
        parts[0] = shapely.force_2d(parts[0])
    assert parts == [Point(), LineString()]


def test_child_with_deleted_parent():
    # test that we can remove a collection while keeping
    # children around
    a = LineString([(0, 0), (1, 1), (1, 2), (2, 2)])
    b = LineString([(0, 0), (1, 1), (2, 1), (2, 2)])
    collection = a.intersection(b)

    child = collection.geoms[0]
    # delete parent of child
    del collection

    # access geometry, this should not seg fault as 1.2.15 did
    assert child.wkt is not None


def test_from_numpy_array():
    geoms = np.array([Point(0, 0), LineString([(1, 1), (2, 2)])])
    geom = GeometryCollection(geoms)
    assert len(geom.geoms) == 2
    np.testing.assert_array_equal(geoms, geom.geoms)


def test_from_geojson(geometrycollection_geojson):
    geom = shape(geometrycollection_geojson)
    assert geom.geom_type == "GeometryCollection"
    assert len(geom.geoms) == 2

    geom_types = [g.geom_type for g in geom.geoms]
    assert "Point" in geom_types
    assert "LineString" in geom_types


def test_geointerface(geometrycollection_geojson):
    geom = shape(geometrycollection_geojson)
    assert geom.__geo_interface__ == geometrycollection_geojson


def test_len_raises(geometrycollection_geojson):
    geom = shape(geometrycollection_geojson)
    with pytest.raises(TypeError):
        len(geom)


def test_numpy_object_array():
    geom = GeometryCollection([LineString([(0, 0), (1, 1)])])
    ar = np.empty(1, object)
    ar[:] = [geom]
    assert ar[0] == geom
