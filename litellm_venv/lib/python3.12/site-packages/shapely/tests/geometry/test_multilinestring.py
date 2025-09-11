import numpy as np
import pytest

from shapely import LineString, MultiLineString
from shapely.errors import EmptyPartError
from shapely.geometry.base import dump_coords
from shapely.tests.geometry.test_multi import MultiGeometryTestCase


class TestMultiLineString(MultiGeometryTestCase):
    def test_multilinestring(self):
        # From coordinate tuples
        geom = MultiLineString([[(1.0, 2.0), (3.0, 4.0)]])
        assert isinstance(geom, MultiLineString)
        assert len(geom.geoms) == 1
        assert dump_coords(geom) == [[(1.0, 2.0), (3.0, 4.0)]]

        # From lines
        a = LineString([(1.0, 2.0), (3.0, 4.0)])
        ml = MultiLineString([a])
        assert len(ml.geoms) == 1
        assert dump_coords(ml) == [[(1.0, 2.0), (3.0, 4.0)]]

        # From another multi-line
        ml2 = MultiLineString(ml)
        assert len(ml2.geoms) == 1
        assert dump_coords(ml2) == [[(1.0, 2.0), (3.0, 4.0)]]

        # Sub-geometry Access
        geom = MultiLineString([(((0.0, 0.0), (1.0, 2.0)))])
        assert isinstance(geom.geoms[0], LineString)
        assert dump_coords(geom.geoms[0]) == [(0.0, 0.0), (1.0, 2.0)]
        with pytest.raises(IndexError):  # index out of range
            geom.geoms[1]

        # Geo interface
        assert geom.__geo_interface__ == {
            "type": "MultiLineString",
            "coordinates": (((0.0, 0.0), (1.0, 2.0)),),
        }

    def test_from_multilinestring_z(self):
        coords1 = [(0.0, 1.0, 2.0), (3.0, 4.0, 5.0)]
        coords2 = [(6.0, 7.0, 8.0), (9.0, 10.0, 11.0)]

        # From coordinate tuples
        ml = MultiLineString([coords1, coords2])
        copy = MultiLineString(ml)
        assert isinstance(copy, MultiLineString)
        assert copy.geom_type == "MultiLineString"
        assert len(copy.geoms) == 2
        assert dump_coords(copy.geoms[0]) == coords1
        assert dump_coords(copy.geoms[1]) == coords2

    def test_numpy(self):
        # Construct from a numpy array
        geom = MultiLineString([np.array(((0.0, 0.0), (1.0, 2.0)))])
        assert isinstance(geom, MultiLineString)
        assert len(geom.geoms) == 1
        assert dump_coords(geom) == [[(0.0, 0.0), (1.0, 2.0)]]

    def test_subgeom_access(self):
        line0 = LineString([(0.0, 1.0), (2.0, 3.0)])
        line1 = LineString([(4.0, 5.0), (6.0, 7.0)])
        self.subgeom_access_test(MultiLineString, [line0, line1])

    def test_create_multi_with_empty_component(self):
        msg = "Can't create MultiLineString with empty component"
        with pytest.raises(EmptyPartError, match=msg):
            MultiLineString([LineString([(0, 0), (1, 1), (2, 2)]), LineString()]).wkt


def test_numpy_object_array():
    geom = MultiLineString([[[5.0, 6.0], [7.0, 8.0]]])
    ar = np.empty(1, object)
    ar[:] = [geom]
    assert ar[0] == geom
