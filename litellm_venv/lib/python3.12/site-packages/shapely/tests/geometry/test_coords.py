import numpy as np
import pytest

from shapely import LineString, geos_version
from shapely.tests.common import (
    line_string,
    line_string_m,
    line_string_z,
    line_string_zm,
    point,
    point_m,
    point_z,
    point_zm,
)


class TestCoords:
    """
    Shapely assumes contiguous C-order float64 data for internal ops.
    Data should be converted to contiguous float64 if numpy exists.
    c9a0707 broke this a little bit.
    """

    def test_data_promotion(self):
        coords = np.array([[12, 34], [56, 78]], dtype=np.float32)
        processed_coords = np.array(LineString(coords).coords)

        assert coords.tolist() == processed_coords.tolist()

    def test_data_destriding(self):
        coords = np.array([[12, 34], [56, 78]], dtype=np.float32)

        # Easy way to introduce striding: reverse list order
        processed_coords = np.array(LineString(coords[::-1]).coords)

        assert coords[::-1].tolist() == processed_coords.tolist()


class TestCoordsGetItem:
    def test_index_coords(self):
        c = [(float(x), float(-x)) for x in range(4)]
        g = LineString(c)
        for i in range(-4, 4):
            assert g.coords[i] == c[i]
        with pytest.raises(IndexError):
            g.coords[4]
        with pytest.raises(IndexError):
            g.coords[-5]

    def test_index_coords_z(self):
        c = [(float(x), float(-x), float(x * 2)) for x in range(4)]
        g = LineString(c)
        for i in range(-4, 4):
            assert g.coords[i] == c[i]
        with pytest.raises(IndexError):
            g.coords[4]
        with pytest.raises(IndexError):
            g.coords[-5]

    def test_index_coords_misc(self):
        g = LineString()  # empty
        with pytest.raises(IndexError):
            g.coords[0]
        with pytest.raises(TypeError):
            g.coords[0.0]

    def test_slice_coords(self):
        c = [(float(x), float(-x)) for x in range(4)]
        g = LineString(c)
        assert g.coords[1:] == c[1:]
        assert g.coords[:-1] == c[:-1]
        assert g.coords[::-1] == c[::-1]
        assert g.coords[::2] == c[::2]
        assert g.coords[:4] == c[:4]
        assert g.coords[4:] == c[4:] == []

    def test_slice_coords_z(self):
        c = [(float(x), float(-x), float(x * 2)) for x in range(4)]
        g = LineString(c)
        assert g.coords[1:] == c[1:]
        assert g.coords[:-1] == c[:-1]
        assert g.coords[::-1] == c[::-1]
        assert g.coords[::2] == c[::2]
        assert g.coords[:4] == c[:4]
        assert g.coords[4:] == c[4:] == []


class TestXY:
    """New geometry/coordseq method 'xy' makes numpy interop easier"""

    def test_arrays(self):
        x, y = LineString([(0, 0), (1, 1)]).xy
        assert len(x) == 2
        assert list(x) == [0.0, 1.0]
        assert len(y) == 2
        assert list(y) == [0.0, 1.0]


@pytest.mark.parametrize("geom", [point, point_z, line_string, line_string_z])
def test_coords_array_copy(geom):
    """Test CoordinateSequence.__array__ method."""
    coord_seq = geom.coords
    assert np.array(coord_seq) is not np.array(coord_seq)
    assert np.array(coord_seq, copy=True) is not np.array(coord_seq, copy=True)

    # Behaviour of copy=False is different between NumPy 1.x and 2.x
    if int(np.version.short_version.split(".", 1)[0]) >= 2:
        with pytest.raises(ValueError, match="A copy is always created"):
            np.array(coord_seq, copy=False)
    else:
        assert np.array(coord_seq, copy=False) is np.array(coord_seq, copy=False)


@pytest.mark.skipif(geos_version < (3, 12, 0), reason="GEOS < 3.12")
def test_coords_with_m():
    assert point_m.coords[:] == [(2.0, 3.0, 5.0)]
    assert point_zm.coords[:] == [(2.0, 3.0, 4.0, 5.0)]
    assert line_string_m.coords[:] == [
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 2.0),
        (1.0, 1.0, 3.0),
    ]
    assert line_string_zm.coords[:] == [
        (0.0, 0.0, 4.0, 1.0),
        (1.0, 0.0, 4.0, 2.0),
        (1.0, 1.0, 4.0, 3.0),
    ]
