import unittest

import pytest

from shapely import geometry
from shapely.constructive import BufferCapStyle, BufferJoinStyle
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE


@pytest.mark.parametrize("distance", [float("nan"), float("inf")])
def test_non_finite_distance(distance):
    g = geometry.Point(0, 0)
    with pytest.raises(ValueError, match="distance must be finite"):
        g.buffer(distance)


class BufferTests(unittest.TestCase):
    """Test Buffer Point/Line/Polygon with and without single_sided params"""

    def test_empty(self):
        g = geometry.Point(0, 0)
        h = g.buffer(0)
        assert h.is_empty

    def test_point(self):
        g = geometry.Point(0, 0)
        h = g.buffer(1, quad_segs=1)
        assert h.geom_type == "Polygon"
        expected_coord = [(1.0, 0.0), (0, -1.0), (-1.0, 0), (0, 1.0), (1.0, 0.0)]
        for index, coord in enumerate(h.exterior.coords):
            assert coord[0] == pytest.approx(expected_coord[index][0])
            assert coord[1] == pytest.approx(expected_coord[index][1])

    def test_point_single_sidedd(self):
        g = geometry.Point(0, 0)
        h = g.buffer(1, quad_segs=1, single_sided=True)
        assert h.geom_type == "Polygon"
        expected_coord = [(1.0, 0.0), (0, -1.0), (-1.0, 0), (0, 1.0), (1.0, 0.0)]
        for index, coord in enumerate(h.exterior.coords):
            assert coord[0] == pytest.approx(expected_coord[index][0])
            assert coord[1] == pytest.approx(expected_coord[index][1])

    def test_line(self):
        g = geometry.LineString([[0, 0], [0, 1]])
        h = g.buffer(1, quad_segs=1)
        assert h.geom_type == "Polygon"
        expected_coord = [
            (-1.0, 1.0),
            (0, 2.0),
            (1.0, 1.0),
            (1.0, 0.0),
            (0, -1.0),
            (-1.0, 0.0),
            (-1.0, 1.0),
        ]
        for index, coord in enumerate(h.exterior.coords):
            assert coord[0] == pytest.approx(expected_coord[index][0])
            assert coord[1] == pytest.approx(expected_coord[index][1])

    def test_line_single_sideded_left(self):
        g = geometry.LineString([[0, 0], [0, 1]])
        h = g.buffer(1, quad_segs=1, single_sided=True)
        assert h.geom_type == "Polygon"
        expected_coord = [(0.0, 1.0), (0.0, 0.0), (-1.0, 0.0), (-1.0, 1.0), (0.0, 1.0)]
        for index, coord in enumerate(h.exterior.coords):
            assert coord[0] == pytest.approx(expected_coord[index][0])
            assert coord[1] == pytest.approx(expected_coord[index][1])

    def test_line_single_sideded_right(self):
        g = geometry.LineString([[0, 0], [0, 1]])
        h = g.buffer(-1, quad_segs=1, single_sided=True)
        assert h.geom_type == "Polygon"
        expected_coord = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
        for index, coord in enumerate(h.exterior.coords):
            assert coord[0] == pytest.approx(expected_coord[index][0])
            assert coord[1] == pytest.approx(expected_coord[index][1])

    def test_polygon(self):
        g = geometry.Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        h = g.buffer(1, quad_segs=1)
        assert h.geom_type == "Polygon"
        expected_coord = [
            (-1.0, 0.0),
            (-1.0, 1.0),
            (0.0, 2.0),
            (1.0, 2.0),
            (2.0, 1.0),
            (2.0, 0.0),
            (1.0, -1.0),
            (0.0, -1.0),
            (-1.0, 0.0),
        ]
        for index, coord in enumerate(h.exterior.coords):
            assert coord[0] == pytest.approx(expected_coord[index][0])
            assert coord[1] == pytest.approx(expected_coord[index][1])

    def test_polygon_single_sideded(self):
        g = geometry.Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        h = g.buffer(1, quad_segs=1, single_sided=True)
        assert h.geom_type == "Polygon"
        expected_coord = [
            (-1.0, 0.0),
            (-1.0, 1.0),
            (0.0, 2.0),
            (1.0, 2.0),
            (2.0, 1.0),
            (2.0, 0.0),
            (1.0, -1.0),
            (0.0, -1.0),
            (-1.0, 0.0),
        ]
        for index, coord in enumerate(h.exterior.coords):
            assert coord[0] == pytest.approx(expected_coord[index][0])
            assert coord[1] == pytest.approx(expected_coord[index][1])

    def test_enum_values(self):
        assert CAP_STYLE.round == 1
        assert CAP_STYLE.round == BufferCapStyle.round
        assert CAP_STYLE.flat == 2
        assert CAP_STYLE.flat == BufferCapStyle.flat
        assert CAP_STYLE.square == 3
        assert CAP_STYLE.square == BufferCapStyle.square

        assert JOIN_STYLE.round == 1
        assert JOIN_STYLE.round == BufferJoinStyle.round
        assert JOIN_STYLE.mitre == 2
        assert JOIN_STYLE.mitre == BufferJoinStyle.mitre
        assert JOIN_STYLE.bevel == 3
        assert JOIN_STYLE.bevel == BufferJoinStyle.bevel

    def test_cap_style(self):
        g = geometry.LineString([[0, 0], [1, 0]])
        h = g.buffer(1, cap_style=BufferCapStyle.round)
        assert h == g.buffer(1, cap_style=CAP_STYLE.round)
        assert h == g.buffer(1, cap_style="round")

        h = g.buffer(1, cap_style=BufferCapStyle.flat)
        assert h == g.buffer(1, cap_style=CAP_STYLE.flat)
        assert h == g.buffer(1, cap_style="flat")

        h = g.buffer(1, cap_style=BufferCapStyle.square)
        assert h == g.buffer(1, cap_style=CAP_STYLE.square)
        assert h == g.buffer(1, cap_style="square")

    def test_buffer_style(self):
        g = geometry.LineString([[0, 0], [1, 0]])
        h = g.buffer(1, join_style=BufferJoinStyle.round)
        assert h == g.buffer(1, join_style=JOIN_STYLE.round)
        assert h == g.buffer(1, join_style="round")

        h = g.buffer(1, join_style=BufferJoinStyle.mitre)
        assert h == g.buffer(1, join_style=JOIN_STYLE.mitre)
        assert h == g.buffer(1, join_style="mitre")

        h = g.buffer(1, join_style=BufferJoinStyle.bevel)
        assert h == g.buffer(1, join_style=JOIN_STYLE.bevel)
        assert h == g.buffer(1, join_style="bevel")


def test_deprecated_quadsegs():
    point = geometry.Point(0, 0)
    with pytest.warns(FutureWarning):
        result = point.buffer(1, quadsegs=1)
    expected = point.buffer(1, quad_segs=1)
    assert result.equals(expected)


def test_deprecated_resolution():
    point = geometry.Point(0, 0)
    with pytest.deprecated_call(match="Use 'quad_segs' instead"):
        result = point.buffer(1, resolution=1)
    expected = point.buffer(1, quad_segs=1)
    assert result.equals(expected)
