import pytest
from numpy.testing import assert_allclose

from shapely import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    box,
    get_coordinates,
)
from shapely.ops import orient
from shapely.plotting import patch_from_polygon, plot_line, plot_points, plot_polygon

pytest.importorskip("matplotlib")


def test_patch_from_polygon():
    poly = box(0, 0, 1, 1)
    artist = patch_from_polygon(poly, facecolor="red", edgecolor="blue", linewidth=3)
    assert equal_color(artist.get_facecolor(), "red")
    assert equal_color(artist.get_edgecolor(), "blue")
    assert artist.get_linewidth() == 3


def test_patch_from_polygon_with_interior():
    poly = box(0, 0, 1, 1).difference(box(0.2, 0.2, 0.5, 0.5))
    artist = patch_from_polygon(poly, facecolor="red", edgecolor="blue", linewidth=3)
    assert equal_color(artist.get_facecolor(), "red")
    assert equal_color(artist.get_edgecolor(), "blue")
    assert artist.get_linewidth() == 3


def test_patch_from_multipolygon():
    poly = box(0, 0, 1, 1).union(box(2, 2, 3, 3))
    artist = patch_from_polygon(poly, facecolor="red", edgecolor="blue", linewidth=3)
    assert equal_color(artist.get_facecolor(), "red")
    assert equal_color(artist.get_edgecolor(), "blue")
    assert artist.get_linewidth() == 3


def test_plot_polygon():
    poly = box(0, 0, 1, 1)
    artist, _ = plot_polygon(poly)
    plot_coords = artist.get_path().vertices
    assert_allclose(plot_coords, get_coordinates(poly))

    # overriding default styling
    artist = plot_polygon(poly, add_points=False, color="red", linewidth=3)
    assert equal_color(artist.get_facecolor(), "red", alpha=0.3)
    assert equal_color(artist.get_edgecolor(), "red", alpha=1.0)
    assert artist.get_linewidth() == 3


def test_plot_polygon_with_interior():
    poly = box(0, 0, 1, 1).difference(box(0.2, 0.2, 0.5, 0.5))
    artist, _ = plot_polygon(poly)
    plot_coords = artist.get_path().vertices
    assert_allclose(plot_coords, get_coordinates(orient(poly)))


def test_plot_multipolygon():
    poly = box(0, 0, 1, 1).union(box(2, 2, 3, 3))
    artist, _ = plot_polygon(poly)
    plot_coords = artist.get_path().vertices
    assert_allclose(plot_coords, get_coordinates(poly))


def test_plot_multipolygon_with_interior():
    poly1 = box(0, 0, 1, 1).difference(box(0.2, 0.2, 0.5, 0.5))
    poly2 = box(3, 3, 6, 6).difference(box(4, 4, 5, 5))
    poly = MultiPolygon([poly1, poly2])

    artist, _ = plot_polygon(poly)
    plot_coords = artist.get_path().vertices
    assert_allclose(plot_coords, get_coordinates(orient(poly)))


def test_plot_line():
    line = LineString([(0, 0), (1, 0), (1, 1)])
    artist, _ = plot_line(line)
    plot_coords = artist.get_path().vertices
    assert_allclose(plot_coords, get_coordinates(line))

    # overriding default styling
    artist = plot_line(line, add_points=False, color="red", linewidth=3)
    assert equal_color(artist.get_edgecolor(), "red")
    assert equal_color(artist.get_facecolor(), "none")
    assert artist.get_linewidth() == 3


def test_plot_multilinestring():
    line = MultiLineString(
        [LineString([(0, 0), (1, 0), (1, 1)]), LineString([(2, 2), (3, 3)])]
    )
    artist, _ = plot_line(line)
    plot_coords = artist.get_path().vertices
    assert_allclose(plot_coords, get_coordinates(line))


def test_plot_points():
    for geom in [Point(0, 0), LineString([(0, 0), (1, 0), (1, 1)]), box(0, 0, 1, 1)]:
        artist = plot_points(geom)
        plot_coords = artist.get_path().vertices
        assert_allclose(plot_coords, get_coordinates(geom))
        assert artist.get_linestyle() == "None"

    # overriding default styling
    geom = Point(0, 0)
    artist = plot_points(geom, color="red", marker="+", fillstyle="top")
    assert artist.get_color() == "red"
    assert artist.get_marker() == "+"
    assert artist.get_fillstyle() == "top"


def equal_color(actual, expected, alpha=None):
    from matplotlib import colors

    conv = colors.colorConverter

    return actual == conv.to_rgba(expected, alpha=alpha)
