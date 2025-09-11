"""Geometry classes and factories."""

from shapely.geometry.base import CAP_STYLE, JOIN_STYLE
from shapely.geometry.collection import GeometryCollection
from shapely.geometry.geo import box, mapping, shape
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.point import Point
from shapely.geometry.polygon import LinearRing, Polygon

__all__ = [
    "CAP_STYLE",
    "JOIN_STYLE",
    "GeometryCollection",
    "LineString",
    "LinearRing",
    "MultiLineString",
    "MultiPoint",
    "MultiPolygon",
    "Point",
    "Polygon",
    "box",
    "mapping",
    "shape",
]
