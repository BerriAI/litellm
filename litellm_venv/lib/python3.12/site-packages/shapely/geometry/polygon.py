"""Polygons and their linear ring components."""

import numpy as np

import shapely
from shapely import _geometry_helpers
from shapely.algorithms.cga import signed_area  # noqa
from shapely.errors import TopologicalError
from shapely.geometry.base import BaseGeometry
from shapely.geometry.linestring import LineString
from shapely.geometry.point import Point

__all__ = ["LinearRing", "Polygon", "orient"]


def _unpickle_linearring(wkb):
    linestring = shapely.from_wkb(wkb)
    srid = shapely.get_srid(linestring)
    linearring = _geometry_helpers.linestring_to_linearring(linestring)
    if srid:
        linearring = shapely.set_srid(linearring, srid)
    return linearring


class LinearRing(LineString):
    """Geometry type composed of one or more line segments that forms a closed loop.

    A LinearRing is a closed, one-dimensional feature.
    A LinearRing that crosses itself or touches itself at a single point is
    invalid and operations on it may fail.

    Parameters
    ----------
    coordinates : sequence
        A sequence of (x, y [,z]) numeric coordinate pairs or triples, or
        an array-like with shape (N, 2) or (N, 3).
        Also can be a sequence of Point objects.

    Notes
    -----
    Rings are automatically closed. There is no need to specify a final
    coordinate pair identical to the first.

    Examples
    --------
    Construct a square ring.

    >>> from shapely import LinearRing
    >>> ring = LinearRing( ((0, 0), (0, 1), (1 ,1 ), (1 , 0)) )
    >>> ring.is_closed
    True
    >>> list(ring.coords)
    [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    >>> ring.length
    4.0

    """

    __slots__ = []

    def __new__(self, coordinates=None):
        """Create a new LinearRing geometry."""
        if coordinates is None:
            # empty geometry
            # TODO better way?
            return shapely.from_wkt("LINEARRING EMPTY")
        elif isinstance(coordinates, LineString):
            if type(coordinates) is LinearRing:
                # return original objects since geometries are immutable
                return coordinates
            elif not coordinates.is_valid:
                raise TopologicalError("An input LineString must be valid.")
            else:
                # LineString
                # TODO convert LineString to LinearRing more directly?
                coordinates = coordinates.coords

        else:
            if hasattr(coordinates, "__array__"):
                coordinates = np.asarray(coordinates)
            if isinstance(coordinates, np.ndarray) and np.issubdtype(
                coordinates.dtype, np.number
            ):
                pass
            else:
                # check coordinates on points
                def _coords(o):
                    if isinstance(o, Point):
                        return o.coords[0]
                    else:
                        return [float(c) for c in o]

                coordinates = np.array([_coords(o) for o in coordinates])
                if not np.issubdtype(coordinates.dtype, np.number):
                    # conversion of coords to 2D array failed, this might be due
                    # to inconsistent coordinate dimensionality
                    raise ValueError("Inconsistent coordinate dimensionality")

        if len(coordinates) == 0:
            # empty geometry
            # TODO better constructor + should shapely.linearrings handle this?
            return shapely.from_wkt("LINEARRING EMPTY")

        geom = shapely.linearrings(coordinates)
        if not isinstance(geom, LinearRing):
            raise ValueError("Invalid values passed to LinearRing constructor")
        return geom

    @property
    def __geo_interface__(self):
        """Return a GeoJSON-like mapping of the LinearRing geometry."""
        return {"type": "LinearRing", "coordinates": tuple(self.coords)}

    def __reduce__(self):
        """Pickle support.

        WKB doesn't differentiate between LineString and LinearRing so we
        need to move the coordinate sequence into the correct geometry type
        """
        return (_unpickle_linearring, (shapely.to_wkb(self, include_srid=True),))

    @property
    def is_ccw(self):
        """True if the ring is oriented counter clock-wise."""
        return bool(shapely.is_ccw(self))

    @property
    def is_simple(self):
        """True if the geometry is simple.

        Simple means that any self-intersections are only at boundary points.
        """
        return bool(shapely.is_simple(self))


shapely.lib.registry[2] = LinearRing


class InteriorRingSequence:
    _parent = None
    _ndim = None
    _index = 0
    _length = 0

    def __init__(self, parent):
        self._parent = parent
        self._ndim = parent._ndim

    def __iter__(self):
        self._index = 0
        self._length = self.__len__()
        return self

    def __next__(self):
        if self._index < self._length:
            ring = self._get_ring(self._index)
            self._index += 1
            return ring
        else:
            raise StopIteration

    def __len__(self):
        return shapely.get_num_interior_rings(self._parent)

    def __getitem__(self, key):
        m = self.__len__()
        if isinstance(key, int):
            if key + m < 0 or key >= m:
                raise IndexError("index out of range")
            if key < 0:
                i = m + key
            else:
                i = key
            return self._get_ring(i)
        elif isinstance(key, slice):
            res = []
            start, stop, stride = key.indices(m)
            for i in range(start, stop, stride):
                res.append(self._get_ring(i))
            return res
        else:
            raise TypeError("key must be an index or slice")

    def _get_ring(self, i):
        return shapely.get_interior_ring(self._parent, i)


class Polygon(BaseGeometry):
    """A geometry type representing an area that is enclosed by a linear ring.

    A polygon is a two-dimensional feature and has a non-zero area. It may
    have one or more negative-space "holes" which are also bounded by linear
    rings. If any rings cross each other, the feature is invalid and
    operations on it may fail.

    Parameters
    ----------
    shell : sequence
        A sequence of (x, y [,z]) numeric coordinate pairs or triples, or
        an array-like with shape (N, 2) or (N, 3).
        Also can be a sequence of Point objects.
    holes : sequence
        A sequence of objects which satisfy the same requirements as the
        shell parameters above

    Attributes
    ----------
    exterior : LinearRing
        The ring which bounds the positive space of the polygon.
    interiors : sequence
        A sequence of rings which bound all existing holes.

    Examples
    --------
    Create a square polygon with no holes

    >>> from shapely import Polygon
    >>> coords = ((0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.))
    >>> polygon = Polygon(coords)
    >>> polygon.area
    1.0

    """

    __slots__ = []

    def __new__(self, shell=None, holes=None):
        """Create a new Polygon geometry."""
        if shell is None:
            # empty geometry
            # TODO better way?
            return shapely.from_wkt("POLYGON EMPTY")
        elif isinstance(shell, Polygon):
            # return original objects since geometries are immutable
            return shell
        else:
            shell = LinearRing(shell)

        if holes is not None:
            if len(holes) == 0:
                # shapely constructor cannot handle holes=[]
                holes = None
            else:
                holes = [LinearRing(ring) for ring in holes]

        geom = shapely.polygons(shell, holes=holes)
        if not isinstance(geom, Polygon):
            raise ValueError("Invalid values passed to Polygon constructor")
        return geom

    @property
    def exterior(self):
        """Return the exterior ring of the polygon."""
        return shapely.get_exterior_ring(self)

    @property
    def interiors(self):
        """Return the sequence of interior rings of the polygon."""
        if self.is_empty:
            return []
        return InteriorRingSequence(self)

    @property
    def coords(self):
        """Not implemented for polygons."""
        raise NotImplementedError(
            "Component rings have coordinate sequences, but the polygon does not"
        )

    @property
    def __geo_interface__(self):
        """Return a GeoJSON-like mapping of the Polygon geometry."""
        if self.exterior == LinearRing():
            coords = []
        else:
            coords = [tuple(self.exterior.coords)]
            for hole in self.interiors:
                coords.append(tuple(hole.coords))
        return {"type": "Polygon", "coordinates": tuple(coords)}

    def svg(self, scale_factor=1.0, fill_color=None, opacity=None):
        """Return SVG path element for the Polygon geometry.

        Parameters
        ----------
        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is to use "#66cc99" if
            geometry is valid, and "#ff3333" if invalid.
        opacity : float
            Float number between 0 and 1 for color opacity. Default value is 0.6

        """
        if self.is_empty:
            return "<g />"
        if fill_color is None:
            fill_color = "#66cc99" if self.is_valid else "#ff3333"
        if opacity is None:
            opacity = 0.6
        exterior_coords = [["{},{}".format(*c) for c in self.exterior.coords]]
        interior_coords = [
            ["{},{}".format(*c) for c in interior.coords] for interior in self.interiors
        ]
        path = " ".join(
            [
                "M {} L {} z".format(coords[0], " L ".join(coords[1:]))
                for coords in exterior_coords + interior_coords
            ]
        )
        return (
            f'<path fill-rule="evenodd" fill="{fill_color}" stroke="#555555" '
            f'stroke-width="{2.0 * scale_factor}" opacity="{opacity}" d="{path}" />'
        )

    @classmethod
    def from_bounds(cls, xmin, ymin, xmax, ymax):
        """Construct a `Polygon()` from spatial bounds."""
        return cls([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])


shapely.lib.registry[3] = Polygon


def orient(polygon, sign=1.0):
    """Return an oriented polygon.

    It is recommended to use :func:`shapely.orient_polygons` instead.

    Parameters
    ----------
    polygon : shapely.Polygon
    sign : float, default 1.
        The sign of the result's signed area.
        A non-negative sign means that the coordinates of the geometry's exterior
        rings will be oriented counter-clockwise.

    Returns
    -------
    Geometry or array_like

    Refer to :func:`shapely.orient_polygons` for full documentation.

    """
    return shapely.orient_polygons(polygon, exterior_cw=sign < 0.0)
