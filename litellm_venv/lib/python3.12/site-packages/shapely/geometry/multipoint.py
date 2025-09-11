"""Collections of points and related utilities."""

import numpy as np

import shapely
from shapely.errors import EmptyPartError
from shapely.geometry import point
from shapely.geometry.base import BaseMultipartGeometry

__all__ = ["MultiPoint"]


class MultiPoint(BaseMultipartGeometry):
    """A collection of one or more Points.

    A MultiPoint has zero area and zero length.

    Parameters
    ----------
    points : sequence
        A sequence of Points, or a sequence of (x, y [,z]) numeric coordinate
        pairs or triples, or an array-like of shape (N, 2) or (N, 3).

    Attributes
    ----------
    geoms : sequence
        A sequence of Points

    Examples
    --------
    Construct a MultiPoint containing two Points

    >>> from shapely import MultiPoint, Point
    >>> ob = MultiPoint([[0.0, 0.0], [1.0, 2.0]])
    >>> len(ob.geoms)
    2
    >>> type(ob.geoms[0]) == Point
    True

    """

    __slots__ = []

    def __new__(self, points=None):
        """Create a new MultiPoint geometry."""
        if points is None:
            # allow creation of empty multipoints, to support unpickling
            # TODO better empty constructor
            return shapely.from_wkt("MULTIPOINT EMPTY")
        elif isinstance(points, MultiPoint):
            return points
        elif len(points) == 0:
            return shapely.from_wkt("MULTIPOINT EMPTY")

        if isinstance(points, np.ndarray) and np.issubdtype(points.dtype, np.number):
            subs = shapely.points(points)
            if not subs.ndim == 1:
                raise ValueError("Invalid values passed to MultiPoint constructor")
            if shapely.is_empty(subs).any():
                raise EmptyPartError("Can't create MultiPoint with empty component")
        else:
            subs = []
            for item in points:
                p = point.Point(item)
                if p.is_empty:
                    raise EmptyPartError("Can't create MultiPoint with empty component")
                subs.append(p)

        return shapely.multipoints(subs)

    @property
    def __geo_interface__(self):
        """Return a GeoJSON-like mapping interface for this MultiPoint."""
        return {
            "type": "MultiPoint",
            "coordinates": tuple(g.coords[0] for g in self.geoms),
        }

    def svg(self, scale_factor=1.0, fill_color=None, opacity=None):
        """Return a group of SVG circle elements for the MultiPoint geometry.

        Parameters
        ----------
        scale_factor : float
            Multiplication factor for the SVG circle diameters.  Default is 1.
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
        return (
            "<g>"
            + "".join(p.svg(scale_factor, fill_color, opacity) for p in self.geoms)
            + "</g>"
        )


shapely.lib.registry[4] = MultiPoint
