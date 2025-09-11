"""Collections of polygons and related utilities."""

import shapely
from shapely.geometry import polygon
from shapely.geometry.base import BaseMultipartGeometry

__all__ = ["MultiPolygon"]


class MultiPolygon(BaseMultipartGeometry):
    """A collection of one or more Polygons.

    If component polygons overlap the collection is invalid and some
    operations on it may fail.

    Parameters
    ----------
    polygons : sequence
        A sequence of Polygons, or a sequence of (shell, holes) tuples
        where shell is the sequence representation of a linear ring
        (see LinearRing) and holes is a sequence of such linear rings.

    Attributes
    ----------
    geoms : sequence
        A sequence of `Polygon` instances

    Examples
    --------
    Construct a MultiPolygon from a sequence of coordinate tuples

    >>> from shapely import MultiPolygon, Polygon
    >>> ob = MultiPolygon([
    ...     (
    ...     ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
    ...     [((0.1,0.1), (0.1,0.2), (0.2,0.2), (0.2,0.1))]
    ...     )
    ... ])
    >>> len(ob.geoms)
    1
    >>> type(ob.geoms[0]) == Polygon
    True

    """

    __slots__ = []

    def __new__(self, polygons=None):
        """Create a new MultiPolygon geometry."""
        if polygons is None:
            # allow creation of empty multipolygons, to support unpickling
            # TODO better empty constructor
            return shapely.from_wkt("MULTIPOLYGON EMPTY")
        elif isinstance(polygons, MultiPolygon):
            return polygons

        polygons = getattr(polygons, "geoms", polygons)
        # remove None and empty polygons from list of Polygons
        polygons = [p for p in polygons if p]

        L = len(polygons)

        # Bail immediately if we have no input points.
        if L == 0:
            return shapely.from_wkt("MULTIPOLYGON EMPTY")

        # This function does not accept sequences of MultiPolygons: there is
        # no implicit flattening.
        if any(isinstance(p, MultiPolygon) for p in polygons):
            raise ValueError("Sequences of multi-polygons are not valid arguments")

        subs = []
        for i in range(L):
            ob = polygons[i]
            if not isinstance(ob, polygon.Polygon):
                shell = ob[0]
                if len(ob) > 1:
                    holes = ob[1]
                else:
                    holes = None
                p = polygon.Polygon(shell, holes)
            else:
                p = polygon.Polygon(ob)
            subs.append(p)

        return shapely.multipolygons(subs)

    @property
    def __geo_interface__(self):
        """Return a GeoJSON-like mapping of the MultiPolygon geometry."""
        allcoords = []
        for geom in self.geoms:
            coords = []
            coords.append(tuple(geom.exterior.coords))
            for hole in geom.interiors:
                coords.append(tuple(hole.coords))
            allcoords.append(tuple(coords))
        return {"type": "MultiPolygon", "coordinates": allcoords}

    def svg(self, scale_factor=1.0, fill_color=None, opacity=None):
        """Return group of SVG path elements for the MultiPolygon geometry.

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
        return (
            "<g>"
            + "".join(p.svg(scale_factor, fill_color, opacity) for p in self.geoms)
            + "</g>"
        )


shapely.lib.registry[6] = MultiPolygon
