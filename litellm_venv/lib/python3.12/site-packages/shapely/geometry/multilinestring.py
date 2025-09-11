"""Collections of linestrings and related utilities."""

import shapely
from shapely.errors import EmptyPartError
from shapely.geometry import linestring
from shapely.geometry.base import BaseMultipartGeometry

__all__ = ["MultiLineString"]


class MultiLineString(BaseMultipartGeometry):
    """A collection of one or more LineStrings.

    A MultiLineString has non-zero length and zero area.

    Parameters
    ----------
    lines : sequence
        A sequence LineStrings, or a sequence of line-like coordinate
        sequences or array-likes (see accepted input for LineString).

    Attributes
    ----------
    geoms : sequence
        A sequence of LineStrings

    Examples
    --------
    Construct a MultiLineString containing two LineStrings.

    >>> from shapely import MultiLineString
    >>> lines = MultiLineString([[[0, 0], [1, 2]], [[4, 4], [5, 6]]])

    """

    __slots__ = []

    def __new__(self, lines=None):
        """Create a new MultiLineString geometry."""
        if not lines:
            # allow creation of empty multilinestrings, to support unpickling
            # TODO better empty constructor
            return shapely.from_wkt("MULTILINESTRING EMPTY")
        elif isinstance(lines, MultiLineString):
            return lines

        lines = getattr(lines, "geoms", lines)
        subs = []
        for item in lines:
            line = linestring.LineString(item)
            if line.is_empty:
                raise EmptyPartError(
                    "Can't create MultiLineString with empty component"
                )
            subs.append(line)

        if len(lines) == 0:
            return shapely.from_wkt("MULTILINESTRING EMPTY")

        return shapely.multilinestrings(subs)

    @property
    def __geo_interface__(self):
        """Return a GeoJSON-like mapping interface for this MultiLineString."""
        return {
            "type": "MultiLineString",
            "coordinates": tuple(tuple(c for c in g.coords) for g in self.geoms),
        }

    def svg(self, scale_factor=1.0, stroke_color=None, opacity=None):
        """Return a group of SVG polyline elements for the LineString geometry.

        Parameters
        ----------
        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        stroke_color : str, optional
            Hex string for stroke color. Default is to use "#66cc99" if
            geometry is valid, and "#ff3333" if invalid.
        opacity : float
            Float number between 0 and 1 for color opacity. Default value is 0.8

        """
        if self.is_empty:
            return "<g />"
        if stroke_color is None:
            stroke_color = "#66cc99" if self.is_valid else "#ff3333"
        return (
            "<g>"
            + "".join(p.svg(scale_factor, stroke_color, opacity) for p in self.geoms)
            + "</g>"
        )


shapely.lib.registry[5] = MultiLineString
