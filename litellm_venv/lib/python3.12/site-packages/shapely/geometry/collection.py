"""Multi-part collections of geometries."""

import shapely
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry


class GeometryCollection(BaseMultipartGeometry):
    """Collection of one or more geometries that can be of different types.

    Parameters
    ----------
    geoms : list
        A list of shapely geometry instances, which may be of varying geometry
        types.

    Attributes
    ----------
    geoms : sequence
        A sequence of Shapely geometry instances

    Examples
    --------
    Create a GeometryCollection with a Point and a LineString

    >>> from shapely import GeometryCollection, LineString, Point
    >>> p = Point(51, -1)
    >>> l = LineString([(52, -1), (49, 2)])
    >>> gc = GeometryCollection([p, l])

    """

    __slots__ = []

    def __new__(self, geoms=None):
        """Create a new GeometryCollection."""
        if isinstance(geoms, BaseGeometry):
            # TODO(shapely-2.0) do we actually want to split Multi-part geometries?
            # this is needed for the split() tests
            if hasattr(geoms, "geoms"):
                geoms = geoms.geoms
            else:
                geoms = [geoms]
        elif geoms is None or len(geoms) == 0:
            # TODO better empty constructor
            return shapely.from_wkt("GEOMETRYCOLLECTION EMPTY")

        return shapely.geometrycollections(geoms)

    @property
    def __geo_interface__(self):
        """Return a GeoJSON-like mapping of the geometry collection."""
        geometries = []
        for geom in self.geoms:
            geometries.append(geom.__geo_interface__)
        return dict(type="GeometryCollection", geometries=geometries)


shapely.lib.registry[7] = GeometryCollection
