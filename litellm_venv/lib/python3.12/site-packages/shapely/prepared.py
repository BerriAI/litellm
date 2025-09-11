"""Support for GEOS prepared geometry operations."""

from pickle import PicklingError

import shapely


class PreparedGeometry:
    """A geometry prepared for efficient comparison to a set of other geometries.

    Examples
    --------
    >>> from shapely.prepared import prep
    >>> from shapely.geometry import Point, Polygon
    >>> triangle = Polygon([(0.0, 0.0), (1.0, 1.0), (1.0, -1.0)])
    >>> p = prep(triangle)
    >>> p.intersects(Point(0.5, 0.5))
    True

    """

    def __init__(self, context):
        """Prepare a geometry for efficient comparison to other geometries."""
        if isinstance(context, PreparedGeometry):
            self.context = context.context
        else:
            shapely.prepare(context)
            self.context = context
        self.prepared = True

    def contains(self, other):
        """Return True if the geometry contains the other, else False."""
        return self.context.contains(other)

    def contains_properly(self, other):
        """Return True if the geometry properly contains the other, else False."""
        return self.context.contains_properly(other)

    def covers(self, other):
        """Return True if the geometry covers the other, else False."""
        return self.context.covers(other)

    def crosses(self, other):
        """Return True if the geometries cross, else False."""
        return self.context.crosses(other)

    def disjoint(self, other):
        """Return True if geometries are disjoint, else False."""
        return self.context.disjoint(other)

    def intersects(self, other):
        """Return True if geometries intersect, else False."""
        return self.context.intersects(other)

    def overlaps(self, other):
        """Return True if geometries overlap, else False."""
        return self.context.overlaps(other)

    def touches(self, other):
        """Return True if geometries touch, else False."""
        return self.context.touches(other)

    def within(self, other):
        """Return True if geometry is within the other, else False."""
        return self.context.within(other)

    def __reduce__(self):
        """Pickling is not supported."""
        raise PicklingError("Prepared geometries cannot be pickled.")


def prep(ob):
    """Create and return a prepared geometric object."""
    return PreparedGeometry(ob)
