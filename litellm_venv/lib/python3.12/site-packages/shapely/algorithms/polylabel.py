"""Provides functions for finding the pole of inaccessibility for a given polygon."""

from shapely._geometry import get_point
from shapely.constructive import maximum_inscribed_circle


def polylabel(polygon, tolerance=1.0):
    """Find pole of inaccessibility for a given polygon.

    Based on Vladimir Agafonkin's https://github.com/mapbox/polylabel

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Polygon for which to find the pole of inaccessibility.
    tolerance : int or float, optional
        `tolerance` represents the highest resolution in units of the
        input geometry that will be considered for a solution. (default
        value is 1.0).

    Returns
    -------
    shapely.geometry.Point
        A point representing the pole of inaccessibility for the given input
        polygon.

    Raises
    ------
    shapely.errors.TopologicalError
        If the input polygon is not a valid geometry.

    Examples
    --------
    >>> from shapely.ops import polylabel
    >>> from shapely import LineString
    >>> polygon = LineString([(0, 0), (50, 200), (100, 100), (20, 50),
    ... (-100, -20), (-150, -200)]).buffer(100)
    >>> polylabel(polygon, tolerance=0.001)
    <POINT (59.733 111.33)>

    """
    line = maximum_inscribed_circle(polygon, tolerance)
    return get_point(line, 0)
