import math
from itertools import islice

import numpy as np

import shapely
from shapely.affinity import affine_transform


def _oriented_envelope_min_area(geometry, **kwargs):
    """Compute the oriented envelope (minimum rotated rectangle).

    This is a fallback implementation for GEOS < 3.12 to have the correct
    minimum area behaviour.
    """
    if geometry is None:
        return None
    if geometry.is_empty:
        return shapely.from_wkt("POLYGON EMPTY")

    # first compute the convex hull
    hull = geometry.convex_hull
    try:
        coords = hull.exterior.coords
    except AttributeError:  # may be a Point or a LineString
        return hull
    # generate the edge vectors between the convex hull's coords
    edges = (
        (pt2[0] - pt1[0], pt2[1] - pt1[1])
        for pt1, pt2 in zip(coords, islice(coords, 1, None))
    )

    def _transformed_rects():
        for dx, dy in edges:
            # compute the normalized direction vector of the edge
            # vector.
            length = math.sqrt(dx**2 + dy**2)
            ux, uy = dx / length, dy / length
            # compute the normalized perpendicular vector
            vx, vy = -uy, ux
            # transform hull from the original coordinate system to
            # the coordinate system defined by the edge and compute
            # the axes-parallel bounding rectangle.
            transf_rect = affine_transform(hull, (ux, uy, vx, vy, 0, 0)).envelope
            # yield the transformed rectangle and a matrix to
            # transform it back to the original coordinate system.
            yield (transf_rect, (ux, vx, uy, vy, 0, 0))

    # check for the minimum area rectangle and return it
    transf_rect, inv_matrix = min(_transformed_rects(), key=lambda r: r[0].area)
    return affine_transform(transf_rect, inv_matrix)


_oriented_envelope_min_area_vectorized = np.frompyfunc(
    _oriented_envelope_min_area, 1, 1
)
