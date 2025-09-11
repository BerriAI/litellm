"""Shapely CGA algorithms."""

import numpy as np

import shapely


def signed_area(ring):
    """Return the signed area enclosed by a ring in linear time.

    Algorithm used: https://web.archive.org/web/20080209143651/http://cgafaq.info:80/wiki/Polygon_Area
    """
    coords = np.array(ring.coords)[:, :2]
    xs, ys = np.vstack([coords, coords[1]]).T
    return np.sum(xs[1:-1] * (ys[2:] - ys[:-2])) / 2.0


def _reverse_conditioned(rings, condition):
    """Return a copy of the rings potentially reversed depending on `condition`."""
    condition = np.asarray(condition)
    if np.all(condition):
        rings = shapely.reverse(rings)
    elif np.any(condition):
        rings = np.array(rings)
        rings[condition] = shapely.reverse(rings[condition])
    return rings


def _orient_polygon(geometry, exterior_cw=False):
    if geometry is None:
        return None
    if geometry.geom_type in ["MultiPolygon", "GeometryCollection"]:
        return geometry.__class__(
            [_orient_polygon(geom, exterior_cw) for geom in geometry.geoms]
        )
    # elif geometry.geom_type in ["LinearRing"]:
    #     return reverse_conditioned(geometry, is_ccw(geometry) != ccw)
    elif geometry.geom_type == "Polygon":
        rings = np.array([geometry.exterior, *geometry.interiors])
        reverse_condition = shapely.is_ccw(rings)
        reverse_condition[0] = not reverse_condition[0]
        if exterior_cw:
            reverse_condition = np.logical_not(reverse_condition)
        if np.any(reverse_condition):
            rings = _reverse_conditioned(rings, reverse_condition)
            return geometry.__class__(rings[0], rings[1:])
    return geometry


_orient_polygons_vectorized = np.frompyfunc(_orient_polygon, nin=2, nout=1)
