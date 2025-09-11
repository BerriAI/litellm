"""Plot single geometries using Matplotlib.

Note: this module is experimental, and mainly targeting (interactive)
exploration, debugging and illustration purposes.

"""

import numpy as np

import shapely


def _default_ax():
    import matplotlib.pyplot as plt

    ax = plt.gca()
    ax.grid(True)
    ax.set_aspect("equal")
    return ax


def _path_from_polygon(polygon):
    from matplotlib.path import Path

    from shapely.ops import orient

    if isinstance(polygon, shapely.MultiPolygon):
        return Path.make_compound_path(
            *[_path_from_polygon(poly) for poly in polygon.geoms]
        )
    else:
        polygon = orient(polygon)
        return Path.make_compound_path(
            Path(np.asarray(polygon.exterior.coords)[:, :2]),
            *[Path(np.asarray(ring.coords)[:, :2]) for ring in polygon.interiors],
        )


def patch_from_polygon(polygon, **kwargs):
    """Get a Matplotlib patch from a (Multi)Polygon.

    Note: this function is experimental, and mainly targeting (interactive)
    exploration, debugging and illustration purposes.

    Parameters
    ----------
    polygon : shapely.Polygon or shapely.MultiPolygon
        The polygon to convert to a Matplotlib Patch.
    **kwargs
        Additional keyword arguments passed to the matplotlib Patch.

    Returns
    -------
    Matplotlib artist (PathPatch)

    """
    from matplotlib.patches import PathPatch

    return PathPatch(_path_from_polygon(polygon), **kwargs)


def plot_polygon(
    polygon,
    ax=None,
    add_points=True,
    color=None,
    facecolor=None,
    edgecolor=None,
    linewidth=None,
    **kwargs,
):
    """Plot a (Multi)Polygon.

    Note: this function is experimental, and mainly targeting (interactive)
    exploration, debugging and illustration purposes.

    Parameters
    ----------
    polygon : shapely.Polygon or shapely.MultiPolygon
        The polygon to plot.
    ax : matplotlib Axes, default None
        The axes on which to draw the plot. If not specified, will get the
        current active axes or create a new figure.
    add_points : bool, default True
        If True, also plot the coordinates (vertices) as points.
    color : matplotlib color specification
        Color for both the polygon fill (face) and boundary (edge). By default,
        the fill is using an alpha of 0.3. You can specify `facecolor` and
        `edgecolor` separately for greater control.
    facecolor : matplotlib color specification
        Color for the polygon fill.
    edgecolor : matplotlib color specification
        Color for the polygon boundary.
    linewidth : float
        The line width for the polygon boundary.
    **kwargs
        Additional keyword arguments passed to the matplotlib Patch.

    Returns
    -------
    Matplotlib artist (PathPatch), if `add_points` is false.
    A tuple of Matplotlib artists (PathPatch, Line2D), if `add_points` is true.

    """
    from matplotlib import colors

    if ax is None:
        ax = _default_ax()

    if color is None:
        color = "C0"
    color = colors.to_rgba(color)

    if facecolor is None:
        facecolor = list(color)
        facecolor[-1] = 0.3
        facecolor = tuple(facecolor)

    if edgecolor is None:
        edgecolor = color

    patch = patch_from_polygon(
        polygon, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, **kwargs
    )
    ax.add_patch(patch)
    ax.autoscale_view()

    if add_points:
        line = plot_points(polygon, ax=ax, color=color)
        return patch, line

    return patch


def plot_line(line, ax=None, add_points=True, color=None, linewidth=2, **kwargs):
    """Plot a (Multi)LineString/LinearRing.

    Note: this function is experimental, and mainly targeting (interactive)
    exploration, debugging and illustration purposes.

    Parameters
    ----------
    line : shapely.LineString or shapely.LinearRing
        The line to plot.
    ax : matplotlib Axes, default None
        The axes on which to draw the plot. If not specified, will get the
        current active axes or create a new figure.
    add_points : bool, default True
        If True, also plot the coordinates (vertices) as points.
    color : matplotlib color specification
        Color for the line (edgecolor under the hood) and points.
    linewidth : float, default 2
        The line width for the polygon boundary.
    **kwargs
        Additional keyword arguments passed to the matplotlib Patch.

    Returns
    -------
    Matplotlib artist (PathPatch)

    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    if ax is None:
        ax = _default_ax()

    if color is None:
        color = "C0"

    if isinstance(line, shapely.MultiLineString):
        path = Path.make_compound_path(
            *[Path(np.asarray(mline.coords)[:, :2]) for mline in line.geoms]
        )
    else:
        path = Path(np.asarray(line.coords)[:, :2])

    patch = PathPatch(
        path, facecolor="none", edgecolor=color, linewidth=linewidth, **kwargs
    )
    ax.add_patch(patch)
    ax.autoscale_view()

    if add_points:
        line = plot_points(line, ax=ax, color=color)
        return patch, line

    return patch


def plot_points(geom, ax=None, color=None, marker="o", **kwargs):
    """Plot a Point/MultiPoint or the vertices of any other geometry type.

    Parameters
    ----------
    geom : shapely.Geometry
        Any shapely Geometry object, from which all vertices are extracted
        and plotted.
    ax : matplotlib Axes, default None
        The axes on which to draw the plot. If not specified, will get the
        current active axes or create a new figure.
    color : matplotlib color specification
        Color for the filled points. You can use `markeredgecolor` and
        `markerfacecolor` to have different edge and fill colors.
    marker : str, default "o"
        The matplotlib marker for the points.
    **kwargs
        Additional keyword arguments passed to matplotlib `plot` (Line2D).

    Returns
    -------
    Matplotlib artist (Line2D)

    """
    if ax is None:
        ax = _default_ax()

    coords = shapely.get_coordinates(geom)
    (line,) = ax.plot(
        coords[:, 0], coords[:, 1], linestyle="", marker=marker, color=color, **kwargs
    )
    return line
