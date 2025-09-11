"""STRtree spatial index for efficient spatial queries."""

from collections.abc import Iterable
from typing import Any

import numpy as np

from shapely import lib
from shapely._enum import ParamEnum
from shapely.decorators import UnsupportedGEOSVersionError
from shapely.geometry.base import BaseGeometry
from shapely.predicates import is_empty, is_missing

__all__ = ["STRtree"]


class BinaryPredicate(ParamEnum):
    """The enumeration of GEOS binary predicates types."""

    intersects = 1
    within = 2
    contains = 3
    overlaps = 4
    crosses = 5
    touches = 6
    covers = 7
    covered_by = 8
    contains_properly = 9


class STRtree:
    """A query-only R-tree spatial index.

    It is created using the Sort-Tile-Recursive (STR) [1]_ algorithm.

    The tree indexes the bounding boxes of each geometry.  The tree is
    constructed directly at initialization and nodes cannot be added or
    removed after it has been created.

    All operations return indices of the input geometries.  These indices
    can be used to index into anything associated with the input geometries,
    including the input geometries themselves, or custom items stored in
    another object of the same length as the geometries.

    Bounding boxes limited to two dimensions and are axis-aligned (equivalent to
    the ``bounds`` property of a geometry); any Z values present in geometries
    are ignored for purposes of indexing within the tree.

    Any mixture of geometry types may be stored in the tree.

    Note: the tree is more efficient for querying when there are fewer
    geometries that have overlapping bounding boxes and where there is greater
    similarity between the outer boundary of a geometry and its bounding box.
    For example, a MultiPolygon composed of widely-spaced individual Polygons
    will have a large overall bounding box compared to the boundaries of its
    individual Polygons, and the bounding box may also potentially overlap many
    other geometries within the tree.  This means that the resulting tree may be
    less efficient to query than a tree constructed from individual Polygons.

    Parameters
    ----------
    geoms : sequence
        A sequence of geometry objects.
    node_capacity : int, default 10
        The maximum number of child nodes per parent node in the tree.

    References
    ----------
    .. [1] Leutenegger, Scott T.; Edgington, Jeffrey M.; Lopez, Mario A.
       (February 1997). "STR: A Simple and Efficient Algorithm for
       R-Tree Packing".
       https://ia600900.us.archive.org/27/items/nasa_techdoc_19970016975/19970016975.pdf

    """

    def __init__(self, geoms: Iterable[BaseGeometry], node_capacity: int = 10):
        """Create a new STRtree spatial index."""
        # Keep references to geoms in a copied array so that this array is not
        # modified while the tree depends on it remaining the same
        self._geometries = np.array(geoms, dtype=np.object_, copy=True)

        # initialize GEOS STRtree
        self._tree = lib.STRtree(self.geometries, node_capacity)

    def __len__(self):
        """Return the number of geometries in the tree."""
        return self._tree.count

    def __reduce__(self):
        """Pickle support."""
        return (STRtree, (self.geometries,))

    @property
    def geometries(self):
        """Geometries stored in the tree in the order used to construct the tree.

        The order of this array corresponds to the tree indices returned by
        other STRtree methods.

        Do not attempt to modify items in the returned array.

        Returns
        -------
        ndarray of Geometry objects

        """
        return self._geometries

    def query(self, geometry, predicate=None, distance=None):
        """Get the index combinations of all possibly intersecting geometries.

        Returns the integer indices of all combinations of each input geometry
        and tree geometries where the bounding box of each input geometry
        intersects the bounding box of a tree geometry.

        If the input geometry is a scalar, this returns an array of shape (n, ) with
        the indices of the matching tree geometries.  If the input geometry is an
        array_like, this returns an array with shape (2,n) where the subarrays
        correspond to the indices of the input geometries and indices of the
        tree geometries associated with each.  To generate an array of pairs of
        input geometry index and tree geometry index, simply transpose the
        result.

        If a predicate is provided, the tree geometries are first queried based
        on the bounding box of the input geometry and then are further filtered
        to those that meet the predicate when comparing the input geometry to
        the tree geometry:
        predicate(geometry, tree_geometry)

        The 'dwithin' predicate requires GEOS >= 3.10.

        Bounding boxes are limited to two dimensions and are axis-aligned
        (equivalent to the ``bounds`` property of a geometry); any Z values
        present in input geometries are ignored when querying the tree.

        Any input geometry that is None or empty will never match geometries in
        the tree.

        Parameters
        ----------
        geometry : Geometry or array_like
            Input geometries to query the tree and filter results using the
            optional predicate.
        predicate : {None, 'intersects', 'within', 'contains', 'overlaps', 'crosses',\
'touches', 'covers', 'covered_by', 'contains_properly', 'dwithin'}, optional
            The predicate to use for testing geometries from the tree
            that are within the input geometry's bounding box.
        distance : number or array_like, optional
            Distances around each input geometry within which to query the tree
            for the 'dwithin' predicate.  If array_like, shape must be
            broadcastable to shape of geometry.  Required if predicate='dwithin'.

        Returns
        -------
        ndarray with shape (n,) if geometry is a scalar
            Contains tree geometry indices.

        OR

        ndarray with shape (2, n) if geometry is an array_like
            The first subarray contains input geometry indices.
            The second subarray contains tree geometry indices.

        Examples
        --------
        >>> from shapely import box, Point, STRtree
        >>> import numpy as np
        >>> points = [Point(0, 0), Point(1, 1), Point(2,2), Point(3, 3)]
        >>> tree = STRtree(points)

        Query the tree using a scalar geometry:

        >>> indices = tree.query(box(0, 0, 1, 1))
        >>> indices.tolist()
        [0, 1]

        Query using an array of geometries:

        >>> boxes = np.array([box(0, 0, 1, 1), box(2, 2, 3, 3)])
        >>> arr_indices = tree.query(boxes)
        >>> arr_indices.tolist()
        [[0, 0, 1, 1], [0, 1, 2, 3]]

        Or transpose to get all pairs of input and tree indices:

        >>> arr_indices.T.tolist()
        [[0, 0], [0, 1], [1, 2], [1, 3]]

        Retrieve the tree geometries by results of query:

        >>> tree.geometries.take(indices).tolist()
        [<POINT (0 0)>, <POINT (1 1)>]

        Retrieve all pairs of input and tree geometries:

        >>> np.array([boxes.take(arr_indices[0]),\
tree.geometries.take(arr_indices[1])]).T.tolist()
        [[<POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))>, <POINT (0 0)>],
         [<POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))>, <POINT (1 1)>],
         [<POLYGON ((3 2, 3 3, 2 3, 2 2, 3 2))>, <POINT (2 2)>],
         [<POLYGON ((3 2, 3 3, 2 3, 2 2, 3 2))>, <POINT (3 3)>]]

        Query using a predicate:

        >>> tree = STRtree([box(0, 0, 0.5, 0.5), box(0.5, 0.5, 1, 1), box(1, 1, 2, 2)])
        >>> tree.query(box(0, 0, 1, 1), predicate="contains").tolist()
        [0, 1]
        >>> tree.query(Point(0.75, 0.75), predicate="dwithin", distance=0.5).tolist()
        [0, 1, 2]

        >>> tree.query(boxes, predicate="contains").tolist()
        [[0, 0], [0, 1]]
        >>> tree.query(boxes, predicate="dwithin", distance=0.5).tolist()
        [[0, 0, 0, 1], [0, 1, 2, 2]]

        Retrieve custom items associated with tree geometries (records can
        be in whatever data structure so long as geometries and custom data
        can be extracted into arrays of the same length and order):

        >>> records = [
        ...     {"geometry": Point(0, 0), "value": "A"},
        ...     {"geometry": Point(2, 2), "value": "B"}
        ... ]
        >>> tree = STRtree([record["geometry"] for record in records])
        >>> items = np.array([record["value"] for record in records])
        >>> items.take(tree.query(box(0, 0, 1, 1))).tolist()
        ['A']


        Notes
        -----
        In the context of a spatial join, input geometries are the "left"
        geometries that determine the order of the results, and tree geometries
        are "right" geometries that are joined against the left geometries. This
        effectively performs an inner join, where only those combinations of
        geometries that can be joined based on overlapping bounding boxes or
        optional predicate are returned.

        """
        geometry = np.asarray(geometry)
        is_scalar = False
        if geometry.ndim == 0:
            geometry = np.expand_dims(geometry, 0)
            is_scalar = True

        if predicate is None:
            indices = self._tree.query(geometry, 0)
            return indices[1] if is_scalar else indices

        # Requires GEOS >= 3.10
        elif predicate == "dwithin":
            if lib.geos_version < (3, 10, 0):
                raise UnsupportedGEOSVersionError(
                    "dwithin predicate requires GEOS >= 3.10"
                )
            if distance is None:
                raise ValueError(
                    "distance parameter must be provided for dwithin predicate"
                )
            distance = np.asarray(distance, dtype="float64")
            if distance.ndim > 1:
                raise ValueError("Distance array should be one dimensional")

            try:
                distance = np.broadcast_to(distance, geometry.shape)
            except ValueError:
                raise ValueError("Could not broadcast distance to match geometry")

            indices = self._tree.dwithin(geometry, distance)
            return indices[1] if is_scalar else indices

        predicate = BinaryPredicate.get_value(predicate)
        indices = self._tree.query(geometry, predicate)
        return indices[1] if is_scalar else indices

    def nearest(self, geometry) -> Any | None:
        """Return the index of the nearest geometry in the tree.

        This is determined for each input geometry based on distance within
        two-dimensional Cartesian space.

        This distance will be 0 when input geometries intersect tree geometries.

        If there are multiple equidistant or intersected geometries in the tree,
        only a single result is returned for each input geometry, based on the
        order that tree geometries are visited; this order may be
        nondeterministic.

        If any input geometry is None or empty, an error is raised.  Any Z
        values present in input geometries are ignored when finding nearest
        tree geometries.

        Parameters
        ----------
        geometry : Geometry or array_like
            Input geometries to query the tree.

        Returns
        -------
        scalar or ndarray
            Indices of geometries in tree. Return value will have the same shape
            as the input.

            None is returned if this index is empty. This may change in
            version 2.0.

        See Also
        --------
        query_nearest: returns all equidistant geometries, exclusive geometries, \
and optional distances

        Examples
        --------
        >>> from shapely import Point, STRtree
        >>> tree = STRtree([Point(i, i) for i in range(10)])

        Query the tree for nearest using a scalar geometry:

        >>> index = tree.nearest(Point(2.2, 2.2))
        >>> index
        2
        >>> tree.geometries.take(index)
        <POINT (2 2)>

        Query the tree for nearest using an array of geometries:

        >>> indices = tree.nearest([Point(2.2, 2.2), Point(4.4, 4.4)])
        >>> indices.tolist()
        [2, 4]
        >>> tree.geometries.take(indices).tolist()
        [<POINT (2 2)>, <POINT (4 4)>]

        Nearest only return one object if there are multiple equidistant results:

        >>> tree = STRtree ([Point(0, 0), Point(0, 0)])
        >>> tree.nearest(Point(0, 0))
        0

        """
        if self._tree.count == 0:
            return None

        geometry_arr = np.asarray(geometry, dtype=object)
        if is_missing(geometry_arr).any() or is_empty(geometry_arr).any():
            raise ValueError(
                "Cannot determine nearest geometry for empty geometry or "
                "missing value (None)."
            )
        # _tree.nearest returns ndarray with shape (2, 1) -> index in input
        # geometries and index into tree geometries
        indices = self._tree.nearest(np.atleast_1d(geometry_arr))[1]

        if geometry_arr.ndim == 0:
            return indices[0]
        else:
            return indices

    def query_nearest(
        self,
        geometry,
        max_distance=None,
        return_distance=False,
        exclusive=False,
        all_matches=True,
    ):
        """Return the index of the nearest geometries in the tree.

        This is determined for each input geometry based on distance within
        two-dimensional Cartesian space.

        This distance will be 0 when input geometries intersect tree geometries.

        If there are multiple equidistant or intersected geometries in tree and
        `all_matches` is True (the default), all matching tree geometries are
        returned; otherwise only the first matching tree geometry is returned.
        Tree indices are returned in the order they are visited for each input
        geometry and may not be in ascending index order; no meaningful order is
        implied.

        The max_distance used to search for nearest items in the tree may have a
        significant impact on performance by reducing the number of input
        geometries that are evaluated for nearest items in the tree.  Only those
        input geometries with at least one tree geometry within +/- max_distance
        beyond their envelope will be evaluated.  However, using a large
        max_distance may have a negative performance impact because many tree
        geometries will be queried for each input geometry.

        The distance, if returned, will be 0 for any intersected geometries in
        the tree.

        Any geometry that is None or empty in the input geometries is omitted
        from the output.  Any Z values present in input geometries are ignored
        when finding nearest tree geometries.

        Parameters
        ----------
        geometry : Geometry or array_like
            Input geometries to query the tree.
        max_distance : float, optional
            Maximum distance within which to query for nearest items in tree.
            Must be greater than 0.
        return_distance : bool, default False
            If True, will return distances in addition to indices.
        exclusive : bool, default False
            If True, the nearest tree geometries that are equal to the input
            geometry will not be returned.
        all_matches : bool, default True
            If True, all equidistant and intersected geometries will be returned
            for each input geometry.
            If False, only the first nearest geometry will be returned.

        Returns
        -------
        tree indices or tuple of (tree indices, distances) if geometry is a scalar
            indices is an ndarray of shape (n, ) and distances (if present) an
            ndarray of shape (n, )

        OR

        indices or tuple of (indices, distances)
            indices is an ndarray of shape (2,n) and distances (if present) an
            ndarray of shape (n).
            The first subarray of indices contains input geometry indices.
            The second subarray of indices contains tree geometry indices.

        See Also
        --------
        nearest: returns singular nearest geometry for each input

        Examples
        --------
        >>> import numpy as np
        >>> from shapely import box, Point, STRtree
        >>> points = [Point(0, 0), Point(1, 1), Point(2,2), Point(3, 3)]
        >>> tree = STRtree(points)

        Find the nearest tree geometries to a scalar geometry:

        >>> indices = tree.query_nearest(Point(0.25, 0.25))
        >>> indices.tolist()
        [0]

        Retrieve the tree geometries by results of query:

        >>> tree.geometries.take(indices).tolist()
        [<POINT (0 0)>]

        Find the nearest tree geometries to an array of geometries:

        >>> query_points = np.array([Point(2.25, 2.25), Point(1, 1)])
        >>> arr_indices = tree.query_nearest(query_points)
        >>> arr_indices.tolist()
        [[0, 1], [2, 1]]

        Or transpose to get all pairs of input and tree indices:

        >>> arr_indices.T.tolist()
        [[0, 2], [1, 1]]

        Retrieve all pairs of input and tree geometries:

        >>> list(zip(query_points.take(arr_indices[0]), tree.geometries.take(arr_indices[1])))
        [(<POINT (2.25 2.25)>, <POINT (2 2)>), (<POINT (1 1)>, <POINT (1 1)>)]

        All intersecting geometries in the tree are returned by default:

        >>> tree.query_nearest(box(1,1,3,3)).tolist()
        [1, 2, 3]

        Set all_matches to False to to return a single match per input geometry:

        >>> tree.query_nearest(box(1,1,3,3), all_matches=False).tolist()
        [1]

        Return the distance to each nearest tree geometry:

        >>> index, distance = tree.query_nearest(Point(0.5, 0.5), return_distance=True)
        >>> index.tolist()
        [0, 1]
        >>> distance.round(4).tolist()
        [0.7071, 0.7071]

        Return the distance for each input and nearest tree geometry for an array
        of geometries:

        >>> indices, distance = tree.query_nearest([Point(0.5, 0.5), Point(1, 1)], return_distance=True)
        >>> indices.tolist()
        [[0, 0, 1], [0, 1, 1]]
        >>> distance.round(4).tolist()
        [0.7071, 0.7071, 0.0]

        Retrieve custom items associated with tree geometries (records can
        be in whatever data structure so long as geometries and custom data
        can be extracted into arrays of the same length and order):

        >>> records = [
        ...     {"geometry": Point(0, 0), "value": "A"},
        ...     {"geometry": Point(2, 2), "value": "B"}
        ... ]
        >>> tree = STRtree([record["geometry"] for record in records])
        >>> items = np.array([record["value"] for record in records])
        >>> items.take(tree.query_nearest(Point(0.5, 0.5))).tolist()
        ['A']

        """  # noqa: E501
        geometry = np.asarray(geometry, dtype=object)
        is_scalar = False
        if geometry.ndim == 0:
            geometry = np.expand_dims(geometry, 0)
            is_scalar = True

        if max_distance is not None:
            if not np.isscalar(max_distance):
                raise ValueError("max_distance parameter only accepts scalar values")

            if max_distance <= 0:
                raise ValueError("max_distance must be greater than 0")

        # a distance of 0 means no max_distance is used
        max_distance = max_distance or 0

        if not np.isscalar(exclusive):
            raise ValueError("exclusive parameter only accepts scalar values")

        if exclusive not in {True, False}:
            raise ValueError("exclusive parameter must be boolean")

        if not np.isscalar(all_matches):
            raise ValueError("all_matches parameter only accepts scalar values")

        if all_matches not in {True, False}:
            raise ValueError("all_matches parameter must be boolean")

        results = self._tree.query_nearest(
            geometry, max_distance, exclusive, all_matches
        )

        # output indices are shape (n, )
        if is_scalar:
            if not return_distance:
                return results[0][1]

            else:
                return (results[0][1], results[1])

        # output indices are shape (2, n)
        if not return_distance:
            return results[0]

        return results
