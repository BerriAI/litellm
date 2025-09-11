"""Coordinate sequence utilities."""

from array import array


class CoordinateSequence:
    """Access to coordinate tuples from the parent geometry's coordinate sequence.

    Examples
    --------
    >>> from shapely.wkt import loads
    >>> g = loads('POINT (0.0 0.0)')
    >>> list(g.coords)
    [(0.0, 0.0)]
    >>> g = loads('POINT M (1 2 4)')
    >>> g.coords[:]
    [(1.0, 2.0, 4.0)]

    """

    def __init__(self, coords):
        """Initialize the CoordinateSequence.

        Parameters
        ----------
        coords : array
            The coordinate array.

        """
        self._coords = coords

    def __len__(self):
        """Return the length of the CoordinateSequence.

        Returns
        -------
        int
            The length of the CoordinateSequence.

        """
        return self._coords.shape[0]

    def __iter__(self):
        """Iterate over the CoordinateSequence."""
        for i in range(self.__len__()):
            yield tuple(self._coords[i].tolist())

    def __getitem__(self, key):
        """Get the item at the specified index or slice.

        Parameters
        ----------
        key : int or slice
            The index or slice.

        Returns
        -------
        tuple or list
            The item at the specified index or slice.

        """
        m = self.__len__()
        if isinstance(key, int):
            if key + m < 0 or key >= m:
                raise IndexError("index out of range")
            if key < 0:
                i = m + key
            else:
                i = key
            return tuple(self._coords[i].tolist())
        elif isinstance(key, slice):
            res = []
            start, stop, stride = key.indices(m)
            for i in range(start, stop, stride):
                res.append(tuple(self._coords[i].tolist()))
            return res
        else:
            raise TypeError("key must be an index or slice")

    def __array__(self, dtype=None, copy=None):
        """Return a copy of the coordinate array.

        Parameters
        ----------
        dtype : data-type, optional
            The desired data-type for the array.
        copy : bool, optional
            If None (default) or True, a copy of the array is always returned.
            If False, a ValueError is raised as this is not supported.

        Returns
        -------
        array
            The coordinate array.

        Raises
        ------
        ValueError
            If `copy=False` is specified.

        """
        if copy is False:
            raise ValueError("`copy=False` isn't supported. A copy is always created.")
        elif copy is True:
            return self._coords.copy()
        else:
            return self._coords

    @property
    def xy(self):
        """X and Y arrays."""
        m = self.__len__()
        x = array("d")
        y = array("d")
        for i in range(m):
            xy = self._coords[i].tolist()
            x.append(xy[0])
            y.append(xy[1])
        return x, y
