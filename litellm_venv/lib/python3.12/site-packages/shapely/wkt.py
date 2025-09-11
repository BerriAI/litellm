"""Load/dump geometries using the well-known text (WKT) format.

Also provides pickle-like convenience functions.
"""

import shapely


def loads(data):
    """Load a geometry from a WKT string.

    Parameters
    ----------
    data : str
        A WKT string

    Returns
    -------
    Shapely geometry object

    """
    return shapely.from_wkt(data)


def load(fp):
    """Load a geometry from an open file.

    Parameters
    ----------
    fp :
        A file-like object which implements a `read` method.

    Returns
    -------
    Shapely geometry object

    """
    data = fp.read()
    return loads(data)


def dumps(ob, trim=False, rounding_precision=-1, **kw):
    """Dump a WKT representation of a geometry to a string.

    Parameters
    ----------
    ob :
        A geometry object of any type to be dumped to WKT.
    trim : bool, default False
        Remove excess decimals from the WKT.
    rounding_precision : int, default -1
        Round output to the specified number of digits.
        Default behavior returns full precision.
    **kw : kwargs, optional
        Keyword output options passed to :func:`~shapely.to_wkt`.

    Returns
    -------
    input geometry as WKT string

    """
    return shapely.to_wkt(ob, trim=trim, rounding_precision=rounding_precision, **kw)


def dump(ob, fp, **settings):
    """Dump a geometry to an open file.

    Parameters
    ----------
    ob :
        A geometry object of any type to be dumped to WKT.
    fp :
        A file-like object which implements a `write` method.
    **settings : kwargs, optional
        Keyword output options passed to :func:`~shapely.wkt.dumps`.

    Returns
    -------
    None

    """
    fp.write(dumps(ob, **settings))
