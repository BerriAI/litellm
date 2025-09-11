"""Load/dump geometries using the well-known binary (WKB) format.

Also provides pickle-like convenience functions.
"""

import shapely


def loads(data, hex=False):
    """Load a geometry from a WKB byte string.

    If ``hex=True``, the string will be hex-encoded.

    Raises
    ------
    GEOSException, UnicodeDecodeError
        If ``data`` contains an invalid geometry.

    """
    return shapely.from_wkb(data)


def load(fp, hex=False):
    """Load a geometry from an open file.

    Raises
    ------
    GEOSException, UnicodeDecodeError
        If the given file contains an invalid geometry.

    """
    data = fp.read()
    return loads(data, hex=hex)


def dumps(ob, hex=False, srid=None, **kw):
    """Dump a WKB representation of a geometry to a byte string.

    If ``hex=True``, the string will be hex-encoded.

    Parameters
    ----------
    ob : geometry
        The geometry to export to well-known binary (WKB) representation.
    hex : bool
        If true, export the WKB as a hexadecimal string. The default is to
        return a binary string/bytes object.
    srid : int
        Spatial reference system ID to include in the output. The default value
        means no SRID is included.
    **kw : kwargs, optional
        Keyword output options passed to :func:`~shapely.to_wkb`.

    """
    if srid is not None:
        # clone the object and set the SRID before dumping
        ob = shapely.set_srid(ob, srid)
        kw["include_srid"] = True
    if "big_endian" in kw:
        # translate big_endian=True/False into byte_order=0/1
        # but if not specified, keep the default of byte_order=-1 (native)
        big_endian = kw.pop("big_endian")
        byte_order = 0 if big_endian else 1
        kw.update(byte_order=byte_order)
    return shapely.to_wkb(ob, hex=hex, **kw)


def dump(ob, fp, hex=False, **kw):
    """Dump a geometry to an open file."""
    fp.write(dumps(ob, hex=hex, **kw))
