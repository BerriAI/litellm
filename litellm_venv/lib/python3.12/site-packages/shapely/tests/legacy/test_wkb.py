import binascii
import math
import struct
import sys

import pytest

from shapely import wkt
from shapely.geometry import Point
from shapely.tests.legacy.conftest import shapely20_todo
from shapely.wkb import dump, dumps, load, loads


@pytest.fixture(scope="module")
def some_point():
    return Point(1.2, 3.4)


def bin2hex(value):
    return binascii.b2a_hex(value).upper().decode("utf-8")


def hex2bin(value):
    return binascii.a2b_hex(value)


def hostorder(fmt, value):
    """Re-pack a hex WKB value to native endianness if needed

    This routine does not understand WKB format, so it must be provided a
    struct module format string, without initial indicator character ("@=<>!"),
    which will be interpreted as big- or little-endian with standard sizes
    depending on the endian flag in the first byte of the value.
    """

    if fmt and fmt[0] in "@=<>!":
        raise ValueError("Initial indicator character, one of @=<>!, in fmt")
    if not fmt or fmt[0] not in "cbB":
        raise ValueError("Missing endian flag in fmt")

    (hexendian,) = struct.unpack(fmt[0], hex2bin(value[:2]))
    hexorder = {0: ">", 1: "<"}[hexendian]
    sysorder = {"little": "<", "big": ">"}[sys.byteorder]
    if hexorder == sysorder:
        return value  # Nothing to do

    return bin2hex(
        struct.pack(
            sysorder + fmt,
            {">": 0, "<": 1}[sysorder],
            *struct.unpack(hexorder + fmt, hex2bin(value))[1:],
        )
    )


def test_dumps_srid(some_point):
    result = dumps(some_point)
    assert bin2hex(result) == hostorder(
        "BIdd", "0101000000333333333333F33F3333333333330B40"
    )
    result = dumps(some_point, srid=4326)
    assert bin2hex(result) == hostorder(
        "BIIdd", "0101000020E6100000333333333333F33F3333333333330B40"
    )


def test_dumps_endianness(some_point):
    result = dumps(some_point)
    assert bin2hex(result) == hostorder(
        "BIdd", "0101000000333333333333F33F3333333333330B40"
    )
    result = dumps(some_point, big_endian=False)
    assert bin2hex(result) == "0101000000333333333333F33F3333333333330B40"
    result = dumps(some_point, big_endian=True)
    assert bin2hex(result) == "00000000013FF3333333333333400B333333333333"


def test_dumps_hex(some_point):
    result = dumps(some_point, hex=True)
    assert result == hostorder("BIdd", "0101000000333333333333F33F3333333333330B40")


def test_loads_srid():
    # load a geometry which includes an srid
    geom = loads(hex2bin("0101000020E6100000333333333333F33F3333333333330B40"))
    assert isinstance(geom, Point)
    assert geom.coords[:] == [(1.2, 3.4)]
    # by default srid is not exported
    result = dumps(geom)
    assert bin2hex(result) == hostorder(
        "BIdd", "0101000000333333333333F33F3333333333330B40"
    )
    # include the srid in the output
    result = dumps(geom, include_srid=True)
    assert bin2hex(result) == hostorder(
        "BIIdd", "0101000020E6100000333333333333F33F3333333333330B40"
    )
    # replace geometry srid with another
    result = dumps(geom, srid=27700)
    assert bin2hex(result) == hostorder(
        "BIIdd", "0101000020346C0000333333333333F33F3333333333330B40"
    )


def test_loads_hex(some_point):
    assert loads(dumps(some_point, hex=True), hex=True) == some_point


def test_dump_load_binary(some_point, tmpdir):
    file = tmpdir.join("test.wkb")
    with open(file, "wb") as file_pointer:
        dump(some_point, file_pointer)
    with open(file, "rb") as file_pointer:
        restored = load(file_pointer)

    assert some_point == restored


def test_dump_load_hex(some_point, tmpdir):
    file = tmpdir.join("test.wkb")
    with open(file, "w") as file_pointer:
        dump(some_point, file_pointer, hex=True)
    with open(file) as file_pointer:
        restored = load(file_pointer, hex=True)

    assert some_point == restored


# pygeos handles both bytes and str
@shapely20_todo
def test_dump_hex_load_binary(some_point, tmpdir):
    """Asserts that reading a binary file as text (hex mode) fails."""
    file = tmpdir.join("test.wkb")
    with open(file, "w") as file_pointer:
        dump(some_point, file_pointer, hex=True)

    with pytest.raises(TypeError):
        with open(file, "rb") as file_pointer:
            load(file_pointer)


def test_dump_binary_load_hex(some_point, tmpdir):
    """Asserts that reading a text file (hex mode) as binary fails."""
    file = tmpdir.join("test.wkb")
    with open(file, "wb") as file_pointer:
        dump(some_point, file_pointer)

    # TODO(shapely-2.0) on windows this doesn't seem to error with pygeos,
    # but you get back a point with garbage coordinates
    if sys.platform == "win32":
        with open(file) as file_pointer:
            restored = load(file_pointer, hex=True)
        assert some_point != restored
        return

    with pytest.raises((UnicodeEncodeError, UnicodeDecodeError)):
        with open(file) as file_pointer:
            load(file_pointer, hex=True)


def test_point_empty():
    g = wkt.loads("POINT EMPTY")
    result = dumps(g, big_endian=False)
    # Use math.isnan for second part of the WKB representation  there are
    # many byte representations for NaN)
    assert result[: -2 * 8] == b"\x01\x01\x00\x00\x00"
    coords = struct.unpack("<2d", result[-2 * 8 :])
    assert len(coords) == 2
    assert all(math.isnan(val) for val in coords)


def test_point_z_empty():
    g = wkt.loads("POINT Z EMPTY")
    assert g.wkb_hex == hostorder(
        "BIddd", "0101000080000000000000F87F000000000000F87F000000000000F87F"
    )
