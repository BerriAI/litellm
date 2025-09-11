from math import pi

import pytest

from shapely.geometry import Point
from shapely.wkt import dump, dumps, load, loads


@pytest.fixture(scope="module")
def some_point():
    return Point(pi, -pi)


@pytest.fixture(scope="module")
def empty_geometry():
    return Point()


def test_wkt(some_point):
    """.wkt and wkt.dumps() both do not trim by default."""
    assert some_point.wkt == f"POINT ({pi:.15f} {-pi:.15f})"


def test_wkt_null(empty_geometry):
    assert empty_geometry.wkt == "POINT EMPTY"


def test_dump_load(some_point, tmpdir):
    file = tmpdir.join("test.wkt")
    with open(file, "w") as file_pointer:
        dump(some_point, file_pointer)
    with open(file) as file_pointer:
        restored = load(file_pointer)

    assert some_point == restored


def test_dump_load_null_geometry(empty_geometry, tmpdir):
    file = tmpdir.join("test.wkt")
    with open(file, "w") as file_pointer:
        dump(empty_geometry, file_pointer)
    with open(file) as file_pointer:
        restored = load(file_pointer)

    # This is does not work with __eq__():
    assert empty_geometry.equals(restored)


def test_dumps_loads(some_point):
    assert dumps(some_point) == f"POINT ({pi:.16f} {-pi:.16f})"
    assert loads(dumps(some_point)) == some_point


def test_dumps_loads_null_geometry(empty_geometry):
    assert dumps(empty_geometry) == "POINT EMPTY"
    # This is does not work with __eq__():
    assert loads(dumps(empty_geometry)).equals(empty_geometry)


def test_dumps_precision(some_point):
    assert dumps(some_point, rounding_precision=4) == f"POINT ({pi:.4f} {-pi:.4f})"
