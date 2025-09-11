import pytest

from shapely import Point, Polygon, geos_version


def test_format_invalid():
    # check invalid spec formats
    pt = Point(1, 2)
    test_list = [
        ("5G", ValueError, "invalid format specifier"),
        (".f", ValueError, "invalid format specifier"),
        ("0.2e", ValueError, "invalid format specifier"),
        (".1x", ValueError, "hex representation does not specify precision"),
    ]
    for format_spec, err, match in test_list:
        with pytest.raises(err, match=match):
            format(pt, format_spec)


def get_tst_format_point_params():
    xy1 = (0.12345678901234567, 1.2345678901234567e10)
    xy2 = (-169.910918, -18.997564)
    xyz3 = (630084, 4833438, 76)
    test_list = [
        (".0f", xy1, "POINT (0 12345678901)", True),
        (".1f", xy1, "POINT (0.1 12345678901.2)", True),
        ("0.2f", xy2, "POINT (-169.91 -19.00)", True),
        (".3F", (float("inf"), -float("inf")), "POINT (INF -INF)", True),
    ]
    if geos_version < (3, 10, 0):
        # 'g' format varies depending on GEOS version
        test_list += [
            (".1g", xy1, "POINT (0.1 1e+10)", True),
            (".6G", xy1, "POINT (0.123457 1.23457E+10)", True),
            ("0.12g", xy1, "POINT (0.123456789012 12345678901.2)", True),
            ("0.4g", xy2, "POINT (-169.9 -19)", True),
        ]
    else:
        test_list += [
            (".1g", xy1, "POINT (0.1 12345678901.2)", False),
            (".6G", xy1, "POINT (0.123457 12345678901.234568)", False),
            ("0.12g", xy1, "POINT (0.123456789012 12345678901.234568)", False),
            ("g", xy2, "POINT (-169.910918 -18.997564)", False),
            ("0.2g", xy2, "POINT (-169.91 -19)", False),
        ]
    # without precisions test GEOS rounding_precision=-1; different than Python
    test_list += [
        ("f", (1, 2), f"POINT ({1:.16f} {2:.16f})", False),
        ("F", xyz3, "POINT Z ({:.16f} {:.16f} {:.16f})".format(*xyz3), False),
        ("g", xyz3, "POINT Z (630084 4833438 76)", False),
    ]
    return test_list


@pytest.mark.parametrize(
    "format_spec, coords, expt_wkt, same_python_float", get_tst_format_point_params()
)
def test_format_point(format_spec, coords, expt_wkt, same_python_float):
    pt = Point(*coords)
    # basic checks
    assert f"{pt}" == pt.wkt
    assert format(pt, "") == pt.wkt
    assert format(pt, "x") == pt.wkb_hex.lower()
    assert format(pt, "X") == pt.wkb_hex
    # check formatted WKT to expected
    assert format(pt, format_spec) == expt_wkt, format_spec
    # check Python's format consistency
    text_coords = expt_wkt[expt_wkt.index("(") + 1 : expt_wkt.index(")")]
    is_same = []
    for coord, expt_coord in zip(coords, text_coords.split()):
        py_fmt_float = format(float(coord), format_spec)
        if same_python_float:
            assert py_fmt_float == expt_coord, format_spec
        else:
            is_same.append(py_fmt_float == expt_coord)
    if not same_python_float:
        assert not all(is_same), f"{format_spec!r} with {expt_wkt}"


def test_format_polygon():
    # check basic cases
    poly = Point(0, 0).buffer(10, quad_segs=2)
    assert f"{poly}" == poly.wkt
    assert format(poly, "") == poly.wkt
    assert format(poly, "x") == poly.wkb_hex.lower()
    assert format(poly, "X") == poly.wkb_hex

    # Use f-strings with extra characters and rounding precision
    if geos_version < (3, 13, 0):
        assert f"<{poly:.2f}>" == (
            "<POLYGON ((10.00 0.00, 7.07 -7.07, 0.00 -10.00, -7.07 -7.07, "
            "-10.00 -0.00, -7.07 7.07, -0.00 10.00, 7.07 7.07, 10.00 0.00))>"
        )
    else:
        assert f"<{poly:.2f}>" == (
            "<POLYGON ((10.00 0.00, 7.07 -7.07, 0.00 -10.00, -7.07 -7.07, "
            "-10.00 0.00, -7.07 7.07, 0.00 10.00, 7.07 7.07, 10.00 0.00))>"
        )

    # 'g' format varies depending on GEOS version
    if geos_version < (3, 10, 0):
        assert f"{poly:.2G}" == (
            "POLYGON ((10 0, 7.1 -7.1, 1.6E-14 -10, -7.1 -7.1, "
            "-10 -3.2E-14, -7.1 7.1, -4.6E-14 10, 7.1 7.1, 10 0))"
        )
    else:
        assert f"{poly:.2G}" == (
            "POLYGON ((10 0, 7.07 -7.07, 0 -10, -7.07 -7.07, "
            "-10 0, -7.07 7.07, 0 10, 7.07 7.07, 10 0))"
        )

    # check empty
    empty = Polygon()
    assert f"{empty}" == "POLYGON EMPTY"
    assert format(empty, "") == empty.wkt
    assert format(empty, ".2G") == empty.wkt
    assert format(empty, "x") == empty.wkb_hex.lower()
    assert format(empty, "X") == empty.wkb_hex
