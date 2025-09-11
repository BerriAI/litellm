from decimal import Decimal

import pytest

from shapely import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

items2d = [
    [(0.0, 0.0), (70.0, 120.0), (140.0, 0.0), (0.0, 0.0)],
    [(60.0, 80.0), (80.0, 80.0), (70.0, 60.0), (60.0, 80.0)],
]

items2d_mixed = [
    [
        (Decimal("0.0"), Decimal("0.0")),
        (Decimal("70.0"), 120.0),
        (140.0, Decimal("0.0")),
        (0.0, 0.0),
    ],
    [
        (Decimal("60.0"), Decimal("80.0")),
        (Decimal("80.0"), 80.0),
        (70.0, Decimal("60.0")),
        (60.0, 80.0),
    ],
]

items2d_decimal = [
    [
        (Decimal("0.0"), Decimal("0.0")),
        (Decimal("70.0"), Decimal("120.0")),
        (Decimal("140.0"), Decimal("0.0")),
        (Decimal("0.0"), Decimal("0.0")),
    ],
    [
        (Decimal("60.0"), Decimal("80.0")),
        (Decimal("80.0"), Decimal("80.0")),
        (Decimal("70.0"), Decimal("60.0")),
        (Decimal("60.0"), Decimal("80.0")),
    ],
]

items3d = [
    [(0.0, 0.0, 1), (70.0, 120.0, 2), (140.0, 0.0, 3), (0.0, 0.0, 1)],
    [(60.0, 80.0, 1), (80.0, 80.0, 2), (70.0, 60.0, 3), (60.0, 80.0, 1)],
]

items3d_mixed = [
    [
        (Decimal("0.0"), Decimal("0.0"), Decimal(1)),
        (Decimal("70.0"), 120.0, Decimal(2)),
        (140.0, Decimal("0.0"), 3),
        (0.0, 0.0, 1),
    ],
    [
        (Decimal("60.0"), Decimal("80.0"), Decimal(1)),
        (Decimal("80.0"), 80.0, 2),
        (70.0, Decimal("60.0"), Decimal(3)),
        (60.0, 80.0, 1),
    ],
]

items3d_decimal = [
    [
        (Decimal("0.0"), Decimal("0.0"), Decimal(1)),
        (Decimal("70.0"), Decimal("120.0"), Decimal(2)),
        (Decimal("140.0"), Decimal("0.0"), Decimal(3)),
        (Decimal("0.0"), Decimal("0.0"), Decimal(1)),
    ],
    [
        (Decimal("60.0"), Decimal("80.0"), Decimal(1)),
        (Decimal("80.0"), Decimal("80.0"), Decimal(2)),
        (Decimal("70.0"), Decimal("60.0"), Decimal(3)),
        (Decimal("60.0"), Decimal("80.0"), Decimal(1)),
    ],
]

all_geoms = [
    [
        Point(items[0][0]),
        Point(*items[0][0]),
        MultiPoint(items[0]),
        LinearRing(items[0]),
        LineString(items[0]),
        MultiLineString(items),
        Polygon(items[0]),
        MultiPolygon(
            [
                Polygon(items[1]),
                Polygon(items[0], holes=items[1:]),
            ]
        ),
        GeometryCollection([Point(items[0][0]), Polygon(items[0])]),
    ]
    for items in [
        items2d,
        items2d_mixed,
        items2d_decimal,
        items3d,
        items3d_mixed,
        items3d_decimal,
    ]
]


@pytest.mark.parametrize("geoms", list(zip(*all_geoms)))
def test_decimal(geoms):
    assert geoms[0] == geoms[1] == geoms[2]
    assert geoms[3] == geoms[4] == geoms[5]
