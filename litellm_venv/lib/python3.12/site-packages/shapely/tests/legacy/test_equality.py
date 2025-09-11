from shapely import Point, Polygon


def test_equals_exact():
    p1 = Point(1.0, 1.0)
    p2 = Point(2.0, 2.0)
    p3 = Point(1.0, 1.0 + 1e-7)
    assert not p1.equals(p2)
    assert not p1.equals_exact(p2, 0.001)
    assert not p1.equals_exact(p3)
    assert p1.equals_exact(p3, 1e-6)

    # test polygons
    shell = [(10, 10), (10, -10), (-10, -10), (-10, 10)]
    holes = [[(1, 1), (1, -1), (-1, -1), (-1, 1)]]
    p1 = Polygon(shell, holes)
    p2 = Polygon(shell, holes=[holes[0][::-1]])
    assert p1.equals(p2)
    assert not p1.equals_exact(p2, 1e-5)
    assert p1.equals_exact(p2, 1e-5, normalize=True)

    hole2 = [(1, 1), (1, -1), (-1, -1), (-1, 1.01)]
    p3 = Polygon(shell, holes=[hole2])
    assert not p1.equals_exact(p3, 1e-5)
