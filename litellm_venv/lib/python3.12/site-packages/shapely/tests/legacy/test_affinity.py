import unittest
from math import pi

import numpy as np
import pytest

from shapely import affinity
from shapely.geometry import Point
from shapely.wkt import loads as load_wkt


class AffineTestCase(unittest.TestCase):
    def test_affine_params(self):
        g = load_wkt("LINESTRING(2.4 4.1, 2.4 3, 3 3)")

        with pytest.raises(TypeError):
            affinity.affine_transform(g, None)

        with pytest.raises(ValueError):
            affinity.affine_transform(g, [1, 2, 3, 4, 5, 6, 7, 8, 9])

        with pytest.raises(AttributeError):
            affinity.affine_transform(None, [1, 2, 3, 4, 5, 6])

    def test_affine_geom_types(self):
        # identity matrices, which should result with no transformation
        matrix2d = (1, 0, 0, 1, 0, 0)
        matrix3d = (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)

        # empty in, empty out
        empty2d = load_wkt("MULTIPOLYGON EMPTY")
        assert affinity.affine_transform(empty2d, matrix2d).is_empty

        def test_geom(g2, g3=None):
            assert not g2.has_z
            a2 = affinity.affine_transform(g2, matrix2d)
            assert not a2.has_z
            assert g2.equals(a2)
            if g3 is not None:
                assert g3.has_z
                a3 = affinity.affine_transform(g3, matrix3d)
                assert a3.has_z
                assert g3.equals(a3)

        pt2d = load_wkt("POINT(12.3 45.6)")
        pt3d = load_wkt("POINT(12.3 45.6 7.89)")
        test_geom(pt2d, pt3d)
        ls2d = load_wkt("LINESTRING(0.9 3.4, 0.7 2, 2.5 2.7)")
        ls3d = load_wkt("LINESTRING(0.9 3.4 3.3, 0.7 2 2.3, 2.5 2.7 5.5)")
        test_geom(ls2d, ls3d)
        lr2d = load_wkt("LINEARRING(0.9 3.4, 0.7 2, 2.5 2.7, 0.9 3.4)")
        lr3d = load_wkt("LINEARRING(0.9 3.4 3.3, 0.7 2 2.3, 2.5 2.7 5.5, 0.9 3.4 3.3)")
        test_geom(lr2d, lr3d)
        test_geom(
            load_wkt(
                "POLYGON((0.9 2.3, 0.5 1.1, 2.4 0.8, 0.9 2.3), "
                "(1.1 1.7, 0.9 1.3, 1.4 1.2, 1.1 1.7), "
                "(1.6 1.3, 1.7 1, 1.9 1.1, 1.6 1.3))"
            )
        )
        test_geom(
            load_wkt("MULTIPOINT ((-300 300), (700 300), (-800 -1100), (200 -300))")
        )
        test_geom(
            load_wkt(
                "MULTILINESTRING((0 0, -0.7 -0.7, 0.6 -1), (-0.5 0.5, 0.7 0.6, 0 -0.6))"
            )
        )
        test_geom(
            load_wkt(
                "MULTIPOLYGON(((900 4300, -1100 -400, 900 -800, 900 4300)), "
                "((1200 4300, 2300 4400, 1900 1000, 1200 4300)))"
            )
        )
        test_geom(
            load_wkt(
                "GEOMETRYCOLLECTION(POINT(20 70),"
                " POLYGON((60 70, 13 35, 60 -30, 60 70)),"
                " LINESTRING(60 70, 50 100, 80 100))"
            )
        )

    def test_affine_2d(self):
        g = load_wkt("LINESTRING(2.4 4.1, 2.4 3, 3 3)")
        # custom scale and translate
        expected2d = load_wkt("LINESTRING(-0.2 14.35, -0.2 11.6, 1 11.6)")
        matrix2d = (2, 0, 0, 2.5, -5, 4.1)
        a2 = affinity.affine_transform(g, matrix2d)
        assert a2.equals_exact(expected2d, 1e-6)
        assert not a2.has_z
        # Make sure a 3D matrix does not make a 3D shape from a 2D input
        matrix3d = (2, 0, 0, 0, 2.5, 0, 0, 0, 10, -5, 4.1, 100)
        a3 = affinity.affine_transform(g, matrix3d)
        assert a3.equals_exact(expected2d, 1e-6)
        assert not a3.has_z

    def test_affine_3d(self):
        g2 = load_wkt("LINESTRING(2.4 4.1, 2.4 3, 3 3)")
        g3 = load_wkt("LINESTRING(2.4 4.1 100.2, 2.4 3 132.8, 3 3 128.6)")
        # custom scale and translate
        matrix2d = (2, 0, 0, 2.5, -5, 4.1)
        matrix3d = (2, 0, 0, 0, 2.5, 0, 0, 0, 0.3048, -5, 4.1, 100)
        # Combinations of 2D and 3D geometries and matrices
        a22 = affinity.affine_transform(g2, matrix2d)
        a23 = affinity.affine_transform(g2, matrix3d)
        a32 = affinity.affine_transform(g3, matrix2d)
        a33 = affinity.affine_transform(g3, matrix3d)
        # Check dimensions
        assert not a22.has_z
        assert not a23.has_z
        assert a32.has_z
        assert a33.has_z

        # 2D equality checks
        expected2d = load_wkt("LINESTRING(-0.2 14.35, -0.2 11.6, 1 11.6)")
        expected3d = load_wkt(
            "LINESTRING(-0.2 14.35 130.54096, -0.2 11.6 140.47744, 1 11.6 139.19728)"
        )
        expected32 = load_wkt(
            "LINESTRING(-0.2 14.35 100.2, -0.2 11.6 132.8, 1 11.6 128.6)"
        )
        assert a22.equals_exact(expected2d, 1e-6)
        assert a23.equals_exact(expected2d, 1e-6)
        # Do explicit 3D check of coordinate values
        for a, e in zip(a32.coords, expected32.coords):
            for ap, ep in zip(a, e):
                self.assertAlmostEqual(ap, ep)
        for a, e in zip(a33.coords, expected3d.coords):
            for ap, ep in zip(a, e):
                self.assertAlmostEqual(ap, ep)


class TransformOpsTestCase(unittest.TestCase):
    def test_rotate(self):
        ls = load_wkt("LINESTRING(240 400, 240 300, 300 300)")
        # counter-clockwise degrees
        rls = affinity.rotate(ls, 90)
        els = load_wkt("LINESTRING(220 320, 320 320, 320 380)")
        assert rls.equals(els)
        # retest with named parameters for the same result
        rls = affinity.rotate(geom=ls, angle=90, origin="center")
        assert rls.equals(els)
        # clockwise radians
        rls = affinity.rotate(ls, -pi / 2, use_radians=True)
        els = load_wkt("LINESTRING(320 380, 220 380, 220 320)")
        assert rls.equals(els)
        ## other `origin` parameters
        # around the centroid
        rls = affinity.rotate(ls, 90, origin="centroid")
        els = load_wkt("LINESTRING(182.5 320, 282.5 320, 282.5 380)")
        assert rls.equals(els)
        # around the second coordinate tuple
        rls = affinity.rotate(ls, 90, origin=ls.coords[1])
        els = load_wkt("LINESTRING(140 300, 240 300, 240 360)")
        assert rls.equals(els)
        # around the absolute Point of origin
        rls = affinity.rotate(ls, 90, origin=Point(0, 0))
        els = load_wkt("LINESTRING(-400 240, -300 240, -300 300)")
        assert rls.equals(els)

    def test_rotate_empty(self):
        rls = affinity.rotate(load_wkt("LINESTRING EMPTY"), 90)
        els = load_wkt("LINESTRING EMPTY")
        assert rls.equals(els)

    def test_rotate_angle_array(self):
        ls = load_wkt("LINESTRING(240 400, 240 300, 300 300)")
        els = load_wkt("LINESTRING(220 320, 320 320, 320 380)")
        # check with degrees
        theta = np.array(90.0)
        rls = affinity.rotate(ls, theta)
        assert theta.item() == 90.0
        assert rls.equals(els)
        # check with radians
        theta = np.array(pi / 2)
        rls = affinity.rotate(ls, theta, use_radians=True)
        assert theta.item() == pi / 2
        assert rls.equals(els)

    def test_scale(self):
        ls = load_wkt("LINESTRING(240 400 10, 240 300 30, 300 300 20)")
        # test defaults of 1.0
        sls = affinity.scale(ls)
        assert sls.equals(ls)
        # different scaling in different dimensions
        sls = affinity.scale(ls, 2, 3, 0.5)
        els = load_wkt("LINESTRING(210 500 5, 210 200 15, 330 200 10)")
        assert sls.equals(els)
        # Do explicit 3D check of coordinate values
        for a, b in zip(sls.coords, els.coords):
            for ap, bp in zip(a, b):
                self.assertEqual(ap, bp)
        # retest with named parameters for the same result
        sls = affinity.scale(geom=ls, xfact=2, yfact=3, zfact=0.5, origin="center")
        assert sls.equals(els)
        ## other `origin` parameters
        # around the centroid
        sls = affinity.scale(ls, 2, 3, 0.5, origin="centroid")
        els = load_wkt("LINESTRING(228.75 537.5, 228.75 237.5, 348.75 237.5)")
        assert sls.equals(els)
        # around the second coordinate tuple
        sls = affinity.scale(ls, 2, 3, 0.5, origin=ls.coords[1])
        els = load_wkt("LINESTRING(240 600, 240 300, 360 300)")
        assert sls.equals(els)
        # around some other 3D Point of origin
        sls = affinity.scale(ls, 2, 3, 0.5, origin=Point(100, 200, 1000))
        els = load_wkt("LINESTRING(380 800 505, 380 500 515, 500 500 510)")
        assert sls.equals(els)
        # Do explicit 3D check of coordinate values
        for a, b in zip(sls.coords, els.coords):
            for ap, bp in zip(a, b):
                assert ap == bp

    def test_scale_empty(self):
        sls = affinity.scale(load_wkt("LINESTRING EMPTY"))
        els = load_wkt("LINESTRING EMPTY")
        assert sls.equals(els)

    def test_skew(self):
        ls = load_wkt("LINESTRING(240 400 10, 240 300 30, 300 300 20)")
        # test default shear angles of 0.0
        sls = affinity.skew(ls)
        assert sls.equals(ls)
        # different shearing in x- and y-directions
        sls = affinity.skew(ls, 15, -30)
        els = load_wkt(
            "LINESTRING (253.39745962155615 417.3205080756888, "
            "226.60254037844385 317.3205080756888, "
            "286.60254037844385 282.67949192431126)"
        )
        assert sls.equals_exact(els, 1e-6)
        # retest with radians for the same result
        sls = affinity.skew(ls, pi / 12, -pi / 6, use_radians=True)
        assert sls.equals_exact(els, 1e-6)
        # retest with named parameters for the same result
        sls = affinity.skew(geom=ls, xs=15, ys=-30, origin="center", use_radians=False)
        assert sls.equals_exact(els, 1e-6)
        ## other `origin` parameters
        # around the centroid
        sls = affinity.skew(ls, 15, -30, origin="centroid")
        els = load_wkt(
            "LINESTRING(258.42150697963973 406.49519052838332, "
            "231.6265877365273980 306.4951905283833185, "
            "291.6265877365274264 271.8541743770057337)"
        )
        assert sls.equals_exact(els, 1e-6)
        # around the second coordinate tuple
        sls = affinity.skew(ls, 15, -30, origin=ls.coords[1])
        els = load_wkt(
            "LINESTRING(266.7949192431123038 400, 240 300, 300 265.3589838486224153)"
        )
        assert sls.equals_exact(els, 1e-6)
        # around the absolute Point of origin
        sls = affinity.skew(ls, 15, -30, origin=Point(0, 0))
        els = load_wkt(
            "LINESTRING(347.179676972449101 261.435935394489832, "
            "320.3847577293367976 161.4359353944898317, "
            "380.3847577293367976 126.7949192431122754)"
        )
        assert sls.equals_exact(els, 1e-6)

    def test_skew_empty(self):
        sls = affinity.skew(load_wkt("LINESTRING EMPTY"))
        els = load_wkt("LINESTRING EMPTY")
        assert sls.equals(els)

    def test_skew_xs_ys_array(self):
        ls = load_wkt("LINESTRING(240 400 10, 240 300 30, 300 300 20)")
        els = load_wkt(
            "LINESTRING (253.39745962155615 417.3205080756888, "
            "226.60254037844385 317.3205080756888, "
            "286.60254037844385 282.67949192431126)"
        )
        # check with degrees
        xs_ys = np.array([15.0, -30.0])
        sls = affinity.skew(ls, xs_ys[0, ...], xs_ys[1, ...])
        assert xs_ys[0] == 15.0
        assert xs_ys[1] == -30.0
        assert sls.equals_exact(els, 1e-6)
        # check with radians
        xs_ys = np.array([pi / 12, -pi / 6])
        sls = affinity.skew(ls, xs_ys[0, ...], xs_ys[1, ...], use_radians=True)
        assert xs_ys[0] == pi / 12
        assert xs_ys[1] == -pi / 6
        assert sls.equals_exact(els, 1e-6)

    def test_translate(self):
        ls = load_wkt("LINESTRING(240 400 10, 240 300 30, 300 300 20)")
        # test default offset of 0.0
        tls = affinity.translate(ls)
        assert tls.equals(ls)
        # test all offsets
        tls = affinity.translate(ls, 100, 400, -10)
        els = load_wkt("LINESTRING(340 800 0, 340 700 20, 400 700 10)")
        assert tls.equals(els)
        # Do explicit 3D check of coordinate values
        for a, b in zip(tls.coords, els.coords):
            for ap, bp in zip(a, b):
                assert ap == bp
        # retest with named parameters for the same result
        tls = affinity.translate(geom=ls, xoff=100, yoff=400, zoff=-10)
        assert tls.equals(els)

    def test_translate_empty(self):
        tls = affinity.translate(load_wkt("LINESTRING EMPTY"))
        els = load_wkt("LINESTRING EMPTY")
        self.assertTrue(tls.equals(els))
        assert tls.equals(els)
