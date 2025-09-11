# Tests SVG output and validity
import os
import unittest
from xml.dom.minidom import parseString as parse_xml_string

from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.collection import GeometryCollection


class SvgTestCase(unittest.TestCase):
    def assertSVG(self, geom, expected, **kwrds):
        """Helper function to check XML and debug SVG"""
        svg_elem = geom.svg(**kwrds)
        try:
            parse_xml_string(svg_elem)
        except Exception:
            raise AssertionError("XML is not valid for SVG element: " + str(svg_elem))
        svg_doc = geom._repr_svg_()
        try:
            doc = parse_xml_string(svg_doc)
        except Exception:
            raise AssertionError("XML is not valid for SVG document: " + str(svg_doc))
        svg_output_dir = None
        # svg_output_dir = '.'  # useful for debugging SVG files
        if svg_output_dir:
            fname = geom.geom_type
            if geom.is_empty:
                fname += "_empty"
            if not geom.is_valid:
                fname += "_invalid"
            if kwrds:
                fname += "_" + ",".join(str(k) + "=" + str(kwrds[k]) for k in kwrds)
            svg_path = os.path.join(svg_output_dir, fname + ".svg")
            with open(svg_path, "w") as fp:
                fp.write(doc.toprettyxml())
        assert svg_elem == expected

    def test_point(self):
        # Empty
        self.assertSVG(Point(), "<g />")
        # Valid
        g = Point(6, 7)
        self.assertSVG(
            g,
            '<circle cx="6.0" cy="7.0" r="3.0" stroke="#555555" '
            'stroke-width="1.0" fill="#66cc99" opacity="0.6" />',
        )
        self.assertSVG(
            g,
            '<circle cx="6.0" cy="7.0" r="15.0" stroke="#555555" '
            'stroke-width="5.0" fill="#66cc99" opacity="0.6" />',
            scale_factor=5,
        )

    def test_multipoint(self):
        # Empty
        self.assertSVG(MultiPoint(), "<g />")
        # Valid
        g = MultiPoint([(6, 7), (3, 4)])
        self.assertSVG(
            g,
            '<g><circle cx="6.0" cy="7.0" r="3.0" stroke="#555555" '
            'stroke-width="1.0" fill="#66cc99" opacity="0.6" />'
            '<circle cx="3.0" cy="4.0" r="3.0" stroke="#555555" '
            'stroke-width="1.0" fill="#66cc99" opacity="0.6" /></g>',
        )
        self.assertSVG(
            g,
            '<g><circle cx="6.0" cy="7.0" r="15.0" stroke="#555555" '
            'stroke-width="5.0" fill="#66cc99" opacity="0.6" />'
            '<circle cx="3.0" cy="4.0" r="15.0" stroke="#555555" '
            'stroke-width="5.0" fill="#66cc99" opacity="0.6" /></g>',
            scale_factor=5,
        )

    def test_linestring(self):
        # Empty
        self.assertSVG(LineString(), "<g />")
        # Valid
        g = LineString([(5, 8), (496, -6), (530, 20)])
        self.assertSVG(
            g,
            '<polyline fill="none" stroke="#66cc99" stroke-width="2.0" '
            'points="5.0,8.0 496.0,-6.0 530.0,20.0" opacity="0.8" />',
        )
        self.assertSVG(
            g,
            '<polyline fill="none" stroke="#66cc99" stroke-width="10.0" '
            'points="5.0,8.0 496.0,-6.0 530.0,20.0" opacity="0.8" />',
            scale_factor=5,
        )
        # Invalid
        self.assertSVG(
            LineString([(0, 0), (0, 0)]),
            '<polyline fill="none" stroke="#ff3333" stroke-width="2.0" '
            'points="0.0,0.0 0.0,0.0" opacity="0.8" />',
        )

    def test_multilinestring(self):
        # Empty
        self.assertSVG(MultiLineString(), "<g />")
        # Valid
        self.assertSVG(
            MultiLineString([[(6, 7), (3, 4)], [(2, 8), (9, 1)]]),
            '<g><polyline fill="none" stroke="#66cc99" stroke-width="2.0" '
            'points="6.0,7.0 3.0,4.0" opacity="0.8" />'
            '<polyline fill="none" stroke="#66cc99" stroke-width="2.0" '
            'points="2.0,8.0 9.0,1.0" opacity="0.8" /></g>',
        )
        # Invalid
        self.assertSVG(
            MultiLineString([[(2, 3), (2, 3)], [(2, 8), (9, 1)]]),
            '<g><polyline fill="none" stroke="#ff3333" stroke-width="2.0" '
            'points="2.0,3.0 2.0,3.0" opacity="0.8" />'
            '<polyline fill="none" stroke="#ff3333" stroke-width="2.0" '
            'points="2.0,8.0 9.0,1.0" opacity="0.8" /></g>',
        )

    def test_polygon(self):
        # Empty
        self.assertSVG(Polygon(), "<g />")
        # Valid
        g = Polygon(
            [(35, 10), (45, 45), (15, 40), (10, 20), (35, 10)],
            [[(20, 30), (35, 35), (30, 20), (20, 30)]],
        )
        self.assertSVG(
            g,
            '<path fill-rule="evenodd" fill="#66cc99" stroke="#555555" '
            'stroke-width="2.0" opacity="0.6" d="M 35.0,10.0 L 45.0,45.0 L '
            "15.0,40.0 L 10.0,20.0 L 35.0,10.0 z M 20.0,30.0 L 35.0,35.0 L "
            '30.0,20.0 L 20.0,30.0 z" />',
        )
        self.assertSVG(
            g,
            '<path fill-rule="evenodd" fill="#66cc99" stroke="#555555" '
            'stroke-width="10.0" opacity="0.6" d="M 35.0,10.0 L 45.0,45.0 L '
            "15.0,40.0 L 10.0,20.0 L 35.0,10.0 z M 20.0,30.0 L 35.0,35.0 L "
            '30.0,20.0 L 20.0,30.0 z" />',
            scale_factor=5,
        )
        # Invalid
        self.assertSVG(
            Polygon([(0, 40), (0, 0), (40, 40), (40, 0), (0, 40)]),
            '<path fill-rule="evenodd" fill="#ff3333" stroke="#555555" '
            'stroke-width="2.0" opacity="0.6" d="M 0.0,40.0 L 0.0,0.0 L '
            '40.0,40.0 L 40.0,0.0 L 0.0,40.0 z" />',
        )

    def test_multipolygon(self):
        # Empty
        self.assertSVG(MultiPolygon(), "<g />")
        # Valid
        self.assertSVG(
            MultiPolygon(
                [
                    Polygon([(40, 40), (20, 45), (45, 30), (40, 40)]),
                    Polygon(
                        [(20, 35), (10, 30), (10, 10), (30, 5), (45, 20), (20, 35)],
                        [[(30, 20), (20, 15), (20, 25), (30, 20)]],
                    ),
                ]
            ),
            '<g><path fill-rule="evenodd" fill="#66cc99" stroke="#555555" '
            'stroke-width="2.0" opacity="0.6" d="M 40.0,40.0 L 20.0,45.0 L '
            '45.0,30.0 L 40.0,40.0 z" />'
            '<path fill-rule="evenodd" fill="#66cc99" stroke="#555555" '
            'stroke-width="2.0" opacity="0.6" d="M 20.0,35.0 L 10.0,30.0 L '
            "10.0,10.0 L 30.0,5.0 L 45.0,20.0 L 20.0,35.0 z M 30.0,20.0 L "
            '20.0,15.0 L 20.0,25.0 L 30.0,20.0 z" /></g>',
        )
        # Invalid
        self.assertSVG(
            MultiPolygon(
                [
                    Polygon([(140, 140), (120, 145), (145, 130), (140, 140)]),
                    Polygon([(0, 40), (0, 0), (40, 40), (40, 0), (0, 40)]),
                ]
            ),
            '<g><path fill-rule="evenodd" fill="#ff3333" stroke="#555555" '
            'stroke-width="2.0" opacity="0.6" d="M 140.0,140.0 L '
            '120.0,145.0 L 145.0,130.0 L 140.0,140.0 z" />'
            '<path fill-rule="evenodd" fill="#ff3333" stroke="#555555" '
            'stroke-width="2.0" opacity="0.6" d="M 0.0,40.0 L 0.0,0.0 L '
            '40.0,40.0 L 40.0,0.0 L 0.0,40.0 z" /></g>',
        )

    def test_collection(self):
        # Empty
        self.assertSVG(GeometryCollection(), "<g />")
        # Valid
        self.assertSVG(
            GeometryCollection([Point(7, 3), LineString([(4, 2), (8, 4)])]),
            '<g><circle cx="7.0" cy="3.0" r="3.0" stroke="#555555" '
            'stroke-width="1.0" fill="#66cc99" opacity="0.6" />'
            '<polyline fill="none" stroke="#66cc99" stroke-width="2.0" '
            'points="4.0,2.0 8.0,4.0" opacity="0.8" /></g>',
        )
        # Invalid
        self.assertSVG(
            Point(7, 3).union(LineString([(4, 2), (4, 2)])),
            '<g><circle cx="7.0" cy="3.0" r="3.0" stroke="#555555" '
            'stroke-width="1.0" fill="#ff3333" opacity="0.6" />'
            '<polyline fill="none" stroke="#ff3333" stroke-width="2.0" '
            'points="4.0,2.0 4.0,2.0" opacity="0.8" /></g>',
        )
