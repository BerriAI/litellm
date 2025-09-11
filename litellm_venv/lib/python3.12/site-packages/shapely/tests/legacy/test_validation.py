import unittest

from shapely.geometry import Point
from shapely.validation import explain_validity


class ValidationTestCase(unittest.TestCase):
    def test_valid(self):
        assert explain_validity(Point(0, 0)) == "Valid Geometry"
