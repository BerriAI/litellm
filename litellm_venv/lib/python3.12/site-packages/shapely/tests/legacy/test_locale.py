"""Test locale independence of WKT"""

import locale
import sys
import unittest

from shapely.wkt import dumps, loads

# Set locale to one that uses a comma as decimal separator
# TODO: try a few other common locales
if sys.platform == "win32":
    test_locales = {"Portuguese": "portuguese_brazil", "Italian": "italian_italy"}
else:
    test_locales = {
        "Portuguese": "pt_BR.UTF-8",
        "Italian": "it_IT.UTF-8",
    }

do_test_locale = False


def setUpModule():
    global do_test_locale
    for name in test_locales:
        try:
            test_locale = test_locales[name]
            locale.setlocale(locale.LC_ALL, test_locale)
            do_test_locale = True
            break
        except Exception:
            pass
    if not do_test_locale:
        raise unittest.SkipTest("test locale not found")


def tearDownModule():
    if sys.platform == "win32" or sys.version_info[0:2] >= (3, 11):
        locale.setlocale(locale.LC_ALL, "")
    else:
        # Deprecated since version 3.11, will be removed in version 3.13
        locale.resetlocale()


class LocaleTestCase(unittest.TestCase):
    # @unittest.skipIf(not do_test_locale, 'test locale not found')

    def test_wkt_locale(self):
        # Test reading and writing
        p = loads("POINT (0.0 0.0)")
        assert p.x == 0.0
        assert p.y == 0.0
        wkt = dumps(p)
        assert wkt.startswith("POINT")
        assert "," not in wkt
