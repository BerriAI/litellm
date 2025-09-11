import numpy as np

test_int_types = [int, np.int16, np.int32, np.int64]


class MultiGeometryTestCase:
    def subgeom_access_test(self, cls, geoms):
        geom = cls(geoms)
        for t in test_int_types:
            for i, g in enumerate(geoms):
                assert geom.geoms[t(i)] == g
