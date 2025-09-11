"""Pytest and scipy-doctest configuration for Shapely."""

import numpy
import pytest

from shapely import geos_version_string

try:
    from scipy_doctest.conftest import dt_config

    HAVE_SCPDT = True
except ModuleNotFoundError:
    HAVE_SCPDT = False

shapely20_todo = pytest.mark.xfail(
    strict=True, reason="Not yet implemented for Shapely 2.0"
)
shapely20_wontfix = pytest.mark.xfail(strict=True, reason="Will fail for Shapely 2.0")


def pytest_report_header(config):
    """Header for pytest."""
    vers = [
        f"GEOS version: {geos_version_string}",
        f"NumPy version: {numpy.__version__}",
    ]
    return "\n".join(vers)


if HAVE_SCPDT:
    import doctest
    import warnings
    from contextlib import contextmanager

    @contextmanager
    def warnings_errors_and_rng(test=None):
        """Filter out some warnings."""
        depr_msgs = "|".join(
            [
                # https://github.com/pyproj4/pyproj/issues/1468
                "Conversion of an array with ndim",
            ]
        )

        runtime_msgs = "|".join(
            [
                # https://github.com/libgeos/geos/pull/1226
                "invalid value encountered in coverage_union",
            ]
        )

        with warnings.catch_warnings():
            if depr_msgs:
                warnings.filterwarnings("ignore", depr_msgs, DeprecationWarning)
            if runtime_msgs:
                warnings.filterwarnings("ignore", runtime_msgs, RuntimeWarning)
            yield

    # find and check doctests under this context manager
    dt_config.user_context_mgr = warnings_errors_and_rng

    # relax all NumPy scalar type repr, e.g. `np.int32(0)` matches `0`
    dt_config.strict_check = False

    dt_config.optionflags = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS

    # ignores are for things fail doctest collection (optionals etc)
    dt_config.pytest_extra_ignore = [
        "shapely/geos.py",
    ]
