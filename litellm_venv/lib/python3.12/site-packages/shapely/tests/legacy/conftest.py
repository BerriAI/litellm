import numpy
import pytest

shapely20_todo = pytest.mark.xfail(
    strict=True, reason="Not yet implemented for Shapely 2.0"
)
shapely20_wontfix = pytest.mark.xfail(strict=True, reason="Will fail for Shapely 2.0")


def pytest_report_header(config):
    """Header for pytest."""
    return f"dependencies: numpy-{numpy.__version__}"
