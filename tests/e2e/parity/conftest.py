"""Parity suite plumbing. The suite is deliberately the single module
test_parity_e2e.py (client, strict shape models, and tests all live there, so
the whole parity contract reads top to bottom in one file); this conftest only
registers the covers marker, which pytest.ini's --strict-markers requires.
"""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "covers: registry cell a test covers, e.g. other.config.cost_map.remote_loaded",
    )
