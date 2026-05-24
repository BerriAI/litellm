"""
Auth-level conftest — override the heavy proxy autouse fixture for tests that
only need lightweight litellm.proxy.utils imports (e.g. password-hashing tests).
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_proxy_module_globals():
    """No-op override — these tests don't touch proxy_server module globals."""
    yield
