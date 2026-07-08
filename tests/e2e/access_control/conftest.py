"""Access-control suite client fixture; lifecycle/skip/marker live in the parent conftest."""

import pytest

from access_control_client import AccessControlClient, build_client


@pytest.fixture(scope="session")
def client() -> AccessControlClient:
    return build_client()
