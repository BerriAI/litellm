"""Management suite client fixture; lifecycle/skip/marker live in the parent conftest."""

import pytest

from management_client import ManagementClient, build_client


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "covers: registry cell a test covers, e.g. mgmt.key.generate.persists",
    )


@pytest.fixture(scope="session")
def client() -> ManagementClient:
    return build_client()
