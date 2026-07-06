"""Models-management suite client fixture; lifecycle/skip/marker live in the parent conftest."""

import pytest

from models_mgmt_client import ModelsMgmtClient, build_client


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "covers: registry cell a test covers, e.g. mgmt.model.add.persists",
    )


@pytest.fixture(scope="session")
def client() -> ModelsMgmtClient:
    return build_client()
