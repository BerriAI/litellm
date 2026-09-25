import importlib

import pytest

from tests.unit.litellm_core_utils.fake_secret_vault import FakeSecretVault


@pytest.fixture(autouse=True, scope="session")
def bundled_tiktoken_cache() -> None:
    importlib.import_module("litellm.litellm_core_utils.default_encoding")


@pytest.fixture
def secret_vault_factory() -> type[FakeSecretVault]:
    return FakeSecretVault
