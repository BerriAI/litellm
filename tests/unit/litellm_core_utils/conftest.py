import pytest

from tests.unit.litellm_core_utils.fake_secret_vault import FakeSecretVault


@pytest.fixture
def secret_vault_factory() -> type[FakeSecretVault]:
    return FakeSecretVault
