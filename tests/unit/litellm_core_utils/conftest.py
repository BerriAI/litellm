from importlib import resources

import pytest

import litellm
from tests.unit.litellm_core_utils.fake_secret_vault import FakeSecretVault


@pytest.fixture(autouse=True)
def bundled_tiktoken_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(resources.files(litellm).joinpath("litellm_core_utils/tokenizers")))


@pytest.fixture
def secret_vault_factory() -> type[FakeSecretVault]:
    return FakeSecretVault
