import pytest

import litellm


@pytest.fixture(autouse=True)
def preserve_manager_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "secret_manager_client", litellm.secret_manager_client)
    monkeypatch.setattr(litellm, "_key_management_system", litellm._key_management_system)
    monkeypatch.setattr(litellm, "_key_management_settings", litellm._key_management_settings)
