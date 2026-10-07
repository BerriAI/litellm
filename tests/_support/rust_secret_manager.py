import pytest

import litellm


@pytest.fixture(autouse=True)
def preserve_manager_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "secret_manager_client", litellm.secret_manager_client)
    for name in ("_key_management_system", "_key_management_settings"):
        monkeypatch.setattr(litellm, name, getattr(litellm, name))
