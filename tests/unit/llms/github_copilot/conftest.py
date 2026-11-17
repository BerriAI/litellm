import pytest


@pytest.fixture(autouse=True)
def clear_copilot_api_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_API_VERSION", raising=False)
