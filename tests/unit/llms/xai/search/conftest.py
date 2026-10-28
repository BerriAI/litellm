import pytest


@pytest.fixture(autouse=True)
def _search_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "true")
    for name in ("XAI_API_KEY", "XAI_API_BASE"):
        monkeypatch.delenv(name, raising=False)
