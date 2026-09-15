import pytest

_OPENCODE_ENV_VARS = (
    "OPENCODE_API_KEY",
    "OPENCODE_GO_API_KEY",
    "OPENCODE_ZEN_API_KEY",
    "OPENCODE_GO_API_BASE",
    "OPENCODE_ZEN_API_BASE",
)


@pytest.fixture(autouse=True)
def _no_ambient_opencode_env(monkeypatch):
    for name in _OPENCODE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
