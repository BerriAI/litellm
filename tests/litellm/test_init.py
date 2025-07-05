# tests/litellm/test_init.py

import importlib
import sys

def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "mocked_key")

    if "litellm" in sys.modules:
        del sys.modules["litellm"]
    import litellm
    importlib.reload(litellm)

    assert litellm.api_key == "mocked_key"

def test_api_base_from_env(monkeypatch):
    monkeypatch.setenv("LITELLM_API_BASE", "https://mocked-base.com")

    if "litellm" in sys.modules:
        del sys.modules["litellm"]
    import litellm
    importlib.reload(litellm)

    assert litellm.api_base == "https://mocked-base.com"
