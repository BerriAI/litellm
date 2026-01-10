import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _reload_local_proxy_server():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    for module_name in list(sys.modules):
        if module_name == "litellm" or module_name.startswith("litellm."):
            del sys.modules[module_name]

    import litellm.proxy.proxy_server as proxy_server

    return importlib.reload(proxy_server)


def test_cors_credentials_disabled_with_wildcard(monkeypatch):
    monkeypatch.setenv("LITELLM_CORS_ALLOW_ORIGINS", "*")
    monkeypatch.setenv("LITELLM_CORS_ALLOW_CREDENTIALS", "true")

    proxy_server = _reload_local_proxy_server()

    assert proxy_server.cors_allow_credentials is False
    assert "*" in proxy_server.cors_allow_origins


def test_cors_credentials_enabled_with_explicit_origins(monkeypatch):
    monkeypatch.setenv(
        "LITELLM_CORS_ALLOW_ORIGINS", "https://example.com, https://other.com"
    )
    monkeypatch.setenv("LITELLM_CORS_ALLOW_CREDENTIALS", "true")

    proxy_server = _reload_local_proxy_server()

    assert proxy_server.cors_allow_credentials is True
    assert proxy_server.cors_allow_origins == [
        "https://example.com",
        "https://other.com",
    ]
