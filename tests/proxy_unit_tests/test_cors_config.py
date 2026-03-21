import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CORS_ENV_VARS = (
    "LITELLM_CORS_ALLOW_ORIGINS",
    "LITELLM_CORS_ALLOW_CREDENTIALS",
    "LITELLM_CORS_ALLOW_METHODS",
    "LITELLM_CORS_ALLOW_HEADERS",
)
CORS_MODULES = ("litellm.proxy.proxy_server",)


@pytest.fixture(autouse=True)
def clear_cors_env(monkeypatch):
    for env_var in CORS_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


def _reload_local_proxy_server():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    for module_name in CORS_MODULES:
        sys.modules.pop(module_name, None)

    import litellm.proxy.proxy_server as proxy_server

    return importlib.reload(proxy_server)


def test_normalize_cors_value_list():
    from litellm.proxy.proxy_cli import _normalize_cors_value

    assert _normalize_cors_value(
        ["https://a.com", " https://b.com "], "cors_allow_origins"
    ) == "https://a.com,https://b.com"


def test_normalize_cors_value_empty_list():
    from litellm.proxy.proxy_cli import _normalize_cors_value

    assert _normalize_cors_value([], "cors_allow_origins") == ""


def test_normalize_cors_value_invalid_type_raises():
    from litellm.proxy.proxy_cli import _normalize_cors_value

    with pytest.raises(ValueError, match="expected a string or list of strings"):
        _normalize_cors_value(123, "cors_allow_origins")


def test_cors_defaults_preserve_existing_proxy_behavior():
    proxy_server = _reload_local_proxy_server()

    assert proxy_server.cors_allow_origins == ["*"]
    assert proxy_server.cors_allow_credentials is True
    assert proxy_server.cors_allow_methods == ["*"]
    assert proxy_server.cors_allow_headers == ["*"]


def test_cors_credentials_disabled_with_wildcard(monkeypatch):
    monkeypatch.setenv("LITELLM_CORS_ALLOW_ORIGINS", "*")
    monkeypatch.setenv("LITELLM_CORS_ALLOW_CREDENTIALS", "true")

    proxy_server = _reload_local_proxy_server()

    assert proxy_server.cors_allow_credentials is False
    assert "*" in proxy_server.cors_allow_origins


def test_cors_credentials_disabled_with_mixed_wildcard_origins(monkeypatch):
    monkeypatch.setenv(
        "LITELLM_CORS_ALLOW_ORIGINS", "*, https://dashboard.example.com"
    )
    monkeypatch.setenv("LITELLM_CORS_ALLOW_CREDENTIALS", "true")

    proxy_server = _reload_local_proxy_server()

    assert proxy_server.cors_allow_credentials is False
    assert proxy_server.cors_allow_origins == ["*", "https://dashboard.example.com"]


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


def test_cors_credentials_default_to_disabled_with_explicit_origins(monkeypatch):
    monkeypatch.setenv("LITELLM_CORS_ALLOW_ORIGINS", "https://example.com")

    proxy_server = _reload_local_proxy_server()

    assert proxy_server.cors_allow_credentials is False
    assert proxy_server.cors_allow_origins == ["https://example.com"]


def test_cors_credentials_only_config_is_rejected_with_wildcard_default(monkeypatch):
    monkeypatch.setenv("LITELLM_CORS_ALLOW_CREDENTIALS", "true")

    proxy_server = _reload_local_proxy_server()

    assert proxy_server.cors_allow_credentials is False
    assert proxy_server.cors_allow_origins == ["*"]


def test_explicit_empty_origins_stay_empty(monkeypatch):
    monkeypatch.setenv("LITELLM_CORS_ALLOW_ORIGINS", "")

    proxy_server = _reload_local_proxy_server()

    assert proxy_server.cors_allow_origins == []


def test_methods_and_headers_are_trimmed(monkeypatch):
    monkeypatch.setenv("LITELLM_CORS_ALLOW_METHODS", " GET , POST , OPTIONS ")
    monkeypatch.setenv(
        "LITELLM_CORS_ALLOW_HEADERS", " Authorization , Content-Type "
    )

    proxy_server = _reload_local_proxy_server()

    assert proxy_server.cors_allow_methods == ["GET", "POST", "OPTIONS"]
    assert proxy_server.cors_allow_headers == ["Authorization", "Content-Type"]


@pytest.mark.parametrize(
    "credential_value,expected",
    [
        ("yes", True),
        ("1", True),
        ("True", True),
        ("FALSE", False),
    ],
)
def test_cors_credentials_boolean_formats(monkeypatch, credential_value, expected):
    monkeypatch.setenv("LITELLM_CORS_ALLOW_ORIGINS", "https://example.com")
    monkeypatch.setenv("LITELLM_CORS_ALLOW_CREDENTIALS", credential_value)

    proxy_server = _reload_local_proxy_server()

    assert proxy_server.cors_allow_credentials is expected
