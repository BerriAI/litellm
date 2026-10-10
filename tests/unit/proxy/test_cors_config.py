"""
Tests for CORS configuration security fix.

All tests import _get_cors_config directly from proxy_server so they exercise
real production code rather than a local mirror.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy import proxy_server


def test_cors_wildcard_disables_credentials():
    """should disable credentials when LITELLM_CORS_ORIGINS is not set (defaults to wildcard)."""
    from litellm.proxy.proxy_server import _get_cors_config

    origins, allow_credentials = _get_cors_config(cors_origins_env="")
    assert origins == ["*"]
    assert allow_credentials is False


def test_cors_empty_string_disables_credentials():
    """should disable credentials when LITELLM_CORS_ORIGINS is empty or whitespace."""
    from litellm.proxy.proxy_server import _get_cors_config

    for empty in ("", "   ", "\t"):
        origins, allow_credentials = _get_cors_config(cors_origins_env=empty)
        assert origins == ["*"], f"Expected wildcard for input {repr(empty)}"
        assert allow_credentials is False, f"Expected no credentials for input {repr(empty)}"


def test_cors_single_specific_origin_enables_credentials():
    """should enable credentials when a single explicit origin is configured."""
    from litellm.proxy.proxy_server import _get_cors_config

    origins, allow_credentials = _get_cors_config(cors_origins_env="https://admin.example.com")
    assert origins == ["https://admin.example.com"]
    assert allow_credentials is True


def test_cors_multiple_specific_origins_enables_credentials():
    """should enable credentials and correctly parse comma-separated origins."""
    from litellm.proxy.proxy_server import _get_cors_config

    origins, allow_credentials = _get_cors_config(
        cors_origins_env="https://app.example.com, https://admin.example.com, https://api.example.com"
    )
    assert origins == [
        "https://app.example.com",
        "https://admin.example.com",
        "https://api.example.com",
    ]
    assert allow_credentials is True


def test_cors_wildcard_string_in_env_disables_credentials():
    """should disable credentials when LITELLM_CORS_ORIGINS is explicitly set to '*'."""
    from litellm.proxy.proxy_server import _get_cors_config

    origins, allow_credentials = _get_cors_config(cors_origins_env="*")
    assert "*" in origins
    assert allow_credentials is False


def test_cors_origins_strips_whitespace():
    """should strip surrounding whitespace from each origin entry."""
    from litellm.proxy.proxy_server import _get_cors_config

    origins, _ = _get_cors_config(cors_origins_env="  https://a.com  ,  https://b.com  ")
    assert origins == ["https://a.com", "https://b.com"]


def test_cors_origins_skips_blank_entries():
    """should skip blank entries caused by trailing/double commas."""
    from litellm.proxy.proxy_server import _get_cors_config

    origins, allow_credentials = _get_cors_config(cors_origins_env="https://a.com,,https://b.com,")
    assert origins == ["https://a.com", "https://b.com"]
    assert allow_credentials is True


def test_cors_explicit_credentials_true_overrides_wildcard():
    """should enable credentials when LITELLM_CORS_ALLOW_CREDENTIALS=true even
    if wildcard origins are in use (opt-in for existing deployments)."""
    from litellm.proxy.proxy_server import _get_cors_config

    origins, allow_credentials = _get_cors_config(
        cors_origins_env="",
        cors_credentials_env="true",
    )
    assert "*" in origins
    assert allow_credentials is True


def test_cors_explicit_credentials_false_overrides_specific_origins():
    """should disable credentials when LITELLM_CORS_ALLOW_CREDENTIALS=false even
    if specific origins are configured."""
    from litellm.proxy.proxy_server import _get_cors_config

    origins, allow_credentials = _get_cors_config(
        cors_origins_env="https://admin.example.com",
        cors_credentials_env="false",
    )
    assert origins == ["https://admin.example.com"]
    assert allow_credentials is False


def test_cors_explicit_credentials_case_insensitive():
    """should accept TRUE/FALSE case-insensitively for LITELLM_CORS_ALLOW_CREDENTIALS."""
    from litellm.proxy.proxy_server import _get_cors_config

    _, allow_true = _get_cors_config(cors_origins_env="", cors_credentials_env="TRUE")
    _, allow_false = _get_cors_config(cors_origins_env="https://x.com", cors_credentials_env="FALSE")
    assert allow_true is True
    assert allow_false is False


def test_proxy_server_cors_invariant():
    """should verify that proxy_server module-level origins and allow_cors_credentials
    are consistent — catches any future drift in the module-level call to _get_cors_config.
    """
    import os

    import litellm.proxy.proxy_server as proxy_server

    if os.getenv("LITELLM_CORS_ALLOW_CREDENTIALS") is None:
        assert proxy_server.allow_cors_credentials == ("*" not in proxy_server.origins), (
            f"Invariant broken: allow_cors_credentials={proxy_server.allow_cors_credentials} "
            f"but origins={proxy_server.origins}. "
            "When origins contains '*', allow_credentials must be False."
        )


@pytest.fixture
def cors_client(monkeypatch):
    for key in ("LITELLM_CORS_ORIGINS", "LITELLM_CORS_ALLOW_CREDENTIALS"):
        monkeypatch.delenv(key, raising=False)
    settings = {
        "cors_allow_origins": ["http://localhost:3000"],
        "cors_allow_methods": ["POST"],
        "cors_allow_headers": ["Authorization", "Content-Type"],
        "cors_allow_credentials": False,
    }
    monkeypatch.setattr(proxy_server, "general_settings", settings)
    middleware = next(entry for entry in proxy_server.app.user_middleware if "expose_headers" in entry.kwargs)
    app = FastAPI()

    @app.get("/ping")
    def ping():
        return {"ok": True}

    app.add_middleware(middleware.cls, **middleware.kwargs)
    with TestClient(app) as client:
        yield client, settings


def preflight(client, origin="http://localhost:3000", method="POST", headers="authorization,content-type"):
    return client.options(
        "/ping",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": method,
            "Access-Control-Request-Headers": headers,
        },
    )


def test_loaded_yaml_allowlists_are_applied(cors_client):
    client, _ = cors_client
    response = preflight(client)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert response.headers["access-control-allow-methods"] == "POST"
    assert "authorization" in response.headers["access-control-allow-headers"].lower()
    assert "access-control-allow-credentials" not in response.headers
    assert response.headers["vary"] == "Origin"


@pytest.mark.parametrize(
    "arguments",
    [
        {"origin": "https://untrusted.example"},
        {"method": "GET"},
        {"headers": "x-unlisted"},
    ],
)
def test_unlisted_preflight_is_rejected(cors_client, arguments):
    client, _ = cors_client
    assert preflight(client, **arguments).status_code == 400


def test_simple_response_retains_exposed_headers(cors_client):
    client, _ = cors_client
    response = client.get("/ping", headers={"Origin": "http://localhost:3000"})
    assert response.json() == {"ok": True}
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "x-litellm-cache-key" in response.headers["access-control-expose-headers"]


def test_request_without_origin_is_unchanged(cors_client):
    client, _ = cors_client
    response = client.get("/ping")
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_configuration_rebinding_is_observed(cors_client, monkeypatch):
    client, _ = cors_client
    assert preflight(client).status_code == 200
    monkeypatch.setattr(
        proxy_server,
        "general_settings",
        {
            "cors_allow_origins": ["https://replacement.example"],
            "cors_allow_methods": ["PATCH"],
            "cors_allow_headers": ["x-new-header"],
            "cors_allow_credentials": True,
        },
    )
    assert preflight(client).status_code == 400
    response = preflight(client, origin="https://replacement.example", method="PATCH", headers="x-new-header")
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://replacement.example"
    assert response.headers["access-control-allow-methods"] == "PATCH"
    assert response.headers["access-control-allow-credentials"] == "true"


def test_environment_origins_and_credentials_take_precedence(cors_client, monkeypatch):
    client, settings = cors_client
    settings["cors_allow_credentials"] = True
    monkeypatch.setenv("LITELLM_CORS_ORIGINS", "https://env.example")
    monkeypatch.setenv("LITELLM_CORS_ALLOW_CREDENTIALS", "false")
    response = preflight(client, origin="https://env.example")
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://env.example"
    assert response.headers["access-control-allow-methods"] == "POST"
    assert "access-control-allow-credentials" not in response.headers
    assert preflight(client).status_code == 400


def test_in_place_origin_list_changes_invalidate_cached_policy(cors_client):
    client, settings = cors_client
    assert preflight(client).status_code == 200
    settings["cors_allow_origins"].append("https://added.example")
    assert preflight(client, origin="https://added.example").status_code == 200
    settings["cors_allow_origins"].remove("http://localhost:3000")
    assert preflight(client).status_code == 400


def test_unconfigured_default_remains_wildcard_without_credentials(cors_client):
    client, settings = cors_client
    settings.clear()
    response = preflight(client, origin="https://any.example", method="DELETE", headers="x-anything")
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in response.headers


def test_implicit_credentials_follow_effective_origin_policy(cors_client):
    client, settings = cors_client
    del settings["cors_allow_credentials"]
    response = preflight(client)
    assert response.headers["access-control-allow-credentials"] == "true"
    settings["cors_allow_origins"] = ["*"]
    response = preflight(client)
    assert response.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in response.headers


@pytest.mark.parametrize("origins", [None, ["*"], ["https://trusted.example", "*"]])
def test_yaml_credentials_cannot_enable_wildcard_policy(cors_client, origins):
    client, settings = cors_client
    settings["cors_allow_credentials"] = True
    if origins is None:
        del settings["cors_allow_origins"]
    else:
        settings["cors_allow_origins"] = origins
    response = preflight(client, origin="https://untrusted.example")
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in response.headers
    response = client.get("/ping", headers={"Origin": "https://untrusted.example", "Cookie": "session=fixture"})
    assert response.status_code == 200
    assert "access-control-allow-credentials" not in response.headers


@pytest.mark.parametrize("origins_env", ["", "   ", "\t", "*"])
def test_empty_or_wildcard_environment_disables_yaml_credentials(cors_client, monkeypatch, origins_env):
    client, settings = cors_client
    settings["cors_allow_credentials"] = True
    monkeypatch.setenv("LITELLM_CORS_ORIGINS", origins_env)
    response = preflight(client, origin="https://untrusted.example")
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in response.headers
    response = client.get("/ping", headers={"Origin": "https://untrusted.example", "Cookie": "session=fixture"})
    assert "access-control-allow-credentials" not in response.headers


def test_explicit_yaml_origins_allow_credentials_only_for_listed_origin(cors_client):
    client, settings = cors_client
    settings["cors_allow_credentials"] = True
    response = preflight(client)
    assert response.status_code == 200
    assert response.headers["access-control-allow-credentials"] == "true"
    response = preflight(client, origin="https://untrusted.example")
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers
    response = client.get("/ping", headers={"Origin": "https://untrusted.example", "Cookie": "session=fixture"})
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize("credentials_env", ["true", "false"])
def test_explicit_environment_credentials_override_yaml_wildcard_guard(cors_client, monkeypatch, credentials_env):
    client, settings = cors_client
    settings["cors_allow_origins"] = ["*"]
    settings["cors_allow_credentials"] = credentials_env == "false"
    monkeypatch.setenv("LITELLM_CORS_ALLOW_CREDENTIALS", credentials_env)
    response = preflight(client, origin="https://untrusted.example")
    assert response.status_code == 200
    if credentials_env == "true":
        assert response.headers["access-control-allow-origin"] == "https://untrusted.example"
        assert response.headers["access-control-allow-credentials"] == "true"
    else:
        assert response.headers["access-control-allow-origin"] == "*"
        assert "access-control-allow-credentials" not in response.headers
