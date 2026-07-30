import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy.discovery_endpoints.ui_discovery_endpoints import router
from litellm.types.proxy.control_plane_endpoints import WorkerRegistryEntry


def get_discovery_response(settings):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with (
        patch("litellm.proxy.utils.get_server_root_path", return_value="/"),
        patch("litellm.proxy.utils.get_proxy_base_url", return_value=None),
        patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=False),
        patch("litellm.proxy.proxy_server.general_settings", settings),
    ):
        return client.get("/.well-known/litellm-ui-config")


def native_oidc_settings(**overrides):
    jwtauth = {
        "native_oidc_issuer": "https://idp.example.com",
        "native_oidc_client_id": "litellm-native",
        "native_oidc_scopes": ["openid"],
    }
    jwtauth.update(overrides)
    return {"enable_jwt_auth": True, "litellm_jwtauth": jwtauth}


def test_ui_discovery_endpoints_exposes_valid_native_oidc_config():
    settings = {
        "enable_jwt_auth": True,
        "litellm_jwtauth": {
            "native_oidc_issuer": "https://idp.example.com",
            "native_oidc_client_id": "litellm-native",
            "native_oidc_scopes": ["openid", "profile", "offline_access"],
            # Secrets living alongside the public fields must never be published.
            "client_secret": "must-not-appear",
            "public_key_ttl_seconds": 600,
        },
    }

    response = get_discovery_response(settings)

    assert response.status_code == 200
    assert response.json()["native_oidc"] == {
        "issuer": "https://idp.example.com",
        "client_id": "litellm-native",
        "scopes": ["openid", "profile", "offline_access"],
    }
    assert "must-not-appear" not in response.text


def test_ui_discovery_endpoints_returns_the_issuer_byte_for_byte():
    """The issuer is a trust anchor compared by exact string equality.

    A trailing slash, an explicit port, or mixed case must survive untouched --
    the CLI checks this value against the provider document verbatim.
    """
    issuer = "https://IdP.Example.com:8443/tenant/"
    response = get_discovery_response(native_oidc_settings(native_oidc_issuer=issuer))

    assert response.json()["native_oidc"]["issuer"] == issuer


@pytest.mark.parametrize(
    "settings",
    [
        # JWT auth off, or set to something that is not exactly True.
        {"enable_jwt_auth": False, "litellm_jwtauth": {}},
        {"enable_jwt_auth": "true", "litellm_jwtauth": {}},
        {"enable_jwt_auth": None, "litellm_jwtauth": {}},
        {"enable_jwt_auth": 1, "litellm_jwtauth": {}},
        # Nothing configured at all.
        {"enable_jwt_auth": True, "litellm_jwtauth": {}},
        {"enable_jwt_auth": True, "litellm_jwtauth": "invalid"},
        {"enable_jwt_auth": True, "litellm_jwtauth": None},
        # Partially configured: every field is required.
        native_oidc_settings(native_oidc_issuer=None),
        native_oidc_settings(native_oidc_client_id=None),
        native_oidc_settings(native_oidc_scopes=None),
        # Issuer failures.
        native_oidc_settings(native_oidc_issuer=""),
        native_oidc_settings(native_oidc_issuer="   "),
        native_oidc_settings(native_oidc_issuer="idp.example.com"),
        native_oidc_settings(native_oidc_issuer="ftp://idp.example.com"),
        native_oidc_settings(native_oidc_issuer="javascript:alert(1)"),
        # Plaintext HTTP is only allowed against a numeric loopback address.
        native_oidc_settings(native_oidc_issuer="http://idp.example.com"),
        native_oidc_settings(native_oidc_issuer="http://localhost:8080"),
        # Credentials, query, fragment and malformed ports.
        native_oidc_settings(native_oidc_issuer="https://user:pass@idp.example.com"),
        native_oidc_settings(native_oidc_issuer="https://@idp.example.com"),
        native_oidc_settings(native_oidc_issuer="https://idp.example.com:not-a-port"),
        native_oidc_settings(native_oidc_issuer="https://idp.example.com:65536"),
        native_oidc_settings(native_oidc_issuer="https://idp.example.com?foo=bar"),
        native_oidc_settings(native_oidc_issuer="https://idp.example.com#fragment"),
        native_oidc_settings(native_oidc_issuer="https://idp.example.com/a b"),
        native_oidc_settings(native_oidc_issuer="https://idp.example.com\x00"),
        native_oidc_settings(native_oidc_issuer="https://idp.example.com\n"),
        # Client id failures.
        native_oidc_settings(native_oidc_client_id=""),
        native_oidc_settings(native_oidc_client_id="   "),
        native_oidc_settings(native_oidc_client_id="client\nid"),
        native_oidc_settings(native_oidc_client_id="client id"),
        native_oidc_settings(native_oidc_client_id=" litellm-native "),
        native_oidc_settings(native_oidc_client_id="client\x00id"),
        # Scope failures: RFC 6749 scope-token excludes space, quote and backslash.
        native_oidc_settings(native_oidc_scopes=[]),
        native_oidc_settings(native_oidc_scopes=[""]),
        native_oidc_settings(native_oidc_scopes=["open\nid"]),
        native_oidc_settings(native_oidc_scopes=["open id"]),
        native_oidc_settings(native_oidc_scopes=['open"id']),
        native_oidc_settings(native_oidc_scopes=["open\\id"]),
        native_oidc_settings(native_oidc_scopes=["openid", "openid"]),
        native_oidc_settings(native_oidc_scopes="openid"),
    ],
)
def test_ui_discovery_endpoints_omits_invalid_native_oidc_config(settings):
    response = get_discovery_response(settings)

    assert response.status_code == 200
    assert "native_oidc" not in response.json()
    # Only native_oidc is dropped -- other optional fields still serialize as null.
    assert response.json()["proxy_base_url"] is None


def test_native_oidc_config_forbids_unknown_fields():
    """A future field must not ride along into the public document by accident."""
    from pydantic import ValidationError

    from litellm.types.proxy.discovery_endpoints.ui_discovery_endpoints import (
        NativeOIDCConfig,
    )

    with pytest.raises(ValidationError):
        NativeOIDCConfig(
            issuer="https://idp.example.com",
            client_id="litellm-native",
            scopes=("openid",),
            client_secret="must-not-appear",
        )


def test_ui_discovery_endpoints_never_echoes_rejected_values():
    """The route is unauthenticated, so a rejection must not leak the input."""
    response = get_discovery_response(
        native_oidc_settings(
            native_oidc_issuer="https://user:hunter2@idp.internal.example.com"
        )
    )

    assert response.status_code == 200
    assert "hunter2" not in response.text
    assert "idp.internal.example.com" not in response.text


@pytest.mark.parametrize("issuer", ["http://127.0.0.1:8080", "http://[::1]:8080"])
def test_ui_discovery_endpoints_allows_loopback_http_native_oidc_config(issuer):
    """Plaintext HTTP is permitted for local development, loopback literals only."""
    response = get_discovery_response(native_oidc_settings(native_oidc_issuer=issuer))

    assert response.status_code == 200
    assert response.json()["native_oidc"]["issuer"] == issuer


def test_ui_discovery_endpoints_with_defaults():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with (
        patch("litellm.proxy.utils.get_server_root_path", return_value="/"),
        patch("litellm.proxy.utils.get_proxy_base_url", return_value=None),
        patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=False),
        patch.dict(os.environ, {"DISABLE_ADMIN_UI": "false"}, clear=False),
    ):
        response = client.get("/.well-known/litellm-ui-config")

        assert response.status_code == 200
        data = response.json()
        assert data["server_root_path"] == "/"
        assert data["proxy_base_url"] is None
        assert data["auto_redirect_to_sso"] is False
        assert data["admin_ui_disabled"] is False
        assert data["sso_configured"] is False


def test_ui_discovery_endpoints_with_custom_server_root_path():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with (
        patch("litellm.proxy.utils.get_server_root_path", return_value="/litellm"),
        patch("litellm.proxy.utils.get_proxy_base_url", return_value=None),
        patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=False),
        patch.dict(os.environ, {"DISABLE_ADMIN_UI": "false"}, clear=False),
    ):
        response = client.get("/.well-known/litellm-ui-config")

        assert response.status_code == 200
        data = response.json()
        assert data["server_root_path"] == "/litellm"
        assert data["proxy_base_url"] is None
        assert data["auto_redirect_to_sso"] is False
        assert data["sso_configured"] is False


def test_ui_discovery_endpoints_with_proxy_base_url_when_set():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with (
        patch("litellm.proxy.utils.get_server_root_path", return_value="/"),
        patch(
            "litellm.proxy.utils.get_proxy_base_url",
            return_value="https://proxy.example.com",
        ),
        patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=False),
        patch.dict(os.environ, {"DISABLE_ADMIN_UI": "false"}, clear=False),
    ):
        response = client.get("/litellm/.well-known/litellm-ui-config")

        assert response.status_code == 200
        data = response.json()
        assert data["server_root_path"] == "/"
        assert data["proxy_base_url"] == "https://proxy.example.com"
        assert data["auto_redirect_to_sso"] is False
        assert data["sso_configured"] is False


def test_ui_discovery_endpoints_with_sso_configured_and_auto_redirect_enabled():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with (
        patch("litellm.proxy.utils.get_server_root_path", return_value="/litellm"),
        patch(
            "litellm.proxy.utils.get_proxy_base_url",
            return_value="https://proxy.example.com",
        ),
        patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=True),
        patch.dict(
            os.environ,
            {"AUTO_REDIRECT_UI_LOGIN_TO_SSO": "true", "DISABLE_ADMIN_UI": "false"},
            clear=False,
        ),
    ):
        response = client.get("/.well-known/litellm-ui-config")

        assert response.status_code == 200
        data = response.json()
        assert data["server_root_path"] == "/litellm"
        assert data["proxy_base_url"] == "https://proxy.example.com"
        assert data["auto_redirect_to_sso"] is True
        assert data["sso_configured"] is True


def test_ui_discovery_endpoints_with_sso_configured_and_auto_redirect_not_set_defaults_to_false():
    """When SSO is configured but AUTO_REDIRECT_UI_LOGIN_TO_SSO is not set, defaults to False."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with (
        patch("litellm.proxy.utils.get_server_root_path", return_value="/litellm"),
        patch(
            "litellm.proxy.utils.get_proxy_base_url",
            return_value="https://proxy.example.com",
        ),
        patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=True),
        patch.dict(os.environ, {"DISABLE_ADMIN_UI": "false"}, clear=False),
    ):
        # Ensure AUTO_REDIRECT_UI_LOGIN_TO_SSO is not set (simulate default)
        os.environ.pop("AUTO_REDIRECT_UI_LOGIN_TO_SSO", None)

        response = client.get("/.well-known/litellm-ui-config")

        assert response.status_code == 200
        data = response.json()
        assert data["server_root_path"] == "/litellm"
        assert data["proxy_base_url"] == "https://proxy.example.com"
        assert data["auto_redirect_to_sso"] is False
        assert data["sso_configured"] is True


def test_ui_discovery_endpoints_with_sso_configured_but_auto_redirect_disabled():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with (
        patch("litellm.proxy.utils.get_server_root_path", return_value="/litellm"),
        patch(
            "litellm.proxy.utils.get_proxy_base_url",
            return_value="https://proxy.example.com",
        ),
        patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=True),
        patch.dict(
            os.environ,
            {"AUTO_REDIRECT_UI_LOGIN_TO_SSO": "false", "DISABLE_ADMIN_UI": "false"},
            clear=False,
        ),
    ):
        response = client.get("/.well-known/litellm-ui-config")

        assert response.status_code == 200
        data = response.json()
        assert data["server_root_path"] == "/litellm"
        assert data["proxy_base_url"] == "https://proxy.example.com"
        assert data["auto_redirect_to_sso"] is False
        assert data["sso_configured"] is True


def test_ui_discovery_endpoints_with_sso_not_configured_but_auto_redirect_enabled():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with (
        patch("litellm.proxy.utils.get_server_root_path", return_value="/"),
        patch("litellm.proxy.utils.get_proxy_base_url", return_value=None),
        patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=False),
        patch.dict(
            os.environ,
            {"AUTO_REDIRECT_UI_LOGIN_TO_SSO": "true", "DISABLE_ADMIN_UI": "false"},
            clear=False,
        ),
    ):
        response = client.get("/.well-known/litellm-ui-config")

        assert response.status_code == 200
        data = response.json()
        assert data["server_root_path"] == "/"
        assert data["proxy_base_url"] is None
        assert data["auto_redirect_to_sso"] is False
        assert data["sso_configured"] is False


def test_ui_discovery_endpoints_both_routes_return_same_data():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with (
        patch("litellm.proxy.utils.get_server_root_path", return_value="/litellm"),
        patch(
            "litellm.proxy.utils.get_proxy_base_url",
            return_value="https://proxy.example.com",
        ),
        patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=True),
        patch.dict(
            os.environ,
            {"AUTO_REDIRECT_UI_LOGIN_TO_SSO": "true", "DISABLE_ADMIN_UI": "false"},
            clear=False,
        ),
    ):
        response1 = client.get("/.well-known/litellm-ui-config")
        response2 = client.get("/litellm/.well-known/litellm-ui-config")

        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response1.json() == response2.json()


def test_ui_discovery_endpoints_with_auto_redirect_via_general_settings():
    """When auto_redirect_ui_login_to_sso is set in general_settings (config.yaml), it should be honored."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with (
        patch("litellm.proxy.utils.get_server_root_path", return_value="/"),
        patch("litellm.proxy.utils.get_proxy_base_url", return_value=None),
        patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=True),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"auto_redirect_ui_login_to_sso": True},
        ),
        patch.dict(os.environ, {"DISABLE_ADMIN_UI": "false"}, clear=False),
    ):
        os.environ.pop("AUTO_REDIRECT_UI_LOGIN_TO_SSO", None)

        response = client.get("/.well-known/litellm-ui-config")

        assert response.status_code == 200
        data = response.json()
        assert data["auto_redirect_to_sso"] is True
        assert data["sso_configured"] is True


def test_ui_discovery_endpoints_with_auto_redirect_env_var_overrides_general_settings():
    """Env var and general_settings should both work — either being true enables the feature."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with (
        patch("litellm.proxy.utils.get_server_root_path", return_value="/"),
        patch("litellm.proxy.utils.get_proxy_base_url", return_value=None),
        patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=True),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"auto_redirect_ui_login_to_sso": False},
        ),
        patch.dict(
            os.environ,
            {"AUTO_REDIRECT_UI_LOGIN_TO_SSO": "true", "DISABLE_ADMIN_UI": "false"},
            clear=False,
        ),
    ):
        response = client.get("/.well-known/litellm-ui-config")

        assert response.status_code == 200
        data = response.json()
        assert data["auto_redirect_to_sso"] is True


def test_ui_discovery_endpoints_with_admin_ui_disabled():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with (
        patch("litellm.proxy.utils.get_server_root_path", return_value="/"),
        patch("litellm.proxy.utils.get_proxy_base_url", return_value=None),
        patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=False),
        patch.dict(os.environ, {"DISABLE_ADMIN_UI": "true"}, clear=False),
    ):
        response = client.get("/.well-known/litellm-ui-config")

        assert response.status_code == 200
        data = response.json()
        assert data["server_root_path"] == "/"
        assert data["proxy_base_url"] is None
        assert data["auto_redirect_to_sso"] is False
        assert data["admin_ui_disabled"] is True
        assert data["sso_configured"] is False


def test_ui_discovery_endpoints_with_admin_ui_enabled():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with (
        patch("litellm.proxy.utils.get_server_root_path", return_value="/"),
        patch("litellm.proxy.utils.get_proxy_base_url", return_value=None),
        patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=False),
        patch.dict(os.environ, {"DISABLE_ADMIN_UI": "false"}, clear=False),
    ):
        response = client.get("/.well-known/litellm-ui-config")

        assert response.status_code == 200
        data = response.json()
        assert data["server_root_path"] == "/"
        assert data["proxy_base_url"] is None
        assert data["auto_redirect_to_sso"] is False
        assert data["admin_ui_disabled"] is False
        assert data["sso_configured"] is False


def test_ui_discovery_endpoints_is_control_plane_true_when_workers_configured():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    mock_config = MagicMock()
    mock_config.worker_registry = [
        WorkerRegistryEntry(
            worker_id="team-a", name="Team A", url="https://worker-1:4001"
        ),
    ]

    with (
        patch("litellm.proxy.utils.get_server_root_path", return_value="/"),
        patch("litellm.proxy.utils.get_proxy_base_url", return_value=None),
        patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=False),
        patch("litellm.proxy.proxy_server.proxy_config", mock_config),
        patch.dict(os.environ, {"DISABLE_ADMIN_UI": "false"}, clear=False),
    ):
        response = client.get("/.well-known/litellm-ui-config")

        assert response.status_code == 200
        data = response.json()
        assert data["is_control_plane"] is True
        assert len(data["workers"]) == 1
        assert data["workers"][0]["worker_id"] == "team-a"
        assert data["workers"][0]["name"] == "Team A"
        assert data["workers"][0]["url"] == "https://worker-1:4001"


def test_ui_discovery_endpoints_hide_default_credentials_hint_default_false():
    """Default credentials hint is shown by default (flag false)."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with (
        patch("litellm.proxy.utils.get_server_root_path", return_value="/"),
        patch("litellm.proxy.utils.get_proxy_base_url", return_value=None),
        patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=False),
        patch.dict(os.environ, {"DISABLE_ADMIN_UI": "false"}, clear=False),
    ):
        os.environ.pop("LITELLM_HIDE_DEFAULT_CREDENTIALS_HINT", None)

        response = client.get("/.well-known/litellm-ui-config")

        assert response.status_code == 200
        data = response.json()
        assert data["hide_default_credentials_hint"] is False


def test_ui_discovery_endpoints_hide_default_credentials_hint_via_env_var():
    """LITELLM_HIDE_DEFAULT_CREDENTIALS_HINT=true hides the login-page credentials card."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with (
        patch("litellm.proxy.utils.get_server_root_path", return_value="/"),
        patch("litellm.proxy.utils.get_proxy_base_url", return_value=None),
        patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=False),
        patch.dict(
            os.environ,
            {
                "LITELLM_HIDE_DEFAULT_CREDENTIALS_HINT": "true",
                "DISABLE_ADMIN_UI": "false",
            },
            clear=False,
        ),
    ):
        response = client.get("/.well-known/litellm-ui-config")

        assert response.status_code == 200
        data = response.json()
        assert data["hide_default_credentials_hint"] is True


def test_ui_discovery_endpoints_hide_default_credentials_hint_via_general_settings():
    """general_settings.hide_default_credentials_hint=true also hides the card."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with (
        patch("litellm.proxy.utils.get_server_root_path", return_value="/"),
        patch("litellm.proxy.utils.get_proxy_base_url", return_value=None),
        patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=False),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"hide_default_credentials_hint": True},
        ),
        patch.dict(os.environ, {"DISABLE_ADMIN_UI": "false"}, clear=False),
    ):
        os.environ.pop("LITELLM_HIDE_DEFAULT_CREDENTIALS_HINT", None)

        response = client.get("/.well-known/litellm-ui-config")

        assert response.status_code == 200
        data = response.json()
        assert data["hide_default_credentials_hint"] is True


def test_ui_discovery_endpoints_is_control_plane_false_when_no_workers():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    mock_config = MagicMock()
    mock_config.worker_registry = []

    with (
        patch("litellm.proxy.utils.get_server_root_path", return_value="/"),
        patch("litellm.proxy.utils.get_proxy_base_url", return_value=None),
        patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=False),
        patch("litellm.proxy.proxy_server.proxy_config", mock_config),
        patch.dict(os.environ, {"DISABLE_ADMIN_UI": "false"}, clear=False),
    ):
        response = client.get("/.well-known/litellm-ui-config")

        assert response.status_code == 200
        data = response.json()
        assert data["is_control_plane"] is False
        assert data["workers"] == []
