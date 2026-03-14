import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)

from litellm.proxy.discovery_endpoints.ui_discovery_endpoints import router
from litellm.types.proxy.control_plane_endpoints import WorkerRegistryEntry


def test_ui_discovery_endpoints_with_defaults():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with patch("litellm.proxy.utils.get_server_root_path", return_value="/"), \
         patch("litellm.proxy.utils.get_proxy_base_url", return_value=None), \
         patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=False), \
         patch.dict(os.environ, {"DISABLE_ADMIN_UI": "false"}, clear=False):
        
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

    with patch("litellm.proxy.utils.get_server_root_path", return_value="/litellm"), \
         patch("litellm.proxy.utils.get_proxy_base_url", return_value=None), \
         patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=False), \
         patch.dict(os.environ, {"DISABLE_ADMIN_UI": "false"}, clear=False):
        
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

    with patch("litellm.proxy.utils.get_server_root_path", return_value="/"), \
         patch("litellm.proxy.utils.get_proxy_base_url", return_value="https://proxy.example.com"), \
         patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=False), \
         patch.dict(os.environ, {"DISABLE_ADMIN_UI": "false"}, clear=False):
        
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

    with patch("litellm.proxy.utils.get_server_root_path", return_value="/litellm"), \
         patch("litellm.proxy.utils.get_proxy_base_url", return_value="https://proxy.example.com"), \
         patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=True), \
         patch.dict(os.environ, {"AUTO_REDIRECT_UI_LOGIN_TO_SSO": "true", "DISABLE_ADMIN_UI": "false"}, clear=False):
        
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

    with patch("litellm.proxy.utils.get_server_root_path", return_value="/litellm"), \
         patch("litellm.proxy.utils.get_proxy_base_url", return_value="https://proxy.example.com"), \
         patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=True), \
         patch.dict(os.environ, {"DISABLE_ADMIN_UI": "false"}, clear=False):
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

    with patch("litellm.proxy.utils.get_server_root_path", return_value="/litellm"), \
         patch("litellm.proxy.utils.get_proxy_base_url", return_value="https://proxy.example.com"), \
         patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=True), \
         patch.dict(os.environ, {"AUTO_REDIRECT_UI_LOGIN_TO_SSO": "false", "DISABLE_ADMIN_UI": "false"}, clear=False):
        
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

    with patch("litellm.proxy.utils.get_server_root_path", return_value="/"), \
         patch("litellm.proxy.utils.get_proxy_base_url", return_value=None), \
         patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=False), \
         patch.dict(os.environ, {"AUTO_REDIRECT_UI_LOGIN_TO_SSO": "true", "DISABLE_ADMIN_UI": "false"}, clear=False):
        
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

    with patch("litellm.proxy.utils.get_server_root_path", return_value="/litellm"), \
         patch("litellm.proxy.utils.get_proxy_base_url", return_value="https://proxy.example.com"), \
         patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=True), \
         patch.dict(os.environ, {"AUTO_REDIRECT_UI_LOGIN_TO_SSO": "true", "DISABLE_ADMIN_UI": "false"}, clear=False):
        
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

    with patch("litellm.proxy.utils.get_server_root_path", return_value="/"), \
         patch("litellm.proxy.utils.get_proxy_base_url", return_value=None), \
         patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=True), \
         patch("litellm.proxy.proxy_server.general_settings", {"auto_redirect_ui_login_to_sso": True}), \
         patch.dict(os.environ, {"DISABLE_ADMIN_UI": "false"}, clear=False):
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

    with patch("litellm.proxy.utils.get_server_root_path", return_value="/"), \
         patch("litellm.proxy.utils.get_proxy_base_url", return_value=None), \
         patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=True), \
         patch("litellm.proxy.proxy_server.general_settings", {"auto_redirect_ui_login_to_sso": False}), \
         patch.dict(os.environ, {"AUTO_REDIRECT_UI_LOGIN_TO_SSO": "true", "DISABLE_ADMIN_UI": "false"}, clear=False):

        response = client.get("/.well-known/litellm-ui-config")

        assert response.status_code == 200
        data = response.json()
        assert data["auto_redirect_to_sso"] is True


def test_ui_discovery_endpoints_with_admin_ui_disabled():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with patch("litellm.proxy.utils.get_server_root_path", return_value="/"), \
         patch("litellm.proxy.utils.get_proxy_base_url", return_value=None), \
         patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=False), \
         patch.dict(os.environ, {"DISABLE_ADMIN_UI": "true"}, clear=False):
        
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

    with patch("litellm.proxy.utils.get_server_root_path", return_value="/"), \
         patch("litellm.proxy.utils.get_proxy_base_url", return_value=None), \
         patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=False), \
         patch.dict(os.environ, {"DISABLE_ADMIN_UI": "false"}, clear=False):

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

    with patch("litellm.proxy.utils.get_server_root_path", return_value="/"), \
         patch("litellm.proxy.utils.get_proxy_base_url", return_value=None), \
         patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=False), \
         patch("litellm.proxy.proxy_server.proxy_config", mock_config), \
         patch.dict(os.environ, {"DISABLE_ADMIN_UI": "false"}, clear=False):

        response = client.get("/.well-known/litellm-ui-config")

        assert response.status_code == 200
        data = response.json()
        assert data["is_control_plane"] is True
        assert len(data["workers"]) == 1
        assert data["workers"][0]["worker_id"] == "team-a"
        assert data["workers"][0]["name"] == "Team A"
        assert data["workers"][0]["url"] == "https://worker-1:4001"


def test_ui_discovery_endpoints_is_control_plane_false_when_no_workers():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    mock_config = MagicMock()
    mock_config.worker_registry = []

    with patch("litellm.proxy.utils.get_server_root_path", return_value="/"), \
         patch("litellm.proxy.utils.get_proxy_base_url", return_value=None), \
         patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=False), \
         patch("litellm.proxy.proxy_server.proxy_config", mock_config), \
         patch.dict(os.environ, {"DISABLE_ADMIN_UI": "false"}, clear=False):

        response = client.get("/.well-known/litellm-ui-config")

        assert response.status_code == 200
        data = response.json()
        assert data["is_control_plane"] is False
        assert data["workers"] == []
