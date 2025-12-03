import os
import sys
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)

from litellm.proxy.discovery_endpoints.ui_discovery_endpoints import router


def test_ui_discovery_endpoints_with_defaults():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with patch("litellm.proxy.utils.get_server_root_path", return_value="/"), \
         patch("litellm.proxy.utils.get_proxy_base_url", return_value=None), \
         patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=False):
        
        response = client.get("/.well-known/litellm-ui-config")
        
        assert response.status_code == 200
        data = response.json()
        assert data["server_root_path"] == "/"
        assert data["proxy_base_url"] is None
        assert data["is_sso_configured"] is False


def test_ui_discovery_endpoints_with_custom_server_root_path():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with patch("litellm.proxy.utils.get_server_root_path", return_value="/litellm"), \
         patch("litellm.proxy.utils.get_proxy_base_url", return_value=None), \
         patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=False):
        
        response = client.get("/.well-known/litellm-ui-config")
        
        assert response.status_code == 200
        data = response.json()
        assert data["server_root_path"] == "/litellm"
        assert data["proxy_base_url"] is None
        assert data["is_sso_configured"] is False


def test_ui_discovery_endpoints_with_proxy_base_url_when_set():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with patch("litellm.proxy.utils.get_server_root_path", return_value="/"), \
         patch("litellm.proxy.utils.get_proxy_base_url", return_value="https://proxy.example.com"), \
         patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=False):
        
        response = client.get("/litellm/.well-known/litellm-ui-config")
        
        assert response.status_code == 200
        data = response.json()
        assert data["server_root_path"] == "/"
        assert data["proxy_base_url"] == "https://proxy.example.com"
        assert data["is_sso_configured"] is False


def test_ui_discovery_endpoints_with_sso_configured_when_sso_is_setup():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with patch("litellm.proxy.utils.get_server_root_path", return_value="/litellm"), \
         patch("litellm.proxy.utils.get_proxy_base_url", return_value="https://proxy.example.com"), \
         patch("litellm.proxy.auth.auth_utils._has_user_setup_sso", return_value=True):
        
        response = client.get("/.well-known/litellm-ui-config")
        
        assert response.status_code == 200
        data = response.json()
        assert data["server_root_path"] == "/litellm"
        assert data["proxy_base_url"] == "https://proxy.example.com"
        assert data["is_sso_configured"] is True

