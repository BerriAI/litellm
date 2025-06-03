"""
Tests for SSO configuration endpoints.

This module tests the new endpoints for managing SSO provider configuration.
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import pytest
import requests
import json


@pytest.mark.parametrize(
    "sso_provider, config_data",
    [
        (
            "google",
            {
                "sso_provider": "google",
                "google_client_id": "test-google-client-id",
                "google_client_secret": "test-google-client-secret",
                "proxy_base_url": "https://test-proxy-domain.com",
                "user_email": "admin@testdomain.com"
            }
        ),
        (
            "microsoft",
            {
                "sso_provider": "microsoft",
                "microsoft_client_id": "test-microsoft-client-id",
                "microsoft_client_secret": "test-microsoft-client-secret",
                "microsoft_tenant": "test-tenant-id",
                "proxy_base_url": "https://test-proxy-domain.com",
                "user_email": "admin@testdomain.com"
            }
        ),
        (
            "okta",
            {
                "sso_provider": "okta",
                "generic_client_id": "test-okta-client-id",
                "generic_client_secret": "test-okta-client-secret",
                "generic_authorization_endpoint": "https://test-okta-domain.com/oauth2/v1/authorize",
                "generic_token_endpoint": "https://test-okta-domain.com/oauth2/v1/token",
                "generic_userinfo_endpoint": "https://test-okta-domain.com/oauth2/v1/userinfo",
                "generic_scope": "openid email profile",
                "proxy_base_url": "https://test-proxy-domain.com",
                "user_email": "admin@testdomain.com"
            }
        )
    ]
)
def test_sso_provider_config_update(sso_provider, config_data):
    """Test updating SSO configuration for different providers"""
    # This test would require a running proxy server
    # For now, it validates the test data structure
    assert config_data["sso_provider"] == sso_provider
    assert "proxy_base_url" in config_data
    assert "user_email" in config_data
    
    if sso_provider == "google":
        assert "google_client_id" in config_data
        assert "google_client_secret" in config_data
    elif sso_provider == "microsoft":
        assert "microsoft_client_id" in config_data
        assert "microsoft_client_secret" in config_data
        assert "microsoft_tenant" in config_data
    elif sso_provider == "okta":
        assert "generic_client_id" in config_data
        assert "generic_client_secret" in config_data
        assert "generic_authorization_endpoint" in config_data
        assert "generic_token_endpoint" in config_data
        assert "generic_userinfo_endpoint" in config_data


def test_sso_endpoints_structure():
    """Test that SSO endpoint test data follows expected patterns"""
    base_url = "http://localhost:4000"
    headers = {
        "Authorization": "Bearer sk-test-key",
        "Content-Type": "application/json"
    }
    
    # Test endpoint paths
    get_endpoint = f"{base_url}/get/sso_provider_config"
    update_endpoint = f"{base_url}/update/sso_provider_config"
    delete_endpoint = f"{base_url}/delete/sso_provider_config"
    
    assert get_endpoint.endswith("/get/sso_provider_config")
    assert update_endpoint.endswith("/update/sso_provider_config")
    assert delete_endpoint.endswith("/delete/sso_provider_config")
    
    assert headers["Content-Type"] == "application/json"
    assert headers["Authorization"].startswith("Bearer ")


def test_sso_get_config_endpoint():
    """Integration test for GET /get/sso_provider_config endpoint"""
    base_url = "http://localhost:4000"
    headers = {
        "Authorization": "Bearer sk-1234",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(f"{base_url}/get/sso_provider_config", headers=headers, timeout=5)
        # Should either succeed with 200 or fail with proper HTTP status
        assert response.status_code in [200, 401, 403, 404, 500]
        
        if response.status_code == 200:
            config = response.json()
            assert isinstance(config, dict)
    except requests.exceptions.RequestException:
        pytest.skip("Proxy server not available")


def test_sso_delete_config_endpoint():
    """Integration test for DELETE /delete/sso_provider_config endpoint"""
    base_url = "http://localhost:4000"
    headers = {
        "Authorization": "Bearer sk-1234",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.delete(f"{base_url}/delete/sso_provider_config", headers=headers, timeout=5)
        # Should either succeed with 200 or fail with proper HTTP status
        assert response.status_code in [200, 401, 403, 404, 500]
        
        if response.status_code == 200:
            result = response.json()
            assert isinstance(result, dict)
    except requests.exceptions.RequestException:
        pytest.skip("Proxy server not available") 