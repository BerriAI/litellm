import pytest
import os
from fastapi.testclient import TestClient
from litellm.proxy.proxy_server import app

def test_get_image_redirect_behavior(monkeypatch):
    """Verify 307 Redirect behavior for remote URLs."""
    client = TestClient(app)
    
    # Safely set the environment variable and ensure auto-cleanup
    monkeypatch.setenv("UI_LOGO_PATH", "http://example.com/logo.png")
    
    response = client.get("/get_image", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "http://example.com/logo.png"

def test_get_image_local_fallback(monkeypatch):
    """Verify fallback to default logo when environment variable is missing."""
    client = TestClient(app)
    monkeypatch.delenv("UI_LOGO_PATH", raising=False)
    
    response = client.get("/get_image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"