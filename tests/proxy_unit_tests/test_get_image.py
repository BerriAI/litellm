import pytest
from fastapi.testclient import TestClient
from litellm.proxy.proxy_server import app

client = TestClient(app)

def test_get_image_redirect_behavior(monkeypatch):
    """
    Test that remote URLs in UI_LOGO_PATH trigger a 307 Redirect.
    """
    test_url = "http://example.com/logo.png"
    monkeypatch.setenv("UI_LOGO_PATH", test_url)
    
    # We set follow_redirects=False to catch the 307 itself
    response = client.get("/get_image", follow_redirects=False)
    
    assert response.status_code == 307
    assert response.headers["location"] == test_url

def test_get_image_local_fallback(monkeypatch):
    """
    Test that local paths return a 200 FileResponse.
    """
    # Ensure no remote URL is set
    monkeypatch.delenv("UI_LOGO_PATH", raising=False)
    
    response = client.get("/get_image")
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
