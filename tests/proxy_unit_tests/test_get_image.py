import pytest
from fastapi.testclient import TestClient
from litellm.proxy.proxy_server import app
import os

def test_get_image_redirect_behavior():
    """
    Update existing tests to verify the new Redirect behavior for remote URLs.
    """
    client = TestClient(app)
    
    # Test case: Remote URL should 307 Redirect
    os.environ["UI_LOGO_PATH"] = "http://example.com/logo.png"
    
    # Ensure no stale cache interferes
    if os.path.exists("cached_logo.jpg"):
        os.remove("cached_logo.jpg")
        
    response = client.get("/get_image", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "http://example.com/logo.png"

def test_get_image_local_fallback():
    """
    Verify that if no URL is provided, it still returns a 200 for local files.
    """
    client = TestClient(app)
    os.environ.pop("UI_LOGO_PATH", None)
    
    response = client.get("/get_image")
    # This should return the default logo (200 OK)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
