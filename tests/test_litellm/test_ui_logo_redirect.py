import pytest
from fastapi.testclient import TestClient
from litellm.proxy.proxy_server import app
import os

def test_get_image_redirect():
    # 1. Setup Client
    client = TestClient(app)
    test_url = "http://example.com/logo.png"
    
    # 2. Force the environment variable
    os.environ["UI_LOGO_PATH"] = test_url
    
    # 3. FIX: Ensure no stale cached files exist that would cause a 200 instead of 307
    if os.path.exists("cached_logo.jpg"):
        os.remove("cached_logo.jpg")

    # 4. Call endpoint (do not follow redirects so we can catch the 307)
    response = client.get("/get_image", follow_redirects=False)
    
    # 5. Assertions
    assert response.status_code == 307
    assert response.headers["location"] == test_url
