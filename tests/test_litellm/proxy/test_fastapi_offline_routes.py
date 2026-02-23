"""
Unit test for testing /routes endpoint with FastAPIOffline app initialization.

This test verifies that the /routes endpoint works correctly when the proxy 
server is initialized using FastAPIOffline instead of regular FastAPI.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import pytest
from fastapi.testclient import TestClient
from fastapi_offline import FastAPIOffline


class TestFastAPIOfflineRoutes:
    """Test that /routes endpoint works with FastAPIOffline app initialization."""
    
    def test_routes_endpoint_with_fastapi_offline(self):
        """
        Test that /routes endpoint responds correctly when using FastAPIOffline.
        
        This test verifies that when the proxy server app is initialized using 
        FastAPIOffline instead of regular FastAPI, the /routes endpoint still 
        functions properly without throwing the StaticFiles AttributeError.
        """
        from litellm.proxy.proxy_server import router

        # Initialize app using FastAPIOffline instead of regular FastAPI
        app = FastAPIOffline()
        
        # Add a simple root endpoint to verify app is working
        @app.get("/")
        async def root():
            return {"message": "Hello World"}
        
        # Include the litellm proxy router which contains the /routes endpoint
        app.include_router(router)
        
        # Create test client
        client = TestClient(app)
        
        # Test the root endpoint first to ensure app is working
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Hello World"}
        
        # Test the /routes endpoint - this should not fail even with FastAPIOffline
        # The important part is that it doesn't fail with the StaticFiles AttributeError
        response = client.get("/routes")
        
        # Print response for debugging
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text}")
        
        # The key test: we should NOT get a 500 (Internal Server Error)
        # which would indicate the StaticFiles AttributeError bug
        assert response.status_code != 500, f"Got 500 error: {response.text}"
        
        # We accept either 200 (success) or 401 (auth required) - both are valid
        assert response.status_code in [200, 401], f"Unexpected status: {response.status_code}"
        
        if response.status_code == 200:
            # If successful, verify it has the expected structure
            response_json = response.json()
            assert "routes" in response_json
            assert isinstance(response_json["routes"], list)
            print("✓ /routes endpoint returns valid routes data with FastAPIOffline")
        else:
            # If auth fails, ensure it's a proper JSON error response
            response_json = response.json()
            assert "detail" in response_json
            print("✓ /routes endpoint handles auth properly with FastAPIOffline")
        
        # If we get here without any AttributeError exceptions, the fix is working
        print("✓ /routes endpoint handles FastAPIOffline initialization correctly")

    def test_routes_endpoint_with_auth_token_fastapi_offline(self):
        """
        Test /routes endpoint with auth token using FastAPIOffline.
        
        This test provides a mock auth token to actually test the routes response.
        """
        from unittest.mock import patch

        from litellm.proxy.proxy_server import router

        # Initialize app using FastAPIOffline
        app = FastAPIOffline()
        
        @app.get("/")
        async def root():
            return {"message": "Hello World"}
        
        app.include_router(router)
        client = TestClient(app)
        
        # Mock the authentication to bypass the auth requirement
        with patch('litellm.proxy.auth.user_api_key_auth.user_api_key_auth') as mock_auth:
            # Configure mock to return a successful auth response
            mock_auth.return_value = {"user_id": "test_user", "api_key": "test_key"}
            
            # Test with Authorization header
            headers = {"Authorization": "Bearer sk-test-token"}
            response = client.get("/routes", headers=headers)
            
            # If authentication is properly mocked, we should get a 200 response
            # If not, we might get 401, but we should NOT get 500 (AttributeError)
            assert response.status_code in [200, 401], f"Unexpected status code: {response.status_code}"
            
            if response.status_code == 200:
                # If we get a successful response, verify it has the expected structure
                response_json = response.json()
                assert "routes" in response_json
                assert isinstance(response_json["routes"], list)
                print("✓ /routes endpoint returns valid response with FastAPIOffline")
            else:
                # Even if auth fails, ensure it's a proper JSON error response
                response_json = response.json()
                assert "detail" in response_json
                print("✓ /routes endpoint handles auth properly with FastAPIOffline")