import datetime
import json
import os
import sys
from datetime import timezone

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

import jwt

from litellm.proxy.auth.handle_jwt import JWTHandler
from litellm.proxy.management_helpers.ui_session_handler import UISessionHandler


class TestJWTHandler:
    @pytest.fixture
    def jwt_handler(self):
        handler = JWTHandler()
        handler.leeway = 60  # Set leeway for testing
        return handler

    @pytest.fixture
    def setup_mocks(self, monkeypatch):
        # Mock master_key
        test_master_key = "test_master_key"
        monkeypatch.setattr("litellm.proxy.proxy_server.master_key", test_master_key)

        # Mock UISessionHandler.is_ui_session_token to return True for our test token
        def mock_is_ui_session_token(payload):
            return "ui_session_id" in payload

        monkeypatch.setattr(
            UISessionHandler, "is_ui_session_token", mock_is_ui_session_token
        )

        return test_master_key

    def test_validate_valid_ui_token(self, jwt_handler, setup_mocks):
        # Setup
        test_master_key = setup_mocks

        # Create a valid UI token
        valid_payload = {
            "ui_session_id": "test_session_id",
            "exp": datetime.datetime.now(tz=timezone.utc).timestamp() + 3600,
            "iat": datetime.datetime.now(tz=timezone.utc).timestamp(),
            "aud": "litellm-ui",
        }
        valid_token = jwt.encode(valid_payload, test_master_key, algorithm="HS256")

        # Test valid UI token
        result = jwt_handler._validate_ui_token(valid_token)
        assert result is not None
        assert result["ui_session_id"] == "test_session_id"

    def test_validate_expired_ui_token(self, jwt_handler, setup_mocks):
        # Setup
        test_master_key = setup_mocks

        # Create an expired UI token
        expired_payload = {
            "ui_session_id": "test_session_id",
            "exp": datetime.datetime.now(tz=timezone.utc).timestamp() - 3600,
            "iat": datetime.datetime.now(tz=timezone.utc).timestamp() - 7200,
            "aud": "litellm-ui",
        }
        expired_token = jwt.encode(expired_payload, test_master_key, algorithm="HS256")

        # Test expired UI token
        with pytest.raises(ValueError, match="Invalid UI token"):
            jwt_handler._validate_ui_token(expired_token)

    def test_validate_invalid_signature_ui_token(self, jwt_handler, setup_mocks):
        # Setup
        test_master_key = setup_mocks

        # Create a token with invalid signature
        valid_payload = {
            "ui_session_id": "test_session_id",
            "exp": datetime.datetime.now(tz=timezone.utc).timestamp() + 3600,
            "iat": datetime.datetime.now(tz=timezone.utc).timestamp(),
            "aud": "litellm-ui",
        }
        invalid_token = jwt.encode(valid_payload, "wrong_key", algorithm="HS256")

        # Test UI token with invalid signature
        with pytest.raises(ValueError, match="Invalid UI token"):
            jwt_handler._validate_ui_token(invalid_token)

    def test_validate_non_ui_token(self, jwt_handler, setup_mocks):
        # Setup
        test_master_key = setup_mocks

        # Create a non-UI token
        non_ui_payload = {
            "sub": "user123",
            "exp": datetime.datetime.now(tz=timezone.utc).timestamp() + 3600,
        }
        non_ui_token = jwt.encode(non_ui_payload, test_master_key, algorithm="HS256")

        # Test non-UI token
        result = jwt_handler._validate_ui_token(non_ui_token)
        assert result is None
