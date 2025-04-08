import json
import os
import sys
from typing import Optional, cast
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.auth.handle_jwt import JWTHandler
from litellm.proxy.management_endpoints.types import CustomOpenID
from litellm.proxy.management_endpoints.ui_sso import MicrosoftSSOHandler


def test_microsoft_sso_handler_openid_from_response():
    # Arrange
    # Create a mock response similar to what Microsoft SSO would return
    mock_response = {
        "mail": "test@example.com",
        "displayName": "Test User",
        "id": "user123",
        "givenName": "Test",
        "surname": "User",
        "some_other_field": "value",
    }

    # Create a mock JWTHandler that returns predetermined team IDs
    mock_jwt_handler = MagicMock(spec=JWTHandler)
    expected_team_ids = ["team1", "team2"]
    mock_jwt_handler.get_team_ids_from_jwt.return_value = expected_team_ids

    # Act
    # Call the method being tested
    result = MicrosoftSSOHandler.openid_from_response(
        response=mock_response, jwt_handler=mock_jwt_handler
    )

    # Assert
    # Verify the JWT handler was called with the correct parameters
    mock_jwt_handler.get_team_ids_from_jwt.assert_called_once_with(
        cast(dict, mock_response)
    )

    # Check that the result is a CustomOpenID object with the expected values
    assert isinstance(result, CustomOpenID)
    assert result.email == "test@example.com"
    assert result.display_name == "Test User"
    assert result.provider == "microsoft"
    assert result.id == "user123"
    assert result.first_name == "Test"
    assert result.last_name == "User"
    assert result.team_ids == expected_team_ids


def test_microsoft_sso_handler_with_empty_response():
    # Arrange
    # Test with None response
    mock_jwt_handler = MagicMock(spec=JWTHandler)
    mock_jwt_handler.get_team_ids_from_jwt.return_value = []

    # Act
    result = MicrosoftSSOHandler.openid_from_response(
        response=None, jwt_handler=mock_jwt_handler
    )

    # Assert
    assert isinstance(result, CustomOpenID)
    assert result.email is None
    assert result.display_name is None
    assert result.provider == "microsoft"
    assert result.id is None
    assert result.first_name is None
    assert result.last_name is None
    assert result.team_ids == []

    # Make sure the JWT handler was called with an empty dict
    mock_jwt_handler.get_team_ids_from_jwt.assert_called_once_with({})
