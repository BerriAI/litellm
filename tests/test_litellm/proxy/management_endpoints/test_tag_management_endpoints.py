import json
import os
import sys
from typing import Any, Dict, Optional

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from unittest.mock import patch

import litellm
from litellm.proxy.proxy_server import app
from litellm.types.tag_management import TagDeleteRequest, TagInfoRequest, TagNewRequest

client = TestClient(app)


@pytest.mark.asyncio
async def test_create_and_get_tag():
    """
    Test creation of a new tag and retrieving its information
    """
    # Mock the prisma client and _get_tags_config and _save_tags_config
    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.management_endpoints.tag_management_endpoints._get_tags_config"
    ) as mock_get_tags, patch(
        "litellm.proxy.management_endpoints.tag_management_endpoints._save_tags_config"
    ) as mock_save_tags, patch(
        "litellm.proxy.management_endpoints.tag_management_endpoints._add_tag_to_deployment"
    ) as mock_add_tag, patch(
        "litellm.proxy.management_endpoints.tag_management_endpoints._get_model_names"
    ) as mock_get_models:
        # Setup mocks
        mock_get_tags.return_value = {}
        mock_get_models.return_value = {"model-1": "gpt-3.5-turbo"}

        # Create a new tag
        tag_data = {
            "name": "test-tag",
            "description": "Test tag for unit testing",
            "models": ["model-1"],
        }

        # Set admin access for the test
        headers = {"Authorization": f"Bearer sk-1234"}

        # Test tag creation
        response = client.post("/tag/new", json=tag_data, headers=headers)
        assert response.status_code == 200
        result = response.json()
        assert result["message"] == "Tag test-tag created successfully"
        assert result["tag"]["name"] == "test-tag"
        assert result["tag"]["description"] == "Test tag for unit testing"

        # Mock updated tag config for the get request
        mock_get_tags.return_value = {
            "test-tag": {
                "name": "test-tag",
                "description": "Test tag for unit testing",
                "models": ["model-1"],
                "model_info": {"model-1": "gpt-3.5-turbo"},
            }
        }

        # Test retrieving tag info
        info_data = {"names": ["test-tag"]}
        response = client.post("/tag/info", json=info_data, headers=headers)
        assert response.status_code == 200
        result = response.json()
        assert "test-tag" in result
        assert result["test-tag"]["description"] == "Test tag for unit testing"


@pytest.mark.asyncio
async def test_update_tag():
    """
    Test updating an existing tag
    """
    # Mock the prisma client and _get_tags_config and _save_tags_config
    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.management_endpoints.tag_management_endpoints._get_tags_config"
    ) as mock_get_tags, patch(
        "litellm.proxy.management_endpoints.tag_management_endpoints._save_tags_config"
    ) as mock_save_tags, patch(
        "litellm.proxy.management_endpoints.tag_management_endpoints._get_model_names"
    ) as mock_get_models:
        # Setup mocks for existing tag
        mock_get_tags.return_value = {
            "test-tag": {
                "name": "test-tag",
                "description": "Original description",
                "models": ["model-1"],
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00",
                "created_by": "user-123",
            }
        }
        mock_get_models.return_value = {"model-1": "gpt-3.5-turbo", "model-2": "gpt-4"}

        # Update tag data
        update_data = {
            "name": "test-tag",
            "description": "Updated description",
            "models": ["model-1", "model-2"],
        }

        # Set admin access for the test
        headers = {"Authorization": f"Bearer sk-1234"}

        # Test tag update
        response = client.post("/tag/update", json=update_data, headers=headers)
        assert response.status_code == 200
        result = response.json()
        assert result["message"] == "Tag test-tag updated successfully"
        assert result["tag"]["description"] == "Updated description"
        assert len(result["tag"]["models"]) == 2
        assert "model-2" in result["tag"]["models"]


@pytest.mark.asyncio
async def test_delete_tag():
    """
    Test deleting a tag
    """
    # Mock the prisma client and _get_tags_config and _save_tags_config
    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.management_endpoints.tag_management_endpoints._get_tags_config"
    ) as mock_get_tags, patch(
        "litellm.proxy.management_endpoints.tag_management_endpoints._save_tags_config"
    ) as mock_save_tags:
        # Setup mocks for existing tag
        mock_get_tags.return_value = {
            "test-tag": {
                "name": "test-tag",
                "description": "Test tag for deletion",
                "models": ["model-1"],
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00",
                "created_by": "user-123",
            }
        }

        # Delete tag data
        delete_data = {"name": "test-tag"}

        # Set admin access for the test
        headers = {"Authorization": f"Bearer sk-1234"}

        # Test tag deletion
        response = client.post("/tag/delete", json=delete_data, headers=headers)
        assert response.status_code == 200
        result = response.json()
        assert result["message"] == "Tag test-tag deleted successfully"

        # Verify _save_tags_config was called without the deleted tag
        mock_save_tags.assert_called_once()
