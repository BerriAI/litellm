"""
Test suite for Container API router functionality.
Tests that the router method gets called correctly for container operations.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import litellm


class TestContainerRouter:
    """Test suite for Container API router functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.container_name = "Test Container"
        self.container_id = "cntr_123456789"

    @patch("litellm.containers.main.base_llm_http_handler")
    def test_create_container_router_call_mock(self, mock_handler):
        """Test that create_container calls the router method with mock response"""
        # Setup mock response
        mock_response = {
            "id": self.container_id,
            "object": "container",
            "created_at": 1747857508,
            "status": "running",
            "expires_after": {
                "anchor": "last_active_at",
                "minutes": 20
            },
            "last_active_at": 1747857508,
            "name": self.container_name
        }
        
        # Configure the mock handler
        mock_handler.container_create_handler.return_value = mock_response
        
        # Call the create_container function with mock response
        result = litellm.create_container(
            name=self.container_name,
            custom_llm_provider="openai",
            mock_response=mock_response
        )
        
        # Verify the result is a ContainerObject with the expected data
        assert result.id == mock_response["id"]
        assert result.object == mock_response["object"]
        assert result.name == mock_response["name"]
        assert result.status == mock_response["status"]
        assert result.created_at == mock_response["created_at"]

    @patch("litellm.containers.main.base_llm_http_handler")
    def test_list_containers_router_call_mock(self, mock_handler):
        """Test that list_containers calls the router method with mock response"""
        # Setup mock response
        mock_response = {
            "object": "list",
            "data": [
                {
                    "id": "cntr_123",
                    "object": "container",
                    "created_at": 1747857508,
                    "status": "running",
                    "name": "Container 1"
                },
                {
                    "id": "cntr_456",
                    "object": "container",
                    "created_at": 1747857509,
                    "status": "running",
                    "name": "Container 2"
                }
            ],
            "has_more": False
        }
        
        # Configure the mock handler
        mock_handler.container_list_handler.return_value = mock_response
        
        # Call the list_containers function with mock response
        result = litellm.list_containers(
            custom_llm_provider="openai",
            mock_response=mock_response
        )
        
        # Verify the result is a ContainerListResponse with the expected data
        assert result.object == "list"
        assert len(result.data) == 2
        assert result.data[0].id == "cntr_123"
        assert result.data[1].id == "cntr_456"
        assert result.has_more is False

    @patch("litellm.containers.main.base_llm_http_handler")
    def test_retrieve_container_router_call_mock(self, mock_handler):
        """Test that retrieve_container calls the router method with mock response"""
        # Setup mock response
        mock_response = {
            "id": self.container_id,
            "object": "container",
            "created_at": 1747857508,
            "status": "running",
            "expires_after": {
                "anchor": "last_active_at",
                "minutes": 20
            },
            "last_active_at": 1747857508,
            "name": self.container_name
        }
        
        # Configure the mock handler
        mock_handler.container_retrieve_handler.return_value = mock_response
        
        # Call the retrieve_container function with mock response
        result = litellm.retrieve_container(
            container_id=self.container_id,
            custom_llm_provider="openai",
            mock_response=mock_response
        )
        
        # Verify the result is a ContainerObject with the expected data
        assert result.id == mock_response["id"]
        assert result.object == mock_response["object"]
        assert result.name == mock_response["name"]
        assert result.status == mock_response["status"]

    @patch("litellm.containers.main.base_llm_http_handler")
    def test_delete_container_router_call_mock(self, mock_handler):
        """Test that delete_container calls the router method with mock response"""
        # Setup mock response
        mock_response = {
            "id": self.container_id,
            "object": "container.deleted",
            "deleted": True
        }
        
        # Configure the mock handler
        mock_handler.container_delete_handler.return_value = mock_response
        
        # Call the delete_container function with mock response
        result = litellm.delete_container(
            container_id=self.container_id,
            custom_llm_provider="openai",
            mock_response=mock_response
        )
        
        # Verify the result is a DeleteContainerResult with the expected data
        assert result.id == mock_response["id"]
        assert result.object == mock_response["object"]
        assert result.deleted is True

    @pytest.mark.asyncio
    @patch("litellm.containers.main.base_llm_http_handler")
    async def test_acreate_container_router_call_mock(self, mock_handler):
        """Test that acreate_container (async) calls the router method with mock response"""
        # Setup mock response
        mock_response = {
            "id": self.container_id,
            "object": "container",
            "created_at": 1747857508,
            "status": "running",
            "name": self.container_name
        }
        
        # Configure the mock handler
        mock_handler.container_create_handler.return_value = mock_response
        
        # Call the async create_container function with mock response
        result = await litellm.acreate_container(
            name=self.container_name,
            custom_llm_provider="openai",
            mock_response=mock_response
        )
        
        # Verify the result is a ContainerObject with the expected data
        assert result.id == mock_response["id"]
        assert result.object == mock_response["object"]
        assert result.name == mock_response["name"]
        assert result.status == mock_response["status"]

    @pytest.mark.asyncio
    @patch("litellm.containers.main.base_llm_http_handler")
    async def test_alist_containers_router_call_mock(self, mock_handler):
        """Test that alist_containers (async) calls the router method with mock response"""
        # Setup mock response
        mock_response = {
            "object": "list",
            "data": [
                {
                    "id": "cntr_123",
                    "object": "container",
                    "created_at": 1747857508,
                    "status": "running",
                    "name": "Container 1"
                }
            ],
            "has_more": False
        }
        
        # Configure the mock handler
        mock_handler.container_list_handler.return_value = mock_response
        
        # Call the async list_containers function with mock response
        result = await litellm.alist_containers(
            custom_llm_provider="openai",
            mock_response=mock_response
        )
        
        # Verify the result is a ContainerListResponse with the expected data
        assert result.object == "list"
        assert len(result.data) == 1
        assert result.data[0].id == "cntr_123"

